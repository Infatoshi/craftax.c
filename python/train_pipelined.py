"""Pipelined PPO: n-buffered rollouts on separate CUDA streams.

Architecture
------------
  N RolloutBuffers in a ring. At any time one is being *trained on* (the
  consumer) and one is being *filled* (the producer). With N=1 these
  serialize (= baseline). With N>=2 they overlap:

      iter k-1:  [train on buf_{k-1}]   |   [fill buf_k       ]
      iter k  :  [train on buf_k    ]   |   [fill buf_{k+1}   ]

  Producer = rollout loop (policy.forward_eval on rollout_stream + CPU env).
  Consumer = PPO update (minibatches on train_stream).
  CUDA events bridge stream dependencies so train waits on its buffer
  being filled before starting.

Since the CPU env is already parallel (16 OMP threads inside
craftax_step_batch), the only thing we need to orchestrate is the GPU
side: queue policy.forward_eval on rollout_stream, queue update work on
train_stream, buffer obs/actions/rewards/etc on the GPU to avoid
train/rollout aliasing.

Usage
-----
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \\
    CRAFTAX_LIB=$PWD/libcraftax.so PYTHONPATH=python \\
    /path/to/pufferlib/venv/bin/python python/train_pipelined.py \\
        --n-buffers 2 --num-envs 1024 --horizon 64 --iters 30
"""
from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

import pufferlib.pytorch
from pufferlib.environments.craftax.torch import (
    Policy as _Policy,
    N_MAP,
    N_FLAT,
)

from craftax_c.environment import CraftaxCEnv


# ----------------------------------------------------------------------------
# Policy: accept state=None so PPO-style train loops can call policy(obs, None).
# ----------------------------------------------------------------------------
class Policy(_Policy):
    def forward(self, observations, state=None):
        hidden, lookup = self.encode_observations(observations)
        action, value = self.decode_actions(hidden, lookup)
        return action, value

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


class _E:
    class _S:
        n = 17
    single_action_space = _S()


# ----------------------------------------------------------------------------
# Rollout buffer -- all tensors on GPU; producer writes, consumer reads.
# ----------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, horizon, num_envs, obs_dim, device):
        H, E = horizon, num_envs
        self.obs      = torch.zeros(H, E, obs_dim, device=device)
        self.actions  = torch.zeros(H, E, dtype=torch.int64, device=device)
        self.rewards  = torch.zeros(H, E, device=device)
        self.dones    = torch.zeros(H, E, device=device)
        self.values   = torch.zeros(H, E, device=device)
        self.logprobs = torch.zeros(H, E, device=device)
        self.obs_last = torch.zeros(E, obs_dim, device=device)
        self.filled_event = torch.cuda.Event() if device == "cuda" else None
        self.consumed_event = torch.cuda.Event() if device == "cuda" else None


@torch.no_grad()
def fill_rollout(env, policy, buf, horizon, last_obs_cpu, last_obs_gpu,
                 rollout_stream, action_scratch_cpu, sample_fn):
    """Fill a rollout buffer using the given env + policy on rollout_stream.

    `last_obs_cpu` is the CPU float32 obs buffer the env writes into (owned
    by CraftaxCEnv). We async-copy it to `last_obs_gpu` each step so the
    GPU policy reads from a persistent gpu tensor and the CPU env can
    proceed in parallel with the GPU work on another stream.
    """
    ctx = torch.cuda.stream(rollout_stream) if rollout_stream else nullcontext()
    with ctx:
        for t in range(horizon):
            # Stage 1: async copy current CPU obs -> GPU tensor.
            last_obs_gpu.copy_(
                torch.from_numpy(last_obs_cpu), non_blocking=True
            )
            buf.obs[t].copy_(last_obs_gpu, non_blocking=True)

            # Stage 2: policy on the rollout stream.
            logits, value = policy.forward_eval(last_obs_gpu, None)
            # Categorical sample + logprob (single fused call path).
            action, logprob = sample_fn(logits)
            buf.actions[t] = action
            buf.values[t] = value.squeeze(-1)
            buf.logprobs[t] = logprob

            # Stage 3: D2H for actions (needed by CPU env).
            # Use pinned host scratch so the copy is async.
            action_scratch_cpu.copy_(action.to(torch.int32), non_blocking=True)
    # Ensure the D2H completes before CPU reads the action array.
    if rollout_stream is not None:
        rollout_stream.synchronize()

    # Stage 4: step the env. This runs on CPU (16 OMP threads) and does NOT
    # contend with the train stream on the GPU.
    env_obs, env_rew, env_term, env_trunc, _info = env.step(
        action_scratch_cpu.numpy()
    )
    # Scatter per-step rewards/dones into the buffer. We step the env H times
    # in the caller loop, not inside this function -- so we actually want a
    # different structure. See fill_rollout_loop below.


def sample_gumbel(logits):
    """Gumbel-max sampling (matches PufferLib's style): action, logprob."""
    u = torch.rand_like(logits).clamp_(1e-8, 1.0)
    a = (logits - torch.log(-torch.log(u))).argmax(-1)
    lp = logits.log_softmax(-1).gather(-1, a.unsqueeze(-1)).squeeze(-1)
    return a, lp


# ----------------------------------------------------------------------------
# Fill one rollout buffer: interleave policy forward (GPU) and env step (CPU).
# ----------------------------------------------------------------------------
@torch.no_grad()
def fill_one_rollout(env, policy, buf, horizon, obs_cpu_ref, obs_gpu,
                     action_int32_cpu, stream):
    ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx:
        for t in range(horizon):
            # CPU obs -> GPU (async on this stream)
            obs_gpu.copy_(
                torch.from_numpy(obs_cpu_ref[0]), non_blocking=True
            )
            buf.obs[t].copy_(obs_gpu, non_blocking=True)

            logits, value = policy.forward_eval(obs_gpu, None)
            action, logprob = sample_gumbel(logits)
            buf.actions[t] = action
            buf.values[t] = value.squeeze(-1)
            buf.logprobs[t] = logprob

            # D2H of the action to CPU buffer the env will consume.
            action_int32_cpu.copy_(action.to(torch.int32), non_blocking=True)

        # CPU env step must see finished D2H -- sync this stream only.
        if stream is not None:
            stream.synchronize()
        # Actually we also need to block on the *previous* step's synchronize
        # between each t for correctness. We do env.step inside the t loop
        # (see outer), so this single-rollout fn isn't quite right. Rework:
    raise RuntimeError("use fill_one_rollout_v2")


@torch.no_grad()
def fill_one_rollout_v2(env, policy, buf, horizon, obs_cpu_ref,
                        obs_gpu, action_int32_cpu, stream):
    """Correct interleaved rollout: one env.step per t, synced against policy."""
    ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    for t in range(horizon):
        # --- GPU side: copy obs, run policy, D2H action ---
        with ctx:
            obs_gpu.copy_(torch.from_numpy(obs_cpu_ref[0]), non_blocking=True)
            buf.obs[t].copy_(obs_gpu, non_blocking=True)
            logits, value = policy.forward_eval(obs_gpu, None)
            action, logprob = sample_gumbel(logits)
            buf.actions[t] = action
            buf.values[t] = value.squeeze(-1)
            buf.logprobs[t] = logprob
            action_int32_cpu.copy_(action.to(torch.int32), non_blocking=True)
        # Sync just this stream -- doesn't block the train stream.
        if stream is not None:
            stream.synchronize()
        # --- CPU env step (16 OMP threads) ---
        new_obs, rew, term, trunc, _info = env.step(action_int32_cpu.numpy())
        # Copy reward / done into buffer (GPU).
        buf.rewards[t].copy_(
            torch.from_numpy(np.asarray(rew, dtype=np.float32)),
            non_blocking=True,
        )
        buf.dones[t].copy_(
            torch.from_numpy(np.asarray(term, dtype=np.float32)),
            non_blocking=True,
        )
        obs_cpu_ref[0] = new_obs  # update reference for next step
    # Capture last obs for GAE bootstrap.
    with ctx:
        buf.obs_last.copy_(
            torch.from_numpy(obs_cpu_ref[0]), non_blocking=True
        )
    if stream is not None:
        buf.filled_event.record(stream)


# ----------------------------------------------------------------------------
# PPO update on a specific stream.
# ----------------------------------------------------------------------------
def ppo_update(policy, opt, buf, n_epochs, minibatch_size, clip=0.2,
               ent_coef=0.01, vf_coef=0.5, gamma=0.99, lam=0.8, train_stream=None):
    H, E = buf.obs.shape[:2]
    obs_flat   = buf.obs.reshape(H*E, -1)
    act_flat   = buf.actions.reshape(-1)
    rew_flat   = buf.rewards
    done_flat  = buf.dones
    val_flat   = buf.values
    logp_flat  = buf.logprobs

    ctx = torch.cuda.stream(train_stream) if train_stream is not None else nullcontext()
    with ctx:
        if train_stream is not None:
            train_stream.wait_event(buf.filled_event)

        # Bootstrap value for last obs.
        with torch.no_grad():
            _, last_v = policy(buf.obs_last)
            last_v = last_v.squeeze(-1)
        # GAE
        adv = torch.zeros_like(val_flat)
        lastgaelam = torch.zeros(E, device=val_flat.device)
        for t in reversed(range(H)):
            nextnonterm = 1.0 - done_flat[t]
            nextv = last_v if t == H-1 else val_flat[t+1]
            delta = rew_flat[t] + gamma * nextv * nextnonterm - val_flat[t]
            lastgaelam = delta + gamma * lam * nextnonterm * lastgaelam
            adv[t] = lastgaelam
        ret = adv + val_flat
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        adv_f  = adv.reshape(-1)
        ret_f  = ret.reshape(-1)
        val_f  = val_flat.reshape(-1)

        N = H * E
        for _ in range(n_epochs):
            perm = torch.randperm(N, device=obs_flat.device)
            for start in range(0, N, minibatch_size):
                idx = perm[start:start+minibatch_size]
                logits, v_new = policy(obs_flat[idx])
                v_new = v_new.squeeze(-1)
                lp_all = logits.log_softmax(-1)
                nlp = lp_all.gather(-1, act_flat[idx].unsqueeze(-1)).squeeze(-1)
                ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                ratio = (nlp - logp_flat.reshape(-1)[idx]).exp()
                a_mb = adv_f[idx]
                pg = -torch.min(
                    ratio * a_mb,
                    ratio.clamp(1-clip, 1+clip) * a_mb,
                ).mean()
                vc = val_f[idx] + (v_new - val_f[idx]).clamp(-clip, clip)
                vl = 0.5 * torch.max(
                    (v_new - ret_f[idx]).square(),
                    (vc - ret_f[idx]).square(),
                ).mean()
                loss = pg + vf_coef * vl - ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        if train_stream is not None:
            buf.consumed_event.record(train_stream)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--n-buffers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-streams", action="store_true",
                    help="Serialize rollout+train (N=1 semantics regardless of --n-buffers)")
    args = ap.parse_args()

    device = args.device
    env = CraftaxCEnv(num_envs=args.num_envs)
    obs_initial, _ = env.reset()

    obs_dim = env.single_observation_space.shape[0]
    policy = Policy(env).to(device)
    opt = torch.optim.Adam(
        policy.parameters(), lr=args.lr, eps=1e-5,
        fused=(device == "cuda"),
    )

    if device == "cuda" and not args.no_streams:
        rollout_stream = torch.cuda.Stream()
        train_stream   = torch.cuda.Stream()
    else:
        rollout_stream = None
        train_stream   = None

    N = args.n_buffers
    buffers = [RolloutBuffer(args.horizon, args.num_envs, obs_dim, device) for _ in range(N)]
    # One pinned obs staging buffer per ring slot (we only ever use one CPU obs
    # at a time -- the env owns it -- so we keep a single obs_gpu tensor and
    # a single CPU action scratch).
    obs_gpu = torch.zeros(args.num_envs, obs_dim, device=device)
    action_int32_cpu = torch.zeros(args.num_envs, dtype=torch.int32,
                                   device="cpu", pin_memory=(device == "cuda"))
    obs_cpu_ref = [obs_initial]   # boxed so inner fn can mutate

    # Warmup: fill all N-1 buffers so the pipeline is primed at steady state.
    prefetch = max(0, N - 1)
    for i in range(prefetch):
        fill_one_rollout_v2(env, policy, buffers[i], args.horizon,
                            obs_cpu_ref, obs_gpu, action_int32_cpu, rollout_stream)

    if device == "cuda":
        torch.cuda.synchronize()

    from concurrent.futures import ThreadPoolExecutor
    t_start = time.time()
    step_count = 0
    slot_fill = prefetch % N
    slot_train = 0

    if args.no_streams or N == 1:
        # Serial version (baseline).
        for it in range(args.iters):
            fill_one_rollout_v2(env, policy, buffers[slot_fill], args.horizon,
                                obs_cpu_ref, obs_gpu, action_int32_cpu, rollout_stream)
            ppo_update(policy, opt, buffers[slot_train], args.epochs, args.minibatch,
                       train_stream=train_stream)
            slot_fill  = (slot_fill  + 1) % N
            slot_train = (slot_train + 1) % N
            step_count += args.horizon * args.num_envs
    else:
        # Threaded: run fill and ppo_update concurrently in Python (GIL is
        # released in CUDA/C ops), each on its own stream.
        ex = ThreadPoolExecutor(max_workers=2)
        for it in range(args.iters):
            f1 = ex.submit(
                fill_one_rollout_v2, env, policy, buffers[slot_fill], args.horizon,
                obs_cpu_ref, obs_gpu, action_int32_cpu, rollout_stream,
            )
            f2 = ex.submit(
                ppo_update, policy, opt, buffers[slot_train], args.epochs,
                args.minibatch, 0.2, 0.01, 0.5, 0.99, 0.8, train_stream,
            )
            f1.result(); f2.result()
            slot_fill  = (slot_fill  + 1) % N
            slot_train = (slot_train + 1) % N
            step_count += args.horizon * args.num_envs
        ex.shutdown(wait=True)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t_start
    sps = step_count / elapsed
    print(f"N={N}  horizon={args.horizon}  num_envs={args.num_envs}  iters={args.iters}")
    print(f"  time: {elapsed:.2f}s   steps: {step_count:,}   SPS: {sps:,.0f}")


if __name__ == "__main__":
    main()
