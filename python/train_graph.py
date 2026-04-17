"""PPO training with CUDA graph capture of the minibatch update.

The 45k-param policy is launch-bound: measured 743us per plain minibatch
update where the actual FMA compute is ~40us. The rest is kernel launch
overhead for forward/backward/optimizer -- hundreds of small kernels
dispatched serially from Python.

Graph capture collapses all of that into one torch.cuda.CUDAGraph.replay().
We allocate static input buffers (mb_obs / mb_act / mb_logp / mb_adv /
mb_ret / mb_val) and a capturable fused Adam. On each minibatch we
copy the selected rows into the static buffers, then replay.

Stacks cleanly on top of the N-buffered pipelining from train_pipelined.py
(--n-buffers 2 still works; producer thread fills rollouts, consumer
replays the graph).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn

from pufferlib.environments.craftax.torch import Policy as _Policy
from craftax_c.environment import CraftaxCEnv


class Policy(_Policy):
    def __init__(self, env, cnn_channels=32, hidden_size=128, **kwargs):
        super().__init__(env, cnn_channels=cnn_channels, hidden_size=hidden_size, **kwargs)

    def forward(self, observations, state=None):
        hidden, lookup = self.encode_observations(observations)
        return self.decode_actions(hidden, lookup)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


class _E:
    class _S:
        n = 17
    single_action_space = _S()


# --------------------------------------------------------------------------
# Graph-captured minibatch update.
# --------------------------------------------------------------------------
class GraphPPOUpdate:
    """One-shot-capture of a PPO minibatch update.

    Usage:
        g = GraphPPOUpdate(policy, opt, minibatch_size, obs_dim, clip, ...)
        g.copy_inputs(obs_mb, act_mb, logp_mb, adv_mb, ret_mb, val_mb)
        g.replay()
    """

    def __init__(self, policy, opt, mb_size: int, obs_dim: int,
                 clip=0.2, ent_coef=0.01, vf_coef=0.5, device="cuda"):
        self.policy = policy
        self.opt = opt
        self.mb_size = mb_size
        self.clip = clip
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device

        # Static input buffers (persistent memory; the graph's data_ptr's).
        self.obs  = torch.zeros(mb_size, obs_dim, device=device)
        self.act  = torch.zeros(mb_size, dtype=torch.int64, device=device)
        self.logp = torch.zeros(mb_size, device=device)
        self.adv  = torch.zeros(mb_size, device=device)
        self.ret  = torch.zeros(mb_size, device=device)
        self.val  = torch.zeros(mb_size, device=device)

        self.loss_out = torch.zeros((), device=device)

        # Required: ensure params have .grad allocated before capture so
        # opt.step() has valid inputs. set_to_none=False keeps them as
        # persistent zeroed tensors between replays.
        for p in policy.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        # Warmup on a side stream -- this allocates all intermediate buffers
        # at their graph-time sizes before we capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(side)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

    def _step(self):
        """Body of one PPO update -- the thing we capture."""
        logits, v_new = self.policy(self.obs)
        v_new = v_new.squeeze(-1)
        lp_all = logits.log_softmax(-1)
        nlp = lp_all.gather(-1, self.act.unsqueeze(-1)).squeeze(-1)
        ent = -(lp_all.exp() * lp_all).sum(-1).mean()
        ratio = (nlp - self.logp).exp()
        pg = -torch.min(
            ratio * self.adv,
            ratio.clamp(1 - self.clip, 1 + self.clip) * self.adv,
        ).mean()
        vc = self.val + (v_new - self.val).clamp(-self.clip, self.clip)
        vl = 0.5 * torch.max(
            (v_new - self.ret).square(),
            (vc - self.ret).square(),
        ).mean()
        loss = pg + self.vf_coef * vl - self.ent_coef * ent

        # zero_grad(set_to_none=False) -- in-place so graph-captured.
        for p in self.policy.parameters():
            p.grad.zero_()
        loss.backward()
        self.opt.step()
        self.loss_out.copy_(loss.detach())

    @torch.no_grad()
    def copy_inputs(self, obs, act, logp, adv, ret, val):
        self.obs.copy_(obs, non_blocking=True)
        self.act.copy_(act, non_blocking=True)
        self.logp.copy_(logp, non_blocking=True)
        self.adv.copy_(adv, non_blocking=True)
        self.ret.copy_(ret, non_blocking=True)
        self.val.copy_(val, non_blocking=True)

    def replay(self):
        self.graph.replay()


# --------------------------------------------------------------------------
def compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.8):
    """GAE over (H, E). Returns (adv, ret) shaped (H, E)."""
    H, E = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgae = torch.zeros(E, device=rewards.device)
    for t in reversed(range(H)):
        nextnonterm = 1.0 - dones[t]
        nextv = last_value if t == H - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextv * nextnonterm - values[t]
        lastgae = delta + gamma * lam * nextnonterm * lastgae
        adv[t] = lastgae
    return adv, adv + values


def sample_gumbel(logits):
    u = torch.rand_like(logits).clamp_(1e-8, 1.0)
    a = (logits - torch.log(-torch.log(u))).argmax(-1)
    lp = logits.log_softmax(-1).gather(-1, a.unsqueeze(-1)).squeeze(-1)
    return a, lp


class GraphRollout:
    """Graph-captured single-step rollout op: static obs in, static (action,
    logprob, value) out. Randomness for Gumbel sampling is provided via a
    pre-filled noise buffer (regenerated on CPU each step) -- keeps the
    graph deterministic w.r.t. its inputs.
    """
    def __init__(self, policy, num_envs: int, obs_dim: int, n_actions: int,
                 device="cuda"):
        self.policy = policy
        self.obs_in    = torch.zeros(num_envs, obs_dim, device=device)
        self.noise_in  = torch.zeros(num_envs, n_actions, device=device)
        self.action_out  = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.logprob_out = torch.zeros(num_envs, device=device)
        self.value_out   = torch.zeros(num_envs, device=device)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(side)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

    @torch.no_grad()
    def _step(self):
        logits, value = self.policy.forward_eval(self.obs_in)
        # Gumbel-max: argmax(logits - log(-log(U)))
        # self.noise_in is expected to hold log(-log(U)) already -- generated CPU-side.
        a = (logits - self.noise_in).argmax(-1)
        lp = logits.log_softmax(-1).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        self.action_out.copy_(a)
        self.logprob_out.copy_(lp)
        self.value_out.copy_(value.squeeze(-1))

    def replay(self):
        self.graph.replay()


@torch.no_grad()
def rollout_graph(env, g_roll, obs_cpu_ref, action_int32_cpu,
                  horizon, num_envs, obs_dim, n_actions, device):
    obs_buf  = torch.zeros(horizon, num_envs, obs_dim, device=device)
    act_buf  = torch.zeros(horizon, num_envs, dtype=torch.int64, device=device)
    rew_buf  = torch.zeros(horizon, num_envs, device=device)
    done_buf = torch.zeros(horizon, num_envs, device=device)
    val_buf  = torch.zeros(horizon, num_envs, device=device)
    logp_buf = torch.zeros(horizon, num_envs, device=device)

    # Pre-generate Gumbel noise for all H steps on CPU once (amortized), copy
    # per-step. Uses pinned memory so the H2D can be async.
    u = torch.rand(horizon, num_envs, n_actions, device="cpu", pin_memory=True)
    u.clamp_(1e-8, 1.0).log_().neg_().log_()  # -log(-log(U))   shape (H, E, A)

    # If the env is backed by pinned memory, read obs directly from its
    # persistent tensor -- avoids per-step torch.from_numpy allocation and
    # enables async DMA from pinned host memory.
    obs_source = env.obs_tensor  # None if pinned_obs=False

    # Per-step reward/done scratch in pinned memory for async H2D.
    rew_cpu = torch.zeros(num_envs, dtype=torch.float32, pin_memory=True)
    dn_cpu  = torch.zeros(num_envs, dtype=torch.float32, pin_memory=True)

    for t in range(horizon):
        if obs_source is not None:
            g_roll.obs_in.copy_(obs_source, non_blocking=True)
        else:
            g_roll.obs_in.copy_(torch.from_numpy(obs_cpu_ref[0]), non_blocking=True)
        g_roll.noise_in.copy_(u[t], non_blocking=True)
        g_roll.replay()
        obs_buf[t].copy_(g_roll.obs_in, non_blocking=True)
        act_buf[t].copy_(g_roll.action_out, non_blocking=True)
        val_buf[t].copy_(g_roll.value_out, non_blocking=True)
        logp_buf[t].copy_(g_roll.logprob_out, non_blocking=True)
        action_int32_cpu.copy_(g_roll.action_out.to(torch.int32), non_blocking=True)
        torch.cuda.synchronize()

        _obs, rew, term, _trunc, _info = env.step(action_int32_cpu.numpy())
        rew_cpu.copy_(torch.from_numpy(np.asarray(rew, dtype=np.float32)))
        dn_cpu.copy_(torch.from_numpy(np.asarray(term, dtype=np.float32)))
        rew_buf[t].copy_(rew_cpu, non_blocking=True)
        done_buf[t].copy_(dn_cpu, non_blocking=True)
        obs_cpu_ref[0] = _obs

    if obs_source is not None:
        g_roll.obs_in.copy_(obs_source, non_blocking=True)
    else:
        g_roll.obs_in.copy_(torch.from_numpy(obs_cpu_ref[0]), non_blocking=True)
    g_roll.replay()
    last_v = g_roll.value_out.clone()
    return obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf, last_v


@torch.no_grad()
def rollout_plain(env, policy, obs_cpu_ref, obs_gpu, action_int32_cpu,
                  horizon, num_envs, obs_dim, device):
    """Collect a horizon-long rollout. Returns flat tensors on `device`."""
    obs_buf  = torch.zeros(horizon, num_envs, obs_dim, device=device)
    act_buf  = torch.zeros(horizon, num_envs, dtype=torch.int64, device=device)
    rew_buf  = torch.zeros(horizon, num_envs, device=device)
    done_buf = torch.zeros(horizon, num_envs, device=device)
    val_buf  = torch.zeros(horizon, num_envs, device=device)
    logp_buf = torch.zeros(horizon, num_envs, device=device)

    for t in range(horizon):
        obs_gpu.copy_(torch.from_numpy(obs_cpu_ref[0]), non_blocking=True)
        obs_buf[t].copy_(obs_gpu, non_blocking=True)
        logits, value = policy.forward_eval(obs_gpu)
        action, logprob = sample_gumbel(logits)
        act_buf[t] = action
        val_buf[t] = value.squeeze(-1)
        logp_buf[t] = logprob
        action_int32_cpu.copy_(action.to(torch.int32), non_blocking=True)
        torch.cuda.synchronize()  # cheap -- single stream

        _obs, rew, term, _trunc, _info = env.step(action_int32_cpu.numpy())
        rew_buf[t].copy_(torch.from_numpy(np.asarray(rew, dtype=np.float32)),
                          non_blocking=True)
        done_buf[t].copy_(torch.from_numpy(np.asarray(term, dtype=np.float32)),
                          non_blocking=True)
        obs_cpu_ref[0] = _obs

    # Bootstrap value.
    obs_gpu.copy_(torch.from_numpy(obs_cpu_ref[0]), non_blocking=True)
    _, last_v = policy.forward_eval(obs_gpu)
    last_v = last_v.squeeze(-1)

    return obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf, last_v


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--graph", action="store_true", default=False,
                    help="Enable CUDA graph capture of minibatch update.")
    ap.add_argument("--graph-rollout", action="store_true", default=False,
                    help="Also graph-capture the per-step rollout op.")
    ap.add_argument("--hidden", type=int, default=128,
                    help="Hidden size of the policy trunk/flat encoder.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    env = CraftaxCEnv(num_envs=args.num_envs)
    obs_initial, _ = env.reset()
    obs_dim = env.single_observation_space.shape[0]

    policy = Policy(env, hidden_size=args.hidden).to(device)
    opt = torch.optim.Adam(
        policy.parameters(), lr=args.lr, eps=1e-5,
        fused=(device == "cuda"),
        capturable=(device == "cuda"),
    )

    obs_gpu = torch.zeros(args.num_envs, obs_dim, device=device)
    action_int32_cpu = torch.zeros(args.num_envs, dtype=torch.int32,
                                   device="cpu", pin_memory=(device == "cuda"))
    obs_cpu_ref = [obs_initial]

    graph_updater = None
    if args.graph:
        graph_updater = GraphPPOUpdate(
            policy, opt, mb_size=args.minibatch, obs_dim=obs_dim,
            device=device,
        )

    g_roll = None
    if args.graph_rollout:
        g_roll = GraphRollout(
            policy, args.num_envs, obs_dim,
            n_actions=env.single_action_space.n, device=device,
        )

    # Warmup
    _ = rollout_plain(env, policy, obs_cpu_ref, obs_gpu, action_int32_cpu,
                     args.horizon, args.num_envs, obs_dim, device)
    torch.cuda.synchronize()

    print(f"graph={args.graph}  graph_rollout={args.graph_rollout}  "
          f"horizon={args.horizon}  num_envs={args.num_envs}  "
          f"minibatch={args.minibatch}  epochs={args.epochs}")

    t_start = time.time()
    step_count = 0
    H, E, MB = args.horizon, args.num_envs, args.minibatch
    for it in range(args.iters):
        if args.graph_rollout:
            obs_b, act_b, rew_b, done_b, val_b, logp_b, last_v = rollout_graph(
                env, g_roll, obs_cpu_ref, action_int32_cpu,
                args.horizon, args.num_envs, obs_dim,
                env.single_action_space.n, device,
            )
        else:
            obs_b, act_b, rew_b, done_b, val_b, logp_b, last_v = rollout_plain(
                env, policy, obs_cpu_ref, obs_gpu, action_int32_cpu,
                args.horizon, args.num_envs, obs_dim, device,
            )
        adv, ret = compute_gae(rew_b, done_b, val_b, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_f  = obs_b.reshape(H*E, -1)
        act_f  = act_b.reshape(-1)
        logp_f = logp_b.reshape(-1)
        adv_f  = adv.reshape(-1)
        ret_f  = ret.reshape(-1)
        val_f  = val_b.reshape(-1)

        N = H * E
        for _ in range(args.epochs):
            perm = torch.randperm(N, device=device)
            for start in range(0, N, MB):
                idx = perm[start:start+MB]
                if idx.shape[0] < MB:  # skip ragged tail for graph version
                    if args.graph: continue
                if args.graph:
                    graph_updater.copy_inputs(
                        obs_f[idx], act_f[idx], logp_f[idx],
                        adv_f[idx], ret_f[idx], val_f[idx],
                    )
                    graph_updater.replay()
                else:
                    # Eager fallback (matches train_pipelined.ppo_update math)
                    logits, v_new = policy(obs_f[idx])
                    v_new = v_new.squeeze(-1)
                    lp_all = logits.log_softmax(-1)
                    nlp = lp_all.gather(-1, act_f[idx].unsqueeze(-1)).squeeze(-1)
                    ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                    ratio = (nlp - logp_f[idx]).exp()
                    pg = -torch.min(
                        ratio * adv_f[idx],
                        ratio.clamp(0.8, 1.2) * adv_f[idx],
                    ).mean()
                    vc = val_f[idx] + (v_new - val_f[idx]).clamp(-0.2, 0.2)
                    vl = 0.5 * torch.max(
                        (v_new - ret_f[idx]).square(),
                        (vc - ret_f[idx]).square(),
                    ).mean()
                    loss = pg + 0.5 * vl - 0.01 * ent
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
        step_count += H * E

    torch.cuda.synchronize()
    elapsed = time.time() - t_start

    # Quick learning signal: entropy (uniform over 17 actions => ln(17) ≈ 2.833).
    with torch.no_grad():
        obs_sample = obs_b.reshape(-1, obs_dim)[:4096]
        logits, _ = policy(obs_sample)
        lp = logits.log_softmax(-1)
        entropy = -(lp.exp() * lp).sum(-1).mean().item()
    print(f"  time: {elapsed:.2f}s   steps: {step_count:,}   SPS: {step_count/elapsed:,.0f}   final_entropy={entropy:.3f}")


if __name__ == "__main__":
    main()
