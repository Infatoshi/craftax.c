"""Per-stage timing of the graph-captured training loop.

Instruments every phase with cuda.synchronize at boundaries so we can see
which stage is the actual wall-clock bottleneck at a fixed config. No
averaging noise -- run 20 iters, report median of each per-iter stage.
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

from pufferlib.environments.craftax.torch import Policy as _Policy
from craftax_c.environment import CraftaxCEnv

from train_graph import (
    Policy, GraphPPOUpdate, GraphRollout, compute_gae, rollout_plain,
)
from craftax_c.bindings import OBS_DIM_COMPACT


def sync():
    torch.cuda.synchronize()


def now():
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--compact", action="store_true", default=False)
    args = ap.parse_args()

    device = "cuda"
    env = CraftaxCEnv(num_envs=args.num_envs, compact_obs=args.compact)
    if args.compact:
        env.reset_compact()
        obs0 = env.observations
    else:
        obs0, _ = env.reset()
    obs_dim = env.single_observation_space.shape[0]
    n_actions = env.single_action_space.n

    policy = Policy(env).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5,
                           fused=True, capturable=True)

    g_upd  = GraphPPOUpdate(policy, opt, args.minibatch, obs_dim, device=device)
    g_roll = GraphRollout(policy, args.num_envs, obs_dim, n_actions, device=device,
                          compact=args.compact)

    action_i32_cpu = torch.zeros(args.num_envs, dtype=torch.int32,
                                  device="cpu", pin_memory=True)
    obs_cpu_ref = [obs0]

    H, E, MB = args.horizon, args.num_envs, args.minibatch

    # Warmup one full iter (same code path as below).
    obs_buf  = torch.zeros(H, E, obs_dim, device=device)
    act_buf  = torch.zeros(H, E, dtype=torch.int64, device=device)
    rew_buf  = torch.zeros(H, E, device=device)
    done_buf = torch.zeros(H, E, device=device)
    val_buf  = torch.zeros(H, E, device=device)
    logp_buf = torch.zeros(H, E, device=device)

    # Dict-of-lists for per-stage times (ms).
    stages = {
        "roll_obs_h2d":    [],  # CPU obs -> GPU, per horizon step, summed over H
        "roll_noise_copy": [],
        "roll_graph":      [],
        "roll_buf_append": [],
        "roll_act_d2h":    [],
        "roll_sync":       [],
        "env_step_cpu":    [],  # env.step runtime (CPU-side)
        "roll_rew_done":   [],
        "gae":             [],  # GAE python loop + norm
        "reshape":         [],
        "mb_index_copy":   [],  # per epoch, summed over (#MB * epochs)
        "mb_graph":        [],  # per epoch, summed
        "perm":            [],  # randperm per epoch, summed
        "total_rollout":   [],
        "total_update":    [],
        "total_iter":      [],
    }

    for it in range(args.iters + 2):  # +2 warmup
        t_iter = now()
        # ---------------- Rollout ----------------
        t_roll_start = now()
        u = torch.rand(H, E, n_actions, device="cpu", pin_memory=True)
        u.clamp_(1e-8, 1.0).log_().neg_().log_()

        t_obs = t_noise = t_graph = t_buf = t_d2h = t_sync = t_env = t_rew = 0.0
        if args.compact:
            obs_source = env.compact_obs_tensor  # pinned uint8 (E, 145)
        else:
            obs_source = env.obs_tensor  # pinned float (E, 1345)
        for t in range(H):
            s = now()
            if obs_source is not None:
                g_roll.obs_in.copy_(obs_source, non_blocking=True)
            else:
                g_roll.obs_in.copy_(torch.from_numpy(obs_cpu_ref[0]), non_blocking=True)
            sync(); t_obs += now() - s
            s = now()
            g_roll.noise_in.copy_(u[t], non_blocking=True)
            sync(); t_noise += now() - s
            s = now()
            g_roll.replay()
            sync(); t_graph += now() - s
            s = now()
            # In compact mode, the (B,1345) float obs for the rollout buffer
            # lives in g_roll.obs_expanded, not obs_in (which is uint8 145).
            obs_buf[t].copy_(
                g_roll.obs_expanded if args.compact else g_roll.obs_in,
                non_blocking=True,
            )
            act_buf[t].copy_(g_roll.action_out, non_blocking=True)
            val_buf[t].copy_(g_roll.value_out, non_blocking=True)
            logp_buf[t].copy_(g_roll.logprob_out, non_blocking=True)
            sync(); t_buf += now() - s
            s = now()
            action_i32_cpu.copy_(g_roll.action_out.to(torch.int32), non_blocking=True)
            sync(); t_d2h += now() - s
            s = now(); sync(); t_sync += now() - s
            s = now()
            if args.compact:
                _obs, rew, term, _trunc, _info = env.step_compact(action_i32_cpu.numpy())
            else:
                _obs, rew, term, _trunc, _info = env.step(action_i32_cpu.numpy())
            t_env += now() - s
            s = now()
            rew_buf[t].copy_(
                torch.from_numpy(np.asarray(rew, dtype=np.float32)), non_blocking=True)
            done_buf[t].copy_(
                torch.from_numpy(np.asarray(term, dtype=np.float32)), non_blocking=True)
            sync(); t_rew += now() - s
            obs_cpu_ref[0] = _obs

        if obs_source is not None:
            g_roll.obs_in.copy_(obs_source, non_blocking=True)
        else:
            g_roll.obs_in.copy_(torch.from_numpy(obs_cpu_ref[0]), non_blocking=True)
        g_roll.replay()
        last_v = g_roll.value_out.clone()
        sync()
        t_roll_total = now() - t_roll_start

        # ---------------- GAE ----------------
        s = now()
        adv, ret = compute_gae(rew_buf, done_buf, val_buf, last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        sync(); t_gae = now() - s

        # ---------------- Reshape ----------------
        s = now()
        obs_f  = obs_buf.reshape(H*E, -1)
        act_f  = act_buf.reshape(-1)
        logp_f = logp_buf.reshape(-1)
        adv_f  = adv.reshape(-1)
        ret_f  = ret.reshape(-1)
        val_f  = val_buf.reshape(-1)
        sync(); t_reshape = now() - s

        # ---------------- PPO update ----------------
        t_upd_start = now()
        t_perm = t_idx = t_g = 0.0
        N = H * E
        for _ in range(args.epochs):
            s = now()
            perm = torch.randperm(N, device=device)
            sync(); t_perm += now() - s
            for start in range(0, N, MB):
                idx = perm[start:start+MB]
                if idx.shape[0] < MB: continue
                s = now()
                g_upd.copy_inputs(
                    obs_f[idx], act_f[idx], logp_f[idx],
                    adv_f[idx], ret_f[idx], val_f[idx],
                )
                sync(); t_idx += now() - s
                s = now()
                g_upd.replay()
                sync(); t_g += now() - s
        t_upd_total = now() - t_upd_start

        t_iter_total = now() - t_iter

        if it < 2:
            continue  # warmup

        stages["roll_obs_h2d"].append(t_obs * 1e3)
        stages["roll_noise_copy"].append(t_noise * 1e3)
        stages["roll_graph"].append(t_graph * 1e3)
        stages["roll_buf_append"].append(t_buf * 1e3)
        stages["roll_act_d2h"].append(t_d2h * 1e3)
        stages["roll_sync"].append(t_sync * 1e3)
        stages["env_step_cpu"].append(t_env * 1e3)
        stages["roll_rew_done"].append(t_rew * 1e3)
        stages["gae"].append(t_gae * 1e3)
        stages["reshape"].append(t_reshape * 1e3)
        stages["perm"].append(t_perm * 1e3)
        stages["mb_index_copy"].append(t_idx * 1e3)
        stages["mb_graph"].append(t_g * 1e3)
        stages["total_rollout"].append(t_roll_total * 1e3)
        stages["total_update"].append(t_upd_total * 1e3)
        stages["total_iter"].append(t_iter_total * 1e3)

    print(f"\nConfig: NE={args.num_envs} H={args.horizon} "
          f"epochs={args.epochs} MB={args.minibatch}")
    print(f"Per PPO iter = {args.horizon} rollout steps + "
          f"{args.epochs*(args.horizon*args.num_envs//args.minibatch)} minibatch updates\n")

    # Only report median.
    def med(k): return statistics.median(stages[k])
    total = med("total_iter")

    print(f"{'stage':24s} {'ms/iter':>10s}  {'% of iter':>10s}")
    print("-" * 52)
    rollout_sub = [
        "roll_obs_h2d", "roll_noise_copy", "roll_graph",
        "roll_buf_append", "roll_act_d2h", "roll_sync",
        "env_step_cpu", "roll_rew_done",
    ]
    for k in rollout_sub:
        v = med(k)
        print(f"  {k:22s} {v:10.2f}  {100*v/total:10.1f}")
    print(f"  {'ROLLOUT TOTAL':22s} {med('total_rollout'):10.2f}  {100*med('total_rollout')/total:10.1f}")
    print()
    print(f"  {'gae + norm':22s} {med('gae'):10.2f}  {100*med('gae')/total:10.1f}")
    print(f"  {'reshape':22s} {med('reshape'):10.2f}  {100*med('reshape')/total:10.1f}")
    print()
    for k in ["perm", "mb_index_copy", "mb_graph"]:
        v = med(k)
        print(f"  {k:22s} {v:10.2f}  {100*v/total:10.1f}")
    print(f"  {'UPDATE TOTAL':22s} {med('total_update'):10.2f}  {100*med('total_update')/total:10.1f}")
    print("-" * 52)
    print(f"  {'ITER TOTAL':22s} {total:10.2f}")
    sps = args.horizon * args.num_envs / (total / 1000.0)
    print(f"\n  Implied SPS: {sps:,.0f}")

    # Print the single slowest stage
    candidates = {k: med(k) for k in rollout_sub + ["gae", "reshape", "perm", "mb_index_copy", "mb_graph"]}
    worst = max(candidates.items(), key=lambda kv: kv[1])
    print(f"\n  Slowest stage: {worst[0]} at {worst[1]:.2f} ms ({100*worst[1]/total:.1f}% of iter)")


if __name__ == "__main__":
    main()
