"""torch.profiler sweep of the PPO minibatch update.

Captures kernel-level timing of forward + backward + optimizer on the
45k-param Craftax policy, with and without graph capture / bf16 autocast,
so we can see where Blackwell (sm_120) leverage points live.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity, record_function, schedule

from pufferlib.environments.craftax.torch import (
    Policy as _Policy, N_MAP, N_FLAT,
)


class Policy(_Policy):
    def forward(self, observations, state=None):
        hidden, lookup = self.encode_observations(observations)
        return self.decode_actions(hidden, lookup)


class _E:
    class _S: n = 17
    single_action_space = _S()


def build(minibatch: int, device: str = "cuda", dtype=torch.float32, hidden=128):
    policy = Policy(_E(), hidden_size=hidden).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5,
                           fused=True, capturable=True)
    obs  = torch.randn(minibatch, N_MAP + N_FLAT, device=device, dtype=dtype)
    act  = torch.randint(0, 17, (minibatch,), device=device)
    logp = torch.randn(minibatch, device=device, dtype=dtype)
    adv  = torch.randn(minibatch, device=device, dtype=dtype)
    ret  = torch.randn(minibatch, device=device, dtype=dtype)
    val  = torch.randn(minibatch, device=device, dtype=dtype)
    for p in policy.parameters():
        if p.grad is None: p.grad = torch.zeros_like(p)
    return policy, opt, (obs, act, logp, adv, ret, val)


def one_ppo_step(policy, opt, obs, act, logp, adv, ret, val, autocast_dtype=None):
    for p in policy.parameters(): p.grad.zero_()
    if autocast_dtype is not None:
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            logits, v_new = policy(obs)
            v_new = v_new.squeeze(-1)
            lp_all = logits.log_softmax(-1)
            nlp = lp_all.gather(-1, act.unsqueeze(-1)).squeeze(-1)
            ent = -(lp_all.exp() * lp_all).sum(-1).mean()
            ratio = (nlp - logp).exp()
            pg = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            vc = val + (v_new - val).clamp(-0.2, 0.2)
            vl = 0.5 * torch.max((v_new - ret).square(), (vc - ret).square()).mean()
            loss = pg + 0.5 * vl - 0.01 * ent
    else:
        logits, v_new = policy(obs)
        v_new = v_new.squeeze(-1)
        lp_all = logits.log_softmax(-1)
        nlp = lp_all.gather(-1, act.unsqueeze(-1)).squeeze(-1)
        ent = -(lp_all.exp() * lp_all).sum(-1).mean()
        ratio = (nlp - logp).exp()
        pg = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
        vc = val + (v_new - val).clamp(-0.2, 0.2)
        vl = 0.5 * torch.max((v_new - ret).square(), (vc - ret).square()).mean()
        loss = pg + 0.5 * vl - 0.01 * ent
    loss.backward()
    opt.step()


def bench(policy, opt, bufs, iters, autocast_dtype=None):
    # Warmup
    for _ in range(10): one_ppo_step(policy, opt, *bufs, autocast_dtype=autocast_dtype)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): one_ppo_step(policy, opt, *bufs, autocast_dtype=autocast_dtype)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e6  # us


def profile_run(label, policy, opt, bufs, steps=20, autocast_dtype=None):
    print(f"\n--- profiler trace: {label} ---")
    # Warmup
    for _ in range(10): one_ppo_step(policy, opt, *bufs, autocast_dtype=autocast_dtype)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(steps):
            with record_function("ppo_step"):
                one_ppo_step(policy, opt, *bufs, autocast_dtype=autocast_dtype)
        torch.cuda.synchronize()

    # Summary by CUDA self time
    events = prof.key_averages()
    total_self_cuda = sum(e.self_device_time_total for e in events)
    rows = sorted(events, key=lambda e: e.self_device_time_total, reverse=True)
    print(f"  total CUDA self time: {total_self_cuda/steps:.0f} us/step")
    print(f"  {'op':55s} {'cuda us/step':>12s} {'% of step':>10s} {'calls':>7s}")
    cum = 0.0
    for e in rows[:20]:
        us = e.self_device_time_total / steps
        if us < 1: continue
        cum += us
        pct = 100 * e.self_device_time_total / max(total_self_cuda, 1)
        print(f"  {e.key[:55]:55s} {us:>12.1f} {pct:>9.1f}% {e.count/steps:>7.1f}")
    print(f"  {'...top 20 total...':55s} {cum:>12.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minibatch", type=int, default=16384)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--profile-steps", type=int, default=20)
    args = ap.parse_args()

    # Quick wallclock bench across dtypes
    print(f"Wallclock one-PPO-step at MB={args.minibatch}, hidden={args.hidden}:")
    for (label, dtype, autocast) in [
        ("fp32",            torch.float32, None),
        ("fp32+autocast bf16", torch.float32, torch.bfloat16),
        ("fp32+autocast fp16", torch.float32, torch.float16),
    ]:
        policy, opt, bufs = build(args.minibatch, dtype=dtype, hidden=args.hidden)
        us = bench(policy, opt, bufs, args.iters, autocast_dtype=autocast)
        sps = args.minibatch * 1e6 / us
        print(f"  {label:25s}  {us:>7.1f} us/step  ->  {sps:>12,.0f} samples/sec/step")

    # Kernel-level profile on fp32
    policy, opt, bufs = build(args.minibatch, dtype=torch.float32, hidden=args.hidden)
    profile_run("fp32 eager", policy, opt, bufs, steps=args.profile_steps)

    # Kernel-level profile on autocast bf16
    policy, opt, bufs = build(args.minibatch, dtype=torch.float32, hidden=args.hidden)
    profile_run("autocast bf16", policy, opt, bufs, steps=args.profile_steps,
                autocast_dtype=torch.bfloat16)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
