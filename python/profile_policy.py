"""Shape + timing dump for PufferLib's default Craftax-Classic Policy.

What each layer produces, how many params, and how long forward / backward /
optimizer take at realistic minibatch sizes. Useful for deciding which part
of the training loop to optimize.
"""
from __future__ import annotations

import argparse
import time
import torch
from torch import nn

from pufferlib.environments.craftax.torch import Policy, N_MAP, N_FLAT, CRAFTAX_ROWS, CRAFTAX_COLS, CRAFTAX_CHANNELS
import pufferlib


class DummyEnv:
    class _Space:
        def __init__(self, n): self.n = n
    single_action_space = _Space(17)


def hook_and_forward(policy, obs):
    """Run forward, capture per-module output shapes."""
    shapes: list[tuple[str, tuple[int, ...], int]] = []
    params: dict[str, int] = {}
    for name, m in policy.named_modules():
        if name == "" or list(m.children()):  # skip root and containers
            continue
        params[name] = sum(p.numel() for p in m.parameters(recurse=False))
        def _mk(n):
            def hook(mod, inp, out):
                shapes.append((n, tuple(out.shape), params[n]))
            return hook
        m.register_forward_hook(_mk(name))
    policy(obs)
    return shapes


def measure(name, fn, iters, device):
    # warmup
    for _ in range(5): fn()
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn()
    if device == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e6  # µs/iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    device = args.device

    print(f"Device: {device}   minibatch: {args.minibatch}   iters: {args.iters}")
    print(f"N_MAP={N_MAP}  map={CRAFTAX_ROWS}x{CRAFTAX_COLS}x{CRAFTAX_CHANNELS}  N_FLAT={N_FLAT}")

    env = DummyEnv()
    policy = Policy(env).to(device)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total params: {total_params:,}")

    obs = torch.randn(args.minibatch, N_MAP + N_FLAT, device=device)

    # --- layer shape dump (single forward)
    print("\n-- Layer shapes (batch=4) --")
    small = torch.randn(4, N_MAP + N_FLAT, device=device)
    shapes = hook_and_forward(policy, small)
    for name, shp, pm in shapes:
        print(f"  {name:38s}  shape={str(shp):28s}  params={pm:>8,}")

    # --- timing
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)

    def fwd_only():
        with torch.no_grad():
            policy(obs)

    def fwd_train():
        policy(obs)

    def full_step():
        opt.zero_grad(set_to_none=True)
        logits, value = policy(obs)
        # Fake PPO-style losses
        loss = (logits.square().mean() + value.square().mean() + 0.01 * logits.abs().mean())
        loss.backward()
        opt.step()

    def bwd_only():
        opt.zero_grad(set_to_none=True)
        logits, value = policy(obs)
        (logits.square().mean() + value.square().mean()).backward()

    t_fwd_nograd = measure("fwd_nograd", fwd_only, args.iters, device)
    t_fwd_train  = measure("fwd_train",  fwd_train, args.iters, device)
    t_bwd        = measure("fwd+bwd",    bwd_only, args.iters, device)
    t_full       = measure("full_step",  full_step, args.iters, device)

    print("\n-- Per-minibatch timings (µs) --")
    print(f"  forward (no_grad):       {t_fwd_nograd:8.1f} µs")
    print(f"  forward (train mode):    {t_fwd_train:8.1f} µs")
    print(f"  forward + backward:      {t_bwd:8.1f} µs   (backward alone ~ {t_bwd - t_fwd_train:.1f})")
    print(f"  forward + bwd + Adam:    {t_full:8.1f} µs   (Adam alone     ~ {t_full - t_bwd:.1f})")

    # Throughput equivalents
    sps_full = args.minibatch / (t_full / 1e6)
    print(f"\n  minibatch throughput: {sps_full:>12,.0f} samples/sec")
    print(f"  updates/sec at this minibatch: {1e6 / t_full:.0f}")


if __name__ == "__main__":
    main()
