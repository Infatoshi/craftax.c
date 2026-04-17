"""Sweep minibatch size + test torch.compile + fused Adam. Quantify the
launch-overhead-vs-compute tradeoff on the default Craftax policy."""
import time
import torch
from pufferlib.environments.craftax.torch import Policy, N_MAP, N_FLAT

class _E:
    class _S:
        n = 17
    single_action_space = _S()

def time_full_step(policy, opt, obs, iters=300):
    # Warmup
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        logits, value = policy(obs)
        (logits.square().mean() + value.square().mean()).backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        logits, value = policy(obs)
        (logits.square().mean() + value.square().mean()).backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e6  # µs/iter

def main():
    device = "cuda"
    results = []
    for mb in [1024, 4096, 16384, 65536]:
        obs = torch.randn(mb, N_MAP + N_FLAT, device=device)
        # plain
        p = Policy(_E()).to(device)
        o = torch.optim.Adam(p.parameters(), lr=3e-4, eps=1e-5)
        t_plain = time_full_step(p, o, obs)

        # fused adam
        p = Policy(_E()).to(device)
        o = torch.optim.Adam(p.parameters(), lr=3e-4, eps=1e-5, fused=True)
        t_fused = time_full_step(p, o, obs)

        # compile
        p = Policy(_E()).to(device)
        p_c = torch.compile(p, mode="reduce-overhead")
        o = torch.optim.Adam(p.parameters(), lr=3e-4, eps=1e-5, fused=True)
        # compile warmup
        def step_compile():
            o.zero_grad(set_to_none=True)
            logits, value = p_c(obs)
            (logits.square().mean() + value.square().mean()).backward()
            o.step()
        for _ in range(30): step_compile()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(300): step_compile()
        torch.cuda.synchronize()
        t_compile = (time.time() - t0) / 300 * 1e6

        sps_plain = mb / (t_plain / 1e6)
        sps_fused = mb / (t_fused / 1e6)
        sps_comp  = mb / (t_compile / 1e6)
        results.append((mb, t_plain, t_fused, t_compile, sps_plain, sps_fused, sps_comp))
        print(f"MB={mb:>6}  plain={t_plain:>7.1f}µs  fused={t_fused:>7.1f}µs  compile={t_compile:>7.1f}µs"
              f"   SPS: plain={sps_plain:>12,.0f}  fused={sps_fused:>12,.0f}  compile={sps_comp:>12,.0f}")

if __name__ == "__main__":
    main()
