"""Check Triton fused PPO loss matches the eager reference, and microbench."""
import time
import torch
from triton_ppo import ppo_loss_eager, ppo_loss_triton


def check_correctness(B=16384, A=17, tol=1e-3):
    torch.manual_seed(0)
    device = "cuda"
    logits   = torch.randn(B, A, device=device, requires_grad=True)
    v_new    = torch.randn(B, device=device, requires_grad=True)
    act      = torch.randint(0, A, (B,), device=device)
    logp_old = torch.randn(B, device=device)
    adv      = torch.randn(B, device=device)
    ret      = torch.randn(B, device=device)
    val_old  = torch.randn(B, device=device)

    # Eager
    l_eager = ppo_loss_eager(logits, v_new, act, logp_old, adv, ret, val_old)
    g_lo_e, g_vn_e = torch.autograd.grad(l_eager, [logits, v_new], retain_graph=False)

    # Triton
    logits2 = logits.detach().clone().requires_grad_(True)
    v_new2  = v_new.detach().clone().requires_grad_(True)
    l_tri = ppo_loss_triton(logits2, v_new2, act, logp_old, adv, ret, val_old)
    g_lo_t, g_vn_t = torch.autograd.grad(l_tri, [logits2, v_new2])

    def err(a, b, name):
        d = (a - b).abs()
        rel = d / (b.abs() + 1e-6)
        print(f"  {name:20s}  max_abs={d.max().item():.2e}  "
              f"max_rel={rel.max().item():.2e}  mean_abs={d.mean().item():.2e}")
        return d.max().item()

    print(f"Correctness at B={B}, A={A}:")
    e1 = abs(l_tri.item() - l_eager.item())
    print(f"  loss scalar diff     abs={e1:.2e}  eager={l_eager.item():.6f}")
    e2 = err(g_lo_t, g_lo_e, "grad_logits")
    e3 = err(g_vn_t, g_vn_e, "grad_v_new")
    ok = max(e1, e2, e3) < tol
    print(f"  {'PASS' if ok else 'FAIL'} (tol={tol})")
    return ok


def microbench(B=16384, A=17, iters=1000):
    torch.manual_seed(0)
    device = "cuda"
    def mk(requires_grad=False):
        return (
            torch.randn(B, A, device=device, requires_grad=requires_grad),
            torch.randn(B, device=device, requires_grad=requires_grad),
            torch.randint(0, A, (B,), device=device),
            torch.randn(B, device=device),
            torch.randn(B, device=device),
            torch.randn(B, device=device),
            torch.randn(B, device=device),
        )

    def bench(fn, label):
        # Warmup
        for _ in range(50):
            logits, v_new, act, logp_old, adv, ret, val_old = mk(True)
            loss = fn(logits, v_new, act, logp_old, adv, ret, val_old)
            loss.backward()
        torch.cuda.synchronize()
        # Actual measurement with static tensors to avoid alloc overhead
        logits, v_new, act, logp_old, adv, ret, val_old = mk(True)
        t0 = time.time()
        for _ in range(iters):
            logits.grad = None
            v_new.grad = None
            loss = fn(logits, v_new, act, logp_old, adv, ret, val_old)
            loss.backward()
        torch.cuda.synchronize()
        us = (time.time() - t0) / iters * 1e6
        print(f"  {label:20s}  {us:>7.2f} us/call")
        return us

    print(f"\nMicrobench at B={B}, A={A}, iters={iters}:")
    t_eager = bench(ppo_loss_eager,  "eager (15 ops)")
    t_tri   = bench(ppo_loss_triton, "Triton fused")
    print(f"  speedup: {t_eager / t_tri:.2f}x")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    ok = check_correctness(B=16384, A=17)
    microbench(B=16384, A=17, iters=500)
    # Also check at smaller/larger sizes for stability
    print("\n--- smaller batch ---")
    check_correctness(B=4096, A=17)
    print("\n--- larger batch ---")
    check_correctness(B=65536, A=17)
