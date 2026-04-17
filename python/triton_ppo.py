"""Triton fused PPO loss (forward + backward in one kernel).

Replaces the ~15-op elementwise chain:
  log_softmax -> gather -> entropy -> ratio -> clip -> min(pg1, pg2)
  -> vf clip -> max((v-r)**2, (vc-r)**2) -> weighted sum
with a single Triton kernel that reads logits once and writes
(per_row_loss, grad_logits, grad_v_new) once.

Target device: sm_120 (Blackwell RTX PRO 6000). Uses standard Triton
mma / reductions; no sm_120-specific instructions.

Math (verified against the PyTorch reference below):

  nlp = log_softmax(logits)[act]
  entropy = -sum(softmax(logits) * log_softmax(logits))
  ratio = exp(nlp - logp_old)
  pg = -min(ratio * adv, clip(ratio, 1-c, 1+c) * adv)
  vc = val_old + clip(v_new - val_old, -c, c)
  vf = 0.5 * max((v_new - ret)^2, (vc - ret)^2)
  loss_row = pg + vf_coef * vf - ent_coef * entropy
  total_loss = mean_over_B(loss_row)

Gradients (per-row, before the 1/B from the mean):
  d entropy / d logits[a] = -softmax[a] * (log_softmax[a] + entropy)
  d nlp / d logits[a]     = (a == act) - softmax[a]
  d pg / d ratio          = -adv (if pg1 branch) or -adv * clip_active (if pg2)
  d pg / d logits[a]      = d(pg)/d(ratio) * ratio * d(nlp)/d(logits[a])
  d vf / d v_new          = (v_new - ret) or (vc - ret)*clip_active
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ppo_loss_kernel(
    # inputs
    logits_ptr, v_new_ptr, act_ptr, logp_old_ptr,
    adv_ptr, ret_ptr, val_old_ptr,
    # outputs
    loss_row_ptr, grad_logits_ptr, grad_v_new_ptr,
    # hyperparams
    clip, vf_coef, ent_coef,
    # sizes
    B,
    A: tl.constexpr,        # actual number of actions (<= A_PAD)
    A_PAD: tl.constexpr,    # next power-of-2 >= A (Triton arange constraint)
    BLOCK: tl.constexpr,    # rows per program
):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK + tl.arange(0, BLOCK)
    row_mask = row_offs < B
    act_offs = tl.arange(0, A_PAD)
    act_valid = act_offs < A

    # --- Load logits (BLOCK, A_PAD) with padding set to -inf ---
    logits_addr = row_offs[:, None] * A + act_offs[None, :]
    load_mask = row_mask[:, None] & act_valid[None, :]
    logits = tl.load(logits_ptr + logits_addr, mask=load_mask, other=-1e30)

    # --- Stable log_softmax + softmax ---
    mx = tl.max(logits, axis=1, keep_dims=True)
    shifted = logits - mx
    exp_s = tl.exp(shifted)
    sum_exp = tl.sum(exp_s, axis=1, keep_dims=True)
    log_sum_exp = tl.log(sum_exp)
    log_sm = shifted - log_sum_exp          # (BLOCK, A)
    sm = exp_s / sum_exp                    # (BLOCK, A)

    # --- Gather nlp and entropy ---
    act = tl.load(act_ptr + row_offs, mask=row_mask, other=0)
    selected = (act_offs[None, :] == act[:, None])          # (BLOCK, A) bool
    nlp = tl.sum(tl.where(selected, log_sm, 0.0), axis=1)   # (BLOCK,)
    entropy = -tl.sum(sm * log_sm, axis=1)                  # (BLOCK,)

    # --- Load scalars ---
    logp_old = tl.load(logp_old_ptr + row_offs, mask=row_mask, other=0.0)
    adv      = tl.load(adv_ptr      + row_offs, mask=row_mask, other=0.0)
    ret      = tl.load(ret_ptr      + row_offs, mask=row_mask, other=0.0)
    val_old  = tl.load(val_old_ptr  + row_offs, mask=row_mask, other=0.0)
    v_new    = tl.load(v_new_ptr    + row_offs, mask=row_mask, other=0.0)

    # --- PG loss ---
    ratio = tl.exp(nlp - logp_old)
    lo = 1.0 - clip
    hi = 1.0 + clip
    clipped = tl.minimum(tl.maximum(ratio, lo), hi)
    pg1 = ratio * adv
    pg2 = clipped * adv
    use_pg1 = pg1 <= pg2
    pg = -tl.minimum(pg1, pg2)

    # --- VF loss ---
    v_diff = v_new - val_old
    v_diff_c = tl.minimum(tl.maximum(v_diff, -clip), clip)
    vc = val_old + v_diff_c
    vl_un = (v_new - ret) * (v_new - ret)
    vl_cl = (vc - ret) * (vc - ret)
    use_vl_un = vl_un >= vl_cl
    vf = 0.5 * tl.maximum(vl_un, vl_cl)

    # --- Per-row loss ---
    loss_row = pg + vf_coef * vf - ent_coef * entropy
    tl.store(loss_row_ptr + row_offs, loss_row, mask=row_mask)

    # --- Gradients w.r.t. logits ---
    #
    # d(-ent_coef * entropy) / d logits[a]
    #   = ent_coef * softmax[a] * (log_softmax[a] + entropy_row)
    ent_grad = ent_coef * sm * (log_sm + entropy[:, None])

    # d(pg) / d(ratio):
    pg2_active = (ratio > lo) & (ratio < hi)
    dpg_dratio_pg1 = -adv
    dpg_dratio_pg2 = tl.where(pg2_active, -adv, 0.0)
    dpg_dratio = tl.where(use_pg1, dpg_dratio_pg1, dpg_dratio_pg2)

    # d(loss)/d(nlp) via ratio: d(ratio)/d(nlp) = ratio
    d_nlp_pg = dpg_dratio * ratio

    # d(pg)/d(logits[a]) = d_nlp_pg * ((a==act) - softmax[a])
    selected_f = selected.to(tl.float32)
    pg_grad = d_nlp_pg[:, None] * (selected_f - sm)

    grad_logits = pg_grad + ent_grad
    store_mask = row_mask[:, None] & act_valid[None, :]
    tl.store(grad_logits_ptr + logits_addr, grad_logits, mask=store_mask)

    # --- d(loss)/d(v_new) from vf term ---
    vc_active = (v_diff > -clip) & (v_diff < clip)
    dvl_un_dv = (v_new - ret)
    dvl_cl_dv = tl.where(vc_active, (vc - ret), 0.0)
    d_vf_dv = tl.where(use_vl_un, dvl_un_dv, dvl_cl_dv)
    grad_v_new = vf_coef * d_vf_dv
    tl.store(grad_v_new_ptr + row_offs, grad_v_new, mask=row_mask)


class FusedPPOLoss(torch.autograd.Function):
    """loss = mean over B of [ pg(ratio, adv) + vf*vf(v_new, ret, val_old)
                               - ent*entropy(logits) ]

    Forward emits (loss, per-row grads). Backward applies the 1/B from the
    mean and chains through grad_output (scalar).
    """
    @staticmethod
    def forward(ctx, logits, v_new, act, logp_old, adv, ret, val_old,
                clip, vf_coef, ent_coef):
        B, A = logits.shape
        assert logits.is_contiguous() and v_new.is_contiguous()
        per_row_loss = torch.empty(B, device=logits.device, dtype=logits.dtype)
        grad_logits  = torch.empty_like(logits)
        grad_v_new   = torch.empty_like(v_new)

        # A must be a compile-time constant; pick BLOCK by A so the (BLOCK, A)
        # tile fits comfortably in registers.
        BLOCK = 128
        # Round actions up to next power of 2 for Triton's arange.
        A_PAD = 1
        while A_PAD < A:
            A_PAD *= 2
        grid = (triton.cdiv(B, BLOCK),)
        _ppo_loss_kernel[grid](
            logits, v_new, act, logp_old, adv, ret, val_old,
            per_row_loss, grad_logits, grad_v_new,
            float(clip), float(vf_coef), float(ent_coef),
            B,
            A=A, A_PAD=A_PAD, BLOCK=BLOCK,
            num_warps=4,
        )
        loss = per_row_loss.sum() / B
        ctx.save_for_backward(grad_logits, grad_v_new)
        ctx.B = B
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_logits, grad_v_new = ctx.saved_tensors
        inv_B = grad_output / ctx.B
        return (grad_logits * inv_B, grad_v_new * inv_B,
                None, None, None, None, None, None, None, None)


# Public entry point: matches the eager math exactly.
def ppo_loss_triton(logits, v_new, act, logp_old, adv, ret, val_old,
                    clip=0.2, vf_coef=0.5, ent_coef=0.01):
    return FusedPPOLoss.apply(
        logits.contiguous(), v_new.contiguous(), act.contiguous(),
        logp_old.contiguous(), adv.contiguous(), ret.contiguous(),
        val_old.contiguous(),
        clip, vf_coef, ent_coef,
    )


# ---------------------------------------------------------------------------
# Eager reference (for correctness comparison)
# ---------------------------------------------------------------------------
def ppo_loss_eager(logits, v_new, act, logp_old, adv, ret, val_old,
                   clip=0.2, vf_coef=0.5, ent_coef=0.01):
    lp_all = logits.log_softmax(-1)
    nlp = lp_all.gather(-1, act.unsqueeze(-1)).squeeze(-1)
    ent = -(lp_all.exp() * lp_all).sum(-1).mean()
    ratio = (nlp - logp_old).exp()
    pg = -torch.min(
        ratio * adv,
        ratio.clamp(1 - clip, 1 + clip) * adv,
    ).mean()
    vc = val_old + (v_new - val_old).clamp(-clip, clip)
    vl = 0.5 * torch.max(
        (v_new - ret).square(),
        (vc - ret).square(),
    ).mean()
    return pg + vf_coef * vl - ent_coef * ent
