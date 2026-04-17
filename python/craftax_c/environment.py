"""PufferLib-compatible env for craftax.c.

Treats the batched C env as a single PufferEnv with num_agents = num_envs,
matching the pattern used by PufferLib's GymnaxPufferEnv wrapper. No per-env
crossing of the C/Python boundary -- one ctypes call per training step.

The observation layout matches PufferLib's default Craftax-Classic-Symbolic-v1
encoder exactly (7 rows x 9 cols x 21 channels map + 22 flat scalars, 1345
floats total), so you can reuse the default policy unchanged.
"""
from __future__ import annotations

import functools
import numpy as np
import gymnasium

import pufferlib

from craftax_c.bindings import (
    CraftaxBatch,
    OBS_DIM,
    OBS_DIM_COMPACT,
    NUM_ACTIONS,
)


# --------------------------------------------------------------------------
# GPU-side expansion of the 145-byte compact obs into the 1345-float layout
# expected by PufferLib's default Craftax policy.
# --------------------------------------------------------------------------
def expand_compact_obs(compact: "torch.Tensor") -> "torch.Tensor":
    """Expand compact uint8 obs (B, 145) to float32 (B, 1345).

    Byte layout of compact (see craftax.h):
        [0..63)    block_id per tile      (values 0..16)
        [63..126)  mob bitmask per tile   (bit0=zombie, bit1=cow, bit2=skel, bit3=arrow)
        [126..138) inventory (12 slots, 0..9)
        [138..142) health, food, drink, energy (0..9)
        [142]      player_dir (0..4)
        [143]      is_sleeping (0/1)
        [144]      light_level (quantized 0..255)

    Output layout (must match craftax_build_obs exactly):
        63 tiles * 21 channels (17 block one-hot + 4 mob) = 1323 floats
        + 12 inv / 10 + 4 HFDE / 10 + 4 dir one-hot + 1 light + 1 sleep = 22 floats
        total: 1345
    """
    import torch
    import torch.nn.functional as F
    B = compact.shape[0]
    device = compact.device

    # Map part --------------------------------------------------------------
    block_ids = compact[:, :63].long()                   # (B, 63)
    block_oh  = F.one_hot(block_ids, num_classes=17)     # (B, 63, 17) int64
    block_oh  = block_oh.to(torch.float32)

    mob_mask = compact[:, 63:126]                        # (B, 63)  uint8
    # Unpack four bits into (B, 63, 4) floats.
    bits = torch.stack(
        [(mob_mask >> i) & 1 for i in range(4)], dim=-1
    ).to(torch.float32)                                  # (B, 63, 4)

    map_part = torch.cat([block_oh, bits], dim=-1).reshape(B, 63 * 21)

    # Flat scalars ----------------------------------------------------------
    inv  = compact[:, 126:138].to(torch.float32) * (1.0 / 10.0)   # (B, 12)
    hfde = compact[:, 138:142].to(torch.float32) * (1.0 / 10.0)   # (B, 4)

    pdir = compact[:, 142].long()
    # player_dir is in [1, 4] after reset; one_hot over 5 slots then drop slot 0.
    dir_oh = F.one_hot(pdir.clamp_(0, 4), num_classes=5)[:, 1:5].to(torch.float32)

    light = compact[:, 144:145].to(torch.float32) * (1.0 / 255.0)  # (B, 1)
    sleep = compact[:, 143:144].to(torch.float32)                   # (B, 1)

    flat = torch.cat([inv, hfde, dir_oh, light, sleep], dim=-1)    # (B, 22)

    return torch.cat([map_part, flat], dim=-1)                     # (B, 1345)


class CraftaxCEnv(pufferlib.PufferEnv):
    """PufferLib-compatible env backed by craftax.c.

    Args:
        pinned_obs: if True (default when torch+CUDA available), back self.observations
            with a torch pinned-memory tensor. The C env writes obs directly into pinned
            host memory, and training code can do a single async DMA from `env.obs_tensor`
            to GPU. Reduces the H2D stage of rollouts by ~3-4x at NE=1024, H=64.
        compact_obs: if True, ALSO allocate a pinned uint8 compact-obs buffer (145
            bytes/env) and expose it via env.compact_obs_tensor. The training loop is
            expected to call env.step_compact() each step and expand on GPU with
            craftax_c.environment.expand_compact_obs(). This cuts PCIe bytes-per-step
            by 37x (5380 -> 145) -- the intended use for graph-captured training at
            large NE where obs H2D becomes the wallclock bottleneck.
    """
    def __init__(self, num_envs: int = 1024, buf=None, seed: int = 0,
                 pinned_obs: bool | None = None, compact_obs: bool = False):
        self.num_agents = int(num_envs)
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,), dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Discrete(NUM_ACTIONS)
        super().__init__(buf)  # allocates self.observations/rewards/terminals/truncations/actions

        # Decide whether to use pinned memory. Default: on if torch+CUDA is
        # available, off otherwise.
        if pinned_obs is None:
            try:
                import torch as _t
                pinned_obs = _t.cuda.is_available()
            except ImportError:
                pinned_obs = False

        self._obs_tensor = None
        if pinned_obs:
            import torch
            # Allocate pinned tensor; expose its numpy view as self.observations
            # so the C env writes directly into pinned memory. The torch tensor
            # wrapping the same storage is also pinned -- training code can do
            # obs_gpu.copy_(env.obs_tensor, non_blocking=True) for a single DMA.
            self._obs_tensor = torch.zeros(
                self.num_agents, OBS_DIM,
                dtype=torch.float32, pin_memory=True,
            )
            self.observations = self._obs_tensor.numpy()

        # Compact obs buffer (uint8 pinned). Same C batch state is shared.
        self._compact_tensor = None
        self._compact_np = None
        if compact_obs:
            import torch
            self._compact_tensor = torch.zeros(
                self.num_agents, OBS_DIM_COMPACT,
                dtype=torch.uint8, pin_memory=(pinned_obs is not False),
            )
            self._compact_np = self._compact_tensor.numpy()

        self._batch = CraftaxBatch(self.num_agents, seed=seed)
        self._seed = seed
        # actions are int32 for our C API
        if self.actions.dtype != np.int32:
            self._actions_i32 = np.ascontiguousarray(self.actions, dtype=np.int32)
        else:
            self._actions_i32 = self.actions

    # ------------------------------------------------------------------
    # Compact path: writes 145 bytes/env into env.compact_obs_tensor.
    # ------------------------------------------------------------------
    def reset_compact(self, seed: int | None = None):
        if self._compact_np is None:
            raise RuntimeError("CraftaxCEnv was created with compact_obs=False")
        s = self._seed if seed is None else int(seed)
        self._batch.reset_compact(self._compact_np, seed=s)
        return self._compact_tensor

    def step_compact(self, actions):
        if self._compact_np is None:
            raise RuntimeError("CraftaxCEnv was created with compact_obs=False")
        if actions.dtype != np.int32:
            np.copyto(self._actions_i32, actions, casting="unsafe")
        else:
            self._actions_i32 = np.ascontiguousarray(actions, dtype=np.int32)
        self._batch.step_compact(
            self._actions_i32, self._compact_np, self.rewards, self.terminals,
        )
        self.truncations[:] = 0
        return self._compact_tensor, self.rewards, self.terminals, self.truncations, []

    @property
    def compact_obs_tensor(self):
        """Pinned uint8 tensor (num_envs, 145) sharing storage with compact obs buffer."""
        return self._compact_tensor

    @property
    def obs_tensor(self):
        """Pinned torch tensor sharing memory with self.observations (if pinned_obs=True).

        Training code should prefer this over torch.from_numpy(env.observations) because
        it avoids the per-step tensor allocation and enables async DMA.
        """
        return self._obs_tensor

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        s = self._seed if seed is None else int(seed)
        self._batch.reset(self.observations, seed=s)
        # truncations stays zero; we never truncate (C env auto-resets on done)
        self.truncations[:] = 0
        return self.observations, []

    def step(self, actions):
        # Actions may arrive as int64 tensors; copy to our int32 buffer.
        if actions.dtype != np.int32:
            np.copyto(self._actions_i32, actions, casting="unsafe")
        else:
            self._actions_i32 = np.ascontiguousarray(actions, dtype=np.int32)
        self._batch.step(
            self._actions_i32, self.observations, self.rewards, self.terminals
        )
        # PufferLib expects bool-ish terminals; our C API writes int8 {0,1}
        # into the same byte buffer, which is compatible with np.bool_ dtype.
        self.truncations[:] = 0
        return (
            self.observations, self.rewards,
            self.terminals, self.truncations, [],
        )

    def close(self):
        pass


# --------------------------------------------------------------------------
# PufferLib-style creator so train scripts can do:
#   env_creator('Craftax-C-Symbolic-v1')(num_envs=1024)
# --------------------------------------------------------------------------
def env_creator(name: str = "Craftax-C-Symbolic-v1"):
    return functools.partial(make, name)


def make(name: str, num_envs: int = 1024, buf=None, seed: int = 0):
    return CraftaxCEnv(num_envs=num_envs, buf=buf, seed=seed)
