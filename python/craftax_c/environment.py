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
    NUM_ACTIONS,
)


class CraftaxCEnv(pufferlib.PufferEnv):
    """PufferLib-compatible env backed by craftax.c.

    Args:
        pinned_obs: if True (default when torch+CUDA available), back self.observations
            with a torch pinned-memory tensor. The C env writes obs directly into pinned
            host memory, and training code can do a single async DMA from `env.obs_tensor`
            to GPU. Reduces the H2D stage of rollouts by ~3-4x at NE=1024, H=64.
    """
    def __init__(self, num_envs: int = 1024, buf=None, seed: int = 0,
                 pinned_obs: bool | None = None):
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

        self._batch = CraftaxBatch(self.num_agents, seed=seed)
        self._seed = seed
        # actions are int32 for our C API
        if self.actions.dtype != np.int32:
            self._actions_i32 = np.ascontiguousarray(self.actions, dtype=np.int32)
        else:
            self._actions_i32 = self.actions

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
