"""ctypes wrapper around libcraftax.so.

Exposes the batched API directly -- one Python call kicks off the OpenMP
parallel step across num_envs environments. No per-env crossing of the
C/Python boundary.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Constants (mirror craftax.h)
# ---------------------------------------------------------------------------
STATE_SIZE       = 4864
OBS_DIM          = 1345
OBS_DIM_COMPACT  = 145
NUM_ACTIONS      = 17


def _find_lib() -> str:
    """Locate libcraftax.so. Search order: CRAFTAX_LIB env var, repo root, cwd."""
    env = os.environ.get("CRAFTAX_LIB")
    if env:
        return env
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "libcraftax.so",  # repo root (python/craftax_c/ -> root)
        here.parent / "libcraftax.so",
        Path.cwd() / "libcraftax.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "libcraftax.so not found. Build it with `make libcraftax.so` in the "
        "craftax.c repo, or set CRAFTAX_LIB to its absolute path."
    )


_lib = ctypes.CDLL(_find_lib(), mode=ctypes.RTLD_GLOBAL)

# void craftax_reset_batch(EnvState* states, float* obs, int num_envs, uint64_t seed);
_lib.craftax_reset_batch.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_uint64,
]
_lib.craftax_reset_batch.restype = None

# void craftax_step_batch(EnvState*, const int32_t* actions, float* obs,
#                         float* rewards, int8_t* dones,
#                         int num_envs, uint64_t reset_seed);
_lib.craftax_step_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_int, ctypes.c_uint64,
]
_lib.craftax_step_batch.restype = None

# void craftax_reset_batch_compact(EnvState*, uint8_t* obs, int, uint64_t);
_lib.craftax_reset_batch_compact.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int, ctypes.c_uint64,
]
_lib.craftax_reset_batch_compact.restype = None

# void craftax_step_batch_compact(EnvState*, const int32_t*, uint8_t* obs,
#                                 float* rewards, int8_t* dones,
#                                 int num_envs, uint64_t reset_seed);
_lib.craftax_step_batch_compact.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int8),
    ctypes.c_int, ctypes.c_uint64,
]
_lib.craftax_step_batch_compact.restype = None


class CraftaxBatch:
    """Owns the C-side state buffer; references external numpy arrays for I/O.

    Obs/rewards/terminals are NOT owned -- caller passes them in (typically
    PufferLib's pre-allocated buffers) so there's no copy on the hot path.
    """

    def __init__(self, num_envs: int, seed: int = 0):
        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        # 64-byte aligned state buffer (one EnvState per env).
        n_bytes = self.num_envs * STATE_SIZE
        self._states = np.zeros(n_bytes, dtype=np.uint8)
        if self._states.ctypes.data % 64 != 0:
            # numpy usually aligns; if not, reallocate via posix_memalign-ish dance.
            pad = np.zeros(n_bytes + 64, dtype=np.uint8)
            off = (-pad.ctypes.data) % 64
            self._states = pad[off:off + n_bytes]
        assert self._states.ctypes.data % 64 == 0
        self._step_count = 0

    # ------------------------------------------------------------------
    def reset(self, obs: np.ndarray, seed: int | None = None) -> None:
        """Fill obs[num_envs, OBS_DIM] with initial observations."""
        self._check_obs(obs)
        s = self.seed if seed is None else int(seed)
        _lib.craftax_reset_batch(
            self._states.ctypes.data,
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.num_envs, ctypes.c_uint64(s),
        )
        self._step_count = 0

    def step(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
    ) -> None:
        """Step all envs. actions int32 [num_envs]; writes obs/rewards/terminals."""
        if actions.dtype != np.int32:
            raise TypeError(f"actions must be int32, got {actions.dtype}")
        if actions.shape != (self.num_envs,):
            raise ValueError(
                f"actions shape {actions.shape}, expected ({self.num_envs},)"
            )
        self._check_obs(obs)
        if rewards.dtype != np.float32 or rewards.shape != (self.num_envs,):
            raise ValueError("rewards must be float32[num_envs]")
        # Accept bool or int8 terminals -- same underlying byte layout.
        if terminals.dtype not in (np.int8, np.bool_, np.uint8):
            raise TypeError(f"terminals must be int8/bool/uint8, got {terminals.dtype}")
        if terminals.shape != (self.num_envs,):
            raise ValueError("terminals must be 1-D length num_envs")

        self._step_count += 1
        reset_seed = self.seed + self._step_count * 1_000_003
        _lib.craftax_step_batch(
            self._states.ctypes.data,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            terminals.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self.num_envs, ctypes.c_uint64(reset_seed),
        )

    # ------------------------------------------------------------------
    # Compact API -- 145 bytes/env instead of 5380. Expand on GPU.
    def reset_compact(self, obs_u8: np.ndarray, seed: int | None = None) -> None:
        self._check_compact(obs_u8)
        s = self.seed if seed is None else int(seed)
        _lib.craftax_reset_batch_compact(
            self._states.ctypes.data,
            obs_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.num_envs, ctypes.c_uint64(s),
        )
        self._step_count = 0

    def step_compact(self, actions: np.ndarray, obs_u8: np.ndarray,
                     rewards: np.ndarray, terminals: np.ndarray) -> None:
        if actions.dtype != np.int32 or actions.shape != (self.num_envs,):
            raise ValueError("actions must be int32[num_envs]")
        self._check_compact(obs_u8)
        if rewards.dtype != np.float32 or rewards.shape != (self.num_envs,):
            raise ValueError("rewards must be float32[num_envs]")
        if terminals.dtype not in (np.int8, np.bool_, np.uint8):
            raise TypeError("terminals must be int8/bool/uint8")
        self._step_count += 1
        reset_seed = self.seed + self._step_count * 1_000_003
        _lib.craftax_step_batch_compact(
            self._states.ctypes.data,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            obs_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            terminals.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self.num_envs, ctypes.c_uint64(reset_seed),
        )

    def _check_obs(self, obs: np.ndarray) -> None:
        if obs.dtype != np.float32:
            raise TypeError(f"obs must be float32, got {obs.dtype}")
        if obs.shape != (self.num_envs, OBS_DIM):
            raise ValueError(
                f"obs shape {obs.shape}, expected ({self.num_envs}, {OBS_DIM})"
            )
        if not obs.flags["C_CONTIGUOUS"]:
            raise ValueError("obs must be C-contiguous")

    def _check_compact(self, obs: np.ndarray) -> None:
        if obs.dtype != np.uint8:
            raise TypeError(f"compact obs must be uint8, got {obs.dtype}")
        if obs.shape != (self.num_envs, OBS_DIM_COMPACT):
            raise ValueError(
                f"obs shape {obs.shape}, expected ({self.num_envs}, {OBS_DIM_COMPACT})"
            )
        if not obs.flags["C_CONTIGUOUS"]:
            raise ValueError("obs must be C-contiguous")
