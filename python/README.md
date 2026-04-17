# craftax_c — PufferLib integration

Python bindings + a PufferLib `PufferEnv` wrapper for the batched C env.
Uses the default `pufferlib.environments.craftax.torch.Policy` (same 1345-float
observation layout as the JAX env), so no encoder changes are needed to train.

## Quickstart

```bash
# 1. Build the shared library (in the repo root)
make libcraftax.so

# 2. Point to it, train. Uses PufferLib's Craftax-Classic PPO hyperparams.
export CRAFTAX_LIB=$PWD/libcraftax.so
export PYTHONPATH=python
export OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores

# CPU device (works everywhere; env cost is ~0% of training time)
CRAFTAX_DEVICE=cpu /path/to/pufferlib/venv/bin/python python/train_craftax_c.py

# GPU device (requires a PufferLib build with a CUDA kernel for your arch;
# sm_120 / Blackwell is currently unsupported upstream as of this writing)
CRAFTAX_DEVICE=cuda /path/to/pufferlib/venv/bin/python python/train_craftax_c.py
```

Env overrides: `CRAFTAX_NUM_ENVS`, `CRAFTAX_TIMESTEPS`, `CRAFTAX_BATCH`,
`CRAFTAX_MINIBATCH`, `CRAFTAX_DEVICE`.

## How the integration works

`pufferlib.environments.craftax.environment.py` wraps the JAX Craftax env into
a `GymnaxPufferEnv` with `num_agents = num_envs`. One Python call per step
dispatches a `jax.vmap`'d step over the whole batch — no per-env crossing of
the Python boundary.

We do exactly the same pattern. `CraftaxCEnv` inherits from `pufferlib.PufferEnv`
with `num_agents = num_envs`, and `step()` makes one ctypes call into
`craftax_step_batch` which fans out across 16 OpenMP threads internally.

```
step()  ──>  ctypes ──>  craftax_step_batch ──>  #pragma omp parallel for
                                                 over num_envs envs
```

The observation buffer, reward buffer, and terminal buffer that PufferLib
pre-allocates (via `buf=` in `pufferlib.PufferEnv.__init__`) are passed to the
C code as raw pointers, so the C env writes directly into them with zero copy.

## Observation compatibility

craftax.c's `craftax_build_obs` produces the exact same 1345-float layout as
the JAX env and the CUDA port:

```
[ 7 rows x 9 cols x 21 channels ]  =  1323 floats
    per tile: 17 block-type one-hot + 4 mob presence (zombie/cow/skel/arrow)
[ 12 inventory  +  4 intrinsics  +  4 direction one-hot  +  1 light  +  1 sleep ]
                                                    =  22 floats
```

`pufferlib.environments.craftax.torch.Policy` splits the flat vector into the
map CNN input and the flat MLP input at exactly this boundary.

## Files

```
craftax_c/
  __init__.py      Re-exports the public API
  bindings.py      ctypes loader + CraftaxBatch class (low-level batched API)
  environment.py   PufferEnv-compatible CraftaxCEnv
train_craftax_c.py PPO training script using PufferLib's default policy
```

## API

```python
# Low level -- raw batched C calls, bring your own buffers
from craftax_c import CraftaxBatch, OBS_DIM, NUM_ACTIONS
b = CraftaxBatch(num_envs=4096, seed=0)
b.reset(obs_array)
b.step(actions_i32, obs_array, rewards_array, terminals_array)

# PufferLib env
from craftax_c import CraftaxCEnv, env_creator
env = CraftaxCEnv(num_envs=4096)
obs, info = env.reset()
obs, rew, term, trunc, info = env.step(actions)

# For pufferlib.vector.make -- matches the JAX env's creator signature
make_env = env_creator("Craftax-C-Symbolic-v1")
```

## Known issues

- **Blackwell (sm_120) GPU training** — PufferLib's `compute_puff_advantage`
  custom CUDA kernel currently fails with "no kernel image is available for
  execution on the device". Track upstream; CPU device works fine.
- **Action dtype** — PufferEnv allocates actions as int64; we cast to int32
  via `np.copyto(..., casting='unsafe')` each step. Cheap but non-zero.
