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

- **Action dtype** — PufferEnv allocates actions as int64; we cast to int32
  via `np.copyto(..., casting='unsafe')` each step. Cheap but non-zero.

## Blackwell GPU training

PufferLib's shipped `_C.so` ships only sm_86 cubins, so Blackwell
(RTX 50 / RTX PRO 6000, sm_120) fails with "no kernel image is available
for execution on the device". Fix is a rebuild on a Blackwell host -- torch
auto-detects the right arch and emits sm_120 cubins. Our fork at
[Infatoshi/PufferLib @ blackwell-sm120-support](https://github.com/Infatoshi/PufferLib/tree/blackwell-sm120-support)
also adds a `fused_adam` config option and fixes a state_dict recursion
bug in the `torch.compile` path. To use:

```bash
git clone -b blackwell-sm120-support https://github.com/Infatoshi/PufferLib
cd PufferLib
TORCH_CUDA_ARCH_LIST="8.6;9.0;12.0" python -m pip install --no-build-isolation -e .
cuobjdump --list-elf pufferlib/_C.*.so    # should list sm_120 alongside sm_86, sm_90
```

## Policy profile

The default Craftax-Classic policy is a tiny 45k-param model:

```
Conv2d(21, 32, k=3, s=2)  ->  (B, 32, 3, 4)   6,080 params
Conv2d(32, 32, k=3, s=1)  ->  (B, 32, 1, 2)   9,248 params
Flatten                   ->  (B, 64)
Linear(22, 128)           ->  (B, 128)        2,944 params  (flat features)
Linear(192, 128)          ->  (B, 128)       24,704 params  (projection)
Linear(128, 17)   actor                       2,193 params
Linear(128, 1)    critic                        129 params
```

At the training minibatch size (4096) on a Blackwell RTX PRO 6000, one
forward+backward+Adam step takes ~743 us plain / ~576 us with fused Adam.
Most of that is kernel launch overhead on a model this small; compute is
only 4 Gflop per minibatch (~40 us of pure FMA on this GPU).

Empirically `torch.compile(mode="reduce-overhead")` gives 2x on fixed-shape
microbenchmarks but regresses inside the PufferLib PPO loop (varying shapes
trigger recompiles, and max-autotune-no-cudagraphs fusion wins less than
the recompile cost on a 45k-param model). Enabling it via
`args["train"]["compile"] = True` is opt-in.
