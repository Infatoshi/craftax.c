# craftax.c

Pure C / AVX-512 port of [Craftax-Classic](https://github.com/MichaelTMatthews/Craftax), optimized for a modern AMD CPU. Ports the CUDA port at [craftax.cu](https://github.com/Infatoshi/craftax.cu) back to CPU and goes faster than the GPU.

The premise: Craftax's game loop is branchy and latency-sensitive. CPUs are good at that. With enough cores, AVX-512 where it matters, and a little care about the cache hierarchy, a CPU can outrun a datacenter GPU on this workload.

## Results

All numbers on a single AMD Ryzen 9 9950X3D (16c / 32t, dual CCD, one with 3D V-Cache).

| Implementation | Hardware | SPS (NE=32k) |
|---|---|---:|
| JAX (Craftax) | RTX 3090 | ~0.3M |
| CUDA ([craftax.cu](https://github.com/Infatoshi/craftax.cu)) | RTX 3090 | 7.8M |
| CUDA ([craftax.cu](https://github.com/Infatoshi/craftax.cu)) | RTX Pro 6000 Blackwell | 15.1M |
| **This repo (C, AVX-512)** | **Ryzen 9 9950X3D** | **47.8M** |

That's **3.2x the Blackwell RTX Pro 6000** on this workload, on a $600 CPU.

## Build

```bash
make                # bench, bench_hot, bench_tp, bench_tp_pool
```

Requires AVX-512 (Zen 4 / Zen 5 / Intel Ice Lake+). `-march=native` is used.

## Run

```bash
# Mixed NE bench (libgomp, inline reset): float obs vs compact obs
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./bench

# Hot-path bench at NE=32k, compact obs
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./bench_hot 32768 5000

# Best configuration: custom threadpool + pipelined world pool
./bench_tp_pool 32768 5000
```

`bench_tp_pool` runs three configs for comparison:
- `A` — libgomp + inline reset
- `B` — custom spin-barrier thread pool + inline reset
- `C` — thread pool on CCD0 (V-Cache) consumers + dedicated CCD1 producers pre-generating worlds

## How it got fast

Rough chronology of what moved the number, at NE=32k hot-path bench:

```
 5.6M  naive C port, OpenMP parallel over envs
13.3M  mob bitmaps, precomputed sincos, obs compaction, aligned state
25.4M  share Perlin floor/frac/interp setup across the 4 noise layers
28.0M  pipelined world-gen pool (CCD1 producers, CCD0 consumers)
35.0M  AVX-512 Perlin via permutexvar (no gathers)
47.8M  combined thread pool + world pool + SIMD Perlin
```

Profiling-driven, not guessed. Key findings:

- **Perlin worldgen dominated**, not the step loop. Once we vectorized the gradient lookups, the step body became the majority of time.
- **Obs construction was memory-bound.** A compact 145-byte observation beats the 1345-float one-hot encoding by ~3x at large batch sizes. See `OBS_DIM_COMPACT` in `craftax.h`.
- **The dual-CCD layout matters.** Consumers on the V-Cache CCD, producers on the other one. Put them wrong and the pool makes things 2x slower.
- **Scalar code, mostly.** `craftax_step` is entirely scalar — per-env branchy logic doesn't vectorize across envs without an ISPC-style rewrite. The win came from SIMD within worldgen, not within the step loop.

## Files

```
craftax.h           Public API, constants, EnvState struct, PCG32 RNG
craftax.c           Game logic: worldgen, step, mob AI, obs, AVX-512 Perlin
craftax_pool.c      Pipelined world-generation pool (producer threads + ring buffer)
worker_pool.c/.h    Minimal spin-barrier thread pool (optional alternative to OMP)

bench.c             Mixed NE bench: float obs vs compact obs
bench_hot.c         Focused hot-path bench (NE=32k, compact obs)
bench_tp.c          Thread pool vs libgomp A/B
bench_tp_pool.c     Full config comparison (libgomp/threadpool, inline/pooled reset)
sanity.c            Correctness check: achievements, reset rate under random policy
```

## API

```c
#include "craftax.h"

EnvState* states = aligned_alloc(64, sizeof(EnvState) * num_envs);
uint8_t*  obs    = aligned_alloc(64, num_envs * OBS_DIM_COMPACT);
float*    rew    = aligned_alloc(64, num_envs * sizeof(float));
int8_t*   done   = aligned_alloc(64, num_envs);
int32_t*  act    = aligned_alloc(64, num_envs * sizeof(int32_t));

craftax_reset_batch_compact(states, obs, num_envs, /*seed=*/42);
for (int step = 0; step < N; step++) {
    // fill `act` with policy outputs ...
    craftax_step_batch_compact(states, act, obs, rew, done, num_envs, step);
}
```

For the fastest config, use the world pool + thread pool variant:

```c
int consumer_cpus[16] = {0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23};  // CCD0 phys+SMT
setenv("CRAFTAX_POOL_CPUS", "8,9,10,11,12,13,14,15", 1);              // CCD1 producers
ThreadPool* tp = worker_pool_create(16, consumer_cpus);
WorldPool*  wp = craftax_pool_create(/*capacity=*/4096, /*producers=*/8, 42);

craftax_step_batch_compact_pool_tp(tp, wp, states, act, obs, rew, done, num_envs);
```

## Correctness

`sanity.c` runs 256 envs × 500 steps of uniform-random actions and reports achievement counts + reset rate. Same distribution as CUDA reference (wood / table / sapling / plant / wake_up dominate under random play).

Full JAX-reference validation from the CUDA port is not ported here (the PCG32 RNG differs from JAX's Philox so trajectories won't match bit-for-bit), but the game logic is identical and agent behavior under training should match.

## Acknowledgements

Original Craftax environment: Matthews et al., ICML 2024. CUDA port that this repo derives from: [craftax.cu](https://github.com/Infatoshi/craftax.cu).

## Citation

```bibtex
@software{craftax_c,
  title={craftax.c: CPU Craftax-Classic},
  url={https://github.com/Infatoshi/craftax.c},
  year={2026}
}
```
