// Env-only SPS benchmark: float obs (1345) vs compact obs (145).
#include "craftax.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void bench_one(int num_envs, int iters) {
    EnvState* states     = (EnvState*)aligned_alloc(64, sizeof(EnvState) * (size_t)num_envs);
    float*    obs_f      = (float*)   aligned_alloc(64, sizeof(float)    * (size_t)num_envs * OBS_DIM);
    uint8_t*  obs_c      = (uint8_t*) aligned_alloc(64, (size_t)num_envs * OBS_DIM_COMPACT);
    float*    rewards    = (float*)   aligned_alloc(64, sizeof(float)    * (size_t)num_envs);
    int8_t*   dones      = (int8_t*)  aligned_alloc(64, sizeof(int8_t)   * (size_t)num_envs);
    int32_t*  actions    = (int32_t*) aligned_alloc(64, sizeof(int32_t)  * (size_t)num_envs);
    if (!states || !obs_f || !obs_c || !rewards || !dones || !actions) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    memset(states, 0, sizeof(EnvState) * (size_t)num_envs);

    pcg32_t ar; pcg32_seed(&ar, 42, 1);
    for (int i = 0; i < num_envs; i++) actions[i] = (int32_t)(pcg32_next(&ar) % NUM_ACTIONS);

    // --- float obs path ---
    craftax_reset_batch(states, obs_f, num_envs, 42);
    for (int w = 0; w < 5; w++)
        craftax_step_batch(states, actions, obs_f, rewards, dones, num_envs, 42 + (uint64_t)w);
    double t0 = now_s();
    for (int k = 0; k < iters; k++)
        craftax_step_batch(states, actions, obs_f, rewards, dones, num_envs, 42 + (uint64_t)(k + 100));
    double dt_f = now_s() - t0;
    double sps_f = (double)num_envs * (double)iters / dt_f;

    // --- compact obs path (reset states fresh for fair comparison) ---
    memset(states, 0, sizeof(EnvState) * (size_t)num_envs);
    craftax_reset_batch_compact(states, obs_c, num_envs, 42);
    for (int w = 0; w < 5; w++)
        craftax_step_batch_compact(states, actions, obs_c, rewards, dones, num_envs, 42 + (uint64_t)w);
    t0 = now_s();
    for (int k = 0; k < iters; k++)
        craftax_step_batch_compact(states, actions, obs_c, rewards, dones, num_envs, 42 + (uint64_t)(k + 100));
    double dt_c = now_s() - t0;
    double sps_c = (double)num_envs * (double)iters / dt_c;

    printf("  NE=%6d  float=%12.0f SPS   compact=%12.0f SPS   speedup=%.2fx\n",
           num_envs, sps_f, sps_c, sps_c / sps_f);
}

int main(int argc, char** argv) {
    int iters_small = 2000, iters_big = 400;
    if (argc >= 2) iters_small = atoi(argv[1]);
    if (argc >= 3) iters_big = atoi(argv[2]);

    printf("--- ENV-ONLY SPS (C, OpenMP) ---\n");
    printf("  state size: %zu bytes\n", sizeof(EnvState));
    printf("  obs: float=%d floats (%d B)  compact=%d B\n",
           OBS_DIM, (int)(OBS_DIM * sizeof(float)), OBS_DIM_COMPACT);
    bench_one(1024,  iters_small);
    bench_one(4096,  iters_small);
    bench_one(8192,  iters_big);
    bench_one(32768, iters_big);
    return 0;
}
