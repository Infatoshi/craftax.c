// Best-of-both: custom thread pool + world pool pipeline.
#include "craftax.h"
#include "worker_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc, char** argv) {
    int NE    = argc>1 ? atoi(argv[1]) : 32768;
    int ITERS = argc>2 ? atoi(argv[2]) : 5000;

    EnvState* st = aligned_alloc(64, sizeof(EnvState)*(size_t)NE);
    uint8_t*  ob = aligned_alloc(64, (size_t)NE*OBS_DIM_COMPACT);
    float*    rw = aligned_alloc(64, sizeof(float)*(size_t)NE);
    int8_t*   dn = aligned_alloc(64, NE);
    int32_t*  ac = aligned_alloc(64, sizeof(int32_t)*(size_t)NE);
    memset(st, 0, sizeof(EnvState)*(size_t)NE);

    pcg32_t r; pcg32_seed(&r,42,1);
    for (int i=0;i<NE;i++) ac[i] = pcg32_next(&r)%NUM_ACTIONS;

    // --- A: libgomp + inline reset (old baseline) ---
    craftax_reset_batch_compact(st, ob, NE, 42);
    for (int w=0;w<20;w++) craftax_step_batch_compact(st,ac,ob,rw,dn,NE,42+w);
    double t0 = now_s();
    for (int k=0;k<ITERS;k++) craftax_step_batch_compact(st,ac,ob,rw,dn,NE,42+k+100);
    double dt_A = now_s()-t0;

    // --- B: custom threadpool + inline reset ---
    int cpus16[16]; for (int i=0;i<16;i++) cpus16[i] = i;
    ThreadPool* tp16 = worker_pool_create(16, cpus16);
    memset(st, 0, sizeof(EnvState)*(size_t)NE);
    craftax_reset_batch_compact(st, ob, NE, 42);
    for (int w=0;w<20;w++) craftax_step_batch_compact_tp(tp16,st,ac,ob,rw,dn,NE,42+w);
    t0 = now_s();
    for (int k=0;k<ITERS;k++) craftax_step_batch_compact_tp(tp16,st,ac,ob,rw,dn,NE,42+k+100);
    double dt_B = now_s()-t0;
    worker_pool_destroy(tp16);

    // --- C: custom threadpool (CCD0 phys+SMT) + world pool (CCD1 producers) ---
    // 16 consumer threads on cores {0..7, 16..23}, 8 producers on {8..15}.
    setenv("CRAFTAX_POOL_CPUS", "8,9,10,11,12,13,14,15", 1);
    int ccpus[16] = {0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23};
    ThreadPool* tp_ccd0 = worker_pool_create(16, ccpus);
    WorldPool* wp = craftax_pool_create(4096, 8, 42);

    memset(st, 0, sizeof(EnvState)*(size_t)NE);
    craftax_reset_batch_compact(st, ob, NE, 42);
    for (int w=0;w<20;w++) craftax_step_batch_compact_pool_tp(tp_ccd0, wp, st,ac,ob,rw,dn,NE);
    uint64_t p0,c0,f0; int r0;
    craftax_pool_stats(wp, &p0, &c0, &f0, &r0);
    t0 = now_s();
    for (int k=0;k<ITERS;k++) craftax_step_batch_compact_pool_tp(tp_ccd0, wp, st,ac,ob,rw,dn,NE);
    double dt_C = now_s()-t0;
    uint64_t p1,c1,f1; int r1;
    craftax_pool_stats(wp, &p1, &c1, &f1, &r1);
    craftax_pool_destroy(wp);
    worker_pool_destroy(tp_ccd0);

    printf("NE=%d iters=%d\n", NE, ITERS);
    printf("  A  libgomp + inline reset (16 cores):        %.3fs  SPS=%10.0f  (1.00x)\n",
           dt_A, (double)NE*ITERS/dt_A);
    printf("  B  threadpool + inline reset (16 cores):     %.3fs  SPS=%10.0f  (%.2fx)\n",
           dt_B, (double)NE*ITERS/dt_B, dt_A/dt_B);
    printf("  C  threadpool + world pool (16C+8P):         %.3fs  SPS=%10.0f  (%.2fx)\n",
           dt_C, (double)NE*ITERS/dt_C, dt_A/dt_C);
    printf("     pool: consumed=%lu produced=%lu fallbacks=%lu ready=%d\n",
           c1-c0, p1-p0, f1-f0, r1);
    return 0;
}
