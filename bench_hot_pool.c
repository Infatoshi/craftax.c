// Focused bench comparing inline-reset vs pipelined-pool reset at NE=32k.
#include "craftax.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc, char** argv) {
    int NE    = argc>1 ? atoi(argv[1]) : 32768;
    int ITERS = argc>2 ? atoi(argv[2]) : 2000;
    int POOL  = argc>3 ? atoi(argv[3]) : 2048;
    int NPROD = argc>4 ? atoi(argv[4]) : 4;

    EnvState* st = aligned_alloc(64, sizeof(EnvState)*(size_t)NE);
    uint8_t*  ob = aligned_alloc(64, (size_t)NE*OBS_DIM_COMPACT);
    float*    rw = aligned_alloc(64, sizeof(float)*(size_t)NE);
    int8_t*   dn = aligned_alloc(64, NE);
    int32_t*  ac = aligned_alloc(64, sizeof(int32_t)*(size_t)NE);
    memset(st, 0, sizeof(EnvState)*(size_t)NE);

    pcg32_t r; pcg32_seed(&r,42,1);
    for (int i=0;i<NE;i++) ac[i] = pcg32_next(&r)%NUM_ACTIONS;

    // --- baseline: inline reset ---
    craftax_reset_batch_compact(st, ob, NE, 42);
    for (int w=0;w<20;w++) craftax_step_batch_compact(st,ac,ob,rw,dn,NE,42+w);
    double t0 = now_s();
    for (int k=0;k<ITERS;k++) craftax_step_batch_compact(st,ac,ob,rw,dn,NE,42+k+100);
    double dt_base = now_s()-t0;
    double sps_base = (double)NE*ITERS/dt_base;

    // --- pool variant ---
    printf("Creating pool (cap=%d, producers=%d)...\n", POOL, NPROD);
    WorldPool* p = craftax_pool_create(POOL, NPROD, 42);
    uint64_t pre_p, pre_c; int ready0;
    craftax_pool_stats(p, &pre_p, &pre_c, NULL, &ready0);
    printf("  pool prewarm: produced=%lu ready=%d\n", pre_p, ready0);

    memset(st, 0, sizeof(EnvState)*(size_t)NE);
    craftax_reset_batch_compact(st, ob, NE, 42);
    for (int w=0;w<20;w++) craftax_step_batch_compact_pool(p,st,ac,ob,rw,dn,NE);

    uint64_t p0_prod, p0_cons, p0_fb; int ready1;
    craftax_pool_stats(p, &p0_prod, &p0_cons, &p0_fb, &ready1);

    t0 = now_s();
    for (int k=0;k<ITERS;k++) craftax_step_batch_compact_pool(p,st,ac,ob,rw,dn,NE);
    double dt_pool = now_s()-t0;
    double sps_pool = (double)NE*ITERS/dt_pool;

    uint64_t p1_prod, p1_cons, p1_fb; int ready2;
    craftax_pool_stats(p, &p1_prod, &p1_cons, &p1_fb, &ready2);

    printf("\n--- NE=%d iters=%d ---\n", NE, ITERS);
    printf("baseline (inline reset):  %.3fs  SPS=%.0f\n", dt_base, sps_base);
    printf("pooled reset:             %.3fs  SPS=%.0f   (%.2fx)\n", dt_pool, sps_pool, sps_pool/sps_base);
    printf("pool consumed: %lu   produced-during-run: %lu   fallbacks: %lu   ready_end=%d\n",
           p1_cons - p0_cons, p1_prod - p0_prod, p1_fb - p0_fb, ready2);

    craftax_pool_destroy(p);
    return 0;
}
