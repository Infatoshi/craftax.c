// Focused bench: NE=32768, compact obs only. One long run so perf captures the
// steady-state hot path without the sync barriers of tiny-batch runs.
#include "craftax.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc, char** argv) {
    int NE = argc>1 ? atoi(argv[1]) : 32768;
    int ITERS = argc>2 ? atoi(argv[2]) : 2000;

    EnvState* st = aligned_alloc(64, sizeof(EnvState)*(size_t)NE);
    uint8_t*  ob = aligned_alloc(64, (size_t)NE*OBS_DIM_COMPACT);
    float*    rw = aligned_alloc(64, sizeof(float)*(size_t)NE);
    int8_t*   dn = aligned_alloc(64, NE);
    int32_t*  ac = aligned_alloc(64, sizeof(int32_t)*(size_t)NE);
    memset(st, 0, sizeof(EnvState)*(size_t)NE);

    pcg32_t r; pcg32_seed(&r,42,1);
    for (int i=0;i<NE;i++) ac[i] = pcg32_next(&r)%NUM_ACTIONS;
    craftax_reset_batch_compact(st, ob, NE, 42);

    for (int w=0;w<20;w++) craftax_step_batch_compact(st,ac,ob,rw,dn,NE,42+w);

    double t0 = now_s();
    for (int k=0;k<ITERS;k++) craftax_step_batch_compact(st,ac,ob,rw,dn,NE,42+k+100);
    double dt = now_s()-t0;
    printf("NE=%d iters=%d time=%.3fs SPS=%.0f\n", NE, ITERS, dt, (double)NE*ITERS/dt);
    return 0;
}
