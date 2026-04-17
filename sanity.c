// Quick sanity check: run 500 steps, summarize rewards/dones/achievements.
#include "craftax.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const int NE = 256, STEPS = 500;
    EnvState* st = aligned_alloc(64, sizeof(EnvState) * NE);
    float* obs = aligned_alloc(64, sizeof(float) * NE * OBS_DIM);
    float* rew = aligned_alloc(64, sizeof(float) * NE);
    int8_t* done = aligned_alloc(64, NE);
    int32_t* act = aligned_alloc(64, sizeof(int32_t) * NE);
    memset(st, 0, sizeof(EnvState) * NE);

    craftax_reset_batch(st, obs, NE, 42);
    pcg32_t r; pcg32_seed(&r, 7, 1);

    double sum_r = 0; int tot_done = 0; int ach_hits[NUM_ACHIEVEMENTS] = {0};
    for (int k = 0; k < STEPS; k++) {
        for (int i = 0; i < NE; i++) act[i] = pcg32_next(&r) % NUM_ACTIONS;
        craftax_step_batch(st, act, obs, rew, done, NE, 42 + k);
        for (int i = 0; i < NE; i++) { sum_r += rew[i]; tot_done += done[i]; }
    }
    for (int i = 0; i < NE; i++)
        for (int a = 0; a < NUM_ACHIEVEMENTS; a++)
            ach_hits[a] += st[i].achievements[a];

    printf("NE=%d STEPS=%d\n", NE, STEPS);
    printf("mean reward/step/env = %.4f\n", sum_r / (double)(NE * STEPS));
    printf("total dones = %d (reset rate = %.2f%%)\n", tot_done,
           100.0 * tot_done / (double)(NE * STEPS));
    printf("achievements unlocked across envs:\n");
    const char* names[] = {"wood","table","cow","sapling","drink","wpick","wsword","plant",
        "zombie","stone","pstone","eplant","skel","spick","ssword","wake","furnace","coal",
        "iron","diamond","ipick","isword"};
    for (int a = 0; a < NUM_ACHIEVEMENTS; a++)
        if (ach_hits[a]) printf("  %-10s  %d envs\n", names[a], ach_hits[a]);
    return 0;
}
