// Pipelined world-generation pool.
// Producer pthreads (pinned to SMT siblings) continuously pre-generate worlds
// and publish them to a ring buffer. Consumers pop from the ring on reset.
//
// Slot state machine:  EMPTY (0) -> FILLING (1) -> READY (2) -> DRAINING (3) -> EMPTY
// Producers CAS EMPTY->FILLING, fill the slot, store READY.
// Consumers CAS READY->DRAINING, copy out, store EMPTY.
//
// produce_hint/consume_hint are non-authoritative atomics used as rolling
// starting points so producers and consumers don't collide on the same slots.
#define _GNU_SOURCE
#include "craftax.h"
#include "worker_pool.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern void _craftax_generate_world(EnvState* s, uint64_t seed, uint64_t env_id);

enum { SLOT_EMPTY=0, SLOT_FILLING=1, SLOT_READY=2, SLOT_DRAINING=3 };

struct WorldPool {
    int capacity;
    int num_producers;
    uint64_t master_seed;

    EnvState* worlds;               // [capacity]
    _Atomic uint32_t* state;        // [capacity]

    _Atomic uint64_t produce_hint;
    _Atomic uint64_t consume_hint;
    _Atomic uint64_t producer_env_id;   // monotonically-increasing env_id for seeding

    _Atomic uint64_t total_produced;
    _Atomic uint64_t total_consumed;
    _Atomic uint64_t total_fallbacks;

    _Atomic int running;
    pthread_t* threads;
};

static void pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

typedef struct { WorldPool* pool; int producer_idx; int cpu; } ProducerArg;

static void* producer_main(void* arg_) {
    ProducerArg* a = (ProducerArg*)arg_;
    WorldPool* p = a->pool;
    if (a->cpu >= 0) pin_to_cpu(a->cpu);

    while (atomic_load_explicit(&p->running, memory_order_relaxed)) {
        // Find an EMPTY slot starting from the rolling hint.
        uint64_t h = atomic_fetch_add_explicit(&p->produce_hint, 1, memory_order_relaxed);
        uint32_t idx = (uint32_t)(h % (uint64_t)p->capacity);

        uint32_t expect = SLOT_EMPTY;
        int claimed = atomic_compare_exchange_strong_explicit(
            &p->state[idx], &expect, SLOT_FILLING,
            memory_order_acquire, memory_order_relaxed);

        if (!claimed) {
            // Either filling/ready/draining -- try again with next hint.
            // If the whole ring is full (READY everywhere), yield briefly.
            if (expect == SLOT_READY) {
                // Pool full; back off a bit.
                for (int k = 0; k < 64; k++) __builtin_ia32_pause();
            }
            continue;
        }

        // Fill the slot.
        uint64_t eid = atomic_fetch_add_explicit(&p->producer_env_id, 1, memory_order_relaxed);
        _craftax_generate_world(&p->worlds[idx], p->master_seed, eid);

        atomic_store_explicit(&p->state[idx], SLOT_READY, memory_order_release);
        atomic_fetch_add_explicit(&p->total_produced, 1, memory_order_relaxed);
    }
    free(a);
    return NULL;
}

// Try to pop one ready world into dst. Returns true on success.
static inline bool pool_try_pop(WorldPool* p, EnvState* dst) {
    // Try up to capacity slots before giving up.
    for (int tries = 0; tries < p->capacity; tries++) {
        uint64_t h = atomic_fetch_add_explicit(&p->consume_hint, 1, memory_order_relaxed);
        uint32_t idx = (uint32_t)(h % (uint64_t)p->capacity);

        uint32_t expect = SLOT_READY;
        if (atomic_compare_exchange_strong_explicit(
                &p->state[idx], &expect, SLOT_DRAINING,
                memory_order_acquire, memory_order_relaxed)) {
            memcpy(dst, &p->worlds[idx], sizeof(EnvState));
            atomic_store_explicit(&p->state[idx], SLOT_EMPTY, memory_order_release);
            atomic_fetch_add_explicit(&p->total_consumed, 1, memory_order_relaxed);
            return true;
        }
    }
    return false;
}

WorldPool* craftax_pool_create(int capacity, int num_producers, uint64_t master_seed) {
    WorldPool* p = (WorldPool*)calloc(1, sizeof(*p));
    p->capacity = capacity;
    p->num_producers = num_producers;
    p->master_seed = master_seed;
    p->worlds = (EnvState*)aligned_alloc(64, sizeof(EnvState) * (size_t)capacity);
    p->state  = (_Atomic uint32_t*)aligned_alloc(64, sizeof(uint32_t) * (size_t)capacity);
    memset(p->worlds, 0, sizeof(EnvState) * (size_t)capacity);
    for (int i = 0; i < capacity; i++)
        atomic_store_explicit(&p->state[i], SLOT_EMPTY, memory_order_relaxed);
    atomic_store(&p->running, 1);

    // SMT siblings of physical cores live at +16 on this CPU topology:
    //   CCD0: cores 0..7  (SMT 16..23)
    //   CCD1: cores 8..15 (SMT 24..31)
    // Producers go on CCD1's SMT siblings (24..31), so consumers on the
    // physical cores can run steady-state step logic unimpeded.
    // Producer CPU list from env var CRAFTAX_POOL_CPUS (e.g. "12,13,14,15").
    // Default: CCD1 SMT siblings 24..31 (may collide with OMP, slow).
    const char* cpus_env = getenv("CRAFTAX_POOL_CPUS");
    int cpus[64]; int ncpus = 0;
    if (cpus_env && *cpus_env) {
        const char* s = cpus_env;
        while (*s && ncpus < 64) {
            while (*s == ',' || *s == ' ') s++;
            if (!*s) break;
            cpus[ncpus++] = atoi(s);
            while (*s && *s != ',') s++;
        }
    }
    p->threads = (pthread_t*)calloc((size_t)num_producers, sizeof(pthread_t));
    for (int i = 0; i < num_producers; i++) {
        ProducerArg* a = (ProducerArg*)malloc(sizeof(*a));
        a->pool = p;
        a->producer_idx = i;
        a->cpu = ncpus > 0 ? cpus[i % ncpus] : (24 + (i % 8));
        pthread_create(&p->threads[i], NULL, producer_main, a);
    }

    // Pre-fill: wait until pool has at least 3/4 ready worlds (bounded).
    for (int waits = 0; waits < 2000; waits++) {
        int ready = 0;
        for (int i = 0; i < capacity; i++)
            if (atomic_load_explicit(&p->state[i], memory_order_relaxed) == SLOT_READY) ready++;
        if (ready >= 3 * capacity / 4) break;
        usleep(1000);
    }
    return p;
}

void craftax_pool_destroy(WorldPool* p) {
    if (!p) return;
    atomic_store(&p->running, 0);
    for (int i = 0; i < p->num_producers; i++) pthread_join(p->threads[i], NULL);
    free(p->threads);
    free(p->worlds);
    free((void*)p->state);
    free(p);
}

void craftax_pool_stats(WorldPool* p, uint64_t* produced, uint64_t* consumed,
                        uint64_t* fallbacks, int* ready_count) {
    if (produced)  *produced  = atomic_load_explicit(&p->total_produced,  memory_order_relaxed);
    if (consumed)  *consumed  = atomic_load_explicit(&p->total_consumed,  memory_order_relaxed);
    if (fallbacks) *fallbacks = atomic_load_explicit(&p->total_fallbacks, memory_order_relaxed);
    if (ready_count) {
        int r = 0;
        for (int i = 0; i < p->capacity; i++)
            if (atomic_load_explicit(&p->state[i], memory_order_relaxed) == SLOT_READY) r++;
        *ready_count = r;
    }
}

void craftax_step_batch_compact_pool(
    WorldPool* pool,
    EnvState* states, const int32_t* actions,
    uint8_t* obs, float* rewards, int8_t* dones,
    int num_envs)
{
    uint64_t local_fallbacks = 0;
    #pragma omp parallel for schedule(static) reduction(+:local_fallbacks)
    for (int i = 0; i < num_envs; i++) {
        float r; int d;
        craftax_step(&states[i], (int)actions[i], &r, &d);
        rewards[i] = r;
        dones[i] = (int8_t)d;
        if (d) {
            if (!pool_try_pop(pool, &states[i])) {
                _craftax_generate_world(&states[i],
                    pool->master_seed,
                    atomic_fetch_add_explicit(&pool->producer_env_id, 1, memory_order_relaxed));
                local_fallbacks++;
            }
        }
        craftax_build_obs_compact(&states[i], obs + (size_t)i * OBS_DIM_COMPACT);
    }
    atomic_fetch_add_explicit(&pool->total_fallbacks, local_fallbacks, memory_order_relaxed);
}

// ============================================================
// Thread-pool + world-pool combined variant.
// ============================================================
typedef struct {
    WorldPool* wpool;
    EnvState* states;
    const int32_t* actions;
    uint8_t* obs;
    float* rewards;
    int8_t* dones;
    _Atomic uint64_t fallbacks;
} PoolStepWork;

static void pool_step_worker(void* arg, int id, int nw, int begin, int end) {
    (void)id; (void)nw;
    PoolStepWork* w = (PoolStepWork*)arg;
    uint64_t local_fb = 0;
    for (int i = begin; i < end; i++) {
        float r; int d;
        craftax_step(&w->states[i], (int)w->actions[i], &r, &d);
        w->rewards[i] = r;
        w->dones[i] = (int8_t)d;
        if (d) {
            if (!pool_try_pop(w->wpool, &w->states[i])) {
                _craftax_generate_world(&w->states[i],
                    w->wpool->master_seed,
                    atomic_fetch_add_explicit(&w->wpool->producer_env_id, 1,
                                              memory_order_relaxed));
                local_fb++;
            }
        }
        craftax_build_obs_compact(&w->states[i],
            w->obs + (size_t)i * OBS_DIM_COMPACT);
    }
    atomic_fetch_add_explicit(&w->fallbacks, local_fb, memory_order_relaxed);
}

void craftax_step_batch_compact_pool_tp(struct ThreadPool* tp, WorldPool* pool,
                                        EnvState* states, const int32_t* actions,
                                        uint8_t* obs, float* rewards, int8_t* dones,
                                        int num_envs) {
    PoolStepWork w = {
        .wpool = pool, .states = states, .actions = actions,
        .obs = obs, .rewards = rewards, .dones = dones,
    };
    atomic_store(&w.fallbacks, 0);
    worker_pool_run(tp, num_envs, pool_step_worker, &w);
    atomic_fetch_add_explicit(&pool->total_fallbacks,
        atomic_load(&w.fallbacks), memory_order_relaxed);
}
