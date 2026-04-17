// Implementation: persistent worker threads spin on a generation counter.
// Main publishes work (fn/arg/total + bump generation); each worker grabs its
// slice, executes, and increments done_count. Main spins on done_count.
//
// Padding ensures done_count and generation don't share a cache line with
// other state (false-sharing kills spin-barrier latency).
#define _GNU_SOURCE
#include "worker_pool.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define CACHE_LINE 64
#define CL_ALIGN __attribute__((aligned(CACHE_LINE)))

struct ThreadPool {
    int num_workers;
    pthread_t* threads;
    int* cpus;

    // Hot control state on separate cache lines (avoid false sharing).
    CL_ALIGN _Atomic uint64_t generation;
    char _pad0[CACHE_LINE - sizeof(_Atomic uint64_t)];
    CL_ALIGN _Atomic int done_count;
    char _pad1[CACHE_LINE - sizeof(_Atomic int)];
    CL_ALIGN _Atomic int shutdown;
    char _pad2[CACHE_LINE - sizeof(_Atomic int)];

    // Payload (written by main before bumping generation, read by workers).
    work_fn_t fn;
    void* arg;
    int total;
} CL_ALIGN;

typedef struct { ThreadPool* pool; int id; } WorkerArg;

static void pin_cpu(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

static void* worker_main(void* arg_) {
    WorkerArg* a = (WorkerArg*)arg_;
    ThreadPool* p = a->pool;
    int id = a->id;
    if (p->cpus) pin_cpu(p->cpus[id]);

    uint64_t seen = 0;
    for (;;) {
        // Spin until a new generation is published.
        uint64_t g;
        for (;;) {
            g = atomic_load_explicit(&p->generation, memory_order_acquire);
            if (g != seen) break;
            if (atomic_load_explicit(&p->shutdown, memory_order_relaxed)) {
                free(a); return NULL;
            }
            _mm_pause();
        }
        seen = g;
        if (atomic_load_explicit(&p->shutdown, memory_order_relaxed)) { free(a); return NULL; }

        int total = p->total;
        int nw = p->num_workers;
        // Balanced chunking: workers < (total%nw) take chunk+1, rest take chunk.
        int base = total / nw;
        int rem  = total - base * nw;
        int begin = id * base + (id < rem ? id : rem);
        int extra = (id < rem) ? 1 : 0;
        int end = begin + base + extra;
        if (begin < end) p->fn(p->arg, id, nw, begin, end);

        atomic_fetch_add_explicit(&p->done_count, 1, memory_order_release);
    }
}

ThreadPool* worker_pool_create(int num_workers, const int* cpus) {
    ThreadPool* p = (ThreadPool*)aligned_alloc(CACHE_LINE, sizeof(*p));
    memset(p, 0, sizeof(*p));
    p->num_workers = num_workers;
    p->threads = (pthread_t*)calloc((size_t)num_workers, sizeof(pthread_t));
    if (cpus) {
        p->cpus = (int*)malloc((size_t)num_workers * sizeof(int));
        memcpy(p->cpus, cpus, (size_t)num_workers * sizeof(int));
    }
    atomic_store(&p->generation, 0);
    atomic_store(&p->done_count, 0);
    atomic_store(&p->shutdown, 0);
    for (int i = 0; i < num_workers; i++) {
        WorkerArg* a = (WorkerArg*)malloc(sizeof(*a));
        a->pool = p; a->id = i;
        pthread_create(&p->threads[i], NULL, worker_main, a);
    }
    return p;
}

void worker_pool_destroy(ThreadPool* p) {
    if (!p) return;
    atomic_store(&p->shutdown, 1);
    // Wake workers so they observe shutdown.
    atomic_fetch_add_explicit(&p->generation, 1, memory_order_release);
    for (int i = 0; i < p->num_workers; i++) pthread_join(p->threads[i], NULL);
    free(p->threads);
    free(p->cpus);
    free(p);
}

void worker_pool_run(ThreadPool* p, int total, work_fn_t fn, void* arg) {
    p->fn = fn; p->arg = arg; p->total = total;
    // Reset done_count, then publish new generation (release ordering so
    // workers see fn/arg/total after they see the bumped generation).
    atomic_store_explicit(&p->done_count, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->generation, 1, memory_order_release);
    int nw = p->num_workers;
    // Spin until all workers have incremented done_count.
    for (;;) {
        int d = atomic_load_explicit(&p->done_count, memory_order_acquire);
        if (d >= nw) break;
        _mm_pause();
    }
}

int worker_pool_num_workers(const ThreadPool* p) { return p->num_workers; }
