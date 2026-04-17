// Minimal spin-barrier thread pool. Replaces #pragma omp parallel for to
// avoid libgomp team-barrier overhead (~30% of hot-path time at high thread
// counts when batches are tight).
//
// Usage:
//   int cpus[] = {0,1,2,...,15};
//   ThreadPool* tp = worker_pool_create(16, cpus);
//   worker_pool_run(tp, num_items, my_work_fn, my_arg);
//   worker_pool_destroy(tp);
//
// work_fn_t is called once per worker with its [begin, end) slice of [0, total).
#pragma once
#include <stdint.h>

typedef struct ThreadPool ThreadPool;

typedef void (*work_fn_t)(void* arg, int worker_id, int num_workers, int begin, int end);

// cpus: optional array of CPU ids (one per worker) for pthread_setaffinity.
// Pass NULL to leave threads unpinned.
ThreadPool* worker_pool_create(int num_workers, const int* cpus);
void        worker_pool_destroy(ThreadPool* p);

// Spin-based parallel for. Blocks until all workers finish.
void worker_pool_run(ThreadPool* p, int total, work_fn_t fn, void* arg);

int  worker_pool_num_workers(const ThreadPool* p);
