CC      ?= gcc
CFLAGS  ?= -O3 -march=native -mtune=native -ffast-math -fno-math-errno \
           -funroll-loops -flto -fopenmp -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -flto -fopenmp -lm

all: bench

bench: bench.o craftax.o craftax_pool.o worker_pool.o
	$(CC) -o $@ $^ $(LDFLAGS) -lpthread

bench_hot_pool: bench_hot_pool.o craftax.o craftax_pool.o worker_pool.o
	$(CC) -o $@ $^ $(LDFLAGS) -lpthread

bench_tp: bench_tp.o craftax.o craftax_pool.o worker_pool.o
	$(CC) -o $@ $^ $(LDFLAGS) -lpthread

bench_tp_pool: bench_tp_pool.o craftax.o craftax_pool.o worker_pool.o
	$(CC) -o $@ $^ $(LDFLAGS) -lpthread

%.o: %.c craftax.h
	$(CC) $(CFLAGS) -c -o $@ $<

run: bench
	OMP_PROC_BIND=close OMP_PLACES=cores ./bench

clean:
	rm -f *.o bench

.PHONY: all run clean
