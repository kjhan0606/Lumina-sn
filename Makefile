# Makefile — LUMINA-SN
CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

# OpenMP support (set OMP=1 to enable)
ifdef OMP
CFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

# Source files (in src/)
SOURCES = src/lumina_main.c src/lumina_transport.c src/lumina_plasma.c src/lumina_atomic.c src/lumina_cmfgen.c
HEADERS = src/lumina.h
TARGET = lumina

# CUDA source
CUDA_SRC = src/lumina_cuda.cu
NVCC = nvcc
GPU_ARCH ?= sm_86
NVFLAGS = -O2 -arch=$(GPU_ARCH) -std=c++14 -Xcompiler -fopenmp -DLUMINA_HAS_CUDA_BF_GEMM
NVLDFLAGS = -lm -lcublas -Xcompiler -fopenmp

# Default target
all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# CUDA build target (compile C sources alongside .cu)
CUDA_BF_GEMM    = src/lumina_bf_gemm.cu
CUDA_NLTE_GEMM  = src/lumina_nlte_gemm.cu

cuda: lumina_cuda
lumina_cuda: $(CUDA_SRC) $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) src/lumina_atomic.c src/lumina_plasma.c src/lumina_cmfgen.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o lumina_cuda $(CUDA_SRC) $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) src/lumina_atomic.c src/lumina_plasma.c src/lumina_cmfgen.c $(NVLDFLAGS)

# Task #39 validation harness
bench_bf_gemm: bench_bf_gemm.c $(CUDA_BF_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o bench_bf_gemm bench_bf_gemm.c $(CUDA_BF_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(NVLDFLAGS)

# Task #40 validation harness
bench_nlte_rates: bench_nlte_rates.c $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o bench_nlte_rates bench_nlte_rates.c $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(NVLDFLAGS)

# Clean
clean:
	rm -f $(TARGET) lumina_cuda *.o

# Run with defaults
run: $(TARGET)
	./$(TARGET) data/tardis_reference

# Quick test with fewer packets
test: $(TARGET)
	./$(TARGET) data/tardis_reference 10000 5

# OpenMP build
omp:
	$(MAKE) OMP=1

.PHONY: all clean run test omp cuda
