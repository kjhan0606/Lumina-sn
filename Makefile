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
SOURCES = src/lumina_main.c src/lumina_transport.c src/lumina_plasma.c src/lumina_atomic.c
HEADERS = src/lumina.h
TARGET = lumina

# CUDA source
CUDA_SRC = src/lumina_cuda.cu
NVCC = nvcc
NVFLAGS = -O2 -arch=sm_89 -std=c++14 -Xcompiler -fopenmp
NVLDFLAGS = -lm -Xcompiler -fopenmp

# Default target
all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# CUDA build target (compile C sources alongside .cu)
cuda: lumina_cuda
lumina_cuda: $(CUDA_SRC) src/lumina_atomic.c src/lumina_plasma.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o lumina_cuda $(CUDA_SRC) src/lumina_atomic.c src/lumina_plasma.c $(NVLDFLAGS)

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
