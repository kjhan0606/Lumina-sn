# Makefile — Phase 5: Build LUMINA-SN
CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

# Phase 5: OpenMP support (set OMP=1 to enable)
ifdef OMP
CFLAGS += -fopenmp
LDFLAGS += -fopenmp
endif

# Phase 5: Source files
SOURCES = lumina_main.c lumina_transport.c lumina_plasma.c lumina_atomic.c
HEADERS = lumina.h
TARGET = lumina

# Phase 5: CUDA source (Phase 6)
CUDA_SRC = lumina_cuda.cu
NVCC = nvcc
NVFLAGS = -O2 -arch=sm_89 -std=c++14

# Phase 5: Default target
all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# Phase 6: CUDA build target (compile C sources alongside .cu)
cuda: lumina_cuda
lumina_cuda: $(CUDA_SRC) lumina_atomic.c lumina_plasma.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o lumina_cuda $(CUDA_SRC) lumina_atomic.c lumina_plasma.c $(LDFLAGS)

# Phase 5: Clean
clean:
	rm -f $(TARGET) lumina_cuda *.o

# Phase 5: Run with defaults
run: $(TARGET)
	./$(TARGET) tardis_reference

# Phase 5: Quick test with fewer packets
test: $(TARGET)
	./$(TARGET) tardis_reference 10000 5

# Phase 5: OpenMP build
omp:
	$(MAKE) OMP=1

.PHONY: all clean run test omp cuda
