# LUMINA-SN Makefile
# Monte Carlo Radiative Transfer in C with LUMINA Rotation Algorithm
#
# Usage:
#   make                  - Build all targets
#   make test-kernels     - Run physics kernel validation
#   make test-transport   - Build full transport test
#   make test-atomic      - Build and run atomic data loader test
#   make simulate         - Run 100k packet simulation
#   make OPENMP=1         - Build with OpenMP parallelization
#   make cuda             - Build CUDA-accelerated transport (requires NVCC)

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math
CFLAGS_DEBUG = -Wall -Wextra -g -O0 -DDEBUG
LDFLAGS = -lm

# HDF5 library paths (user's local installation)
HDF5_ROOT = $(HOME)/local
HDF5_INCLUDE = -I$(HDF5_ROOT)/include
HDF5_LIB = -L$(HDF5_ROOT)/lib -Wl,-rpath,$(HDF5_ROOT)/lib -lhdf5 -lhdf5_hl

# CUDA configuration
NVCC = nvcc
CUDA_ARCH = -arch=native  # Auto-detect GPU architecture (or set to sm_75, sm_80, etc.)
NVCC_FLAGS = -O3 -use_fast_math $(CUDA_ARCH) -Xcompiler -fPIC
NVCC_FLAGS_OMP = $(NVCC_FLAGS) -Xcompiler -fopenmp
# Auto-detect CUDA path from nvcc location if CUDA_HOME not set
CUDA_ROOT ?= $(shell dirname $$(dirname $$(which nvcc)))
CUDA_INCLUDE = -I$(CUDA_ROOT)/include
CUDA_LIB = -L$(CUDA_ROOT)/lib64 -Wl,-rpath,$(CUDA_ROOT)/lib64 -lcudart -lpthread -lstdc++

# OpenMP support
ifdef OPENMP
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
    $(info OpenMP enabled)
endif

# Headers
HDRS = physics_kernels.h rpacket.h validation.h lumina_rotation.h atomic_data.h plasma_physics.h simulation_state.h
CUDA_HDRS = cuda_atomic.h cuda_interface.h

# Library sources
LIB_SRCS = rpacket.c validation.c lumina_rotation.c atomic_loader.c plasma_physics.c simulation_state.c debug_rng.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

# CUDA sources
CUDA_SRCS = cuda_transport.cu cuda_atomic.cu gpu_transport.cu
CUDA_OBJS = cuda_transport.o cuda_atomic.o gpu_transport.o

# Main targets
.PHONY: all clean test-kernels test-atomic test-plasma simulate simulate-integrated debug help cuda

all: liblumin.a test_kernels test_transport test_atomic test_plasma test_integrated

# Static library
liblumin.a: $(LIB_OBJS)
	ar rcs $@ $^

# Object files
%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

# Physics kernel tests
test_kernels: test_kernels.c physics_kernels.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

# Full transport test (needs HDF5 for liblumin.a)
test_transport: test_transport.c liblumin.a
	$(CC) $(CFLAGS) $(HDF5_INCLUDE) $< -L. -llumin $(LDFLAGS) $(HDF5_LIB) -o $@

# Full transport test with CUDA support (Task Order #019)
# Links C code with CUDA via cuda_interface.h
test_transport_cuda: test_transport.c liblumin.a gpu_transport.o
	$(CC) $(CFLAGS) -DENABLE_CUDA -DHAVE_HDF5 -fopenmp $(HDF5_INCLUDE) $< gpu_transport.o -L. -llumin $(LDFLAGS) $(HDF5_LIB) $(CUDA_LIB) -o $@

# Atomic data loader test
test_atomic: test_atomic.c atomic_loader.o atomic_data.h
	$(CC) $(CFLAGS) $(HDF5_INCLUDE) $< atomic_loader.o $(LDFLAGS) $(HDF5_LIB) -o $@

# Atomic loader object (needs HDF5)
atomic_loader.o: atomic_loader.c atomic_data.h
	$(CC) $(CFLAGS) $(HDF5_INCLUDE) -c $< -o $@

# Plasma physics object (needs HDF5 for atomic data)
plasma_physics.o: plasma_physics.c plasma_physics.h atomic_data.h
	$(CC) $(CFLAGS) $(HDF5_INCLUDE) -c $< -o $@

# Plasma physics test
test_plasma: test_plasma.c plasma_physics.o atomic_loader.o atomic_data.h plasma_physics.h
	$(CC) $(CFLAGS) $(HDF5_INCLUDE) $< plasma_physics.o atomic_loader.o $(LDFLAGS) $(HDF5_LIB) -o $@

# Simulation state object (needs HAVE_HDF5 for plasma state injection)
simulation_state.o: simulation_state.c simulation_state.h plasma_physics.h atomic_data.h rpacket.h
	$(CC) $(CFLAGS) -DHAVE_HDF5 $(HDF5_INCLUDE) -c $< -o $@

# Macro-atom object
macro_atom.o: macro_atom.c macro_atom.h atomic_data.h rpacket.h physics_kernels.h
	$(CC) $(CFLAGS) $(HDF5_INCLUDE) -c $< -o $@

# Virtual packet object (TARDIS-style spectrum synthesis)
virtual_packet.o: virtual_packet.c virtual_packet.h simulation_state.h physics_kernels.h
	$(CC) $(CFLAGS) $(HDF5_INCLUDE) -c $< -o $@

# Integrated plasma-transport test
test_integrated: test_integrated.c simulation_state.o plasma_physics.o atomic_loader.o lumina_rotation.o macro_atom.o virtual_packet.o
	$(CC) $(CFLAGS) $(HDF5_INCLUDE) $< simulation_state.o plasma_physics.o atomic_loader.o lumina_rotation.o macro_atom.o virtual_packet.o $(LDFLAGS) $(HDF5_LIB) -o $@

# TARDIS trace generator for validation
test_tardis_trace: test_tardis_trace.c rpacket.o simulation_state.o plasma_physics.o atomic_loader.o
	$(CC) $(CFLAGS) $(HDF5_INCLUDE) $< rpacket.o simulation_state.o plasma_physics.o atomic_loader.o $(LDFLAGS) $(HDF5_LIB) -o $@

# ============================================================================
# CUDA TARGETS
# ============================================================================

# CUDA object files
cuda_transport.o: cuda_transport.cu cuda_atomic.h
	$(NVCC) $(NVCC_FLAGS) $(HDF5_INCLUDE) -c $< -o $@

cuda_atomic.o: cuda_atomic.cu cuda_atomic.h atomic_data.h plasma_physics.h simulation_state.h
	$(NVCC) $(NVCC_FLAGS) $(HDF5_INCLUDE) -c $< -o $@

# GPU transport interface (Task Order #019/020: C-CUDA bridge + kernels)
gpu_transport.o: gpu_transport.cu cuda_interface.h cuda_shared.h
	$(NVCC) $(NVCC_FLAGS_OMP) -c $< -o $@

# Standalone GPU transport test
test_gpu_transport: gpu_transport.cu cuda_interface.h
	$(NVCC) $(NVCC_FLAGS_OMP) -DGPU_TRANSPORT_STANDALONE $< -o $@

# CUDA test driver (uses -Xlinker for rpath since nvcc doesn't accept -Wl,)
HDF5_LIB_NVCC = -L$(HDF5_ROOT)/lib -Xlinker -rpath -Xlinker $(HDF5_ROOT)/lib -lhdf5 -lhdf5_hl
CUDA_LIB_NVCC = -L$(CUDA_HOME)/lib64 -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 -lcudart
test_cuda: test_cuda.cu $(CUDA_OBJS) simulation_state.o plasma_physics.o atomic_loader.o
	$(NVCC) $(NVCC_FLAGS) $(HDF5_INCLUDE) $< $(CUDA_OBJS) simulation_state.o plasma_physics.o atomic_loader.o -lm $(HDF5_LIB_NVCC) $(CUDA_LIB_NVCC) -lpthread -lstdc++ -o $@

# Build all CUDA components
cuda: test_cuda test_gpu_transport
	@echo ""
	@echo "=== CUDA Transport Built ==="
	@echo "Run with: ./test_cuda atomic/kurucz_cd23_chianti_H_He.h5 1000000 spectrum_cuda.csv"

# Test CUDA infrastructure (Task Order #019)
test-cuda-interface: test_gpu_transport
	@echo ""
	@echo "=== Testing CUDA Interface (Task Order #019) ==="
	./test_gpu_transport

# Build CUDA-enabled transport test
cuda-transport: test_transport_cuda
	@echo ""
	@echo "=== CUDA-Enabled Transport Built ==="
	@echo "Run with: ./test_transport_cuda --simulate 10000"

# Run CUDA simulation
simulate-cuda: test_cuda
	@echo ""
	@echo "=== Running CUDA Monte Carlo (1M packets) ==="
	./test_cuda atomic/kurucz_cd23_chianti_H_He.h5 1000000 spectrum_cuda.csv

# Run kernel validation
test-kernels: test_kernels
	@echo ""
	@echo "=== Running Physics Kernel Tests ==="
	./test_kernels

# Run atomic data loader test
test-atomic: test_atomic
	@echo ""
	@echo "=== Running Atomic Data Loader Test ==="
	./test_atomic atomic/kurucz_cd23_chianti_H_He.h5

# Run plasma physics test
test-plasma: test_plasma
	@echo ""
	@echo "=== Running Plasma Physics Test ==="
	./test_plasma atomic/kurucz_cd23_chianti_H_He.h5

# Run TARDIS comparison
validate-plasma: test_plasma
	@echo ""
	@echo "=== Running C Plasma Test ==="
	./test_plasma atomic/kurucz_cd23_chianti_H_He.h5 10000 1e-10
	@echo ""
	@echo "=== Running Python Validation ==="
	python3 validate_plasma_tardis.py --temperature 10000 --density 1e-10

# Run simulation
simulate: test_transport
	@echo ""
	@echo "=== Running Monte Carlo Simulation ==="
	./test_transport --simulate 100000 --spectrum spectrum.csv

# Quick simulation (fewer packets, for testing)
simulate-quick: test_transport
	./test_transport --simulate 10000 --spectrum spectrum_quick.csv

# Integrated simulation with realistic atomic opacities
simulate-integrated: test_integrated
	@echo ""
	@echo "=== Running Integrated Plasma-Transport Simulation ==="
	./test_integrated atomic/kurucz_cd23_chianti_H_He.h5 100000 spectrum_integrated.csv

# Quick integrated simulation
simulate-integrated-quick: test_integrated
	./test_integrated atomic/kurucz_cd23_chianti_H_He.h5 10000 spectrum_integrated_quick.csv

# Validation against Python
validate: test_transport
	@echo ""
	@echo "=== Generating Python Trace ==="
	python3 validation_dump.py --output trace_python.bin --csv trace_python.csv
	@echo ""
	@echo "=== Running C Validation ==="
	./test_transport --validate trace_python.bin --trace trace_c.bin --csv trace_c.csv

# Debug build
debug: CFLAGS = $(CFLAGS_DEBUG)
debug: clean all

# Clean
clean:
	rm -f *.o *.a test_kernels test_transport test_transport_cuda test_atomic test_plasma test_integrated test_cuda test_gpu_transport
	rm -f *.bin *.csv spectrum*.csv trace*.csv

# Deep clean (including backup files)
distclean: clean
	rm -f *~ *.bak

# Help
help:
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║                    LUMINA-SN Build System                     ║"
	@echo "╠═══════════════════════════════════════════════════════════════╣"
	@echo "║                                                               ║"
	@echo "║  Targets:                                                     ║"
	@echo "║    all                 - Build library and executables        ║"
	@echo "║    test-kernels        - Physics kernel validation (19 tests) ║"
	@echo "║    test-atomic         - Test atomic data HDF5 loader         ║"
	@echo "║    test-plasma         - Test Saha-Boltzmann plasma solver    ║"
	@echo "║    simulate            - Run 100k packet MC (dummy opacities) ║"
	@echo "║    simulate-integrated - Run 100k with real atomic opacities  ║"
	@echo "║    validate            - Compare C vs Python implementation   ║"
	@echo "║    clean               - Remove build artifacts               ║"
	@echo "║                                                               ║"
	@echo "║  Options:                                                     ║"
	@echo "║    OPENMP=1            - Enable OpenMP parallelization        ║"
	@echo "║                                                               ║"
	@echo "║  Examples:                                                    ║"
	@echo "║    make simulate-integrated  # Full physics simulation        ║"
	@echo "║    make OPENMP=1 simulate    # Parallel simulation            ║"
	@echo "║    ./test_integrated data.h5 50000 output.csv                 ║"
	@echo "║                                                               ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"

# Dependency tracking
rpacket.o: rpacket.c physics_kernels.h rpacket.h
validation.o: validation.c physics_kernels.h rpacket.h validation.h
lumina_rotation.o: lumina_rotation.c physics_kernels.h lumina_rotation.h validation.h
