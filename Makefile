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
SOURCES = src/lumina_main.c src/lumina_transport.c src/a2_02c_segment_capture.c src/radiation_field.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c src/lumina_cmfgen.c
HEADERS = src/lumina.h src/a2_02c_segment_capture.h src/radiation_field.h
TARGET = lumina

# CUDA source
CUDA_SRC = src/lumina_cuda.cu
NVCC = nvcc
# Multi-arch fatbin by default so one binary runs on A100(sm_80)/A40(sm_86)/H100(sm_90).
# Override with e.g. `make GPU_ARCH=sm_86` for a single-arch build.
GPU_ARCH ?=
ifeq ($(strip $(GPU_ARCH)),)
GPU_GENCODE = -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90
else
GPU_GENCODE = -arch=$(GPU_ARCH)
endif
NVFLAGS = -O2 $(GPU_GENCODE) -std=c++14 -Xcompiler -fopenmp -DLUMINA_HAS_CUDA_BF_GEMM
NVLDFLAGS = -lm -lcublas -Xcompiler -fopenmp

# Default target
all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

# CUDA build target (compile C sources alongside .cu)
CUDA_BF_GEMM    = src/lumina_bf_gemm.cu
CUDA_NLTE_GEMM  = src/lumina_nlte_gemm.cu
CUDA_NLTE_ASM   = src/lumina_nlte_assemble.cu
CUDA_CMF_SOLVE  = src/lumina_cmf_solve.cu

cuda: lumina_cuda
lumina_cuda: $(CUDA_SRC) $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) $(CUDA_NLTE_ASM) $(CUDA_CMF_SOLVE) src/lumina_atomic.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_cmfgen.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o lumina_cuda $(CUDA_SRC) $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) $(CUDA_NLTE_ASM) $(CUDA_CMF_SOLVE) src/lumina_atomic.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_cmfgen.c $(NVLDFLAGS)

# Task #39 validation harness
bench_bf_gemm: bench_bf_gemm.c $(CUDA_BF_GEMM) src/lumina_atomic.c src/lumina_plasma.c src/lumina_element_wide.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o bench_bf_gemm bench_bf_gemm.c $(CUDA_BF_GEMM) src/lumina_atomic.c src/lumina_plasma.c src/lumina_element_wide.c $(NVLDFLAGS)

# Task #40 validation harness
bench_nlte_rates: bench_nlte_rates.c $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) src/lumina_atomic.c src/lumina_plasma.c src/lumina_element_wide.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o bench_nlte_rates bench_nlte_rates.c $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) src/lumina_atomic.c src/lumina_plasma.c src/lumina_element_wide.c $(NVLDFLAGS)

# GPU bound-bound assembly self-check
selftest_nlte_assemble: selftest_nlte_assemble.c $(CUDA_NLTE_ASM) $(CUDA_NLTE_GEMM) $(CUDA_BF_GEMM) src/lumina_atomic.c src/lumina_plasma.c src/lumina_element_wide.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o selftest_nlte_assemble selftest_nlte_assemble.c $(CUDA_NLTE_ASM) $(CUDA_NLTE_GEMM) $(CUDA_BF_GEMM) src/lumina_atomic.c src/lumina_plasma.c src/lumina_element_wide.c $(NVLDFLAGS)

# Known-answer (Saha) self-test of the ionization/photoionization path — CPU only
selftest_ioniz_saha: selftest_ioniz_saha.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -o selftest_ioniz_saha selftest_ioniz_saha.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

# Gate-B Phase-1.6 deterministic frozen-cell oracle (CPU single-thread observer)
bench_frozen_oracle: bench_frozen_oracle.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
		-o bench_frozen_oracle bench_frozen_oracle.c \
		src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

# Wave-3.2 A4 operational-RC negative control.  Only the frozen entry symbol is
# wrapped; production allocation and judgment paths remain unchanged.
bench_frozen_oracle_rc: bench_frozen_oracle.c tests/wave32_ew_rc_wrap.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
		-Dnlte_element_wide_run_labeled=wave32_rc_wrapped_nlte_element_wide_run_labeled \
		-c bench_frozen_oracle.c -o /tmp/w32_rc_bench.o
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE -Isrc \
		-o bench_frozen_oracle_rc /tmp/w32_rc_bench.o tests/wave32_ew_rc_wrap.c \
		src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_ew_rc: tests/wave32_ew_rc_selftest.c tests/wave32_ew_rc_wrap.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
		-ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
		-o selftest_wave32_ew_rc tests/wave32_ew_rc_selftest.c \
		tests/wave32_ew_rc_wrap.c src/lumina_plasma.c \
		src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_ew_io: tests/wave32_ew_io_selftest.c src/lumina_element_wide.c src/lumina_plasma.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE \
		-ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
		-o selftest_wave32_ew_io tests/wave32_ew_io_selftest.c \
		src/lumina_element_wide.c src/lumina_plasma.c \
		src/lumina_atomic.c $(LDFLAGS)

# Wave-3.2 R7 offline binary writer fixture (no model/GPU execution).
selftest_cmf_chieta_dump: scripts/cmf_chieta_writer_fixture.c src/lumina_cmfgen.c src/lumina_cmfgen.h $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_cmf_chieta_dump \
		scripts/cmf_chieta_writer_fixture.c src/lumina_cmfgen.c $(LDFLAGS)

# [CMF-LINEPOP T2] offline population-native line-dump writer fixture.
selftest_cmf_linepop_dump: scripts/cmf_linepop_writer_fixture.c src/lumina_cmfgen.c src/lumina_cmfgen.h $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_cmf_linepop_dump \
		scripts/cmf_linepop_writer_fixture.c src/lumina_cmfgen.c $(LDFLAGS)

# Stage 3.2 Rung 1: model-free writer + KA-3.2.3 positive/negative controls.
selftest_stage32_rung1: scripts/stage32_rung1_writer_fixture.c src/lumina_cmfgen.c src/lumina_cmfgen.h $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_stage32_rung1_writer \
		scripts/stage32_rung1_writer_fixture.c src/lumina_cmfgen.c $(LDFLAGS)
	python3 scripts/stage32_rung1_selftest.py

# E5 in-situ A/B/B2 assembly fixture, including a seeded n_u corruption.
selftest_emiss_ab_insitu: tests/emiss_ab_insitu_fixture.c src/lumina_cmfgen.c src/lumina_cmfgen.h $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_emiss_ab_insitu \
		tests/emiss_ab_insitu_fixture.c src/lumina_cmfgen.c $(LDFLAGS)

# E11 canonical fluorescence-matrix reader/writer, seeded edge corruption, and
# OFF-path byte-invariant fixture (offline only; no model or GPU execution).
selftest_emiss_e11_fluor_matrix:
	python3 -m py_compile scripts/emiss_e11_fluor_matrix.py \
		scripts/emiss_e11_seeded_fixture.py scripts/emiss_e11_off_byte_check.py \
		scripts/emiss_e10_apply_redistribution.py
	python3 scripts/emiss_e11_seeded_fixture.py --out-dir /tmp/emiss_e11_fixture

# Wave-3.2 D6 independent-ledger seeded debit corruption fixture.
selftest_wave32_matrix_debit: tests/wave32_matrix_debit_seed.c src/lumina_element_wide.c src/lumina_plasma.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_wave32_matrix_debit \
		tests/wave32_matrix_debit_seed.c src/lumina_element_wide.c \
		src/lumina_plasma.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_within_sl_oom: tests/wave32_within_sl_oom.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -Wl,--wrap=malloc -o selftest_wave32_within_sl_oom \
		tests/wave32_within_sl_oom.c src/lumina_plasma.c \
		src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_boundary_q: tests/wave32_boundary_q_seed.c src/lumina_element_wide.c src/lumina_plasma.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_wave32_boundary_q \
		tests/wave32_boundary_q_seed.c src/lumina_element_wide.c \
		src/lumina_plasma.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_counter_atomic: tests/wave32_counter_atomic_selftest.c src/lumina_element_wide.c src/lumina_plasma.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -DWAVE32_COUNTER_SELFTEST -fopenmp -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_wave32_counter_atomic \
		tests/wave32_counter_atomic_selftest.c src/lumina_element_wide.c \
		src/lumina_plasma.c src/lumina_atomic.c $(LDFLAGS)

# A2-03 canonical RadiationField schema/lifecycle and failure-injection fixture.
selftest_a2_03_radiation_field: tests/a2_03_radiation_field_selftest.c src/radiation_field.c src/radiation_field.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_03_radiation_field_selftest.c src/radiation_field.c $(LDFLAGS)

selftest_a2_03_producer_parity_fixture: tests/a2_03_producer_parity_fixture.c src/lumina_transport.c src/radiation_field.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -ffunction-sections -fdata-sections \
		-Isrc -Wl,--gc-sections -o $@ tests/a2_03_producer_parity_fixture.c \
		src/lumina_transport.c src/radiation_field.c $(LDFLAGS)

# A2-04 single producer-commit API, atomic generation and failure injection.
selftest_a2_04_commit: tests/a2_04_commit_selftest.c src/radiation_field.c src/radiation_field.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_04_commit_selftest.c src/radiation_field.c $(LDFLAGS)

selftest_a2_04_replay_commit: tests/a2_04_replay_commit.c src/radiation_field.c src/radiation_field.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_04_replay_commit.c src/radiation_field.c $(LDFLAGS)

# Clean
clean:
	rm -f $(TARGET) lumina_cuda bench_frozen_oracle_rc selftest_wave32_ew_rc \
		selftest_wave32_ew_io \
		selftest_cmf_chieta_dump \
		selftest_emiss_ab_insitu \
		selftest_wave32_matrix_debit selftest_wave32_within_sl_oom \
		selftest_wave32_boundary_q selftest_wave32_counter_atomic \
		selftest_a2_03_radiation_field \
		selftest_a2_03_producer_parity_fixture selftest_a2_04_commit \
		selftest_a2_04_replay_commit *.o

# Run with defaults
run: $(TARGET)
	./$(TARGET) data/tardis_reference

# Quick test with fewer packets
test: $(TARGET)
	./$(TARGET) data/tardis_reference 10000 5

# OpenMP build
omp:
	$(MAKE) OMP=1

.PHONY: all clean run test omp cuda selftest_emiss_e11_fluor_matrix \
	selftest_a2_03_radiation_field selftest_a2_03_producer_parity_fixture \
	selftest_a2_04_commit selftest_a2_04_replay_commit
