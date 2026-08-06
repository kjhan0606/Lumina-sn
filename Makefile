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
SOURCES = src/lumina_main.c src/lumina_transport.c src/a2_02c_segment_capture.c src/radiation_field.c src/bf_rate_jnu.c src/line_jbar.c src/population_contract.c src/opacity_publication.c src/emissivity_publication.c src/radeq_publication.c src/gpu_physics_contract.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c src/lumina_cmfgen.c
HEADERS = src/lumina.h src/a2_02c_segment_capture.h src/radiation_field.h src/bf_rate_jnu.h src/line_jbar.h src/population_contract.h src/opacity_publication.h src/emissivity_publication.h src/radeq_publication.h src/gpu_physics_contract.h
TARGET = lumina
POPULATION_SRC = src/population_contract.c
A2_PUBLICATION_SRC = src/opacity_publication.c src/emissivity_publication.c src/radeq_publication.c

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
CUDA_RF_MIRROR  = src/gpu_radiation_field.cu
GPU_RF_CONTRACT = src/gpu_radiation_field_contract.c
GPU_PHYSICS_CONTRACT = src/gpu_physics_contract.c
GPU_PHYSICS_KERNELS = src/gpu_physics_kernels.cu
GPU_OPACITY_KERNELS = src/gpu_opacity_kernels.cu
GPU_EMISSIVITY_KERNELS = src/gpu_emissivity_kernels.cu

cuda: lumina_cuda
lumina_cuda: $(CUDA_SRC) $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) $(CUDA_NLTE_ASM) $(CUDA_CMF_SOLVE) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) $(GPU_PHYSICS_CONTRACT) $(GPU_PHYSICS_KERNELS) $(GPU_OPACITY_KERNELS) $(GPU_EMISSIVITY_KERNELS) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c src/lumina_cmfgen.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o lumina_cuda $(CUDA_SRC) $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) $(CUDA_NLTE_ASM) $(CUDA_CMF_SOLVE) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) $(GPU_PHYSICS_CONTRACT) $(GPU_PHYSICS_KERNELS) $(GPU_OPACITY_KERNELS) $(GPU_EMISSIVITY_KERNELS) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c src/lumina_cmfgen.c $(NVLDFLAGS)

# Task #39 validation harness
bench_bf_gemm: bench_bf_gemm.c $(CUDA_BF_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o bench_bf_gemm bench_bf_gemm.c $(CUDA_BF_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c $(NVLDFLAGS)

# Task #40 validation harness
bench_nlte_rates: bench_nlte_rates.c $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o bench_nlte_rates bench_nlte_rates.c $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c $(NVLDFLAGS)

# GPU bound-bound assembly self-check
selftest_nlte_assemble: selftest_nlte_assemble.c $(CUDA_NLTE_ASM) $(CUDA_NLTE_GEMM) $(CUDA_BF_GEMM) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o selftest_nlte_assemble selftest_nlte_assemble.c $(CUDA_NLTE_ASM) $(CUDA_NLTE_GEMM) $(CUDA_BF_GEMM) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c $(NVLDFLAGS)

selftest_a2_12_contract: tests/a2_12_contract_selftest.c $(GPU_RF_CONTRACT) src/gpu_radiation_field_contract.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_12_contract_selftest.c $(GPU_RF_CONTRACT)

selftest_a2_13_15_contract: tests/a2_13_15_contract_selftest.c $(GPU_PHYSICS_CONTRACT) src/gpu_physics_contract.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_13_15_contract_selftest.c $(GPU_PHYSICS_CONTRACT)

selftest_a2_12_gpu_lifecycle: tests/a2_12_gpu_lifecycle_selftest.cu $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) src/radiation_field.c src/gpu_radiation_field.h src/gpu_radiation_field_contract.h src/radiation_field.h
	$(NVCC) $(NVFLAGS) -Isrc -o $@ tests/a2_12_gpu_lifecycle_selftest.cu \
		$(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) src/radiation_field.c $(NVLDFLAGS)

selftest_a2_13_gpu_oracle: tests/a2_13_gpu_oracle.cu $(GPU_PHYSICS_KERNELS) $(GPU_OPACITY_KERNELS) $(GPU_EMISSIVITY_KERNELS) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) $(GPU_PHYSICS_CONTRACT) src/bf_rate_jnu.c src/radiation_field.c
	$(NVCC) $(NVFLAGS) -Isrc -o $@ tests/a2_13_gpu_oracle.cu \
		$(GPU_PHYSICS_KERNELS) $(GPU_OPACITY_KERNELS) $(GPU_EMISSIVITY_KERNELS) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) \
		$(GPU_PHYSICS_CONTRACT) src/bf_rate_jnu.c src/radiation_field.c $(NVLDFLAGS)

# Known-answer (Saha) self-test of the ionization/photoionization path — CPU only
selftest_ioniz_saha: selftest_ioniz_saha.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -o selftest_ioniz_saha selftest_ioniz_saha.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

# Gate-B Phase-1.6 deterministic frozen-cell oracle (CPU single-thread observer)
bench_frozen_oracle: bench_frozen_oracle.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
		-o bench_frozen_oracle bench_frozen_oracle.c \
		src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

# Wave-3.2 A4 operational-RC negative control.  Only the frozen entry symbol is
# wrapped; production allocation and judgment paths remain unchanged.
bench_frozen_oracle_rc: bench_frozen_oracle.c tests/wave32_ew_rc_wrap.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
		-Dnlte_element_wide_run_labeled=wave32_rc_wrapped_nlte_element_wide_run_labeled \
		-c bench_frozen_oracle.c -o /tmp/w32_rc_bench.o
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE -Isrc \
		-o bench_frozen_oracle_rc /tmp/w32_rc_bench.o tests/wave32_ew_rc_wrap.c \
		src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_ew_rc: tests/wave32_ew_rc_selftest.c tests/wave32_ew_rc_wrap.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
		-ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
		-o selftest_wave32_ew_rc tests/wave32_ew_rc_selftest.c \
		tests/wave32_ew_rc_wrap.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c \
		src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_ew_io: tests/wave32_ew_io_selftest.c src/lumina_element_wide.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE \
		-ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
		-o selftest_wave32_ew_io tests/wave32_ew_io_selftest.c \
		src/lumina_element_wide.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c \
		src/lumina_atomic.c $(LDFLAGS)

# Wave-3.2 R7 offline binary writer fixture (no model/GPU execution).
selftest_cmf_chieta_dump: scripts/cmf_chieta_writer_fixture.c src/lumina_cmfgen.c $(A2_PUBLICATION_SRC) src/lumina_cmfgen.h $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_cmf_chieta_dump \
		scripts/cmf_chieta_writer_fixture.c src/lumina_cmfgen.c $(A2_PUBLICATION_SRC) $(LDFLAGS)

# [CMF-LINEPOP T2] offline population-native line-dump writer fixture.
selftest_cmf_linepop_dump: scripts/cmf_linepop_writer_fixture.c src/lumina_cmfgen.c $(A2_PUBLICATION_SRC) src/lumina_cmfgen.h $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_cmf_linepop_dump \
		scripts/cmf_linepop_writer_fixture.c src/lumina_cmfgen.c $(A2_PUBLICATION_SRC) $(LDFLAGS)

# Stage 3.2 Rung 1: model-free writer + KA-3.2.3 positive/negative controls.
selftest_stage32_rung1: scripts/stage32_rung1_writer_fixture.c src/lumina_cmfgen.c $(A2_PUBLICATION_SRC) src/lumina_cmfgen.h $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_stage32_rung1_writer \
		scripts/stage32_rung1_writer_fixture.c src/lumina_cmfgen.c $(A2_PUBLICATION_SRC) $(LDFLAGS)
	python3 scripts/stage32_rung1_selftest.py

# E5 in-situ A/B/B2 assembly fixture, including a seeded n_u corruption.
selftest_emiss_ab_insitu: tests/emiss_ab_insitu_fixture.c src/lumina_cmfgen.c $(A2_PUBLICATION_SRC) src/lumina_cmfgen.h $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_emiss_ab_insitu \
		tests/emiss_ab_insitu_fixture.c src/lumina_cmfgen.c $(A2_PUBLICATION_SRC) $(LDFLAGS)

# E11 canonical fluorescence-matrix reader/writer, seeded edge corruption, and
# OFF-path byte-invariant fixture (offline only; no model or GPU execution).
selftest_emiss_e11_fluor_matrix:
	python3 -m py_compile scripts/emiss_e11_fluor_matrix.py \
		scripts/emiss_e11_seeded_fixture.py scripts/emiss_e11_off_byte_check.py \
		scripts/emiss_e10_apply_redistribution.py
	python3 scripts/emiss_e11_seeded_fixture.py --out-dir /tmp/emiss_e11_fixture

# Wave-3.2 D6 independent-ledger seeded debit corruption fixture.
selftest_wave32_matrix_debit: tests/wave32_matrix_debit_seed.c src/lumina_element_wide.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_wave32_matrix_debit \
		tests/wave32_matrix_debit_seed.c src/lumina_element_wide.c \
		src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_within_sl_oom: tests/wave32_within_sl_oom.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -Wl,--wrap=malloc -o selftest_wave32_within_sl_oom \
		tests/wave32_within_sl_oom.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c \
		src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_boundary_q: tests/wave32_boundary_q_seed.c src/lumina_element_wide.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_wave32_boundary_q \
		tests/wave32_boundary_q_seed.c src/lumina_element_wide.c \
		src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_counter_atomic: tests/wave32_counter_atomic_selftest.c src/lumina_element_wide.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -DWAVE32_COUNTER_SELFTEST -fopenmp -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_wave32_counter_atomic \
		tests/wave32_counter_atomic_selftest.c src/lumina_element_wide.c \
		src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/lumina_atomic.c $(LDFLAGS)

# A2-03 canonical RadiationField schema/lifecycle and failure-injection fixture.
selftest_a2_03_radiation_field: tests/a2_03_radiation_field_selftest.c src/radiation_field.c src/radiation_field.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_03_radiation_field_selftest.c src/radiation_field.c $(LDFLAGS)

selftest_a2_03_producer_parity_fixture: tests/a2_03_producer_parity_fixture.c src/lumina_transport.c src/radiation_field.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -ffunction-sections -fdata-sections \
		-Isrc -Wl,--gc-sections -o $@ tests/a2_03_producer_parity_fixture.c \
		src/lumina_transport.c src/radiation_field.c src/line_jbar.c $(LDFLAGS)

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
		selftest_a2_04_replay_commit selftest_a2_12_contract \
		selftest_a2_13_15_contract \
		selftest_a2_12_gpu_lifecycle selftest_a2_13_gpu_oracle *.o

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
	selftest_a2_04_commit selftest_a2_04_replay_commit \
	selftest_a2_12_contract selftest_a2_12_gpu_lifecycle \
	selftest_a2_13_15_contract selftest_a2_13_gpu_oracle

# A2-05 canonical-view bf rate selftest (analytic closed forms + R6 validity)
selftest_a2_05_bf_rate: tests/a2_05_bf_rate_selftest.c src/bf_rate_jnu.c src/radiation_field.c $(HEADERS)
	$(CC) -O2 -std=c11 -Isrc -o selftest_a2_05_bf_rate \
		tests/a2_05_bf_rate_selftest.c src/bf_rate_jnu.c src/radiation_field.c $(LDFLAGS)

# A2-05 L-1bf gate fixture (deterministic/MC commit -> per-level Gamma)
l1bf_fixture: tests/a2_05_l1bf_fixture.c src/bf_rate_jnu.c src/radiation_field.c $(HEADERS)
	$(CC) -O2 -std=c11 -Isrc -o l1bf_fixture \
		tests/a2_05_l1bf_fixture.c src/bf_rate_jnu.c src/radiation_field.c $(LDFLAGS)

# A2-06 selective line-Jbar estimator selftest (closed-form phi integrals,
# packet-population variance identity, Q-set hash determinism)
selftest_a2_06_line_jbar: tests/a2_06_line_jbar_selftest.c src/line_jbar.c src/line_jbar.h
	$(CC) -O2 -std=gnu11 -Isrc -o selftest_a2_06_line_jbar \
		tests/a2_06_line_jbar_selftest.c src/line_jbar.c $(LDFLAGS)

# A2-06 dual-view commit selftest (partial-commit invariance, view codes, SE)
selftest_a2_06_dual_commit: tests/a2_06_dual_commit_selftest.c src/radiation_field.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -Isrc -o selftest_a2_06_dual_commit \
		tests/a2_06_dual_commit_selftest.c src/radiation_field.c $(LDFLAGS)

selftest_a2_08_signed_opacity: tests/a2_08_signed_opacity_selftest.c src/opacity_publication.c src/opacity_publication.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_08_signed_opacity_selftest.c src/opacity_publication.c $(LDFLAGS)
	python3 scripts/run_a2_08_selftest.py --binary ./selftest_a2_08_signed_opacity

selftest_a2_09_emissivity: tests/a2_09_emissivity_selftest.c src/emissivity_publication.c src/emissivity_publication.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_09_emissivity_selftest.c src/emissivity_publication.c $(LDFLAGS)
	python3 scripts/run_a2_09_selftest.py --binary ./selftest_a2_09_emissivity

selftest_a2_10_radeq: tests/a2_10_radeq_selftest.c src/radeq_publication.c src/population_contract.c src/radeq_publication.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_10_radeq_selftest.c src/radeq_publication.c src/population_contract.c $(LDFLAGS)
	python3 scripts/run_a2_10_selftest.py --binary ./selftest_a2_10_radeq

# A2-07 generation-bound Z(T_e), validity and transactional-publish contract.
selftest_a2_07_population: tests/a2_07_population_selftest.c src/population_contract.c src/population_contract.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o selftest_a2_07_population \
		tests/a2_07_population_selftest.c src/population_contract.c $(LDFLAGS)
