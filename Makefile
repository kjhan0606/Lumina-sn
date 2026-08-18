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
SOURCES = src/lumina_main.c src/lumina_transport.c src/a2_02c_segment_capture.c src/radiation_field.c src/bf_rate_jnu.c src/line_jbar.c src/seed_capability.c src/jnu_seed.c src/population_contract.c src/atomic_internal_energy.c src/opacity_publication.c src/emissivity_publication.c src/radeq_publication.c src/line_net_rate.c src/cmfgen_adiabatic.c src/cmf_exact_sliding.c src/cmf_error_envelope.c src/nlte_population_candidate.c src/physics_comparison.c src/gpu_physics_contract.c src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c src/lumina_cmfgen.c
# ★2026-08-07: 이 목록이 12개였고, src 가 실제로 #include 하는 헤더는 21개였다.
#   빠진 9개는 바뀌어도 리빌드가 일어나지 않는다 — 그 중에 **계약 헤더**들이 있었다
#   (gpu_radiation_field_contract · gpu_physics_kernels · lumina_cmfgen).
#   T3 잡 226529 진단 중 발각: env_universe.h 를 483→496 으로 재생성했는데
#   `make` 가 "up to date" 라 했고 바이너리는 옛 전집을 유지했다.
#   목록은 `python3 scripts/check_makefile_headers.py` 가 기계로 대조한다.
HEADERS = src/lumina.h src/a2_02c_segment_capture.h src/radiation_field.h src/bf_rate_jnu.h \
          src/lumina_frequency_grid.h \
          src/line_jbar.h src/seed_capability.h src/jnu_seed.h src/population_contract.h \
          src/atomic_internal_energy.h src/line_net_rate.h \
          src/opacity_publication.h src/emissivity_publication.h src/radeq_publication.h \
          src/cmfgen_adiabatic.h \
          src/cmf_exact_sliding.h \
          src/cmf_exact_multigpu.h \
          src/cmf_error_envelope.h \
          src/nlte_population_candidate.h \
          src/physics_comparison.h \
          src/gpu_physics_contract.h \
          src/env_universe.h src/gpu_emissivity_kernels.h src/gpu_opacity_kernels.h \
          src/bf_event_measure_access.h \
          src/gpu_physics_kernels.h src/gpu_radiation_field.h src/gpu_radiation_field_contract.h \
          src/lumina_cmf_field.h src/lumina_cmfgen.h src/lumina_radeq_col_pairs.h
TARGET = lumina
POPULATION_SRC = src/population_contract.c
A2_PUBLICATION_SRC = src/opacity_publication.c src/emissivity_publication.c src/radeq_publication.c src/line_net_rate.c src/cmfgen_adiabatic.c
NLTE_CANDIDATE_SRC = src/nlte_population_candidate.c
ATOMIC_INTERNAL_SRC = src/atomic_internal_energy.c

# CUDA source
CUDA_SRC = src/lumina_cuda.cu
NVCC = nvcc
LUMINA_CUDA_OUTPUT ?= lumina_cuda
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
CUDA_CMF_EXACT  = src/cmf_exact_multigpu.cu
CUDA_RF_MIRROR  = src/gpu_radiation_field.cu
GPU_RF_CONTRACT = src/gpu_radiation_field_contract.c
GPU_PHYSICS_CONTRACT = src/gpu_physics_contract.c
GPU_PHYSICS_KERNELS = src/gpu_physics_kernels.cu
GPU_OPACITY_KERNELS = src/gpu_opacity_kernels.cu
GPU_EMISSIVITY_KERNELS = src/gpu_emissivity_kernels.cu

cuda: lumina_cuda
lumina_cuda: $(CUDA_SRC) $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) $(CUDA_NLTE_ASM) $(CUDA_CMF_SOLVE) $(CUDA_CMF_EXACT) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) $(GPU_PHYSICS_CONTRACT) $(GPU_PHYSICS_KERNELS) $(GPU_OPACITY_KERNELS) $(GPU_EMISSIVITY_KERNELS) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) $(ATOMIC_INTERNAL_SRC) $(A2_PUBLICATION_SRC) $(NLTE_CANDIDATE_SRC) src/physics_comparison.c src/bf_rate_jnu.c src/radiation_field.c src/line_jbar.c src/cmf_exact_sliding.c src/cmf_error_envelope.c src/seed_capability.c src/lumina_element_wide.c src/lumina_cmfgen.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o $(LUMINA_CUDA_OUTPUT) $(CUDA_SRC) $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) $(CUDA_NLTE_ASM) $(CUDA_CMF_SOLVE) $(CUDA_CMF_EXACT) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) $(GPU_PHYSICS_CONTRACT) $(GPU_PHYSICS_KERNELS) $(GPU_OPACITY_KERNELS) $(GPU_EMISSIVITY_KERNELS) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) $(ATOMIC_INTERNAL_SRC) $(A2_PUBLICATION_SRC) $(NLTE_CANDIDATE_SRC) src/physics_comparison.c src/bf_rate_jnu.c src/radiation_field.c src/line_jbar.c src/cmf_exact_sliding.c src/cmf_error_envelope.c src/seed_capability.c src/lumina_element_wide.c src/lumina_cmfgen.c $(NVLDFLAGS)

selftest_cmf_exact_sliding: tests/cmf_exact_sliding_selftest.c \
		src/cmf_exact_sliding.c src/cmf_exact_sliding.h \
		src/cmf_error_envelope.c src/cmf_error_envelope.h
	$(CC) -O2 -Wall -Wextra -std=c11 -fopenmp -Isrc -o $@ \
		tests/cmf_exact_sliding_selftest.c src/cmf_exact_sliding.c \
		src/cmf_error_envelope.c -lm -fopenmp

selftest_cmf_error_envelope: tests/cmf_error_envelope_selftest.c \
		src/cmf_error_envelope.c src/cmf_error_envelope.h
	$(CC) -O2 -Wall -Wextra -Werror -pedantic -std=c11 -Isrc -o $@ \
		tests/cmf_error_envelope_selftest.c src/cmf_error_envelope.c -lm

selftest_cmf_exact_multigpu: tests/cmf_exact_multigpu_selftest.cu \
		src/cmf_exact_multigpu.cu src/cmf_exact_multigpu.h \
		src/cmf_exact_sliding.c src/cmf_exact_sliding.h \
		src/cmf_error_envelope.c src/cmf_error_envelope.h
	$(NVCC) $(NVFLAGS) -Isrc -o $@ \
		tests/cmf_exact_multigpu_selftest.cu src/cmf_exact_multigpu.cu \
		src/cmf_exact_sliding.c src/cmf_error_envelope.c $(NVLDFLAGS)

selftest_cmf_exact_epoch_scan: tests/cmf_exact_epoch_scan_selftest.cu
	$(NVCC) $(NVFLAGS) -Isrc -o $@ \
		tests/cmf_exact_epoch_scan_selftest.cu $(NVLDFLAGS)

bench_cmf_exact_multigpu_reduced: tests/cmf_exact_multigpu_reduced_bench.cu \
		src/cmf_exact_multigpu.cu src/cmf_exact_multigpu.h \
		src/cmf_exact_sliding.c src/cmf_exact_sliding.h \
		src/cmf_error_envelope.c src/cmf_error_envelope.h
	$(NVCC) $(NVFLAGS) -Isrc -o $@ \
		tests/cmf_exact_multigpu_reduced_bench.cu src/cmf_exact_multigpu.cu \
		src/cmf_exact_sliding.c src/cmf_error_envelope.c $(NVLDFLAGS)

# Task #39 validation harness
bench_bf_gemm: bench_bf_gemm.c $(CUDA_BF_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o bench_bf_gemm bench_bf_gemm.c $(CUDA_BF_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c $(NVLDFLAGS)

# Task #40 validation harness
bench_nlte_rates: bench_nlte_rates.c $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o bench_nlte_rates bench_nlte_rates.c $(CUDA_BF_GEMM) $(CUDA_NLTE_GEMM) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c $(NVLDFLAGS)

# GPU bound-bound assembly self-check
selftest_nlte_assemble: selftest_nlte_assemble.c $(CUDA_NLTE_ASM) $(CUDA_NLTE_GEMM) $(CUDA_BF_GEMM) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c $(HEADERS)
	$(NVCC) $(NVFLAGS) -o selftest_nlte_assemble selftest_nlte_assemble.c $(CUDA_NLTE_ASM) $(CUDA_NLTE_GEMM) $(CUDA_BF_GEMM) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) src/lumina_atomic.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c $(NVLDFLAGS)

selftest_a2_12_contract: tests/a2_12_contract_selftest.c $(GPU_RF_CONTRACT) src/gpu_radiation_field_contract.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_12_contract_selftest.c $(GPU_RF_CONTRACT)

selftest_a2_13_15_contract: tests/a2_13_15_contract_selftest.c $(GPU_PHYSICS_CONTRACT) src/gpu_physics_contract.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_13_15_contract_selftest.c $(GPU_PHYSICS_CONTRACT)

selftest_a2_12_gpu_lifecycle: tests/a2_12_gpu_lifecycle_selftest.cu $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) src/radiation_field.c src/seed_capability.c src/gpu_radiation_field.h src/gpu_radiation_field_contract.h src/radiation_field.h
	$(NVCC) $(NVFLAGS) -Isrc -o $@ tests/a2_12_gpu_lifecycle_selftest.cu \
		$(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) src/radiation_field.c src/seed_capability.c $(NVLDFLAGS)

selftest_a2_13_gpu_oracle: tests/a2_13_gpu_oracle.cu $(GPU_PHYSICS_KERNELS) $(GPU_OPACITY_KERNELS) $(GPU_EMISSIVITY_KERNELS) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) $(GPU_PHYSICS_CONTRACT) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c
	$(NVCC) $(NVFLAGS) -Isrc -o $@ tests/a2_13_gpu_oracle.cu \
		$(GPU_PHYSICS_KERNELS) $(GPU_OPACITY_KERNELS) $(GPU_EMISSIVITY_KERNELS) $(CUDA_RF_MIRROR) $(GPU_RF_CONTRACT) \
		$(GPU_PHYSICS_CONTRACT) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c $(NVLDFLAGS)

# Known-answer (Saha) self-test of the ionization/photoionization path — CPU only
selftest_ioniz_saha: selftest_ioniz_saha.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) $(NLTE_CANDIDATE_SRC) $(ATOMIC_INTERNAL_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -o selftest_ioniz_saha selftest_ioniz_saha.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) $(NLTE_CANDIDATE_SRC) $(ATOMIC_INTERNAL_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

# seed T_e 부트스트랩 발행의 음성대조 배터리 (docs/RUNG_SEED_TE_PUBLICATION.md G2/G4/G5)
selftest_seed_te_publish: tests/seed_te_publish_selftest.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) $(NLTE_CANDIDATE_SRC) $(ATOMIC_INTERNAL_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -o $@ tests/seed_te_publish_selftest.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) $(NLTE_CANDIDATE_SRC) $(ATOMIC_INTERNAL_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

# MC-EVT NE4 구조 게이트 — 접근자를 우회해 chi_bf 를 직접 읽는 소비자가 없는가.
# ★Codex 가 src/Makefile 을 새로 만들어 냈으나 저장소 빌드 주체는 루트 Makefile 이다.
event-measure-check:
	python3 scripts/check_event_measure_access.py
	python3 scripts/compare_event_measure_spectra.py --selftest

selftest-sh-radeq-source:
	python3 scripts/check_a209_source_failclosed.py

selftest_mc_evt_access: tests/mc_evt_access_selftest.cu src/bf_event_measure_access.h src/lumina.h
	$(NVCC) -O2 $(GPU_GENCODE) -std=c++14 -Isrc -o $@ $<

# 격자 포함 계약(안 B) 왕복 항등식 — docs/RUNG_GRID_CONTAINMENT_CONTRACT.md B-2
# ★Codex 가 소스만 내고 규칙을 빠뜨려 손으로 빌드해야 했다.  드리프트 검사 대상에 넣는다.
selftest_grid_roundtrip: src/radiation_field_roundtrip_selftest.c src/radiation_field.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -Isrc -o $@ src/radiation_field_roundtrip_selftest.c src/radiation_field.c $(LDFLAGS)

# SH-GRID stale/corrupt CMFGEN sigma assets must fail before a fallback model
# can silently replace the requested cross sections.
selftest_sh_grid_loader: tests/sh_grid_loader_selftest.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -ffunction-sections \
		-fdata-sections -Isrc -Wl,--gc-sections -o $@ \
		tests/sh_grid_loader_selftest.c src/lumina_atomic.c $(LDFLAGS)

# L1-1 부트스트랩 창의 음성대조 배터리 (docs/RUNG_L1_1_BOOTSTRAP_SUPPLIER.md G3/G5)
selftest_bootstrap_window: tests/bootstrap_window_selftest.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) $(NLTE_CANDIDATE_SRC) $(ATOMIC_INTERNAL_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -o $@ tests/bootstrap_window_selftest.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) $(NLTE_CANDIDATE_SRC) $(ATOMIC_INTERNAL_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

# Gate-B Phase-1.6 deterministic frozen-cell oracle (CPU single-thread observer)
bench_frozen_oracle: bench_frozen_oracle.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
		-o bench_frozen_oracle bench_frozen_oracle.c \
		src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

# Wave-3.2 A4 operational-RC negative control.  Only the frozen entry symbol is
# wrapped; production allocation and judgment paths remain unchanged.
bench_frozen_oracle_rc: bench_frozen_oracle.c tests/wave32_ew_rc_wrap.c src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
		-Dnlte_element_wide_run_labeled=wave32_rc_wrapped_nlte_element_wide_run_labeled \
		-c bench_frozen_oracle.c -o /tmp/w32_rc_bench.o
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE -Isrc \
		-o bench_frozen_oracle_rc /tmp/w32_rc_bench.o tests/wave32_ew_rc_wrap.c \
		src/lumina_plasma.c $(POPULATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_ew_rc: tests/wave32_ew_rc_selftest.c tests/wave32_ew_rc_wrap.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
		-ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
		-o selftest_wave32_ew_rc tests/wave32_ew_rc_selftest.c \
		tests/wave32_ew_rc_wrap.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c \
		src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_ew_io: tests/wave32_ew_io_selftest.c src/lumina_element_wide.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE \
		-ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
		-o selftest_wave32_ew_io tests/wave32_ew_io_selftest.c \
		src/lumina_element_wide.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c \
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
selftest_wave32_matrix_debit: tests/wave32_matrix_debit_seed.c src/lumina_element_wide.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_wave32_matrix_debit \
		tests/wave32_matrix_debit_seed.c src/lumina_element_wide.c \
		src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_within_sl_oom: tests/wave32_within_sl_oom.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_element_wide.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -Wl,--wrap=malloc -o selftest_wave32_within_sl_oom \
		tests/wave32_within_sl_oom.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c \
		src/lumina_element_wide.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_boundary_q: tests/wave32_boundary_q_seed.c src/lumina_element_wide.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_wave32_boundary_q \
		tests/wave32_boundary_q_seed.c src/lumina_element_wide.c \
		src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_atomic.c $(LDFLAGS)

selftest_wave32_counter_atomic: tests/wave32_counter_atomic_selftest.c src/lumina_element_wide.c src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_atomic.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -DWAVE32_COUNTER_SELFTEST -fopenmp -ffunction-sections -fdata-sections -Isrc \
		-Wl,--gc-sections -o selftest_wave32_counter_atomic \
		tests/wave32_counter_atomic_selftest.c src/lumina_element_wide.c \
		src/lumina_plasma.c $(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c src/lumina_atomic.c $(LDFLAGS)

# A2-03 canonical RadiationField schema/lifecycle and failure-injection fixture.
selftest_a2_03_radiation_field: tests/a2_03_radiation_field_selftest.c src/radiation_field.c src/seed_capability.c src/radiation_field.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_03_radiation_field_selftest.c src/radiation_field.c src/seed_capability.c $(LDFLAGS)

selftest_a2_03_producer_parity_fixture: tests/a2_03_producer_parity_fixture.c src/lumina_transport.c src/radiation_field.c src/seed_capability.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -ffunction-sections -fdata-sections \
		-Isrc -Wl,--gc-sections -o $@ tests/a2_03_producer_parity_fixture.c \
		src/lumina_transport.c src/radiation_field.c src/seed_capability.c src/line_jbar.c $(LDFLAGS)

# A2-04 single producer-commit API, atomic generation and failure injection.
selftest_a2_04_commit: tests/a2_04_commit_selftest.c src/radiation_field.c src/seed_capability.c src/radiation_field.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_04_commit_selftest.c src/radiation_field.c src/seed_capability.c $(LDFLAGS)

selftest_a2_04_replay_commit: tests/a2_04_replay_commit.c src/radiation_field.c src/seed_capability.c src/radiation_field.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_04_replay_commit.c src/radiation_field.c src/seed_capability.c $(LDFLAGS)

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
		selftest_cmfgen_adiabatic \
		selftest_atomic_internal_energy \
		selftest_nlte_population_candidate \
		selftest_nlte_candidate_tau \
		selftest_a2_10_seed_commit \
		selftest_nlte_candidate_adiabatic \
		selftest_physics_comparison \
		selftest_physics_comparison_regrid \
		selftest_line_net_rate \
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
	event-measure-check selftest-sh-radeq-source selftest_mc_evt_access \
	selftest-bf-edge-census bf-edge-census selftest-tau-writer-census \
	selftest_a2_03_radiation_field selftest_a2_03_producer_parity_fixture \
	selftest_a2_04_commit selftest_a2_04_replay_commit \
	selftest_a2_12_contract selftest_a2_12_gpu_lifecycle \
	selftest_a2_13_15_contract selftest_a2_13_gpu_oracle \
	selftest_line_net_rate selftest-a2-10-cmfgen-mapped-line

# A2-05 canonical-view bf rate selftest (analytic closed forms + R6 validity)
selftest_a2_05_bf_rate: tests/a2_05_bf_rate_selftest.c src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c $(HEADERS)
	$(CC) -O2 -std=c11 -Isrc -o selftest_a2_05_bf_rate \
		tests/a2_05_bf_rate_selftest.c src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c $(LDFLAGS)

# A2-05 L-1bf gate fixture (deterministic/MC commit -> per-level Gamma)
l1bf_fixture: tests/a2_05_l1bf_fixture.c src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c $(HEADERS)
	$(CC) -O2 -std=c11 -Isrc -o l1bf_fixture \
		tests/a2_05_l1bf_fixture.c src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c $(LDFLAGS)

# A2-06 selective line-Jbar estimator selftest (closed-form phi integrals,
# packet-population variance identity, Q-set hash determinism)
selftest_a2_06_line_jbar: tests/a2_06_line_jbar_selftest.c src/line_jbar.c src/line_jbar.h
	$(CC) -O2 -std=gnu11 -Isrc -o selftest_a2_06_line_jbar \
		tests/a2_06_line_jbar_selftest.c src/line_jbar.c $(LDFLAGS)

# A2-06 dual-view commit selftest (partial-commit invariance, view codes, SE)
selftest_a2_06_dual_commit: tests/a2_06_dual_commit_selftest.c src/radiation_field.c src/seed_capability.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -Isrc -o selftest_a2_06_dual_commit \
		tests/a2_06_dual_commit_selftest.c src/radiation_field.c src/seed_capability.c $(LDFLAGS)

selftest_a2_08_signed_opacity: tests/a2_08_signed_opacity_selftest.c src/opacity_publication.c src/opacity_publication.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_08_signed_opacity_selftest.c src/opacity_publication.c $(LDFLAGS)
	python3 scripts/run_a2_08_selftest.py --binary ./selftest_a2_08_signed_opacity

selftest_a2_09_emissivity: tests/a2_09_emissivity_selftest.c src/emissivity_publication.c src/emissivity_publication.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_09_emissivity_selftest.c src/emissivity_publication.c $(LDFLAGS)
	python3 scripts/run_a2_09_selftest.py --binary ./selftest_a2_09_emissivity

# MC-EVT prerequisite: prove whether BF thresholds exist below NLTE_NU_MIN.
# The selftest is green; the deck census deliberately returns rc=3 when
# SH-GRID must be reopened, so it is kept as an explicit audit target.
selftest-bf-edge-census:
	python3 scripts/census_bf_edges_below_grid.py --selftest

selftest-tau-writer-census:
	python3 scripts/check_tau_writer_generation.py

bf-edge-census:
	python3 scripts/census_bf_edges_below_grid.py \
		--ref-dir data/tardis_reference_toy06_19p48d_sivcaiv_active

selftest_a2_10_radeq: tests/a2_10_radeq_selftest.c src/radeq_publication.c src/opacity_publication.c src/population_contract.c src/radeq_publication.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/a2_10_radeq_selftest.c src/radeq_publication.c src/opacity_publication.c src/population_contract.c $(LDFLAGS)
	python3 scripts/run_a2_10_selftest.py --binary ./selftest_a2_10_radeq

selftest-a2-10-cancellation-census:
	python3 tests/a2_10_cancellation_census_selftest.py

selftest-a2-10-refinement-comparison:
	python3 tests/a2_10_refinement_comparison_selftest.py

selftest-a2-10-targeted-gate:
	python3 tests/a2_10_targeted_gate_selftest.py

selftest-a2-10-targeted-reference:
	python3 tests/a2_10_targeted_reference_selftest.py

selftest-a2-10-line-ion-owners:
	python3 tests/a2_10_line_ion_owner_summary_selftest.py
	python3 tests/a2_10_line_saturation_summary_selftest.py
	python3 tests/cmfgen_lineheat_ion_owner_summary_selftest.py
	python3 tests/cmfgen_line_components_ion_owner_selftest.py
	python3 tests/a2_10_cmfgen_ion_owner_comparison_selftest.py
	python3 tests/a2_10_cmfgen_ion_component_comparison_selftest.py
	python3 tests/a2_10_line_owner_closure_monitor_selftest.py
	python3 tests/a2_10_line_owner_component_monitor_selftest.py

selftest-a2-10-line-saturation:
	python3 tests/a2_10_line_saturation_summary_selftest.py
	python3 tests/a2_10_line_saturation_intersection_selftest.py
	python3 tests/a2_10_phase_baseline_streams_selftest.py
	python3 tests/a2_10_cmfgen_line_saturation_comparison_selftest.py
	python3 tests/a2_10_line_saturation_per_ion_coverage_selftest.py
	python3 tests/a2_10_line_saturation_per_ion_monitor_selftest.py

selftest-a2-10-cmfgen-mapped-line:
	python3 tests/a2_10_cmfgen_mapped_line_comparison_selftest.py

selftest_line_net_rate: tests/line_net_rate_selftest.c \
		src/line_net_rate.c src/line_net_rate.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/line_net_rate_selftest.c src/line_net_rate.c $(LDFLAGS)
	./selftest_line_net_rate

selftest_cmfgen_adiabatic: tests/cmfgen_adiabatic_selftest.c \
		src/cmfgen_adiabatic.c src/cmfgen_adiabatic.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/cmfgen_adiabatic_selftest.c src/cmfgen_adiabatic.c $(LDFLAGS)
	./selftest_cmfgen_adiabatic

selftest_atomic_internal_energy: tests/atomic_internal_energy_selftest.c \
		src/atomic_internal_energy.c src/atomic_internal_energy.h \
		src/population_contract.c src/population_contract.h src/lumina.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/atomic_internal_energy_selftest.c \
		src/atomic_internal_energy.c src/population_contract.c $(LDFLAGS)
	./selftest_atomic_internal_energy

selftest_nlte_population_candidate: tests/nlte_population_candidate_selftest.c \
		src/nlte_population_candidate.c src/nlte_population_candidate.h \
		src/population_contract.c src/population_contract.h $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/nlte_population_candidate_selftest.c \
		src/nlte_population_candidate.c src/population_contract.c \
		src/opacity_publication.c src/emissivity_publication.c \
		src/atomic_internal_energy.c src/cmfgen_adiabatic.c $(LDFLAGS)
	./selftest_nlte_population_candidate

selftest_nlte_candidate_adiabatic: tests/nlte_candidate_adiabatic_selftest.c \
		src/nlte_population_candidate.c src/nlte_population_candidate.h \
		src/atomic_internal_energy.c src/atomic_internal_energy.h \
		src/cmfgen_adiabatic.c src/cmfgen_adiabatic.h \
		src/population_contract.c src/population_contract.h $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/nlte_candidate_adiabatic_selftest.c \
		src/nlte_population_candidate.c src/atomic_internal_energy.c \
		src/cmfgen_adiabatic.c src/population_contract.c \
		src/opacity_publication.c src/emissivity_publication.c $(LDFLAGS)
	./selftest_nlte_candidate_adiabatic

selftest_nlte_candidate_tau: tests/nlte_candidate_tau_selftest.c \
		src/nlte_population_candidate.c src/nlte_population_candidate.h \
		src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c \
		$(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c \
		src/radiation_field.c src/seed_capability.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections \
		-Isrc -Wl,--gc-sections -o $@ tests/nlte_candidate_tau_selftest.c \
		src/nlte_population_candidate.c src/lumina_plasma.c \
		src/lumina_element_wide.c src/lumina_atomic.c $(POPULATION_SRC) \
		$(ATOMIC_INTERNAL_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c \
		src/seed_capability.c $(LDFLAGS)
	./selftest_nlte_candidate_tau

# A2-INIT seed-material commit: fail-closed generation/provenance controls +
# positive material-changes/Te-preserved control.
selftest_a2_10_seed_commit: tests/a2_10_seed_commit_selftest.c \
		src/nlte_population_candidate.c src/nlte_population_candidate.h \
		src/lumina_plasma.c src/lumina_element_wide.c src/lumina_atomic.c \
		$(POPULATION_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c \
		src/radiation_field.c src/seed_capability.c $(HEADERS)
	$(CC) -O2 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections \
		-Isrc -Wl,--gc-sections -o $@ tests/a2_10_seed_commit_selftest.c \
		src/nlte_population_candidate.c src/lumina_plasma.c \
		src/lumina_element_wide.c src/lumina_atomic.c $(POPULATION_SRC) \
		$(ATOMIC_INTERNAL_SRC) $(A2_PUBLICATION_SRC) src/bf_rate_jnu.c src/radiation_field.c \
		src/seed_capability.c $(LDFLAGS)
	./selftest_a2_10_seed_commit

selftest_physics_comparison: tests/physics_comparison_selftest.c \
		src/physics_comparison.c src/physics_comparison.h \
		src/atomic_internal_energy.c src/radeq_publication.c $(HEADERS)
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o $@ \
		tests/physics_comparison_selftest.c src/physics_comparison.c \
		src/atomic_internal_energy.c src/radeq_publication.c \
		src/population_contract.c src/opacity_publication.c \
		src/emissivity_publication.c $(LDFLAGS)
	./selftest_physics_comparison

selftest_physics_comparison_regrid:
	python3 tests/physics_comparison_regrid_selftest.py

# A2-07 generation-bound Z(T_e), validity and transactional-publish contract.
selftest_a2_07_population: tests/a2_07_population_selftest.c src/population_contract.c src/population_contract.h
	$(CC) -O2 -Wall -Wextra -std=c11 -Isrc -o selftest_a2_07_population \
		tests/a2_07_population_selftest.c src/population_contract.c $(LDFLAGS)

# A2-16 scalar-seed capability selftest (state machine + N16-1..5)
selftest_a2_16_seed: tests/a2_16_seed_capability_selftest.c src/seed_capability.c src/seed_capability.h
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -Isrc -o selftest_a2_16_seed \
		tests/a2_16_seed_capability_selftest.c src/seed_capability.c $(LDFLAGS)

# A2-17 native J_nu seed codec. The legacy W/T_rad parser is deliberately
# confined to the separate offline executable and is never in SOURCES.
selftest_a2_17_jnu_seed: tests/a2_17_jnu_seed_selftest.c src/jnu_seed.c \
		src/jnu_seed.h src/radiation_field.c src/seed_capability.c
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -Isrc -o $@ \
		tests/a2_17_jnu_seed_selftest.c src/jnu_seed.c \
		src/radiation_field.c src/seed_capability.c $(LDFLAGS)

lumina_legacy_seed_converter: tools/lumina_legacy_seed_converter.c \
		src/jnu_seed.c src/jnu_seed.h src/radiation_field.c src/seed_capability.c
	$(CC) -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -Isrc -o $@ \
		tools/lumina_legacy_seed_converter.c src/jnu_seed.c \
		src/radiation_field.c src/seed_capability.c $(LDFLAGS)
