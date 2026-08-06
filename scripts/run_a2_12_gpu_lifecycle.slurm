#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p validation/a2_12
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-UNSET}"
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv,noheader
sha256sum src/gpu_radiation_field.cu src/gpu_radiation_field_contract.c \
  tests/a2_12_gpu_lifecycle_selftest.cu
make selftest_a2_12_gpu_lifecycle
sha256sum selftest_a2_12_gpu_lifecycle
python3 scripts/run_a2_12_gpu_lifecycle.py \
  --binary ./selftest_a2_12_gpu_lifecycle --out-dir validation/a2_12
