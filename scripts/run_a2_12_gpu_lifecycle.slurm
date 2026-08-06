#!/usr/bin/env bash
set -euo pipefail

# sbatch 는 스크립트를 스풀 복사하므로 BASH_SOURCE 로는 저장소를 찾을 수 없다.
# SLURM_SUBMIT_DIR(제출 디렉터리)을 정본으로 쓰고, 대화형 실행 시에만 fallback.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  repo_root="$SLURM_SUBMIT_DIR"
else
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
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
