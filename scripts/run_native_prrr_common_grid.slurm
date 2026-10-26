#!/bin/bash -l
# 원70/J70 PRRR 공통격자 read-only 율 진단. sbatch 인수: 새 격리 root.
#SBATCH --job-name=lumina-prrr-common
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
ulimit -c 0
project=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
root=$(realpath "${1:?새 격리 root 필요}")
test -d "$root"
test ! -e "$root/run"
test "${SLURM_CPUS_PER_TASK:-}" = 1
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$project"
timeout 3400s python3 scripts/compare_native_prrr_common_grid.py \
  --base /gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json \
  --prrr-dir /gpfs/kjhan/cmfgen_runs/lumina_verbose_se70_20261026-FUDAJh/run/material \
  --continuum /gpfs/kjhan/cmfgen_runs/lumina_verbose_se70_20261026-FUDAJh/run/material/CONT_FREQ \
  --prior /gpfs/kjhan/cmfgen_runs/lumina_rate_contract_20261026-bCgTkl/rates.csv \
  --output "$root/run" --node 49
