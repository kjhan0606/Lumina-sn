#!/bin/bash
# 사용자 승인: 물질70의 직접 연속항 CMFGEN 복사장, 단회·CPU16·32GiB·30분.
# 제출할 때 --chdir와 --output은 새 격리 root로 지정한다.
#SBATCH --job-name=lumina-direct-m70
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue

source /etc/profile
module purge
module load gnu13
set -euo pipefail
umask 027

printf 'PREFLIGHT job=%s cpus=%s mem=%s root=%s\n' \
  "${SLURM_JOB_ID:-unset}" "${SLURM_CPUS_PER_TASK:-unset}" \
  "${SLURM_MEM_PER_NODE:-unset}" "$PWD"
test "${SLURM_CPUS_PER_TASK:-}" = 16
test "${SLURM_MEM_PER_NODE:-}" = 32768
test "${LUMINA_DIRECT_ROOT:-}" = "$PWD"
test -d "$PWD/scripts"
test ! -e "$PWD/run"

export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/scripts"
ulimit -s unlimited

python3 -m pytest -q -p no:cacheprovider \
  --confcutdir="$PWD/tests" --basetemp="$PWD/pytest-allocation" \
  tests/test_cmfgen_direct_continuum.py
timeout -k 15 1780s python3 scripts/run_cmfgen_direct_continuum.py \
  --reference /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429062 \
  --out "$PWD/run" --execute
