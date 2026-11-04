#!/bin/bash -l
# 승인된 같은 상태 h=0.5/0.25 포획 각1회, 순차 실행, 재시도 없음.
#SBATCH --job-name=lumina-fixedj-grid
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
umask 027
module purge
module load gnu13/13.2.0
ulimit -s unlimited
ulimit -c 0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
project=$(realpath "${2:?repository root required}")
test -f "$project/scripts/run_cmfgen_fixedj_grid_bundle.py"
exec python3 -B "$project/scripts/run_cmfgen_fixedj_grid_bundle.py" \
    --bundle-root "${1:?isolated bundle root required}"
