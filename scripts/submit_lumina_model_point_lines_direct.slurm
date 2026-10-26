#!/bin/bash
# 완성 CMFGEN 직접 J만 바꾼 원49점 전복사선 BB. 새 물질/수송/SE 없음.
#SBATCH --job-name=lumina-lines-direct
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:0
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --no-requeue

source /etc/profile
set -euo pipefail
umask 027
test "${SLURM_CPUS_PER_TASK:-}" = 1
test "${SLURM_MEM_PER_NODE:-}" = 4096
test -n "${LUMINA_PROJECT:-}"
test -n "${LUMINA_PROJECTION:-}"
test ! -e "$PWD/run"
test ! -e "$PWD/lines.so"
jobroot=$PWD
j_path=${LUMINA_PROJECTION%/*}/J_direct_native.npy
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$LUMINA_PROJECT"
gcc -O2 -std=c11 -fPIC -shared -Wl,-z,defs -Isrc -o "$jobroot/lines.so" src/model_source_kernels.c -lm
timeout -k 15 570s python3 scripts/run_lumina_model_point_lines.py \
  --base /gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json \
  --j "$j_path" \
  --field-projection "$LUMINA_PROJECTION" --library "$jobroot/lines.so" --out "$jobroot/run"
