#!/bin/bash
# 봉인 원49점 C 후보를 제어 재생하고 준위용해만 추가한 private 후보를 비교한다.
#SBATCH --job-name=lumina-pseudo-overlay
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue

source /etc/profile
set -euo pipefail
umask 027
test "${SLURM_CPUS_PER_TASK:-}" = 1
test "${SLURM_MEM_PER_NODE:-}" = 4096
test -n "${LUMINA_PROJECT:-}"
test -n "${LUMINA_BASELINE:-}"
test -n "${LUMINA_PSEUDO_RESULT:-}"
test -n "${LUMINA_BOUND_RESULT:-}"
test -n "${LUMINA_COLL_RESULT:-}"
test ! -e "$PWD/run"
jobroot=$PWD
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$LUMINA_PROJECT"
timeout -k 15 3300s python3 scripts/run_lumina_pseudo_candidate_overlay.py \
  --base /gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json \
  --baseline "$LUMINA_BASELINE" --pseudo-result "$LUMINA_PSEUDO_RESULT" \
  --bound-result "$LUMINA_BOUND_RESULT" --collision-result "$LUMINA_COLL_RESULT" \
  --node 49 --out "$jobroot/run"
