#!/bin/bash
# 완성 직접장·bound BF를 재사용하는 원래점 Si/S 준위용해율 단회 시험.
# 새 CMFGEN/GPU 수송·물질 갱신·SE/RE 없음. 제출 시 --chdir로 새 root 지정.
#SBATCH --job-name=lumina-dis-direct
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
test -n "${LUMINA_PROJECTION:-}"
test -n "${LUMINA_BOUND_RESULT:-}"
test -n "${LUMINA_POINT_LIBRARY:-}"
test -n "${LUMINA_REFERENCE_RATES:-}"
test ! -e "$PWD/run"
jobroot=$PWD
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$LUMINA_PROJECT"
python3 -m pytest -q -p no:cacheprovider --confcutdir="$PWD/tests" \
  --basetemp="$jobroot/pytest" tests/test_model_point_dissolution.py
timeout -k 15 3300s python3 scripts/run_lumina_model_point_dissolution_direct.py \
  --base /gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json \
  --field-projection "$LUMINA_PROJECTION" \
  --full-bound-result "$LUMINA_BOUND_RESULT" \
  --library "$LUMINA_POINT_LIBRARY" \
  --reference-rates "$LUMINA_REFERENCE_RATES" \
  --node "${LUMINA_NATIVE_NODE:-49}" --out "$jobroot/run"
