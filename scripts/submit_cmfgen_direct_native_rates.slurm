#!/bin/bash
# 전주파수 직접 CMFGEN 장의 정상 완료 뒤 같은 물질70 Si/S 광율만 비교한다.
# 새 수송·물질/SE/RE 갱신은 없고, 부분장은 Python 소비자가 거부한다.
#SBATCH --job-name=lumina-direct-rates
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue

set -euo pipefail
test "${SLURM_CPUS_PER_TASK:-}" = 1
test "${SLURM_MEM_PER_NODE:-}" = 4096
test -n "${LUMINA_PROJECT:-}"
test -n "${LUMINA_DIRECT_ROOT:-}"
test -d "$LUMINA_PROJECT/scripts"
test -f "$LUMINA_DIRECT_ROOT/job.slurm"
test ! -e "$PWD/result"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$LUMINA_PROJECT/scripts"
python3 -m pytest -q -p no:cacheprovider --basetemp="$PWD/pytest-allocation" \
  "$LUMINA_PROJECT/tests/test_compare_cmfgen_direct_native_rates.py"
python3 "$LUMINA_PROJECT/scripts/compare_cmfgen_direct_native_rates.py" \
  --base /gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json \
  --direct-root "$LUMINA_DIRECT_ROOT" \
  --resident /gpfs/kjhan/cmfgen_runs/lumina_resident_field_20261026-qFP5rJ/run/resident-field.json \
  --out "$PWD/result"
