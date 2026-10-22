#!/bin/bash -l
#SBATCH --job-name=lumina-siii-compact
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
module purge
module load gnu13

export OMP_NUM_THREADS=16
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${LUMINA_SIII_CAPTURE_PARENT:?새 포획 부모 경로가 필요합니다}"
test -d "$LUMINA_SIII_CAPTURE_PARENT"
test ! -e "$LUMINA_SIII_CAPTURE_PARENT/capture"
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn

exec /home/kjhan/miniconda3/bin/python3 scripts/run_cmfgen_siii_compact_capture.py \
  --previous /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429062 \
  --build /gpfs/kjhan/cmfgen_runs/lumina_siii_compact_build_20261023-nOoVGp/build \
  --root "$LUMINA_SIII_CAPTURE_PARENT/capture"
