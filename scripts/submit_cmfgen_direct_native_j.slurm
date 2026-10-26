#!/bin/bash
# 같은 물질70의 완성 CMFGEN 직접 J를 자체 원90점/ν 격자에만 보간한다.
# 새 수송/SE/RE는 없으며 출력 root는 새 격리 공간이어야 한다.
#SBATCH --job-name=lumina-direct-native-j
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:0
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --no-requeue

source /etc/profile
set -euo pipefail
umask 027
module purge
test "${SLURM_CPUS_PER_TASK:-}" = 1
test "${SLURM_MEM_PER_NODE:-}" = 4096
test -n "${LUMINA_PROJECT:-}"
test -n "${LUMINA_DIRECT_ROOT:-}"
test -d "$LUMINA_PROJECT/scripts"
test ! -e "$PWD/projected"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$LUMINA_PROJECT/scripts"
timeout -k 15 1170s python3 "$LUMINA_PROJECT/scripts/project_cmfgen_direct_native_j.py" \
  --base /gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json \
  --direct-root "$LUMINA_DIRECT_ROOT" --out "$PWD/projected"
