#!/bin/bash
# 동일 물질70/충돌에서 BF·BB 입력 J만 직접 CMFGEN으로 교체한 지정 원래점 후보.
#SBATCH --job-name=lumina-fixed-direct
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
test -n "${LUMINA_DIRECT_J_PROJECTION:-}"
test -n "${LUMINA_FULL_BOUND_RESULT:-}"
test -n "${LUMINA_DIRECT_BB_RESULT:-}"
test -n "${LUMINA_COLL_RESULT:-}"
test ! -e "$PWD/native-occupation.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
timeout -k 15 570s bash "$LUMINA_PROJECT/tests/run_bf_point_tests.sh" \
  "$LUMINA_PROJECT" "$PWD" model-native "$LUMINA_COLL_RESULT"
