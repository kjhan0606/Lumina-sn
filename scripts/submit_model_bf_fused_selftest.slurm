#!/bin/bash
# BF Milne/SE 행렬 결합의 합성 시험과 실제 native49 계수 소비를 대조한다.
# 별도 출력 root에서 CPU1·4GiB·5분, 새 복사장/물질 갱신 없음.
#SBATCH --job-name=lumina-bf-fused-test
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
test -d "$LUMINA_PROJECT/src"
test ! -e "$PWD/source-before.sha256"
if test -n "${LUMINA_COLL_RESULT:-}"; then
    bash "$LUMINA_PROJECT/tests/run_bf_point_tests.sh" \
      "$LUMINA_PROJECT" "$PWD" model-native "$LUMINA_COLL_RESULT"
else
    bash "$LUMINA_PROJECT/tests/run_bf_point_tests.sh" \
      "$LUMINA_PROJECT" "$PWD" model-native
fi
