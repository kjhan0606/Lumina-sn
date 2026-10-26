#!/bin/bash
# 원 물질70·봉인 J_A의 전 rawPHOT bound 율 완성. 새 수송/물질/SE 해는 아님.
# 제출 시 --chdir와 --output을 새 격리 root로 지정한다.
#SBATCH --job-name=lumina-point-bound
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --no-requeue

source /etc/profile
set -euo pipefail
umask 027
test "${SLURM_CPUS_PER_TASK:-}" = 1
test "${SLURM_MEM_PER_NODE:-}" = 4096
test "${LUMINA_BOUND_ROOT:-}" = "$PWD"
test -d "$PWD/scripts"
test ! -e "$PWD/run"
test ! -e "$PWD/point.so"
export LUMINA_BOUND_WALL_SECONDS=7200
timeout -k 15 7140s bash tests/run_model_point_bound.sh \
  /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn "$PWD"
