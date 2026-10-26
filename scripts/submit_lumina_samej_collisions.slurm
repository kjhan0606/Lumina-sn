#!/bin/bash -l
# 완료된 동일J 계수의 충돌 스위치 단독 대조. 새 계수/수송/native 없음.
#SBATCH --job-name=lumina-sameJ-CI
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
set -euo pipefail
umask 027
ulimit -c 0
module purge
module load gnu13/13.2.0
project=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$project/scripts"
parent=$(realpath "${1:?새 parent 필요}")
test -d "$parent"
test ! -e "$parent/run"
cd "$project"
python3 -m pytest -q -p no:cacheprovider tests/test_samej_collision_replay.py
timeout -k 15 500s python3 scripts/replay_lumina_samej_collisions.py \
  --source /gpfs/kjhan/cmfgen_runs/lumina_sameJ_nativeBB_20261027-y7f92T \
  --out "$parent/run"
