#!/bin/bash -l
# 전 PHOT 경로/원90점 공통장 BF 입력 단회 생산. 새 수송/native/물질 갱신 없음.
#SBATCH --job-name=lumina-material-bf90
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
set -euo pipefail
umask 027
ulimit -c 0
module purge
module load gnu13/13.2.0
project=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$project/scripts"
parent=$(realpath "${1:?새 parent 필요}")
test -d "$parent"
test ! -e "$parent/run"
cd "$project"
timeout -k 30 6900s python3 scripts/run_lumina_material_bf90.py \
  --source-commit "${2:?봉인 커밋 필요}" --out "$parent/run"
