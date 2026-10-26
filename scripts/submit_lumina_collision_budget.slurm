#!/bin/bash -l
# 완료된 동일J/충돌 witness의 읽기 전용 분해. 물리 실행/계수 적분 없음.
#SBATCH --job-name=lumina-collision-budget
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
module purge
module load gnu13/13.2.0
project=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$project/scripts"
parent=$(realpath "${1:?새 parent 필요}")
test -d "$parent"
test ! -e "$parent/result.json"
cd "$project"
python3 -m pytest -q -p no:cacheprovider tests/test_samej_collision_budget.py
timeout -k 15 500s python3 scripts/analyze_lumina_samej_collision_budget.py \
  --response /gpfs/kjhan/cmfgen_runs/lumina_sameJ_dissolution_rows_20261027-QajPr2/run/result.json \
  --out "$parent/result.json"
