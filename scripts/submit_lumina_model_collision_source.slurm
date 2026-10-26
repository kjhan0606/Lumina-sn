#!/bin/bash -l
# 공통장 물질응답 묶음 내부의 전원소 Ω(T) 원자료 시험. 새 물리 해법 실행 없음.
#SBATCH --job-name=lumina-model-collisions
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
test ! -e "$parent/result.json"
cd "$project"
python3 -m pytest -q -p no:cacheprovider tests/test_model_collision_source.py
timeout -k 15 540s python3 scripts/check_lumina_model_collision_source.py \
 --levels /gpfs/kjhan/cmfgen_runs/lumina_material_cells_20261020-PjlGds/cells/atomic/levels.csv \
 --material /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429038/material \
 --out "$parent/result.json" "${@:2}"
