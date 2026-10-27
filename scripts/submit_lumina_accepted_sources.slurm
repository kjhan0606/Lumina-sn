#!/bin/bash -l
# 새 물질의 전주파수 계수·경계1회. 실제 수송/물질응답은 묶음 내 후속 작업이다.
#SBATCH --job-name=lumina-accepted-sources
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
repository=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
parent=$(realpath "${1:?새 parent 필요}")
source_commit=${2:?봉인 commit 필요}
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test -d "$parent"
test ! -e "$parent/source"
test ! -e "$parent/run"
mkdir "$parent/source"
git -C "$repository" archive "$source_commit" src scripts data | tar -x -C "$parent/source"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$parent/source/scripts"
cd "$parent/source"
timeout -k 30 6900s python3 scripts/run_lumina_accepted_sources.py \
    --out "$parent/run" --source-commit "$source_commit"
