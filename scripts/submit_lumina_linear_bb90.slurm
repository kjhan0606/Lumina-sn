#!/bin/bash -l
# 원 J70의 명시적 선형 표현만 적분. 수송/물질 갱신 없음.
#SBATCH --job-name=lumina-linear-bb90
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
parent=$(realpath "${1:?새 parent 필요}")
commit=${2:?봉인 커밋 필요}
test -d "$parent"
test ! -e "$parent/source"
mkdir "$parent/source" "$parent/build"
git -C "$project" archive "$commit" scripts src tests | tar -x -C "$parent/source"
cd "$parent/source"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$parent/source/scripts"
timeout -k 10 180s python3 -m unittest discover -s tests -p test_model_line_linear_jbar.py -v \
  > "$parent/unit.stdout" 2> "$parent/unit.stderr"
gcc -O2 -std=c11 -Wall -Wextra -Werror -shared -fPIC src/model_line_linear_jbar.c -lm -o "$parent/build/lines.so"
timeout -k 15 510s python3 scripts/run_lumina_linear_bb90.py --library "$parent/build/lines.so" \
  --out "$parent/run" --source-commit "$commit"
