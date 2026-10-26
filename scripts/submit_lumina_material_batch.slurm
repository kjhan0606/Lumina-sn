#!/bin/bash -l
# 전90점 율 생산자 직접시험. 대표 PHOT 경로만 대조하며 새 수송/native 없음.
#SBATCH --job-name=lumina-material-batch
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
test ! -e "$parent/build"
test ! -e "$parent/run"
mkdir "$parent/build"
cd "$project"
sha256sum src/bf_rate_jnu.c src/bf_rate_jnu.h scripts/lumina_model_material_rates.py \
  tests/test_model_material_rates.py > "$parent/build/source-before.sha256"
gcc -O2 -std=c11 -fopenmp -fPIC -shared -Wl,-z,defs -Isrc -o "$parent/build/point.so" \
  src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c -lm
timeout -k 15 90s python3 -m pytest -q -p no:cacheprovider tests/test_model_material_rates.py \
  tests/test_model_point_reference.py tests/test_model_point_dissolution.py \
  > "$parent/tests.stdout" 2> "$parent/tests.stderr"
timeout -k 15 480s python3 scripts/check_lumina_material_batch.py \
  --library "$parent/build/point.so" --out "$parent/run"
sha256sum -c "$parent/build/source-before.sha256" > "$parent/build/source-after-check.txt"
