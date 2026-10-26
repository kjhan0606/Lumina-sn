#!/bin/bash -l
# 완료한 BF90과 직접시험된 동일 C 바이너리만 재사용한다. 적분/수송/물질 갱신 없음.
#SBATCH --job-name=lumina-bf90-consumer
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
set -euo pipefail
umask 027
ulimit -c 0
module purge
module load gnu13/13.2.0
parent=$(realpath "${1:?새 parent 필요}")
producer=$(realpath "${2:?완료한 C 직접시험 parent 필요}")
result=${3:?완료 BF90 result.json 필요}
test -d "$parent"
test ! -e "$parent/run"
test -f "$producer/run/result.json"
test -f "$producer/build/source-after-check.txt"
test -f "$result"
cd "$producer/source"
sha256sum -c "$producer/build/source-before.sha256" > "$parent/source-check.txt"
cd "$producer/build"
sha256sum -c binary.sha256 > "$parent/binary-check.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$producer/source/scripts"
timeout -k 15 540s python3 "$producer/source/scripts/consume_lumina_material_bf90.py" \
  --source "$result" --projection "$producer/build/model_projection" --out "$parent/run"
