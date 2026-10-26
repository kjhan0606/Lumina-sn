#!/bin/bash -l
# 동일 원70/J70의 직접 CI1 대조와 준위용해 충돌항의 실제 연결 시험.
#SBATCH --job-name=lumina-sameJ-dissolution
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
timeout -k 15 300s bash tests/run_bf_point_tests.sh "$project" "$parent/build" \
  > "$parent/build.stdout" 2> "$parent/build.stderr"
python3 -m pytest -q -p no:cacheprovider tests/test_samej_collision_replay.py \
  tests/test_lumina_reference_response.py tests/test_native_line_quadrature.py
timeout -k 15 240s python3 scripts/replay_lumina_samej_collisions.py \
  --source /gpfs/kjhan/cmfgen_runs/lumina_sameJ_nativeBB_20261027-y7f92T \
  --ci-baseline /gpfs/kjhan/cmfgen_runs/lumina_sameJ_CI_20261027-OTSjsr/run/result.json \
  --projection "$parent/build/model_projection" --with-dissolution --out "$parent/run" \
  "${@:2}"
