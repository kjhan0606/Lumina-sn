#!/bin/bash -l
# 기존 원소 충돌식을 블록 소비로 연결한다. 동일 commit 소스만 Slurm 안에서 추출.
# 목표: 실제 Si/S518행 불변과 용해/열량 보존. 전90점 공동수렴 주장은 하지 않는다.
#SBATCH --job-name=lumina-collision-stream
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
repository=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
parent=$(realpath "${1:?새 parent 필요}")
source_commit=${2:?검증할 정확한 commit 필요}
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test -d "$parent"
test ! -e "$parent/source"
test ! -e "$parent/build"
test ! -e "$parent/run"
git -C "$repository" cat-file -e "$source_commit^{commit}"
mkdir "$parent/source" "$parent/build"
git -C "$repository" archive "$source_commit" src scripts tests data | tar -x -C "$parent/source"
project="$parent/source"
# 버전 관리 밖의 봉인된 원자 기준표 한 개를 그대로 전달한다.
test "$(sha256sum "$repository/data/atomic/ionization_reference.csv" | cut -d ' ' -f 1)" = \
  660141ef55e5b9028f6a8f374a28a149bcf81e67f1f888fc0e157397aea4fa9a
test ! -e "$project/data/atomic/ionization_reference.csv"
cp -- "$repository/data/atomic/ionization_reference.csv" "$project/data/atomic/ionization_reference.csv"
test "$(sha256sum "$project/data/atomic/ionization_reference.csv" | cut -d ' ' -f 1)" = \
  660141ef55e5b9028f6a8f374a28a149bcf81e67f1f888fc0e157397aea4fa9a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$project/scripts"
printf 'source_commit=%s\n' "$source_commit"
cd "$project"
timeout -k 15 300s bash tests/run_bf_point_tests.sh "$project" "$parent/build" \
  > "$parent/build.stdout" 2> "$parent/build.stderr"
python3 -m pytest -q -p no:cacheprovider tests/test_samej_collision_replay.py \
  tests/test_lumina_reference_response.py tests/test_native_line_quadrature.py
timeout -k 15 240s python3 scripts/replay_lumina_samej_collisions.py \
  --source /gpfs/kjhan/cmfgen_runs/lumina_sameJ_nativeBB_20261027-y7f92T \
  --ci-baseline /gpfs/kjhan/cmfgen_runs/lumina_sameJ_CI_20261027-OTSjsr/run/result.json \
  --projection "$parent/build/model_projection" --with-dissolution --require-ci-heat \
  --out "$parent/run"
