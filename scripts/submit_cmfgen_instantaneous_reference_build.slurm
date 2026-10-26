#!/bin/bash -l
# 빌드 두 개의 총 한도. full CMFGEN 계산·수송·온도 해제는 실행하지 않는다.
# 인수: mktemp -d로 만든 새 parent, 봉인 source commit.
#SBATCH --job-name=lumina-native-decay-build
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
source_commit=${2:?봉인 commit 필요}
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
test -d "$parent"
test ! -e "$parent/source"
test ! -e "$parent/build"
mkdir "$parent/source"
git -C "$repository" archive "$source_commit" scripts tests | tar -x -C "$parent/source"
project="$parent/source"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project/scripts"
cd "$project"
python3 -m unittest discover -s tests -p test_cmfgen_instantaneous_reference_build.py -v
timeout -k 15 480s python3 scripts/build_cmfgen_instantaneous_reference.py \
  --native /gpfs/kjhan/cmfgen_src/cur_cmf --out "$parent/build"
