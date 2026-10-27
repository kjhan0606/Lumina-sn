#!/bin/bash -l
# corrected native의 자체장 fixed-T 응답 최대4회. 자동재실행·T-release 없음.
# 인수: NEW_PARENT SEALED_COMMIT BUILD REPORT_SHA [MEMORY_GIB=32] [SECONDS=6600].
# 승인된128GiB는 마지막 인수128 및 sbatch --mem=128G를 함께 명시한다.
#SBATCH --job-name=lumina-native-instant-fixedt
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
set -euo pipefail
umask 027
ulimit -s unlimited
ulimit -c 0
module purge
module load gnu13/13.2.0
repository=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
parent=$(realpath "${1:?새 parent 필요}")
source_commit=${2:?봉인 commit 필요}
build=$(realpath "${3:?새 초기화 교정 build 필요}")
build_report_sha=${4:?봉인 build 보고서 SHA256 필요}
native_memory_gib=${5:-32}
native_seconds=${6:-6600}
[[ "$native_memory_gib" = 32 || "$native_memory_gib" = 128 ]]
[[ "$native_seconds" = 6600 || "$native_seconds" = 42600 ]]
[[ "${SLURM_MEM_PER_NODE:?명시 메모리 할당 필요}" = "$((native_memory_gib * 1024))" ]]
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
[[ "$build_report_sha" =~ ^[0-9a-f]{64}$ ]]
test "$(sha256sum "$build/result.json" | cut -d ' ' -f 1)" = "$build_report_sha"
test -d "$parent"
test ! -e "$parent/source"
test ! -e "$parent/run"
mkdir "$parent/source"
git -C "$repository" archive "$source_commit" scripts tests | tar -x -C "$parent/source"
project="$parent/source"
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project/scripts"
cd "$project"
python3 -m unittest discover -s tests -p test_cmfgen_instantaneous_fixedt.py -v
timeout -k 30 "$((native_seconds + 300))s" python3 scripts/run_cmfgen_instantaneous_fixedt.py \
  --original /gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702 \
  --source70 /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429038/material \
  --native /gpfs/kjhan/cmfgen_src/cur_cmf \
  --build "$build" --build-report-sha256 "$build_report_sha" \
  --out "$parent/run" --source-commit "$source_commit" --memory-gib "$native_memory_gib" \
  --seconds "$native_seconds"
