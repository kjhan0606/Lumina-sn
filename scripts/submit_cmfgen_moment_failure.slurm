#!/bin/bash -l
# 승인된74→78 실패 상태 포획 단회. NUM_ITS16은 이전 경로 보존용이며78에서 종료.
# NEW_PARENT COMMIT SOURCE70 SEED_RUN ION_REF BUILD BUILD_SHA NATIVE CAPTURE CAPTURE_SHA FAILED_INPUTS FAILED_SHA [TARGET_FREQ TARGET_OUTER CAPTURE_PARENT CAPTURE_NATIVE]
#SBATCH --job-name=lumina-mom78-capture
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
#SBATCH --chdir=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
set -euo pipefail
umask 027
[[ $# -eq 12 || $# -eq 16 ]]
ulimit -s unlimited
ulimit -c 0
module purge
module load gnu13/13.2.0
repository=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
parent=$(realpath "${1:?새 parent 필요}")
source_commit=${2:?봉인 commit 필요}
source70=$(realpath "${3:?원70 물질 필요}")
seed_run=$(realpath "${4:?완료74 run 필요}")
ion_reference=$(realpath "${5:?74 독립 추출 결과 필요}")
build=$(realpath "${6:?교정 full build 필요}")
build_sha=${7:?교정 build SHA 필요}
native=$(realpath "${8:?실제 native 경로 필요}")
capture=$(realpath -m "${9:?포획 build 필요}")
capture_sha=${10:?포획 build SHA 필요}
failure_inputs=$(realpath "${11:?기존 실패 입력 기록 필요}")
failure_sha=${12:?기존 실패 입력 SHA 필요}
target_frequency=${13:-}
target_outer=${14:-}
capture_parent=${15:-}
capture_native=${16:-}
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
for digest in "$build_sha" "$failure_sha"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
done
target_flags=()
if [[ -n "$target_frequency" || -n "$target_outer" || -n "$capture_parent" || -n "$capture_native" ]]; then
  [[ -n "$target_frequency" && -n "$target_outer" && -n "$capture_parent" && -n "$capture_native" ]]
  [[ "$capture_sha" = BUILD ]]
  [[ "$target_frequency" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]
  [[ "$target_outer" =~ ^[1-9][0-9]*$ ]]
  (( 10#$target_outer <= 2147483647 ))
  capture_parent=$(realpath "$capture_parent")
  capture_native=$(realpath "$capture_native")
  test -d "$capture_parent" && test -d "$capture_native" && test ! -e "$capture"
else
  [[ "$capture_sha" =~ ^[0-9a-f]{64}$ ]]
fi
[[ "${SLURM_CPUS_PER_TASK:?CPU16 필요}" = 16 ]]
[[ "${SLURM_MEM_PER_NODE:?128GiB 필요}" = 131072 ]]
test -d "$parent" && test ! -e "$parent/source" && test ! -e "$parent/run-parent"
mkdir "$parent/source" "$parent/run-parent"
git -C "$repository" archive "$source_commit" scripts tests | tar -x -C "$parent/source"
project="$parent/source"
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project/scripts"
cd "$project"
python3 -m unittest tests.test_cmfgen_instantaneous_fixedt -v
python3 -m pytest tests/test_cmfgen_failure_capture.py -q
if [[ -n "$target_frequency" ]]; then
  python3 scripts/prepare_cmfgen_mom_capture.py --parent "$capture_parent" \
    --native "$capture_native" --out "$capture" --entry-only \
    --target-frequency "$target_frequency" --target-outer "$target_outer" >/dev/null
  capture_sha=$(sha256sum "$capture/capture-build.json" | awk '{print $1}')
  target_flags+=(--capture-target-frequency "$target_frequency" --capture-target-outer "$target_outer")
fi
timeout 14000s python3 scripts/continue_cmfgen_instantaneous_fixedt.py \
  --original /gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702 \
  --source70 "$source70" --seed-run "$seed_run" --native "$native" \
  --ion-reference "$ion_reference" --build "$build" --build-report-sha256 "$build_sha" \
  --out "$parent/run-parent/run" --source-commit "$source_commit" \
  --memory-gib 128 --seconds 13680 --max-iterations 16 \
  --capture-build "$capture" --capture-build-sha256 "$capture_sha" \
  --capture-failure-inputs "$failure_inputs" --capture-failure-inputs-sha256 "$failure_sha" \
  "${target_flags[@]}"
