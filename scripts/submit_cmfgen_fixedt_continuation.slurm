#!/bin/bash -l
# 미수렴 CMFGEN fixed-T 완료 record에서만 새 격리 복제본으로 추가 반복.
# 인수: NEW_PARENT SEALED_COMMIT SOURCE70 SEED_RUN ION_REFERENCE BUILD BUILD_SHA NATIVE [ITERATIONS=16].
# 승인 자원: CPU16/OMP16/128GiB/GPU0/최대12시간/재시도0.
#SBATCH --job-name=lumina-cmfgen-extend74
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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
source70=$(realpath "${3:?원70 material 필요}")
seed_run=$(realpath "${4:?완료74 run 필요}")
ion_reference=$(realpath "${5:?SCRTEMP/POP 독립 대조 추출기 결과 필요}")
build=$(realpath "${6:?봉인 full build 필요}")
build_sha=${7:?봉인 build SHA 필요}
native=$(realpath "${8:?CMFGEN native 경로 필요}")
iterations=${9:-16}
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
[[ "$build_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$iterations" =~ ^([1-9]|1[0-9]|2[0-9]|3[0-2])$ ]]
[[ "${SLURM_MEM_PER_NODE:?128GiB 할당 필요}" = 131072 ]]
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
timeout 43000s python3 scripts/continue_cmfgen_instantaneous_fixedt.py \
  --original /gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702 \
  --source70 "$source70" --seed-run "$seed_run" --native "$native" \
  --ion-reference "$ion_reference" \
  --build "$build" --build-report-sha256 "$build_sha" \
  --out "$parent/run-parent/run" --source-commit "$source_commit" \
  --memory-gib 128 --seconds 42600 --max-iterations "$iterations"
