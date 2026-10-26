#!/bin/bash -l
# 원70/J70 고정장 물질 단회: 실제ν 지수 갱신을 고친 private native build만 사용.
# sbatch 인수: mktemp -d로 만든 새 parent. 원본·구형 출력은 보존한다.
#SBATCH --job-name=lumina-fixedj-exp
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
module purge
module load gnu13/13.2.0
ulimit -s unlimited
ulimit -c 0
test "${SLURM_CPUS_PER_TASK:-}" = 16
test "${SLURM_MEM_PER_NODE:-}" = 32768
parent=$(realpath "${1:?새 parent 필요}")
test -d "$parent"
test ! -e "$parent/build"
test ! -e "$parent/run"
project=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
base_build=/gpfs/kjhan/cmfgen_runs/lumina_se_witness_build_v3_20261020-44LxrD/build
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project/scripts"
cd "$project"
timeout -k 30 300s python3 scripts/prepare_cmfgen_fixedj_exponent.py \
  --parent "$base_build" --out "$parent/build" > "$parent/build.stdout" 2> "$parent/build.stderr"
timeout -k 30 6900s python3 scripts/run_cmfgen_verbose_se70.py \
  --original /gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702 \
  --source70 /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429038/material \
  --reference71 /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429062/material \
  --native /gpfs/kjhan/cmfgen_src/cur_cmf --build "$base_build" \
  --corrected-build "$parent/build" --root "$parent/run"
