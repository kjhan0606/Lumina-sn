#!/bin/bash
# 원70/J70→71 native SE의 충돌·연속항 진단을 격리 복제본에서 단회 재실행.
# --chdir/--output은 제출 시 새 root의 상위 디렉터리로 지정한다.
#SBATCH --job-name=lumina-se70-phase
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue

source /etc/profile
module purge
module load gnu13
set -euo pipefail
umask 027
test "${SLURM_CPUS_PER_TASK:-}" = 16
test "${SLURM_MEM_PER_NODE:-}" = 32768
test -n "${LUMINA_PHASE_PARENT:-}"
test -d "$LUMINA_PHASE_PARENT"
test ! -e "$LUMINA_PHASE_PARENT/run"
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts
ulimit -s unlimited

timeout -k 30 6900s python3 /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_cmfgen_verbose_se70.py \
  --original /gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702 \
  --source70 /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429038/material \
  --reference71 /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_429062/material \
  --native /gpfs/kjhan/cmfgen_src/cur_cmf \
  --build /gpfs/kjhan/cmfgen_runs/lumina_se_witness_build_v3_20261020-44LxrD/build \
  --root "$LUMINA_PHASE_PARENT/run"
