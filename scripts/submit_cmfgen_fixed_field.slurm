#!/usr/bin/env bash
# 완성 J60 고정 · native 고정T SE/ne 네 번 갱신. RT/열/스펙트럼 합격 아님.
#SBATCH --job-name=cmf-fixed-J60
#SBATCH --partition=a10,a40,a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/fixed-field-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/fixed-field-%j.err
set -euo pipefail
umask 027
module purge
module load gnu13
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ulimit -s unlimited 2>/dev/null || true
readonly FIXED_FIELD_RUNNER_DIR=${1:?sealed runner is required}
readonly FIXED_FIELD_RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_fixed_field_${SLURM_JOB_ID}
[[ ${SLURM_CPUS_PER_TASK:-} == 16 && ${SLURM_MEM_PER_NODE:-} == 131072 && ! -e "$FIXED_FIELD_RUN" ]] || exit 23
(cd "$FIXED_FIELD_RUNNER_DIR" && sha256sum -c runner.sha256)
/home/kjhan/miniconda3/bin/python3 "$FIXED_FIELD_RUNNER_DIR/run_cmfgen_fixed_field.py" \
 --seed /gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702 \
 --field /gpfs/kjhan/cmfgen_runs/toy06_19p48d_formal_primary_428782 \
 --run "$FIXED_FIELD_RUN" --native /gpfs/kjhan/cmfgen_src/cur_cmf
