#!/usr/bin/env bash
# 검증된70B의 새 장 한 번과 원래 step 한 번. 자동 후속은 없다.
#SBATCH --job-name=cmf-damp71B
#SBATCH --partition=a10,a40,a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/damping71B-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/damping71B-%j.err
set -euo pipefail
umask 027
module purge
module load gnu13
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ulimit -s unlimited 2>/dev/null || true
readonly DAMPING_RUNNER=${1:?sealed runner required}
readonly DAMPING_PLAN=${2:?sealed plan required}
readonly DAMPING_PLAN_SHA=${3:?plan sha256 required}
readonly DAMPING_SOURCE_ROOT=${4:?completed70B root required}
readonly DAMPING_SOURCE_OWNER=${5:?actual70B C1 required}
readonly DAMPING_SOURCE_SHA=${6:?completed70B runtime sha256 required}
(cd "$DAMPING_RUNNER" && sha256sum -c runner.sha256)
/home/kjhan/miniconda3/bin/python3 "$DAMPING_RUNNER/run_cmfgen_damping71_branch.py" \
 --plan "$DAMPING_PLAN" --plan-sha256 "$DAMPING_PLAN_SHA" \
 --source-root "$DAMPING_SOURCE_ROOT" --source-owner "$DAMPING_SOURCE_OWNER" \
 --source-runtime-sha256 "$DAMPING_SOURCE_SHA" \
 --run "/gpfs/kjhan/cmfgen_runs/toy06_19p48d_damping71B_${SLURM_JOB_ID}"
