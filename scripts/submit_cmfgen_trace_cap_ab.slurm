#!/usr/bin/env bash
# 같은63/J64의 대조 시험. 수송0/native1, MAX_LAM만2→1e6.
#SBATCH --job-name=cmf-trace-cap-ab
#SBATCH --partition=a10,a40,a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/trace-cap-ab-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/trace-cap-ab-%j.err
set -euo pipefail
umask 027
module purge
module load gnu13
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ulimit -s unlimited 2>/dev/null || true
readonly TRACE_AB_RUNNER=${1:?sealed runner is required}
readonly TRACE_AB_RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_trace_ab_${SLURM_JOB_ID}
[[ ${SLURM_CPUS_PER_TASK:-} == 16 && ${SLURM_MEM_PER_NODE:-} == 32768 && ! -e "$TRACE_AB_RUN" ]] || exit 23
(cd "$TRACE_AB_RUNNER" && sha256sum -c runner.sha256)
/home/kjhan/miniconda3/bin/python3 "$TRACE_AB_RUNNER/run_cmfgen_trace_cap_ab.py" \
 --reference /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_428886 \
 --run "$TRACE_AB_RUN" --native /gpfs/kjhan/cmfgen_src/cur_cmf
