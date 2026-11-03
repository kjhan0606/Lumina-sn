#!/bin/bash -l
# Same-state CMFGEN line-heat capture only; one attempt, no requeue.
# sbatch arguments: isolated attempt root, repository root.
#SBATCH --job-name=lumina-fixedj-heat
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
umask 027
module purge
module load gnu13/13.2.0
ulimit -s unlimited
ulimit -c 0

test "${SLURM_CPUS_PER_TASK:-}" = 16
test "${SLURM_MEM_PER_NODE:-}" = 32768
test -n "${SLURM_JOB_ID:-}"
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# The executable fails closed unless this reaches the srun task.
export LUMINA_FIXEDJ_HEAT_ONLY=1
test "$LUMINA_FIXEDJ_HEAT_ONLY" = 1

project=$(realpath "${2:?repository root required}")
test -f "$project/scripts/run_cmfgen_fixedj_heat_capture.py"
exec python3 -B "$project/scripts/run_cmfgen_fixedj_heat_capture.py" \
  --attempt-root "${1:?isolated attempt root required}"
