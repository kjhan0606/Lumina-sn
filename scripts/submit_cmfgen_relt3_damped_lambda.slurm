#!/bin/bash -l
# One fixed-T LAMBDA damping experiment; inputs are staged into a new root.
#SBATCH --job-name=cmfgen-lam101
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:0
#SBATCH --time=03:00:00
#SBATCH --no-requeue
set -euo pipefail
umask 027
ulimit -c 0

seed=${1:?iteration-55 seed root required}
setup_source=${2:?matching atom-link source required}
run=${3:?new isolated run root required}
executable=${4:?sealed CMFGEN binary required}
repository=${5:?repository root required}

module purge
module load gnu13/13.2.0
module list
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repository/scripts"
cd "$repository"
python3 scripts/run_cmfgen_relt3_damped_lambda.py \
  --seed "$seed" --setup-source "$setup_source" --run "$run" \
  --executable "$executable"
