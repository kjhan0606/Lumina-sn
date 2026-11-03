#!/usr/bin/env bash
#SBATCH --job-name=lumina-gridheat
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${LUMINA_REPO:?set LUMINA_REPO}"
: "${LUMINA_INPUT_ROOT:?set LUMINA_INPUT_ROOT}"
: "${LUMINA_PHOT_LIBRARY:?set LUMINA_PHOT_LIBRARY}"
: "${LUMINA_KERNEL_LIBRARY:?set LUMINA_KERNEL_LIBRARY}"
: "${LUMINA_OUTPUT:?set a new LUMINA_OUTPUT path}"
: "${LUMINA_EXPECTED_COMMIT:?set LUMINA_EXPECTED_COMMIT}"

cd "$LUMINA_REPO"
[[ "$(git rev-parse HEAD)" == "$LUMINA_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain -- scripts/compare_lumina_native_grid_heat.py)" ]]
[[ ! -e "$LUMINA_OUTPUT" ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == 16 ]]

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun --cpu-bind=cores python scripts/compare_lumina_native_grid_heat.py \
  --input-root "$LUMINA_INPUT_ROOT" \
  --phot-library "$LUMINA_PHOT_LIBRARY" \
  --kernel-library "$LUMINA_KERNEL_LIBRARY" \
  --out "$LUMINA_OUTPUT"
