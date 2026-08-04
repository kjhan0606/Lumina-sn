#!/bin/bash
#SBATCH --job-name=toy06_stddeck
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --output=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.out
#SBATCH --error=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.err

set -euo pipefail

# Slurm executes a spool copy; BASH_SOURCE is not a repository locator (job 399901).
: "${REPO_ROOT:?submit with --export=ALL,REPO_ROOT=/absolute/path/to/Lumina-sn}"
CMFGEN_RUN="${CMFGEN_RUN:-/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4}"
NEW_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_standart"
LINKS_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_sivcaiv_links"

# SLURM_TMPDIR was absent in job 400018.  Use an explicit job-scoped GPFS path.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    SCRATCH_ROOT="$SLURM_TMPDIR/toy06_standart"
else
    : "${SLURM_JOB_ID:?SLURM_JOB_ID required when SLURM_TMPDIR is absent}"
    SCRATCH_ROOT="/gpfs/kjhan/lumina_runner2/tmp/toy06_standart_${SLURM_JOB_ID}"
fi
OFF_CONTROL="$SCRATCH_ROOT/r4_off_control"

cd "$REPO_ROOT"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# CPU data preparation only.  No Lumina binary, GPU, or model transport is run.
python3 scripts/build_toy06_standart_deck.py --output "$NEW_DECK"

mkdir -p "$SCRATCH_ROOT"
if [[ -e "$OFF_CONTROL" ]]; then
    echo "ERROR: refusing existing R4 OFF-control: $OFF_CONTROL" >&2
    exit 2
fi
cp -a "$LINKS_DECK" "$OFF_CONTROL"

# Intentionally last: pipefail makes verifier status the Slurm job result.
python3 scripts/verify_toy06_standart_deck.py \
    --deck "$NEW_DECK" \
    --cmf-run "$CMFGEN_RUN" \
    --r4-off-control "$OFF_CONTROL" \
    2>&1 | tee "$NEW_DECK/verification.log"
