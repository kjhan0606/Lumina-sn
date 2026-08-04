#!/bin/bash
#SBATCH --job-name=toy06_cmfcomp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --output=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.out
#SBATCH --error=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.err

set -euo pipefail

# Slurm runs a spool copy, so a path derived from BASH_SOURCE is invalid.
: "${REPO_ROOT:?submit with --export=ALL,REPO_ROOT=/absolute/path/to/Lumina-sn}"
CMFGEN_RUN="${CMFGEN_RUN:-/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4}"
NEW_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_cmfgencomp"
LINKS_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_sivcaiv_links"

# job 400018 showed SLURM_TMPDIR is not guaranteed.  The fallback is explicit,
# job-scoped GPFS storage, never $HOME and never an unresolved broad path.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    SCRATCH_ROOT="$SLURM_TMPDIR/toy06_cmfgencomp"
else
    : "${SLURM_JOB_ID:?SLURM_JOB_ID is required when SLURM_TMPDIR is absent}"
    SCRATCH_ROOT="/gpfs/kjhan/lumina_runner2/tmp/toy06_cmfgencomp_${SLURM_JOB_ID}"
fi
OFF_CONTROL="$SCRATCH_ROOT/r4_off_control"

cd "$REPO_ROOT"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# CPU data preparation only: no model binary, GPU, or deck submission occurs.
python3 scripts/build_toy06_cmfgencomp_deck.py \
    --cmf-run "$CMFGEN_RUN" \
    --output "$NEW_DECK"

mkdir -p "$SCRATCH_ROOT"
if [[ -e "$OFF_CONTROL" ]]; then
    echo "ERROR: refusing existing R4 OFF-control path: $OFF_CONTROL" >&2
    exit 2
fi
cp -a "$LINKS_DECK" "$OFF_CONTROL"

# Intentionally last.  pipefail propagates the verifier status as the Slurm
# job result; a failed physical gate is reported without adjustment or retry.
python3 scripts/verify_toy06_cmfgencomp_deck.py \
    --deck "$NEW_DECK" \
    --cmf-run "$CMFGEN_RUN" \
    --r4-off-control "$OFF_CONTROL" \
    2>&1 | tee "$NEW_DECK/verification.log"
