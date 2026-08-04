#!/bin/bash
#SBATCH --job-name=deck_atomic_quarantine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.out
#SBATCH --error=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.err

set -euo pipefail

# Slurm executes a spool copy, so an exported REPO_ROOT is authoritative.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CMFGEN_RUN="${CMFGEN_RUN:-/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4}"
CMFGEN_LINKS="${CMFGEN_LINKS:-$CMFGEN_RUN/atomic_links.txt}"
SOURCE_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
NEW_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_sivcaiv_active"
LINKS_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_sivcaiv_links"

# job 400018 showed that SLURM_TMPDIR is not universal.  R4_OFF_DIR is the
# explicit operator escape hatch and takes precedence when supplied.
if [[ -n "${R4_OFF_DIR:-}" ]]; then
    OFF_DECK="$R4_OFF_DIR"
elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
    OFF_DECK="$SLURM_TMPDIR/r4_ftos_offcontrol"
else
    echo "ERROR: neither R4_OFF_DIR nor SLURM_TMPDIR is available" >&2
    echo "       resubmit with --export=ALL,R4_OFF_DIR=/scratch/..." >&2
    exit 2
fi
IDENTITY_REPORT_DIR="${ATOMIC_IDENTITY_REPORT_DIR:-${OFF_DECK}.identity_reports}"

cd "$REPO_ROOT"
unset CUDA_VISIBLE_DEVICES
export CMFGEN_RUN CMFGEN_LINKS R4_OFF_DIR="$OFF_DECK"
export CMFGEN_FULL_LEVELS=1
export CMFGEN_SUPER_LEVELS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

bake_sidecars() {
    local deck="$1"
    python3 scripts/finalize_cmfgen_ref_npy.py "$deck" 50
    python3 scripts/build_cmfgen_coldata_all.py \
      --ref-dir "$deck" \
      --source-manifest "$deck/atomic_vintage_manifest.csv" \
      --write
    python3 - "$deck" <<'PY'
import importlib.util
from pathlib import Path
import sys

out = Path(sys.argv[1])
script = Path("scripts/bake_level_multiplicity.py").resolve()
spec = importlib.util.spec_from_file_location("bake_level_multiplicity", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.REF_DIR = out
module.LEVELS_CSV = out / "levels.csv"
module.OUT_CSV = out / "level_multiplicity.csv"
module.main()
PY
    python3 scripts/build_ma_radrecomb_target.py "$deck"
}

# Required CPU-only fixture proof.  No deck/model/GPU work occurs here.
python3 scripts/atomic_quarantine_fixture.py

# Preserve every existing R1 and R4 gate against the immutable R4 source deck.
# The OFF control is ephemeral and never aliases or writes _links/_ftos.
unset CMFGEN_LINK_FTOS
python3 scripts/deck_regen_r4_offcontrol_driver.py
bake_sidecars "$OFF_DECK"
python3 scripts/verify_deck_r4_ftos.py \
  --new "$SOURCE_DECK" \
  --links "$CMFGEN_LINKS" \
  --cmf-run "$CMFGEN_RUN" \
  --links-deck "$LINKS_DECK" \
  --off-control "$OFF_DECK"

# Build the new active view.  The driver refuses to overwrite any target.
export CMFGEN_LINK_FTOS=1
python3 scripts/deck_quarantine_driver.py
bake_sidecars "$NEW_DECK"
python3 scripts/seal_atomic_quarantine.py --deck "$NEW_DECK"

# Must remain the final command.  pipefail propagates the validator's exit code
# as the Slurm job result; tee only records that final read-only verdict.
python3 scripts/verify_atomic_quarantine_identity.py \
  --deck "$NEW_DECK" \
  --cmf-run "$CMFGEN_RUN" \
  --links "$CMFGEN_LINKS" \
  --report-dir "$IDENTITY_REPORT_DIR" \
  2>&1 | tee "$NEW_DECK/verification.log"
