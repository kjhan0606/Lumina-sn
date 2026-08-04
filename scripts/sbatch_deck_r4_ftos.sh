#!/bin/bash
#SBATCH --job-name=deck_r4_ftos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.out
#SBATCH --error=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.err

set -euo pipefail

# Slurm copies this script below /var/spool/slurmd/scripts.  The exported
# REPO_ROOT is authoritative; BASH_SOURCE is only a local-shell fallback.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NEW_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_sivcaiv_ftos"
LINKS_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_sivcaiv_links"
CMFGEN_RUN="${CMFGEN_RUN:-/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4}"
# R4_OFF_DIR is the explicit operator escape hatch and takes precedence on
# clusters where SLURM_TMPDIR is not provisioned.
if [[ -n "${R4_OFF_DIR:-}" ]]; then
    OFF_DECK="$R4_OFF_DIR"
elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
    OFF_DECK="$SLURM_TMPDIR/r4_ftos_offcontrol"
else
    echo "ERROR: neither R4_OFF_DIR nor SLURM_TMPDIR is available" >&2
    echo "       resubmit with --export=ALL,R4_OFF_DIR=/scratch/..." >&2
    exit 2
fi
cd "$REPO_ROOT"

# CPU-only data bakes: request no GRES and hide inherited accelerator devices.
unset CUDA_VISIBLE_DEVICES
export CMFGEN_FULL_LEVELS=1
export CMFGEN_SUPER_LEVELS=1
export CMFGEN_LINKS="${CMFGEN_LINKS:-$CMFGEN_RUN/atomic_links.txt}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python3 scripts/r4_ftos_fixture.py
python3 scripts/audit_r4_ftos.py --links "$CMFGEN_LINKS" --deck "$LINKS_DECK"
CMFGEN_LINK_FTOS=1 python3 scripts/r4_ftos_wiring_fixture.py

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

# Exact gate-OFF control.  It is node-local and is retained until Slurm cleans
# SLURM_TMPDIR, so the final read-only verifier can compare every byte.
unset CMFGEN_LINK_FTOS
export R4_OFF_DIR="$OFF_DECK"
python3 scripts/deck_regen_r4_offcontrol_driver.py
bake_sidecars "$OFF_DECK"
python3 scripts/verify_deck_r1_vintage.py \
  --new "$OFF_DECK" --cmf-run "$CMFGEN_RUN" \
  2>&1 | tee "$OFF_DECK/verification.log"

# R4 output.  The driver refuses to overwrite an existing target.
export CMFGEN_LINK_FTOS=1
python3 scripts/deck_regen_r4_ftos_driver.py
bake_sidecars "$NEW_DECK"

# Read-only validator is the final command.  pipefail makes its exit code the
# Slurm job result; a failure is reported without adjustment or rewrite.
python3 scripts/verify_deck_r4_ftos.py \
  --new "$NEW_DECK" \
  --links "$CMFGEN_LINKS" \
  --cmf-run "$CMFGEN_RUN" \
  --links-deck "$LINKS_DECK" \
  --off-control "$OFF_DECK" \
  2>&1 | tee "$NEW_DECK/verification.log"
