#!/bin/bash
#SBATCH --job-name=deck_r1_links
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.out
#SBATCH --error=/gpfs/kjhan/lumina_runner2/slurm/%x_%j.err

set -euo pipefail

# Slurm copies this script below /var/spool/slurmd/scripts.  An exported
# REPO_ROOT is therefore authoritative; BASH_SOURCE is only a local-shell
# fallback.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NEW_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_sivcaiv_links"
CMFGEN_RUN="${CMFGEN_RUN:-/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4}"
cd "$REPO_ROOT"

# CPU-only data bake: request no GRES and hide inherited accelerator devices.
unset CUDA_VISIBLE_DEVICES
export CMFGEN_FULL_LEVELS=1
export CMFGEN_SUPER_LEVELS=1
export CMFGEN_LINKS="${CMFGEN_LINKS:-$CMFGEN_RUN/atomic_links.txt}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python3 scripts/deck_regen_r1_vintage_driver.py
python3 scripts/finalize_cmfgen_ref_npy.py "$NEW_DECK" 50
python3 scripts/build_cmfgen_coldata_all.py \
  --ref-dir "$NEW_DECK" \
  --source-manifest "$NEW_DECK/atomic_vintage_manifest.csv" \
  --write

python3 - "$NEW_DECK" <<'PY'
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

python3 scripts/build_ma_radrecomb_target.py "$NEW_DECK"

# Read-only verifier is last.  set -o pipefail preserves its nonzero status and
# the pipeline stops without attempting to tune or rewrite a failed deck.
python3 scripts/verify_deck_r1_vintage.py \
  --new "$NEW_DECK" --cmf-run "$CMFGEN_RUN" \
  2>&1 | tee "$NEW_DECK/verification.log"
