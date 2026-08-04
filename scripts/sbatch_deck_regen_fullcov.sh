#!/bin/bash
#SBATCH --job-name=deck_fullcov
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# sbatch copies this file into /var/spool/slurmd/scripts, so BASH_SOURCE cannot
# locate the repo under slurm.  Honour an exported REPO_ROOT when present.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
NEW_DECK="$REPO_ROOT/data/tardis_reference_toy06_19p48d_sivcaiv_fullcov"
cd "$REPO_ROOT"

# CPU-only job: this script requests no GRES and hides any inherited devices.
unset CUDA_VISIBLE_DEVICES
export CMFGEN_FULL_LEVELS=1
export CMFGEN_SUPER_LEVELS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python3 scripts/deck_regen_fullcov_driver.py
python3 scripts/finalize_cmfgen_ref_npy.py "$NEW_DECK" 50
python3 scripts/build_cmfgen_coldata_all.py --ref-dir "$NEW_DECK" --write

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

# The verifier is intentionally last and exits nonzero on any gate failure.
python3 scripts/verify_deck_regen_fullcov.py --new "$NEW_DECK" \
  2>&1 | tee "$NEW_DECK/verification.log"
