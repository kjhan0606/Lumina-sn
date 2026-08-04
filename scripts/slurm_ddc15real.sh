#!/bin/bash
#SBATCH --job-name=ddc15real
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Run on the REAL DDC15 composition+density (data/tardis_reference_ddc15real,
# built by scripts/build_ddc15_real_composition.py from the Blondin+2015 public
# hydro file, iron-peak isotopes decayed 0.976d->17.76d). Replaces the
# hand-rolled ddc15_profile_v3 approximation that starved the Si layer and used
# Ni-as-Co. The ref already carries geometry/density/abundances/config, so this
# script does NOT regenerate structure at runtime — it runs the ref as-is.
#
# Self-consistent thermal solve (no LUMINA_FIXED_TRAD_PROFILE): the real
# composition changes the opacity/thermal structure, so freezing to the old
# scatter profile would be inconsistent.
#
#   sbatch --export=ALL,LINE_INT=scatter   scripts/slurm_ddc15real.sh
#   sbatch --export=ALL,LINE_INT=macroatom scripts/slurm_ddc15real.sh

LINE_INT=${LINE_INT:-macroatom}
BINNEDJ=1
DIFFUSE_BC=1
N_PKT=200000
N_ITER=10
RUN_TAG="ddc15real_${LINE_INT}"

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/tardis_reference_ddc15real"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if echo "$GPU_NAME" | grep -qiE "H100|H200"; then
    BIN="$ROOT/lumina_cuda_binnedj_h100_macap"
else
    BIN="$ROOT/lumina_cuda_binnedj_a40_macap"
fi

label="paperDDC15real_2002bo_${RUN_TAG}"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

# Symlink the ENTIRE ref (structure + atomic data) as-is; no regeneration.
REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15-REAL composition run  LINE_INT=$LINE_INT  DIFFUSE_BC=$DIFFUSE_BC  N_PKT=$N_PKT N_ITER=$N_ITER ==="
echo "Host: $(hostname)  GPU: $GPU_NAME"
echo "Binary: $BIN  Ref: $REF  Time: $(date)"
python3 -c "import pandas as pd; d=pd.read_csv('$REF/config.json'.replace('.json','.json')) if False else None; import json; c=json.load(open('$REF/config.json')); print(f'  t_exp={c[\"time_explosion_s\"]:.0f}s  L={c[\"luminosity_inner_erg_s\"]:.3e}  T_inner_init={c[\"T_inner_K\"]}K  n_shells={c[\"n_shells\"]}')"

env LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=$REF/cmfgen_sigma_bf.bin \
    LUMINA_DYNAMIC_TRANSPROB=1 \
    LUMINA_NLTE_SKIP_Z=14 \
    LUMINA_NLTE_START_ITER=2 \
    LUMINA_NLTE_FLOOR_REG=1 \
    LUMINA_NLTE_INV_CEIL=1e4 \
    LUMINA_LINE_INTERACTION=$LINE_INT \
    LUMINA_MAX_INTERACTIONS=200 \
    LUMINA_BINNED_J_ESTIMATOR=$BINNEDJ \
    LUMINA_TAU_BY_ION=1 \
    LUMINA_NLTE_LEVEL_DUMP=1 \
    LUMINA_DIFFUSE_INNER_BC=$DIFFUSE_BC \
    LUMINA_ENERGY_BUDGET=1 \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo ""
echo "--- transprob mode + interaction cap ---"
grep -E "Transition probs:|\[CAP\]|truncat" stdout.log | head -10
echo ""
echo "--- energy budget per iteration ---"
grep -E "E-BUDGET" stdout.log
echo ""
echo "--- T_inner trajectory ---"
grep -E 'T_inner:|T_inner final' stdout.log | tail -14
echo ""
echo "--- ion fractions (Si/Fe sanity) ---"
grep -E "ION-FRAC|Si II|Fe III" stdout.log | head -20
echo "Done: $(date)"
