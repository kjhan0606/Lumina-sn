#!/bin/bash
#SBATCH --job-name=ddc15_radeqesc
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Follow-up to the RADEQ A/B (job 163554): the RADEQ T_e solver decoupled T_e
# from T_rad and fixed the T_inner luminosity runaway, but OVER-COOLED the
# photosphere (T_e[0]~2932K vs CMFGEN 4374K) because radeq_line_cool summed the
# full ~2.5M bb-line census with lagged dilute-Boltzmann(T_rad) pops that are NOT
# in SE with the solved T_e -> spurious one-way coolant (C_coll swamped H_photo
# ~20x at shell 0). Physics-verified faithful fix = radiative-escape bb cooling
# (LUMINA_RADEQ_COOL_ESCAPE=1): C_bb = sum beta_esc(tau)*A_ul*n_up*dE, beta_esc on
# the RADIATIVE term (NOT a beta-weighted collisional difference, which the physics
# review rejected as double-discounting trapping). Manifestly >=0 so the nonneg
# floor is moot; thick lines (tau>>1 -> beta_esc->1/tau) are correctly trapped and
# stop cooling. NLTE-tracked n_up (LUMINA_RADEQ_COOL_NLTE_ONLY=1). No double-count:
# macro-atom is purely radiative here (LUMINA_KPACKET off).
# Binary = lumina_cuda_radeqesc (carries the fix). Same DDC15 0.976d ref + knobs.

LINE_INT=${LINE_INT:-macroatom}
BINNEDJ=1
DIFFUSE_BC=1
N_PKT=${N_PKT:-200000}
N_ITER=${N_ITER:-10}
MAX_INT=${MAX_INT:-200}
SPEC_MODE=${SPEC_MODE:-spectrum}

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/${DDC15_REF:-tardis_reference_ddc15_0p976d}"
BIN="$ROOT/lumina_cuda_radeqesc"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
label="ddc15_radeqesc"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d RADEQ-Te + radiative-escape bb cooling ==="
echo "Host: $(hostname)  GPU: $GPU_NAME"
echo "Binary: $BIN  Ref: $REF  Time: $(date)"
echo "ls -l binary:"; ls -l "$BIN"

env LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=$REF/cmfgen_sigma_bf.bin \
    LUMINA_DYNAMIC_TRANSPROB=1 \
    LUMINA_NLTE_SKIP_Z=14 \
    LUMINA_NLTE_START_ITER=2 \
    LUMINA_NLTE_FLOOR_REG=1 \
    LUMINA_NLTE_INV_CEIL=1e4 \
    LUMINA_RADEQ_TE=1 \
    LUMINA_RADEQ_DIAG=1 \
    LUMINA_RADEQ_COOL_NLTE_ONLY=1 \
    LUMINA_RADEQ_COOL_ESCAPE=1 \
    LUMINA_LINE_INTERACTION=$LINE_INT \
    LUMINA_MAX_INTERACTIONS=$MAX_INT \
    LUMINA_BINNED_J_ESTIMATOR=$BINNEDJ \
    LUMINA_TAU_BY_ION=1 \
    LUMINA_NLTE_LEVEL_DUMP=1 \
    LUMINA_DIFFUSE_INNER_BC=$DIFFUSE_BC \
    LUMINA_ENERGY_BUDGET=1 \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$SPEC_MODE" nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo ""
echo "--- [RADEQ] T_e/T_rad summary lines ---"
grep -E "\[RADEQ\]" stdout.log | tail -12
echo ""
echo "--- [RADEQ-DIAG] term-by-term (shell0 + mid) ---"
grep -E "\[RADEQ-DIAG" stdout.log | tail -8
echo ""
echo "--- final T_rad / T_e trajectory ---"
grep -E 'T_inner:|T_inner final|T_e/T_rad' stdout.log | tail -10
echo "Done: $(date)"
