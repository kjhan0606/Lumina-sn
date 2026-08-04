#!/bin/bash
#SBATCH --job-name=ddc15_radeqB
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Path (B): faithful electron-energy / integral radiative-equilibrium closure.
# 5-agent literature verdict (Kubat-Puls-Pauldrach 1999, Lucy 2003, Hillier &
# Dessart 2012) established TWO errors in the escape-form RADEQ run:
#   (1) the bb LINE term in the ELECTRON energy balance must be the COLLISIONAL
#       NET  Sum (n_lo C_lu - n_up C_ul) dE, NOT the radiative escape
#       beta_esc*A_ul*n_up*dE (radiative rates exchange energy with the RADIATION
#       FIELD, not the electron pool; escape over-counts by B_lu*J/C_lu >> 1 at a
#       radiatively-pumped photosphere -> the floor collapse).
#   (2) the photosphere is pinned by THERMALIZATION: at tau>>1, J_nu -> B_nu(T_e),
#       and the bound-free heating/cooling detailed-balance pair (already in
#       radeq: H_photo = Sum n_lev * int sigma_bf f_above J_nu dnu  vs
#       C_rec = Sum n_lev * int sigma_bf f_above B_nu(T_e) dnu) is the J->B(T_e)
#       restoring force. With the escape coolant removed, this pair drives T_e ~
#       T_rad at the photosphere.
# This run flips the gas-energy line term back to the SIGNED collisional net over
# SE-satisfying NLTE-tracked levels (no rebuild needed; the code path exists):
#   LUMINA_RADEQ_COOL_ESCAPE=0   -> use collisional-difference, NOT escape
#   LUMINA_RADEQ_COOL_NONNEG=0   -> SIGNED net (no per-line floor; the floor was
#                                   what rectified lagged residuals into a phantom
#                                   one-way coolant in the earlier coolfix run)
#   LUMINA_RADEQ_COOL_NLTE_ONLY=1-> SE pops only (untracked lagged-Boltzmann lines
#                                   are not in SE with the solved T_e)
# Binary = lumina_cuda_radeqesc (multi-arch sm_80/86/90). Same DDC15 0.976d ref.

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
label="ddc15_radeqB"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d RADEQ path (B): collisional-net line + bf thermalization ==="
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
    LUMINA_RADEQ_COOL_ESCAPE=0 \
    LUMINA_RADEQ_COOL_NONNEG=0 \
    LUMINA_RADEQ_COOL_NLTE_ONLY=1 \
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
echo "Done: $(date)"
