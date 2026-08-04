#!/bin/bash
#SBATCH --job-name=ddc15_radeq
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-1
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# Approach-1 (iv) validation: does the radiative-equilibrium T_e solver
# (compute_radiative_equilibrium_te, plasma.c:3289 = "Task #20", binned-J
# photoheating vs adiabatic/recomb/line cooling, bisection) land on CMFGEN's
# outer T_e ~2540K instead of the band-aid T_e = 0.9*T_rad?
# The solver is already wired (cuda.cu:3205) but gated OFF by default, so every
# prior production run silently used the 0.9 ratio. This A/B just flips the gate.
#   cell 0 — OFF : T_e = 0.9*T_rad  (current production default; baseline)
#   cell 1 — ON  : LUMINA_RADEQ_TE=1 + LUMINA_RADEQ_DIAG=1 (term-by-term H/C dump)
# Both arms identical otherwise. Same DDC15 0.976d CMFGEN self-test reference and
# same physics knobs as slurm_ddc15_frozenin_ab.sh (macroatom, binnedj). No
# frozen-in here — RADEQ T_e is isolated on the steady-state plasma.

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
BIN="$ROOT/lumina_cuda"

LABELS=(off on)
RADEQ_VALS=(0 1)
tag=${LABELS[$SLURM_ARRAY_TASK_ID]}
radeq=${RADEQ_VALS[$SLURM_ARRAY_TASK_ID]}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
label="ddc15_radeq_${tag}"
work_root="$ROOT/logs/${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d RADEQ-Te A/B  cell=$tag (RADEQ_TE=$radeq) ==="
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
    LUMINA_RADEQ_TE=$radeq \
    LUMINA_RADEQ_DIAG=$radeq \
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
