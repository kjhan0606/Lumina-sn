#!/bin/bash
#SBATCH --job-name=ddc15_radeqB2
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Path B2: HYBRID radiative-equilibrium closure (3-agent verdict, 2026-06-06).
# Path B (job 163822) proved the signed collisional-net + bf-thermalization line
# term FIXES the photosphere (T_e[0]=4326 K vs CMFGEN 4434 K) but DIVERGES UPWARD
# in the optically-thin frozen outer zone (sh24-48 -> 6000-8400 K vs target 2505 K)
# because photoheating collapses (H_photo->1e-19) so the heating=cooling bisection
# has no anchor and the lagged-non-SE collisional net goes spuriously negative
# (heating). 3 physics agents (Hillier&Dessart 2012, Dessart&Hillier 2008,
# Kozma&Fransson 1992, Kerzendorf&Sim 2014) recommend the B2 HYBRID:
#   - inner/photosphere (tau_rec/t_exp < 1): keep the thermalization-anchored
#     heating=cooling bisection (validated, 2.4% at sh0);
#   - outer frozen zone  (tau_rec/t_exp >= 1): set T_e = ratio*T_rad, the trapped
#     dilute-Planck color temperature the frozen electrons couple to (TARDIS form).
# The switchover uses the SAME tau_rec/t_exp criterion that parameterizes the
# frozen-in IONIZATION -> one parameter-free physical statement, not two knobs.
#   LUMINA_RADEQ_HYBRID=1          -> enable the frozen-zone color-temp branch
#   LUMINA_RADEQ_HYBRID_TAUREC=1.0 -> freeze-out threshold (tau_rec/t_exp)
# All other knobs identical to path B (collisional-net line, escape OFF). Binary =
# lumina_cuda_radeqhybrid (multi-arch sm_80/86/90, carries the hybrid branch).

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
BIN="$ROOT/lumina_cuda_radeqhybrid"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
label="ddc15_radeqB2"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d RADEQ path (B2): hybrid collisional-net + frozen color-temp ==="
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
    LUMINA_RADEQ_HYBRID=1 \
    LUMINA_RADEQ_HYBRID_TAUREC=1.0 \
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
echo "--- [RADEQ] T_e/T_rad + hybrid summary lines ---"
grep -E "\[RADEQ\]" stdout.log | tail -14
echo ""
echo "--- [RADEQ-DIAG] term-by-term (shell0 + mid) ---"
grep -E "\[RADEQ-DIAG" stdout.log | tail -8
echo "Done: $(date)"
