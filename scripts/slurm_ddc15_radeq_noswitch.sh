#!/bin/bash
#SBATCH --job-name=ddc15_radeq_noswitch
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# H2 DE-RISK / ZONE-SWITCH ISOLATION TEST (2-agent design review, 2026-06-07).
# The free K-averaging test proved the blanket run's residual outer T_e spike is a
# DETERMINISTIC shell-to-shell BIAS (RMS stuck ~22% over K=1..6 averages, not 1/sqrt(K))
# with two structures: sh24-31 HIGH = the tau_rec/t_exp ZONE-SWITCH crossover kink,
# sh40-48 LOW = packet-starved inputs. Both physics+code agents say the cheapest
# decisive next test is to REMOVE the zone-switch entirely (0 code): set the freeze
# threshold so large that tau_rec >= thr*t_exp is NEVER true -> EVERY shell takes the
# SINGLE normal heating=cooling bisection (the inner branch), eliminating the closure
# switch, while everything else stays byte-identical to the blanket run (164032).
#   sh24-31 kink DISAPPEARS  -> the switch made it -> H2 (delete the mode/gate block)
#                               is confirmed as the cure for component (i).
#   sh24-31 kink PERSISTS    -> the bias is in the INPUTS (n_e/pops change character
#                               across the freeze boundary) -> input-smoothing (H3/H1)
#                               is mandatory regardless, learned before writing the
#                               deterministic EUV field.
# ONLY change vs slurm_ddc15_radeq_blanket.sh: LUMINA_RADEQ_HYBRID_TAUREC 1.0 -> 1e9.
# Binary = lumina_cuda_radeqhybrid (unchanged).

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
label="ddc15_radeq_noswitch"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d RADEQ: zone-switch REMOVED (TAUREC=1e9, single bisection all shells) ==="
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
    LUMINA_RADEQ_HYBRID_TAUREC=1e9 \
    LUMINA_RADEQ_HYBRID_MODE=blanket \
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
