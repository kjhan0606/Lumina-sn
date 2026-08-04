#!/bin/bash
#SBATCH --job-name=ddc15_frozenin
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Task #7 — FROZEN-IN recombination freeze-out, first C run (2026-06-07).
# Byte-identical to the blanket B2-hybrid run (job 164032, which matched the
# outer T_e ~2550 K plateau) EXCEPT two added knobs:
#   LUMINA_FROZENIN=1               -> apply the time-dependent freeze-out cascade
#                                      (Milne-RR alpha + parameter-free t_0 +
#                                       homologous recomb ODE) to shells with t_0<t_exp
#   LUMINA_NLTE_PER_ION_RESCALE=1   -> pin each ion's level sum to the frozen total
#                                      so the downstream NLTE solve does NOT re-solve
#                                      ion balance and undo the frozen partition
# Acceptance test = outer n_e vs CMFGEN (current steady-state is ~1000x too low;
# the frozen-in plateau should recover the ~0.5 <Z> fossil). Uses the freshly
# built lumina_cuda (carries frozenin + hybrid + rescale, sm_86).

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

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
label="ddc15_frozenin"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d FROZEN-IN freeze-out (blanket B2-hybrid + frozen-in cascade) ==="
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
    LUMINA_RADEQ_HYBRID_MODE=blanket \
    LUMINA_FROZENIN=1 \
    LUMINA_NLTE_PER_ION_RESCALE=1 \
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
echo "--- [FROZENIN] freeze-out lines ---"
grep -E "\[FROZENIN\]" stdout.log | tail -14
echo ""
echo "--- plasma_state head ---"
head -6 lumina_plasma_state.csv 2>/dev/null
echo "Done: $(date)"
