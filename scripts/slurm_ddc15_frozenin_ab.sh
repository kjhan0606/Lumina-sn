#!/bin/bash
#SBATCH --job-name=ddc15_frozenin
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --array=0-1
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%A_%a.err

# Task #7 A/B: frozen-in recombination freeze-out (Chugai/Potashov single-epoch).
# Does LUMINA_FROZENIN=1 reproduce CMFGEN's outer <Z>~0.53 plateau that
# steady-state ionization cannot? Both arms identical except the frozen-in flag.
#   cell 0 — OFF : steady-state NLTE (= current production; must match baseline)
#   cell 1 — ON  : LUMINA_FROZENIN=1 + LUMINA_NLTE_PER_ION_RESCALE=1. The rescale
#                  gate pins each ion's NLTE level sum to the frozen-in per-ion
#                  total (plasma.c:4438) WITHOUT the transport-only plasma freeze
#                  that LUMINA_NLTE_ION_LOCK triggers (cuda.cu:3188) — so
#                  apply_frozenin_freezeout actually runs every iteration and the
#                  frozen totals survive the NLTE solve.
# Same DDC15 0.976d CMFGEN self-test reference + same physics knobs as
# slurm_ddc15_initial.sh (macroatom, binnedj). Binary = freshly built lumina_cuda.

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
BIN="$ROOT/lumina_cuda"   # fresh build that carries the frozen-in code

LABELS=(off on)
FROZEN_VALS=(0 1)
RESCALE_VALS=(0 1)
tag=${LABELS[$SLURM_ARRAY_TASK_ID]}
frozen=${FROZEN_VALS[$SLURM_ARRAY_TASK_ID]}
irescale=${RESCALE_VALS[$SLURM_ARRAY_TASK_ID]}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
label="ddc15_frozenin_${tag}"
work_root="$ROOT/logs/${label}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d FROZEN-IN A/B  cell=$tag (FROZENIN=$frozen PER_ION_RESCALE=$irescale) ==="
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
    LUMINA_NLTE_PER_ION_RESCALE=$irescale \
    LUMINA_FROZENIN=$frozen \
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
echo "--- [FROZENIN] markers ---"
grep -E "\[FROZENIN\]" stdout.log | head -12
echo ""
echo "--- final n_e[0] / n_e[last] ---"
grep -E "n_e\[0\]=" stdout.log | tail -4
echo ""
echo "--- T_inner / T_rad trajectory ---"
grep -E 'T_inner:|T_inner final|T_rad' stdout.log | tail -10
echo "Done: $(date)"
