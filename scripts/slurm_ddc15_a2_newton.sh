#!/bin/bash
#SBATCH --job-name=ddc15_a2newton
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# PATH-A / A1 smoke (2026-06-08): replace the TARDIS T_inner feedback controller
# with the fixed-L diffusion inner BC (LUMINA_DIFFUSION_INNER_BC=1).
# Byte-identical to scripts/slurm_ddc15_frozenin.sh EXCEPT the one added knob.
# Acceptance: T_inner must NOT ping-pong (4430->91163->...->5643 K) when full
# NLTE engages at iter 2; it should hold ~4430 K (Stefan-Boltzmann L_req value),
# and inner sh0 n_e/ionization should stop over-shooting (ratio was 9.7-19x).

LINE_INT=${LINE_INT:-macroatom}
BINNEDJ=1
DIFFUSE_BC=1
N_PKT=${N_PKT:-200000}
N_ITER=${N_ITER:-6}
MAX_INT=${MAX_INT:-200}
SPEC_MODE=${SPEC_MODE:-spectrum}

# NLTE rate-matrix assembly is 96.6% of NLTE wall time and is OpenMP-parallel
# over shells; OMP=1 serialized all 49 (236 s/CE-iter). cpus-per-task=4 -> 4x.
export OMP_NUM_THREADS=4
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/${DDC15_REF:-tardis_reference_ddc15_0p976d}"
BIN="$ROOT/lumina_cuda"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
label="ddc15_a2newton"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d A2 coupled-Newton {n_e,T_e} smoke ==="
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
    LUMINA_DIFFUSION_INNER_BC=1 \
    LUMINA_COUPLED_NEWTON=1 \
    LUMINA_COUPLED_TDEP=${LUMINA_COUPLED_TDEP:-1} \
    LUMINA_NLTE_SKIP_DEAD=1 \
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
echo "--- T_inner trajectory (A1 should hold ~4430 K) ---"
grep -E "T_inner:" stdout.log
echo ""
echo "--- COUPLED-NEWTON convergence + RADEQ T_e ---"
grep -E "COUPLED-NEWTON|\[RADEQ\] T_e/T_rad" stdout.log
echo ""
echo "--- plasma_state head ---"
head -6 lumina_plasma_state.csv 2>/dev/null
echo "Done: $(date)"
