#!/bin/bash
#SBATCH --job-name=ddc15_a3lstar
#SBATCH --partition=a100
#SBATCH --nodelist=syn102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# PATH-A / A3 incr-1 (2026-06-08): diagonal-Λ* dilution-preserving radiation
# response inside the coupled Newton (LUMINA_COUPLED_LAMBDA_STAR=1). J_ν follows
# W·B(T_e) where the gas is thick to its own bf continuum (Λ*=1−e^{−τ_bf}), so the
# bf heating tracks the trial T_e instead of staying frozen at B(T_field≈4434K).
# IMPORTANT per the 2-agent design review: VALIDATE Λ* ALONE — keep J_ν photoion
# OFF (LUMINA_COUPLED_JNU_PHOTOION=0) so the charge closure stays φ_neb. Acceptance:
# beat the φ_neb baseline (job 164504) inner0-12 mean +0.543 dex, and the
# photosphere T_e[0] must NOT collapse to the 1000K floor (the JNU death-spiral).

LINE_INT=${LINE_INT:-macroatom}
BINNEDJ=${BINNEDJ:-1}
DIFFUSE_BC=1
N_PKT=${N_PKT:-200000}
N_ITER=${N_ITER:-6}
MAX_INT=${MAX_INT:-200}
SPEC_MODE=${SPEC_MODE:-spectrum}

export OMP_NUM_THREADS=8
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/${DDC15_REF:-tardis_reference_ddc15_0p976d}"
BIN="$ROOT/lumina_cuda"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
label="ddc15_a3lstar"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d A3 incr-1 diagonal-Λ* radiation response smoke ==="
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
    LUMINA_RADEQ_COOL_ESCAPE=${LUMINA_RADEQ_COOL_ESCAPE:-0} \
    LUMINA_RADEQ_COOL_NONNEG=${LUMINA_RADEQ_COOL_NONNEG:-0} \
    LUMINA_RADEQ_LINE_RESPOND=${LUMINA_RADEQ_LINE_RESPOND:-0} \
    LUMINA_RADEQ_COOL_NLTE_ONLY=1 \
    LUMINA_RADEQ_HYBRID=1 \
    LUMINA_RADEQ_HYBRID_TAUREC=1.0 \
    LUMINA_RADEQ_HYBRID_MODE=blanket \
    LUMINA_FROZENIN=1 \
    LUMINA_NLTE_PER_ION_RESCALE=1 \
    LUMINA_DIFFUSION_INNER_BC=1 \
    LUMINA_COUPLED_NEWTON=1 \
    LUMINA_COUPLED_NEWTON_OMP=1 \
    LUMINA_COUPLED_TDEP=${LUMINA_COUPLED_TDEP:-1} \
    LUMINA_COUPLED_JNU_PHOTOION=${LUMINA_COUPLED_JNU_PHOTOION:-0} \
    LUMINA_COUPLED_LAMBDA_STAR=${LUMINA_COUPLED_LAMBDA_STAR:-1} \
    LUMINA_COUPLED_JNU_LSTAR=${LUMINA_COUPLED_JNU_LSTAR:-0} \
    LUMINA_COUPLED_LAMBDA_TAUSCALE=${LUMINA_COUPLED_LAMBDA_TAUSCALE:-1.0} \
    LUMINA_COUPLED_NEWTON_TRACE=${LUMINA_COUPLED_NEWTON_TRACE:-0} \
    LUMINA_NLTE_SKIP_DEAD=1 \
    LUMINA_LINE_INTERACTION=$LINE_INT \
    LUMINA_MAX_INTERACTIONS=$MAX_INT \
    LUMINA_BINNED_J_ESTIMATOR=$BINNEDJ \
    LUMINA_TAU_BY_ION=1 \
    LUMINA_NLTE_LEVEL_DUMP=1 \
    LUMINA_ION_POP_DUMP=${LUMINA_ION_POP_DUMP:-0} \
    LUMINA_JNU_SED_DUMP=${LUMINA_JNU_SED_DUMP:-0} \
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
echo "--- COUPLED-NEWTON convergence + RADEQ T_e + LSTAR ---"
grep -E "COUPLED-NEWTON|\[RADEQ\] T_e/T_rad|\[LSTAR" stdout.log
echo ""
echo "--- plasma_state head ---"
head -6 lumina_plasma_state.csv 2>/dev/null
echo "Done: $(date)"
