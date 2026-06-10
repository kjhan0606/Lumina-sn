#!/bin/bash
#SBATCH --job-name=ddc15_pure_cmfgen
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# PURE-CMFGEN PARALLEL PATH smoke + first physics (2026-06-07).
# Deterministic comoving-frame tangent-ray short-characteristics formal solver
# (src/lumina_cmfgen.c) replaces the Monte-Carlo radiation field entirely:
#   - expansion (Sobolev-binned) line opacity subsumes the comoving d/dnu term
#   - chi = e-scatter + bf + ff + line-expansion; coherent e-scatter closed by
#     diagonal ALI; core rays emit B(T_inner), outer BC I^- = 0
#   - J_nu(shell,bin) written into NLTEConfig.J_nu on the existing 1000-bin grid
#   - ALL downstream solvers (RADEQ T_e, plasma ionization, bf, NLTE) reused.
# Uses the GPU `lumina_cuda` binary (env-gated LUMINA_PURE_CMFGEN=1) so the reused
# NLTE step runs on the GPU (cuBLAS GEMM) — the deterministic radiation field is
# CPU but the per-iter NLTE solve dominates cost and is impractical on CPU.
# This is a FIRST-LIGHT smoke: verify no NaN/segfault + sanity of T_e(v) vs CMFGEN
# (do NOT trust absolute numbers until the J->S thick and W*B thin limits are
# checked). N_PKT is ignored (MC bypassed).

module load cuda/13.0.2 2>/dev/null || true

N_PKT=${N_PKT:-1000}
N_ITER=${N_ITER:-8}
SPEC_MODE=${SPEC_MODE:-spectrum}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/${DDC15_REF:-tardis_reference_ddc15_0p976d}"
BIN="$ROOT/lumina_cuda"

label="ddc15_pure_cmfgen"
work_root="$ROOT/logs/${label}_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"

REF_DIR="$work_root/ref"
mkdir -p "$REF_DIR"
for f in "$REF"/*; do
    ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"
done

echo "=== DDC15 0.976d PURE-CMFGEN deterministic radiation (MC bypassed) ==="
echo "Host: $(hostname)  Time: $(date)"
echo "Binary: $BIN  Ref: $REF"
echo "ls -l binary:"; ls -l "$BIN"

env LUMINA_PURE_CMFGEN=1 \
    LUMINA_PURE_CMFGEN_ITER=$N_ITER \
    LUMINA_CMFGEN_ALI_ITER=${LUMINA_CMFGEN_ALI_ITER:-8} \
    LUMINA_CMFGEN_CELLDIAG=${LUMINA_CMFGEN_CELLDIAG:-} \
    LUMINA_BF_OPACITY=1 \
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
    LUMINA_LINE_INTERACTION=macroatom \
    LUMINA_TAU_BY_ION=1 \
    LUMINA_DIFFUSE_INNER_BC=1 \
    LUMINA_ENERGY_BUDGET=1 \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$SPEC_MODE" nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo ""
echo "--- [CMFGEN] driver lines ---"
grep -E "\[CMFGEN\]" stdout.log | tail -20
echo ""
echo "--- plasma_state head ---"
head -6 lumina_plasma_state.csv 2>/dev/null
echo "Done: $(date)"
