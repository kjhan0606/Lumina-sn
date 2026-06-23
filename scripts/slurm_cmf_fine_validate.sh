#!/bin/bash
#SBATCH --job-name=cmf_fine_validate
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# P7 PRODUCER validation (gate II-2 / production I-2,I-3): run the pure-CMFGEN
# driver a few iters with LUMINA_CMF_LINERES_JBAR=1 + LUMINA_CMF_FINE_DIAG=1 so
# cmfgen_fine_jbar() exercises the fine-grid deposit/solve/extract on the REAL
# DDC15 line list. Narrow window (default 3000-3200 A) keeps cmf_solve_J cheap
# for a fast correctness check: the [cmf_fine] tie-back ratio must be ~1.0 and
# mean Jbar/B must be physical (O(0.1-2)). Widen window once correctness holds.

module load cuda/13.0.2 2>/dev/null || true

N_PKT=${N_PKT:-1000}
N_ITER=${N_ITER:-5}
SPEC_MODE=${SPEC_MODE:-spectrum}
CONSUME=${CONSUME:-0}   # II-3 A/B knob: 0=producer-only(baseline pops), 1=deterministic consumer
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/${DDC15_REF:-tardis_reference_ddc15_0p976d}"
BIN="$ROOT/lumina_cuda"

work_root="$ROOT/logs/cmf_fine_validate_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"
REF_DIR="$work_root/ref"; mkdir -p "$REF_DIR"
for f in "$REF"/*; do ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"; done

echo "=== P7 PRODUCER fine-grid Jbar_l validation (DDC15 0.976d) ==="
echo "Host: $(hostname)  Time: $(date)  window=${LAMLO:-3000}-${LAMHI:-3200} A  N_ITER=$N_ITER"
ls -l "$BIN"

env LUMINA_PURE_CMFGEN=1 \
    LUMINA_PURE_CMFGEN_ITER=$N_ITER \
    LUMINA_CMFGEN_ALI_ITER=${LUMINA_CMFGEN_ALI_ITER:-8} \
    LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=$REF/cmfgen_sigma_bf.bin \
    LUMINA_DYNAMIC_TRANSPROB=1 \
    LUMINA_NLTE_SKIP_Z=14 \
    LUMINA_NLTE_START_ITER=2 \
    LUMINA_NLTE_FLOOR_REG=1 \
    LUMINA_NLTE_INV_CEIL=1e4 \
    LUMINA_RADEQ_TE=1 \
    LUMINA_RADEQ_COOL_ESCAPE=0 \
    LUMINA_RADEQ_COOL_NONNEG=0 \
    LUMINA_RADEQ_COOL_NLTE_ONLY=1 \
    LUMINA_LINE_INTERACTION=macroatom \
    LUMINA_TAU_BY_ION=1 \
    LUMINA_DIFFUSE_INNER_BC=1 \
    LUMINA_ENERGY_BUDGET=1 \
    LUMINA_CMF_LINERES_JBAR=1 \
    LUMINA_CMF_LINERES_CONSUME=$CONSUME \
    LUMINA_CMF_FINE_DIAG=1 \
    LUMINA_CMF_FINE_LAMLO=${LAMLO:-3000} \
    LUMINA_CMF_FINE_LAMHI=${LAMHI:-3200} \
    LUMINA_CMF_FINE_VDOP=${VDOP:-1e6} \
    LUMINA_CMF_FINE_PPD=${PPD:-12} \
    LUMINA_CMF_FINE_ALI=${FINE_ALI:-16} \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$SPEC_MODE" nlte \
    > stdout.log 2> stderr.log
rc=$?
echo "  exit=$rc  work_root=$work_root"
echo ""
echo "--- [cmf_fine] producer diagnostics (mean Jbar/B + in-window S_l/B per iter) ---"
grep -E "\[cmf_fine\]" stderr.log stdout.log
echo "--- [cmf_consume] consumer activation ---"
grep -E "\[cmf_consume\]" stderr.log stdout.log
echo ""
echo "--- [CMFGEN] driver lines ---"
grep -E "\[CMFGEN\]" stdout.log | tail -8
echo ""
echo "--- any NaN/segfault/error ---"
grep -iE "nan|segfault|error|abort|alloc failed" stderr.log stdout.log | head
