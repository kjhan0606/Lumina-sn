#!/bin/bash
# Manual (non-slurm) run on LagEunha RTX 5000 Ada (sm_89) — B3-1 + SED/ion-pop dumps.
# Mirrors scripts/slurm_ddc15_a3_lstar.sh env block. Run via ssh.
set -e
module load cuda/12.8.1 2>/dev/null || true

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/tardis_reference_ddc15_0p976d"
BIN="$ROOT/lumina_cuda_ada"
N_PKT=${N_PKT:-120000}
N_ITER=${N_ITER:-6}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
unset OMP_PLACES

work_root="$ROOT/logs/ddc15_ada_diag_$$"
mkdir -p "$work_root"; cd "$work_root"
REF_DIR="$work_root/ref"; mkdir -p "$REF_DIR"
for f in "$REF"/*; do ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"; done

echo "=== DDC15 0.976d B3-1 Ada diag (manual) ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Binary: $BIN  workdir: $work_root  Time: $(date)"

env LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF=$REF/cmfgen_sigma_bf.bin \
    LUMINA_DYNAMIC_TRANSPROB=1 \
    LUMINA_NLTE_SKIP_Z=14 \
    LUMINA_NLTE_START_ITER=2 \
    LUMINA_NLTE_FLOOR_REG=1 \
    LUMINA_NLTE_INV_CEIL=1e4 \
    LUMINA_RADEQ_TE=1 \
    LUMINA_RADEQ_DIAG=1 \
    LUMINA_RADEQ_COOL_NLTE_ONLY=1 \
    LUMINA_RADEQ_HYBRID=1 \
    LUMINA_RADEQ_HYBRID_TAUREC=1.0 \
    LUMINA_RADEQ_HYBRID_MODE=blanket \
    LUMINA_FROZENIN=1 \
    LUMINA_NLTE_PER_ION_RESCALE=1 \
    LUMINA_DIFFUSION_INNER_BC=1 \
    LUMINA_COUPLED_NEWTON=1 \
    LUMINA_COUPLED_NEWTON_OMP=1 \
    LUMINA_COUPLED_TDEP=1 \
    LUMINA_COUPLED_JNU_PHOTOION=0 \
    LUMINA_COUPLED_LAMBDA_STAR=1 \
    LUMINA_COUPLED_JNU_LSTAR=1 \
    LUMINA_COUPLED_LAMBDA_TAUSCALE=1.0 \
    LUMINA_NLTE_SKIP_DEAD=1 \
    LUMINA_LINE_INTERACTION=macroatom \
    LUMINA_MAX_INTERACTIONS=200 \
    LUMINA_BINNED_J_ESTIMATOR=1 \
    LUMINA_TAU_BY_ION=1 \
    LUMINA_NLTE_LEVEL_DUMP=0 \
    LUMINA_ION_POP_DUMP=1 \
    LUMINA_JNU_SED_DUMP=1 \
    LUMINA_DIFFUSE_INNER_BC=1 \
    LUMINA_ENERGY_BUDGET=1 \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log
rc=$?
echo "exit=$rc  Done: $(date)"
echo "WORKDIR=$work_root"
ls -l lumina_jnu_sed.csv lumina_ion_pops.csv lumina_plasma_state.csv 2>/dev/null
