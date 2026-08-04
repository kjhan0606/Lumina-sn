#!/bin/bash
#SBATCH --job-name=ddc15_nefix
#SBATCH --partition=a100,a40,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# n_e-fix validation + coupled-Newton profiler (LUMINA_CN_PROF=1).
# Overlapping partitions (a100,a40,h100) so slurm takes the first free GPU.
# Same collapsed config as scripts/run_ada_lstar_diag.sh (JNU_PHOTOION=0,
# LAMBDA_STAR=1, JNU_LSTAR=1). Multi-arch lumina_cuda covers sm_80/86/90.

module load cuda/13.0.2 2>/dev/null || true

N_PKT=${N_PKT:-120000}
N_ITER=${N_ITER:-6}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF="$ROOT/data/tardis_reference_ddc15_0p976d"
BIN="$ROOT/lumina_cuda"

work_root="$ROOT/logs/ddc15_nefix_prof_${SLURM_JOB_ID}"
mkdir -p "$work_root"; cd "$work_root"
REF_DIR="$work_root/ref"; mkdir -p "$REF_DIR"
for f in "$REF"/*; do ln -sf "$(readlink -f "$f")" "$REF_DIR/$(basename "$f")"; done

echo "=== DDC15 0.976d n_e-fix + CN profiler ==="
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "OMP=$OMP_NUM_THREADS  Binary: $BIN  Time: $(date)"

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
    LUMINA_JNU_SED_DUMP=0 \
    LUMINA_DIFFUSE_INNER_BC=1 \
    LUMINA_ENERGY_BUDGET=1 \
    LUMINA_CN_PROF=1 \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log
rc=$?
echo "exit=$rc  Done: $(date)"
echo "--- negative n_e check ---"
awk -F, 'NR>1 && $4+0<0{n++} END{print "negative_ne_rows="n+0}' lumina_plasma_state.csv 2>/dev/null
echo "--- CN profiler ---"
grep "CN-PROF" stdout.log
echo "--- coupled-newton convergence ---"
grep "COUPLED-NEWTON" stdout.log | tail -3
echo "WORKDIR=$work_root"
