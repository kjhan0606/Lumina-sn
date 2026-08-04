#!/bin/bash
#SBATCH --job-name=ionlock_lte
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Test: ion-lock + LTE-at-T_rad level pops.
# v2 154238 confirmed all 14 ions LOCKED but spectrum RMS 0.582 (vs CHAMP 0.319).
# Hypothesis: NLTE rate-solved level distribution within each ion is what hurts
# the spectrum, not the ion totals. Setting LUMINA_NLTE_FORCE_LTE_LEVELS=1 makes
# every shell take the Boltzmann@T_rad fallback while keeping ion-lock per-ion
# rescale -- isolates whether ion-lock alone (with LTE levels) is harmless.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_ionlock_lte"

N_PKT=200000
N_ITER=10

cell_dir="$ROOT/logs/ionlock_lte_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== Ion-lock + LTE-at-T_rad levels ==="
echo "Host: $(hostname) GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_ION_LOCK=1 \
LUMINA_NLTE_FORCE_LTE_LEVELS=1 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- ion totals at shell 0 (last NLTE iter) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- spectrum landings ---"
ls -la lumina_spectrum_formal.csv 2>&1 | tail -1
echo "Done: $(date)"
