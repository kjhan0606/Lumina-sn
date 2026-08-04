#!/bin/bash
#SBATCH --job-name=dr_floor_aggro
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_e%a_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_e%a_%j.err
#SBATCH --array=4,5,6,7

# Heavy.2 / aggressive DR floor probe.
# 1e-9..1e-12 sweep was null (W err 363-372% across 4 decades).
# This probe pushes floor to physically *unreasonable* values to ask:
# IF recomb magnitude could be arbitrary, would the gap close? If no knee
# even at 1e-4 cm^3/s -> bug is structural (matrix solve / nebular fallback)
# not magnitude.
#
# Also enables LUMINA_NLTE_RATE_DUMP=1 -> per-(Z,ion,shell) sum_R_bf,
# sum_R_rec rows in nlte_rate_balance.csv for direct diagnostic.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_dr_floor"
EXPN=$SLURM_ARRAY_TASK_ID
FLOOR=$(awk -v e=$EXPN 'BEGIN{printf "%.0e", 10^(-e)}')

N_PKT=200000
N_ITER=10
MODE=spectrum

cell_dir="$ROOT/logs/dr_floor_aggro_e${EXPN}_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== DR aggressive floor probe, alpha=$FLOOR cm^3/s (e=$EXPN) ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time:    $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_RATE_DUMP=1 \
LUMINA_DR_FLOOR_CMS=$FLOOR \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error\|Mean \|T_rad error\|' stdout.log | tail -4
echo "--- rate-balance head + tail ---"
head -3 nlte_rate_balance.csv 2>/dev/null
echo "..."
tail -5 nlte_rate_balance.csv 2>/dev/null
echo "--- ion totals (last NLTE iter, shell 0) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- spectrum landings ---"
ls -la nlte_rate_balance.csv lumina_spectrum_formal.csv 2>&1 | tail -3
echo "Done:   $(date)"
