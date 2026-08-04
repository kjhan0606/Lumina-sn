#!/bin/bash
#SBATCH --job-name=nomllock_smoke
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# (2) Remove Mihalas-Lucy soft lock on NLTE pair total.
#   LUMINA_NLTE_NO_ML_LOCK=1 → conservation row b[N-1] = n_element (mass conservation)
# Prediction: iron-peak is mostly II+III at SN T_e, so n_element ~ n(II)+n(III) and
# this should leave W err near baseline. If W err shifts significantly, phi_neb total
# (not the II/III split) is part of the cooling-feedback loop.
# Baseline: 155358 cefix W err 547.00%.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_nomllock"

N_PKT=50000
N_ITER=8

cell_dir="$ROOT/logs/nomllock_smoke_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== NLTE no-ML-lock smoke probe ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary:    $BIN"
echo "n_pkt=$N_PKT  n_iter=$N_ITER  NLTE_START=5"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_NO_ML_LOCK=1 \
LUMINA_NLTE_RATE_DUMP=1 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- knob banner ---"
grep -E 'no-ML-lock' stdout.log | head -2
echo "--- Fe II / Fe III shell 0 last iter ---"
if [ -f nlte_rate_balance.csv ]; then
    head -1 nlte_rate_balance.csv
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $3==0' | tail -2
    echo "--- Co II/III shell 0 ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==27 && $3==0' | tail -2
    echo "--- Ni II/III shell 0 ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==28 && $3==0' | tail -2
    echo "--- Si II/III shell 0 ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==14 && $3==0' | tail -2
    echo "--- Ca II/III shell 0 ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==20 && $3==0' | tail -2
fi
echo "Done: $(date)"
