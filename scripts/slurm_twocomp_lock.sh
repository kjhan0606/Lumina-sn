#!/bin/bash
#SBATCH --job-name=twocomp_lock
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# P3: Two-component ion-lock — W-threshold shell-conditional LTE-at-T_e.
# Inner shells (W > W_thresh) use phi = phi_LTE_at_Te for masked transitions.
# Default IONMASK=1 (II→III lock): less inner-shell ionization → less Fe/Co/Ni III
# at UV-blanketing radii. Outer shells continue with M-L nebular hybrid.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_zlock"

N_PKT=50000
N_ITER=8

W_THRESH=${LK_WTHRESH:-1.5}
ZMASK=${LK_ZMASK:-21,22,23,24,25,26,27,28}
IONMASK=${LK_IONMASK:-1}

cell_dir="$ROOT/logs/twocomp_lock_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== 2-component ion-lock probe ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary:    $BIN"
echo "n_pkt=$N_PKT  n_iter=$N_ITER  NLTE_START=5"
echo "2-comp lock: W_THRESH=$W_THRESH  ZMASK=$ZMASK  IONMASK=$IONMASK"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_RATE_DUMP=1 \
LUMINA_LOCK_W_THRESH=$W_THRESH \
LUMINA_LOCK_ZMASK=$ZMASK \
LUMINA_LOCK_IONMASK=$IONMASK \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- 2-comp-lock banner ---"
grep -E '\[2-comp-lock\]' stdout.log | head -1
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- Fe II / Fe III shell 0 last iter ---"
if [ -f nlte_rate_balance.csv ]; then
    head -1 nlte_rate_balance.csv
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $3==0' | tail -2
    echo "--- Fe II / Fe III shell 15 (W~1) ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $3==15' | tail -2
    echo "--- Co II/III shell 0 ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==27 && $3==0' | tail -2
fi
echo "Done: $(date)"
