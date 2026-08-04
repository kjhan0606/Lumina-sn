#!/bin/bash
#SBATCH --job-name=dr_boost
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_b%a_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_b%a_%j.err
#SBATCH --array=1,10,100,1000

# Heavy.2 / Task #142: empirical DR magnitude sweep.
# Each array task scales DR by SLURM_ARRAY_TASK_ID (boost = 1, 10, 100, 1000).
# Goal: locate knee in W err vs boost — tells us how badly Mazzotta LS
# underestimates near-threshold DR resonances at SN T_e.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_dr_boost"
BOOST=$SLURM_ARRAY_TASK_ID

N_PKT=200000
N_ITER=10
MODE=spectrum

cell_dir="$ROOT/logs/dr_boost_b${BOOST}_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== DR boost sweep, factor=$BOOST ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Binary:  $BIN"
echo "Ref:     $REF_DIR"
echo "Time:    $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_DR_BOOST=$BOOST \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error\|Mean \|T_rad error\|' stdout.log
echo "--- iter-by-iter shell-0 W ---"
grep -A1 'Shell  W_LUMINA' stdout.log | grep -E '^    0' | awk '{print NR, $2, $3}' | tail -10
echo "--- ion totals (last NLTE iter, shell 0) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- spectrum landings ---"
ls -la nlte_rate_balance.csv lumina_spectrum_formal.csv 2>&1 | tail -3
echo "Done:   $(date)"
