#!/bin/bash
#SBATCH --job-name=dr_floor
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_e%a_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_e%a_%j.err
#SBATCH --array=9,10,11,12

# Heavy.2 / Task #143: phenomenological DR floor sweep.
# Each array task sets LUMINA_DR_FLOOR_CMS = 1e-${SLURM_ARRAY_TASK_ID} cm^3/s
#   exponent=9  -> alpha_floor=1e-9
#   exponent=10 -> alpha_floor=1e-10
#   exponent=11 -> alpha_floor=1e-11
#   exponent=12 -> alpha_floor=1e-12
# Goal: T-independent recombination floor decouples "any extra recomb" from
# Mazzotta's collapsed Boltzmann factors at SN T_e ~ 5000-10000 K.
# If a knee appears in W err vs floor -> magnitude tells us required IC-DR
# strength. If no convergence at any floor -> recomb side is not the gap.

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

cell_dir="$ROOT/logs/dr_floor_e${EXPN}_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== DR floor sweep, alpha_floor=$FLOOR cm^3/s (exponent=$EXPN) ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Binary:  $BIN"
echo "Ref:     $REF_DIR"
echo "Time:    $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_DR_FLOOR_CMS=$FLOOR \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error\|Mean \|T_rad error\|' stdout.log | tail -4
echo "--- iter-by-iter shell-0 W (last 10 lines) ---"
grep -A1 'Shell  W_LUMINA' stdout.log | grep -E '^    0' | awk '{print NR, $2, $3}' | tail -10
echo "--- ion totals (last NLTE iter, shell 0) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- spectrum landings ---"
ls -la nlte_rate_balance.csv lumina_spectrum_formal.csv 2>&1 | tail -3
echo "Done:   $(date)"
