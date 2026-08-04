#!/bin/bash
#SBATCH --job-name=uvcap_smoke
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Validate UV J_nu cap (LUMINA_J_NU_UV_CAP) — clip super-Planckian J_nu in UV
# at W_cap * B_nu(T_rad) inside the NLTE photoionization rate integral.
# Baseline reference:
#   154431: pre-Si-DR        (lumina_cuda_h100_dr_7ion)
#   154938: +Nahar Si DR     (lumina_cuda_h100_dr_si)
# This run binary has Si DR baked in + UV-cap env knobs active.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_uvcap_v2"

N_PKT=50000
N_ITER=8

# v2 cap: bracket [LAMBDA_MIN, LAMBDA_MAX] avoids Wien tail (deep UV B_ν→0).
# Defaults: 2000 < lambda < 3500 A, W_cap=1.5 (moderate first probe).
W_CAP=${UVCAP_W:-1.5}
LAM_MAX=${UVCAP_LAMBDA_MAX:-3500}
LAM_MIN=${UVCAP_LAMBDA_MIN:-2000}

cell_dir="$ROOT/logs/uvcap_smoke_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== UV J_nu cap smoke: rate-balance dump ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary:    $BIN"
echo "n_pkt=$N_PKT  n_iter=$N_ITER  NLTE_START=5"
echo "UV cap:    W <= $W_CAP    for $LAM_MIN < lambda < $LAM_MAX A"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_RATE_DUMP=1 \
LUMINA_J_NU_UV_CAP=1 \
LUMINA_J_NU_UV_CAP_LAMBDA_MAX=$LAM_MAX \
LUMINA_J_NU_UV_CAP_LAMBDA_MIN=$LAM_MIN \
LUMINA_J_NU_UV_W_CAP=$W_CAP \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- UV cap stats (per iter) ---"
grep -E '\[J_nu UV cap\]' stdout.log | tail -20

echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4

echo "--- Si II / Si III shell 0 last iter ---"
if [ -f nlte_rate_balance.csv ]; then
    head -1 nlte_rate_balance.csv
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==14 && $3==0' | tail -4
    echo "--- Fe II / Fe III shell 0 last iter (the σ_bf-saturation target) ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $3==0' | tail -4
    echo "--- Ca II / Ca III shell 0 last iter ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==20 && $3==0' | tail -4
fi
echo "Done: $(date)"
