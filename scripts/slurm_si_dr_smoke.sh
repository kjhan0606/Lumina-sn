#!/bin/bash
#SBATCH --job-name=si_dr_smoke
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Validate Nahar Si II→I + Si III→II DR addition (lumina_cuda_h100_dr_si).
# Baseline reference: logs/diag_ratebal_154431/nlte_rate_balance.csv (lumina_cuda_h100_dr_7ion)
# Expected: Si II shell 0 ratio_RrecRbf 3.11e-10 → ~1.0 (n_e·α≈7.4e-3 ≳ R_bf 5.9e-3).

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_dr_si"

N_PKT=50000
N_ITER=8

cell_dir="$ROOT/logs/si_dr_smoke_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== Si DR smoke: rate-balance dump (compare to diag_ratebal_154431) ==="
echo "Host: $(hostname) GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary: $BIN  (Nahar Si II/III added to DR_TABLE)"
echo "n_pkt=$N_PKT n_iter=$N_ITER NLTE_START=5"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_RATE_DUMP=1 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4

echo "--- Si II / Si III  shell 0 last iter ---"
if [ -f nlte_rate_balance.csv ]; then
    head -1 nlte_rate_balance.csv
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==14 && $3==0' | tail -4
    echo "--- Ca II / Ca III shell 0 last iter (control: unchanged) ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==20 && $3==0' | tail -4
    echo "--- Fe II shell 0 last iter (control) ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $2==1 && $3==0' | tail -2
fi
echo "Done: $(date)"
