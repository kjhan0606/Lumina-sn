#!/bin/bash
#SBATCH --job-name=cefix_smoke
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# CE detailed-balance sign-flip fix smoke probe.
# Bug: lumina_plasma.c:2703 used fabs(delta_E_eV) → reverse rate always
#      exp(+|ΔE|/kT) (k_rev > k_fwd), regardless of exothermic forward.
# Fix: drop fabs → signed delta_E_eV → reverse Boltzmann-suppressed correctly.
# Compare to baseline 154431/155042: W err 541.78%, Fe II sh0 n_ion=2.19e4.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_cefix"

N_PKT=50000
N_ITER=8

cell_dir="$ROOT/logs/cefix_smoke_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== CE sign-flip fix smoke probe ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary:    $BIN"
echo "n_pkt=$N_PKT  n_iter=$N_ITER  NLTE_START=5"

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
echo "--- Fe II / Fe III shell 0 last iter ---"
if [ -f nlte_rate_balance.csv ]; then
    head -1 nlte_rate_balance.csv
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $3==0' | tail -2
    echo "--- Co II/III shell 0 ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==27 && $3==0' | tail -2
    echo "--- Ni II/III shell 0 ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==28 && $3==0' | tail -2
    echo "--- Si II/III shell 0 (CE: Ca-Si pair) ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==14 && $3==0' | tail -2
    echo "--- Ca II/III shell 0 ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==20 && $3==0' | tail -2
fi
echo "Done: $(date)"
