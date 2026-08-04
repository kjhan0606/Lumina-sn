#!/bin/bash
#SBATCH --job-name=diag_ratebal
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Track B (parallel with #140 CE expansion):
# Diagnose σ_bf vs DR rate balance per (Z, ion_lo, shell, iter) to confirm
# the over-ionization is dominated by photoionization rate (R_bf) 1000× exceeding
# DR + recombination (R_rec). Existing dump path in lumina_plasma.c:2255+,
# triggered by LUMINA_NLTE_RATE_DUMP=1. No rebuild required.
# Uses 7-ion plan-C DR binary (Sc/Ti/V/Cr/Mn/Co/Ni III).

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_dr_7ion"

# Short run: ~50K pkts × 8 iter; we just need rate-balance trajectory, not spectrum quality
N_PKT=50000
N_ITER=8

cell_dir="$ROOT/logs/diag_ratebal_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== Diagnostic: rate-balance dump (σ_bf vs DR per iter) ==="
echo "Host: $(hostname) GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary: $BIN"
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
echo "--- nlte_rate_balance.csv summary ---"
if [ -f nlte_rate_balance.csv ]; then
    n_rows=$(wc -l < nlte_rate_balance.csv)
    echo "  rows: $n_rows"
    echo "  unique (Z,ion_lo): $(tail -n +2 nlte_rate_balance.csv | awk -F, '{print $1","$2}' | sort -u | wc -l)"
    echo "  unique shells: $(tail -n +2 nlte_rate_balance.csv | awk -F, '{print $3}' | sort -un | tr '\n' ' ')"
    echo "  R_bf range: $(tail -n +2 nlte_rate_balance.csv | awk -F, 'NR==1{mn=$11;mx=$11}{if($11<mn)mn=$11;if($11>mx)mx=$11}END{print mn" .. "mx}')"
    echo "  ratio_RrecRbf range: $(tail -n +2 nlte_rate_balance.csv | awk -F, 'NR==1{mn=$17;mx=$17}{if($17<mn)mn=$17;if($17>mx)mx=$17}END{print mn" .. "mx}')"
    echo "  --- last iter snapshot (iron-peak III) ---"
    tail -n 60 nlte_rate_balance.csv | awk -F, '$1>=21 && $1<=28 && $2==2 {printf "Z=%2d ion_lo=%d sh=%2d N=%4d Te=%.0f W=%.2e n_e=%.2e R_bf=%.2e R_rec=%.2e ratio=%.2e\n",$1,$2,$3,$4,$5,$7,$8,$11,$12,$17}' | head -20
fi
echo "--- spectrum landings ---"
ls -la lumina_spectrum*.csv 2>&1 | tail -2
echo "Done: $(date)"
