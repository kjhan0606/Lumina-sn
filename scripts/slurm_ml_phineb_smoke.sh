#!/bin/bash
#SBATCH --job-name=ml_phineb_smoke
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Path (d): Mihalas-Lucy phi_neb correction probe.
# Apply f(T_e/T_rad)^exp + global boost to phi_neb in compute_ion_populations()
# and compute_electron_density() to dial down over-ionization at SN photosphere.
#
# Baseline references:
#   154431: pre-Si-DR no UV cap                 (W err 541.78%)
#   155042: +Si DR +UV cap v2 [2000,3500]W=1.5  (W err 542.90%)
# This probe: pre-UV-cap binary path + M-L phi_neb correction.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_ml_phineb"

N_PKT=50000
N_ITER=8

# Probe configuration:
#   T_RATIO_EXP=2: at typical T_e/T_rad=0.9, factor=0.81 (modest 19% reduction)
#   T_RATIO_EXP=4: factor=0.66 (~34% reduction)
#   T_RATIO_EXP=6: factor=0.53 (~47% reduction)
# Apply to all iron-peak III (ion=2 -> ion=3) where over-ionization dominates.
T_EXP=${ML_T_EXP:-4}
BOOST=${ML_BOOST:-1.0}
ZMASK=${ML_ZMASK:-21,22,23,24,25,26,27,28}     # Sc..Ni
IONMASK=${ML_IONMASK:-2}                        # only k=2 (II->III balance)

cell_dir="$ROOT/logs/ml_phineb_smoke_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== M-L phi_neb correction smoke: rate-balance dump ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary:    $BIN"
echo "n_pkt=$N_PKT  n_iter=$N_ITER  NLTE_START=5"
echo "M-L knobs: BOOST=$BOOST  T_RATIO_EXP=$T_EXP  ZMASK=$ZMASK  IONMASK=$IONMASK"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_RATE_DUMP=1 \
LUMINA_ML_PHI_NEB_BOOST=$BOOST \
LUMINA_ML_PHI_NEB_T_RATIO_EXP=$T_EXP \
LUMINA_ML_PHI_NEB_ZMASK=$ZMASK \
LUMINA_ML_PHI_NEB_IONMASK=$IONMASK \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- M-L phi_neb banner (should print once at iter 0) ---"
grep -E '\[M-L phi_neb\]' stdout.log | head -2
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4

echo "--- Si II / Si III shell 0 last iter ---"
if [ -f nlte_rate_balance.csv ]; then
    head -1 nlte_rate_balance.csv
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==14 && $3==0' | tail -4
    echo "--- Fe II / Fe III shell 0 last iter (saturation target) ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $3==0' | tail -4
    echo "--- Ca II / Ca III shell 0 last iter ---"
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==20 && $3==0' | tail -4
fi
echo "Done: $(date)"
