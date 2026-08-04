#!/bin/bash
#SBATCH --job-name=uv_idown_smoke
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# P1: Macro-atom UV internal-down suppress.
# Iron-peak (Sc..Ni) II+III internal_down transitions where dest line λ<thresh
# are scaled by FACTOR (<1). Probability mass redistributes to BB-emit and
# internal_up, breaking the UV→UV cascade observed in macro_atom_fate (69%).
#
# Baselines:
#   154431: pre-Si-DR no UV cap                   W err 541.78%
#   155042: +Si DR +UV cap v2                     W err 542.90%
# This binary: full pipeline + UV-IDOWN knob.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_uv_idown"

N_PKT=50000
N_ITER=8

# Probe knobs:
#   FACTOR=0.5  → 50% reduction of UV internal-down
#   FACTOR=0.25 → 75% reduction (more aggressive)
#   THRESH=4000 Å → UV+blue boundary (4000Å covers Fe II UV1, UV2, UV3 multiplets)
#   ZMASK default = 21..28 (Sc..Ni)
#   IONMASK default = 1,2 (II + III)
FACTOR=${UVID_FACTOR:-0.5}
THRESH=${UVID_THRESH:-4000}
ZMASK=${UVID_ZMASK:-21,22,23,24,25,26,27,28}
IONMASK=${UVID_IONMASK:-1,2}

cell_dir="$ROOT/logs/uv_idown_smoke_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== UV internal-down suppress smoke ==="
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary:    $BIN"
echo "n_pkt=$N_PKT  n_iter=$N_ITER  NLTE_START=5"
echo "UV-IDOWN:  FACTOR=$FACTOR  THRESH=${THRESH}Å  ZMASK=$ZMASK  IONMASK=$IONMASK"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_RATE_DUMP=1 \
LUMINA_MACRO_UV_IDOWN_FACTOR=$FACTOR \
LUMINA_MACRO_UV_IDOWN_THRESH=$THRESH \
LUMINA_MACRO_UV_IDOWN_ZMASK=$ZMASK \
LUMINA_MACRO_UV_IDOWN_IONMASK=$IONMASK \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- TransProb banner (UV-IDOWN engaged) ---"
grep -E '\[TransProb\] Recomputed' stdout.log | head -2
echo "--- Macro-atom fate (UV→ bands) ---"
grep -E '\[MA-FATE\]' stdout.log | tail -8
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- Fe II / Fe III shell 0 last iter ---"
if [ -f nlte_rate_balance.csv ]; then
    head -1 nlte_rate_balance.csv
    tail -n 800 nlte_rate_balance.csv | awk -F, '$1==26 && $3==0' | tail -4
fi
echo "Done: $(date)"
