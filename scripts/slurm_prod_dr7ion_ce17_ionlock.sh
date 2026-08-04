#!/bin/bash
#SBATCH --job-name=prod_dr7_ce17_ionlock
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Production smoke: DR-7ion + CE-17 + Mihalas-Lucy ion-lock + Si-skip on
# champion 152761 baseline (tardis_reference_strat6_higherL_aulboost_L19,
# uvopt_emit_boost=1.7). Validates that new NLTE physics doesn't regress
# the RMS=0.045 baseline spectrum quality. Track B/A diagnosis showed
# CE/DR alone cannot fix σ_bf saturation; ion-lock pins ion totals while
# DR-7ion + CE-17 polish level populations and cross-ion redistribution.
#
# Smoke = 200K pkts × 10 iter for ~10 min turnaround. Full-PROD scale
# (800K × 12) only after smoke clears.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost_L19"
BIN="$ROOT/lumina_cuda_h100_dr_7ion_ce"

N_PKT=200000
N_ITER=10

cell_dir="$ROOT/logs/prod_dr7_ce17_ionlock_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== Production smoke: DR-7ion + CE-17 + ion-lock + Si-skip ==="
echo "Host: $(hostname) GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary: $BIN"
echo "RefDir: $REF_DIR"
echo "n_pkt=$N_PKT n_iter=$N_ITER"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_ION_LOCK=1 \
LUMINA_NLTE_SKIP_Z=14 \
LUMINA_UVOPT_EMIT_BOOST=1.7 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- ion totals at shell 0 (last NLTE iter) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- iron-peak ions (Sc/Ti/V/Cr/Mn/Fe/Co/Ni) shell 0 ---"
grep -iE 'Sc II|Ti II|V II|Cr II|Mn II|Fe II|Co II|Ni II' stdout.log | grep 'shell 0' | tail -10
echo "--- spectrum landings ---"
ls -la lumina_spectrum*.csv 2>&1 | tail -2
echo "Done: $(date)"
