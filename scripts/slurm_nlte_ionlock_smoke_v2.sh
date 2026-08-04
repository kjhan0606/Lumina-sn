#!/bin/bash
#SBATCH --job-name=nlte_ionlock_v2
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Task #146: Mihalas-Lucy ion-lock smoke test, v2 (Ni II fallback fix).
#
# Smoke 154235 results: 13/14 ion pairs LOCKED EXACTLY (NLTE n_total = nebular
# n_ion). Ni II broken because Ni II/III's 2000x2000 matrix produces non-finite
# cuBLAS output -> Boltzmann fallback was using COMBINED rescale, dumping
# Ni III's nebular into the Ni II level range (15.6x over-ionization).
#
# v2 patches both CPU (lumina_plasma.c) and GPU (lumina_cuda.cu) fallbacks to
# do per-ion rescale when LUMINA_NLTE_ION_LOCK=1. Also emits a [NLTE-FALLBACK]
# stderr line (capped at 16 events) so we can see how often Ni II/III hits the
# fallback per iteration.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_ionlock_v2"

N_PKT=200000
N_ITER=10
MODE=spectrum

cell_dir="$ROOT/logs/nlte_ionlock_v2_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== Mihalas-Lucy ion-lock smoke v2 (Ni II fallback fix) ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Binary:  $BIN"
echo "Time:    $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_ION_LOCK=1 \
LUMINA_NLTE_RATE_DUMP=1 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- ion totals at shell 0 (last NLTE iter) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- fallback events ---"
grep '\[NLTE-FALLBACK\]' stderr.log | head -20
echo "--- spectrum landings ---"
ls -la nlte_rate_balance.csv lumina_spectrum_formal.csv 2>&1 | tail -3
echo "Done:   $(date)"
