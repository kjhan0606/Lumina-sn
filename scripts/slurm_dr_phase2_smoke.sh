#!/bin/bash
#SBATCH --job-name=dr_phase2_smoke
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Heavy.1 (CMFGEN sigma_bf -> NLTE photoionization rates) +
# Heavy.2 (Badnell + NORAD + Mazzotta DR Burgess fits): smoke test.
# Goal: verify the Phase-1 W err 7172% blow-up collapses once Phase-2 DR
# provides the missing recombination channel.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_dr_phase2"

N_PKT=200000
N_ITER=10
MODE=spectrum

cell_dir="$ROOT/logs/dr_phase2_smoke_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== DR Phase 1+2 smoke ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Binary:  $BIN"
echo "Ref:     $REF_DIR"
echo "Time:    $(date)"

# Phase 1: pass absolute CMFGEN sigma_bf path so it resolves regardless of cwd.
# (Previous run used "=1" which is relative -> not-found -> Kramers fallback.)
LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_RATE_DUMP=1 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- CMFGEN sigma_bf load ---"
grep -E 'CMFGEN sigma_bf|σ_bf source' stdout.log | head -5
echo "--- spectrum / rate-balance landings ---"
ls -la nlte_rate_balance.csv lumina_spectrum_formal.csv 2>&1 | tail -10
echo "Done:   $(date)"
