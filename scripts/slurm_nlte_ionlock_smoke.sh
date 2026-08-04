#!/bin/bash
#SBATCH --job-name=nlte_ionlock
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Task #145: Mihalas-Lucy ion-lock smoke test.
# DR boost (1..1000) and DR floor (1e-12..1e-9) sweeps were null:
# W err 363-377% across 8 runs. Rate-balance dump showed R_bf=1e8-1e9/s
# vs Milne-R_rec=1e-1..1e-2/s -- 10 orders out, no DR can balance.
#
# This run pins n_ion_lo and n_ion_hi to the nebular W·ζ-Saha estimate
# (TARDIS exact mode) and lets NLTE only redistribute excited levels
# within each ion. If W err drops near the LTE-Saha baseline (~0.7%)
# while spectrum RMS holds, ion-lock is the right path forward.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_ionlock"

N_PKT=200000
N_ITER=10
MODE=spectrum

cell_dir="$ROOT/logs/nlte_ionlock_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== Mihalas-Lucy ion-lock smoke test ==="
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
grep -E 'Mean \|W error\|Mean \|T_rad error\|T_inner final' stdout.log | tail -4
echo "--- ion totals at shell 0 (last NLTE iter) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- rate-balance dump (last iter, shell 0): R_bf vs R_rec_eff per ion ---"
awk -F, 'NR==1{print "ion_pair,sum_R_bf,sum_R_rec,ratio"; next} $3==0 && NR>1 {print $1","$2","$11","$12","$17}' \
    nlte_rate_balance.csv 2>/dev/null | tail -16
echo "--- spectrum landings ---"
ls -la nlte_rate_balance.csv lumina_spectrum_formal.csv 2>&1 | tail -3
echo "Done:   $(date)"
