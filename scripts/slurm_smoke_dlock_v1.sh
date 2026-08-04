#!/bin/bash
#SBATCH --job-name=smoke_dlock_v1
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Smoke v1: delayed ion-lock (LOCK_START_ITER=10) on champion baseline.
# n_iter=15: iters 0-4 pre-NLTE, iters 5-9 free NLTE (T_inner self-correct via
# overshoot), iters 10-14 locked (level-pop polish on stable T_inner). Goal:
# match Gate B-5 T_inner err 3.83% AND keep ion-lock benefit on iron-peak III.
# Compare to smoke 154434 (lock from iter 5, W err 1653%) and Gate B-5 154432
# (no lock, W err 366%).

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost_L19"
BIN="$ROOT/lumina_cuda_h100_dr_7ion_ce_dlock"

N_PKT=200000
N_ITER=15

cell_dir="$ROOT/logs/smoke_dlock_v1_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== Smoke dlock-v1: delayed ion-lock (LOCK_START=10) on champion L19 ==="
echo "Host: $(hostname) GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "Binary: $BIN"
echo "RefDir: $REF_DIR"
echo "n_pkt=$N_PKT n_iter=$N_ITER NLTE_START=5 LOCK_START=10 SKIP_Z=14"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_ION_LOCK=1 \
LUMINA_NLTE_LOCK_START_ITER=10 \
LUMINA_NLTE_SKIP_Z=14 \
LUMINA_UVOPT_EMIT_BOOST=1.7 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- iter trajectory (T_inner + L_em) ---"
grep -E "Iteration|T_inner:" stdout.log | head -40
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- ion totals at shell 0 (last NLTE iter) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- iron-peak ions (Sc/Ti/V/Cr/Mn/Fe/Co/Ni) shell 0 ---"
grep -iE 'Sc II|Ti II|V II|Cr II|Mn II|Fe II|Co II|Ni II' stdout.log | grep 'shell 0' | tail -10
echo "--- spectrum landings ---"
ls -la lumina_spectrum*.csv 2>&1 | tail -2
echo "Done: $(date)"
