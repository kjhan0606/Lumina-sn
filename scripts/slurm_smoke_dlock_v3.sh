#!/bin/bash
#SBATCH --job-name=smoke_dlock_v3
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Smoke v3: full freeze (transport-only) once lock activates.
# Iters 0-9: free NLTE → T_inner self-correct via overshoot (matches Gate B-5).
# Iters 10-14: lock active → skip solve_radiation_field, gamma_dep, T_e,
# compute_plasma_state, BF, NLTE, transprob, T_inner. Pure packet propagation
# on frozen plasma — TARDIS spectrum-iter pattern.
# Compare to v1 154450 (W err 3497%, lock-only) and v2 154461 (W err 1809%,
# T_inner err 2.46%, lock+T_inner-freeze; but L_em monotonic collapse from
# locked-totals NLTE level-pop drift).

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost_L19"
BIN="$ROOT/lumina_cuda_h100_dr_7ion_ce_dlock_v3"

N_PKT=200000
N_ITER=15

cell_dir="$ROOT/logs/smoke_dlock_v3_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== Smoke dlock-v3: full plasma freeze + transport-only after lock ==="
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
grep -E "Iteration|T_inner:|plasma frozen" stdout.log | head -60
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- ion totals at shell 0 (last NLTE iter, pre-lock) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- iron-peak ions shell 0 ---"
grep -iE 'Sc II|Ti II|V II|Cr II|Mn II|Fe II|Co II|Ni II' stdout.log | grep 'shell 0' | tail -10
echo "--- spectrum landings ---"
ls -la lumina_spectrum*.csv 2>&1 | tail -2
echo "Done: $(date)"
