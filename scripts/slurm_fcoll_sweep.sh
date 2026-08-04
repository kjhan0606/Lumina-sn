#!/bin/bash
#SBATCH --job-name=fcoll_sweep
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#SBATCH --array=0-3

# Path 3 sweep: pure Compton-adiabatic (boost=0) vs hybrid (1,3) vs current (12).
# T_e ∝ (Γ_eff·T_rad)/(Γ_eff+Γ_ad), Γ_eff = Γ_C·(1+boost). At inner shell with
# t_exp=18d, T_rad=10500K: boost=0 → T_e≈7300K, boost=12 → T_e≈10164K.
# Goal: confirm low boost → low T_e → 10²× higher Milne recomb → R_rec/R_bf up.

export OMP_NUM_THREADS=1
unset OMP_PLACES

BOOSTS=(0 1 3 12)
BOOST=${BOOSTS[$SLURM_ARRAY_TASK_ID]}

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost_L19"
BIN="$ROOT/lumina_cuda_h100_fcoll"

N_PKT=200000
N_ITER=10

cell_dir="$ROOT/logs/fcoll_b${BOOST}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== fcoll sweep: LUMINA_F_COLL_BOOST=$BOOST ==="
echo "Host: $(hostname) GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"
echo "n_pkt=$N_PKT n_iter=$N_ITER NLTE_START=5 SKIP_Z=14"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_NLTE_SKIP_Z=14 \
LUMINA_UVOPT_EMIT_BOOST=1.7 \
LUMINA_SELF_CONSISTENT_TE=1 \
LUMINA_F_COLL_BOOST=$BOOST \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final iter T_rad/T_e ---"
grep -E "Shell  W_LUMINA|^  [0-9]" stdout.log | tail -16
echo "--- iter trajectory ---"
grep -E "Iteration|T_inner:|L_em" stdout.log | head -30
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- ion totals shell 0 (NLTE) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- spectrum landings ---"
ls -la lumina_spectrum*.csv 2>&1 | tail -2
echo "Done: $(date)"
