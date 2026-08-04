#!/bin/bash
#SBATCH --job-name=gateB_dr_7ion
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Gate B-4: full iron-peak III→II AS plan-C DR stack.
# 7 AS plan-C ions: Sc III, Co III, Cr III, Ni III, Ti III, V III, Mn III
# (Fe III kept at Nahar 1997 NORAD — plan-C = 0.09× Nahar, literature wins).
# vs B-3 (Sc/Co/Cr/Ni only): adds Ti/V/Mn DR, all of which had only Mazzotta floor.
# Diagnostic from B-3: Ni II 37× over-recombined → expected to also need rebalance
# of partner ions for charge conservation in shell 0.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"
BIN="$ROOT/lumina_cuda_h100_dr_7ion"

N_PKT=200000
N_ITER=10

cell_dir="$ROOT/logs/gateB_dr_7ion_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== Gate B-4: 7-ion AS plan-C iron-peak DR stack ==="
echo "Host: $(hostname) GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Time: $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF="$ROOT/data/atomic/cmfgen_sigma_bf.bin" \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" spectrum nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
echo "--- final convergence ---"
grep -E 'Mean \|W error|Mean \|T_rad error|T_inner final' stdout.log | tail -4
echo "--- ion totals at shell 0 (last NLTE iter) ---"
grep 'shell 0: NLTE n_total' stdout.log | tail -16
echo "--- iron-peak ions (Sc/Ti/V/Cr/Mn/Fe/Co/Ni) ---"
grep -iE 'Sc II|Ti II|V II|Cr II|Mn II|Fe II|Co II|Ni II' stdout.log | grep 'shell 0' | tail -10
echo "--- spectrum landings ---"
ls -la lumina_spectrum_formal.csv 2>&1 | tail -1
echo "Done: $(date)"
