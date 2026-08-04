#!/bin/bash
#SBATCH --job-name=phaseA_ablation
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Phase A: 4-cell NLTE × σ_bf ablation matrix (152629 base)
# Goal: identify whether ²P° upper-level over-population is driven by
#   (1) σ_bf-driven recombination cascade, (2) NLTE rate-equation result, or both.
# Compare n_NLTE / n_LTE_Boltzmann for Si II / Ca II 4p²P° across the 4 cells.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
case "$CC" in
    90) BIN="$ROOT/lumina_cuda_h100_levelpop" ;;
    86) BIN="$ROOT/lumina_cuda_a40_levelpop" ;;
    *)  echo "WARN unknown compute_cap=$CC; falling back to h100 binary"
        BIN="$ROOT/lumina_cuda_h100_levelpop" ;;
esac

N_PKT=200000
N_ITER=10
MODE=spectrum

# 4 cells: (NLTE on/off) x (σ_bf on/off)
declare -a CELLS=("NN" "NF" "FN" "FF")
declare -A NLTE=( ["NN"]="nlte" ["NF"]="nlte" ["FN"]="lte"  ["FF"]="lte"  )
declare -A SBF=(  ["NN"]="1"    ["NF"]="0"    ["FN"]="1"    ["FF"]="0"    )

echo "=== Phase A 4-cell ablation: NLTE × σ_bf at 152629 base ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) (sm_$CC)"
echo "Binary:  $BIN"
echo "RefDir:  $REF_DIR"
echo "Args:    $N_PKT pkts, $N_ITER iters, $MODE"
echo "Date:    $(date)"
echo

for cell in "${CELLS[@]}"; do
    nlte_arg="${NLTE[$cell]}"
    sbf_val="${SBF[$cell]}"
    cell_dir="$ROOT/logs/phaseA_ablate_${cell}_${SLURM_JOB_ID}"
    mkdir -p "$cell_dir"
    cd "$cell_dir"

    echo
    echo "--- CELL $cell : NLTE=$nlte_arg, sigma_bf=$sbf_val ---"
    echo "Workdir: $cell_dir"
    echo "Time:    $(date)"

    # Common 152629 settings; toggle σ_bf (LUMINA_CMFGEN_SIGMA_BF) and NLTE (positional arg).
    # LUMINA_NLTE_LEVEL_DUMP=1 -> dumps nlte_levels_iter*.csv per iteration (last is converged)
    LUMINA_BF_OPACITY=1 \
    LUMINA_CMFGEN_SIGMA_BF="$sbf_val" \
    LUMINA_DYNAMIC_TRANSPROB=1 \
    LUMINA_NLTE_START_ITER=5 \
    LUMINA_NLTE_LEVEL_DUMP=1 \
    LUMINA_UVOPT_EMIT_BOOST=1.0 \
    LUMINA_UVOPT_EMIT_LAM_MIN=1700 \
    LUMINA_UVOPT_EMIT_LAM_MAX=3000 \
    LUMINA_UVOPT_EMIT_BOOST2=0.15 \
    LUMINA_UVOPT_EMIT_LAM_MIN2=5800 \
    LUMINA_UVOPT_EMIT_LAM_MAX2=7000 \
    LUMINA_UVOPT_EMIT_BOOST3=0.65 \
    LUMINA_UVOPT_EMIT_LAM_MIN3=3200 \
    LUMINA_UVOPT_EMIT_LAM_MAX3=3800 \
    "$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" "$nlte_arg" \
        > stdout.log 2> stderr.log

    rc=$?
    echo "  exit=$rc"
    ls -la lumina_spectrum_formal.csv nlte_levels_iter*.csv 2>&1 | tail -5
done

echo
echo "=== Phase A ablation complete: $(date) ==="
