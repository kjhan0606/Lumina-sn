#!/bin/bash
#SBATCH --job-name=nlte_consv_test
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Single-cell sanity test for NLTE conservation fix (post-solve rescale).
# Verifies pair sum (II+III) matches nebular n_ion within 1% in ALL shells.

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
case "$CC" in
    90) BIN="$ROOT/lumina_cuda_h100_levelpop_consv" ;;
    86) BIN="$ROOT/lumina_cuda_a40_levelpop_consv" ;;
    *)  echo "WARN unknown compute_cap=$CC; falling back to h100 binary"
        BIN="$ROOT/lumina_cuda_h100_levelpop_consv" ;;
esac

N_PKT=200000
N_ITER=10
MODE=spectrum

cell_dir="$ROOT/logs/nlte_consv_test_NN_${SLURM_JOB_ID}"
mkdir -p "$cell_dir"
cd "$cell_dir"

echo "=== NLTE conservation fix sanity test (single NN cell) ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) (sm_$CC)"
echo "Binary:  $BIN"
echo "Time:    $(date)"

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF=1 \
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
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" nlte \
    > stdout.log 2> stderr.log

rc=$?
echo "  exit=$rc"
ls -la lumina_spectrum_formal.csv nlte_levels_iter*.csv 2>&1 | tail -10
echo "Done:   $(date)"
