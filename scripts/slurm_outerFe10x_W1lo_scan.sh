#!/bin/bash
#SBATCH --job-name=outerFe10x_W1lo
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_a%a_%A.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_a%a_%A.err
#SBATCH --array=0-2

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
W1_LIST=(0.0 0.25 0.5)
W1=${W1_LIST[$SLURM_ARRAY_TASK_ID]}

REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost_L19_outerFe10x"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
case "$CC" in
    90) BIN="$ROOT/lumina_cuda_h100_w4_bump" ;;
    86) BIN="$ROOT/lumina_cuda_a40" ;;
    80) BIN="$ROOT/lumina_cuda_a100" ;;
    *)  BIN="$ROOT/lumina_cuda_h100_w4_bump" ;;
esac

N_PKT=1000000
N_ITER=10
MODE=spectrum

WORKDIR="$ROOT/logs/outerFe10x_W1${W1//./p}_1M_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== outerFe 10× W1 LOW scan: W1=${W1} (1M pkts) ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) (sm_$CC)"
echo "Binary:  $BIN"
echo "RefDir:  $REF_DIR"
echo "Goal: drop UVbl from ~1.5 (W1=1.0) toward 1.0 by reducing UV emit boost."
echo "W3=0.65, W2=0.10 fixed. 1M packets for reliable UVbl variance."
echo "Date:    $(date)"
echo

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF=1 \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_UVOPT_EMIT_BOOST=$W1 \
LUMINA_UVOPT_EMIT_LAM_MIN=1700 \
LUMINA_UVOPT_EMIT_LAM_MAX=3000 \
LUMINA_UVOPT_EMIT_BOOST2=0.10 \
LUMINA_UVOPT_EMIT_LAM_MIN2=5800 \
LUMINA_UVOPT_EMIT_LAM_MAX2=7000 \
LUMINA_UVOPT_EMIT_BOOST3=0.65 \
LUMINA_UVOPT_EMIT_LAM_MIN3=3200 \
LUMINA_UVOPT_EMIT_LAM_MAX3=3800 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" nlte
