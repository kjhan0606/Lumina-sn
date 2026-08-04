#!/bin/bash
#SBATCH --job-name=w4_bump_probe
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
case "$CC" in
    90) BIN="$ROOT/lumina_cuda_h100_w4_bump" ;;
    *)  echo "ERROR: this binary is sm_90; got compute_cap=$CC"; exit 2 ;;
esac

REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost_L19"
N_PKT="${N_PKT:-200000}"
N_ITER="${N_ITER:-10}"
MODE=spectrum

WORKDIR="$ROOT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

W4_BOOST="${W4_BOOST:-0.5}"
W4_ZMASK="${W4_ZMASK:-24,26,27}"
W4_IONMASK="${W4_IONMASK:-2}"

echo "=== W4 [3000,3100] iron-peak III bump suppress probe ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "Binary:  $BIN"
echo "RefDir:  $REF_DIR"
echo "Args:    $N_PKT pkts, $N_ITER iters, $MODE, nlte"
echo "Champion: W1=1.7 W2=0.10 W3=0.65 wide on Fe+Ni II"
echo "W4:      boost=$W4_BOOST [3000,3100] ZMASK=$W4_ZMASK IONMASK=$W4_IONMASK"
echo "Date:    $(date)"
echo

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF=1 \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
LUMINA_UVOPT_EMIT_BOOST=1.7 \
LUMINA_UVOPT_EMIT_LAM_MIN=1700 \
LUMINA_UVOPT_EMIT_LAM_MAX=3000 \
LUMINA_UVOPT_EMIT_BOOST2=0.10 \
LUMINA_UVOPT_EMIT_LAM_MIN2=5800 \
LUMINA_UVOPT_EMIT_LAM_MAX2=7000 \
LUMINA_UVOPT_EMIT_BOOST3=0.65 \
LUMINA_UVOPT_EMIT_LAM_MIN3=3200 \
LUMINA_UVOPT_EMIT_LAM_MAX3=3800 \
LUMINA_UVOPT_EMIT_BOOST4=$W4_BOOST \
LUMINA_UVOPT_EMIT_LAM_MIN4=3000 \
LUMINA_UVOPT_EMIT_LAM_MAX4=3100 \
LUMINA_UVOPT_EMIT_ZMASK4=$W4_ZMASK \
LUMINA_UVOPT_EMIT_IONMASK4=$W4_IONMASK \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" nlte
