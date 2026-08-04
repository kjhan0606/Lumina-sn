#!/bin/bash
#SBATCH --job-name=stratV2EF
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_a%a_%A.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_a%a_%A.err
#SBATCH --array=0-1

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
TAGS=(stratV2E_FeBoth03 stratV2F_FeBoth01)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost_L19_${TAG}"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
case "$CC" in
    90) BIN="$ROOT/lumina_cuda_h100_w4_bump" ;;
    86) BIN="$ROOT/lumina_cuda_a40" ;;
    80) BIN="$ROOT/lumina_cuda_a100" ;;
    *)  BIN="$ROOT/lumina_cuda_h100_w4_bump" ;;
esac

WORKDIR="$ROOT/logs/${TAG}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== ${TAG} on $(hostname) sm_$CC === Date: $(date)"

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
"$BIN" "$REF_DIR" 200000 10 spectrum nlte
