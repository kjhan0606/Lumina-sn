#!/bin/bash
#SBATCH --job-name=mafate_bump_diag
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
    80|86|89) BIN="$ROOT/lumina_cuda_h100_mafate_bump"
              echo "WARN: built for sm_90 only; $CC may JIT-fail" ;;
    90)       BIN="$ROOT/lumina_cuda_h100_mafate_bump" ;;
    *)        echo "ERROR: unknown compute_cap=$CC"; exit 2 ;;
esac

REF_DIR="$ROOT/data/tardis_reference_strat6_higherL_aulboost_L19"
N_PKT=200000
N_ITER=5
MODE=spectrum

WORKDIR="$ROOT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== MA-FATE [3000,3100] BUMP per-(Z,ion) diagnostic ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) (sm_$CC)"
echo "Binary:  $BIN"
echo "RefDir:  $REF_DIR"
echo "Args:    $N_PKT pkts, $N_ITER iters, $MODE, nlte"
echo "Date:    $(date)"
echo

# Champion 152761 env: W1=1.7, W2=0.10, W3=0.65 wide on [3200,3800]
LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF=1 \
LUMINA_NLTE_START_ITER=2 \
LUMINA_UVOPT_EMIT_BOOST=1.7 \
LUMINA_UVOPT_EMIT_LAM_MIN=1700 \
LUMINA_UVOPT_EMIT_LAM_MAX=3000 \
LUMINA_UVOPT_EMIT_BOOST2=0.10 \
LUMINA_UVOPT_EMIT_LAM_MIN2=5800 \
LUMINA_UVOPT_EMIT_LAM_MAX2=7000 \
LUMINA_UVOPT_EMIT_BOOST3=0.65 \
LUMINA_UVOPT_EMIT_LAM_MIN3=3200 \
LUMINA_UVOPT_EMIT_LAM_MAX3=3800 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" nlte
