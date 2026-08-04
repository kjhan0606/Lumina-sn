#!/bin/bash
#SBATCH --job-name=lumina_h100
#SBATCH --partition=h100
#SBATCH --nodelist=syn08
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Required to avoid OMP futex barrier blocking on multi-CPU host
export OMP_NUM_THREADS=1
unset OMP_PLACES

# H100 = sm_90, use prebuilt h100 binary or build with GPU_ARCH=sm_90
BINARY="${LUMINA_BIN:-/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/lumina_cuda_h100}"
REF_DIR="${REF_DIR:-/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference}"
N_PKT="${N_PKT:-200000}"
N_ITER="${N_ITER:-10}"
MODE="${MODE:-spectrum}"
NLTE_FLAG="${NLTE_FLAG:-nlte}"

# Per-job working dir under logs/ to keep CSV outputs separated
WORKDIR="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== LUMINA H100 run ==="
echo "Host:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Binary:    $BINARY"
echo "RefDir:    $REF_DIR"
echo "Workdir:   $WORKDIR"
echo "Args:      $N_PKT pkts, $N_ITER iters, mode=$MODE, nlte=$NLTE_FLAG"
echo "NLTE_START_ITER: ${LUMINA_NLTE_START_ITER:-5}"
echo "Date:      $(date)"
echo

LUMINA_NLTE_START_ITER="${LUMINA_NLTE_START_ITER:-5}" \
"$BINARY" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" "$NLTE_FLAG"
