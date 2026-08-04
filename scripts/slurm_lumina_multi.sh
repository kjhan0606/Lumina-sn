#!/bin/bash
#SBATCH --job-name=lumina_multi
#SBATCH --partition=h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

# Required to avoid OMP futex barrier blocking on multi-CPU host
export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn

# Detect GPU compute capability and pick matching binary
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
case "$CC" in
    80) BIN_DEFAULT="$ROOT/lumina_cuda_a100_ionmask" ;;
    86) BIN_DEFAULT="$ROOT/lumina_cuda_a40_ionmask"  ;;
    89) BIN_DEFAULT="$ROOT/lumina_cuda_a40_ionmask"  ;;  # sm_86 binary fwd-compat on Ada
    90) BIN_DEFAULT="$ROOT/lumina_cuda_h100_ionmask" ;;
    *)  echo "ERROR: unknown compute_cap=$CC"; exit 2 ;;
esac

BINARY="${LUMINA_BIN:-$BIN_DEFAULT}"
REF_DIR="${REF_DIR:-$ROOT/data/tardis_reference}"
N_PKT="${N_PKT:-200000}"
N_ITER="${N_ITER:-10}"
MODE="${MODE:-spectrum}"
NLTE_FLAG="${NLTE_FLAG:-nlte}"

WORKDIR="$ROOT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== LUMINA multi-partition run ==="
echo "Host:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (sm_$CC)"
echo "Binary:    $BINARY"
echo "RefDir:    $REF_DIR"
echo "Workdir:   $WORKDIR"
echo "Args:      $N_PKT pkts, $N_ITER iters, mode=$MODE, nlte=$NLTE_FLAG"
echo "NLTE_START_ITER: ${LUMINA_NLTE_START_ITER:-5}"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Date:      $(date)"
echo

LUMINA_NLTE_START_ITER="${LUMINA_NLTE_START_ITER:-5}" \
"$BINARY" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" "$NLTE_FLAG"
