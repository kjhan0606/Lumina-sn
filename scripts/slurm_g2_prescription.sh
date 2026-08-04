#!/bin/bash
#SBATCH --job-name=g2_prescription
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err

export OMP_NUM_THREADS=1
unset OMP_PLACES

ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
REF_DIR="$ROOT/data/tardis_reference_g2_prescription"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
case "$CC" in
    90) BIN="$ROOT/lumina_cuda_h100_w4_bump" ;;
    86) BIN="$ROOT/lumina_cuda_a40_ionmask" ;;
    80) BIN="$ROOT/lumina_cuda_a100_ionmask" ;;
    *)  BIN="$ROOT/lumina_cuda_h100_w4_bump" ;;
esac

N_PKT=1000000
N_ITER=10
MODE=spectrum

WORKDIR="$ROOT/logs/g2_prescription_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== G2 inverse-regression prescription LUMINA verification (1M pkts) ==="
echo "Host:    $(hostname)  GPU:$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) (sm_$CC)"
echo "Binary:  $BIN"
echo "RefDir:  $REF_DIR"
echo "Theta:   lineridge α=0.75 winner (emulator RMS 0.6980 → 0.5355, -23.3%)"
echo "Key:     log_L=43.805, v_inner=14528, t_exp=21.39d, density_exp=-14.0"
echo "         X_Fe core/wall/outer = 0.183/0.423/0.145, X_Si_wall=0.462, X_Ni=0.125"
echo "Goal: does emulator's parametric prescription land at low RMS in real LUMINA?"
echo "No W1/W2/W3 emit boosts — raw G2 only (clean physical test)."
echo "Date:    $(date)"
echo

LUMINA_BF_OPACITY=1 \
LUMINA_CMFGEN_SIGMA_BF=1 \
LUMINA_DYNAMIC_TRANSPROB=1 \
LUMINA_NLTE_START_ITER=5 \
"$BIN" "$REF_DIR" "$N_PKT" "$N_ITER" "$MODE" nlte
