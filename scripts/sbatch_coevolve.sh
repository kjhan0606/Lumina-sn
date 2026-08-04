#!/bin/bash
#SBATCH --job-name=coevolve_s01
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
# COEVOLVE Stage 0+1 validation on a big-memory GPU (h200/h100/a100 only —
# the CMF solve needs ~55 GB, so a40/a10 would CPU-fallback).
#   arg1 = mode: coev (Stage-1 color, default) | byteid (R1 byte-identity)
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-100000} NITER=${NITER:-12}
# arg2 = LUMINA_MC_COEVOLVE_INJECT (0=surface J-CDF [Stage-1], 2=deposition volumetric [Stage-2])
export LUMINA_MC_COEVOLVE_INJECT=${2:-0}
echo "LUMINA_MC_COEVOLVE_INJECT=$LUMINA_MC_COEVOLVE_INJECT"
bash scripts/run_coevolve_s01.sh "${1:-coev}"
