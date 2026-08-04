#!/bin/bash
#SBATCH --job-name=p0_batchA
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# Batch A (fluorescence plan P0): the DECIDING ionization-vs-pump falsifiers.
# Runs the epay27 CHAMPION consumer (run_coevolve_s01.sh consume) three ways,
# SEQUENTIALLY (each run writes fixed-name lumina_*.csv to the repo-root CWD, so
# concurrent runs would clobber each other — serialize to keep them clean).
# Baseline (A1-1) = the existing champion run in logs/coevolve_consume/ (same
# binary); jbardump below reproduces it AND adds the jbar census (A2).
#
#   A1-2 frozenin   : LUMINA_FROZENIN=1            -> freeze ionization to input Saha
#   A1-3 nophotoion : LUMINA_COUPLED_JNU_PHOTOION=0 -> drop the too-blue-J photoion drive
#   A2   jbardump   : LUMINA_JBAR_LINE_DUMP=1       -> baseline + per-line jbar coverage
#
# Falsifier: if green(5000-6500) rises toward ~22% AND UV(2500-3000) drops toward
# <15% in ANY variant (read the copied lumina_spectrum_formal.csv + S/Si II f(II)
# in lumina_levelpop.csv), ionization is the binding layer -> proceed to P1.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}

run_variant () {
  local tag="$1"; shift
  echo "=============================================================="
  echo "[BatchA] variant=$tag  extra_env: $*  PKTS=$PKTS NITER=$NITER"
  echo "=============================================================="
  # env for THIS variant only (subshell so it does not leak to the next run)
  ( export P0TAG="$tag" "$@"; bash scripts/run_coevolve_s01.sh consume )
  echo "[BatchA] variant=$tag DONE -> logs/coevolve_consume_$tag/"
}

run_variant jbardump   LUMINA_JBAR_LINE_DUMP=1
run_variant frozenin   LUMINA_FROZENIN=1
run_variant nophotoion LUMINA_COUPLED_JNU_PHOTOION=0

echo "ALL BATCH-A VARIANTS DONE"
