#!/bin/bash
#SBATCH --job-name=p1_photoion
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=UNLIMITED
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# P1: feed the lagged, softened MC shadow field to the photoionization RATE
# (LUMINA_COEVOLVE_PHOTOION_MC=1) so S/Si stop over-ionizing -> S II carrier
# returns -> green lights up / UV drops. Symmetric to the proven bb jbar_line
# rewiring. Runs the champion consumer with a serial alpha sweep (mild->strong,
# so a divergent strong-alpha arm can't waste the safe ones). Control = the
# existing gate-OFF baseline (logs/coevolve_consume, byte-identical binary).
#   Falsifier: any arm moves green(5-6.5k) UP + UV(2.5-3k) DOWN + S II f(II) UP.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}

run_alpha () {
  local a="$1"; local tag="p1_a${a/./}"
  echo "=============================================================="
  echo "[P1] alpha=$a  tag=$tag  PKTS=$PKTS NITER=$NITER"
  echo "=============================================================="
  ( export P0TAG="$tag" LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA="$a"
    bash scripts/run_coevolve_s01.sh consume )
  echo "[P1] alpha=$a DONE -> logs/coevolve_consume_$tag/"
}

# alphas from args (e.g. "0.5" for a fast single-arm signal), else the full sweep
ALPHAS="${@:-0.3 0.5 0.8}"
for a in $ALPHAS; do run_alpha "$a"; done
echo "ALL P1 ALPHA ARMS DONE"
