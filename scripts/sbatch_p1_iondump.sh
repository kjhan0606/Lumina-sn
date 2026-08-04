#!/bin/bash
#SBATCH --job-name=p1_iondump
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:30:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# f(II) evidence repair: every consume run so far copied a STALE Jul-7
# repo-root lumina_ion_pops.csv (LUMINA_ION_POP_DUMP was never set), so the
# falsifier's S II f(II) leg has never actually been measured. Deterministic
# Jbar seed => byte-reproducible physics; this just re-runs the control and
# the P1 alpha=0.5 arm with the ion dump (final + per-iter) enabled.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_ION_POP_DUMP_ITER=1

run_arm () {
  local tag="$1"; shift
  echo "=============================================================="
  echo "[IONDUMP] tag=$tag  extra_env: $*"
  echo "=============================================================="
  rm -f lumina_ion_pops_iter*.csv
  ( export P0TAG="$tag"
    for kv in "$@"; do export "$kv"; done
    bash scripts/run_coevolve_s01.sh consume )
  cp -f lumina_ion_pops_iter*.csv "logs/coevolve_consume_$tag/" 2>/dev/null || true
  echo "[IONDUMP] $tag DONE -> logs/coevolve_consume_$tag/"
}

run_arm base_iondump
run_arm p1_a05_iondump LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=0.5
echo "ALL IONDUMP ARMS DONE"
