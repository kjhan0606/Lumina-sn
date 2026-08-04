#!/bin/bash
#SBATCH --job-name=base_n32
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# Fixed-point hunt: NITER=12 arms were UNCONVERGED (f(II) peaks at iter 5-6
# then slides down ~0.05/iter with no plateau; only alpha=1.0 converges, flat
# by iter 6). This runs the deterministic-J baseline out to NITER=32 with
# per-iter ion dumps to find where the self-reinforcing re-ionization drift
# actually lands (falsifier-grade demonstration of the P1 mechanism).
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=32
export LUMINA_ION_POP_DUMP=1 LUMINA_ION_POP_DUMP_ITER=1

rm -f lumina_ion_pops_iter*.csv
( export P0TAG="base_n32"
  bash scripts/run_coevolve_s01.sh consume )
cp -f lumina_ion_pops_iter*.csv logs/coevolve_consume_base_n32/ 2>/dev/null || true
echo "base_n32 DONE -> logs/coevolve_consume_base_n32/"
