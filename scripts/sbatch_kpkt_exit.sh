#!/bin/bash
#SBATCH --job-name=a10_kx
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ARTIS single-exit k-packet validation (task #13): alpha=1.0 MC co-evolve arm
# with the collisional channel ON but ARTIS macroatom.cc single-exit semantics
# (LUMINA_KPACKET=1 + LUMINA_KPACKET_EXIT=1). Baselines: a10 (KPACKET=0, channel
# off, too-red family) and epay27 (KPACKET=1 re-inject runaway, UV 99.5%).
# Judge: S f(II) front s6-8 vs ARTIS (0.244/0.733/0.965) + UV/green fractions.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_ION_POP_DUMP_ITER=1

rm -f lumina_ion_pops_iter*.csv
( export P0TAG="a10_kx"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
cp -f lumina_ion_pops_iter*.csv logs/coevolve_consume_a10_kx/ 2>/dev/null || true
cp -f lumina_coevolve_field.csv logs/coevolve_consume_a10_kx/ 2>/dev/null || true
echo "a10_kx DONE -> logs/coevolve_consume_a10_kx/"
