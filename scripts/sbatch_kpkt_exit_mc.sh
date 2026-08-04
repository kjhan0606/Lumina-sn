#!/bin/bash
#SBATCH --job-name=a10_kx_r2
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# a10_kx byte-reproduction on the new binary that writes the FINAL-iter MC
# emergent spectrum (lumina_spectrum_coevolve_mc.csv) from the in-loop co-evolve
# transport escape tally. THEN_MC is bypassed in co-evolve mode (176488 lesson),
# so this writer is the only fair MC observable vs ARTIS's MC spectrum.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_ION_POP_DUMP_ITER=1

rm -f lumina_ion_pops_iter*.csv lumina_spectrum_coevolve_mc.csv
( export P0TAG="a10_kx_r2"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
cp -f lumina_ion_pops_iter*.csv logs/coevolve_consume_a10_kx_r2/ 2>/dev/null || true
for csv in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv; do
  [ -f "$csv" ] && cp -f "$csv" "logs/coevolve_consume_a10_kx_r2/$csv"
done
echo "a10_kx_r2 DONE -> logs/coevolve_consume_a10_kx_r2/"
