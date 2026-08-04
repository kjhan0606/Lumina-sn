#!/bin/bash
#SBATCH --job-name=a10_kx_tepin
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# CAUSALITY FALSIFIER for the hot-band -> EUV -> Si II annihilation chain.
# Pins s40-49 T_e to the CMFGEN profile (12765..24600 K linear) via the new
# diagnostic gate. Pre-registered predictions if the chain is real:
#   1. mc_J(531-758A) at s25 collapses >5x; outward-rising profile flattens
#   2. Si f(II) s15/s25 recovers >=10x (toward ARTIS direction)
#   3. S f(II) stays ~0.97 (gnt-locked control leg)
#   4. Kromer: S II emit share <70%, Si II appears (>1%)
# NULL branch: EUV survives the pin -> unification refuted, source is
# mid-shell emission -> hot root and palette revert to independent defects.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_DIAG_TE_PIN=40:49:12765:24600

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv
( export P0TAG="a10_kx_tepin"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for csv in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv; do
  [ -f "$csv" ] && cp -f "$csv" "logs/coevolve_consume_a10_kx_tepin/$csv"
done
echo "a10_kx_tepin DONE -> logs/coevolve_consume_a10_kx_tepin/"
