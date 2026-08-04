#!/bin/bash
#SBATCH --job-name=a10_kx_si
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# DECISIVE PALETTE EXPERIMENT: un-skip Si (LUMINA_NLTE_SKIP_Z="") in the a10_kx
# config. toy06 envelope (s12+) is Si 0.55 / S 0.35 / Ca 0.10, yet Si has been
# NLTE/macro-atom-DEAD via legacy SKIP_Z=14 (super-thermal-S_l-era guard; Si has
# 863 levels in the atomic data). Kromer verdict on a10_kx: S II emits 87.6% of
# ALL escaped flux (4470A window 98.2%) = wrong fluorescence palette.
# Pre-registered predictions if palette-root: S II emit share <50%, Si II top-2
# emitter/absorber, 4350-4620 share 17.8%->toward ARTIS 8.9%, corr(MC,ARTIS)
# 0.33 -> >0.5. If NLTE blows up (NaN/ill-conditioning) => documents the
# original skip reason; fix path becomes Si conditioning, not just un-skip.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
# NOTE: virtual (LUMINA_SPEC_ARG=both) removed — final-pass cost >4.5h at
# 400k pkts (176523 TIMEOUT at 6h). Do not re-enable without LOS/subsample redesign.
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_NLTE_SKIP_Z=""

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_spectrum_virtual_coevolve.csv
( export P0TAG="a10_kx_si"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for csv in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv \
           lumina_kromer_coevolve.csv lumina_spectrum_virtual_coevolve.csv; do
  [ -f "$csv" ] && cp -f "$csv" "logs/coevolve_consume_a10_kx_si/$csv"
done
echo "a10_kx_si DONE -> logs/coevolve_consume_a10_kx_si/"
