#!/bin/bash
#SBATCH --job-name=a10_kx_ce
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# A/B for the MA-cap-exit fix: LUMINA_MA_CAP_EMIT=1 turns the silent coherent
# fall-through (confirmed defect) into a proper resonant deactivation.
# Falsifier: the 4470A spike (17.8% of flux in 4350-4620 vs ARTIS 8.9%)
# collapses toward ARTIS if fall-through was the culprit; [MA-CAP-EXIT]
# per-iter counter quantifies the event rate either way. Virtual-packet
# estimator also on (report must label virtual-based results).
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1 LUMINA_SPEC_ARG=both

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_spectrum_virtual_coevolve.csv
( export P0TAG="a10_kx_ce"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  export LUMINA_MA_CAP_EMIT=1
  bash scripts/run_coevolve_s01.sh consume )
for csv in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv \
           lumina_kromer_coevolve.csv lumina_spectrum_virtual_coevolve.csv; do
  [ -f "$csv" ] && cp -f "$csv" "logs/coevolve_consume_a10_kx_ce/$csv"
done
echo "a10_kx_ce DONE -> logs/coevolve_consume_a10_kx_ce/"
