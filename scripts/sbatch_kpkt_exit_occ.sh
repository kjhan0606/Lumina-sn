#!/bin/bash
#SBATCH --job-name=a10_kx_occ
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# A/B for the occupancy guard (LUMINA_COEVOLVE_PHOTOION_OCC=1): unsampled MC
# bins keep deterministic J in the photoion integral. Falsifier: far-outer
# hot root (s44-47 T_e 55-82kK) returns toward CMFGEN x1.3 band; Ca II 8542 /
# red-NIR excess relax. Pre-registered null-possibility: final-state zero-count
# bins were 0% at s40-47, so the guard may barely fire ([OCC] lines quantify
# per-iter) -> then escalate to stratification / inverse-variance.
# No virtual (timeout lesson), no THEN_MC. Kromer on for palette bookkeeping.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv
( export P0TAG="a10_kx_occ"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  export LUMINA_COEVOLVE_PHOTOION_OCC=1
  bash scripts/run_coevolve_s01.sh consume )
for csv in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv; do
  [ -f "$csv" ] && cp -f "$csv" "logs/coevolve_consume_a10_kx_occ/$csv"
done
echo "a10_kx_occ DONE -> logs/coevolve_consume_a10_kx_occ/"
