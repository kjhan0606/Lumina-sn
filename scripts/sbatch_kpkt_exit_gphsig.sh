#!/bin/bash
#SBATCH --job-name=a10_kx_gphsig
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# PHYSICAL FIX ARM: detailed-balance-consistent photoionization cross-section.
# Gph loop swaps hydrogenic Kramers sigma_bf for the ground-state CMFGEN
# sigma_bf (same table the recombination Milne integral uses). Audit predicts
# the deterministic-field Si over-ionization shrinks ~15x; under the MC field
# the effect reweights the 531-758A window by the real sigma(nu) shape.
# Single-variable arm: no T_e pin here.
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv
( export P0TAG="a10_kx_gphsig"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for csv in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv; do
  [ -f "$csv" ] && cp -f "$csv" "logs/coevolve_consume_a10_kx_gphsig/$csv"
done
echo "a10_kx_gphsig DONE -> logs/coevolve_consume_a10_kx_gphsig/"
