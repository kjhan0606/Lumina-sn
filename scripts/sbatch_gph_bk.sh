#!/bin/bash
#SBATCH --job-name=a10_kx_gphbk
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# FINAL TEST C: b_k-weighted all-level Gph (detailed-balance, NLTE populations).
#   = B(all-level) but photoionization weights use ACTUAL NLTE level populations
#     instead of Boltzmann@T_e. Fixes IGE under-ionization WITHOUT IME over-ionization.
# Root (b_k dump): IGE excited b_k=10-57 (real, amplifies Gph -> IV), IME excited
#   b_k=0.015 (depressed -> suppresses Gph -> stays II/III). Boltzmann(b=1) in B
#   over-ionized IME 66x -> spectrum regressed (narrow-corr 0.474->0.372).
# Pre-registered predictions (C vs A vs B):
#  1. [GPH-ALLLEVEL-NLTE] banner + G_nlte/G_boltz > 1 (Fe III amplified)
#  2. IGE core Fe/Co <q> 2.9 -> ~3.0 (>=B, toward CMFGEN 3.0); f(IV) -> ~0.98
#  3. IME (Si/S) over-ionization RESOLVED: s5 Si <q> back toward 2.13 (not 2.60),
#     s9-11 S back to II-dominant (~1.2, not railed 2.00) = A-like or better vs bench
#  4. spectrum narrow-corr >= A (0.474), optical features (Si II/S II/Fe-Mg) restored
#     (NOT B's -0.4..-0.8 suppression) = the decisive falsifier
#  5. NO core runaway; far-edge hot root unchanged (separate, task#15)
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_GPH_ALLLEVEL=1
export LUMINA_GPH_ALLLEVEL_NLTE=1

TAG="a10_kx_gphbk"
mkdir -p logs/coevolve_consume_${TAG}
rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin
( export P0TAG="$TAG"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for f in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin; do
  [ -f "$f" ] && cp -f "$f" "logs/coevolve_consume_${TAG}/$f"
done
echo "${TAG} DONE -> logs/coevolve_consume_${TAG}/"
