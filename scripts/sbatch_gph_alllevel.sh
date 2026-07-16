#!/bin/bash
#SBATCH --job-name=a10_kx_gphA
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# DETAILED-BALANCE FIX A/B: simul photoionization Gph.
#   MODE=ground : GPH_SIGMA_CMFGEN only (ground-level-only Gph) = baseline A
#   MODE=all    : + GPH_ALLLEVEL (all-level population-weighted Gph) = fix B
# Root (offline-confirmed): simul r=G/(n_e·alpha); alpha=Milne all-level but
# Gph=ground-only -> net over-recombination -> IGE under-ionized (Fe/Co III vs
# benchmark IV). All-level Gph restores detailed balance.
# Pre-registered predictions (B vs A):
#  1. [GPH-ALLLEVEL] armed banner + G_all/G_ground diagnostic > 1 (Fe III)
#  2. Fe/Co core (s0-4) f(IV) rises: ground~0.01 -> all >= 0.3 (benchmark IV~0.5-1)
#  3. <q>(Fe,Co) core 2.0 -> toward 3.0 (CMFGEN); Si/S/Ca ~unchanged
#  4. n_e core rises toward CMFGEN (was 3x low)
#  5. NO runaway: T_e core stays bounded (no 40kK+); far-edge hot root unchanged (separate)
#  6. Ni: may stay railed (bare-top-stage, separate issue) — not required to fix
#  7. spectrum (re-fixed 4 criteria) as ultimate falsifier
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
MODE=${MODE:-all}
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  MODE=$MODE  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
[ "$MODE" = "all" ] && export LUMINA_GPH_ALLLEVEL=1

TAG="a10_kx_gph${MODE}"
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
