#!/bin/bash
#SBATCH --job-name=a10_kx_evlog
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ARCHIVAL RUN: si config (a10_kx + SKIP_Z="") + LUMINA_EVENT_LOG=1.
# Dumps the final-iteration packet event stream (lumina_events.bin) +
# line mapping (lumina_events_lines.bin). All subsequent diagnostics on
# this config (EUV emission provenance by process/line, Kromer++ palettes,
# single-packet forensics) become OFFLINE queries via scripts/read_events.py
# -- no more re-runs for new observables. KROMER kept ON as a built-in
# cross-check (escape palette must be re-derivable from etype=6 events).
# First query (task #19): which process/lines emit at comoving 700-758 A
# in shells 10-30 (the Si II killer window).
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv \
      lumina_events.bin lumina_events_lines.bin
( export P0TAG="a10_kx_evlog"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for f in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv \
         lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin; do
  [ -f "$f" ] && cp -f "$f" "logs/coevolve_consume_a10_kx_evlog/$f"
done
echo "a10_kx_evlog DONE -> logs/coevolve_consume_a10_kx_evlog/"
