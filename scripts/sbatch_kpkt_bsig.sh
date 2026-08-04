#!/bin/bash
#SBATCH --job-name=a10_kx_bsig
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# COMBINED ORTHODOX ARM: internal-up Sobolev beta (IUP_BETA) + detailed-balance
# sigma_bf (GPH_SIGMA_CMFGEN) on the si config. Both are physics-correctness
# fixes, not knobs. Speculative-chained after iupb: CANCEL before start if the
# iupb solo verdict shows beta restoration is destructive.
# Pre-registered predictions (additive expectation):
#  1. killer S III EUV lines (700-738A) em/abs ratio 8.8 -> toward 1 (events)
#  2. Si f(II) s15/s25 recovery (gate: >=10x from 1e-3)
#  3. 4470 spike stays ~8-9% (sigma effect retained)
#  4. palette: S II emit < 70%
#  5. S f(II) stays 0.85-0.97 band
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_MACROATOM_IUP_BETA=1
export LUMINA_GPH_SIGMA_CMFGEN=1

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin
( export P0TAG="a10_kx_bsig"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for f in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin; do
  [ -f "$f" ] && cp -f "$f" "logs/coevolve_consume_a10_kx_bsig/$f"
done
echo "a10_kx_bsig DONE -> logs/coevolve_consume_a10_kx_bsig/"
