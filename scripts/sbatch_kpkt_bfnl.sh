#!/bin/bash
#SBATCH --job-name=a10_kx_bfnl
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# BLUE-BUDGET ARM (economy audit rank #1 + #3): chi_bf from NLTE level pops
# (LUMINA_BF_NLTE_POPS=1) + physical C_fb in p_fb (KPKT_FB_MULTI=1), on the
# best-physics stack (si + GPH_SIGMA + IUP_JBLUE). Unified narrative test:
# over-ionization -> chi_bf collapse (41.3 -> 0.075) -> UV/EUV free-stream ->
# (Si II annihilation + no UV reprocessing) -> too-red.
# Pre-registered predictions:
#  1. [BF+FF] UV chi_bf/chi_e collapse stops/reverses (diseased endpoint 0.075)
#  2. bf-absorption ratio etype3/etype1 >> diseased 1.7e-3
#  3. blue >= 40% (jbl 31.1), redNIR <= 25% (45.3), corr(MC,ARTIS) > 0.3 (-0.19)
#  4. Si f(II): s15 retained >= 0.02, s25 FIRST recovery > 0.001
#  5. killer S III em/abs 7.5 -> < 3 (EUV photons now absorbed)
#  6. [FB-MULTI] fb_emit > 0 for the first time (p_fb corrected ~1e-5)
#  7. [JBLUE-ANCHOR] thin-line log-mean ~ 0 (estimator innocent)
#  8. re-fixed gate: Si II 6355 absorption P-Cygni appearance
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_IUP_JBLUE=1
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_BF_NLTE_POPS=1
export LUMINA_KPKT_FB_MULTI=1

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin
( export P0TAG="a10_kx_bfnl"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for f in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin; do
  [ -f "$f" ] && cp -f "$f" "logs/coevolve_consume_a10_kx_bfnl/$f"
done
echo "a10_kx_bfnl DONE -> logs/coevolve_consume_a10_kx_bfnl/"
