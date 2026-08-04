#!/bin/bash
#SBATCH --job-name=a10_kx_bfnl3
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# FB THERMAL-LEDGER FIX (LUMINA_FB_COOL_KT=1): ARTIS bfcooling_integrand
# weights fb thermal cooling by h(nu-nu_edge)~kTe ONLY (chi belongs to the
# ionization ledger). Our (chi+kT) form over-charged the electron heat bath
# ~40x in the bfnl2 trap state at FOUR sites (simul_r1 valve kill, C_fb_real
# p_fb->1 trap closure, edge-CDF hard-edge bias, radeq_fb_rate_eval).
# Economy-audit erratum: the "(h nu_edge + kTe)" recommendation is REFUTED
# by ARTIS source; only the alpha_rr rate part of the audit stands.
# Pre-registered predictions (vs bfnl2 177955 baselines):
#  1. p_fb NO ratchet: stays < 0.05 all shells/iters (bfnl2 s16 0.019->0.956)
#  2. NO trap: fb_emit per iter < 10M (bfnl2 it9 1.69e9); chain p50 < 50 (588)
#  3. thermal valve LIVES: pins back to O(bfnl) level (<= hi10/lo5 vs 44/50),
#     T_e[25] evolves (not frozen at 9481K), no 21kK runaway either (bfnl)
#  4. chi_bf/chi_e UV arrest retained: endpoint 0.1-3 (bfnl 0.58, disease 0.075)
#  5. EUV 700-758 ledger stays closed: em/(line+bf abs) ~ 1
#  6. Si f(II) s15 >= 0.02 (bfnl 0.005, bfnl2 0.000) -- the prize
#  7. color: blue >= 40 (bfnl 39.9), redNIR <= 25, S II emit < 50%
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
export LUMINA_FB_COOL_KT=1

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin
( export P0TAG="a10_kx_bfnl3"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for f in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin; do
  [ -f "$f" ] && cp -f "$f" "logs/coevolve_consume_a10_kx_bfnl3/$f"
done
echo "a10_kx_bfnl3 DONE -> logs/coevolve_consume_a10_kx_bfnl3/"
