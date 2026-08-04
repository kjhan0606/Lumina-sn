#!/bin/bash
#SBATCH --job-name=a10_kx_bfnl2
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# BFNL RERUN after the [KPKT-FBUP] wiring fix: the Path A k-packet continuum
# exit probs (p_ff/p_fb + legacy fb edge freq) were init-upload-only (before
# the first table build) -> device copies stayed 0 -> ff/fb exits NEVER fired
# in ANY archived run (etype4=etype5=0 in bfnl AND euvf). bfnl (177732) thus
# ran "bf absorption ON / fb re-emission OFF" = a Kirchhoff-inconsistent
# absorb-only lamp; its mid-shell T_e runaway (s25 9.5->21kK, J[mid] 100x) is
# hypothesized to be this unpaid pair.
# Pre-registered predictions (vs bfnl 177732 baselines):
#  1. [FB-MULTI] fb_emit > 0 first time; etype4(ff) > 0, etype5(fb) > 0
#  2. T_e runaway softens: T_e[25] final < 21kK (absorb-only lamp was the driver);
#     ideally plateau toward CMFGEN ~9.5-11kK
#  3. Si f(II) s15 >= 0.02 (bfnl 0.005 — hot re-ionization killed it) if #2 holds
#  4. EUV 700-758 ledger stays closed: em/(line+bf abs) ~ 1 (bfnl 1.00)
#  5. color retained/improved: blue >= 40 (39.9), redNIR <= 25 (23.7),
#     S II emit share < 50 (56.2)
#  6. UV chi_bf/chi_e endpoint stays arrested >= 0.3 (bfnl 0.58, diseased 0.075)
#  7. narrow-band median|log r| vs ARTIS < 0.305
#  8. re-fixed gate: Si II 6355 absorption P-Cygni (bfnl depth -0.13, ARTIS +0.40)
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
( export P0TAG="a10_kx_bfnl2"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for f in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin; do
  [ -f "$f" ] && cp -f "$f" "logs/coevolve_consume_a10_kx_bfnl2/$f"
done
echo "a10_kx_bfnl2 DONE -> logs/coevolve_consume_a10_kx_bfnl2/"
