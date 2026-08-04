#!/bin/bash
#SBATCH --job-name=a10_kx_iupb
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# EUV OVER-PUMP FIX ARM: LUMINA_MACROATOM_IUP_BETA=1 (existing gate).
# Audit verdict: macro-atom internal-up rate = B_lu*J with NO Sobolev-beta
# (plasma.c:1870-1871), over-counting the pump on optically-thick EUV lines
# by 1/beta = tau = 1e2-1e6, closing a self-reinforcing loop (EUV packets ->
# jbar_line -> over-pump -> more EUV). ARTIS applies beta to the up-rate
# ((B_lu - B_ul nu/nl) * beta * J_blue) — this arm restores that symmetry
# (beta already on internal-down via IDOWN_BETA=1).
# Pre-registered predictions if the loop is the EUV floor's engine:
#   1. mc_J(531-758A) floor s12-28 collapses >=10x toward cs_J
#   2. EUV escape share 5.1% -> <1%
#   3. Si f(II) s15/s25 recovers >=10x (toward CMFGEN/ARTIS)
#   4. Kromer: S II emit <70%, Si II appears
#   5. S f(II) ~0.97 unchanged (gnt control leg)
# CAUTION: beta touches ALL lines' up-rates -> spectrum-wide shifts expected;
# judge fluorescence metrics honestly (narrow-corr may move either way).
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

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin
( export P0TAG="a10_kx_iupb"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for csv in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin; do
  [ -f "$csv" ] && cp -f "$csv" "logs/coevolve_consume_a10_kx_iupb/$csv"
done
echo "a10_kx_iupb DONE -> logs/coevolve_consume_a10_kx_iupb/"
