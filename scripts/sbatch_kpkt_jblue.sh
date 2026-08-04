#!/bin/bash
#SBATCH --job-name=a10_kx_jbl
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ARTIS-EXACT UP-RATE ARM (differential #2 complete form):
# rate = (B_lu - B_ul*n_u/n_l) * beta * J_blue, with the new blue-wing
# estimator (LUMINA_IUP_JBLUE=1). Fixes iupb's beta^2 double-suppression
# (beta*J_line where J_line already carries (1-beta)S saturation).
# Pre-registered predictions:
#  1. Si f(II) recovery RETAINED: s9 >= 0.04, s15 >= 0.01 (iupb level+)
#  2. too-red recovered: blue >= 40% (iupb 32.1), redNIR <= 25% (36.8),
#     corr(MC,ARTIS) > 0.3 (iupb 0.035)
#  3. spike lands 5-12% (iupb over-collapsed to 1.6, disease 18.3)
#  4. EUV window s15 stays dead (mc/cs < 3)
#  5. S f(II) 0.89-0.97 unchanged
#  6. [IUP-JBLUE] counters: jblue-used >> fallback from iter 2+
#  7. aspirational gate: Si II 6355 absorption P-Cygni first appearance
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

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin
( export P0TAG="a10_kx_jbl"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for f in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv lumina_events.bin lumina_events_lines.bin; do
  [ -f "$f" ] && cp -f "$f" "logs/coevolve_consume_a10_kx_jbl/$f"
done
echo "a10_kx_jbl DONE -> logs/coevolve_consume_a10_kx_jbl/"
