#!/bin/bash
#SBATCH --job-name=a10_kx_fbm
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# FIX (a) A/B: per-continuum free-bound sampling (LUMINA_KPKT_FB_MULTI=1)
# on the si config (Si in NLTE, SKIP_Z=""). Root (audited): single
# representative-edge fb emission puts all k-packet fb photons AT the Si II
# ionization threshold (758.5 A) in shells 7-36 -> self-reinforcing Si
# over-ionization (f(II) 1e-3 vs CMFGEN 0.04-0.15).
# Pre-registered predictions:
#  1. [FB-MULTI] SiII-edge share of fb emission << 100% (physical weight)
#  2. Si f(II) s15/s25 recovers >=10x from 1e-3/1e-4 toward CMFGEN 0.04/0.15
#  3. mc field 700-758A excess (was 19-27x cs) -> toward <=3x
#  4. Kromer: S II emit share 84.5% drops; Si II emission appears (>1%)
#  5. corr(MC emergent, ARTIS) rises (si_hs baseline 0.295 @1.6M, kr 0.328 @400k)
#  6. control: S f(II) stays ~0.97 (S edge lies in the dark zone regardless)
# Null branch: SiII-edge share stays high & no f(II) recovery => fb RATE
# excess or missing on-the-spot reabsorption -> escalate to variant (b).
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_KPKT_FB_MULTI=1

rm -f lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv
( export P0TAG="a10_kx_fbm"
  export LUMINA_COEVOLVE_PHOTOION_MC=1 LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0
  export LUMINA_KPACKET=1 LUMINA_KPACKET_EXIT=1
  bash scripts/run_coevolve_s01.sh consume )
for csv in lumina_coevolve_field.csv lumina_spectrum_coevolve_mc.csv lumina_kromer_coevolve.csv; do
  [ -f "$csv" ] && cp -f "$csv" "logs/coevolve_consume_a10_kx_fbm/$csv"
done
echo "a10_kx_fbm DONE -> logs/coevolve_consume_a10_kx_fbm/"
