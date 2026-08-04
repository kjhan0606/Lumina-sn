#!/bin/bash
#SBATCH --job-name=a10_kx_mabf
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# MACROATOM_BF A/B — the REPAIR-SHAPED falsifier for the Co IV funnel cause
# ============================================================================
# Cause (validation/cmfgen_toy06_19p48d/analysis/coiv_funnel_trace/VERDICT.md):
# deep Co IV (non-NLTE, stage-IV) macro-atom has NO thermalization exit —
# collisions physically negligible (p_k ~ 8e-10 at n_e 4.4e9) and the
# bound-free recombination cascade exit is gate-OFF by default
# (build_recomb_topology, lumina_plasma.c:1245-52). Every erg absorbed into
# deep Co IV re-emits as Co IV UV lines => the 1490-1650A funnel (42% of s0
# energy in one bin; mc/cs=39x @1526A), reddened bath, cold radeq root,
# dead FUV gradient. CMFGEN's deep smoothness comes from SE bf/continuum
# coupling — the exact channel this gate enables.
#
# This is an ENV-ONLY A/B: the champion ./lumina_cuda already contains the
# gate (commit 66586d6). No binary swap. B-run env verbatim + ONE variable:
#     export LUMINA_MACROATOM_BF=1
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES (verbatim — do not move goalposts after the run):
#   PASS (cause CONVICTED as repairable) = ALL of:
#    (i)  s0 u-fraction 1290-2000A drops 0.514 -> <=0.40 AND the ~1508A
#         single-bin share 0.42 -> <=0.15  (funnel collapse);
#    (ii) source-function split heals: mc_J/cs_J @1526A drops 39x -> <=3x
#         AND mc_J/cs_J @1700-2100A rises 0.04 -> >=0.3;
#    (iii) NO LTHERM-style collateral: EUV(300-450A, s0) does NOT collapse
#         (>= B-run 3.9e-12; recomb continua should ADD genuine EUV/FUV edge
#         photons) AND deep Fe f(IV) s0 does not crater (>= 0.5; B-run 0.79);
#    (iv) directional (not gating): FUV(918-1290, s0) rises; T_e(s0) rises
#         above 13120K; slope s0->s8 steepens.
#   PARTIAL: (i) passes but (ii) or (iii) fails => bf exit is necessary but
#            not sufficient; report which leg failed.
#   NULL: (i) fails => check wiring FIRST:
#         "[MACROATOM_BF] recomb cascade topology: N entries" (init, N>0)
#         "[MACROATOM_BF] recomb cascade ENABLED on device"   (init)
#         If topology N=0 or lines absent: gate never armed (cmfgen_loaded /
#         cmfgen_has_sigma prerequisites) — wiring no-op, NOT a physics null.
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- B-run (a10_kx_gphall MODE=all) environment, verbatim ----------------------
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_GPH_ALLLEVEL=1
unset LUMINA_NLTE_BK_CEIL

# --- the ONE new variable ------------------------------------------------------
export LUMINA_MACROATOM_BF=1

TAG="a10_kx_mabf"
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
echo "[mabf] verify wiring: grep MACROATOM_BF logs/coevolve_consume_${TAG}/stdout.log"
