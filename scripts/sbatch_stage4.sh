#!/bin/bash
#SBATCH --job-name=a10_kx_stage4
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# FORK A -- STAGE-IV NLTE/SE PROMOTION (all elements)   (PREPARE ONLY; driver submits)
# ============================================================================
# Exact clone of the B-run recipe (a10_kx_gphall MODE=all == sbatch_tincol.sh's
# env == run_coevolve_s01.sh consume) PLUS the TWO new gate variables:
#     export LUMINA_NLTE_STAGE4=1     # promote (III,IV) pairs for the 7 elements
#                                     #   with stage-IV data: Si,Fe,Co,Ni,Ti,Cr,Al
#     export LUMINA_MACROATOM_BF=1    # thermal-exit companion (A.5 of the design
#                                     #   doc): the recomb cascade that lets the
#                                     #   now-SE-populated deep IV manifolds drain
# Binary: lumina_cuda.withStage4 (carries the stage-IV + BF-coverage gates; ALL
#         env-OFF by default => byte-identical to the B-run when the two gates are
#         unset; carries jtable+tetab+tincol+ltherm gates too, all env-OFF here).
#
# WHAT CHANGES (physics):
#   - Fe/Co/Ni/Si/Ti/Cr/Al IV move OUT of the dilute-Boltzmann fallback INTO the
#     NLTE/SE set (each as the top of an adjacent (III,IV) pair; O-triplet pattern).
#   - S_l is now WRITTEN for stage-IV lines (line-map driven; cuda.cu:1467 /
#     plasma.c:10951) => the deterministic cs and the MC macro-atom share ONE
#     source function for those lines (removes the mc/cs=39x inconsistency at 1526A).
#   - The top-ion continuum drain (plasma.c:TOPSTAGE_IV) auto-detects that III is
#     no longer the top stage (IV in the set) => the real (III,IV) SE pair replaces
#     the Saha-IV proxy; IV (Co/Ni, no V in the dataset) closes the ladder as top.
#   - Gph(III) for the promoted combs switches from LTE b_k=1 to SE weighting now
#     that IV provides the continuum drain (ADDENDUM: closes the Co III 22x deficit
#     WITHOUT the ~300x over-correction of a naive NLTE weighting on an undrained
#     top ion).
#   - build_recomb_topology source coverage is extended to the promoted stage-IV
#     EXCITED manifolds (Co IV lev144 etc.), so the recomb thermal exit is
#     reachable from where the funnel traps packets, not only from the IV ground.
#
# WHY (VERDICT coiv_funnel_trace + STAGE4_NLTE_REPAIR_DESIGN Fork A):
#   Deep-shell Co/Fe/Ni IV were non-NLTE => nebular dilute-Boltzmann pops + no S_l
#   => the MC macro-atom recycled their UV forest resonantly (eps_eff~8e-10) while
#   the cs thermalized it to ~B(T_e). Promotion gives SE pops + a shared S_l; the
#   BF recomb exit (now reachable) drains the deep IV manifolds the way CMFGEN does.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES (verbatim -- do not move the goalposts after the run).
# All numbers at deep shell s0 unless noted. Reference: B-run a10_kx_gphall +
# self-run CMFGEN toy06 @19.48d.
#   (i)   FUNNEL: 1290-2000A u-frac  <= 0.40   AND  1508A single-bin frac <= 0.15
#         (B-run: pile-band u-frac 0.51, 1508A dominant).
#   (ii)  mc_J/cs_J @ 1526.17A  <= 3x  (B-run 39x; MACROATOM_BF-only A/B was 42x).
#   (iii) Co f(IV) s0  -> toward CMFGEN 0.99+  and GENUINELY high (SE pops + a real
#         thermal exit), i.e. NOT the funnel-artifact-maintained value.
#   (iv)  f(Fe IV) s0  >= 0.9  (CMFGEN 1.000).
#   (v)   EUV band (300-450A) mc_J does NOT collapse (stays within ~0.5 dex of B-run;
#         the SE EUV emitters Fe II/III must be preserved -- unlike the blunt LTHERM).
#   (vi)  DIRECTIONAL: T_e(s0) rises, deep FUV(918-1290A) rises, and the s0->s8 mc_J
#         FUV slope steepens toward the DDC15/CMFGEN gradient.
#   (vii) [ADDENDUM] Co f(IV) s8 rises from 0.006 toward CMFGEN 0.099 (the 17x deep
#         under-ionization closes as the SE-drained Co III comb feeds III->IV).
#   PARTIAL: (ii) lands 3x..10x, OR (iii)/(vii) move the right direction but under-
#            shoot -> promotion + exit are partially effective (report which gate).
#   NULL:   mc_J/cs_J does NOT move OR f(IV) craters -> FIRST check the wiring prints
#           (below), THEN re-open the VERDICT.
#
# WIRING CHECKS (grep the stdout FIRST on any null -- a missing print is a build/
# env no-op, NOT a physics null):
#   [STAGE4] LUMINA_NLTE_STAGE4=1: 38 NLTE slots ...     (once, nlte_init)
#   [STAGE4]   Z=26 IV (slot 7): <N> NLTE levels          (per promoted ion; N>0)
#   [STAGE4]   Z=27 IV (slot 12): ...   Z=28 IV (slot 15): ...  (Co/Ni)
#   [STAGE4-BF] recomb source-coverage extension ON: <M> of <S> sources ...
#              (M = promoted stage-IV excited levels; must be >> 0)
#   [MACROATOM_BF] recomb cascade topology: <E> entries over <S> source levels
#              (S must GROW vs the 34-source B-run probe once the extension fires)
#   [NLTE] Lines mapped to NLTE ions: <L2>/2565342   (L2 must exceed the B-run
#              2410046 by ~12.6k -- the stage-IV lines now carry S_l)
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withStage4 binary as ./lumina_cuda for run_coevolve_s01.sh -----
# (that runner hardcodes ./lumina_cuda; back up the current default and restore it
#  on exit, even on error/SIGTERM, so no other run's ./lumina_cuda is clobbered.)
ST_BIN=lumina_cuda.withStage4
[ -x "$ST_BIN" ] || { echo "FATAL: $ST_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[stage4] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$ST_BIN" lumina_cuda
echo "[stage4] installed $ST_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

# --- B-run (a10_kx_gphall MODE=all) environment, verbatim -----------------------
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_GPH_ALLLEVEL=1
# B-run had LUMINA_NLTE_BK_CEIL ABSENT (b_k cap OFF). Clear any inherited value.
unset LUMINA_NLTE_BK_CEIL

# --- Fork A: stage-IV NLTE/SE promotion + the thermal-exit companion (the ONLY
#     two new variables vs the B-run) ------------------------------------------
export LUMINA_NLTE_STAGE4=1
export LUMINA_MACROATOM_BF=1

TAG="a10_kx_stage4"
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
echo "[stage4] verify wiring: grep -E '\[STAGE4|MACROATOM_BF|Lines mapped' logs/coevolve_consume_${TAG}/stdout.log"
