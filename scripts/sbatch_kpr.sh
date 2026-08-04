#!/bin/bash
#SBATCH --job-name=a10_kx_kpr
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# KP_EMISS_REPAIR + STAGE4-ROUND2  (the composed mastermind repair)
#                                                    (PREPARE ONLY -- driver submits)
# ============================================================================
# Exact clone of the B-run recipe (a10_kx_gphall MODE=all -- same env as
# sbatch_bsrc.sh's B-run block) PLUS the composed upstream repair:
#   STAGE4-R2 (Part A, active under LUMINA_NLTE_STAGE4=1 round-2 semantics):
#     A1  depth-gate + b_k cap on the promoted III-comb Gph weighting
#         (LUMINA_STAGE4_GPH_WTHR=0.13, LUMINA_STAGE4_BK_CAP=1000)   plasma.c:5490,5537
#     A2  top-ion Saha closure: zero the level-less Ni/Co V ladder rung
#         (default ON under stage4; LUMINA_SIMUL_CAP_TOPION override)  plasma.c:4956,~4969
#     A3  drop Ti (Z=22) IV singular pair from the NLTE solve            cuda.cu:~870
#   KPEMISS_REPAIR (Part B, master gate LUMINA_KPEMISS_REPAIR=1; all knobs off => byte-id):
#     B1  KPEMISS_SE_POPS: build kp_emiss from SE/NLTE pops, not dilute-Boltzmann  plasma.c:2117
#     B2  KPKT_FB_MULTI: real per-edge frozenin_alpha_rr recombination fb SED (existing gate)
#     B3  KPEMISS_BSRC_TAU=0.13: B(T_e) k-packet exit (-4) where W>tau (deep only)  cuda.cu:~3080
#   Shipped WITH Fork B (LUMINA_LINE_BSRC=1 MODE=1) per KP_EMISS_REPAIR_DESIGN §Fork B
#   (belt-and-suspenders; the Fork-B-off subsumption A/B is a LATER arm, not this run).
# Binary: lumina_cuda.withKpr (carries A1/A2/A3 + B1/B2/B3 gates; ALL default-OFF so
#         master-gate-off + stage4-off => byte-identical to the champion B-run).
#
# ONE physics mode, one gate, three composable knobs (KP_EMISS_REPAIR_DESIGN §3):
#   make the 94%-of-deep k-packet channel re-emit the thick-zone THERMAL SOURCE
#   (from correct SE pops, with a real continuum floor) instead of the resonant
#   Co IV 1490-1650A line forest.  A1-A3 supply the correct stage-IV ionization/pops
#   WITHOUT the round-1 blowup; B1-B3 make the channel sample B(T_e)/continuum deep.
#
# ---------------------------------------------------------------------------
# PER-CRIME PRE-REGISTERED CHECKLIST  (VERBATIM, criminal_record/crime_table.csv;
#   crime | observable | CMFGEN vs Lumina | grade | repair component)
# ---------------------------------------------------------------------------
#  C1  deep FUV collapse   mc_J(918-1290,s0)  83.60 vs 1.922 (-1.54dex)   CONVICTED  i+iii(+Te)
#  C2  dead Fe recomb grad f(FeIV) slope       +5.09 vs +0.65 dex          CONVICTED  i+iii
#  C3  deep T_e cold       T_e(s0)             18760 vs 13120 K            CONVICTED  i+ii+iii(+NITER)
#  C4  deep u low          u_bol(s0)           694.8 vs 400.2 (ForkB->264) CONVICTED  ii(+hotTe)  [ForkB-alone FAILS]
#  C5  EUV starvation      mc_J(300-450,s0)    2.05e-2 vs 4.60e-6 (-3.65)  CONVICTED  i+ii(selective iii)
#  C6  phot FUV+over-ion   mc_J(s8)+f(FeIV,s8) 0.022 vs 0.461              CONVICTED  iii(+S topstage SE)
#  C7  Co III Gph deficit  Gph(Co III)         22x low (twin 17.26x)       AGGRAVATED Gph-SE-weight (SEPARATE) -> residual
#  C8  unfilled deep valley mc_J(1650-2100,s0) mc/CMFGEN -1.55dex          CONVICTED-down  deep-field amp (ii+hotTe) NOT iii-clamp
#  C9  T_rad pin uniform   T_rad all shells    uniq=1 @10470K              INDEP-instrument -> residual (no change)
#  C10 Co rate ~10x low    f(Co IV) twin       5-20x under-ion, T+J pinned INDEP-of-kp_emiss -> residual (no change)
#  C11 deep Co IV 1500 pile mc_J(1290-2000,s0) Co IV=80.9% of deep emit-E  CONVICTED  i+iii
#  C12 split-field mc/cs   Gph(mc_J)/cool(cs_J) band 7-77x divergence      MECHANISM  i (unifies once kp_emiss thermal)
#  H1  fluor gap UV 51v23  emergent UV frac    23.3-23.8% vs 42.9-51.6%    CONVICTED  i+iii (=C6, one epoch removed)
#  H2  Kromer S III UV     S III emission      52% of emergent UV          CONVICTED  iii(+S topstage SE)
#  (H3 ii+iii, H4 ii+iii, H5 TOPSTAGE-IV; H6-H10 = OTHER repairs, pre-register NO change.)
#
# CURE GATES (KP_EMISS_REPAIR_DESIGN §4; quantity @ location -> gate):
#  K1 Co IV emit-share s0 : 91% -> <=30% ; mc/cs @1526A -> ~1            (iii+i)
#  K2 mc_J(918-1290,s0)   : 1.9e-6 -> ~2e-4                              (iii+ii)
#  K3 mc_J(300-450,s0)    : recovers AND stays NON-thermal (selective iii)(ii)
#  K4 u_bol(s0)           : 400/264 -> climbs toward 695 (needs ii+hotTe+NITER)
#  K5 T_e(s0)             : 13120 -> past 15-16 kK toward 18277
#  K6 f(FeIV) slope s0->s8: +0.65 -> +5.09 (field un-flattened)         (i+iii)
#  K7 S III emit-share s8 : collapses WITHOUT F4 ; UV 51% -> ~23%       (i round-2 pops + iii)
#  K8 ff/fb exit share s0 : 0.02% -> RISES with KPKT_FB_MULTI (measure) (ii)
#  G-FUNNEL (LATER, Fork B OFF): mc/cs @1526A s0 <= ~2 => iii subsumes Fork B
#
# HEADLINE GATES (this run; the driver reads these, do NOT move goalposts):
#   PASS/direction:
#     * deep u(s0) recovers from 264 toward 695  (>=400 = direction PASS ; >=550 strong)
#     * T_e(s0) > 15000 K
#     * FUV(918-1290, s0) >= +0.5 dex vs B-run 5.81e-6
#     * EUV(300-450, s0) NOT collapsed (>= 1e-12)     [selective-iii guard, crime C5]
#     * funnel dead: mc/cs @1450-1650 <= 3
#     * photospheric S III FUV share falls from 84-88% (event check) WITHOUT any F4 gate
#     * f(FeIV) s8 <= 0.25  (toward 0.022, from 0.46)
#     * slope s0->s8 >= +0.5 (toward +2.42)
#   RESIDUALS -- PRE-REGISTERED as NO-CHANGE (do NOT claim improvement):
#     * Co f(IV) twin-style rate deficit (C7/C10)  -- needs SE-weighted Gph, outside i/ii/iii
#     * T_rad pin 10470 uniq (C9)                  -- separate estimator fix
#     * MC blue-tilt / far-outer hot-band (H9/H10) -- separate physics, untouched
#   HARD KILL (KP_EMISS_REPAIR_DESIGN §5): if Co IV emit-share s0 does NOT fall < 30%
#     while EUV stays non-thermal -> the thick-zone source thesis is wrong; escalate to
#     a transport-side deep continuum-optical-depth (tau_cont>=1) audit.
#   Wiring (check FIRST on any null):
#     [KPR] LUMINA_KPEMISS_REPAIR=1 knobs: SE_POPS=1 KPKT_FB_MULTI=1 BSRC_TAU=0.130 ... ARMED  (once, init)
#     [STAGE4] LUMINA_NLTE_STAGE4=1: NLTE slots ... promoted (III,IV) pairs                    (once, init)
#     [KPR] it NN: bteq_exits=<n> cdf_exits=<m>          (once per iter; n==0 with the gate ON
#            prints a *** WARNING *** = wiring no-op, NOT a physics null. Expect n in the deep
#            shells only, W>0.13 = s0-s2.)
#     [BSRC] it NN: thermalized=<N> (mode=1)             (Fork B belt-and-suspenders, still on)
#     [FB-MULTI] it NN: fb_emit=<N> ...                  (B2 continuum floor active)
#     (0 info=199 NLTE-FALLBACK lines expected -> A3 dropped the singular Ti pair)
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr binary as ./lumina_cuda for run_coevolve_s01.sh --------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[kpr] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[kpr] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

# --- B-run (a10_kx_gphall MODE=all) environment, verbatim ----------------------
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_EVENT_LOG_ESCATTER=1
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_GPH_ALLLEVEL=1
# B-run had LUMINA_NLTE_BK_CEIL ABSENT (b_k cap OFF). Clear any inherited value.
unset LUMINA_NLTE_BK_CEIL
# Keep the #33-transplant / thermostat / other-repair gates OFF: T_e and the field
# are FREE to respond (the whole point). unset even if inherited from the shell.
unset LUMINA_GPH_JTABLE LUMINA_TE_TABLE LUMINA_TINNER_COLOR \
      LUMINA_MACROATOM_BF LUMINA_LINE_THERM

# --- STAGE4-ROUND2 (Part A) -----------------------------------------------------
export LUMINA_NLTE_STAGE4=1          # round-2 semantics (A1 depth-gate default 0.13,
                                     #   A2 top-ion clamp default ON, A3 Ti dropped)
export LUMINA_STAGE4_GPH_WTHR=0.13   # A1 depth gate: NLTE-weight III combs only where W>this
export LUMINA_STAGE4_BK_CAP=1000     # A1 per-level b_k cap inside the gate

# --- KPEMISS_REPAIR (Part B) master gate + knobs --------------------------------
export LUMINA_KPEMISS_REPAIR=1       # master gate (off => byte-identical)
export LUMINA_KPEMISS_SE_POPS=1      # B1 SE/NLTE pops into kp_emiss (plasma.c:2117)
export LUMINA_KPKT_FB_MULTI=1        # B2 real per-edge fb recombination continuum floor
export LUMINA_KPEMISS_BSRC_TAU=0.13  # B3 B(T_e) k-packet exit where W>this (deep only)

# --- Fork B per-line thermal source (shipped WITH the repair; A/B-off is a later arm)
export LUMINA_LINE_BSRC=1
export LUMINA_LINE_BSRC_MODE=1

TAG="a10_kx_kpr"
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
echo "[kpr] verify wiring: grep -E '\[KPR\]|\[STAGE4\]|\[FB-MULTI\]|\[BSRC\]' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr] verify A3 (Ti):  grep -c 'info=199' logs/coevolve_consume_${TAG}/stderr.log   # expect 0"
