#!/bin/bash
#SBATCH --job-name=a10_kx_kpr8
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# KPR8 = KPR7 BAND-SCOPED so the case-B/OTS + BSRC_PHOT remove ONLY the
#        pathological photospheric field, PRESERVING the legitimate 405-2000A
#        excited-level continuum (over_recomb_s4: the kpr6/kpr7 repair suppressed
#        Gph(FeIII) 30x at s4 because it thermalized the 405-2000A continuum that
#        CMFGEN uses for 96% of its Fe III photoionization).
# ============================================================================
# SYMPTOM-SCOPING attempt, NOT a root-cause fix. The root cause of why Lumina's
# self-consistent photospheric field does not converge to CMFGEN's band-shape is a
# REGISTERED OPEN COLD-CASE. This run's narrow job: stop the repair from over-
# suppressing the LEGITIMATE excited-level continuum, and see if the photospheric
# ionization lands. A pre-registered STOP rule (below) guards against whack-a-mole.
#
# Exact clone of scripts/sbatch_kpr7.sh (STAGE4-R2 + KPEMISS_REPAIR B1/B2/B3 SRC=2
# + DB_FB + COOLGUARD + Fork B + PUMP_FIELD alpha=1 + FLOORM + PUMPF + Prong A
# BSRC_PHOT + Prong B FB_OTS + PHYSICAL tau_bf graded gate), with TWO independent
# env-gated tightenings (both default = kpr7 behavior; the tau_bf DEPTH gate is
# UNCHANGED -- only the FREQUENCY/CHANNEL scope of what gets thermalized changes):
#
#   PRONG B (FB_OTS) NUMIN TIGHTEN:  912A(3.29e15) -> 405A(7.40e15).
#     over_recomb_s4 found NUMIN=912A redirects ALL fb ground edges from 912A
#     bluewards, thermalizing the 405-912A band. LOAD-BEARING PHYSICS (verified in
#     source, plasma.c:2327-2338): the FB-MULTI edge is find_ioniz_energy(Z,stage-1)
#     = the recombination-product ION's GROUND ionization threshold, ONE edge per
#     (Z,stage) continuum -- destination-BLIND (the machinery does NOT model recomb-
#     to-excited-levels; every edge is a ground edge). So the cut does NOT separate
#     "ground vs excited DESTINATION" (that does not exist here); it separates high-IE
#     ion GROUND edges (EUV, <=405A: Fe III 404.5A/7.4115e15, S III 356A, Si III 370A,
#     Co III 370A, Ni III 352A) -- the pathological Fe IV->III over-emission the design
#     targets -- from low-IE ion ground edges (405-912A: Ti III 451A/6.65e15, Fe II
#     766A/3.91e15, S II 531A, Si II 758A) which carry legitimate FUV/EUV continuum.
#     NUMIN=7.40e15 captures Fe III (7.4115e15 > 7.40e15, 0.15% margin) and spares
#     the 405-912A band (nearest spared edge Ti III at 6.65e15, ~10% gap => robust).
#
#   PRONG A (BSRC_PHOT) CROSS-ION SCOPE:  LUMINA_KPEMISS_BSRC_PHOT_XION=1 (NEW gate).
#     VERIFIED in source: BSRC_PHOT is ALREADY line-only -- the k-packet -2 ff and
#     -3 fb continuum branches RETURN before the -4 B(Te) exit (lumina_cuda.cu k-packet
#     ladder), so NO continuum is thermalized. The 912-2000A FUV deficit (12.6x down)
#     is therefore SAME-ION FUV LINE emission thermalized to Wien-faint B(Te~12kK).
#     Cross-ion tagging IS feasible: n_macro_levels==atom.n_levels, so level_Z/level_ion
#     give a re-excited macro-level's species. When XION=1 the phot-tier -4 exit draws
#     the re-excitation CDF, and thermalizes ONLY the CROSS-ION re-excite (emitter
#     species != the macro-atom's ACTIVATING species = the A2 S III attractor signature);
#     SAME-ION cascades keep the legitimate radiative walk => their 912-2000A FUV LINE
#     emission survives. Default OFF => phot tier redirects ALL (== kpr7).
#
# Binary: lumina_cuda.withKpr8 (= withKpr7 source + the XION gate; FB_OTS_NUMIN default
#         3.29e15, BSRC_PHOT_XION default 0 => with the two new card values UNSET the
#         behavior is kpr7; with BSRC_PHOT=0 && FB_OTS=0 it is byte-identical to withKpr5
#         -- the new act_zi capture + XION block draw ZERO uniforms off-path).
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES + STOP RULE  (do NOT move)   Yardstick = CMFGEN toy06 @19.48d.
# ---------------------------------------------------------------------------
#  PASS (the photosphere lands) -- a MONOTONIC outward-recombining Fe IV profile:
#    * f(FeIV, s4) in 0.6-0.8       (CMFGEN 0.727; kpr7 over-recombined to 0.116)
#    * f(FeIV, s6) in 0.03-0.15     (CMFGEN 0.069)
#    * f(FeIV, s8) in 0.01-0.06     (CMFGEN 0.022) -- MONOTONIC s4>s6>s8
#                                     (kpr7 was NON-monotonic: s8 > s6)
#    * Gph(FeIII, s4) within 3x of CMFGEN 0.170 /s  (kpr7: 30x deficit = 0.00561)
#  DEEP gains HELD (Prong A/B act only on thick-tier PHOT/mid shells; deep untouched):
#    * FUV(918-1290, s0)  >= 1.5e-4
#    * FUV gradient slope  +2.0 .. +3.0
#    * funnel dead: mc/cs @1450-1650  <= 3x
#    * deep f(FeIV, s0)  >= 0.95
#  T_e HELD:  T_e(s8) in 10.0-11.4 kK  (CMFGEN 10.383 kK)
#  RESIDUALS -- PRE-REGISTERED NO-CHANGE (do NOT claim improvement):
#    * Co twin f(IV) rate deficit (~10x) unchanged; MC blue-tilt; T_rad pin (C9).
#
#  STOP RULE (pre-registered): if this SINGLE scoped run does NOT achieve the
#    monotonic photospheric profile above, DO NOT iterate further on scoping -- the
#    result is reported AS-IS, the campaign consolidates the DEEP WIN and proceeds to
#    the Phase-2 spectrum; the photospheric band-shape non-convergence is the
#    registered COLD-CASE for post-spectrum re-investigation. (Guard vs whack-a-mole.)
#
#  WIRING (check FIRST on any null):
#    [OTS-TAUBF] criterion = tau_bf ... MODE=2 (graded ...) TAU_THR=1.00        (init)
#    [FB_OTS] ... NUMIN=7.400e+15 Hz (405 A) ...                                (init)
#    [BSRC_PHOT] ... phot tier ...                                             (init)
#    [BSRC_PHOT_XION] ... CROSS-ION re-excites only ... map uploaded           (init)
#    [KPR] itNN: ... phot_bteq=<n> fb_ots_redirects=<m> xion_kept=<k>   (per iter;
#         phot_bteq>0 AND xion_kept>0 after it 0 => XION is firing and KEEPING same-ion
#         cascades; if xion_kept==0 the S III attractor is 100% of phot re-excites, or
#         the map/gate mis-armed -> check the [BSRC_PHOT_XION] init banner.)
#  FALSIFIERS:
#    (i)  if f(FeIV,s8) stays > 0.5 while both prongs fire, the field is not the whole
#         lock -> re-open ionization balance (alpha/DR), not transport.
#    (ii) if Gph(FeIII,s4) is now OVER 3x above 0.170 (over-restored) OR the deep FUV
#         gate breaks, the scope is too loose -> the STOP rule still applies (report,
#         do not iterate).
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr8 binary as ./lumina_cuda for run_coevolve_s01.sh --------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr8
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[kpr8] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[kpr8] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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
export LUMINA_KPEMISS_BSRC_SRC=2     # B3 refinement: deep -4 exit nu ~ chi_line(nu)*B_nu(Te)

# --- KPR2: the principled thermal-ledger fix (inherited) ------------------------
export LUMINA_RADEQ_DB_FB=1          # simul_r1 bf cooling = detailed-balance partner of H_photo
export LUMINA_KPEMISS_COOLGUARD=1    # skip B3+FB-MULTI thermal exits where f(FeV)>0.5

# --- Fork B per-line thermal source (shipped WITH the repair; A/B-off is a later arm)
export LUMINA_LINE_BSRC=1
export LUMINA_LINE_BSRC_MODE=1

# --- PHASE-1a: unify the radeq line-pump field onto the Gph alpha-blend (inherited)
export LUMINA_RADEQ_PUMP_FIELD=1     # simul_line_term Jb = alpha*mc_J + (1-alpha)*cs_J

# --- PHASE-1 FINAL LEVER: floor policy + zero-count pump fallback (inherited) -----
export LUMINA_NLTE_FLOOR_MODE=1      # FIX-1: LTE-relative floor + b_k cap (was flat 1e-30)
export LUMINA_NLTE_FLOOR_BKMAX=1000  # FIX-1: departure cap b_k<=1000
export LUMINA_RADEQ_PUMP_FALLBACK=1  # FIX-2: zero-count mc bins -> B_nu(Te), not cs_J

# --- PHOTOSPHERIC EUV REPAIR: Prong A + Prong B composed, PHYSICAL tau_bf gate ----
export LUMINA_KPEMISS_BSRC_PHOT=1          # Prong A: extend -4 B(Te) exit to tau_bf-
                                           #   qualified phot shells
export LUMINA_KPEMISS_BSRC_PHOT_SRC=1      # phot-tier -4 exit = pure Planck(Te) (Wien-dead EUV);
                                           #   deep tier keeps SRC=2 (BSRC_SRC above)
export LUMINA_KPEMISS_FB_OTS=1             # Prong B: case-B/OTS redirect of EUV ground-edge
                                           #   (-3) fb -> B(Te) draw where tau_bf-qualified
# --- KPR8 SCOPING #1: Prong-B NUMIN tightened 912A -> 405A (spare the 405-912A band)
export LUMINA_KPEMISS_FB_OTS_NUMIN=7.40e15 # 405A: redirect ONLY Fe III(404.5A/7.4115e15) &
                                           #   bluer IGE EUV ground edges; SPARE 405-912A
                                           #   low-IE edges (Ti III 451A, Fe II 766A, ...).
                                           #   (was 3.29e15=912A in kpr7, which killed 405-912A)
# --- KPR8 SCOPING #2: Prong-A cross-ion scope (thermalize ONLY the S III attractor)
export LUMINA_KPEMISS_BSRC_PHOT_XION=1     # phot -4 exit thermalizes CROSS-ION re-excites
                                           #   only; same-ion cascades KEPT => preserve the
                                           #   912-2000A FUV LINE emission (was: redirect ALL)
# --- the OVERFITTING-KILLER: physical case-A/B criterion (replaces W>WFLOOR) ------
export LUMINA_KPEMISS_OTS_MODE=2           # 2=GRADED P(OTS)=1-exp(-tau_bf) (default); 1=binary
export LUMINA_KPEMISS_OTS_TAU=1.0          # binary threshold = physical tau=1 boundary (sens. only)
# LUMINA_KPEMISS_BSRC_PHOT_WFLOOR intentionally UNSET => W-floor guard OFF => pure tau_bf.

TAG="a10_kx_kpr8"
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
echo "[kpr8] verify OTS CRITERION: grep -E '\[OTS-TAUBF\]' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr8] verify PRONG-B NUMIN:  grep -E '\[FB_OTS\]' logs/coevolve_consume_${TAG}/stdout.log  # NUMIN=7.400e+15 Hz (405 A)"
echo "[kpr8] verify PRONG-A XION:   grep -E '\[BSRC_PHOT_XION\]' logs/coevolve_consume_${TAG}/stdout.log  # map uploaded"
echo "[kpr8] verify wiring:         grep -E '\[KPR\] it' logs/coevolve_consume_${TAG}/stdout.log  # phot_bteq>0 AND xion_kept>0 after it0"
echo "[kpr8] verify gates:          grep -E '\[OTS-TAUBF\]|\[BSRC_PHOT\]|\[BSRC_PHOT_XION\]|\[FB_OTS\]|\[KPR\]|\[FLOORM\]|\[PUMPF\]|\[STAGE4\]|\[FB-MULTI\]|\[DBFB\]|\[SIMUL\] done' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr8] PASS: MONOTONIC front f(FeIV) s4 0.6-0.8, s6 0.03-0.15, s8 0.01-0.06 (s4>s6>s8); Gph(FeIII,s4) within 3x of 0.170; DEEP FUV(s0)>=1.5e-4, slope +2.0..+3.0, funnel<=3x, f(IV,s0)>=0.95; T_e(s8) 10.0-11.4kK; residuals unchanged"
echo "[kpr8] STOP RULE: if the monotonic front is NOT achieved, report AS-IS, consolidate deep win, proceed to Phase-2 spectrum. Do NOT iterate on scoping."
