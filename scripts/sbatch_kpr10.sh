#!/bin/bash
#SBATCH --job-name=a10_kx_kpr10
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# KPR10 = KPR9 + LUMINA_NLTE_METACOLL_MODE=2 (COMPLETE the metastable drain to
#         match CMFGEN's collision TOPOLOGY) + retire the STAGE4 b_k cap.
# ============================================================================
# WHY (verified DIRECTLY from CMFGEN's FeIII_COL_DATA, Zhang 1996,
#   /gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/col_data):
#   KPR9's METACOLL (MODE=1) drains each drainless-metastable to GROUND ONLY with
#   Axelrod Omega=1. This UNDER-drains: Fe III level 17 stays b_k~40 at s8, so the
#   b_k cap stays load-bearing. CMFGEN instead drains every metastable to ALL LOWER
#   LEVELS of the same ion, with the documented forbidden floor Omega=0.1 ("Value
#   for OMEGA if f=0: 0.1", col_data header). So ground-only-Omega=1 is structurally
#   wrong; it should be all-lower-levels with the CMFGEN forbidden floor per channel.
#
# THE CHANGE (LUMINA_NLTE_METACOLL_MODE=2, parameter-free approximation):
#   For each drainless-metastable m, add a collisional de-excitation channel to
#   EVERY lower level l (E_l < E_m) of the same ion, each Omega=METACOLL_OMEGA(0.1):
#     C_down(m->l) = n_e*8.629e-6/(g_m*sqrt(Te))*Omega
#     C_up  (l->m) = n_e*8.629e-6/(g_l*sqrt(Te))*Omega*exp(-dE_ml/kTe)
#   Detailed balance exact PER CHANNEL: C_up/C_down = (g_m/g_l)*exp(-dE/kTe). This
#   approximates CMFGEN's per-transition Zhang96 Omega (to be imported later as the
#   fidelity endpoint) with its forbidden floor + all-lower topology.
#   MODE=1 (default) is byte-identical to kpr9 (ground-only, Omega=1).
#
# Binary: lumina_cuda.withKpr10 (= withKpr9 source + the METACOLL_MODE dispatch).
#   MODE=1 default => rate matrix byte-identical to withKpr9; MODE=2 activates the
#   all-lower-levels drain. Host-side matrix arithmetic only; no RNG touched.
#
# ---------------------------------------------------------------------------
# !! OPEN DECISION FOR THE DRIVER (env-chain audit -- decide BEFORE submit) !!
#   The pre-registered gate compares against "the cap-off AUD-2 run" (b_k -> 2214).
#   That AUD-2 baseline retired BOTH b_k ceilings: STAGE4_BK_CAP=0 AND
#   `unset LUMINA_NLTE_FLOOR_MODE LUMINA_NLTE_FLOOR_BKMAX`. This card, per the
#   mission's explicit 3-delta spec, sets ONLY STAGE4_BK_CAP=0 and KEEPS kpr9's
#   FLOOR_MODE=1 + FLOOR_BKMAX=1000. But FLOOR_BKMAX=1000 is a SECOND, in-series
#   b_k cap: it clamps the STORED nlte_level_populations at b_k<=1000 on writeback
#   (plasma.c:5536-5537 [FLOORM] "resolved: cap at b_k<=BKMAX"). Both the Gph NLTE
#   weighting (plasma.c:5864 reads the STORED pops, then STAGE4_BK_CAP re-caps) and
#   the [METACOLL-PROBE] (reads the STORED pops) therefore CANNOT exceed b_k~1000
#   while FLOOR_BKMAX=1000 remains. Consequence:
#     * PASS (b_k O(1-3)) is still cleanly observable.
#     * NULL/PARTIAL magnitude is MASKED at 1000 -- the probe can never show the
#       2214 runaway the gate references, so NULL vs PARTIAL is muddied and the cap
#       is not FULLY retired.
#   To make this a genuine cap-retirement test vs AUD-2, ALSO uncomment the FLOOR
#   retire line below. Left commented to honor the mission's explicit env list.
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES  (do NOT move)          Yardstick = CMFGEN toy06 @19.48d.
# ---------------------------------------------------------------------------
#  PASS (METACOLL complete, cap retirable) = with BK_CAP=0: Fe III level-17 b_k at
#    s8 stays O(1-3) (NOT runaway to 2214 like the cap-off AUD-2 run); f(FeIV) s6/s8
#    recombine toward CMFGEN (s6 0.03-0.15 [0.069], s8 0.01-0.06 [0.022]); iron-group
#    over-ionization drops toward <3x; DEEP WIN HELD (FUV s0>=1.5e-4, slope>=+2.0,
#    funnel<=3x, deep f(FeIV) s0>=0.95, u(s0)>=450); T_e(s8) toward 10.4kK.
#  PARTIAL = b_k drops well below the 2214 cap-off runaway but not to ~1, and
#    f(FeIV) improves but not to CMFGEN => the collisional completion helps but a
#    residual (likely the super-thermal photospheric FIELD pumping the cascade)
#    remains => report the residual for the next front.
#  NULL = b_k still runs away (~hundreds+) with cap off => all-lower Omega=0.1 still
#    insufficient => escalate to the real Zhang96 Omega import.
#  WIRING (check FIRST on any null):
#    [METACOLL] init mode -> "  [METACOLL] mode=2 Omega=0.10: drainless-metastable
#                              -> all-lower coupling; FeIII=2 ... (total=563)"
#    [METACOLL-PROBE] per iter -> "  [METACOLL-PROBE] FeIII lvl17 b_k/gnd: s0=.. ..
#                                   sLAST=.. (Te sLAST=..K)"  (MODE=2 drain should
#                                   collapse the deep photospheric b_k17 toward O(1-3);
#                                   AUD-2 cap-off pinned ~2214.)
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr10 binary as ./lumina_cuda for run_coevolve_s01.sh -------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr10
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[kpr10] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[kpr10] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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

# ============================================================================
# KPR10 ROOT-FIX COMPLETION: the metastable collisional drain, now all-lower-
#   levels with the CMFGEN forbidden floor (matches FeIII_COL_DATA topology).
# ============================================================================
export LUMINA_NLTE_METASTABLE_COLL=1   # arm the drainless-metastable drain pass
export LUMINA_NLTE_METACOLL_MODE=2     # 2 = ALL-lower-levels (was ground-only in kpr9)
export LUMINA_NLTE_METACOLL_OMEGA=0.1  # CMFGEN's f=0 forbidden floor (col_data header)

# --- STAGE4-ROUND2 (Part A) -----------------------------------------------------
export LUMINA_NLTE_STAGE4=1          # round-2 semantics (A1 depth-gate default 0.13,
                                     #   A2 top-ion clamp default ON, A3 Ti dropped)
export LUMINA_STAGE4_GPH_WTHR=0.13   # A1 depth gate: NLTE-weight III combs only where W>this
export LUMINA_STAGE4_BK_CAP=0        # RETIRE the STAGE4 per-level b_k cap (=0 disables the
                                     #   clamp; unset would leave the 1000 default). TEST:
                                     #   does the completed all-lower drain make it unnecessary?
# --- OPEN DECISION (see header): to fully match AUD-2 cap-off (b_k free to 2214+),
#     ALSO retire the FLOOR writeback cap. Left commented per the mission's env list.
unset LUMINA_NLTE_FLOOR_MODE LUMINA_NLTE_FLOOR_BKMAX  # [driver] enabled: genuine both-cap-off test vs AUD-2

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
export LUMINA_NLTE_FLOOR_BKMAX=1000  # FIX-1: departure cap b_k<=1000  (SEE OPEN DECISION)
export LUMINA_RADEQ_PUMP_FALLBACK=1  # FIX-2: zero-count mc bins -> B_nu(Te), not cs_J

# --- PHOTOSPHERIC EUV REPAIR: Prong A + Prong B composed, PHYSICAL tau_bf gate ----
export LUMINA_KPEMISS_BSRC_PHOT=1          # Prong A: extend -4 B(Te) exit to tau_bf-
                                           #   qualified phot shells
export LUMINA_KPEMISS_BSRC_PHOT_SRC=1      # phot-tier -4 exit = pure Planck(Te) (Wien-dead EUV);
                                           #   deep tier keeps SRC=2 (BSRC_SRC above)
export LUMINA_KPEMISS_FB_OTS=1             # Prong B: case-B/OTS redirect of EUV ground-edge
                                           #   (-3) fb -> B(Te) draw where tau_bf-qualified
export LUMINA_KPEMISS_FB_OTS_NUMIN=7.40e15 # 405A: redirect ONLY Fe III & bluer IGE EUV ground
                                           #   edges; SPARE the 405-912A low-IE band.
export LUMINA_KPEMISS_BSRC_PHOT_XION=1     # phot -4 exit thermalizes CROSS-ION re-excites only;
                                           #   same-ion cascades KEPT (preserve 912-2000A FUV)
export LUMINA_KPEMISS_OTS_MODE=2           # 2=GRADED P(OTS)=1-exp(-tau_bf) (default); 1=binary
export LUMINA_KPEMISS_OTS_TAU=1.0          # binary threshold = physical tau=1 boundary (sens. only)
# LUMINA_KPEMISS_BSRC_PHOT_WFLOOR intentionally UNSET => W-floor guard OFF => pure tau_bf.

TAG="a10_kx_kpr10"
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
echo "[kpr10] verify METACOLL init:  grep -E '\[METACOLL\]' logs/coevolve_consume_${TAG}/stdout.log  # mode=2 Omega=0.10 all-lower; FeIII=2 ... (total=563)"
echo "[kpr10] verify METACOLL probe: grep -E '\[METACOLL-PROBE\]' logs/coevolve_consume_${TAG}/stdout.log  # FeIII lvl17 b_k/gnd sLAST should collapse toward O(1-3) (AUD-2 cap-off pinned ~2214)"
echo "[kpr10] verify gates:          grep -E '\[METACOLL\]|\[METACOLL-PROBE\]|\[OTS-TAUBF\]|\[BSRC_PHOT\]|\[BSRC_PHOT_XION\]|\[FB_OTS\]|\[KPR\]|\[FLOORM\]|\[PUMPF\]|\[STAGE4\]|\[FB-MULTI\]|\[DBFB\]|\[SIMUL\] done' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr10] PASS: FeIII lvl17 b_k/gnd(sLAST) O(1-3); f(FeIV) s6 0.03-0.15, s8 0.01-0.06; IGE over-ion Lumina/CMFGEN <3x; DEEP FUV(s0)>=1.5e-4, slope>=+2.0, funnel<=3x, f(IV,s0)>=0.95, u(s0)>=450; T_e(s8) toward 10.4kK"
