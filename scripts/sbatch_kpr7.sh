#!/bin/bash
#SBATCH --job-name=a10_kx_kpr7
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# KPR7 = KPR6 with the case-B/OTS threshold CONSTANT killed (PHYSICAL tau_bf)
# ============================================================================
# Exact clone of scripts/sbatch_kpr6.sh (STAGE4-R2 + KPEMISS_REPAIR B1/B2/B3
# SRC=2 + DB_FB + COOLGUARD + Fork B + PUMP_FIELD alpha=1 + FLOORM + PUMPF +
# Prong A BSRC_PHOT + Prong B FB_OTS), with ONE change: the "continuum-thick"
# decision that gated BOTH prongs is no longer the hard constant W>WFLOOR(0.02) --
# a case-specific dilution-factor PROXY that overfits toy06 and would fail on other
# SNe/epochs (user directive: forbidden). It is now the PHYSICAL per-cell optical
# depth of each shell to its OWN ground-edge recombination continuum:
#
#     tau_bf(edge) = chi_bf(nu0) * L_shell                                (per cell)
#       chi_bf(nu0) = Sigma_l n_level * sigma_bf(CMFGEN)  @ Fe III 404A ground edge
#                     = the SAME population-weighted bf opacity transport re-absorbs
#                       on (compute_bf_opacity, plasma.c:3784; d_bf_get_chi, cuda.cu)
#       L_shell     = r_outer - r_inner  (comoving shell width, cm)
#
# Case B (on-the-spot re-absorption) applies where the shell is optically thick to
# that edge (tau_bf >~ 1); case A (free-stream) where tau_bf < 1. CMFGEN's SMOOTH
# recombination front emerges because tau_bf crosses 1 at the physically-correct
# shell -- a DIFFERENT shell for different SNe/epochs, with NO constant to tune.
#
#   OTS_MODE=2 (GRADED, default) -- a photon is re-absorbed with the physical escape
#     probability P(OTS)=1-exp(-tau_bf), drawn per-event on the device. Shells with
#     tau_bf~0.3-3 get PARTIAL OTS, reproducing CMFGEN's smooth s4-s8 front instead
#     of the kpr6 binary cliff (which over-corrected: recomb front ~2 shells too
#     deep, deep field over-bright). MODE=1 = binary tau>=TAU (diagnostic only).
#   OTS_TAU=1.0 -- the PHYSICAL case-A/B boundary tau=1, NOT a fitted proxy (used
#     only in binary mode; exposed for sensitivity checks, default 1.0).
#   The W>WFLOOR env is RETIRED as the criterion; kept only as an OPTIONAL hard
#     floor guard, DEFAULT OFF (WFLOOR-guard=0.0) => pure tau_bf physics decides.
#
# ONE physical criterion now gates BOTH prongs (no constants):
#   PRONG A (LUMINA_KPEMISS_BSRC_PHOT) -- the -4 B(Te) k-packet exit extends below
#     the deep boundary to any tau_bf-qualified phot shell (was W>WFLOOR). Graded:
#     phot k-packets take -4 with P(OTS); phot shells use PURE PLANCK (SRC=1).
#   PRONG B (LUMINA_KPEMISS_FB_OTS) -- an EUV GROUND-edge fb (-3) recomb photon
#     (comoving nu0 > 912A) in a tau_bf-qualified cell is redirected to the -4 B(Te)
#     draw with P(OTS). FUV/optical edges (>912A) untouched (protects deep FUV).
#
# Binary: lumina_cuda.withKpr7 (= withKpr6 source + tau_bf criterion; BSRC_PHOT=0 &&
#         FB_OTS=0 => byte-identical to withKpr5 -- both masters gate every new draw).
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES  (do NOT move)   Yardstick = CMFGEN toy06 @19.48d at Lumina v.
# ---------------------------------------------------------------------------
#  PRIMARY -- the SMOOTH recombination front (the kpr6 over-correction, cured):
#    * f(FeIV, s4) in 0.5-0.8      (kpr6 cliff: 0.116, ~2 shells too deep; CMFGEN 0.727)
#    * f(FeIV, s6) in 0.03-0.15    (a graded transition, NOT a binary cliff)
#    * f(FeIV, s8) in 0.01-0.05    (toward CMFGEN 0.022)
#  DEEP not over-brightened (pull back from the kpr6 overshoot u=854, slope +4.44):
#    * u_bol(s0) in 500-750        (CMFGEN 695; kpr6 over-bright 854)
#    * FUV gradient slope in +2.0..+3.0   (CMFGEN +2.42; kpr6 too steep +4.44)
#    * FUV(918-1290, s0) in 1.5e-4..2.5e-4
#    * funnel dead: mc/cs @1450-1650 <= 3x
#  T_e HELD (keep the kpr6 gain -- mid/outer matched CMFGEN to <330K):
#    * T_e(mid), T_e(outer) within +-500 K of CMFGEN
#  RESIDUALS -- PRE-REGISTERED NO-CHANGE (do NOT claim improvement):
#    * Co twin f(IV) rate deficit (~10x) unchanged; MC blue-tilt; T_rad pin (C9).
#  WIRING (check FIRST on any null):
#    [OTS-TAUBF] criterion = tau_bf ... MODE=2 (graded ...) TAU_THR=1.00        (init)
#    [OTS-TAUBF] Fe III ground edge nu0=~7.4e15 Hz (~404 A)                     (init)
#    [OTS-TAUBF] itNN mode=2 thr=1.00: s2[tau=.. P=.. ..] s4[..] s6[..] s8[..]  (per iter)
#         -> tau_bf must DECREASE outward and cross ~1 across s2->s8; P(OTS) must
#            grade 1->0 across the same shells (the case-A/B front landing physically).
#    [BSRC_PHOT] ... tau_bf-qualified ...   [FB_OTS] ... where tau_bf-qualified ... (init)
#    [KPR] itNN: ... phot_bteq=<n> fb_ots_redirects=<m>   (per iter; both > 0 after it 0)
#    [FLOORM]/[PUMPF]/[FB-MULTI]/[DBFB]/[SIMUL] done       (inherited kpr5/kpr6 wiring)
#  FALSIFIERS:
#    (i)  if tau_bf never crosses ~1 across s2->s8 (all >>1 or all <<1), the edge/grid
#         is wrong -> check nu0 (~404A) and chi_bf population BEFORE any physics claim.
#    (ii) if the front is still a cliff with MODE=2, the graded draw is not firing ->
#         check per-event P(OTS) draws (fb_ots_redirects should be < thick emissions).
#    (iii) if f(FeIV,s8) stays > 0.5 while both prongs fire, the field is not the whole
#          lock -> re-open ionization balance (alpha/DR), not transport.
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr7 binary as ./lumina_cuda for run_coevolve_s01.sh --------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr7
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[kpr7] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[kpr7] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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
                                           #   qualified phot shells (was W>WFLOOR)
export LUMINA_KPEMISS_BSRC_PHOT_SRC=1      # phot-tier -4 exit = pure Planck(Te) (Wien-dead EUV);
                                           #   deep tier keeps SRC=2 (BSRC_SRC above)
export LUMINA_KPEMISS_FB_OTS=1             # Prong B: case-B/OTS redirect of EUV ground-edge
                                           #   (-3) fb -> B(Te) draw where tau_bf-qualified
export LUMINA_KPEMISS_FB_OTS_NUMIN=3.29e15 # EUV cutoff = 912A; FUV/optical edges keep raw fb
# --- the OVERFITTING-KILLER: physical case-A/B criterion (replaces W>WFLOOR) ------
export LUMINA_KPEMISS_OTS_MODE=2           # 2=GRADED P(OTS)=1-exp(-tau_bf) (default); 1=binary
export LUMINA_KPEMISS_OTS_TAU=1.0          # binary threshold = physical tau=1 boundary (sens. only)
# LUMINA_KPEMISS_BSRC_PHOT_WFLOOR intentionally UNSET => W-floor guard OFF => pure tau_bf.

TAG="a10_kx_kpr7"
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
echo "[kpr7] verify OTS CRITERION: grep -E '\[OTS-TAUBF\]' logs/coevolve_consume_${TAG}/stdout.log  # init banner + nu0 + per-iter tau_bf/P(OTS) at s2/s4/s6/s8"
echo "[kpr7] verify PRONG-A wiring: grep -E '\[BSRC_PHOT\]|phot_bteq' logs/coevolve_consume_${TAG}/stdout.log  # init + per-iter phot_bteq>0"
echo "[kpr7] verify PRONG-B wiring: grep -E '\[FB_OTS\]|fb_ots_redirects' logs/coevolve_consume_${TAG}/stdout.log  # init + per-iter redirects>0"
echo "[kpr7] verify gates:         grep -E '\[OTS-TAUBF\]|\[BSRC_PHOT\]|\[FB_OTS\]|\[KPR\]|\[FLOORM\]|\[PUMPF\]|\[STAGE4\]|\[FB-MULTI\]|\[DBFB\]|\[SIMUL\] done' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr7] PRIMARY: SMOOTH front f(FeIV) s4 0.5-0.8, s6 0.03-0.15, s8 0.01-0.05; DEEP u(s0) 500-750, slope +2.0..+3.0, FUV(s0) 1.5-2.5e-4, funnel<=3x; T_e mid/outer +-500K of CMFGEN"
