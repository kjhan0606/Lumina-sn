#!/bin/bash
#SBATCH --job-name=a10_kx_kpr9
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# KPR9 = KPR8 + LUMINA_NLTE_METASTABLE_COLL (the COLD-CASE-P ROOT FIX).
# ============================================================================
# ROOT CAUSE (verified DIRECTLY from data/tardis_reference_toy06_19p48d):
#   Fe III level 17 (E=3.731 eV, g=7) is flagged metastable=1 in levels.csv but
#   has 371 UPWARD lines (lower=17, max f_lu=0.467) and ZERO DOWNWARD lines
#   (upper=17) -- its forbidden decay-to-ground is absent from the atomic data.
#   Consequence: no radiative de-excitation (line absent) AND no collisional
#   de-excitation (the NLTE collision assembly loops over LINES only; no line =>
#   no C_up/C_down channel). Level 17 fills by cascade and cannot drain => its
#   b_k pins at the STAGE4 cap (LUMINA_STAGE4_BK_CAP=1000) => it holds ~51% of
#   Fe III at s8 => its 461 A (EUV) photoionization drives f(FeIV) 12-31x over
#   CMFGEN. This IS the photospheric IGE over-ionization = the registered
#   COLD-CASE-P. CMFGEN thermalizes it (b_k median ~1.04, max 1.69) because it
#   CARRIES the forbidden collision strengths that TARDIS's line list omits.
#
# THE FIX (LUMINA_NLTE_METASTABLE_COLL=1, parameter-free, transferable):
#   In the rate-matrix assembly, for EVERY level flagged metastable=1 that has
#   NO downward radiative line (precomputed drainless_metastable[] in nlte_init),
#   ADD an Axelrod forbidden-collision floor (Omega=1) coupling it (as "upper")
#   to its ion's GROUND level (as "lower"):
#     C_down(meta->ground) = n_e*8.629e-6/(g_meta  *sqrt(Te))*Omega,  Omega=1
#     C_up  (ground->meta) = n_e*8.629e-6/(g_ground*sqrt(Te))*Omega*exp(-dE/kTe)
#   Detailed balance exact: C_up/C_down = (g_meta/g_ground)*exp(-dE/kTe). GROUND
#   is the single partner (it holds the ion population the pileup drains into; one
#   Omega=1 channel per level is the parameter-free Axelrod floor; coupling to ALL
#   lower levels would over-thermalize the allowed manifold). Off-diagonal
#   placement is byte-identical to the line-based collisions. This is the SURGICAL
#   alternative to LTE_NCRIT (which over-thermalizes allowed levels too).
#   COLL_FIX/ION_LOCK/FALLBACK_TE (already ON below) CANNOT help: they fix the
#   collision FORMULA for EXISTING transitions; the defect is a MISSING transition.
#   Scope is IRON-GROUP-GENERAL (all NLTE ions), not Fe-only.
#
# Binary: lumina_cuda.withKpr9 (= withKpr8 source + the metastable-coll pass).
#   LUMINA_NLTE_METASTABLE_COLL default OFF => the rate matrix is byte-identical to
#   withKpr8; the only added-code that runs OFF-gate is the (physics-inert)
#   drainless_metastable precompute in nlte_init.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES  (do NOT move)          Yardstick = CMFGEN toy06 @19.48d.
# ---------------------------------------------------------------------------
#  PASS (COLD-CASE-P fixed) = Fe III b_k at s8: the metastable pileup collapses --
#    level 17 b_k from 1000 (capped) toward O(1-10), and the Fe III excited
#    manifold no longer super-thermal; f(FeIV) s6/s7/s8 recombine toward CMFGEN
#    (s6 0.03-0.15 [CMFGEN 0.069], s8 0.01-0.06 [0.022]); the iron-group
#    over-ionization ratio Lumina/CMFGEN drops from 12-31x toward <3x;
#    DEEP gains HELD (FUV s0>=1.5e-4, slope +2.0..+3.0, funnel<=3x,
#    deep f(IV) s0>=0.95); T_e(s8) toward 10.4kK; residuals (Co rate deficit,
#    T_rad pin) as-is.
#  WIRING (check FIRST on any null):
#    [METACOLL] init count      -> "  [METACOLL] drainless-metastable levels coupled
#                                    to ground (Omega=1): FeIII=2 ... (total=N)"
#                                  (expect FeIII=2: levels 17 & 97; CoIII, NiIII,
#                                   and the IGE stage-IVs carry the largest counts.)
#    [METACOLL-PROBE] per iter  -> "  [METACOLL-PROBE] FeIII lvl17 b_k/gnd: s0=.. s..
#                                    =.. sLAST=..  (Te sLAST=..K)"  (b_k/gnd is the
#                                   departure of level 17 relative to ground; the
#                                   ground-coupled drain pins it toward 1. Baseline
#                                   pins ~1000; a working fix collapses the deepest
#                                   photospheric shell toward O(1-10).)
#  FALSIFIERS:
#    (i)  if [METACOLL-PROBE] sLAST stays ~1000 while the init count is nonzero,
#         the drain is under-powered (n_e too low / cascade fill dominates) -> the
#         missing-transition diagnosis holds but Omega=1-to-ground is insufficient;
#         report AS-IS (do NOT bump Omega -- that would be a tuning knob).
#    (ii) if f(FeIV,s8) recombines but the DEEP FUV gate BREAKS (over-thermalized
#         the allowed manifold), the ground-only coupling leaked -> re-open the
#         WHICH-lower-levels design decision.
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr9 binary as ./lumina_cuda for run_coevolve_s01.sh --------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr9
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[kpr9] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[kpr9] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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
# KPR9 ROOT FIX: the MISSING collisional de-excitation channel for
#                metastable levels lacking radiative decay (COLD-CASE-P).
# ============================================================================
export LUMINA_NLTE_METASTABLE_COLL=1   # add Axelrod (Omega=1) meta->ground drain
                                       #   for every drainless-metastable level

# --- STAGE4-ROUND2 (Part A) -----------------------------------------------------
export LUMINA_NLTE_STAGE4=1          # round-2 semantics (A1 depth-gate default 0.13,
                                     #   A2 top-ion clamp default ON, A3 Ti dropped)
export LUMINA_STAGE4_GPH_WTHR=0.13   # A1 depth gate: NLTE-weight III combs only where W>this
export LUMINA_STAGE4_BK_CAP=1000     # A1 per-level b_k cap inside the gate (the ceiling the
                                     #   metastable pileup pins against; the fix should make
                                     #   the deep level-17 b_k fall FAR below this cap)

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
export LUMINA_KPEMISS_FB_OTS_NUMIN=7.40e15 # 405A: redirect ONLY Fe III & bluer IGE EUV ground
                                           #   edges; SPARE the 405-912A low-IE band.
export LUMINA_KPEMISS_BSRC_PHOT_XION=1     # phot -4 exit thermalizes CROSS-ION re-excites only;
                                           #   same-ion cascades KEPT (preserve 912-2000A FUV)
export LUMINA_KPEMISS_OTS_MODE=2           # 2=GRADED P(OTS)=1-exp(-tau_bf) (default); 1=binary
export LUMINA_KPEMISS_OTS_TAU=1.0          # binary threshold = physical tau=1 boundary (sens. only)
# LUMINA_KPEMISS_BSRC_PHOT_WFLOOR intentionally UNSET => W-floor guard OFF => pure tau_bf.

TAG="a10_kx_kpr9"
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
echo "[kpr9] verify METACOLL init:  grep -E '\[METACOLL\]' logs/coevolve_consume_${TAG}/stdout.log  # FeIII=2 ... (total=N)"
echo "[kpr9] verify METACOLL probe: grep -E '\[METACOLL-PROBE\]' logs/coevolve_consume_${TAG}/stdout.log  # FeIII lvl17 b_k/gnd sLAST should fall from ~1000 toward O(1-10)"
echo "[kpr9] verify gates:          grep -E '\[METACOLL\]|\[METACOLL-PROBE\]|\[OTS-TAUBF\]|\[BSRC_PHOT\]|\[BSRC_PHOT_XION\]|\[FB_OTS\]|\[KPR\]|\[FLOORM\]|\[PUMPF\]|\[STAGE4\]|\[FB-MULTI\]|\[DBFB\]|\[SIMUL\] done' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr9] PASS: FeIII lvl17 b_k/gnd(sLAST) O(1-10); f(FeIV) s6 0.03-0.15, s8 0.01-0.06; IGE over-ion Lumina/CMFGEN <3x; DEEP FUV(s0)>=1.5e-4, slope +2.0..+3.0, funnel<=3x, f(IV,s0)>=0.95; T_e(s8) toward 10.4kK; residuals (Co deficit, T_rad pin) unchanged"
