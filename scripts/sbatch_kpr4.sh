#!/bin/bash
#SBATCH --job-name=a10_kx_kpr4
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# KPR4 = KPR3 + PHASE-1a PUMP-FIELD UNIFICATION: kill the split-field residue
#                                                              (PREPARE ONLY)
#   the radeq ETLA line pump (simul_line_term) now reads THE SAME alpha-blended
#   field the Gph/Hex photoion loop consumes, instead of the hard-wired
#   deterministic cs_J.                            (LUMINA_RADEQ_PUMP_FIELD=1)
# ============================================================================
# Exact clone of scripts/sbatch_kpr3.sh (STAGE4-R2 A1/A2/A3 + KPR B1/B2/B3 +
# DB_FB + COOLGUARD + Fork B + BSRC_SRC=2), with ONE new gate:
#
#   THE DEFECT KPR3 LEFT (te_bias_budget, Phase 1a): kpr2/kpr3 T_e is nearly FLAT
#   (16.8-20.5 kK) while CMFGEN cools steeply outward (18.8->10.4 kK); the warm
#   bias GROWS with depth-out (+1785/+3790/+6402 K at s0/s4/s8). The budget's
#   dominant lever (C2, CONFIRMED) is a SPLIT FIELD inside one energy balance:
#     - the line pump simul_line_term baked its per-line Jb from the HARD-WIRED
#       deterministic cs_J (nlte_get_J_at_nu; plasma.c line-build ~5940-5951),
#     - while Gph photoion + bf-heating Hex in the SAME simul_r1 balance consume
#       the alpha-blended MC field (J = alpha*mc_J + (1-alpha)*cs_J, alpha=1.0;
#       plasma.c ~5748-5768/5804-5824/5867-5891).
#   With the DBFB-repaired (brighter, differently-shaped) mc_J, the cs_J-fed pump
#   under-cools / pump-heats: it FLIPS Lambda_line to net HEATING at s4/s8. The
#   pump arm alone carries -664 / -1563 / -3863 K of the bias (37/41/60 %),
#   growing monotonically with depth -- exactly tracking the flat-vs-cooling gap.
#
#   THE FIX (env-gated, default OFF => byte-identical to withKpr3):
#     LUMINA_RADEQ_PUMP_FIELD=1  plasma.c: the line-build Jb is read from THE SAME
#                             field the Gph loop consumes --
#                               J_line = alpha*mc_J(bin) + (1-alpha)*cs_J(bin)
#                             with alpha the LIVE g_photoion_mc_alpha from the
#                             existing LUMINA_COEVOLVE_PHOTOION_ALPHA machinery
#                             (ONE field definition, no new free parameter). Each
#                             line is mapped to its NLTE bin with the SAME
#                             convention as the cs lookup (nlte_get_J_at_nu log
#                             grid) so ONLY the field SOURCE changes -- binning,
#                             cooling and stimulated structure of simul_line_term
#                             are untouched. Startup: before the first transport
#                             pass mc_J is NULL and every line falls back to cs_J,
#                             mirroring the Gph guard (counted [PUMPF] line_Jb).
#     NOTE (budget C1, REJECTED): do NOT add the full-Planck DBFB variant
#     (LUMINA_RADEQ_DB_FB=2) -- the Wien->Planck partner moves the root <30 K
#     (bf emission is EUV-dominated, hnu/kT ~ 15-30). It is NOT the disease.
#
# Binary: lumina_cuda.withKpr4 (= withKpr3 source + the PUMP_FIELD gate; gate
#         unset/=0 => byte-identical to withKpr3, so master-off => byte-identical
#         baseline).
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES  (do NOT move -- from te_bias_budget's OWN predictions)
#   Yardstick = CMFGEN toy06 @19.48d at Lumina velocities. Unifying the pump onto
#   the same field the heating consumes should let the photosphere COOL and
#   steepen the T_e profile outward like CMFGEN:
# ---------------------------------------------------------------------------
#  PRIMARY (the flat-vs-cooling profile this fix attacks):
#    * T_e(s0)  18000-19500 K   (CMFGEN 18760; bias was +1785)
#    * T_e(s4)  13000-15500 K   (CMFGEN 13657; bias was +3790; must DROP from ~17.4k)
#    * T_e(s8)  10000-12000 K   (CMFGEN 10383; bias was +6402; must DROP from ~16.8k)
#      => the near-flat kpr2/kpr3 profile MUST steepen outward toward CMFGEN.
#    * f(FeIV, s8)  <= 0.25     (toward CMFGEN 0.022; III coolant restored as the
#                                pump stops burning it out; was 0.983)
#  RETAINED KPR3/KPR2 GAINS (must hold -- the pump fix must not undo them):
#    * u_bol(s0)  >= 400        (the SRC=2 forest-re-trap arm carries this; NOTE:
#                                the kpr3-arm readout may adjust this number once
#                                the pump also cools/re-shapes the deep field --
#                                leave a note, do not fail solely on a small shift)
#    * FUV(918-1290, s0) >= 1e-5           (deep FUV not re-collapsed)
#    * funnel dead: mc/cs @1450-1650 <= 3x (Co IV pile stays killed)
#    * FUV gradient slope >= +1.5          (outward steepening restored)
#    * EUV(s0) >= 1e-9                      (non-thermal EUV survives)
#  RESIDUALS -- PRE-REGISTERED NO-CHANGE (do NOT claim improvement):
#    * Co twin rate deficit (~10x, C7/C10) unchanged -- this is a RATE defect, the
#      pump fix is a FIELD-SOURCE swap, orthogonal.
#    * MC blue-tilt / far-outer hot-band (H9/H10) untouched; T_rad pin 10470 (C9).
#  WIRING (check FIRST on any null):
#    [PUMPF] LUMINA_RADEQ_PUMP_FIELD=1: ... alpha-blend (alpha=1.00) ...  (once, init;
#            confirms the gate armed + alpha source)
#    [PUMPF] line_Jb: blended=<n> cs_fallback=<m>   (per iter; blended>0 AFTER the
#            first transport pass => field-source swap firing; blended=0 on iter 0
#            is the expected pre-transport mc_J-NULL startup fallback)
#    [DBFB] selfcheck s0: net(J=B)/H = <x>          (once; MUST be < 1e-6)
#    [KPR] ... src=2 (chi_line*B_nu forest)         (once, init; SRC=2 still armed)
#    [KPR] it NN: src=2 bteq_exits=<n> cdf_exits=<m> chi_fallback=<k>   (per iter)
#    [SIMUL] done: pins hi=<n> lo=<m> of 50          (per iter)
#  HARD KILL: if the T_e profile does NOT steepen (s8 stays >= 13 kK) while
#    [PUMPF] blended>0 (gate demonstrably firing) and f(FeIV,s8) stays > 0.5, the
#    "split-field pump" thesis for the flat profile is wrong -> escalate per the
#    budget: the residual is the over-ionization arm (unify Gph onto the CMFGEN
#    J-table too, LUMINA_GPH_JTABLE, budget option 2) or the deep MC emission
#    COLOR (campaign F3, DIFFUSE_INNER_BC / Co IV fluorescence).
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr4 binary as ./lumina_cuda for run_coevolve_s01.sh --------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr4
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[kpr4] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[kpr4] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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
export LUMINA_KPEMISS_BSRC_SRC=2     # B3 refinement: -4 exit nu ~ chi_line(nu)*B_nu(Te)  (retained)

# --- KPR2: the principled thermal-ledger fix (inherited) ------------------------
export LUMINA_RADEQ_DB_FB=1          # simul_r1 bf cooling = detailed-balance partner of H_photo
export LUMINA_KPEMISS_COOLGUARD=1    # skip B3+FB-MULTI thermal exits where f(FeV)>0.5

# --- Fork B per-line thermal source (shipped WITH the repair; A/B-off is a later arm)
export LUMINA_LINE_BSRC=1
export LUMINA_LINE_BSRC_MODE=1

# --- PHASE-1a: unify the radeq line-pump field onto the Gph alpha-blend  << NEW --
export LUMINA_RADEQ_PUMP_FIELD=1     # simul_line_term Jb = alpha*mc_J + (1-alpha)*cs_J
                                     #   (alpha = LUMINA_COEVOLVE_PHOTOION_ALPHA below)

TAG="a10_kx_kpr4"
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
echo "[kpr4] verify PUMPF wiring:  grep -E '\[PUMPF\]' logs/coevolve_consume_${TAG}/stdout.log   # init(alpha=1.00) + per-iter blended/cs_fallback"
echo "[kpr4] verify SRC=2 wiring:  grep -E '\[KPR\] .*src=2' logs/coevolve_consume_${TAG}/stdout.log   # init + per-iter"
echo "[kpr4] verify gates:         grep -E '\[PUMPF\]|\[KPR\]|\[STAGE4\]|\[FB-MULTI\]|\[BSRC\]|\[DBFB\]|\[SIMUL\] done' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr4] PUMPF blended>0 after iter 0 => field-source swap firing; blended=0 on iter 0 is the expected pre-transport (mc_J NULL) fallback"
