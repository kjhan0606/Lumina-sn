#!/bin/bash
#SBATCH --job-name=a10_kx_kpr5jt
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# KPR5 = KPR4 + PHASE-1 FINAL LEVER: the trace-ion NLTE population floor +
#        the zero-count pump fallback                          (PREPARE ONLY)
# ============================================================================
# Exact clone of scripts/sbatch_kpr4.sh (STAGE4-R2 + KPEMISS_REPAIR B1/B2/B3
# SRC=2 + DB_FB + COOLGUARD + Fork B + PUMP_FIELD alpha-blend a=1), with TWO
# new gates from the kpr4 residual_offset_budget VERDICT:
#
#   FIX-1 (dominant, unifying) -- the trace-ion NLTE population FLOOR.
#     ROOT (residual_offset_budget/VERDICT.md, levelpop query): the NLTE writeback
#     clamps negative/collapsed super-levels to a FLAT 1e-30 in solve-space, then
#     rescales the whole ion to its nebular total. For a TRACE ion at the
#     photosphere (Fe III, f<<1) the rate matrix collapses ~1400/1500 excited
#     levels; they all clamp to 1e-30 and rescale to the SAME absolute pop
#     (n_k=1.342729e-3 at s8) -- an ABSOLUTE floor. b_k=(n_k/n_gnd)/boltz then
#     grows as exp(+dE/kTe), pinning the near-threshold comb at b_k up to 3e8.
#     Those floored excited levels photoionize off the OPTICAL field (>2000A =
#     92% of Gamma(FeIII)) -> f(FeIV,s8)=0.980 (CMFGEN 0.022), feed IV->III
#     recomb EUV -> bf over-heating, and burn out the III coolant. Deep s0 (Fe III
#     NOT trace) has 0/1500 floored -- the defect is trace-ion + photospheric.
#     Operative floor: the GPU writeback in src/lumina_cuda.cu (the CUDA run's
#     path); mirrored in src/lumina_plasma.c (CPU nlte_solve_ion_shell).
#
#     THE FIX (env-gated, default OFF => byte-identical to withKpr4):
#       LUMINA_NLTE_FLOOR_MODE=1     replace the flat 1e-30 negative-clamp with an
#                                    LTE-RELATIVE floor + b_k cap: negative/sub-
#                                    resolution levels are floored at their LTE@Te
#                                    value (b_k=1, positive => the numerical guard
#                                    survives, no zeros/negatives), and EVERY level
#                                    is capped at n_k <= BKMAX*Boltzmann@Te.
#       LUMINA_NLTE_FLOOR_BKMAX=1000 the departure cap (b_k<=1000). Literature SN
#                                    Fe/O departures are b~1-50; 1000 is a generous
#                                    numerical guard that still kills the 1e8 comb.
#     The clamp acts in solve-space BEFORE the xfl redistribute + per-ion rescale
#     (both UNCHANGED), so Sum n_k = n_total (per-ion in lock mode) is preserved
#     exactly -- only the SHAPE of the collapsed/over-populated levels changes.
#
#   FIX-2 (secondary, depth-growing) -- the zero-count pump fallback field.
#     kpr4 [PUMPF]: 25.8% of simul_line_term pump evals fall back to cs_J on
#     zero-count mc bins (blended=21503332 cs_fallback=7473805). cs_J is ~100x
#     super-thermal at depth (int J_cs s8 = 5.8e12 vs mc 6.0e10) -> injects pump-
#     heating (up to -1683 K if removed at s8), a warm arm the field cannot
#     self-correct.
#       LUMINA_RADEQ_PUMP_FALLBACK=1 route GENUINE zero-count mc bins (field armed,
#                                    bin never sampled) to the local thermal
#                                    B_nu(T_e) instead of the super-thermal cs_J
#                                    (consistent with the DBFB ledger). The pre-
#                                    transport (mc field NULL) and out-of-grid
#                                    guards are UNTOUCHED. Only takes effect with
#                                    LUMINA_RADEQ_PUMP_FIELD=1 (inherited from kpr4).
#
# Binary: lumina_cuda.withKpr5 (= withKpr4 source + the two gates; both unset/=0
#         => byte-identical to withKpr4, so master-off => byte-identical baseline).
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES  (do NOT move -- from residual_offset_budget's predictions)
#   Yardstick = CMFGEN toy06 @19.48d at Lumina velocities.
# ---------------------------------------------------------------------------
#  PRIMARY (the floor lock + the depth-growing pump warm arm this fix attacks):
#    * T_e(s0)  17760-19760 K   (CMFGEN 18760; within +-1000 K)
#    * T_e(s2)  15351-17351 K   (CMFGEN 16351; within +-1000 K)
#    * T_e(s4)  12657-14657 K   (CMFGEN 13657; within +-1000 K)
#    * T_e(s6)  10929-12929 K   (CMFGEN 11929; within +-1000 K)
#    * T_e(s8)   9383-11383 K   (CMFGEN 10383; within +-1000 K; was 12181)
#    * f(FeIV, s8) <= 0.25      (expect <= 0.10; toward CMFGEN 0.022 -- floor lock
#                                removed => the floored opt+ channel is gone AND the
#                                EUV-emission source falls with the Fe IV reservoir)
#    * f(FeIV, s0) >= 0.9       (recovers toward 1.0: deep Fe III is NOT trace, so
#                                FLOORM barely touches it; Boltzmann pops -> field
#                                (EUV) lock keeps it ionized, was 0.810)
#  RETAINED KPR4 GAINS (must HOLD -- both fixes act on pops/pump, not FUV transport):
#    * FUV(918-1290, s0) >= 1.5e-4     (deep FUV not re-collapsed)
#    * FUV gradient slope >= +2.0      (outward steepening held)
#    * u_bol(s0) >= 450                (bath energy held)
#    * funnel dead: mc/cs @1450-1650 <= 3x  (Co IV pile stays killed)
#  RESIDUALS -- PRE-REGISTERED NO-CHANGE (do NOT claim improvement):
#    * Co twin rate deficit (~10x) unchanged -- a RATE defect, orthogonal to both fixes.
#    * MC blue-tilt / far-outer hot-band untouched; T_rad pin (C9).
#  WIRING (check FIRST on any null):
#    [FLOORM] mode=1 BKMAX=1000: ...                (once, init; confirms FIX-1 armed)
#    [FLOORM] iter NN clamped levels: deep(s0-2)=.. mid(s3-6)=.. phot(s>=7)=..
#            (per CE iter; phot >> deep expected -- the floor bites trace ions
#             at/beyond the photosphere, deep s0-2 should stay ~0)
#    [PUMPF] LUMINA_RADEQ_PUMP_FALLBACK=1: zero-count mc bins -> B_nu(T_e) ...  (once)
#    [PUMPF] line_Jb: blended=<n> cs_fallback=<m> fallback_mode=1 bnu_routed=<k>
#            (per iter; bnu_routed>0 AFTER first transport pass => FIX-2 firing on
#             the zero-count bins; bnu_routed ~ cs_fallback once mc armed)
#    [DBFB] selfcheck s0: net(J=B)/H = <x>          (once; MUST be < 1e-6)
#    [KPR] ... src=2 (chi_line*B_nu forest)         (once, init; SRC=2 still armed)
#    [SIMUL] done: pins hi=<n> lo=<m> of 50          (per iter)
#  FALSIFIERS (from the VERDICT):
#    (i)  if FIX-1 leaves f(FeIV,s8) > 0.5, the floor is NOT the lock -> re-open
#         field-lock / recombination.
#    (ii) if T_e(s8) stays >= 13 kK while [FLOORM] phot>0 (demonstrably firing),
#         the floor/pump thesis for the warm profile is wrong -> escalate to the
#         over-ionization arm (Gph onto the CMFGEN J-table) or deep MC emission COLOR.
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withKpr5 binary as ./lumina_cuda for run_coevolve_s01.sh --------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
KPR_BIN=lumina_cuda.withKpr5
[ -x "$KPR_BIN" ] || { echo "FATAL: $KPR_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[kpr5] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$KPR_BIN" lumina_cuda
echo "[kpr5] installed $KPR_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

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
# (FLOORM owns the writeback clamp when MODE=1, so the legacy bk_ceil path is
#  bypassed regardless; unset keeps the environment clean.)
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

# --- PHASE-1a: unify the radeq line-pump field onto the Gph alpha-blend (inherited)
export LUMINA_RADEQ_PUMP_FIELD=1     # simul_line_term Jb = alpha*mc_J + (1-alpha)*cs_J
                                     #   (alpha = LUMINA_COEVOLVE_PHOTOION_ALPHA below)

# --- PHASE-1 FINAL LEVER: floor policy + zero-count pump fallback  <<<<< NEW ------
export LUMINA_NLTE_FLOOR_MODE=1
export LUMINA_GPH_JTABLE=data/cmfgen_jtable_toy06_19p48d.bin  # [#33 FORCING] CMFGEN J -> photospheric Gph (confirm field-guilty)      # FIX-1: LTE-relative floor + b_k cap (was flat 1e-30)
export LUMINA_NLTE_FLOOR_BKMAX=1000  # FIX-1: departure cap b_k<=1000
export LUMINA_RADEQ_PUMP_FALLBACK=1  # FIX-2: zero-count mc bins -> B_nu(Te), not cs_J

TAG="a10_kx_kpr5jt"
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
echo "[kpr5] verify FLOORM wiring:  grep -E '\[FLOORM\]' logs/coevolve_consume_${TAG}/stdout.log   # init(mode=1) + per-iter deep/mid/phot"
echo "[kpr5] verify PUMPF wiring:   grep -E '\[PUMPF\]' logs/coevolve_consume_${TAG}/stdout.log   # init(fallback=1) + per-iter blended/cs_fallback/bnu_routed"
echo "[kpr5] verify SRC=2 wiring:   grep -E '\[KPR\] .*src=2' logs/coevolve_consume_${TAG}/stdout.log   # init + per-iter"
echo "[kpr5] verify gates:          grep -E '\[FLOORM\]|\[PUMPF\]|\[KPR\]|\[STAGE4\]|\[FB-MULTI\]|\[BSRC\]|\[DBFB\]|\[SIMUL\] done' logs/coevolve_consume_${TAG}/stdout.log"
echo "[kpr5] FLOORM phot(s>=7)>>deep(s0-2) => floor bites trace ions at the photosphere as designed; PUMPF bnu_routed>0 after iter 0 => FIX-2 firing"
