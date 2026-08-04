#!/bin/bash
#SBATCH --job-name=a10_kx_ltherm
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.out
#SBATCH --error=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/%x_%j.err
#
# ============================================================================
# LINE_THERM "ARREST EXPERIMENT" — deep-shell thermal line re-emission probe
#                                  (PREPARE ONLY -- driver submits)
# ============================================================================
# Exact clone of the B-run recipe (the a10_kx_gphall config, same env as
# sbatch_tincol.sh's B-run block) PLUS the TWO new gate variables:
#     export LUMINA_LINE_THERM=1
#     export LUMINA_LINE_THERM_SMAX=2
# Binary: lumina_cuda.withLtherm (carries jtable + tetab + tincol + ltherm gates;
#         ALL env-OFF by default; byte-identical to the B-run when LINE_THERM unset).
# NO tetab, NO tincol, NO jtable: T_e and the field are FREE to respond. The whole
# point is a FREE, CONSERVATIVE evolution — the free response IS the measurement.
#
# WHAT CHANGES (color only, energy conserved):
#   In gated shells 0..SMAX (=0,1,2), whenever a packet undergoes a LINE interaction
#   (resonance scatter, macro-atom/downbranch/fluorescence emission, or MA-cap
#   resonant deactivation) its RE-EMISSION COMOVING FREQUENCY is redrawn from
#   Planck(T_e[shell]) instead of the line/macro-atom-selected frequency — forcing
#   the deep line source function to the CMFGEN-limit end-member (S -> B). Packet
#   ENERGIES/weights, direction convention, T_e, and every NON-line channel
#   (k-packet ff/fb continuum exits, bf/ff re-emit, e-scatter) are BIT-unchanged.
#   Unlike the LUMINA_TE_TABLE pin this is ENERGY-CONSERVING: T_e stays FREE and
#   must climb by itself, so a positive result is CAUSAL proof (not a tautology).
#   Causal question: does forcing deep-shell line re-emission to THERMAL restore the
#   missing deep FUV/EUV amplitude+gradient and let the gas heat itself?
#
# WHY (docs/FUV_GRADIENT_ATTACK_DESIGN.md 2026-07-19; validation/cmfgen_toy06_19p48d/
#   analysis/{trapping_audit,radeq_ledger_audit,reddening_localization}):
#   Lumina's deep-shell (s0-s2) MC field is a Co IV emission-line spectrum instead of
#   CMFGEN's smooth ~B(T) continuum — the deep line forest absorbs NUV/blue+red and
#   re-emits 84% of its energy into the Co IV 1490-1650A complex (42% of s0's total u
#   in one ~1508A bin; mc/cs=39x at 1526A). The line source function fails to
#   thermalize (S ~/~ B) although CMFGEN with the SAME line data holds S~=B(18760K) at
#   these depths (n_e~5e9, tau_sob up to ~1e5). Downstream: EUV/FUV starvation ->
#   radeq (proven faithful) lands the gas at the zero-pump root 13120K -> deep FUV
#   -1.54 dex -> Fe recombination gradient dead. This probe forces the end-member.
#
# ---------------------------------------------------------------------------
# PRE-REGISTERED GATES (verbatim -- do not move the goalposts after the run):
#
#   PASS (mastermind CONVICTED) = ALL of:
#    (i)   s0 u-fraction in 1290-2000A drops 0.51 -> <=0.40 (CMFGEN 0.32) AND the
#          ~1508A single-bin share 0.42 -> <=0.15;
#    (ii)  deep FUV J(918-1290A band-mean, s0) rises >= +1.0 dex
#          (5.81e-6 -> >= 5.8e-5; CMFGEN 2.02e-4);
#    (iii) [AMENDED -- driver addendum 2026-07-19, split-field audit] T_e(s0) rises
#          >= +1400K WITHOUT any pin (13120 -> >= 14500; counterfactual thermal-J root
#          14818, full ladder tops 18277) PROVIDED [TEHOLD] shows the s0 root is
#          actually being re-solved (radeq_root=root-found) in iters >= 3. If [TEHOLD]
#          shows pin_lo/HOLD persisting at s0 while (i)/(ii)/(iv) pass, gate (iii) is
#          VOID and the verdict becomes: bath thermalization CONVICTED for the field,
#          T_e-solver HOLD CONVICTED as a SECOND, independent criminal (the +3400K
#          lever) -- to be unfrozen in a separate fix.
#    (iv)  EUV(300-450A) J(s0) rises >= +1.0 dex (from 3.9e-12).
#
#   PARTIAL = (ii) +0.3..+1.0 dex OR only some of (i)-(iv) => thermalization is a
#             MAJOR but not SOLE mechanism; report which link failed.
#
#   NULL = (ii) < +0.3 dex => CHECK [LTHERM] counters FIRST
#          (0 = wiring no-op, NOT physics).
#
#   Secondary (directional, NOT gating): Fe f(IV) s6 moves down from 0.748 toward
#   CMFGEN 0.069; photospheric FUV excess (s8) expected to PERSIST (Axis-2
#   upconversion is a separate criminal -- its persistence does NOT invalidate PASS).
#
#   Wiring (check FIRST on any NULL):
#     [LTHERM] active SMAX=2 (thermal line re-emission, energy-conserving)  (once, init)
#     [LTHERM] it NN: thermalized_line_reemits=<N>   (once per iter; N==0 with the gate
#             ON prints a *** WARNING *** and is a wiring no-op, NOT a physics null.
#             Expect N ~ deep-shell line-interaction count, i.e. many thousands+/iter.)
#     [TEHOLD] s0/s1/s2/s8(ctrl): T_e=..K (prev=..K) radeq_root=...  (once per iter;
#             the amended-gate(iii) discriminator: root-found vs pin_lo/pin_hi HOLD.)
# ---------------------------------------------------------------------------
set -e
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
module load cuda/13.0.2 2>/dev/null || true
echo "Host: $(hostname)  Part: $SLURM_JOB_PARTITION  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# --- install the withLtherm binary as ./lumina_cuda for run_coevolve_s01.sh ------
# (that runner hardcodes ./lumina_cuda; we must not overwrite it permanently.)
# Back up the current default and restore it on exit, even on error/SIGTERM.
LT_BIN=lumina_cuda.withLtherm
[ -x "$LT_BIN" ] || { echo "FATAL: $LT_BIN missing/not built"; exit 2; }
ORIG_SAVE="$(mktemp -u ./lumina_cuda.origsave.XXXXXX)"
cp -p lumina_cuda "$ORIG_SAVE"
restore_bin() { [ -f "$ORIG_SAVE" ] && cp -p "$ORIG_SAVE" lumina_cuda && rm -f "$ORIG_SAVE" \
                && echo "[ltherm] restored original ./lumina_cuda"; }
trap restore_bin EXIT
cp -p "$LT_BIN" lumina_cuda
echo "[ltherm] installed $LT_BIN as ./lumina_cuda (orig -> $ORIG_SAVE, restored on exit)"

# --- B-run (a10_kx_gphall MODE=all) environment, verbatim ----------------------
export OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close
export PKTS=${PKTS:-400000} NITER=${NITER:-12}
export LUMINA_ION_POP_DUMP=1 LUMINA_KROMER=1
export LUMINA_EVENT_LOG=1 LUMINA_EVENT_LOG_CAP=128
export LUMINA_NLTE_SKIP_Z=""
export LUMINA_GPH_SIGMA_CMFGEN=1
export LUMINA_GPH_ALLLEVEL=1
# B-run had LUMINA_NLTE_BK_CEIL ABSENT (b_k cap OFF). Clear any inherited value so
# this clone matches the B-run env verbatim.
unset LUMINA_NLTE_BK_CEIL

# --- LINE_THERM deep-shell thermal re-emission gate (the ONLY new variables) -----
export LUMINA_LINE_THERM=1
export LUMINA_LINE_THERM_SMAX=2

TAG="a10_kx_ltherm"
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
echo "[ltherm] verify wiring: grep -E '\[LTHERM\]|\[TEHOLD\]' logs/coevolve_consume_${TAG}/stdout.log"
