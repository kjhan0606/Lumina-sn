#!/usr/bin/env bash
# ============================================================================
# run_emergent.sh — canonical launcher for the freq-resolved EMERGENT spectrum.
#
# WHY THIS EXISTS: the slurm harness defaults are NON-champion A/B-test knobs
# (LSTAR=0 LINE_RE=0 TE_RATIO=0.9 FROZENIN=0 LINERES_JBAR=0). A correct emergent
# run needs 10 knobs all set together; setting them by hand wasted hours of GPU
# time twice on 2026-06-25 (wrong-plasma config, then missing producer gate).
# This wrapper bakes in the VERIFIED champion + emergent config so callers only
# pass the diagnostic delta.
#
# USAGE:
#   scripts/run_emergent.sh                       # baseline emergent (FULL)
#   scripts/run_emergent.sh FINE_CONTONLY=1       # continuum-only falsifier
#   scripts/run_emergent.sh FINE_SL_CLAMP=10 ...  # any override(s)
# Every KEY=VAL arg overrides a baked default. The fully resolved config is
# echoed before sbatch so it is impossible to launch the wrong thing silently.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# --- champion plasma config (memory: project_*; lock 165233) ---
: "${LSTAR:=1}"          # Phase-1 faithful Lambda* T_e response
: "${LINE_RE:=1}"        # Option-2 integral-RE line term
: "${TE_RATIO:=1.0}"     # T_e/T_rad seed+fallback (NOT the 0.9 harness default)
: "${FROZENIN:=1}"       # frozen-in freeze-out owns outer shells
: "${JNU_PHOTOION:=1}"   # J_nu-driven photoionization

# --- freq-resolved emergent producer (memory: project_toored_rootcause_ladder) ---
: "${LINERES_JBAR:=1}"   # GATE for cmfgen_fine_jbar (the producer; emergent lives inside it)
: "${FINE_EMERGENT:=1}"  # write lumina_spectrum_freqres.csv
: "${LAMLO:=3000}"       # emergent window lo (A)
: "${LAMHI:=12000}"      # emergent window hi (A) — must be WIDE, not the 3200 harness default
: "${FINE_SL_CLAMP:=1.0}" # thermal line source (S_l<=1.0*B); best color = pure thermal + freq-resolved continuum
: "${N_ITER:=8}"         # converged plasma

# --- apply KEY=VAL overrides from args (after baked defaults so they win) ---
for kv in "$@"; do
  case "$kv" in
    *=*) export "${kv%%=*}=${kv#*=}";;
    *) echo "run_emergent.sh: bad arg '$kv' (expected KEY=VAL)" >&2; exit 2;;
  esac
done
export LSTAR LINE_RE TE_RATIO FROZENIN JNU_PHOTOION \
       LINERES_JBAR FINE_EMERGENT LAMLO LAMHI FINE_SL_CLAMP N_ITER

echo "=== run_emergent.sh resolved config ==="
printf '  %-14s = %s\n' \
  LSTAR "$LSTAR" LINE_RE "$LINE_RE" TE_RATIO "$TE_RATIO" FROZENIN "$FROZENIN" \
  JNU_PHOTOION "$JNU_PHOTOION" LINERES_JBAR "$LINERES_JBAR" \
  FINE_EMERGENT "$FINE_EMERGENT" LAMLO "$LAMLO" LAMHI "$LAMHI" \
  FINE_SL_CLAMP "$FINE_SL_CLAMP" N_ITER "$N_ITER" \
  FINE_CONTONLY "${FINE_CONTONLY:-0}"
# expected dir tag (so caller can eyeball it matches radls1_linere1_ratio1.0_fz1)
echo "  expected dir tag: jnul${JNU_LSTAR:-0}_radls${LSTAR}_linere${LINE_RE}_ratio${TE_RATIO}_pi${JNU_PHOTOION}_fz${FROZENIN}"
echo "========================================"

JOB=$(sbatch --parsable scripts/slurm_ddc15_pure_cmfgen_phase3.sh)
echo "submitted job: $JOB"
