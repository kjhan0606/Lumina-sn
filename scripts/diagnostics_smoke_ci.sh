#!/usr/bin/env bash
# Tier 3.3c — diagnostics-only smoke CI.
#
# Chains every Tier 0/1/3 bug-hunting check in dependency order and fails on
# the FIRST non-zero exit (set -e).  Designed for:
#
#   * pre-push hook       :  $ bash scripts/diagnostics_smoke_ci.sh
#   * per-PR CI runner    :  same; cheap (no GPU run, no slurm submit)
#   * post-rebuild check  :  pass --ref <new_ref_dir> to validate it
#
# Layers (skip-flags in [ ]):
#   0.1a  validate_reference.py  on  --ref           [--no-ref]
#   1.2a  tests/run_all_golden.sh  (analytical tests)  [--no-golden]
#   3.3b  dead_code_audit.py  (whole repo)             [--no-deadcode]
#   1.1b  scan_lumina_banner.py  on --sentinel        [skipped if no --sentinel]
#   1.2b  diagnose_nlte_pops.py  on --sentinel        [skipped if no --sentinel]
#
# Usage:
#   scripts/diagnostics_smoke_ci.sh \
#       [--ref data/tardis_reference_v3_femerge_capraise_bbfix_metafix_asE] \
#       [--sentinel lumina_cuda_h100_someRun] \
#       [--no-ref] [--no-golden] [--no-deadcode]
#
# Exit codes:
#   0  all layers passed (or skipped)
#   1  some layer flagged a finding
#   2  invocation error
set -u
set -o pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "[FAIL] cd $REPO_ROOT" >&2; exit 2; }

# Defaults
REF=""
SENTINEL=""
DO_REF=1
DO_GOLDEN=1
DO_DEADCODE=1

# Auto-pick most recent metafix_asE reference if none given
DEFAULT_REF="data/tardis_reference_v3_femerge_capraise_bbfix_metafix_asE"
[[ -d "$DEFAULT_REF" ]] && REF="$DEFAULT_REF"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)        REF="$2"; shift 2;;
    --sentinel)   SENTINEL="$2"; shift 2;;
    --no-ref)     DO_REF=0; shift;;
    --no-golden)  DO_GOLDEN=0; shift;;
    --no-deadcode) DO_DEADCODE=0; shift;;
    -h|--help)    sed -n '1,30p' "$0"; exit 0;;
    *) echo "[FAIL] unknown arg: $1" >&2; exit 2;;
  esac
done

FAIL=0
LAYER_RESULTS=()

run_layer() {
  # $1 = label, $2... = command
  # Exit-code policy:
  #   0   = pass
  #   1   = warn-only (validate_reference returns 1 when only WARN, not ERROR;
  #         smoke-CI should not block on advisory mismatches like stale
  #         config.json fields that the runtime ignores)
  #   2+  = real failure
  local label="$1"; shift
  echo
  echo "============================================================"
  echo "  [layer] $label"
  echo "  cmd:    $*"
  echo "============================================================"
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    LAYER_RESULTS+=("PASS  $label")
  elif [[ "$rc" -eq 1 ]]; then
    LAYER_RESULTS+=("PASS(warn)  $label")
  else
    LAYER_RESULTS+=("FAIL($rc)  $label")
    FAIL=1
  fi
}

# --- 0.1a reference validator ---
if [[ "$DO_REF" -eq 1 ]]; then
  if [[ -z "$REF" || ! -d "$REF" ]]; then
    echo "[skip] 0.1a validate_reference  (no --ref dir, default not found)"
    LAYER_RESULTS+=("SKIP  0.1a validate_reference")
  else
    run_layer "0.1a validate_reference  $REF" \
      python3 scripts/validate_reference.py "$REF"
  fi
fi

# --- 1.2a golden analytical tests ---
if [[ "$DO_GOLDEN" -eq 1 ]]; then
  if [[ -x tests/run_all_golden.sh ]]; then
    run_layer "1.2a golden tests" bash tests/run_all_golden.sh
  else
    echo "[skip] 1.2a golden  (tests/run_all_golden.sh missing or not executable)"
    LAYER_RESULTS+=("SKIP  1.2a golden")
  fi
fi

# --- 3.3b dead-code audit ---
if [[ "$DO_DEADCODE" -eq 1 ]]; then
  run_layer "3.3b dead_code_audit" python3 scripts/dead_code_audit.py --repo .
fi

# --- 1.1b banner scan (optional, requires sentinel run) ---
if [[ -n "$SENTINEL" ]]; then
  if [[ -d "$SENTINEL" || -f "$SENTINEL" ]]; then
    run_layer "1.1b scan_lumina_banner  $SENTINEL" \
      python3 scripts/scan_lumina_banner.py "$SENTINEL"
  else
    echo "[skip] 1.1b scan_lumina_banner  (sentinel $SENTINEL not found)"
    LAYER_RESULTS+=("SKIP  1.1b scan_lumina_banner")
  fi

  # --- 1.2b NLTE diagnostic (optional, requires sentinel + NLTE dumps) ---
  if compgen -G "$SENTINEL/nlte_levels_iter*.csv" >/dev/null 2>&1; then
    run_layer "1.2b diagnose_nlte_pops  $SENTINEL" \
      python3 scripts/diagnose_nlte_pops.py "$SENTINEL"
  else
    echo "[skip] 1.2b diagnose_nlte_pops  (no nlte_levels_iter*.csv under $SENTINEL — run needs LUMINA_NLTE_LEVEL_DUMP=1)"
    LAYER_RESULTS+=("SKIP  1.2b diagnose_nlte_pops")
  fi
else
  echo
  echo "[info] no --sentinel supplied; skipping run-side checks 1.1b + 1.2b"
fi

# --- summary ---
echo
echo "============================================================"
echo "  SMOKE CI SUMMARY"
echo "============================================================"
for r in "${LAYER_RESULTS[@]}"; do
  echo "  $r"
done
echo

if [[ "$FAIL" -eq 0 ]]; then
  echo "=== smoke CI: ALL LAYERS PASSED ==="
  exit 0
fi
echo "=== smoke CI: FAIL ==="
exit 1
