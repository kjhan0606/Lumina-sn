#!/usr/bin/env bash
# Run the sealed roundoff-aware comparison only after the linear V2 pass.
set -euo pipefail
umask 027

[[ $# -eq 2 ]] || {
  printf 'usage: %s RUN_ROOT POSTPROCESS_BUNDLE\n' "$0" >&2
  exit 70
}
run_root="$1"
bundle="$2"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs ]] || exit 70
[[ "$bundle" = "$run_root"/* && -f "$bundle/READY" ]] || exit 70

control="$run_root/manual_control"
log="$control/line_saturation_roundoff_v3.log"
v2_log="$control/line_saturation_postprocess_v2.log"
v2_verdict="$run_root/LINE_SATURATION_VERDICT_V2.txt"
summary="$run_root/a210_line_saturation_summary_v2.json"
compare_script="$bundle/compare_a210_cmfgen_line_saturation.py"
coordinate="$bundle/cmfgen_coordinate_reference.json"
netrate="$(<"$bundle/cmfgen_netrate_path.txt")"
report="$run_root/a210_cmfgen_line_saturation_comparison_roundoff_v3.json"
verdict="$run_root/LINE_SATURATION_ROUNDOFF_VERDICT_V3.txt"
lock="$control/LINE_SATURATION_ROUNDOFF_V3_LOCK"

mkdir -p "$control"
exec >>"$log" 2>&1
say() { printf 'LINE_SATURATION_ROUNDOFF_V3 utc=%s %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$*"; }

if [[ -f "$verdict" ]] && grep -qx 'status=PASS' "$verdict"; then
  say "status=ALREADY_PASS"
  exit 0
fi
mkdir "$lock" 2>/dev/null || {
  say "status=BLOCKED reason=ANOTHER_V3_MONITOR_OWNS_LOCK"
  exit 4
}
trap 'rmdir "$lock" 2>/dev/null || true' EXIT
say "status=START bundle=$bundle"

verify_bundle() {
  (cd "$bundle" && sha256sum -c POSTPROCESS_MANIFEST.sha256) >/dev/null &&
    [[ "$(sha256sum "$netrate" | cut -d' ' -f1)" == \
       "$(<"$bundle/cmfgen_netrate.sha256")" ]]
}
verify_bundle || {
  say "status=BLOCKED reason=POSTPROCESS_BUNDLE_SHA_DRIFT"
  exit 4
}

while [[ ! -f "$v2_verdict" ]]; do
  if [[ -f "$v2_log" ]] && \
     tail -n 1 "$v2_log" | grep -Eq 'status=(BLOCKED|YIELDED)'; then
    say "status=BLOCKED reason=V2_DID_NOT_PASS detail=$(tail -n 1 "$v2_log")"
    exit 4
  fi
  sleep 20
done

for exact in \
  'status=PASS' \
  'model_rc=1' \
  'natural_result=RADEQ_NO_BRACKET' \
  'physical_mutation=0' \
  'floor=0' 'cap=0' 'clamp=0' 'jitter=0' 'repair=0'; do
  grep -qx "$exact" "$v2_verdict" || {
    say "status=BLOCKED reason=V2_VERDICT_CONTRACT_MISMATCH field=$exact"
    exit 4
  }
done
[[ -f "$summary" && ! -L "$summary" ]] || {
  say "status=BLOCKED reason=MISSING_V2_SUMMARY"
  exit 4
}
summary_sha="$(sha256sum "$summary" | cut -d' ' -f1)"
grep -qx "summary_sha256=$summary_sha" "$v2_verdict" || {
  say "status=BLOCKED reason=V2_SUMMARY_SHA_MISMATCH"
  exit 4
}
verify_bundle || {
  say "status=BLOCKED reason=POSTPROCESS_BUNDLE_SHA_DRIFT_AFTER_V2"
  exit 4
}

set +e
python3 "$compare_script" \
  --summary "$summary" \
  --netrate "$netrate" \
  --coordinate-reference "$coordinate" \
  --report "$report" \
  > "$run_root/line_saturation_comparison_roundoff_v3.stdout" \
  2> "$run_root/line_saturation_comparison_roundoff_v3.stderr"
comparison_rc=$?
set -e
printf '%s\n' "$comparison_rc" \
  > "$run_root/line_saturation_comparison_roundoff_v3.rc"
[[ "$comparison_rc" -eq 0 ]] || {
  say "status=BLOCKED reason=CMFGEN_ROUNDOFF_COMPARISON_FAILED rc=$comparison_rc"
  exit 4
}

comparison_sha="$(sha256sum "$report" | cut -d' ' -f1)"
comparator_sha="$(sha256sum "$compare_script" | cut -d' ' -f1)"
temporary="$verdict.tmp.$$"
printf '%s\n' \
  "status=PASS" \
  "postprocess_bundle=$bundle" \
  "v2_summary_sha256=$summary_sha" \
  "comparison_sha256=$comparison_sha" \
  "comparator_sha256=$comparator_sha" \
  "execution_order=AFTER_V2_PASS" \
  "arithmetic_bound_is_physical_tolerance=0" \
  "physical_mutation=0" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0" \
  > "$temporary"
mv "$temporary" "$verdict"
say "status=PASS summary_sha256=$summary_sha comparison_sha256=$comparison_sha"
