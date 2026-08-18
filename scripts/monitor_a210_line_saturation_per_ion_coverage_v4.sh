#!/usr/bin/env bash
# Bind selected saturation rows to same-run owner totals after V3 passes.
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
log="$control/line_saturation_per_ion_coverage_v4.log"
v2_verdict="$run_root/LINE_SATURATION_VERDICT_V2.txt"
v3_verdict="$run_root/LINE_SATURATION_ROUNDOFF_VERDICT_V3.txt"
v3_log="$control/line_saturation_roundoff_v3.log"
summary="$run_root/a210_line_saturation_summary_v2.json"
owner_report="$run_root/a210_line_ion_owner_report_coverage_v4.json"
coverage_report="$run_root/a210_line_saturation_per_ion_coverage_v4.json"
verdict="$run_root/LINE_SATURATION_PER_ION_COVERAGE_VERDICT_V4.txt"
owner_script="$bundle/summarize_a210_line_ion_owners.py"
coverage_script="$bundle/check_a210_line_saturation_per_ion_coverage.py"
intersection_script="$bundle/compare_a210_line_saturation_intersection.py"
reference_path_file="$bundle/reference_stderr_path.txt"
requested_te="$(<"$bundle/requested_diag_te_K.txt")"
owner_shells="$(<"$bundle/line_ion_owner_shells.txt")"
lock="$control/LINE_SATURATION_PER_ION_COVERAGE_V4_LOCK"

mkdir -p "$control"
exec >>"$log" 2>&1
say() { printf 'LINE_SATURATION_PER_ION_V4 utc=%s %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$*"; }
if [[ -f "$verdict" ]] && grep -Eq '^status=(PASS|UNDERCOVERED)$' "$verdict"; then
  say "status=ALREADY_COMPLETE"
  exit 0
fi
mkdir "$lock" 2>/dev/null || {
  say "status=BLOCKED reason=ANOTHER_V4_MONITOR_OWNS_LOCK"
  exit 4
}
trap 'rmdir "$lock" 2>/dev/null || true' EXIT
say "status=START bundle=$bundle"

verify_bundle() {
  (cd "$bundle" && sha256sum -c POSTPROCESS_MANIFEST.sha256) >/dev/null
}
verify_bundle || {
  say "status=BLOCKED reason=POSTPROCESS_BUNDLE_SHA_DRIFT"
  exit 4
}

while [[ ! -f "$v3_verdict" ]]; do
  if [[ -f "$v3_log" ]] && \
     tail -n 1 "$v3_log" | grep -q 'status=BLOCKED'; then
    say "status=BLOCKED reason=V3_DID_NOT_PASS detail=$(tail -n 1 "$v3_log")"
    exit 4
  fi
  sleep 20
done
for exact in \
  'status=PASS' \
  'execution_order=AFTER_V2_PASS' \
  'arithmetic_bound_is_physical_tolerance=0' \
  'physical_mutation=0' \
  'floor=0' 'cap=0' 'clamp=0' 'jitter=0' 'repair=0'; do
  grep -qx "$exact" "$v3_verdict" || {
    say "status=BLOCKED reason=V3_VERDICT_CONTRACT_MISMATCH field=$exact"
    exit 4
  }
done
for exact in \
  'status=PASS' 'model_rc=1' 'natural_result=RADEQ_NO_BRACKET' \
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
grep -qx "v2_summary_sha256=$summary_sha" "$v3_verdict" || {
  say "status=BLOCKED reason=V3_SUMMARY_SHA_MISMATCH"
  exit 4
}
verify_bundle || {
  say "status=BLOCKED reason=POSTPROCESS_BUNDLE_SHA_DRIFT_AFTER_V3"
  exit 4
}

set +e
python3 "$owner_script" \
  --log "$run_root/stderr.log" \
  --report "$owner_report" \
  --phase REQUESTED_TE \
  --expected-shells "$owner_shells" \
  --expected-temperature-K "$requested_te" \
  > "$run_root/line_saturation_owner_coverage_v4.stdout" \
  2> "$run_root/line_saturation_owner_coverage_v4.stderr"
owner_rc=$?
set -e
printf '%s\n' "$owner_rc" > "$run_root/line_saturation_owner_coverage_v4.rc"
[[ "$owner_rc" -eq 0 ]] || {
  say "status=BLOCKED reason=OWNER_SUMMARY_FAILED rc=$owner_rc"
  exit 4
}

set +e
python3 "$coverage_script" \
  --saturation-summary "$summary" \
  --owner-report "$owner_report" \
  --report "$coverage_report" \
  > "$run_root/line_saturation_per_ion_coverage_v4.stdout" \
  2> "$run_root/line_saturation_per_ion_coverage_v4.stderr"
coverage_rc=$?
set -e
printf '%s\n' "$coverage_rc" > "$run_root/line_saturation_per_ion_coverage_v4.rc"

if [[ "$coverage_rc" -eq 0 ]]; then
  status=PASS
  rerun=0
elif [[ "$coverage_rc" -eq 4 ]] && \
     grep -q '"verdict": "COMBINED_PREFIX_UNDERCOVERS_AT_LEAST_ONE_TARGET_ION"' \
       "$coverage_report"; then
  status=UNDERCOVERED
  rerun=1
else
  say "status=BLOCKED reason=PER_ION_COVERAGE_CHECK_FAILED rc=$coverage_rc"
  exit 4
fi

intersection_sha=NONE
if [[ -f "$reference_path_file" ]]; then
  reference_stderr="$(<"$reference_path_file")"
  [[ "$reference_stderr" = /* && -f "$reference_stderr" && \
     ! -L "$reference_stderr" && -x "$intersection_script" ]] || {
    say "status=BLOCKED reason=UNSAFE_INTERSECTION_INPUT"
    exit 4
  }
  [[ "$(sha256sum "$reference_stderr" | cut -d' ' -f1)" == \
     "$(<"$bundle/reference_stderr.sha256")" ]] || {
    say "status=BLOCKED reason=REFERENCE_STDERR_SHA_DRIFT"
    exit 4
  }
  [[ "$status" == PASS ]] || {
    say "status=BLOCKED reason=UNION_COVERAGE_DID_NOT_PASS"
    exit 4
  }
  intersection_report="$run_root/a210_line_saturation_intersection_v4.json"
  set +e
  python3 "$intersection_script" \
    --reference-log "$reference_stderr" \
    --candidate-log "$run_root/stderr.log" \
    --report "$intersection_report" \
    > "$run_root/line_saturation_intersection_v4.stdout" \
    2> "$run_root/line_saturation_intersection_v4.stderr"
  intersection_rc=$?
  set -e
  printf '%s\n' "$intersection_rc" \
    > "$run_root/line_saturation_intersection_v4.rc"
  [[ "$intersection_rc" -eq 0 ]] || {
    say "status=BLOCKED reason=INTERSECTION_COMPARISON_FAILED rc=$intersection_rc"
    exit 4
  }
  intersection_sha="$(sha256sum "$intersection_report" | cut -d' ' -f1)"
fi

owner_sha="$(sha256sum "$owner_report" | cut -d' ' -f1)"
coverage_sha="$(sha256sum "$coverage_report" | cut -d' ' -f1)"
temporary="$verdict.tmp.$$"
printf '%s\n' \
  "status=$status" \
  "postprocess_bundle=$bundle" \
  "v2_summary_sha256=$summary_sha" \
  "owner_report_sha256=$owner_sha" \
  "coverage_report_sha256=$coverage_sha" \
  "intersection_report_sha256=$intersection_sha" \
  "required_fraction_each_ion=0.9" \
  "rerun_with_per_ion_union_required=$rerun" \
  "arithmetic_bound_is_physical_tolerance=0" \
  "physical_mutation=0" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0" \
  > "$temporary"
mv "$temporary" "$verdict"
say "status=$status coverage_report_sha256=$coverage_sha rerun_required=$rerun"
