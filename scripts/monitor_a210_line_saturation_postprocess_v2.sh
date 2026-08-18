#!/usr/bin/env bash
# Independent linear-time postprocessor for a completed saturation diagnostic.
# The physical run input and executable are never modified by this observer.
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
log="$control/line_saturation_postprocess_v2.log"
summary_script="$bundle/summarize_a210_line_saturation.py"
compare_script="$bundle/compare_a210_cmfgen_line_saturation.py"
coordinate="$bundle/cmfgen_coordinate_reference.json"
netrate="$(<"$bundle/cmfgen_netrate_path.txt")"

exec >>"$log" 2>&1
say() { printf 'LINE_SATURATION_POSTPROCESS_V2 utc=%s %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$*"; }
say "status=START bundle=$bundle"

verify_bundle() {
  (cd "$bundle" && sha256sum -c POSTPROCESS_MANIFEST.sha256) >/dev/null
  [[ "$(sha256sum "$netrate" | cut -d' ' -f1)" == \
     "$(<"$bundle/cmfgen_netrate.sha256")" ]]
}
verify_bundle || {
  say "status=BLOCKED reason=POSTPROCESS_BUNDLE_SHA_DRIFT"
  exit 4
}

while [[ ! -e "$control/SUPERVISOR_ACTIVE" && \
         ! -e "$control/COMPLETED" && ! -e "$control/FAILED" && \
         ! -e "$control/YIELDED" ]]; do
  sleep 5
done
while [[ -e "$control/SUPERVISOR_ACTIVE" ]]; do sleep 20; done

if [[ -e "$control/YIELDED" ]]; then
  say "status=YIELDED"
  exit 75
fi
[[ -f "$run_root/stderr.log" && -f "$run_root/model.rc" ]] || {
  say "status=BLOCKED reason=MISSING_MODEL_OUTPUT"
  exit 4
}
[[ "$(<"$run_root/model.rc")" == 1 ]] || {
  say "status=BLOCKED reason=UNEXPECTED_MODEL_RC value=$(<"$run_root/model.rc")"
  exit 4
}
grep -q '\[A2-10\]\[VECTOR-INTERIOR-SCAN\].*phase=REQUESTED_TE.*solver_result=RADEQ_NO_BRACKET' \
  "$run_root/stderr.log" || {
  say "status=BLOCKED reason=MISSING_REQUESTED_TE_NO_BRACKET"
  exit 4
}
verify_bundle || {
  say "status=BLOCKED reason=POSTPROCESS_BUNDLE_SHA_DRIFT_AFTER_RUN"
  exit 4
}

set +e
python3 "$summary_script" \
  --log "$run_root/stderr.log" \
  --report "$run_root/a210_line_saturation_summary_v2.json" \
  > "$run_root/line_saturation_summary_v2.stdout" \
  2> "$run_root/line_saturation_summary_v2.stderr"
summary_rc=$?
set -e
printf '%s\n' "$summary_rc" > "$run_root/line_saturation_summary_v2.rc"
[[ "$summary_rc" -eq 0 ]] || {
  say "status=BLOCKED reason=SATURATION_SUMMARY_FAILED rc=$summary_rc"
  exit 4
}

set +e
python3 "$compare_script" \
  --summary "$run_root/a210_line_saturation_summary_v2.json" \
  --netrate "$netrate" \
  --coordinate-reference "$coordinate" \
  --report "$run_root/a210_cmfgen_line_saturation_comparison_v2.json" \
  > "$run_root/line_saturation_comparison_v2.stdout" \
  2> "$run_root/line_saturation_comparison_v2.stderr"
comparison_rc=$?
set -e
printf '%s\n' "$comparison_rc" > "$run_root/line_saturation_comparison_v2.rc"
[[ "$comparison_rc" -eq 0 ]] || {
  say "status=BLOCKED reason=CMFGEN_SATURATION_COMPARISON_FAILED rc=$comparison_rc"
  exit 4
}

summary_sha="$(sha256sum "$run_root/a210_line_saturation_summary_v2.json" | cut -d' ' -f1)"
comparison_sha="$(sha256sum "$run_root/a210_cmfgen_line_saturation_comparison_v2.json" | cut -d' ' -f1)"
printf '%s\n' \
  "status=PASS" \
  "postprocess_bundle=$bundle" \
  "summary_sha256=$summary_sha" \
  "comparison_sha256=$comparison_sha" \
  "model_rc=1" \
  "natural_result=RADEQ_NO_BRACKET" \
  "physical_mutation=0" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0" \
  > "$run_root/LINE_SATURATION_VERDICT_V2.txt"
say "status=PASS summary_sha256=$summary_sha comparison_sha256=$comparison_sha"
