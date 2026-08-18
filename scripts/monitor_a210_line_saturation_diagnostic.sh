#!/usr/bin/env bash
# Detached fail-closed postprocessor for one manual A2-10 saturation flight.
set -euo pipefail
umask 027

[[ $# -eq 1 ]] || {
  printf 'usage: %s RUN_ROOT\n' "$0" >&2
  exit 70
}
run_root="$1"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs ]] || exit 70
control="$run_root/manual_control"
input="$run_root/input"
log="$control/line_saturation_monitor.log"
summary_script="$input/summarize_a210_line_saturation.py"
compare_script="$input/compare_a210_cmfgen_line_saturation.py"
coordinate="$input/cmfgen_coordinate_reference.json"
netrate="$(<"$input/cmfgen_netrate_path.txt")"

mkdir -p "$control"
exec >>"$log" 2>&1
say() { printf 'LINE_SATURATION_MONITOR utc=%s %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$*"; }
say "status=START"

sha256sum -c "$input/flight_scripts.sha256" >/dev/null || {
  say "status=BLOCKED reason=STAGED_SCRIPT_SHA_DRIFT"
  exit 4
}
[[ "$(sha256sum "$netrate" | cut -d' ' -f1)" == \
   "$(<"$input/cmfgen_netrate.sha256")" ]] || {
  say "status=BLOCKED reason=CMFGEN_NETRATE_SHA_DRIFT"
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
grep -q '\[A2-10\]\[VECTOR-INTERIOR-SCAN\].*solver_result=RADEQ_NO_BRACKET' \
  "$run_root/stderr.log" || {
  say "status=BLOCKED reason=MISSING_NATURAL_NO_BRACKET"
  exit 4
}

set +e
python3 "$summary_script" \
  --log "$run_root/stderr.log" \
  --report "$run_root/a210_line_saturation_summary.json" \
  > "$run_root/line_saturation_summary.stdout" \
  2> "$run_root/line_saturation_summary.stderr"
summary_rc=$?
set -e
printf '%s\n' "$summary_rc" > "$run_root/line_saturation_summary.rc"
[[ "$summary_rc" -eq 0 ]] || {
  say "status=BLOCKED reason=SATURATION_SUMMARY_FAILED rc=$summary_rc"
  exit 4
}

set +e
python3 "$compare_script" \
  --summary "$run_root/a210_line_saturation_summary.json" \
  --netrate "$netrate" \
  --coordinate-reference "$coordinate" \
  --report "$run_root/a210_cmfgen_line_saturation_comparison.json" \
  > "$run_root/line_saturation_comparison.stdout" \
  2> "$run_root/line_saturation_comparison.stderr"
comparison_rc=$?
set -e
printf '%s\n' "$comparison_rc" > "$run_root/line_saturation_comparison.rc"
[[ "$comparison_rc" -eq 0 ]] || {
  say "status=BLOCKED reason=CMFGEN_SATURATION_COMPARISON_FAILED rc=$comparison_rc"
  exit 4
}

summary_sha="$(sha256sum "$run_root/a210_line_saturation_summary.json" | cut -d' ' -f1)"
comparison_sha="$(sha256sum "$run_root/a210_cmfgen_line_saturation_comparison.json" | cut -d' ' -f1)"
printf '%s\n' \
  "status=PASS" \
  "summary_sha256=$summary_sha" \
  "comparison_sha256=$comparison_sha" \
  "model_rc=1" \
  "natural_result=RADEQ_NO_BRACKET" \
  "physical_mutation=0" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0" \
  > "$run_root/LINE_SATURATION_VERDICT.txt"
say "status=PASS summary_sha256=$summary_sha comparison_sha256=$comparison_sha"
