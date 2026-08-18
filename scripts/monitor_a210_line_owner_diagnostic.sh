#!/usr/bin/env bash
# Detached postprocessor for one manual A2-10 line-owner diagnostic.
set -euo pipefail
umask 027

[[ $# -eq 4 ]] || {
  printf 'usage: %s RUN_ROOT EXPECTED_TE_K EXPECTED_SHELLS SUMMARIZER_SHA256\n' "$0" >&2
  exit 70
}
run_root="$1"
expected_te="$2"
expected_shells="$3"
expected_summarizer_sha="$4"
control="$run_root/manual_control"
log="$control/line_owner_monitor.log"
summarizer="$run_root/input/summarize_a210_line_ion_owners.py"

mkdir -p "$control"
exec >>"$log" 2>&1
say() { printf 'LINE_OWNER_MONITOR utc=%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$*"; }
say "status=START summarizer_sha256=$expected_summarizer_sha"

# Observe the manual supervisor lifecycle without controlling it.
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
actual_summarizer_sha="$(sha256sum "$summarizer" | cut -d' ' -f1)"
[[ "$actual_summarizer_sha" == "$expected_summarizer_sha" ]] || {
  say "status=BLOCKED reason=SUMMARIZER_SHA_DRIFT actual=$actual_summarizer_sha"
  exit 4
}

set +e
python3 "$summarizer" \
  --log "$run_root/stderr.log" \
  --report "$run_root/a210_line_ion_owner_report.json" \
  --phase REQUESTED_TE \
  --expected-shells "$expected_shells" \
  --expected-temperature-K "$expected_te" \
  > "$run_root/line_ion_owner_summary.stdout" \
  2> "$run_root/line_ion_owner_summary.stderr"
summary_rc=$?
set -e
printf '%s\n' "$summary_rc" > "$run_root/line_ion_owner_summary.rc"
if [[ -f "$run_root/a210_line_ion_owner_report.json" ]]; then
  report_sha="$(sha256sum "$run_root/a210_line_ion_owner_report.json" | cut -d' ' -f1)"
else
  report_sha=NONE
fi
case "$summary_rc" in
  0) say "status=PASS report_sha256=$report_sha" ;;
  4) say "status=BLOCKED_INCOMPLETE_CALLBACK report_sha256=$report_sha" ;;
  *) say "status=ERROR summary_rc=$summary_rc report_sha256=$report_sha" ;;
esac
exit "$summary_rc"
