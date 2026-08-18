#!/usr/bin/env bash
# Run the sealed completion audit after the tripwire and R1 reference monitor
# both close successfully.  Inputs and the auditor are hash-pinned at launch.
set -euo pipefail

if [[ "$#" -ne 8 ]]; then
  printf 'usage: %s RUN_ROOT REFINEMENT_JSON REFINEMENT_SHA256 PROOF_JSON PROOF_SHA256 AUDITOR_SHA256 EXPECTED_REFINEMENTS REPORT\n' "$0" >&2
  exit 64
fi

run_root="$1"
refinement="$2"
expected_refinement_sha="$3"
proof_baseline="$4"
expected_proof_sha="$5"
expected_auditor_sha="$6"
expected_refinements="$7"
report="$8"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
auditor="$repo_root/scripts/finalize_a210_nonoverlap_gate.py"
manual="$run_root/manual_control"
active="$manual/COMPLETION_MONITOR_ACTIVE"
log="$manual/completion_monitor.log"

[[ "$run_root" == /gpfs/* && "$run_root" != /gpfs && "$run_root" != /gpfs/ ]] || exit 64
[[ "$expected_refinement_sha" =~ ^[0-9a-f]{64}$ ]] || exit 64
[[ "$expected_proof_sha" =~ ^[0-9a-f]{64}$ ]] || exit 64
[[ "$expected_auditor_sha" =~ ^[0-9a-f]{64}$ ]] || exit 64
[[ "$expected_refinements" =~ ^[1-9][0-9]*$ && "$expected_refinements" -le 64 ]] || exit 64
[[ -d "$manual" && -f "$refinement" && -f "$proof_baseline" && -f "$auditor" ]] || exit 66
[[ ! -e "$active" && ! -e "$report" ]] || exit 73

: > "$active"
cleanup() { rm -f "$active"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'COMPLETION_MONITOR utc=%s status=START auditor_sha256=%s refinements=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$expected_auditor_sha" \
  "$expected_refinements" >> "$log"

while [[ -e "$manual/SUPERVISOR_ACTIVE" ||
         -e "$manual/POST_GATE_MONITOR_ACTIVE" ]]; do
  sleep 5
done

if [[ ! -f "$run_root/r1_k24_reference_comparison.json" ]] ||
   ! tail -n 1 "$manual/post_gate_monitor.log" |
      grep -q 'status=PASS report=r1_k24_reference_comparison.json'; then
  printf 'COMPLETION_MONITOR utc=%s status=NO_REFERENCE_PASS\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> "$log"
  exit 4
fi

actual_refinement_sha="$(sha256sum "$refinement" | cut -d' ' -f1)"
actual_proof_sha="$(sha256sum "$proof_baseline" | cut -d' ' -f1)"
actual_auditor_sha="$(sha256sum "$auditor" | cut -d' ' -f1)"
if [[ "$actual_refinement_sha" != "$expected_refinement_sha" ||
      "$actual_proof_sha" != "$expected_proof_sha" ||
      "$actual_auditor_sha" != "$expected_auditor_sha" ]]; then
  printf 'COMPLETION_MONITOR utc=%s status=INPUT_SHA_MISMATCH refinement=%s proof=%s auditor=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$actual_refinement_sha" \
    "$actual_proof_sha" "$actual_auditor_sha" >> "$log"
  exit 5
fi

set +e
python3 "$auditor" \
  --run-root "$run_root" \
  --refinement-comparison "$refinement" \
  --proof-witness-baseline "$proof_baseline" \
  --expected-refinements "$expected_refinements" \
  --report "$report" >> "$log" 2>&1
audit_rc=$?
set -e
if [[ "$audit_rc" -ne 0 ]]; then
  printf 'COMPLETION_MONITOR utc=%s status=AUDIT_FAIL rc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$audit_rc" >> "$log"
  exit "$audit_rc"
fi

report_sha="$(sha256sum "$report" | cut -d' ' -f1)"
printf 'COMPLETION_MONITOR utc=%s status=PASS report=%s report_sha256=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$report" "$report_sha" >> "$log"
