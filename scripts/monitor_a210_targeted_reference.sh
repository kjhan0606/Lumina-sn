#!/usr/bin/env bash
# After a detached targeted gate closes, compare its first (R1) publication
# with the sealed reference.  The comparator and final stderr are both hashed
# only after the tripwire supervisor has removed its active marker.
set -euo pipefail

if [[ "$#" -ne 4 && "$#" -ne 6 ]]; then
  printf 'usage: %s RUN_ROOT REFERENCE_STDERR COMPARATOR_SHA256 REPORT_NAME [REFERENCE_REFINEMENTS CANDIDATE_REFINEMENTS]\n' "$0" >&2
  exit 64
fi

run_root="$1"
reference_stderr="$2"
expected_comparator_sha="$3"
report_name="$4"
reference_refinements="${5:-}"
candidate_refinements="${6:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
comparator="$repo_root/scripts/compare_a210_targeted_reference.py"
manual="$run_root/manual_control"
active="$manual/POST_GATE_MONITOR_ACTIVE"
log="$manual/post_gate_monitor.log"

[[ "$run_root" == /gpfs/* && "$run_root" != /gpfs && "$run_root" != /gpfs/ ]] || exit 64
[[ "$report_name" =~ ^[A-Za-z0-9._-]+\.json$ ]] || exit 64
[[ "$expected_comparator_sha" =~ ^[0-9a-f]{64}$ ]] || exit 64
if [[ -n "$reference_refinements" || -n "$candidate_refinements" ]]; then
  [[ "$reference_refinements" =~ ^[1-9][0-9]*$ &&
     "$candidate_refinements" =~ ^[1-9][0-9]*$ &&
     "$reference_refinements" -lt "$candidate_refinements" &&
     "$candidate_refinements" -le 64 ]] || exit 64
fi
[[ -d "$manual" && -f "$reference_stderr" && -f "$comparator" ]] || exit 66
[[ ! -e "$active" ]] || exit 73

: > "$active"
cleanup() { rm -f "$active"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'POST_GATE_MONITOR utc=%s status=START comparator_sha256=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$expected_comparator_sha" >> "$log"

while [[ -e "$manual/SUPERVISOR_ACTIVE" ]]; do
  sleep 5
done

if [[ ! -f "$run_root/TARGETED_GATE_VERDICT.txt" ]] ||
   ! grep -q 'A210_TARGETED_GATE_ACCEPT status=PASS' \
      "$run_root/TARGETED_GATE_VERDICT.txt"; then
  printf 'POST_GATE_MONITOR utc=%s status=NO_GATE_PASS\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> "$log"
  exit 4
fi

actual_comparator_sha="$(sha256sum "$comparator" | cut -d' ' -f1)"
if [[ "$actual_comparator_sha" != "$expected_comparator_sha" ]]; then
  printf 'POST_GATE_MONITOR utc=%s status=COMPARATOR_SHA_MISMATCH actual=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$actual_comparator_sha" >> "$log"
  exit 5
fi

proof_args=()
if [[ -n "$reference_refinements" ]]; then
  proof_args=(
    --reference-refinements "$reference_refinements"
    --candidate-refinements "$candidate_refinements"
  )
fi
python3 "$comparator" \
  --reference-stderr "$reference_stderr" \
  --candidate-stderr "$run_root/stderr.log" \
  --reference-occurrence 0 \
  --candidate-occurrence 0 \
  "${proof_args[@]}" \
  --report "$run_root/$report_name" >> "$log" 2>&1
printf 'POST_GATE_MONITOR utc=%s status=PASS report=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$report_name" >> "$log"
