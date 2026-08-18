#!/usr/bin/env bash
# Observe one manual requested-Te diagnostic through natural tripwire closure,
# then seal proof-only R1 identity, complete ion ownership, and the explicitly
# state-unmatched CMFGEN comparison.  This monitor never controls the model.
set -euo pipefail
umask 027

[[ $# -eq 12 ]] || {
  printf 'usage: %s RUN_ROOT REFERENCE_STDERR REFERENCE_STDERR_SHA EXPECTED_TE_K EXPECTED_SHELLS COMPARATOR_SHA SUMMARIZER_SHA OWNER_COMPARE_SHA CMFGEN_OWNER CMFGEN_OWNER_SHA CMFGEN_FINITE CMFGEN_FINITE_SHA\n' "$0" >&2
  exit 64
}
run_root="$1"
reference_stderr="$2"
reference_stderr_sha="$3"
expected_te="$4"
expected_shells="$5"
comparator_sha="$6"
summarizer_sha="$7"
owner_compare_sha="$8"
cmfgen_owner="$9"
cmfgen_owner_sha="${10}"
cmfgen_finite="${11}"
cmfgen_finite_sha="${12}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
comparator="$repo_root/scripts/compare_a210_targeted_reference.py"
summarizer="$repo_root/scripts/summarize_a210_line_ion_owners.py"
owner_compare="$repo_root/scripts/compare_a210_cmfgen_ion_owners.py"
control="$run_root/manual_control"
active="$control/line_owner_closure_monitor.active"
log="$control/line_owner_closure_monitor.log"

[[ "$run_root" == /gpfs/* && "$run_root" != /gpfs && "$run_root" != /gpfs/ ]] || exit 64
[[ "$expected_shells" =~ ^[1-9][0-9]*$ ]] || exit 64
for value in "$reference_stderr_sha" "$comparator_sha" "$summarizer_sha" \
             "$owner_compare_sha" "$cmfgen_owner_sha" "$cmfgen_finite_sha"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || exit 64
done
[[ -d "$control" && -f "$reference_stderr" && -f "$cmfgen_owner" &&
   -f "$cmfgen_finite" && -f "$comparator" && -f "$summarizer" &&
   -f "$owner_compare" ]] || exit 66
[[ ! -e "$active" ]] || exit 73

: > "$active"
cleanup() { rm -f "$active"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec >>"$log" 2>&1
say() { printf 'LINE_OWNER_CLOSURE_MONITOR utc=%s %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$*"; }
say "status=START reference_refinements=30 candidate_refinements=36"

while [[ -e "$control/SUPERVISOR_ACTIVE" ]]; do sleep 20; done
if [[ -e "$control/YIELDED" ]] || grep -Eq 'status=(YIELD|COLLISION)' \
     "$control/supervisor.log"; then
  say "status=YIELDED_OR_COLLISION"
  exit 75
fi
if [[ ! -e "$control/COMPLETED" && ! -e "$control/FAILED" ]]; then
  say "status=BLOCKED reason=MISSING_NATURAL_CLOSURE_MARKER"
  exit 4
fi
[[ -f "$run_root/stderr.log" && -f "$run_root/model.rc" ]] || {
  say "status=BLOCKED reason=MISSING_MODEL_OUTPUT"
  exit 4
}
[[ "$(tr -d '[:space:]' < "$run_root/model.rc")" == 1 ]] || {
  say "status=BLOCKED reason=UNEXPECTED_MODEL_RC"
  exit 4
}
if grep -Eq '(^|[[:space:]])(physical_values_modified|floor|clamp|jitter|repair)=[1-9][0-9]*([[:space:]]|$)' \
     "$run_root/stdout.log" "$run_root/stderr.log"; then
  say "status=BLOCKED reason=PHYSICAL_REPAIR_MARKER"
  exit 4
fi

check_sha() {
  local path="$1" expected="$2" label="$3" actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    say "status=BLOCKED reason=${label}_SHA_MISMATCH actual=$actual"
    exit 4
  fi
}
check_sha "$reference_stderr" "$reference_stderr_sha" REFERENCE_STDERR
check_sha "$comparator" "$comparator_sha" COMPARATOR
check_sha "$summarizer" "$summarizer_sha" SUMMARIZER
check_sha "$owner_compare" "$owner_compare_sha" OWNER_COMPARE
check_sha "$cmfgen_owner" "$cmfgen_owner_sha" CMFGEN_OWNER
check_sha "$cmfgen_finite" "$cmfgen_finite_sha" CMFGEN_FINITE

python3 "$comparator" \
  --reference-stderr "$reference_stderr" \
  --candidate-stderr "$run_root/stderr.log" \
  --reference-occurrence 0 --candidate-occurrence 0 \
  --reference-refinements 30 --candidate-refinements 36 \
  --report "$run_root/r1_k30_k36_proof_comparison.json"
proof_sha="$(sha256sum "$run_root/r1_k30_k36_proof_comparison.json" | cut -d' ' -f1)"
say "status=R1_PROOF_PASS report_sha256=$proof_sha"

set +e
python3 "$summarizer" \
  --log "$run_root/stderr.log" \
  --report "$run_root/a210_line_ion_owner_report_strict.json" \
  --phase REQUESTED_TE --expected-shells "$expected_shells" \
  --expected-temperature-K "$expected_te"
summary_rc=$?
set -e
owner_sha="$(sha256sum "$run_root/a210_line_ion_owner_report_strict.json" | cut -d' ' -f1)"
if [[ "$summary_rc" -eq 4 ]]; then
  say "status=BLOCKED_INCOMPLETE_CALLBACK report_sha256=$owner_sha"
  exit 4
elif [[ "$summary_rc" -ne 0 ]]; then
  say "status=ERROR summary_rc=$summary_rc report_sha256=$owner_sha"
  exit "$summary_rc"
fi
say "status=OWNER_PASS report_sha256=$owner_sha"

python3 "$owner_compare" \
  --lumina-owner "$run_root/a210_line_ion_owner_report_strict.json" \
  --cmfgen-owner "$cmfgen_owner" --cmfgen-finite "$cmfgen_finite" \
  --shell 0 --depth-lo 67 --depth-hi 68 \
  --report "$run_root/a210_cmfgen_ion_owner_comparison.json"
comparison_sha="$(sha256sum "$run_root/a210_cmfgen_ion_owner_comparison.json" | cut -d' ' -f1)"
say "status=PASS comparison=FINITE_COMPARISON_STATE_UNMATCHED report_sha256=$comparison_sha physical_values_modified=0 floor=0 cap=0 clamp=0 jitter=0 repair=0"
