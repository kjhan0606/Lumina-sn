#!/usr/bin/env bash
# Wait for the strict requested-Te owner closure, then produce the separately
# sealed CMFGEN emission/absorption comparison.  This observer never controls
# or signals the model/supervisor process group.
set -euo pipefail
umask 027

[[ $# -eq 7 ]] || {
  printf 'usage: %s RUN_ROOT COMPARATOR_SHA OWNER_DEP_SHA CMFGEN_COMPONENTS CMFGEN_COMPONENTS_SHA CMFGEN_FINITE CMFGEN_FINITE_SHA\n' "$0" >&2
  exit 64
}
run_root="$1"
comparator_sha="$2"
owner_dep_sha="$3"
cmfgen_components="$4"
cmfgen_components_sha="$5"
cmfgen_finite="$6"
cmfgen_finite_sha="$7"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
comparator="$repo_root/scripts/compare_a210_cmfgen_ion_components.py"
owner_dependency="$repo_root/scripts/compare_a210_cmfgen_ion_owners.py"
control="$run_root/manual_control"
closure_active="$control/line_owner_closure_monitor.active"
closure_log="$control/line_owner_closure_monitor.log"
active="$control/line_owner_component_monitor.active"
log="$control/line_owner_component_monitor.log"
owner_report="$run_root/a210_line_ion_owner_report_strict.json"
report="$run_root/a210_cmfgen_ion_component_comparison.json"

[[ "$run_root" == /gpfs/* && "$run_root" != /gpfs && "$run_root" != /gpfs/ ]] || exit 64
for value in "$comparator_sha" "$owner_dep_sha" \
             "$cmfgen_components_sha" "$cmfgen_finite_sha"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || exit 64
done
[[ -d "$control" && -f "$closure_log" && -f "$comparator" &&
   -f "$owner_dependency" && -f "$cmfgen_components" &&
   -f "$cmfgen_finite" ]] || exit 66
[[ ! -e "$active" ]] || exit 73

: > "$active"
cleanup() { rm -f "$active"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
exec >>"$log" 2>&1
say() { printf 'LINE_OWNER_COMPONENT_MONITOR utc=%s %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$*"; }
say "status=START comparison=FINITE_COMPONENT_COMPARISON_STATE_UNMATCHED"

while [[ -e "$closure_active" ]]; do sleep 20; done
if [[ -e "$control/SUPERVISOR_ACTIVE" ]]; then
  say "status=BLOCKED reason=CLOSURE_MONITOR_EXITED_BEFORE_SUPERVISOR"
  exit 4
fi
closure_last="$(tail -n 1 "$closure_log")"
if [[ "$closure_last" != *"status=PASS comparison=FINITE_COMPARISON_STATE_UNMATCHED"* ]]; then
  say "status=BLOCKED reason=STRICT_OWNER_CLOSURE_NOT_PASS"
  exit 4
fi
[[ -f "$owner_report" ]] || {
  say "status=BLOCKED reason=MISSING_STRICT_OWNER_REPORT"
  exit 4
}

check_sha() {
  local path="$1" expected="$2" label="$3" actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    say "status=BLOCKED reason=${label}_SHA_MISMATCH actual=$actual"
    exit 4
  fi
}
check_sha "$comparator" "$comparator_sha" COMPONENT_COMPARATOR
check_sha "$owner_dependency" "$owner_dep_sha" OWNER_DEPENDENCY
check_sha "$cmfgen_components" "$cmfgen_components_sha" CMFGEN_COMPONENTS
check_sha "$cmfgen_finite" "$cmfgen_finite_sha" CMFGEN_FINITE

python3 "$comparator" \
  --lumina-owner "$owner_report" \
  --cmfgen-components "$cmfgen_components" \
  --cmfgen-finite "$cmfgen_finite" \
  --shell 0 --depth-lo 67 --depth-hi 68 \
  --report "$report"
report_sha="$(sha256sum "$report" | cut -d' ' -f1)"
say "status=PASS comparison=FINITE_COMPONENT_COMPARISON_STATE_UNMATCHED report_sha256=$report_sha physical_values_modified=0 floor=0 cap=0 clamp=0 jitter=0 repair=0"
