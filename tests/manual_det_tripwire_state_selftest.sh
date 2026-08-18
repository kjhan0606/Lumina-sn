#!/usr/bin/env bash
# Exercise the scheduler-state parser from the actual manual tripwire script.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
tripwire="$repo_root/scripts/run_manual_det_with_tripwire.sh"
[[ -f "$tripwire" ]] || {
  printf 'FAIL MANUAL_DET_TRIPWIRE_STATE reason=missing_tripwire\n'
  exit 4
}

function_text="$(sed -n '/^node_allocation_state()/,/^}/p' "$tripwire")"
[[ "$function_text" == *'node_allocation_state()'* ]] || {
  printf 'FAIL MANUAL_DET_TRIPWIRE_STATE reason=missing_state_function\n'
  exit 4
}
eval "$function_text"

node=syn101
mock_row=''
mock_failure=0
scontrol() {
  if [[ "$mock_failure" -eq 1 ]]; then
    return 1
  fi
  printf '%s\n' "$mock_row"
}

passed=0
total=0
check_state() {
  local label="$1" expected="$2" actual
  total=$((total + 1))
  actual="$(node_allocation_state)"
  if [[ "$actual" != "$expected" ]]; then
    printf 'FAIL MANUAL_DET_TRIPWIRE_STATE case=%s expected=%s actual=%s\n' \
      "$label" "$expected" "$actual"
    exit 4
  fi
  passed=$((passed + 1))
}

mock_row='NodeName=syn101 CPUAlloc=0 CPUTot=64 State=IDLE+PLANNED CfgTRES=cpu=64,gres/gpu=8 AllocTRES= CurrentWatts=0'
check_state planned_without_allocation CLEAR

mock_row='NodeName=syn101 CPUAlloc=24 CPUTot=64 State=MIXED CfgTRES=cpu=64,gres/gpu=8 AllocTRES=cpu=24,gres/gpu=3 CurrentWatts=0'
check_state cpu_and_gpu_allocation ALLOCATED

mock_row='NodeName=syn101 CPUAlloc=0 CPUTot=64 State=MIXED CfgTRES=cpu=64,gres/gpu=8 AllocTRES=gres/gpu=2 CurrentWatts=0'
check_state gpu_only_allocation ALLOCATED

mock_row='NodeName=syn101 State=UNKNOWN AllocTRES= CurrentWatts=0'
check_state malformed_row UNKNOWN

mock_failure=1
check_state scontrol_failure UNKNOWN

printf 'PASS MANUAL_DET_TRIPWIRE_STATE cases=%s/%s\n' "$passed" "$total"
