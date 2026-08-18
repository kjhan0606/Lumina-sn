#!/usr/bin/env bash
# Relaunch manual tripwire-flight on collision, with bounded retries.
# This is a thin control wrapper; the flight script already guarantees yield on
# foreign GPU activity and writes reasoned diagnostics.
set -euo pipefail
umask 027

usage() {
  printf 'usage: %s RUN_ROOT [GPU_INDICES] [POLL_SECONDS] [CPU_THREADS] [CPU_SET] [MAX_RETRIES] [RETRY_WAIT_S]\n' "$0" >&2
  exit 70
}

[[ $# -ge 1 && $# -le 7 ]] || usage

run_root="$1"
gpu_indices="${2:-7}"
poll_seconds="${3:-2}"
cpu_threads="${4:-24}"
cpu_set="${5:-}"
max_retries="${6:-3}"
retry_wait_seconds="${7:-120}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
launcher="$repo_root/scripts/run_manual_det_with_tripwire.sh"

[[ -n "$run_root" && "$run_root" = /* ]] || exit 70
[[ "$max_retries" =~ ^[0-9]+$ ]] || exit 70
[[ "$retry_wait_seconds" =~ ^[0-9]+$ ]] || exit 70
[[ "$poll_seconds" =~ ^[0-9]+$ ]] || exit 70
[[ "$cpu_threads" =~ ^[0-9]+$ ]] || exit 70
[[ "$gpu_indices" =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 70
[[ -x "$launcher" ]] || exit 70

attempt=0
while :; do
  attempt=$((attempt + 1))
  prior_yield_mtime="0"
  if [[ -f "$run_root/manual_control/YIELDED" ]]; then
    prior_yield_mtime="$(stat -c '%Y%N' "$run_root/manual_control/YIELDED" 2>/dev/null || echo 0)"
  fi
  stamp="$(date -u +%Y%m%dT%H%M%S_%3N)"
  log="$run_root/manual_control/watch_attempt_${attempt}_${stamp}.log"
  printf 'WATCHER_ATTEMPT_START utc=%s attempt=%s run_root=%s gpu=%s poll=%s cpus=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$attempt" "$run_root" "$gpu_indices" "$poll_seconds" "$cpu_threads" \
    | tee -a "$log"

  if ! bash "$launcher" "$run_root" "$gpu_indices" "$poll_seconds" "$cpu_threads" "$cpu_set" \
        >>"$log" 2>&1; then
    launcher_rc=$?
  else
    launcher_rc=0
  fi

  printf 'WATCHER_LAUNCH_END utc=%s attempt=%s rc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$attempt" "$launcher_rc" >>"$log"

  new_yield=0
  if [[ -f "$run_root/manual_control/YIELDED" ]]; then
    current_yield_mtime="$(stat -c '%Y%N' "$run_root/manual_control/YIELDED" 2>/dev/null || echo 0)"
    if [[ "$current_yield_mtime" != "$prior_yield_mtime" ]]; then
      new_yield=1
    fi
  fi

  if (( new_yield == 1 )); then
    reason="$(sed -n 's/^reason=//p' "$run_root/manual_control/YIELDED" | head -n 1)"
    if [[ "$reason" == foreign_gpu_pids:* ]]; then
      printf 'WATCHER_COLLISION utc=%s attempt=%s reason=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$attempt" "$reason" | tee -a "$log"
      if (( attempt >= max_retries )); then
        printf 'WATCHER_STOP utc=%s status=MAX_RETRIES_EXCEEDED reason=%s\n' \
          "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$reason" | tee -a "$log"
        exit 71
      fi
      printf 'WATCHER_WAIT utc=%s status=RETRY wait_seconds=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$retry_wait_seconds" | tee -a "$log"
      sleep "$retry_wait_seconds"
      continue
    fi
    printf 'WATCHER_FINAL utc=%s status=YIELDED_NONCOLLISION reason=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$reason" | tee -a "$log"
    exit 72
  fi

  if [[ -f "$run_root/manual_control/COMPLETED" || -f "$run_root/manual_control/FAILED" || -f "$run_root/model.rc" ]]; then
    child_rc="$(cat "$run_root/manual_control/child.rc" 2>/dev/null || echo "")"
    if [[ -n "$child_rc" ]]; then
      printf 'WATCHER_CHILD_RC utc=%s attempt=%s child_rc=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$attempt" "$child_rc" >>"$log"
    fi
    printf 'WATCHER_FINAL utc=%s status=TERMINATED_WITH_MARKERS completed=%s failed=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" \
      "$([[ -f "$run_root/manual_control/COMPLETED" ]] && echo 1 || echo 0)" \
      "$([[ -f "$run_root/manual_control/FAILED" ]] && echo 1 || echo 0)" >>"$log"
    exit "$launcher_rc"
  fi

done
