#!/usr/bin/env bash
# Run one sealed DET flight outside Slurm on actually idle GPUs, but yield the
# node immediately when Slurm starts allocating it or another process opens any
# selected physical GPU.  Only this launcher's process group is terminated.
set -euo pipefail
umask 027
SYN101_DEADLINE_EPOCH_FILE="/gpfs/kjhan/lumina/syn101_manual_deadline_0900_epoch.txt"

die() {
  printf 'MANUAL_DET_TRIPWIRE_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -ge 1 && $# -le 5 ]] ||
  die "usage: $0 ABSOLUTE_RUN_ROOT [GPU_INDICES] [POLL_SECONDS] [CPU_THREADS] [CPU_SET]"

run_root="$1"
gpu_indices="${2:-7}"
poll_seconds="${3:-2}"
cpu_threads="${4:-8}"
cpu_set="${5:-}"
node="$(hostname -s)"
today_0900_epoch() {
  local today
  today="$(TZ=Asia/Seoul date +%Y-%m-%d)"
  TZ=Asia/Seoul date -d "${today} 09:00:00" +%s
}

syn101_deadline_epoch() {
  local deadline_file_epoch=""
  if [[ -n "${SYN101_MANUAL_DEADLINE_EPOCH:-}" ]]; then
    deadline_file_epoch="$SYN101_MANUAL_DEADLINE_EPOCH"
  elif [[ -r "$SYN101_DEADLINE_EPOCH_FILE" ]]; then
    deadline_file_epoch="$(tr -dc '0-9' < "$SYN101_DEADLINE_EPOCH_FILE")"
  fi
  if [[ "$deadline_file_epoch" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$deadline_file_epoch"
    return
  fi
  today_0900_epoch
}

# ★2026-08-19 user 지시 "syn101 수동 제출은 금지. 해당 노드는 정상 운영중."
#   syn101 은 slurm 정상 운영 노드로 복귀했다.  수동 실행을 **전면 거부**한다.
#   (08-18 의 시각 기반 컷오프는 폐지됐고, 그 자리를 이 전면 금지가 대체한다.)
#   syn101 에 일이 필요하면 slurm 으로 제출한다.
enforce_syn101_manual_deadline() {
  [[ "$node" == "syn101" ]] || return 0
  die "manual launch denied: syn101 is under normal slurm operation (user 2026-08-19). Submit via slurm."
}

[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs ]] ||
  die "unsafe run root: $run_root"
[[ "$gpu_indices" =~ ^[0-9]+(,[0-9]+)*$ ]] ||
  die "invalid GPU indices: $gpu_indices"
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] ||
  die "invalid poll interval: $poll_seconds"
[[ "$cpu_threads" =~ ^[1-9][0-9]*$ ]] ||
  die "invalid CPU thread count: $cpu_threads"
[[ -z "$cpu_set" || "$cpu_set" =~ ^[0-9]+([,-][0-9]+)*$ ]] ||
  die "invalid CPU set: $cpu_set"
if [[ -n "$cpu_set" ]]; then
  command -v taskset >/dev/null || die "taskset unavailable"
  taskset -c "$cpu_set" true >/dev/null 2>&1 ||
    die "CPU set is unavailable: $cpu_set"
fi
[[ -x "$run_root/input/job.slurm" && -f "$run_root/READY" ]] ||
  die "sealed run root is not READY: $run_root"
[[ ! -e "$run_root/manual_control/SUPERVISOR_ACTIVE" ]] ||
  die "a manual supervisor is already active"
enforce_syn101_manual_deadline

control_dir="$run_root/manual_control"
mkdir -p "$control_dir"
exec >>"$control_dir/supervisor.log" 2>&1

say() {
  printf 'MANUAL_DET_TRIPWIRE utc=%s %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$*"
}

IFS=',' read -r -a gpu_index_list <<<"$gpu_indices"
gpu_uuid_list=()
declare -A seen_gpu_indices=()
for gpu_index in "${gpu_index_list[@]}"; do
  [[ -z "${seen_gpu_indices[$gpu_index]:-}" ]] ||
    die "duplicate GPU index: $gpu_index"
  seen_gpu_indices[$gpu_index]=1
  gpu_uuid="$(nvidia-smi -i "$gpu_index" --query-gpu=uuid \
    --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ "$gpu_uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]] ||
    die "failed to resolve physical GPU UUID for index $gpu_index"
  gpu_uuid_list+=("$gpu_uuid")
done
gpu_uuid_csv="$(IFS=,; printf '%s' "${gpu_uuid_list[*]}")"

gpu_pids() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
    2>/dev/null | awk -F',[[:space:]]*' -v selected="$gpu_uuid_csv" '
      BEGIN { n=split(selected, values, ","); for (i=1; i<=n; ++i) wanted[values[i]]=1 }
      wanted[$1] {print $2}'
}

running_jobs() {
  squeue -h -w "$node" -t RUNNING -o '%i' 2>/dev/null || true
}

node_allocation_state() {
  local row cpu_alloc alloc_tres
  if ! row="$(scontrol show node "$node" -o 2>/dev/null)" || [[ -z "$row" ]]; then
    printf 'UNKNOWN\n'
    return
  fi
  cpu_alloc="$(sed -n 's/.* CPUAlloc=\([0-9][0-9]*\) .*/\1/p' <<< "$row")"
  alloc_tres="$(sed -n 's/.* AllocTRES=\(.*\) CurrentWatts=.*/\1/p' <<< "$row")"
  if [[ ! "$cpu_alloc" =~ ^[0-9]+$ ]]; then
    printf 'UNKNOWN\n'
  elif [[ "$cpu_alloc" -ne 0 || -n "$alloc_tres" ]]; then
    printf 'ALLOCATED\n'
  else
    printf 'CLEAR\n'
  fi
}

is_our_descendant() {
  local pid="$1" ppid loops=0
  while [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 && "$loops" -lt 64 ]]; do
    [[ "$pid" -eq "$child_pid" ]] && return 0
    [[ -r "/proc/$pid/stat" ]] || return 1
    ppid="$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null || true)"
    [[ "$ppid" =~ ^[0-9]+$ && "$ppid" != "$pid" ]] || return 1
    pid="$ppid"
    loops=$((loops + 1))
  done
  return 1
}

foreign_gpu_pids() {
  local pid
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    is_our_descendant "$pid" || printf '%s\n' "$pid"
  done < <(gpu_pids)
}

terminate_ours() {
  local reason="$1" i observed_pgid
  observed_pgid="$(ps -o pgid= -p "$child_pid" 2>/dev/null | tr -d '[:space:]')"
  if [[ "$observed_pgid" =~ ^[0-9]+$ ]]; then
    child_pgid="$observed_pgid"
  fi
  say "action=YIELD reason=$reason child_pid=$child_pid pgid=$child_pgid"
  {
    printf 'utc=%s\nreason=%s\nchild_pid=%s\nprocess_group=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$reason" \
      "$child_pid" "$child_pgid"
    printf 'running_jobs=%s\n' "$(running_jobs | paste -sd, -)"
    for gpu_index in "${gpu_index_list[@]}"; do
      nvidia-smi -i "$gpu_index" --query-gpu=index,uuid,memory.used,utilization.gpu \
        --format=csv,noheader || true
    done
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader || true
  } >"$control_dir/YIELDED"

  kill -TERM -- "-$child_pgid" 2>/dev/null || true
  for i in {1..10}; do
    kill -0 "$child_pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$child_pid" 2>/dev/null; then
    say "action=KILL_AFTER_TERM_TIMEOUT child_pid=$child_pid"
    kill -KILL -- "-$child_pgid" 2>/dev/null || true
  fi
  wait "$child_pid" 2>/dev/null || true
  rm -f "$control_dir/SUPERVISOR_ACTIVE"
  say "status=YIELDED supervisor_exit=75"
  exit 75
}

initial_jobs="$(running_jobs)"
[[ -z "$initial_jobs" ]] || die "Slurm job already RUNNING on $node: $initial_jobs"
initial_allocation_state="$(node_allocation_state)"
[[ "$initial_allocation_state" == CLEAR ]] ||
  die "Slurm allocation state is not clear on $node: $initial_allocation_state"
initial_gpu_pids="$(gpu_pids)"
[[ -z "$initial_gpu_pids" ]] ||
  die "GPU set $gpu_indices ($gpu_uuid_csv) already has compute PID(s): $initial_gpu_pids"

touch "$control_dir/SUPERVISOR_ACTIVE"
say "status=START node=$node gpu_indices=$gpu_indices gpu_uuids=$gpu_uuid_csv poll_s=$poll_seconds cpu_threads=$cpu_threads cpu_set=${cpu_set:-unbound} run_root=$run_root supervisor_pid=$$"
for gpu_index in "${gpu_index_list[@]}"; do
  nvidia-smi -i "$gpu_index" --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader | sed 's/^/MANUAL_DET_TRIPWIRE gpu_preflight=/'
done

manual_id="manual_${node}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
launch_prefix=()
[[ -z "$cpu_set" ]] || launch_prefix=(taskset -c "$cpu_set")
setsid "${launch_prefix[@]}" env \
  SLURM_JOB_ID="$manual_id" \
  SLURM_JOB_NODELIST="$node" \
  SLURM_CPUS_PER_TASK="$cpu_threads" \
  CUDA_VISIBLE_DEVICES="$gpu_indices" \
  bash "$run_root/input/job.slurm" "$run_root" \
  >"$control_dir/job_wrapper.stdout" \
  2>"$control_dir/job_wrapper.stderr" &
child_pid=$!
# setsid(1) runs asynchronously after fork.  Wait until the child has actually
# become its own process-group leader; recording the inherited supervisor PGID
# here would make the yield signal target the wrong group.
child_pgid=""
for _ in {1..50}; do
  child_pgid="$(ps -o pgid= -p "$child_pid" 2>/dev/null | tr -d '[:space:]')"
  [[ "$child_pgid" == "$child_pid" ]] && break
  sleep 0.1
done
[[ "$child_pgid" == "$child_pid" ]] ||
  die "child did not enter its own process group: pid=$child_pid pgid=${child_pgid:-unknown}"
printf '%s\n' "$child_pid" >"$control_dir/child.pid"
printf '%s\n' "$child_pgid" >"$control_dir/child.pgid"
printf '%s\n' "$$" >"$control_dir/supervisor.pid"
say "status=CHILD_STARTED child_pid=$child_pid pgid=$child_pgid manual_id=$manual_id"

stop_requested=0
allocation_probe_failures=0
trap 'stop_requested=1' HUP INT TERM
while kill -0 "$child_pid" 2>/dev/null; do
  if [[ "$stop_requested" -eq 1 || -e "$control_dir/STOP" ]]; then
    terminate_ours "operator_stop"
  fi

  jobs="$(running_jobs)"
  if [[ -n "$jobs" ]]; then
    terminate_ours "slurm_running_jobs:${jobs//$'\n'/,}"
  fi
  allocation_state="$(node_allocation_state)"
  case "$allocation_state" in
    ALLOCATED)
      terminate_ours "slurm_node_allocation_detected"
      ;;
    CLEAR)
      allocation_probe_failures=0
      ;;
    UNKNOWN)
      allocation_probe_failures=$((allocation_probe_failures + 1))
      say "status=SLURM_PROBE_RETRY count=$allocation_probe_failures"
      if [[ "$allocation_probe_failures" -ge 3 ]]; then
        terminate_ours "slurm_node_state_unknown_three_consecutive_probes"
      fi
      ;;
    *)
      terminate_ours "slurm_node_state_invalid:$allocation_state"
      ;;
  esac

  foreign="$(foreign_gpu_pids)"
  if [[ -n "$foreign" ]]; then
    terminate_ours "foreign_gpu_pids:${foreign//$'\n'/,}"
  fi
  sleep "$poll_seconds"
done

set +e
wait "$child_pid"
child_rc=$?
set -e
printf '%s\n' "$child_rc" >"$control_dir/child.rc"
rm -f "$control_dir/SUPERVISOR_ACTIVE"
if [[ "$child_rc" -eq 0 ]]; then
  touch "$control_dir/COMPLETED"
  say "status=COMPLETED child_rc=0"
  exit 0
fi
touch "$control_dir/FAILED"
say "status=FAILED child_rc=$child_rc"
exit "$child_rc"
