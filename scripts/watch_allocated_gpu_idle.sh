#!/usr/bin/env bash
# Observe an allocated GPU without changing its state.  A card is only marked
# as a sustained-idle candidate after a configurable run of low-util samples.
set -euo pipefail
umask 027

die() {
  printf 'GPU_IDLE_WATCH_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -ge 5 && $# -le 8 ]] ||
  die "usage: $0 NODE GPU_INDEX GPU_UUID OWNER_JOB_ID LOG_FILE [INTERVAL_S] [IDLE_UTIL_PCT] [IDLE_SAMPLES]"

node="$1"
gpu_index="$2"
expected_uuid="$3"
owner_job_id="$4"
log_file="$5"
interval_seconds="${6:-30}"
idle_util_percent="${7:-2}"
idle_samples_required="${8:-20}"

[[ "$node" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid node"
[[ "$gpu_index" =~ ^[0-9]+$ ]] || die "invalid GPU index"
[[ "$expected_uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]] || die "invalid GPU UUID"
[[ "$owner_job_id" =~ ^[0-9]+(_[0-9]+)?$ ]] || die "invalid owner job id"
[[ "$interval_seconds" =~ ^[1-9][0-9]*$ ]] || die "invalid interval"
[[ "$idle_util_percent" =~ ^[0-9]+$ ]] || die "invalid idle threshold"
[[ "$idle_samples_required" =~ ^[1-9][0-9]*$ ]] || die "invalid idle sample count"

mkdir -p "$(dirname "$log_file")"
exec >> "$log_file" 2>&1

printf 'GPU_IDLE_WATCH_START utc=%s node=%s gpu_index=%s uuid=%s owner_job=%s interval_s=%s idle_util_pct=%s idle_samples=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$node" "$gpu_index" \
  "$expected_uuid" "$owner_job_id" "$interval_seconds" \
  "$idle_util_percent" "$idle_samples_required"

low_streak=0
idle_announced=0
while :; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  owner_state="$(squeue -h -j "$owner_job_id" -o '%T' 2>/dev/null || true)"
  [[ -n "$owner_state" ]] || owner_state=NOT_IN_QUEUE

  if ! gpu_row="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
      nvidia-smi -i "$gpu_index" \
      --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory,pstate \
      --format=csv,noheader,nounits 2>/dev/null)"; then
    printf 'GPU_IDLE_WATCH_SAMPLE utc=%s status=SSH_OR_NVIDIA_SMI_FAILURE owner_state=%s\n' \
      "$timestamp" "$owner_state"
    sleep "$interval_seconds"
    continue
  fi

  IFS=',' read -r observed_index observed_uuid memory_used memory_total \
    gpu_util memory_util pstate <<< "$gpu_row"
  observed_index="${observed_index//[[:space:]]/}"
  observed_uuid="${observed_uuid//[[:space:]]/}"
  memory_used="${memory_used//[[:space:]]/}"
  memory_total="${memory_total//[[:space:]]/}"
  gpu_util="${gpu_util//[[:space:]]/}"
  memory_util="${memory_util//[[:space:]]/}"
  pstate="${pstate//[[:space:]]/}"

  if [[ "$observed_index" != "$gpu_index" || "$observed_uuid" != "$expected_uuid" ]]; then
    printf 'GPU_IDLE_WATCH_STOP utc=%s status=GPU_IDENTITY_CHANGED observed_index=%s observed_uuid=%s owner_state=%s\n' \
      "$timestamp" "$observed_index" "$observed_uuid" "$owner_state"
    exit 71
  fi

  process_rows="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null | grep -F '$expected_uuid' || true" \
    2>/dev/null || true)"
  process_count=0
  [[ -z "$process_rows" ]] || process_count="$(printf '%s\n' "$process_rows" | wc -l)"

  if (( gpu_util <= idle_util_percent )); then
    low_streak=$((low_streak + 1))
  else
    low_streak=0
    idle_announced=0
  fi

  printf 'GPU_IDLE_WATCH_SAMPLE utc=%s owner_state=%s util_pct=%s memory_util_pct=%s memory_mib=%s/%s pstate=%s processes=%s low_streak=%s\n' \
    "$timestamp" "$owner_state" "$gpu_util" "$memory_util" "$memory_used" \
    "$memory_total" "$pstate" "$process_count" "$low_streak"

  if (( low_streak >= idle_samples_required && idle_announced == 0 )); then
    printf 'SUSTAINED_IDLE_CANDIDATE utc=%s node=%s gpu_index=%s uuid=%s owner_job=%s owner_state=%s util_threshold_pct=%s consecutive_samples=%s process_count=%s memory_used_mib=%s\n' \
      "$timestamp" "$node" "$gpu_index" "$expected_uuid" "$owner_job_id" \
      "$owner_state" "$idle_util_percent" "$low_streak" "$process_count" \
      "$memory_used"
    idle_announced=1
  fi

  if [[ "$owner_state" != RUNNING ]]; then
    printf 'GPU_IDLE_WATCH_STOP utc=%s status=OWNER_JOB_NOT_RUNNING owner_state=%s\n' \
      "$timestamp" "$owner_state"
    exit 0
  fi
  sleep "$interval_seconds"
done
