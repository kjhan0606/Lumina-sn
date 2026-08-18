#!/usr/bin/env bash
# Run one sealed A100x2 line-saturation diagnostic under Slurm, then execute
# the V2 -> V3 -> V4 read-only observers in their pre-registered order.
set -euo pipefail
umask 027

die() {
  printf 'A210_LINE_SATURATION_SLURM_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 1 ]] || die "usage: $0 ABSOLUTE_RUN_ROOT"
run_root="$1"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs ]] || \
  die "unsafe run root: $run_root"
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || \
  die "must run under Slurm"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"

gpu_spec="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
gpu_spec="${gpu_spec#gpu:}"
IFS=',' read -r -a assigned_gpus <<< "$gpu_spec"
[[ "${#assigned_gpus[@]}" -eq 2 ]] || \
  die "expected exactly two assigned GPUs, got: ${gpu_spec:-unset}"

gpu_inventory="$(nvidia-smi --query-gpu=index,uuid \
  --format=csv,noheader,nounits)"
compute_inventory="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
  --format=csv,noheader,nounits 2>/dev/null || true)"
assigned_uuids=()
for gpu_token in "${assigned_gpus[@]}"; do
  gpu_token="${gpu_token//[[:space:]]/}"
  if [[ "$gpu_token" =~ ^[0-9]+$ ]]; then
    gpu_uuid="$(awk -F', *' -v index="$gpu_token" \
      '$1 == index { print $2 }' <<< "$gpu_inventory")"
  elif [[ "$gpu_token" == GPU-* ]]; then
    gpu_uuid="$gpu_token"
  else
    die "unrecognized assigned GPU token: $gpu_token"
  fi
  [[ "$gpu_uuid" == GPU-* ]] || \
    die "failed to resolve assigned GPU token: $gpu_token"
  if awk -F', *' -v uuid="$gpu_uuid" '$1 == uuid { found=1 } END { exit !found }' \
      <<< "$compute_inventory"; then
    offenders="$(awk -F', *' -v uuid="$gpu_uuid" \
      '$1 == uuid { print $0 }' <<< "$compute_inventory")"
    die "assigned GPU already has a compute process: $offenders"
  fi
  assigned_uuids+=("$gpu_uuid")
done

control="$run_root/manual_control"
job="$run_root/input/job.slurm"
v2_bundle="$run_root/postprocess_linear_v2"
v3_bundle="$run_root/postprocess_roundoff_v3"
v4_bundle="$run_root/postprocess_per_ion_coverage_v4"
v2_monitor="$v2_bundle/monitor_a210_line_saturation_postprocess_v2.sh"
v3_monitor="$v3_bundle/monitor_a210_line_saturation_roundoff_v3.sh"
v4_monitor="$v4_bundle/monitor_a210_line_saturation_per_ion_coverage_v4.sh"

[[ -f "$run_root/READY" && -x "$job" ]] || die "unsealed run input"
for bundle in "$v2_bundle" "$v3_bundle" "$v4_bundle"; do
  [[ -f "$bundle/READY" ]] || die "unsealed observer bundle: $bundle"
done
for monitor in "$v2_monitor" "$v3_monitor" "$v4_monitor"; do
  [[ -x "$monitor" ]] || die "observer is not executable: $monitor"
done

mkdir -p "$control"
for marker in SUPERVISOR_ACTIVE COMPLETED FAILED YIELDED COLLISION; do
  [[ ! -e "$control/$marker" ]] || die "pre-existing control marker: $marker"
done
[[ ! -e "$run_root/model.rc" && ! -e "$run_root/child.rc" ]] || \
  die "run root already contains exit records"

pipeline_log="$control/slurm_pipeline.log"
exec >>"$pipeline_log" 2>&1
say() {
  printf 'A210_LINE_SATURATION_SLURM utc=%s %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" "$*"
}

active="$control/SUPERVISOR_ACTIVE"
cleanup_active() {
  rm -f "$active"
}
trap cleanup_active EXIT

say "status=START job_id=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST gpu_uuids=${assigned_uuids[*]}"
"$v2_monitor" "$run_root" "$v2_bundle" &
v2_pid=$!
"$v3_monitor" "$run_root" "$v3_bundle" &
v3_pid=$!
"$v4_monitor" "$run_root" "$v4_bundle" &
v4_pid=$!
printf '%s\n' "$v2_pid" > "$control/line_saturation_postprocess_v2.pid"
printf '%s\n' "$v3_pid" > "$control/line_saturation_roundoff_v3.pid"
printf '%s\n' "$v4_pid" > "$control/line_saturation_per_ion_coverage_v4.pid"
touch "$active"

set +e
"$job" "$run_root"
child_rc=$?
set -e
printf '%s\n' "$child_rc" > "$run_root/child.rc"
cleanup_active
if [[ "$child_rc" -eq 0 ]]; then
  touch "$control/COMPLETED"
else
  touch "$control/FAILED"
fi
say "status=MODEL_WRAPPER_EXIT child_rc=$child_rc"

set +e
wait "$v2_pid"
v2_rc=$?
wait "$v3_pid"
v3_rc=$?
wait "$v4_pid"
v4_rc=$?
set -e
printf '%s\n' "$v2_rc" > "$control/line_saturation_postprocess_v2.rc"
printf '%s\n' "$v3_rc" > "$control/line_saturation_roundoff_v3.rc"
printf '%s\n' "$v4_rc" > "$control/line_saturation_per_ion_coverage_v4.rc"
say "status=OBSERVERS_EXIT v2_rc=$v2_rc v3_rc=$v3_rc v4_rc=$v4_rc"

# The diagnostic model must fail closed on its physical no-bracket result.
# The sealed production wrapper consequently exits 70; observer success, not
# that wrapper status, is the completion criterion for this diagnostic job.
[[ "$child_rc" -eq 70 ]] || die "unexpected model wrapper rc=$child_rc"
[[ -f "$run_root/model.rc" && "$(<"$run_root/model.rc")" == 1 ]] || \
  die "diagnostic model did not exit with rc=1"
[[ "$v2_rc" -eq 0 && "$v3_rc" -eq 0 && "$v4_rc" -eq 0 ]] || \
  die "observer failure v2=$v2_rc v3=$v3_rc v4=$v4_rc"
grep -qx 'status=PASS' "$run_root/LINE_SATURATION_VERDICT_V2.txt" || \
  die "V2 did not pass"
grep -qx 'status=PASS' "$run_root/LINE_SATURATION_ROUNDOFF_VERDICT_V3.txt" || \
  die "V3 did not pass"
grep -Eq '^status=(PASS|UNDERCOVERED)$' \
  "$run_root/LINE_SATURATION_PER_ION_COVERAGE_VERDICT_V4.txt" || \
  die "V4 did not reach a registered verdict"

say "status=PIPELINE_COMPLETE"
trap - EXIT
exit 0
