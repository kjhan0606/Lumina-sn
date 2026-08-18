#!/usr/bin/env bash
#SBATCH --job-name=cmf_mgpu_split
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

set -euo pipefail
umask 027

die() {
  printf 'CMF_MGPU_REDUCED_SPLIT_FLIGHT_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 2 || $# -eq 3 ]] || \
  die "usage: $0 ABSOLUTE_RUN_ROOT one|four [weighted|equal]"
run_root="$1"
mode="$2"
partition_mode="${3:-weighted}"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs ]] || \
  die "unsafe run root: $run_root"
[[ "$mode" = one || "$mode" = four ]] || die "invalid mode: $mode"
[[ "$partition_mode" = weighted || "$partition_mode" = equal ]] || \
  die "invalid partition mode: $partition_mode"
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || \
  die "must run under Slurm"
input_dir="$run_root/input"
side_dir="$run_root/$mode"
[[ $# -eq 3 ]] && side_dir="$run_root/$partition_mode"
binary="$input_dir/bench_cmf_exact_multigpu_reduced"
geometry="$input_dir/geometry.csv"
[[ -x "$binary" && -f "$geometry" && -f "$input_dir/manifest.sha256" ]] || \
  die "incomplete staged input"
mkdir -p "$side_dir"

module load cuda/13.0.2 2>/dev/null || module load cuda 2>/dev/null || \
  die "CUDA module unavailable"
command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
(
  cd "$input_dir"
  sha256sum -c manifest.sha256
) > "$side_dir/input_verify.log" 2>&1 || die "input SHA mismatch"

devices=1
[[ "$mode" = four ]] && devices=4
if [[ -n "${CMF_MGPU_DEVICE_ORDER:-}" ]]; then
  [[ "$mode" = four ]] || die "device reordering requires four mode"
  allocated_device_order="$CUDA_VISIBLE_DEVICES"
  [[ "$CMF_MGPU_DEVICE_ORDER" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || \
    die "invalid CMF_MGPU_DEVICE_ORDER"
  [[ "$(tr ',' '\n' <<< "$CMF_MGPU_DEVICE_ORDER" | sort -u | wc -l)" -eq 4 ]] || \
    die "CMF_MGPU_DEVICE_ORDER must be a permutation"
  while IFS= read -r requested_device; do
    [[ ",$allocated_device_order," = *",$requested_device,"* ]] || \
      die "requested device is outside Slurm allocation: $requested_device"
  done < <(tr ',' '\n' <<< "$CMF_MGPU_DEVICE_ORDER")
  export CUDA_VISIBLE_DEVICES="$CMF_MGPU_DEVICE_ORDER"
fi
nvidia-smi --query-gpu=index,name,memory.total,uuid,driver_version,compute_cap \
  --format=csv,noheader,nounits > "$side_dir/gpu.csv"
[[ "$(wc -l < "$side_dir/gpu.csv")" -ge "$devices" ]] || \
  die "fewer GPUs visible than requested"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die "CUDA_VISIBLE_DEVICES is unset"
printf '%s\n' \
  "job_id=$SLURM_JOB_ID" \
  "node=$SLURM_JOB_NODELIST" \
  "partition=${SLURM_JOB_PARTITION:-unknown}" \
  "cuda_visible_devices=$CUDA_VISIBLE_DEVICES" \
  "requested_device_order=${CMF_MGPU_DEVICE_ORDER:-native}" \
  "allocated_device_order=${allocated_device_order:-$CUDA_VISIBLE_DEVICES}" \
  "mode=$mode" \
  "ray_partition=$partition_mode" \
  "devices=$devices" \
  "reduced_bins=${CMF_MGPU_REDUCED_BINS:-8192}" \
  "apply_diagnostic=${CMF_MGPU_REDUCED_APPLY_DIAGNOSTIC:-0}" \
  "compute_sanitizer=${CMF_MGPU_REDUCED_COMPUTE_SANITIZER:-0}" \
  "epoch_frequency_parallel=${CMF_MGPU_REDUCED_EPOCH:-0}" \
  "epoch_block=${CMF_MGPU_EPOCH_BLOCK:-128}" \
  "epoch_batch=${CMF_MGPU_EPOCH_BATCH:-64}" \
  "epoch_replay=${CMF_MGPU_EPOCH_REPLAY:-32}" \
  "coefficient_contract=finite_synthetic_not_cmfgen_physics" \
  > "$side_dir/execution_environment.txt"

export CMF_MGPU_REDUCED_BINS="${CMF_MGPU_REDUCED_BINS:-8192}"
export CMF_MGPU_REDUCED_DEVICES=4
export CMF_MGPU_REDUCED_REFINEMENTS=1
export CMF_MGPU_REDUCED_MODE="$mode"
export CMF_MGPU_REDUCED_PARTITION="$partition_mode"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

trace="$side_dir/vram_trace.csv"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used \
  --format=csv,noheader,nounits --loop-ms=200 > "$trace" 2>&1 &
sampler_pid=$!
stop_sampler() {
  if kill -0 "$sampler_pid" 2>/dev/null; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
trap stop_sampler EXIT

set +e
runner=()
if [[ "${CMF_MGPU_REDUCED_COMPUTE_SANITIZER:-0}" = 1 ]]; then
  command -v compute-sanitizer >/dev/null || \
    die "compute-sanitizer unavailable"
  runner=(compute-sanitizer --tool memcheck --error-exitcode=99)
fi
CMF_MGPU_REDUCED_OUTPUT="$side_dir/result.bin" \
  "${runner[@]}" "$binary" "$geometry" \
  > "$side_dir/bench.log" 2> "$side_dir/bench.err"
rc=$?
set -e
stop_sampler
trap - EXIT
[[ "$rc" -eq 0 ]] || die "benchmark failed mode=$mode rc=$rc"
if [[ "${CMF_MGPU_REDUCED_APPLY_DIAGNOSTIC:-0}" = 1 ]]; then
  rg '^CMF_MGPU_REDUCED_APPLY_DIAG PASS ' "$side_dir/bench.log" \
    > "$side_dir/result.txt" || die "diagnostic PASS marker missing"
  if rg -q 'FAIL|numerical_repairs=[1-9]' \
      "$side_dir/bench.log" "$side_dir/bench.err"; then
    die "diagnostic failure or numerical repair marker present"
  fi
  python3 "$input_dir/summarize_nvidia_smi_vram.py" "$trace" \
    --expected-devices "$devices" \
    > "$side_dir/vram_summary.txt" || die "VRAM summary failed"
  sha256sum "$side_dir/gpu.csv" "$side_dir/execution_environment.txt" \
    "$side_dir/input_verify.log" "$side_dir/bench.log" \
    "$side_dir/bench.err" "$side_dir/result.txt" \
    "$trace" "$side_dir/vram_summary.txt" > "$side_dir/output.sha256"
  printf '%s\n' \
    "CMF_MGPU_REDUCED_SPLIT_DIAGNOSTIC_OK job_id=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST mode=$mode partition=$partition_mode devices=$devices" \
    "binary_sha256=$(sha256sum "$binary" | cut -d' ' -f1)" \
    "repair=0 floor=0 clamp=0 jitter=0" \
    > "$side_dir/RUN_FOOTER.txt"
  exit 0
fi
rg "^CMF_MGPU_REDUCED_SPLIT_RESULT PASS mode=$mode " "$side_dir/bench.log" \
  > "$side_dir/result.txt"
if rg -q 'FAIL|numerical_repairs=[1-9]' \
    "$side_dir/bench.log" "$side_dir/bench.err"; then
  die "failure or numerical repair marker present"
fi
python3 "$input_dir/summarize_nvidia_smi_vram.py" "$trace" \
  --expected-devices "$devices" \
  > "$side_dir/vram_summary.txt" || die "VRAM summary failed"

sha256sum "$side_dir/gpu.csv" "$side_dir/execution_environment.txt" \
  "$side_dir/input_verify.log" "$side_dir/bench.log" \
  "$side_dir/bench.err" "$side_dir/result.bin" "$side_dir/result.txt" \
  "$trace" "$side_dir/vram_summary.txt" > "$side_dir/output.sha256"
printf '%s\n' \
  "CMF_MGPU_REDUCED_SPLIT_FLIGHT_OK job_id=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST mode=$mode partition=$partition_mode devices=$devices" \
  "binary_sha256=$(sha256sum "$binary" | cut -d' ' -f1)" \
  "result_sha256=$(sha256sum "$side_dir/result.bin" | cut -d' ' -f1)" \
  "coefficient_contract=finite_synthetic_not_cmfgen_physics" \
  "repair=0 floor=0 clamp=0 jitter=0" \
  > "$side_dir/RUN_FOOTER.txt"
