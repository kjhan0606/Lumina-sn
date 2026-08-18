#!/usr/bin/env bash
#SBATCH --job-name=cmf_epoch_g5m
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A40:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:20:00

set -euo pipefail
umask 027

die() {
  printf 'CMF_EPOCH_G5M_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 1 ]] || die "usage: $0 ABSOLUTE_RUN_ROOT"
run_root="$1"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs ]] || \
  die "unsafe run root: $run_root"
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || \
  die "must run under Slurm"
input_dir="$run_root/input"
binary="$input_dir/selftest_cmf_exact_epoch_scan"
[[ -x "$binary" && -f "$input_dir/manifest.sha256" ]] || \
  die "incomplete staged input"

module load cuda/13.0.2 2>/dev/null || module load cuda 2>/dev/null || \
  die "CUDA module unavailable"
(
  cd "$input_dir"
  sha256sum -c manifest.sha256
) > "$run_root/input_verify.log" 2>&1 || die "input SHA mismatch"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,uuid,driver_version,compute_cap \
  --format=csv,noheader,nounits > "$run_root/gpu.csv"
[[ "$(wc -l < "$run_root/gpu.csv")" -eq 4 ]] || \
  die "expected exactly four Slurm-visible GPUs"
printf '%s\n' \
  "job_id=$SLURM_JOB_ID" \
  "node=$SLURM_JOB_NODELIST" \
  "partition=${SLURM_JOB_PARTITION:-unknown}" \
  "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}" \
  > "$run_root/execution_environment.txt"

set +e
"$binary" --multi-device > "$run_root/multi_a.log" 2> "$run_root/multi_a.err"
rc_a=$?
"$binary" --multi-device > "$run_root/multi_b.log" 2> "$run_root/multi_b.err"
rc_b=$?
set -e
[[ "$rc_a" -eq 0 && "$rc_b" -eq 0 ]] || \
  die "multi-device test failed rc_a=$rc_a rc_b=$rc_b"
cmp -s "$run_root/multi_a.log" "$run_root/multi_b.log" || \
  die "multi-device stdout differs"
cmp -s "$run_root/multi_a.err" "$run_root/multi_b.err" || \
  die "multi-device stderr differs"
rg -q '^CMF_EXACT_EPOCH_MULTIDEVICE PASS device_counts=1/2/4 ' \
  "$run_root/multi_a.log" || die "PASS marker missing"

command -v compute-sanitizer >/dev/null || die "compute-sanitizer unavailable"
set +e
compute-sanitizer --tool memcheck --leak-check full --error-exitcode=79 \
  "$binary" --multi-device \
  > "$run_root/memcheck.log" 2> "$run_root/memcheck.err"
memcheck_rc=$?
set -e
[[ "$memcheck_rc" -eq 0 ]] || die "memcheck failed rc=$memcheck_rc"
rg -q 'ERROR SUMMARY: 0 errors' \
  "$run_root/memcheck.log" "$run_root/memcheck.err" || \
  die "memcheck zero-error marker missing"
if rg -q ' FAIL |numerical_repairs=[1-9]|floor=[1-9]|cap=[1-9]|clamp=[1-9]|jitter=[1-9]' \
    "$run_root/multi_a.log" "$run_root/multi_a.err" \
    "$run_root/multi_b.log" "$run_root/multi_b.err" \
    "$run_root/memcheck.log" "$run_root/memcheck.err"; then
  die "failure or numerical repair marker present"
fi

sha256sum "$run_root/gpu.csv" "$run_root/execution_environment.txt" \
  "$run_root/input_verify.log" "$run_root/multi_a.log" \
  "$run_root/multi_a.err" "$run_root/multi_b.log" \
  "$run_root/multi_b.err" "$run_root/memcheck.log" \
  "$run_root/memcheck.err" > "$run_root/output.sha256"
printf '%s\n' \
  "CMF_EPOCH_G5M_OK job_id=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST devices=4" \
  "binary_sha256=$(sha256sum "$binary" | cut -d' ' -f1)" \
  "source_sha256=$(sha256sum "$input_dir/cmf_exact_epoch_scan_selftest.cu" | cut -d' ' -f1)" \
  "device_counts=1/2/4 repeated_byte_identical=yes canonical_ray_reduction=yes" \
  "compute_sanitizer_errors=0 numerical_repairs=0 floor=0 cap=0 clamp=0 jitter=0" \
  > "$run_root/RUN_FOOTER.txt"
