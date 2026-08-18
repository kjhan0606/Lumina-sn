#!/usr/bin/env bash
#SBATCH --job-name=cmf_epoch_g01
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
umask 027

die() {
  printf 'CMF_EPOCH_G01_FATAL %s\n' "$*" >&2
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
command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
(
  cd "$input_dir"
  sha256sum -c manifest.sha256
) > "$run_root/input_verify.log" 2>&1 || die "input SHA mismatch"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,uuid,driver_version,compute_cap \
  --format=csv,noheader,nounits > "$run_root/gpu.csv"
[[ "$(wc -l < "$run_root/gpu.csv")" -eq 1 ]] || \
  die "expected exactly one Slurm-visible GPU"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die "CUDA_VISIBLE_DEVICES is unset"
printf '%s\n' \
  "job_id=$SLURM_JOB_ID" \
  "node=$SLURM_JOB_NODELIST" \
  "partition=${SLURM_JOB_PARTITION:-unknown}" \
  "cuda_visible_devices=$CUDA_VISIBLE_DEVICES" \
  > "$run_root/execution_environment.txt"

python3 "$input_dir/verify_cmf_exact_epoch_formula.py" \
  > "$run_root/structure.log" 2> "$run_root/structure.err" || \
  die "Python structural proof failed"
rg -q '^CMF_EXACT_EPOCH_FORMULA PASS cases=6588 ' \
  "$run_root/structure.log" || die "Python structural PASS marker missing"

set +e
"$binary" > "$run_root/selftest_a.log" 2> "$run_root/selftest_a.err"
rc_a=$?
"$binary" > "$run_root/selftest_b.log" 2> "$run_root/selftest_b.err"
rc_b=$?
set -e
[[ "$rc_a" -eq 0 && "$rc_b" -eq 0 ]] || \
  die "full selftest failed rc_a=$rc_a rc_b=$rc_b"
cmp -s "$run_root/selftest_a.log" "$run_root/selftest_b.log" || \
  die "repeated stdout differs"
cmp -s "$run_root/selftest_a.err" "$run_root/selftest_b.err" || \
  die "repeated stderr differs"
rg -q '^CMF_EXACT_EPOCH_SCAN_SELFTEST PASS profile=full base_cases=2196 mode_cases=6588 ' \
  "$run_root/selftest_a.log" || die "full PASS marker missing"

command -v compute-sanitizer >/dev/null || die "compute-sanitizer unavailable"
set +e
compute-sanitizer --tool memcheck --leak-check full --error-exitcode=79 \
  "$binary" --sanitizer-smoke \
  > "$run_root/memcheck.log" 2> "$run_root/memcheck.err"
memcheck_rc=$?
set -e
[[ "$memcheck_rc" -eq 0 ]] || \
  die "CUDA memcheck failed rc=$memcheck_rc"
rg -q '^CMF_EXACT_EPOCH_SCAN_SELFTEST PASS profile=sanitizer-smoke ' \
  "$run_root/memcheck.log" "$run_root/memcheck.err" || \
  die "sanitizer-smoke PASS marker missing"
rg -q 'ERROR SUMMARY: 0 errors' \
  "$run_root/memcheck.log" "$run_root/memcheck.err" || \
  die "CUDA memcheck zero-error marker missing"

if rg -q ' FAIL |numerical_repairs=[1-9]|floor=[1-9]|cap=[1-9]|clamp=[1-9]|jitter=[1-9]' \
    "$run_root/structure.log" "$run_root/structure.err" \
    "$run_root/selftest_a.log" "$run_root/selftest_a.err" \
    "$run_root/selftest_b.log" "$run_root/selftest_b.err" \
    "$run_root/memcheck.log" "$run_root/memcheck.err"; then
  die "failure or numerical repair marker present"
fi

sha256sum "$run_root/gpu.csv" "$run_root/execution_environment.txt" \
  "$run_root/input_verify.log" "$run_root/structure.log" \
  "$run_root/structure.err" "$run_root/selftest_a.log" \
  "$run_root/selftest_a.err" "$run_root/selftest_b.log" \
  "$run_root/selftest_b.err" "$run_root/memcheck.log" \
  "$run_root/memcheck.err" > "$run_root/output.sha256"
printf '%s\n' \
  "CMF_EPOCH_G01_OK job_id=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST devices=1" \
  "binary_sha256=$(sha256sum "$binary" | cut -d' ' -f1)" \
  "source_sha256=$(sha256sum "$input_dir/cmf_exact_epoch_scan_selftest.cu" | cut -d' ' -f1)" \
  "python_cases=6588 cuda_mode_cases=6588 repeated_byte_identical=yes" \
  "compute_sanitizer_errors=0 numerical_repairs=0 floor=0 cap=0 clamp=0 jitter=0" \
  > "$run_root/RUN_FOOTER.txt"
