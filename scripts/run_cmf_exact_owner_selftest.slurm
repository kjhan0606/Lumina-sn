#!/usr/bin/env bash
#SBATCH --job-name=cmf_exact_owner
#SBATCH --partition=a40
#SBATCH --nodelist=syn07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A40:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
umask 027

die() {
  printf 'CMF_EXACT_OWNER_FLIGHT_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 1 ]] || die "usage: $0 ABSOLUTE_RUN_ROOT"
run_root="$1"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs ]] || \
  die "unsafe run root: $run_root"
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || \
  die "must run under Slurm"
input_dir="$run_root/input"
binary="$input_dir/lumina_cuda"
[[ -x "$binary" && -f "$input_dir/manifest.sha256" ]] || \
  die "incomplete staged input"

module load cuda/13.0.2 2>/dev/null || module load cuda 2>/dev/null || \
  die "CUDA module unavailable"
(
  cd "$input_dir"
  sha256sum -c manifest.sha256
) > "$run_root/input_verify.log" 2>&1 || die "input SHA mismatch"
nvidia-smi --query-gpu=index,name,memory.total,uuid,driver_version,compute_cap \
  --format=csv,noheader,nounits > "$run_root/gpu.csv"
[[ "$(wc -l < "$run_root/gpu.csv")" -eq 4 ]] || \
  die "expected exactly four visible GPUs"

export LUMINA_CMF_EXACT_OWNER_SELFTEST=1
export LUMINA_CMF_FINE_MGPU_DEVICES=4
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
env | grep -E '^(LUMINA_|OMP_|CUDA_VISIBLE_DEVICES=)' | LC_ALL=C sort \
  > "$run_root/execution_environment.txt"

set +e
"$binary" > "$run_root/selftest_a.log" 2> "$run_root/selftest_a.err"
rc_a=$?
"$binary" > "$run_root/selftest_b.log" 2> "$run_root/selftest_b.err"
rc_b=$?
set -e
[[ "$rc_a" -eq 0 && "$rc_b" -eq 0 ]] || \
  die "owner selftest failed rc_a=$rc_a rc_b=$rc_b"
cmp -s "$run_root/selftest_a.log" "$run_root/selftest_b.log" || \
  die "stdout determinism mismatch"
cmp -s "$run_root/selftest_a.err" "$run_root/selftest_b.err" || \
  die "stderr determinism mismatch"
rg -q '^CMF_EXACT_OWNER_SELFTEST PASS devices=4 ' \
  "$run_root/selftest_a.log" || die "PASS marker missing"
if rg -q 'SELFTEST_FAIL|repair=[1-9]|floor=[1-9]|clamp=[1-9]|jitter=[1-9]' \
    "$run_root/selftest_a.log" "$run_root/selftest_a.err"; then
  die "failure or numerical-repair marker present"
fi

command -v compute-sanitizer >/dev/null || die "compute-sanitizer unavailable"
set +e
compute-sanitizer --tool memcheck --error-exitcode=79 \
  "$binary" > "$run_root/memcheck.log" 2> "$run_root/memcheck.err"
memcheck_rc=$?
set -e
[[ "$memcheck_rc" -eq 0 ]] || die "CUDA memcheck failed rc=$memcheck_rc"
rg -q 'ERROR SUMMARY: 0 errors' \
  "$run_root/memcheck.log" "$run_root/memcheck.err" || \
  die "CUDA memcheck zero-error marker missing"

sha256sum "$run_root/gpu.csv" "$run_root/execution_environment.txt" \
  "$run_root/input_verify.log" "$run_root/selftest_a.log" \
  "$run_root/selftest_a.err" "$run_root/memcheck.log" \
  "$run_root/memcheck.err" > "$run_root/output.sha256"
printf '%s\n' \
  "CMF_EXACT_OWNER_FLIGHT_OK job_id=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST" \
  "binary_sha256=$(sha256sum "$binary" | cut -d' ' -f1)" \
  "devices=4 transactional=1 repair=0 floor=0 clamp=0 jitter=0" \
  > "$run_root/RUN_FOOTER.txt"
