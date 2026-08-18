#!/usr/bin/env bash
set -euo pipefail
umask 027

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$repo_root/selftest_cmf_exact_epoch_scan"
driver="$repo_root/scripts/run_cmf_exact_epoch_multidevice.slurm"
source_file="$repo_root/tests/cmf_exact_epoch_scan_selftest.cu"
spec="$repo_root/docs/CMF_EXACT_WITHIN_RAY_SCAN_SPEC_2026-08-10.md"
for required in "$binary" "$driver" "$source_file" "$spec"; do
  [[ -f "$required" ]] || {
    printf 'missing staged input: %s\n' "$required" >&2
    exit 2
  }
done
[[ -x "$binary" ]] || {
  printf 'binary is not executable: %s\n' "$binary" >&2
  exit 2
}

binary_sha="$(sha256sum "$binary" | cut -d' ' -f1)"
stamp="$(date -u +%Y%m%dT%H%M%S%NZ)_p$$"
run_root="/gpfs/${USER}/lumina/cmf_exact_epoch_g5m/g5m_${stamp}_${binary_sha:0:12}"
[[ "$run_root" = /gpfs/"${USER}"/lumina/cmf_exact_epoch_g5m/g5m_* ]] || {
  printf 'unsafe run root: %s\n' "$run_root" >&2
  exit 2
}
mkdir -p "$run_root/input"
install -m 0750 "$binary" "$run_root/input/selftest_cmf_exact_epoch_scan"
install -m 0640 "$source_file" "$spec" "$run_root/input/"
install -m 0750 "$driver" "$run_root/input/job.slurm"
(
  cd "$run_root/input"
  sha256sum CMF_EXACT_WITHIN_RAY_SCAN_SPEC_2026-08-10.md \
    cmf_exact_epoch_scan_selftest.cu job.slurm \
    selftest_cmf_exact_epoch_scan > manifest.sha256
)

node_args=()
if [[ -n "${CMF_EPOCH_NODELIST:-}" ]]; then
  [[ "$CMF_EPOCH_NODELIST" =~ ^[A-Za-z0-9._-]+$ ]] || {
    printf 'invalid CMF_EPOCH_NODELIST: %s\n' "$CMF_EPOCH_NODELIST" >&2
    exit 2
  }
  node_args=(--nodelist="$CMF_EPOCH_NODELIST")
fi
job_id="$(sbatch --parsable \
  "${node_args[@]}" \
  --partition=a40 --nodes=1 --ntasks=1 --gres=gpu:A40:4 \
  --cpus-per-task=8 --mem=12G --time=00:20:00 \
  --output="$run_root/slurm-%j.out" \
  --error="$run_root/slurm-%j.err" \
  "$run_root/input/job.slurm" "$run_root")"
printf 'CMF_EPOCH_G5M_SUBMITTED job_id=%s run_root=%s binary_sha256=%s\n' \
  "$job_id" "$run_root" "$binary_sha"
