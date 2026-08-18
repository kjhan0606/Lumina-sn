#!/usr/bin/env bash
set -euo pipefail
umask 027

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$repo_root/selftest_cmf_exact_multigpu"
driver="$repo_root/scripts/run_cmf_exact_multigpu_prototype.slurm"
[[ -x "$binary" ]] || {
  printf 'missing binary: %s\n' "$binary" >&2
  exit 2
}
[[ -f "$driver" ]] || {
  printf 'missing driver: %s\n' "$driver" >&2
  exit 2
}

binary_sha="$(sha256sum "$binary" | cut -d' ' -f1)"
stamp="$(date -u +%Y%m%dT%H%M%S%NZ)_p$$"
run_root="/gpfs/${USER}/lumina/cmf_multigpu_prototype/mgpu_${stamp}_${binary_sha:0:12}"
[[ "$run_root" = /gpfs/"${USER}"/lumina/cmf_multigpu_prototype/mgpu_* ]] || {
  printf 'unsafe run root: %s\n' "$run_root" >&2
  exit 2
}
mkdir -p "$run_root/input"
install -m 0750 "$binary" "$run_root/input/selftest_cmf_exact_multigpu"
install -m 0640 "$repo_root/src/cmf_exact_multigpu.cu" \
  "$repo_root/src/cmf_exact_multigpu.h" \
  "$repo_root/src/cmf_exact_sliding.c" \
  "$repo_root/src/cmf_exact_sliding.h" \
  "$repo_root/src/cmf_error_envelope.c" \
  "$repo_root/src/cmf_error_envelope.h" \
  "$repo_root/tests/cmf_exact_multigpu_selftest.cu" \
  "$run_root/input/"
install -m 0750 "$driver" "$run_root/input/job.slurm"
(
  cd "$run_root/input"
  sha256sum cmf_error_envelope.c cmf_error_envelope.h \
    cmf_exact_multigpu.cu cmf_exact_multigpu.h \
    cmf_exact_sliding.c cmf_exact_sliding.h \
    cmf_exact_multigpu_selftest.cu selftest_cmf_exact_multigpu job.slurm \
    > manifest.sha256
)

node_args=()
if [[ -n "${CMF_MGPU_NODELIST:-}" ]]; then
  [[ "$CMF_MGPU_NODELIST" =~ ^[A-Za-z0-9._-]+$ ]] || {
    printf 'invalid CMF_MGPU_NODELIST: %s\n' "$CMF_MGPU_NODELIST" >&2
    exit 2
  }
  node_args=(--nodelist="$CMF_MGPU_NODELIST")
fi
job_id="$(sbatch --parsable \
  "${node_args[@]}" \
  --partition=a40 --nodes=1 --ntasks=1 --gres=gpu:A40:4 \
  --cpus-per-task=16 --mem=32G --time=00:30:00 \
  --output="$run_root/slurm-%j.out" \
  --error="$run_root/slurm-%j.err" \
  "$run_root/input/job.slurm" "$run_root")"
printf 'CMF_MGPU_SUBMITTED job_id=%s run_root=%s binary_sha256=%s\n' \
  "$job_id" "$run_root" "$binary_sha"
