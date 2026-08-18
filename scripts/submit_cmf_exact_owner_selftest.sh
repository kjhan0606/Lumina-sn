#!/usr/bin/env bash
set -euo pipefail
umask 027

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$repo_root/lumina_cuda"
driver="$repo_root/scripts/run_cmf_exact_owner_selftest.slurm"
[[ -x "$binary" && -f "$driver" ]] || {
  printf 'CMF_EXACT_OWNER_SUBMIT_FATAL missing binary/driver\n' >&2
  exit 70
}
binary_sha="$(sha256sum "$binary" | cut -d' ' -f1)"
stamp="$(date -u +%Y%m%dT%H%M%S%NZ)_p$$"
run_root="/gpfs/${USER}/lumina/cmf_exact_owner/owner_${stamp}_${binary_sha:0:12}"
[[ "$run_root" = /gpfs/"${USER}"/lumina/cmf_exact_owner/owner_* ]] || {
  printf 'CMF_EXACT_OWNER_SUBMIT_FATAL unsafe run root: %s\n' "$run_root" >&2
  exit 70
}
mkdir -p "$run_root/input"
install -m 0750 "$binary" "$run_root/input/lumina_cuda"
install -m 0750 "$driver" "$run_root/input/job.slurm"
install -m 0640 \
  "$repo_root/src/lumina_cmfgen.c" \
  "$repo_root/src/lumina_cmfgen.h" \
  "$repo_root/src/cmf_exact_multigpu.cu" \
  "$repo_root/src/cmf_exact_multigpu.h" \
  "$repo_root/src/cmf_exact_sliding.c" \
  "$repo_root/src/cmf_exact_sliding.h" \
  "$repo_root/src/cmf_error_envelope.c" \
  "$repo_root/src/cmf_error_envelope.h" \
  "$repo_root/Makefile" \
  "$run_root/input/"
(
  cd "$run_root/input"
  sha256sum * > manifest.sha256
)
job_id="$(sbatch --parsable \
  --partition=a40 --nodelist=syn07 --nodes=1 --ntasks=1 \
  --gres=gpu:A40:4 --cpus-per-task=16 --mem=32G --time=00:30:00 \
  --output="$run_root/slurm-%j.out" \
  --error="$run_root/slurm-%j.err" \
  "$run_root/input/job.slurm" "$run_root")"
printf 'CMF_EXACT_OWNER_SUBMITTED job_id=%s run_root=%s binary_sha256=%s\n' \
  "$job_id" "$run_root" "$binary_sha"
