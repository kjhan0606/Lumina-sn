#!/usr/bin/env bash
set -euo pipefail
umask 027

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$repo_root/bench_cmf_exact_multigpu_reduced"
driver="$repo_root/scripts/run_cmf_exact_multigpu_reduced.slurm"
summary="$repo_root/scripts/summarize_nvidia_smi_vram.py"
geometry="$repo_root/data/tardis_reference_toy06_19p48d/geometry.csv"
for path in "$binary" "$driver" "$summary" "$geometry"; do
  [[ -f "$path" ]] || { printf 'missing input: %s\n' "$path" >&2; exit 2; }
done
[[ -x "$binary" ]] || { printf 'binary is not executable: %s\n' "$binary" >&2; exit 2; }

binary_sha="$(sha256sum "$binary" | cut -d' ' -f1)"
stamp="$(date -u +%Y%m%dT%H%M%S%NZ)_p$$"
run_root="/gpfs/${USER}/lumina/cmf_multigpu_reduced/reduced_${stamp}_${binary_sha:0:12}"
[[ "$run_root" = /gpfs/"${USER}"/lumina/cmf_multigpu_reduced/reduced_* ]] || {
  printf 'unsafe run root: %s\n' "$run_root" >&2
  exit 2
}
mkdir -p "$run_root/input"
install -m 0750 "$binary" "$run_root/input/bench_cmf_exact_multigpu_reduced"
install -m 0640 "$repo_root/src/cmf_exact_multigpu.cu" \
  "$repo_root/src/cmf_exact_multigpu.h" \
  "$repo_root/src/cmf_exact_sliding.c" \
  "$repo_root/src/cmf_exact_sliding.h" \
  "$repo_root/src/cmf_error_envelope.c" \
  "$repo_root/src/cmf_error_envelope.h" \
  "$repo_root/tests/cmf_exact_multigpu_reduced_bench.cu" \
  "$summary" "$geometry" "$run_root/input/"
install -m 0750 "$driver" "$run_root/input/job.slurm"
(
  cd "$run_root/input"
  sha256sum bench_cmf_exact_multigpu_reduced \
    cmf_error_envelope.c cmf_error_envelope.h \
    cmf_exact_multigpu.cu cmf_exact_multigpu.h \
    cmf_exact_sliding.c cmf_exact_sliding.h \
    cmf_exact_multigpu_reduced_bench.cu geometry.csv \
    summarize_nvidia_smi_vram.py job.slurm > manifest.sha256
)

job_id="$(sbatch --parsable \
  --partition=a40 --nodes=1 --ntasks=1 --gres=gpu:A40:4 \
  --cpus-per-task=16 --mem=64G --time=01:00:00 \
  --output="$run_root/slurm-%j.out" \
  --error="$run_root/slurm-%j.err" \
  "$run_root/input/job.slurm" "$run_root")"
printf 'CMF_MGPU_REDUCED_SUBMITTED job_id=%s run_root=%s binary_sha256=%s\n' \
  "$job_id" "$run_root" "$binary_sha"
