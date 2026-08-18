#!/usr/bin/env bash
set -euo pipefail
umask 027

selection="${1:-both}"
[[ "$selection" = one || "$selection" = four || "$selection" = both ]] || {
  printf 'usage: %s [one|four|both]\n' "$0" >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$repo_root/bench_cmf_exact_multigpu_reduced"
driver="$repo_root/scripts/run_cmf_exact_multigpu_reduced_split.slurm"
summary="$repo_root/scripts/summarize_nvidia_smi_vram.py"
compare="$repo_root/scripts/compare_cmf_exact_multigpu_reduced.py"
geometry="$repo_root/data/tardis_reference_toy06_19p48d/geometry.csv"
for path in "$binary" "$driver" "$summary" "$compare" "$geometry"; do
  [[ -f "$path" ]] || { printf 'missing input: %s\n' "$path" >&2; exit 2; }
done
[[ -x "$binary" ]] || { printf 'binary is not executable: %s\n' "$binary" >&2; exit 2; }

binary_sha="$(sha256sum "$binary" | cut -d' ' -f1)"
stamp="$(date -u +%Y%m%dT%H%M%S%NZ)_p$$"
run_root="/gpfs/${USER}/lumina/cmf_multigpu_reduced_split/split_${stamp}_${binary_sha:0:12}"
[[ "$run_root" = /gpfs/"${USER}"/lumina/cmf_multigpu_reduced_split/split_* ]] || {
  printf 'unsafe run root: %s\n' "$run_root" >&2
  exit 2
}
mkdir -p "$run_root/input" "$run_root/one" "$run_root/four"
install -m 0750 "$binary" "$run_root/input/bench_cmf_exact_multigpu_reduced"
install -m 0640 "$repo_root/src/cmf_exact_multigpu.cu" \
  "$repo_root/src/cmf_exact_multigpu.h" \
  "$repo_root/src/cmf_exact_sliding.c" \
  "$repo_root/src/cmf_exact_sliding.h" \
  "$repo_root/src/cmf_error_envelope.c" \
  "$repo_root/src/cmf_error_envelope.h" \
  "$repo_root/tests/cmf_exact_multigpu_reduced_bench.cu" \
  "$summary" "$compare" "$geometry" "$run_root/input/"
install -m 0750 "$driver" "$run_root/input/job.slurm"
(
  cd "$run_root/input"
  sha256sum bench_cmf_exact_multigpu_reduced \
    cmf_error_envelope.c cmf_error_envelope.h \
    cmf_exact_multigpu.cu cmf_exact_multigpu.h \
    cmf_exact_sliding.c cmf_exact_sliding.h \
    cmf_exact_multigpu_reduced_bench.cu geometry.csv \
    summarize_nvidia_smi_vram.py compare_cmf_exact_multigpu_reduced.py \
    job.slurm > manifest.sha256
)

one_job="not-submitted"
four_job="not-submitted"
node_args=()
if [[ -n "${CMF_MGPU_NODELIST:-}" ]]; then
  [[ "$CMF_MGPU_NODELIST" =~ ^[A-Za-z0-9._-]+$ ]] || {
    printf 'invalid CMF_MGPU_NODELIST: %s\n' "$CMF_MGPU_NODELIST" >&2
    exit 2
  }
  node_args=(--nodelist="$CMF_MGPU_NODELIST")
fi
allocated_gpus="${CMF_MGPU_ALLOCATED_GPUS:-4}"
[[ "$allocated_gpus" =~ ^[4-7]$ ]] || {
  printf 'invalid CMF_MGPU_ALLOCATED_GPUS: %s\n' "$allocated_gpus" >&2
  exit 2
}
time_limit="${CMF_MGPU_TIME_LIMIT:-01:00:00}"
[[ "$time_limit" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || {
  printf 'invalid CMF_MGPU_TIME_LIMIT: %s\n' "$time_limit" >&2
  exit 2
}
if [[ "$selection" = one || "$selection" = both ]]; then
  one_job="$(sbatch --parsable \
    "${node_args[@]}" \
    --partition=a40 --nodes=1 --ntasks=1 --gres=gpu:A40:1 \
    --cpus-per-task=4 --mem=16G --time=02:00:00 \
    --job-name=cmf_mgpu_one \
    --output="$run_root/one/slurm-%j.out" \
    --error="$run_root/one/slurm-%j.err" \
    "$run_root/input/job.slurm" "$run_root" one)"
fi
if [[ "$selection" = four || "$selection" = both ]]; then
  four_job="$(sbatch --parsable \
    "${node_args[@]}" \
    --partition=a40 --nodes=1 --ntasks=1 --gres=gpu:A40:"$allocated_gpus" \
    --cpus-per-task=16 --mem=64G --time="$time_limit" \
    --job-name=cmf_mgpu_four \
    --output="$run_root/four/slurm-%j.out" \
    --error="$run_root/four/slurm-%j.err" \
    "$run_root/input/job.slurm" "$run_root" four)"
fi
printf 'CMF_MGPU_REDUCED_SPLIT_SUBMITTED one_job=%s four_job=%s run_root=%s binary_sha256=%s\n' \
  "$one_job" "$four_job" "$run_root" "$binary_sha"
