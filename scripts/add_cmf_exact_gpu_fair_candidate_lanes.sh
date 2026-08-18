#!/usr/bin/env bash
# Add H100x2 and A100x2 candidate lanes to an already staged fair-pair root.
set -euo pipefail
umask 027

die() {
  printf 'CMF_EXACT_GPU_CANDIDATE_ADD_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 1 ]] || die "usage: $0 ABSOLUTE_PAIR_ROOT"
pair_root="$1"
[[ "$pair_root" = /gpfs/"$USER"/lumina/cmf_exact_gpu_fair/pair_* ]] ||
  die "unsafe pair root: $pair_root"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_input="$pair_root/fair_h200x1/input"
driver="$repo_root/scripts/run_cmf_exact_gpu_fair.slurm"
[[ -d "$source_input" && -x "$source_input/lumina_cuda" && -x "$driver" ]] ||
  die "source lane or current driver is unavailable"

stage_lane() {
  local lane="$1"
  local run_root="$pair_root/fair_$lane"
  [[ ! -e "$run_root" ]] || die "candidate lane already exists: $run_root"
  mkdir -p "$run_root/input"
  while IFS= read -r -d '' source_file; do
    local base mode
    base="$(basename "$source_file")"
    [[ "$base" != manifest.sha256 && "$base" != job.slurm ]] || continue
    mode=0640
    [[ -x "$source_file" ]] && mode=0750
    install -m "$mode" "$source_file" "$run_root/input/$base"
  done < <(find "$source_input" -maxdepth 1 -type f -print0)
  install -m 0750 "$driver" "$run_root/input/job.slurm"
  (
    cd "$run_root/input"
    find . -maxdepth 1 -type f ! -name manifest.sha256 -printf '%P\0' |
      LC_ALL=C sort -z | xargs -0 sha256sum > manifest.sha256
  )
  printf 'READY\n' > "$run_root/READY"
}

stage_lane h100x2
stage_lane a100x2

h100_root="$pair_root/fair_h100x2"
a100_root="$pair_root/fair_a100x2"
h100_job="$(sbatch --parsable --job-name=cmf_h100x2 \
  --partition=h100 --nodes=1 --ntasks=1 --gres=gpu:H100:2 \
  --cpus-per-task=24 --mem=192G --time=02:30:00 \
  --output="$h100_root/slurm-%j.out" --error="$h100_root/slurm-%j.err" \
  "$h100_root/input/job.slurm" "$h100_root" 2 H100)"
a100_job="$(sbatch --parsable --job-name=cmf_a100x2 \
  --partition=a100 --nodes=1 --ntasks=1 --gres=gpu:A100:2 \
  --cpus-per-task=24 --mem=192G --time=02:30:00 \
  --output="$a100_root/slurm-%j.out" --error="$a100_root/slurm-%j.err" \
  "$a100_root/input/job.slurm" "$a100_root" 2 A100)"
printf '%s\n' "$h100_job" > "$h100_root/job_id.txt"
printf '%s\n' "$a100_job" > "$a100_root/job_id.txt"
printf 'CMF_EXACT_GPU_CANDIDATES_ADDED pair_root=%s h100_job=%s a100_job=%s\n' \
  "$pair_root" "$h100_job" "$a100_job" | tee "$pair_root/MULTIPARTITION_SUBMITTED.txt"
