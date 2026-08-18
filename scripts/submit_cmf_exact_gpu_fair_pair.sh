#!/usr/bin/env bash
# Stage one fat binary and submit the same sealed validation concurrently to
# H200x1/H100x2/A100x2/A40x4.  The lower-memory devices use two or four shards;
# this is an allocation choice only and does not change the sealed physics.
set -euo pipefail
umask 027

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$repo_root/lumina_cuda"
driver="$repo_root/scripts/run_cmf_exact_gpu_fair.slurm"
checker="$repo_root/scripts/check_cmf_exact_cmfgen_ab.py"
sealed_source="${CMF_EXACT_CMFGEN_SEALED_SOURCE:-/gpfs/$USER/lumina/det_convergence/det1234_20260809T060943Z_1591473a3551}"

die() {
  printf 'CMF_EXACT_GPU_FAIR_SUBMIT_FATAL %s\n' "$*" >&2
  exit 70
}

[[ "$sealed_source" = /gpfs/"$USER"/lumina/det_convergence/det1234_* ]] || \
  die "unexpected sealed source: $sealed_source"
[[ -f "$sealed_source/READY" && -d "$sealed_source/input/model" &&
   -d "$sealed_source/input/global_atomic" &&
   -f "$sealed_source/input/deck.sha256" &&
   -f "$sealed_source/input/topion.sha256" &&
   -f "$sealed_source/input/resolved_lumina.exports" ]] || \
  die "sealed source is incomplete"
[[ -x "$binary" && -x "$driver" && -x "$checker" ]] || \
  die "fat binary, driver, or checker missing"

(
  cd "$sealed_source/input/model"
  sha256sum -c "$sealed_source/input/deck.sha256"
) >/dev/null || die "sealed deck hash mismatch"
(
  cd "$sealed_source/input/global_atomic"
  sha256sum -c "$sealed_source/input/topion.sha256"
) >/dev/null || die "sealed atomic hash mismatch"

binary_sha="$(sha256sum "$binary" | cut -d' ' -f1)"
stamp="$(date -u +%Y%m%dT%H%M%S%NZ)_p$$"
pair_root="/gpfs/$USER/lumina/cmf_exact_gpu_fair/pair_${stamp}_${binary_sha:0:12}"
[[ "$pair_root" = /gpfs/"$USER"/lumina/cmf_exact_gpu_fair/pair_* ]] || \
  die "unsafe pair root: $pair_root"
mkdir -p "$pair_root"

stage_lane() {
  local lane="$1"
  local run_root="$pair_root/fair_$lane"
  mkdir -p "$run_root/input"
  install -m 0750 "$binary" "$run_root/input/lumina_cuda"
  install -m 0750 "$driver" "$run_root/input/job.slurm"
  install -m 0750 "$checker" "$run_root/input/check_cmf_exact_cmfgen_ab.py"
  install -m 0640 "$sealed_source/input/deck.sha256" "$run_root/input/deck.sha256"
  install -m 0640 "$sealed_source/input/topion.sha256" "$run_root/input/topion.sha256"
  install -m 0640 "$sealed_source/input/resolved_lumina.exports" \
    "$run_root/input/resolved_lumina.exports"
  install -m 0640 \
    "$repo_root/src/lumina_cmfgen.c" \
    "$repo_root/src/lumina_cmfgen.h" \
    "$repo_root/src/cmf_exact_multigpu.cu" \
    "$repo_root/src/cmf_exact_multigpu.h" \
    "$repo_root/src/cmf_exact_sliding.c" \
    "$repo_root/src/cmf_exact_sliding.h" \
    "$repo_root/src/cmf_error_envelope.c" \
    "$repo_root/src/cmf_error_envelope.h" \
    "$repo_root/src/env_universe.h" \
    "$repo_root/Makefile" "$run_root/input/"
  printf '%s\n' "$sealed_source" > "$run_root/input/sealed_source.txt"
  printf '%s\n' "$binary_sha" > "$run_root/input/binary.sha256"
  (
    cd "$run_root/input"
    find . -maxdepth 1 -type f ! -name manifest.sha256 -printf '%P\0' |
      LC_ALL=C sort -z | xargs -0 sha256sum > manifest.sha256
  )
  printf 'READY\n' > "$run_root/READY"
}

stage_lane h200x1
stage_lane h100x2
stage_lane a100x2
stage_lane a40x4

h200_root="$pair_root/fair_h200x1"
h100_root="$pair_root/fair_h100x2"
a100_root="$pair_root/fair_a100x2"
a40_root="$pair_root/fair_a40x4"
h200_job="$(sbatch --parsable --job-name=cmf_h200x1 \
  --partition=h200 --nodes=1 --ntasks=1 --gres=gpu:H200:1 \
  --cpus-per-task=24 --mem=192G --time=02:30:00 \
  --output="$h200_root/slurm-%j.out" --error="$h200_root/slurm-%j.err" \
  "$h200_root/input/job.slurm" "$h200_root" 1 H200)"
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
a40_job="$(sbatch --parsable --job-name=cmf_a40x4 \
  --partition=a40 --nodelist=syn07 --nodes=1 --ntasks=1 --gres=gpu:A40:4 \
  --cpus-per-task=24 --mem=192G --time=02:30:00 \
  --output="$a40_root/slurm-%j.out" --error="$a40_root/slurm-%j.err" \
  "$a40_root/input/job.slurm" "$a40_root" 4 A40)"
printf '%s\n' "$h200_job" > "$h200_root/job_id.txt"
printf '%s\n' "$h100_job" > "$h100_root/job_id.txt"
printf '%s\n' "$a100_job" > "$a100_root/job_id.txt"
printf '%s\n' "$a40_job" > "$a40_root/job_id.txt"
printf '%s\n' "$pair_root" > "$pair_root/pair_root.txt"
printf 'CMF_EXACT_GPU_FAIR_CANDIDATES_SUBMITTED pair_root=%s binary_sha256=%s h200_job=%s h100_job=%s a100_job=%s a40_job=%s\n' \
  "$pair_root" "$binary_sha" "$h200_job" "$h100_job" "$a100_job" "$a40_job" |
  tee "$pair_root/SUBMITTED.txt"
