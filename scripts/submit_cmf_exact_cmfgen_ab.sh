#!/usr/bin/env bash
# Submit one same-assembled-state CPU versus A40x4 production CMF exact flight.
set -euo pipefail
umask 027

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$repo_root/lumina_cuda"
driver="$repo_root/scripts/run_cmf_exact_cmfgen_ab.slurm"
checker="$repo_root/scripts/check_cmf_exact_cmfgen_ab.py"
sealed_source="${CMF_EXACT_CMFGEN_SEALED_SOURCE:-/gpfs/$USER/lumina/det_convergence/det1234_20260809T060943Z_1591473a3551}"

die() {
  printf 'CMF_EXACT_CMFGEN_AB_SUBMIT_FATAL %s\n' "$*" >&2
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
[[ -x "$binary" && -f "$driver" && -x "$checker" ]] || \
  die "binary, driver, or checker missing"
make -q -C "$repo_root" GPU_ARCH=sm_86 lumina_cuda || \
  die "lumina_cuda is stale for GPU_ARCH=sm_86"

# Verify the old sealed deck before its identity is referenced by a new job.
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
run_root="/gpfs/$USER/lumina/cmf_exact_cmfgen_ab/ab_${stamp}_${binary_sha:0:12}"
[[ "$run_root" = /gpfs/"$USER"/lumina/cmf_exact_cmfgen_ab/ab_* ]] || \
  die "unsafe run root: $run_root"
mkdir -p "$run_root/input"
printf 'STAGING\n' > "$run_root/STAGING"

install -m 0750 "$binary" "$run_root/input/lumina_cuda"
install -m 0750 "$driver" "$run_root/input/job.slurm"
install -m 0750 "$checker" "$run_root/input/check_cmf_exact_cmfgen_ab.py"
install -m 0640 "$sealed_source/input/deck.sha256" \
  "$run_root/input/deck.sha256"
install -m 0640 "$sealed_source/input/topion.sha256" \
  "$run_root/input/topion.sha256"
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
  "$repo_root/Makefile" \
  "$run_root/input/"
printf '%s\n' "$sealed_source" > "$run_root/input/sealed_source.txt"
printf '%s\n' "$binary_sha" > "$run_root/input/binary.sha256"
(
  cd "$run_root/input"
  find . -maxdepth 1 -type f ! -name manifest.sha256 -printf '%P\0' |
    LC_ALL=C sort -z | xargs -0 sha256sum > manifest.sha256
)
rm "$run_root/STAGING"
printf 'READY\n' > "$run_root/READY"

job_id="$(sbatch --parsable \
  --partition=a40 --nodelist=syn07 --nodes=1 --ntasks=1 \
  --gres=gpu:A40:4 --cpus-per-task=32 --mem=192G --time=02:00:00 \
  --output="$run_root/slurm-%j.out" \
  --error="$run_root/slurm-%j.err" \
  "$run_root/input/job.slurm" "$run_root")"
printf '%s\n' "$job_id" > "$run_root/job_id.txt"
printf 'CMF_EXACT_CMFGEN_AB_SUBMITTED job_id=%s run_root=%s binary_sha256=%s sealed_source=%s\n' \
  "$job_id" "$run_root" "$binary_sha" "$sealed_source" | \
  tee "$run_root/SUBMITTED.txt"
