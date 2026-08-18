#!/usr/bin/env bash
# Clone the sealed k=24 A100x2 targeted input by hard link, then replace only
# the executable/provenance files for the line-coefficient identity diagnostic.
set -euo pipefail
umask 027

die() {
  printf 'A210_LINE_IDENTITY_STAGE_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 3 ]] || die "usage: $0 BASE_RUN_ROOT NEW_RUN_ROOT BINARY"
base="$1"
dest="$2"
binary="$3"
repo="$(cd "$(dirname "$0")/.." && pwd -P)"

[[ "$base" = /* && -d "$base/input" && -f "$base/READY" ]] ||
  die "base run is not sealed and READY: $base"
[[ "$dest" = /* && "$dest" != / && "$dest" != /gpfs && ! -e "$dest" ]] ||
  die "destination must be a new safe absolute path: $dest"
[[ "$binary" = /* && -x "$binary" ]] || die "binary is not executable: $binary"

mkdir -p "$dest"
cp -al "$base/input" "$dest/input"

# Never overwrite a hard-linked base file.  Remove only the validated new
# destination link before installing each replacement.
rm "$dest/input/lumina_cuda" "$dest/input/binary.sha256" \
   "$dest/input/git_head.txt" "$dest/input/git_status.txt" \
   "$dest/input/git_diff.stat" "$dest/input/flight_scripts.sha256" \
   "$dest/input/resolved_lumina.exports" \
   "$dest/input/resolved_lumina.sha256"
install -m 0750 "$binary" "$dest/input/lumina_cuda"
sed "s|$base|$dest|g" "$base/input/resolved_lumina.exports" \
  > "$dest/input/resolved_lumina.exports"
sha256sum "$dest/input/lumina_cuda" | awk '{print $1}' \
  > "$dest/input/binary.sha256"
git -C "$repo" rev-parse HEAD > "$dest/input/git_head.txt"
git -C "$repo" status --short > "$dest/input/git_status.txt"
git -C "$repo" diff --stat > "$dest/input/git_diff.stat"
sha256sum \
  "$dest/input/check_det_convergence.py" \
  "$dest/input/summarize_a210_cancellation_census.py" \
  "$dest/input/check_a210_cancellation_witnesses.py" \
  "$dest/input/check_a210_targeted_gate.py" \
  "$dest/input/job.slurm" \
  "$dest/input/run_coevolve_s01.reference.sh" \
  > "$dest/input/flight_scripts.sha256"
sha256sum "$dest/input/resolved_lumina.exports" \
  > "$dest/input/resolved_lumina.sha256"

binary_sha="$(<"$dest/input/binary.sha256")"
sigma_sha="$(<"$dest/input/sigma.sha256")"
printf '%s\n' \
  "schema=A210_LINE_COEFFICIENT_IDENTITY_DIAGNOSTIC_V1" \
  "base_run_root=$base" \
  "binary_sha256=$binary_sha" \
  "sigma_sha256=$sigma_sha" \
  "diagnostic_only=1" \
  "physical_mutation=0" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0" \
  > "$dest/IDENTITY_DIAGNOSTIC.txt"
printf 'STAGED_ONLY run_root=%s binary_sha256=%s sigma_sha256=%s iterations=1 single_total=1 stage4=0 mgpu_devices=2 refinements=24 cpus=24 diagnostic=A210_LINE_COEFFICIENT_IDENTITY\n' \
  "$dest" "$binary_sha" "$sigma_sha" > "$dest/STAGED_ONLY.txt"
printf 'READY\n' > "$dest/READY"
printf 'A210_LINE_IDENTITY_STAGE_OK run_root=%s binary_sha256=%s\n' \
  "$dest" "$binary_sha"
