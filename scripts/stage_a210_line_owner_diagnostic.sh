#!/usr/bin/env bash
# Clone a sealed A100x2 targeted input and add only private A2-10 diagnostics.
set -euo pipefail
umask 027

die() {
  printf 'A210_LINE_OWNER_STAGE_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 6 ]] || \
  die "usage: $0 BASE_RUN_ROOT NEW_RUN_ROOT BINARY REQUESTED_TE_K OWNER_SHELLS REFINEMENTS"
base="$1"
dest="$2"
binary="$3"
requested_te="$4"
owner_shells="$5"
refinements="$6"
repo="$(cd "$(dirname "$0")/.." && pwd -P)"
summarizer="$repo/scripts/summarize_a210_line_ion_owners.py"

[[ "$base" = /* && -d "$base/input" && -f "$base/READY" ]] || \
  die "base run is not sealed and READY: $base"
[[ "$dest" = /* && "$dest" != / && "$dest" != /gpfs && ! -e "$dest" ]] || \
  die "destination must be a new safe absolute path: $dest"
[[ "$binary" = /* && -x "$binary" ]] || die "binary is not executable: $binary"
[[ -x "$summarizer" ]] || die "summarizer is not executable: $summarizer"
[[ "$requested_te" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || \
  die "requested temperature is not a strict decimal: $requested_te"
awk -v value="$requested_te" \
  'BEGIN { exit !(value > 3500.0 && value < 140000.0) }' || \
  die "requested temperature lies outside the open production bracket"
[[ "$owner_shells" =~ ^[1-9][0-9]*$ && "$owner_shells" -le 50 ]] || \
  die "owner shell count must be in 1..50"
[[ "$refinements" =~ ^[1-9][0-9]*$ && "$refinements" -le 64 ]] || \
  die "proof refinements must be in 1..64"
[[ "$(<"$base/input/diagnostic_mode.txt")" == A210_TARGETED_GATE ]] || \
  die "base input is not the targeted A2-10 lane"
[[ "$(<"$base/input/precore_tau_refresh.txt")" == 0 ]] || \
  die "rejected pre-core tau refresh is present"
grep -q '^declare -x LUMINA_RADEQ_DIAG="1"$' \
  "$base/input/resolved_lumina.exports" || die "base diagnostic lane is not armed"
if grep -Eq '^declare -x LUMINA_(A210_LINE_ION_OWNER_SHELLS|RADEQ_DIAG_TE_K)=' \
    "$base/input/resolved_lumina.exports"; then
  die "base already contains line-owner diagnostic variables"
fi

mkdir -p "$dest"
cp -al "$base/input" "$dest/input"

# Every replaced destination is under a just-created path.  Remove its hard
# link before writing so the sealed base bytes remain untouched.
rm "$dest/input/lumina_cuda" "$dest/input/binary.sha256" \
   "$dest/input/git_head.txt" "$dest/input/git_status.txt" \
   "$dest/input/git_diff.stat" "$dest/input/flight_scripts.sha256" \
   "$dest/input/resolved_lumina.exports" \
   "$dest/input/resolved_lumina.sha256" \
   "$dest/input/envelope_refinements.txt"
install -m 0750 "$binary" "$dest/input/lumina_cuda"
install -m 0750 "$summarizer" \
  "$dest/input/summarize_a210_line_ion_owners.py"

tmp_env="$dest/input/resolved_lumina.exports.tmp"
sed -e "s|$base|$dest|g" \
    -e "s|^declare -x LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=\"[0-9]*\"$|declare -x LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=\"$refinements\"|" \
    "$base/input/resolved_lumina.exports" > "$tmp_env"
printf '%s\n' \
  "declare -x LUMINA_A210_LINE_ION_OWNER_SHELLS=\"$owner_shells\"" \
  "declare -x LUMINA_RADEQ_DIAG_TE_K=\"$requested_te\"" >> "$tmp_env"
LC_ALL=C sort -u "$tmp_env" > "$dest/input/resolved_lumina.exports"
rm "$tmp_env"

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
  "$dest/input/summarize_a210_line_ion_owners.py" \
  "$dest/input/job.slurm" \
  "$dest/input/run_coevolve_s01.reference.sh" \
  > "$dest/input/flight_scripts.sha256"
sha256sum "$dest/input/resolved_lumina.exports" \
  > "$dest/input/resolved_lumina.sha256"
printf '%s\n' "$refinements" > "$dest/input/envelope_refinements.txt"
printf '%s\n' "$requested_te" > "$dest/input/requested_diag_te_K.txt"
printf '%s\n' "$owner_shells" > "$dest/input/line_ion_owner_shells.txt"

grep -q "^declare -x LUMINA_A210_LINE_ION_OWNER_SHELLS=\"$owner_shells\"$" \
  "$dest/input/resolved_lumina.exports" || die "owner shell seal missing"
grep -q "^declare -x LUMINA_RADEQ_DIAG_TE_K=\"$requested_te\"$" \
  "$dest/input/resolved_lumina.exports" || die "requested Te seal missing"
grep -q "^declare -x LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=\"$refinements\"$" \
  "$dest/input/resolved_lumina.exports" || die "proof refinement seal missing"

binary_sha="$(<"$dest/input/binary.sha256")"
sigma_sha="$(<"$dest/input/sigma.sha256")"
printf '%s\n' \
  "schema=A210_LINE_ION_OWNER_DIAGNOSTIC_V1" \
  "base_run_root=$base" \
  "binary_sha256=$binary_sha" \
  "sigma_sha256=$sigma_sha" \
  "envelope_refinements=$refinements" \
  "requested_temperature_K=$requested_te" \
  "owner_shells=$owner_shells" \
  "diagnostic_only=1" \
  "physical_mutation=0" \
  "publication_authority=NONE" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0" \
  > "$dest/LINE_ION_OWNER_DIAGNOSTIC.txt"
printf 'STAGED_ONLY run_root=%s binary_sha256=%s sigma_sha256=%s iterations=1 single_total=1 stage4=0 mgpu_devices=2 refinements=%s cpus=24 diagnostic=A210_LINE_ION_OWNER requested_T_K=%s owner_shells=%s\n' \
  "$dest" "$binary_sha" "$sigma_sha" "$refinements" \
  "$requested_te" "$owner_shells" > "$dest/STAGED_ONLY.txt"
printf 'READY\n' > "$dest/READY"
printf 'A210_LINE_OWNER_STAGE_OK run_root=%s binary_sha256=%s refinements=%s requested_T_K=%s owner_shells=%s\n' \
  "$dest" "$binary_sha" "$refinements" "$requested_te" "$owner_shells"
