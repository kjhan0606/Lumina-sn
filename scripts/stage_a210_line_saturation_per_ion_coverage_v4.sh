#!/usr/bin/env bash
# Seal the per-ion coverage audit that runs after the V3 comparison.
set -euo pipefail
umask 027

die() {
  printf 'A210_LINE_SATURATION_PER_ION_V4_STAGE_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 1 || $# -eq 2 ]] || die "usage: $0 RUN_ROOT [REFERENCE_STDERR]"
run_root="$1"
reference_stderr="${2:-}"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs && \
   -d "$run_root" && -f "$run_root/READY" ]] || \
  die "unsafe or unsealed run root: $run_root"

repo="$(cd "$(dirname "$0")/.." && pwd -P)"
bundle="$run_root/postprocess_per_ion_coverage_v4"
owner_script="$repo/scripts/summarize_a210_line_ion_owners.py"
coverage_script="$repo/scripts/check_a210_line_saturation_per_ion_coverage.py"
intersection_script="$repo/scripts/compare_a210_line_saturation_intersection.py"
monitor_script="$repo/scripts/monitor_a210_line_saturation_per_ion_coverage_v4.sh"
[[ ! -e "$bundle" ]] || die "V4 bundle already exists: $bundle"
for source in "$owner_script" "$coverage_script" "$monitor_script" \
              "$run_root/input/requested_diag_te_K.txt" \
              "$run_root/input/line_ion_owner_shells.txt"; do
  [[ -f "$source" && ! -L "$source" ]] || die "missing or unsafe source: $source"
done
if [[ -n "$reference_stderr" ]]; then
  [[ "$reference_stderr" = /* && -f "$reference_stderr" && \
     ! -L "$reference_stderr" ]] || die "unsafe reference stderr"
  [[ -f "$intersection_script" && ! -L "$intersection_script" ]] || \
    die "missing intersection comparator"
fi

requested_te="$(<"$run_root/input/requested_diag_te_K.txt")"
owner_shells="$(<"$run_root/input/line_ion_owner_shells.txt")"
[[ "$requested_te" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid requested Te"
[[ "$owner_shells" =~ ^[1-9][0-9]*$ ]] || die "invalid owner shell count"
target_ion="${LUMINA_A210_LINE_SATURATION_TARGET_ION:-3}"
if [[ -f "$run_root/input/resolved_lumina.exports" && \
      ! -L "$run_root/input/resolved_lumina.exports" ]]; then
  target_line="$(grep '^declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION=' \
    "$run_root/input/resolved_lumina.exports" || true)"
  if [[ -n "$target_line" ]]; then
    target_ion="${target_line#*=}"
    target_ion="${target_ion#\"}"
    target_ion="${target_ion%\"}"
  fi
fi
[[ "$target_ion" =~ ^([0-9]|10)$ ]] || die "invalid target ion: $target_ion"

mkdir -p "$bundle"
install -m 0750 "$owner_script" "$bundle/summarize_a210_line_ion_owners.py"
install -m 0750 "$coverage_script" \
  "$bundle/check_a210_line_saturation_per_ion_coverage.py"
install -m 0750 "$monitor_script" \
  "$bundle/monitor_a210_line_saturation_per_ion_coverage_v4.sh"
if [[ -n "$reference_stderr" ]]; then
  install -m 0750 "$intersection_script" \
    "$bundle/compare_a210_line_saturation_intersection.py"
  printf '%s\n' "$reference_stderr" > "$bundle/reference_stderr_path.txt"
  sha256sum "$reference_stderr" | cut -d' ' -f1 \
    > "$bundle/reference_stderr.sha256"
fi
printf '%s\n' "$requested_te" > "$bundle/requested_diag_te_K.txt"
printf '%s\n' "$owner_shells" > "$bundle/line_ion_owner_shells.txt"
printf '%s\n' \
  "schema=A210_LINE_SATURATION_PER_ION_COVERAGE_V4_BUNDLE_V1" \
  "run_root=$run_root" \
  "execution_order=AFTER_ROUNDOFF_V3_PASS" \
  "target_atomic_numbers=26,27,28" \
  "target_ion_zero_based=$target_ion" \
  "required_fraction_each_ion=0.9" \
  "intersection_required=$([[ -n "$reference_stderr" ]] && printf 1 || printf 0)" \
  "arithmetic_bound_is_physical_tolerance=0" \
  "physical_mutation=0" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0" \
  > "$bundle/PER_ION_COVERAGE_CONTRACT.txt"
printf 'READY\n' > "$bundle/READY"
manifest_files=(
  summarize_a210_line_ion_owners.py
  check_a210_line_saturation_per_ion_coverage.py
  monitor_a210_line_saturation_per_ion_coverage_v4.sh
  requested_diag_te_K.txt
  line_ion_owner_shells.txt
  PER_ION_COVERAGE_CONTRACT.txt
  READY
)
if [[ -n "$reference_stderr" ]]; then
  manifest_files+=(
    compare_a210_line_saturation_intersection.py
    reference_stderr_path.txt
    reference_stderr.sha256
  )
fi
(
  cd "$bundle"
  sha256sum "${manifest_files[@]}" > POSTPROCESS_MANIFEST.sha256
)
printf 'A210_LINE_SATURATION_PER_ION_V4_STAGE_OK bundle=%s coverage_sha256=%s\n' \
  "$bundle" "$(sha256sum "$coverage_script" | cut -d' ' -f1)"
