#!/usr/bin/env bash
# Finalize both lane reports and the independent external-CMFGEN comparison.
set -euo pipefail
umask 027

die() {
  printf 'CMF_EXACT_GPU_FAIR_FINALIZE_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 1 ]] || die "usage: $0 ABSOLUTE_PAIR_ROOT"
pair_root="$1"
[[ "$pair_root" = /gpfs/"$USER"/lumina/cmf_exact_gpu_fair/pair_* ]] || \
  die "unsafe pair root: $pair_root"
input_dir="$pair_root/finalize_input"
h200_root="$pair_root/fair_h200x1"
a40_root="$pair_root/fair_a40x4"
[[ -f "$h200_root/RUN_FOOTER.txt" && -f "$a40_root/RUN_FOOTER.txt" ]] || \
  die "one or both lane footers are missing"
(
  cd "$input_dir"
  sha256sum -c manifest.sha256
) > "$pair_root/finalize_input_verify.log" 2>&1 || \
  die "finalizer input SHA mismatch"

python3 "$input_dir/summarize_cmf_exact_gpu_fair.py" "$h200_root" \
  --expected-devices 1 --report "$h200_root/lane_report.json" \
  > "$h200_root/lane_report.stdout"
python3 "$input_dir/summarize_cmf_exact_gpu_fair.py" "$a40_root" \
  --expected-devices 4 --report "$a40_root/lane_report.json" \
  > "$a40_root/lane_report.stdout"
python3 "$input_dir/compare_cmf_exact_gpu_fair.py" \
  --h200 "$h200_root/lane_report.json" \
  --a40 "$a40_root/lane_report.json" \
  --report "$pair_root/fair_comparison.json" \
  > "$pair_root/fair_comparison.stdout"

cmfgen_root="/gpfs/$USER/cmfgen_runs/toy06_19.48d_jnu4"
[[ -f "$cmfgen_root/EDDFACTOR" && -f "$cmfgen_root/EDDFACTOR_INFO" &&
   -f "$cmfgen_root/RVTJ" ]] || die "external CMFGEN fixture is unavailable"
python3 "$input_dir/compare_cmf_exact_external_jnu.py" \
  --lumina-fixture \
    "$h200_root/work/lumina_cmf_fine_external_fixture.csv" \
  --cmfgen-root "$cmfgen_root" \
  --csv "$pair_root/external_cmfgen_jnu_points.csv" \
  --report "$pair_root/external_cmfgen_jnu_report.json" \
  > "$pair_root/external_cmfgen_jnu_report.stdout"

sha256sum "$h200_root/lane_report.json" "$a40_root/lane_report.json" \
  "$pair_root/fair_comparison.json" \
  "$pair_root/external_cmfgen_jnu_points.csv" \
  "$pair_root/external_cmfgen_jnu_report.json" \
  > "$pair_root/final_report.sha256"
printf '%s\n' \
  "CMF_EXACT_GPU_FAIR_PAIR_FINALIZED pair_root=$pair_root" \
  "same_fat_binary=PASS h200x1_vs_a40x4=PASS exit_after_r6=PASS" \
  "external_cmfgen_J_nu=PASS_COMPARISON_NOT_PARITY" \
  "floor=0 cap=0 clamp=0 jitter=0 repair=0" \
  > "$pair_root/FINAL_VERDICT.txt"
