#!/usr/bin/env bash
# Seal the linear-time saturation summary and CMFGEN comparison observer.
set -euo pipefail
umask 027

die() {
  printf 'A210_LINE_SATURATION_V2_STAGE_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 1 ]] || die "usage: $0 RUN_ROOT"
run_root="$1"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs && \
   -d "$run_root" && -f "$run_root/READY" ]] || \
  die "unsafe or unsealed run root: $run_root"

repo="$(cd "$(dirname "$0")/.." && pwd -P)"
bundle="$run_root/postprocess_linear_v2"
summary_script="$repo/scripts/summarize_a210_line_saturation.py"
compare_script="$repo/scripts/compare_a210_cmfgen_line_saturation.py"
parser_script="$repo/scripts/extract_cmfgen_line_net_fixture.py"
owner_script="$repo/scripts/summarize_cmfgen_lineheat_ion_owners.py"
depth_script="$repo/scripts/summarize_cmfgen_lineheat_depths.py"
identity_script="$repo/scripts/cmfgen_ion_identity.py"
monitor_script="$repo/scripts/monitor_a210_line_saturation_postprocess_v2.sh"
coordinate="$run_root/input/cmfgen_coordinate_reference.json"
netrate_path_file="$run_root/input/cmfgen_netrate_path.txt"
netrate_sha_file="$run_root/input/cmfgen_netrate.sha256"

[[ ! -e "$bundle" ]] || die "V2 bundle already exists: $bundle"
for source in "$summary_script" "$compare_script" "$parser_script" \
              "$owner_script" "$depth_script" "$identity_script" \
              "$monitor_script" "$coordinate" "$netrate_path_file" \
              "$netrate_sha_file"; do
  [[ -f "$source" && ! -L "$source" ]] || die "missing or unsafe source: $source"
done
netrate="$(<"$netrate_path_file")"
expected_netrate_sha="$(<"$netrate_sha_file")"
[[ "$netrate" = /* && -f "$netrate" && ! -L "$netrate" ]] || \
  die "unsafe NETRATE source"
[[ "$expected_netrate_sha" =~ ^[0-9a-f]{64}$ && \
   "$(sha256sum "$netrate" | cut -d' ' -f1)" == "$expected_netrate_sha" ]] || \
  die "NETRATE SHA mismatch"

mkdir -p "$bundle"
install -m 0750 "$summary_script" "$bundle/summarize_a210_line_saturation.py"
install -m 0750 "$compare_script" "$bundle/compare_a210_cmfgen_line_saturation.py"
install -m 0640 "$parser_script" "$bundle/extract_cmfgen_line_net_fixture.py"
install -m 0640 "$owner_script" "$bundle/summarize_cmfgen_lineheat_ion_owners.py"
install -m 0640 "$depth_script" "$bundle/summarize_cmfgen_lineheat_depths.py"
install -m 0640 "$identity_script" "$bundle/cmfgen_ion_identity.py"
install -m 0750 "$monitor_script" \
  "$bundle/monitor_a210_line_saturation_postprocess_v2.sh"
install -m 0640 "$coordinate" "$bundle/cmfgen_coordinate_reference.json"
install -m 0640 "$netrate_path_file" "$bundle/cmfgen_netrate_path.txt"
install -m 0640 "$netrate_sha_file" "$bundle/cmfgen_netrate.sha256"

printf '%s\n' \
  "schema=A210_LINE_SATURATION_POSTPROCESS_V2_BUNDLE_V1" \
  "run_root=$run_root" \
  "selection_contract=COMBINED_PREFIX_OR_PER_ION_UNION" \
  "natural_model_rc=1" \
  "physical_mutation=0" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0" \
  > "$bundle/POSTPROCESS_CONTRACT.txt"
printf 'READY\n' > "$bundle/READY"
(
  cd "$bundle"
  sha256sum \
    summarize_a210_line_saturation.py \
    compare_a210_cmfgen_line_saturation.py \
    extract_cmfgen_line_net_fixture.py \
    summarize_cmfgen_lineheat_ion_owners.py \
    summarize_cmfgen_lineheat_depths.py \
    cmfgen_ion_identity.py \
    monitor_a210_line_saturation_postprocess_v2.sh \
    cmfgen_coordinate_reference.json \
    cmfgen_netrate_path.txt \
    cmfgen_netrate.sha256 \
    POSTPROCESS_CONTRACT.txt \
    READY \
    > POSTPROCESS_MANIFEST.sha256
)
python3 -I -c \
  'import sys; sys.path.insert(0, sys.argv[1]); import compare_a210_cmfgen_line_saturation' \
  "$bundle" || die "isolated CMFGEN comparison import preflight failed"
printf 'A210_LINE_SATURATION_V2_STAGE_OK bundle=%s summary_sha256=%s comparator_sha256=%s\n' \
  "$bundle" \
  "$(sha256sum "$summary_script" | cut -d' ' -f1)" \
  "$(sha256sum "$compare_script" | cut -d' ' -f1)"
