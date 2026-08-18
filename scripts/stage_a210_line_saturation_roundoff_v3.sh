#!/usr/bin/env bash
# Seal the roundoff-aware CMFGEN comparator for sequential execution after V2.
# This observer never changes the physical run or its V2 evidence.
set -euo pipefail
umask 027

die() {
  printf 'A210_LINE_SATURATION_ROUNDOFF_V3_STAGE_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 1 ]] || die "usage: $0 RUN_ROOT"
run_root="$1"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs && \
   -d "$run_root" && -f "$run_root/READY" ]] || \
  die "unsafe or unsealed run root: $run_root"

repo="$(cd "$(dirname "$0")/.." && pwd -P)"
v2="$run_root/postprocess_linear_v2"
bundle="$run_root/postprocess_roundoff_v3"
compare_script="$repo/scripts/compare_a210_cmfgen_line_saturation.py"
parser_script="$repo/scripts/extract_cmfgen_line_net_fixture.py"
owner_script="$repo/scripts/summarize_cmfgen_lineheat_ion_owners.py"
depth_script="$repo/scripts/summarize_cmfgen_lineheat_depths.py"
identity_script="$repo/scripts/cmfgen_ion_identity.py"
monitor_script="$repo/scripts/monitor_a210_line_saturation_roundoff_v3.sh"

[[ -d "$v2" && -f "$v2/READY" ]] || die "missing sealed V2 bundle"
[[ ! -e "$bundle" ]] || die "V3 bundle already exists: $bundle"
for source in "$compare_script" "$parser_script" "$owner_script" \
              "$depth_script" "$identity_script" "$monitor_script" \
              "$v2/cmfgen_coordinate_reference.json" \
              "$v2/cmfgen_netrate_path.txt" "$v2/cmfgen_netrate.sha256"; do
  [[ -f "$source" && ! -L "$source" ]] || die "missing or unsafe source: $source"
done

netrate="$(<"$v2/cmfgen_netrate_path.txt")"
expected_netrate_sha="$(<"$v2/cmfgen_netrate.sha256")"
[[ "$netrate" = /* && -f "$netrate" && ! -L "$netrate" ]] || \
  die "unsafe NETRATE source: $netrate"
[[ "$expected_netrate_sha" =~ ^[0-9a-f]{64}$ ]] || \
  die "invalid NETRATE SHA"
[[ "$(sha256sum "$netrate" | cut -d' ' -f1)" == \
   "$expected_netrate_sha" ]] || die "NETRATE SHA drift"

mkdir -p "$bundle"
install -m 0750 "$compare_script" \
  "$bundle/compare_a210_cmfgen_line_saturation.py"
install -m 0640 "$parser_script" \
  "$bundle/extract_cmfgen_line_net_fixture.py"
install -m 0640 "$owner_script" \
  "$bundle/summarize_cmfgen_lineheat_ion_owners.py"
install -m 0640 "$depth_script" \
  "$bundle/summarize_cmfgen_lineheat_depths.py"
install -m 0640 "$identity_script" "$bundle/cmfgen_ion_identity.py"
install -m 0750 "$monitor_script" \
  "$bundle/monitor_a210_line_saturation_roundoff_v3.sh"
install -m 0640 "$v2/cmfgen_coordinate_reference.json" \
  "$bundle/cmfgen_coordinate_reference.json"
install -m 0640 "$v2/cmfgen_netrate_path.txt" \
  "$bundle/cmfgen_netrate_path.txt"
install -m 0640 "$v2/cmfgen_netrate.sha256" \
  "$bundle/cmfgen_netrate.sha256"

(
  cd "$bundle"
  sha256sum \
    compare_a210_cmfgen_line_saturation.py \
    extract_cmfgen_line_net_fixture.py \
    summarize_cmfgen_lineheat_ion_owners.py \
    summarize_cmfgen_lineheat_depths.py \
    cmfgen_ion_identity.py \
    monitor_a210_line_saturation_roundoff_v3.sh \
    cmfgen_coordinate_reference.json \
    cmfgen_netrate_path.txt \
    cmfgen_netrate.sha256 \
    > POSTPROCESS_MANIFEST.sha256
)
python3 -I -c \
  'import sys; sys.path.insert(0, sys.argv[1]); import compare_a210_cmfgen_line_saturation' \
  "$bundle" || die "isolated CMFGEN comparison import preflight failed"
printf '%s\n' \
  "schema=A210_LINE_SATURATION_ROUNDOFF_V3_BUNDLE_V1" \
  "run_root=$run_root" \
  "v2_bundle=$v2" \
  "comparator_sha256=$(sha256sum "$compare_script" | cut -d' ' -f1)" \
  "cmfgen_netrate_sha256=$expected_netrate_sha" \
  "execution_order=AFTER_V2_PASS" \
  "arithmetic_bound_is_physical_tolerance=0" \
  "physical_mutation=0" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0" \
  > "$bundle/ROUND_OFF_PROOF_CONTRACT.txt"
printf 'READY\n' > "$bundle/READY"
printf 'A210_LINE_SATURATION_ROUNDOFF_V3_STAGE_OK bundle=%s comparator_sha256=%s\n' \
  "$bundle" "$(sha256sum "$compare_script" | cut -d' ' -f1)"
