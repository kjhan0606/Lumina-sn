#!/usr/bin/env bash
# Clone the sealed K36 owner diagnostic and add only the read-only saturation
# census plus its offline CMFGEN comparison tools.
set -euo pipefail
umask 027

die() {
  printf 'A210_LINE_SATURATION_STAGE_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 4 || $# -eq 5 ]] || \
  die "usage: $0 BASE_RUN_ROOT NEW_RUN_ROOT BINARY COORDINATE_REFERENCE [MODE]"
base="$1"
dest="$2"
binary="$3"
coordinate_reference="$4"
mode="${5:-1}"
[[ "$mode" == 1 || "$mode" == 2 ]] || die "MODE must be 1 or 2"
repo="$(cd "$(dirname "$0")/.." && pwd -P)"

summary_script="$repo/scripts/summarize_a210_line_saturation.py"
compare_script="$repo/scripts/compare_a210_cmfgen_line_saturation.py"
parser_script="$repo/scripts/extract_cmfgen_line_net_fixture.py"
owner_script="$repo/scripts/summarize_cmfgen_lineheat_ion_owners.py"
depth_script="$repo/scripts/summarize_cmfgen_lineheat_depths.py"
identity_script="$repo/scripts/cmfgen_ion_identity.py"
monitor_script="$repo/scripts/monitor_a210_line_saturation_diagnostic.sh"
stage4_script="$repo/scripts/a210_stage4_jbar_offline.py"

[[ "$base" = /* && -d "$base/input" && -f "$base/READY" ]] || \
  die "base run is not sealed and READY: $base"
[[ "$dest" = /* && "$dest" != / && "$dest" != /gpfs && ! -e "$dest" ]] || \
  die "destination must be a new safe absolute path: $dest"
[[ "$binary" = /* && -x "$binary" ]] || die "binary is not executable: $binary"
[[ "$coordinate_reference" = /* && -f "$coordinate_reference" && \
   ! -L "$coordinate_reference" ]] || die "unsafe coordinate reference"
for script in "$summary_script" "$compare_script" "$parser_script" \
              "$owner_script" "$depth_script" "$identity_script" \
              "$monitor_script" "$stage4_script"; do
  [[ -f "$script" && ! -L "$script" ]] || die "missing script: $script"
done
[[ "$(<"$base/input/diagnostic_mode.txt")" == A210_TARGETED_GATE ]] || \
  die "base input is not the targeted A2-10 lane"
[[ "$(<"$base/input/precore_tau_refresh.txt")" == 0 ]] || \
  die "rejected pre-core tau refresh is present"
grep -q '^declare -x LUMINA_RADEQ_DIAG="1"$' \
  "$base/input/resolved_lumina.exports" || die "base diagnostic lane is not armed"
grep -Eq '^declare -x LUMINA_A210_LINE_ION_OWNER_SHELLS="[1-9][0-9]*"$' \
  "$base/input/resolved_lumina.exports" || die "base lacks owner closure scope"
grep -Eq '^declare -x LUMINA_RADEQ_DIAG_TE_K="[^"]+"$' \
  "$base/input/resolved_lumina.exports" || die "base lacks requested Te"
if grep -q '^declare -x LUMINA_A210_LINE_SATURATION_DIAG=' \
    "$base/input/resolved_lumina.exports"; then
  die "base already contains saturation diagnostic variable"
fi

mapfile -t cmf_source < <(python3 - "$coordinate_reference" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
s = d.get("source", {})
print(s.get("netrate", ""))
print(s.get("netrate_sha256", ""))
PY
)
[[ "${#cmf_source[@]}" -eq 2 ]] || die "coordinate reference parse failed"
netrate="${cmf_source[0]}"
expected_netrate_sha="${cmf_source[1]}"
[[ "$netrate" = /* && -f "$netrate" && ! -L "$netrate" ]] || \
  die "coordinate reference NETRATE is unsafe: $netrate"
[[ "$expected_netrate_sha" =~ ^[0-9a-f]{64}$ ]] || \
  die "coordinate reference NETRATE SHA is invalid"
actual_netrate_sha="$(sha256sum "$netrate" | cut -d' ' -f1)"
[[ "$actual_netrate_sha" == "$expected_netrate_sha" ]] || \
  die "coordinate reference NETRATE SHA mismatch"

mkdir -p "$dest"
cp -al "$base/input" "$dest/input"
rm "$dest/input/lumina_cuda" "$dest/input/binary.sha256" \
   "$dest/input/git_head.txt" "$dest/input/git_status.txt" \
   "$dest/input/git_diff.stat" "$dest/input/flight_scripts.sha256" \
   "$dest/input/resolved_lumina.exports" \
   "$dest/input/resolved_lumina.sha256"
install -m 0750 "$binary" "$dest/input/lumina_cuda"
install -m 0750 "$summary_script" \
  "$dest/input/summarize_a210_line_saturation.py"
install -m 0750 "$compare_script" \
  "$dest/input/compare_a210_cmfgen_line_saturation.py"
install -m 0640 "$parser_script" \
  "$dest/input/extract_cmfgen_line_net_fixture.py"
install -m 0640 "$owner_script" \
  "$dest/input/summarize_cmfgen_lineheat_ion_owners.py"
install -m 0640 "$depth_script" \
  "$dest/input/summarize_cmfgen_lineheat_depths.py"
install -m 0640 "$identity_script" \
  "$dest/input/cmfgen_ion_identity.py"
install -m 0750 "$monitor_script" \
  "$dest/input/monitor_a210_line_saturation_diagnostic.sh"
install -m 0750 "$stage4_script" "$dest/input/a210_stage4_jbar_offline.py"
install -m 0640 "$coordinate_reference" \
  "$dest/input/cmfgen_coordinate_reference.json"

target_ion="${LUMINA_A210_LINE_SATURATION_TARGET_ION:-3}"
[[ "$target_ion" =~ ^([0-9]|10)$ ]] || die "invalid target ion: $target_ion"
tmp_env="$dest/input/resolved_lumina.exports.tmp"
sed "s|$base|$dest|g" "$base/input/resolved_lumina.exports" > "$tmp_env"
printf '%s\n' \
  "declare -x LUMINA_A210_LINE_SATURATION_DIAG=\"$mode\"" \
  "declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION=\"$target_ion\"" \
  "declare -x LUMINA_A210_INDEPENDENT_CAPTURE=\"1\"" \
  "declare -x LUMINA_A210_SPRODUCER_CAPTURE=\"${LUMINA_A210_SPRODUCER_CAPTURE:-0}\"" \
  >> "$tmp_env"
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
  "$dest/input/summarize_a210_line_saturation.py" \
  "$dest/input/compare_a210_cmfgen_line_saturation.py" \
  "$dest/input/extract_cmfgen_line_net_fixture.py" \
  "$dest/input/summarize_cmfgen_lineheat_ion_owners.py" \
  "$dest/input/summarize_cmfgen_lineheat_depths.py" \
  "$dest/input/cmfgen_ion_identity.py" \
  "$dest/input/monitor_a210_line_saturation_diagnostic.sh" \
  "$dest/input/a210_stage4_jbar_offline.py" \
  "$dest/input/cmfgen_coordinate_reference.json" \
  "$dest/input/job.slurm" \
  "$dest/input/run_coevolve_s01.reference.sh" \
  > "$dest/input/flight_scripts.sha256"
python3 -I -c \
  'import sys; sys.path.insert(0, sys.argv[1]); import compare_a210_cmfgen_line_saturation' \
  "$dest/input" || die "isolated CMFGEN comparison import preflight failed"
sha256sum "$dest/input/resolved_lumina.exports" \
  > "$dest/input/resolved_lumina.sha256"
printf '%s\n' "$netrate" > "$dest/input/cmfgen_netrate_path.txt"
printf '%s\n' "$expected_netrate_sha" > "$dest/input/cmfgen_netrate.sha256"

requested_te="$(<"$dest/input/requested_diag_te_K.txt")"
owner_shells="$(<"$dest/input/line_ion_owner_shells.txt")"
refinements="$(<"$dest/input/envelope_refinements.txt")"
grep -q "^declare -x LUMINA_A210_LINE_SATURATION_DIAG=\"$mode\"$" \
  "$dest/input/resolved_lumina.exports" || die "saturation seal missing"
grep -q "^declare -x LUMINA_RADEQ_DIAG_TE_K=\"$requested_te\"$" \
  "$dest/input/resolved_lumina.exports" || die "requested Te seal drift"

binary_sha="$(<"$dest/input/binary.sha256")"
sigma_sha="$(<"$dest/input/sigma.sha256")"
printf '%s\n' \
  "schema=A210_LINE_SATURATION_DIAGNOSTIC_V1" \
  "base_run_root=$base" \
  "binary_sha256=$binary_sha" \
  "sigma_sha256=$sigma_sha" \
  "cmfgen_netrate_sha256=$expected_netrate_sha" \
  "envelope_refinements=$refinements" \
  "requested_temperature_K=$requested_te" \
  "owner_shells=$owner_shells" \
  "target_atomic_numbers=26,27,28" \
  "target_ion_zero_based=$target_ion" \
  "independent_capture=1" \
  "independent_J_cont=LINE_FREE_EXACT_CONTINUUM" \
  "independent_S_probe=LINE_MATERIAL_SOURCE_FUNCTION" \
  "stage4_formula=Jbar=beta*J_cont+(1-beta)*S_probe" \
  "selection_target_fraction=0.9" \
  "diagnostic_only=1" \
  "physical_mutation=0" \
  "publication_authority=NONE" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0" \
  > "$dest/LINE_SATURATION_DIAGNOSTIC.txt"
if [[ "$mode" == 2 ]]; then
  printf '%s\n' \
    "selection_mode=PER_ION_UNION" \
    "per_ion_order=DESCENDING_SCALED_EMISSION_THEN_ASCENDING_LINE_ID" \
    "per_ion_prefix=MINIMAL_FIRST_REACH_0.9" \
    "physical_cause_claim=0" \
    >> "$dest/LINE_SATURATION_DIAGNOSTIC.txt"
fi
printf 'STAGED_ONLY run_root=%s binary_sha256=%s sigma_sha256=%s iterations=1 single_total=1 stage4=0 mgpu_devices=2 refinements=%s cpus=24 diagnostic=A210_LINE_SATURATION requested_T_K=%s owner_shells=%s selection=0.9\n' \
  "$dest" "$binary_sha" "$sigma_sha" "$refinements" \
  "$requested_te" "$owner_shells" > "$dest/STAGED_ONLY.txt"
printf 'READY\n' > "$dest/READY"
printf 'A210_LINE_SATURATION_STAGE_OK run_root=%s binary_sha256=%s refinements=%s requested_T_K=%s mode=%s\n' \
  "$dest" "$binary_sha" "$refinements" "$requested_te" "$mode"
