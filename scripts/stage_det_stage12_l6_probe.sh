#!/usr/bin/env bash
# Clone the sealed one-iteration IDSEAL lane and apply only the five DET-SPRIM
# execution deltas registered in RUNG_DET_SPRIM_L6 section 7.
set -euo pipefail
umask 027

die() {
  printf 'DET_STAGE12_L6_STAGE_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 3 ]] || \
  die "usage: $0 IDSEAL_RUN_ROOT NEW_RUN_ROOT ABSOLUTE_BINARY"
base="$1"
dest="$2"
binary="$3"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

[[ "$base" = /* && -d "$base/input" && -f "$base/READY" ]] || \
  die "IDSEAL base is not sealed and READY: $base"
[[ "$dest" = /* && "$dest" != / && "$dest" != /gpfs && ! -e "$dest" ]] || \
  die "destination must be a new safe absolute path: $dest"
[[ "$binary" = /* && -x "$binary" && ! -L "$binary" ]] || \
  die "binary is not a safe executable: $binary"
command -v git >/dev/null || die "git unavailable"
command -v sha256sum >/dev/null || die "sha256sum unavailable"
command -v python3 >/dev/null || die "python3 unavailable"

stage_shell_pid="$BASHPID"
temporary_paths=()
cleanup_temporaries() {
  [[ "$BASHPID" == "$stage_shell_pid" ]] || return
  if ((${#temporary_paths[@]} > 0)); then
    rm -f -- "${temporary_paths[@]}"
  fi
}
trap cleanup_temporaries EXIT

new_temporary() {
  local target="$1"
  new_temporary_path="${target}.tmp.$$"
  [[ ! -d "$target" ]] || die "refusing to replace directory: $target"
  rm -f -- "$new_temporary_path"
  temporary_paths+=("$new_temporary_path")
}

atomic_text_file() {
  local target="$1"
  local mode="$2"
  shift 2
  local temporary
  new_temporary "$target"
  temporary="$new_temporary_path"
  printf '%s\n' "$@" > "$temporary"
  chmod "$mode" "$temporary"
  mv -f -- "$temporary" "$target"
}

atomic_capture_file() {
  local target="$1"
  local mode="$2"
  shift 2
  local temporary
  new_temporary "$target"
  temporary="$new_temporary_path"
  "$@" > "$temporary"
  chmod "$mode" "$temporary"
  mv -f -- "$temporary" "$target"
}

atomic_install_file() {
  local source="$1"
  local target="$2"
  local mode="$3"
  local temporary
  new_temporary "$target"
  temporary="$new_temporary_path"
  install -m "$mode" "$source" "$temporary"
  mv -f -- "$temporary" "$target"
}

write_base_input_manifest() {
  local output="$1"
  (
    cd "$base/input"
    LC_ALL=C find . -type f -exec sha256sum -- {} + | LC_ALL=C sort
  ) > "$output"
}

for required in outer_iterations.txt diagnostic_mode.txt \
  resolved_lumina.exports job.slurm check_a210_targeted_gate.py \
  check_det_convergence.py binary.sha256 git_head.txt; do
  [[ -f "$base/input/$required" && ! -L "$base/input/$required" ]] || \
    die "unsafe or missing IDSEAL input: $required"
done
[[ "$(<"$base/input/outer_iterations.txt")" == 1 ]] || \
  die "IDSEAL base is not the sealed one-iteration lane"
[[ "$(<"$base/input/diagnostic_mode.txt")" == A210_TARGETED_GATE ]] || \
  die "IDSEAL base is not the targeted diagnostic lane"

base_env="$base/input/resolved_lumina.exports"
declare -A expected_base=(
  [LUMINA_PURE_CMFGEN_ITER]=1
  [LUMINA_RADEQ_DIAG_TE_K]=19059.411196903675
  [LUMINA_A210_SPRODUCER_CAPTURE]=0
  [LUMINA_A210_INDEPENDENT_CAPTURE]=0
)
for name in "${!expected_base[@]}"; do
  value="${expected_base[$name]}"
  [[ "$(grep -cFx "declare -x $name=\"$value\"" "$base_env" || true)" -eq 1 ]] || \
    die "IDSEAL base value mismatch: $name"
done
grep -qFx 'declare -x LUMINA_FIXED_TE_PROFILE="/gpfs/kjhan/lumina/te_profiles/seed_uniform_10020.txt"' \
  "$base_env" || die "IDSEAL fixed-Te profile seal mismatch"
grep -qFx 'declare -x LUMINA_A210_LINE_SATURATION_DIAG="2"' \
  "$base_env" || die "IDSEAL line-saturation mode mismatch"
grep -qFx 'declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION="3"' \
  "$base_env" || die "IDSEAL target-ion seal mismatch"
grep -qFx 'declare -x LUMINA_CMF_FINE_MGPU_DEVICES="2"' \
  "$base_env" || die "IDSEAL two-device owner mismatch"

head="$(git -C "$repo" rev-parse HEAD)"
[[ "$head" =~ ^[0-9a-f]{40,64}$ ]] || die "invalid repository HEAD"
for path in \
  scripts/run_det_convergence_2026-08-08.slurm \
  scripts/check_a210_targeted_gate.py \
  scripts/stage_det_stage12_l6_probe.sh \
  scripts/analyze_det_stage12_l6.py; do
  git -C "$repo" cat-file -e "HEAD:$path" 2>/dev/null || \
    die "required path is not committed at HEAD: $path"
done

base_manifest_before="$(mktemp)"
base_manifest_after="$(mktemp)"
temporary_paths+=("$base_manifest_before" "$base_manifest_after")
write_base_input_manifest "$base_manifest_before"
base_input_sha256_before="$(sha256sum "$base_manifest_before" | cut -d' ' -f1)"

mkdir -p "$dest"
cp -al "$base/input" "$dest/input"
input="$dest/input"
atomic_install_file "$binary" "$input/lumina_cuda" 0750
for spec in \
  'scripts/run_det_convergence_2026-08-08.slurm:job.slurm' \
  'scripts/check_a210_targeted_gate.py:check_a210_targeted_gate.py' \
  'scripts/stage_det_stage12_l6_probe.sh:stage_det_stage12_l6_probe.sh' \
  'scripts/analyze_det_stage12_l6.py:analyze_det_stage12_l6.py'; do
  source_path="${spec%%:*}"
  target_name="${spec#*:}"
  atomic_capture_file "$input/$target_name" 0750 \
    git -C "$repo" show "HEAD:$source_path"
done

atomic_text_file "$input/outer_iterations.txt" 0640 2
atomic_text_file "$input/diagnostic_mode.txt" 0640 A210_L6_PROBE
if [[ -f "$input/requested_diag_te_K.txt" ]]; then
  atomic_text_file "$input/requested_diag_te_K.txt" 0640 10020
fi

new_env="$(mktemp)"
temporary_paths+=("$new_env")
changed=0
while IFS= read -r line; do
  case "$line" in
    'declare -x LUMINA_PURE_CMFGEN_ITER='*)
      printf '%s\n' 'declare -x LUMINA_PURE_CMFGEN_ITER="2"'
      changed=$((changed + 1))
      ;;
    'declare -x LUMINA_RADEQ_DIAG_TE_K='*)
      printf '%s\n' 'declare -x LUMINA_RADEQ_DIAG_TE_K="10020"'
      changed=$((changed + 1))
      ;;
    'declare -x LUMINA_A210_SPRODUCER_CAPTURE='*)
      printf '%s\n' 'declare -x LUMINA_A210_SPRODUCER_CAPTURE="1"'
      changed=$((changed + 1))
      ;;
    'declare -x LUMINA_A210_INDEPENDENT_CAPTURE='*)
      printf '%s\n' 'declare -x LUMINA_A210_INDEPENDENT_CAPTURE="1"'
      changed=$((changed + 1))
      ;;
    *)
      printf '%s\n' "${line//$base/$dest}"
      ;;
  esac
done < "$base_env" > "$new_env"
[[ "$changed" -eq 4 ]] || die "resolved env did not expose four env deltas"
atomic_capture_file "$input/resolved_lumina.exports" 0640 \
  env LC_ALL=C sort -u "$new_env"

atomic_text_file "$input/git_head.txt" 0640 "$head"
atomic_capture_file "$input/git_status.txt" 0640 \
  git -C "$repo" status --short
atomic_capture_file "$input/git_diff.stat" 0640 \
  git -C "$repo" diff --stat
python_version="$(python3 --version 2>&1)"
atomic_text_file "$input/python3.version.txt" 0640 "$python_version"
binary_sha256="$(sha256sum "$input/lumina_cuda" | cut -d' ' -f1)"
atomic_text_file "$input/binary.sha256" 0640 "$binary_sha256"
resolved_sha256="$(sha256sum "$input/resolved_lumina.exports")"
atomic_text_file "$input/resolved_lumina.sha256" 0640 "$resolved_sha256"
flight_scripts_sha256="$(sha256sum \
  "$input/job.slurm" \
  "$input/check_a210_targeted_gate.py" \
  "$input/stage_det_stage12_l6_probe.sh" \
  "$input/analyze_det_stage12_l6.py" \
  "$input/check_det_convergence.py")"
atomic_text_file "$input/flight_scripts.sha256" 0640 \
  "$flight_scripts_sha256"

for name in \
  'LUMINA_PURE_CMFGEN_ITER="2"' \
  'LUMINA_RADEQ_DIAG_TE_K="10020"' \
  'LUMINA_A210_SPRODUCER_CAPTURE="1"' \
  'LUMINA_A210_INDEPENDENT_CAPTURE="1"'; do
  [[ "$(grep -cFx "declare -x $name" "$input/resolved_lumina.exports" || true)" -eq 1 ]] || \
    die "staged env delta is missing: $name"
done
[[ "$(<"$input/outer_iterations.txt")" == 2 && \
   "$(<"$input/diagnostic_mode.txt")" == A210_L6_PROBE ]] || \
  die "staged execution-mode delta is incomplete"

for spec in \
  'scripts/run_det_convergence_2026-08-08.slurm:job.slurm' \
  'scripts/check_a210_targeted_gate.py:check_a210_targeted_gate.py' \
  'scripts/stage_det_stage12_l6_probe.sh:stage_det_stage12_l6_probe.sh'; do
  source_path="${spec%%:*}"
  target_name="${spec#*:}"
  head_sha="$(git -C "$repo" show "HEAD:$source_path" | sha256sum | cut -d' ' -f1)"
  staged_sha="$(sha256sum "$input/$target_name" | cut -d' ' -f1)"
  [[ "$head_sha" == "$staged_sha" ]] || \
    die "HEAD freshness seal mismatch: $target_name"
done

write_base_input_manifest "$base_manifest_after"
base_input_sha256_after="$(sha256sum "$base_manifest_after" | cut -d' ' -f1)"
[[ "$base_input_sha256_before" == "$base_input_sha256_after" ]] && \
  cmp -s "$base_manifest_before" "$base_manifest_after" || \
  die "base input byte seal mismatch before=$base_input_sha256_before after=$base_input_sha256_after"

atomic_text_file "$dest/L6_STAGE_SEAL.txt" 0640 \
  "schema=DET_STAGE12_L6_STAGE_V1" \
  "base_run_root=$base" \
  "base_input_sha256_before=$base_input_sha256_before" \
  "base_input_sha256_after=$base_input_sha256_after" \
  "base_input_byte_invariant=1" \
  "repo_head=$head" \
  "outer_iterations=2" \
  "diagnostic_mode=A210_L6_PROBE" \
  "resolved_env_delta_count=4" \
  "execution_delta_count=5" \
  "physical_values_modified=0" \
  "floor=0" "cap=0" "clamp=0" "jitter=0" "repair=0"
atomic_text_file "$dest/READY" 0640 READY
printf 'DET_STAGE12_L6_STAGE_OK run_root=%s base=%s head=%s iterations=2 mode=A210_L6_PROBE\n' \
  "$dest" "$base" "$head"
