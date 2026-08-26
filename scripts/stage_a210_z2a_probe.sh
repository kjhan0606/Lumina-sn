#!/usr/bin/env bash
# Stage A210_Z2A_PROBE from sealed inputs without sharing any input inode.
set -euo pipefail
umask 027

readonly SEALED_L6C_ROOT=/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/l6c_20260822T154507Z_cover
readonly SEALED_L6_ROOT=/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe
readonly SEALED_CENSUS_STDERR=/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1/stderr.log

die() {
  printf 'A210_Z2A_PROBE_STAGE_FATAL %s\n' "$*" >&2
  exit 70
}

new_temporary() {
  local target="$1"
  [[ ! -d "$target" ]] || die "refusing to replace directory: $target"
  stage_temporary="${target}.tmp.$$"
  [[ ! -e "$stage_temporary" && ! -L "$stage_temporary" ]] || \
    die "temporary path already exists: $stage_temporary"
}

atomic_text_file() {
  local target="$1" mode="$2"
  shift 2
  new_temporary "$target"
  printf '%s\n' "$@" > "$stage_temporary"
  chmod "$mode" "$stage_temporary"
  mv -f -- "$stage_temporary" "$target"
}

atomic_install_file() {
  local source="$1" target="$2" mode="$3"
  new_temporary "$target"
  install -m "$mode" -- "$source" "$stage_temporary"
  mv -f -- "$stage_temporary" "$target"
}

rewrite_exports() {
  local source="$1" from_root="$2" to_root="$3" target_ion="$4"
  local line
  rewritten_exports=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      'declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION="'*)
        if [[ "$line" == "declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION=\"$target_ion\"" ]]; then
          printf '%s\n' "$line"
        else
          printf 'declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION="%s"\n' \
            "$target_ion"
          rewritten_exports=$((rewritten_exports + 1))
        fi
        ;;
      *)
        printf '%s\n' "${line//"$from_root"/"$to_root"}"
        ;;
    esac
  done < "$source"
}

verify_export_delta() {
  local base_env="$1" staged_env="$2" base_root="$3" dest_root="$4" target_ion="$5"
  local normalized expected rc=0
  normalized="$(mktemp)"
  expected="$(mktemp)"
  sed "s|$dest_root|$base_root|g" "$staged_env" > "$normalized"
  sed \
    "s/^declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION=\"[0-9][0-9]*\"$/declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION=\"$target_ion\"/" \
    "$base_env" > "$expected"
  cmp -s "$normalized" "$expected" || rc=1
  rm -f -- "$normalized" "$expected"
  return "$rc"
}

verify_no_shared_file_inode() {
  local source_root="$1" copied_root="$2" source rel copied
  shared_hardlinks=0
  shared_hardlink_detail=""
  while IFS= read -r -d '' source; do
    rel="${source#"$source_root"/}"
    copied="$copied_root/$rel"
    if [[ ! -f "$copied" || -L "$copied" ]]; then
      shared_hardlink_detail="MISSING_OR_UNSAFE:$rel"
      return 1
    fi
    if [[ "$(stat -c '%d:%i' "$source")" == "$(stat -c '%d:%i' "$copied")" ]]; then
      shared_hardlinks=$((shared_hardlinks + 1))
      shared_hardlink_detail="$rel"
    fi
  done < <(find "$source_root" -type f -print0)
  [[ "$shared_hardlinks" -eq 0 ]]
}

stager_selftest() {
  local scratch base dest base_env actual bad
  scratch="$(mktemp -d)"
  base="$scratch/base"
  dest="$scratch/dest"
  base_env="$base/input/resolved_lumina.exports"
  actual="$scratch/actual.exports"
  bad="$scratch/bad.exports"
  trap 'rm -rf -- "$scratch"' EXIT
  mkdir -p "$base/input" "$dest/input"
  printf '%s\n' payload > "$base/input/ordinary.txt"
  cp -R --reflink=auto --preserve=mode,timestamps -- "$base/input/." "$dest/input/"
  verify_no_shared_file_inode "$base/input" "$dest/input" || \
    die "selftest copied input shares an inode: $shared_hardlink_detail"
  printf 'NC-S1 inject=NONE status=PASS reason=SEALED_BASE_SHARED_HARDLINKS actual=%s\n' \
    "$shared_hardlinks"
  rm -f -- "$dest/input/ordinary.txt"
  ln "$base/input/ordinary.txt" "$dest/input/ordinary.txt"
  if verify_no_shared_file_inode "$base/input" "$dest/input"; then
    die "selftest accepted injected hardlink"
  fi
  printf 'NC-S1 inject=HARDLINKED_INPUT status=FAIL reason=SEALED_BASE_SHARED_HARDLINKS actual=%s detail=%s\n' \
    "$shared_hardlinks" "$shared_hardlink_detail"
  rm -f -- "$dest/input/ordinary.txt"
  cp -- "$base/input/ordinary.txt" "$dest/input/ordinary.txt"
  verify_no_shared_file_inode "$base/input" "$dest/input" || \
    die "selftest restored copy still shares an inode: $shared_hardlink_detail"
  printf 'NC-S1 remove=RESTORE_COPY status=PASS reason=SEALED_BASE_SHARED_HARDLINKS actual=%s\n' \
    "$shared_hardlinks"
  printf '%s\n' \
    'declare -x LUMINA_MODEL_DIR="/sealed/model"' \
    'declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION="1"' \
    'declare -x LUMINA_A210_SPRODUCER_CAPTURE="1"' \
    'declare -x LUMINA_A210_INDEPENDENT_CAPTURE="1"' \
    'declare -x LUMINA_A210_LINE_SATURATION_DIAG="2"' > "$base_env"
  rewrite_exports "$base_env" /sealed /scratch/z2a 2 > "$actual"
  [[ "$rewritten_exports" -eq 1 ]] || die "selftest target-ion delta count mismatch"
  verify_export_delta "$base_env" "$actual" /sealed /scratch/z2a 2 || \
    die "selftest rejected exact target-ion delta"
  printf 'NC-S2 inject=NONE status=PASS reason=EXPORT_DELTA_EXACT actual=%s\n' \
    "$rewritten_exports"
  sed 's/LUMINA_A210_SPRODUCER_CAPTURE="1"/LUMINA_A210_SPRODUCER_CAPTURE="0"/' \
    "$actual" > "$bad"
  if verify_export_delta "$base_env" "$bad" /sealed /scratch/z2a 2; then
    die "selftest accepted injected extra export delta"
  fi
  printf 'NC-S2 inject=EXTRA_EXPORT_DELTA status=FAIL reason=EXPORT_DELTA_NOT_EXACT\n'
  verify_export_delta "$base_env" "$actual" /sealed /scratch/z2a 2 || \
    die "selftest restore of exact target-ion delta failed"
  printf 'NC-S2 remove=RESTORE_EXPORTS status=PASS reason=EXPORT_DELTA_EXACT actual=%s\n' \
    "$rewritten_exports"
  printf 'A210_Z2A_PROBE_STAGER_SELFTEST_PASS controls=NC-S1,NC-S2\n'
  rm -rf -- "$scratch"
  trap - EXIT
}

if [[ $# -eq 1 && "$1" == --selftest ]]; then
  stager_selftest
  exit 0
fi

[[ $# -eq 3 ]] || die "usage: $0 NEW_RUN_ROOT ABSOLUTE_BINARY TARGET_ION"
dest="$1"
binary="$2"
target_ion="$3"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

case "$target_ion" in
  1|2)
    base="$SEALED_L6C_ROOT"
    expected_base_mode=A210_L6C_PROBE
    expected_base_target=1
    ;;
  3)
    base="$SEALED_L6_ROOT"
    expected_base_mode=A210_L6_PROBE
    expected_base_target=3
    ;;
  *) die "Z2A target ion must be 1, 2, or 3: $target_ion" ;;
esac

[[ "$dest" = /* && "$dest" != / && "$dest" != /gpfs && ! -e "$dest" ]] || \
  die "destination must be a new safe absolute path: $dest"
[[ "$binary" = /* && -x "$binary" && ! -L "$binary" ]] || \
  die "binary is not a safe executable: $binary"
[[ -d "$base/input" && -f "$base/READY" && ! -L "$base/READY" ]] || \
  die "sealed base is unavailable or unsealed: $base"
[[ -f "$SEALED_CENSUS_STDERR" && ! -L "$SEALED_CENSUS_STDERR" ]] || \
  die "sealed census stderr is unavailable or unsafe"
for path in \
  "$base/input/resolved_lumina.exports" \
  "$base/input/outer_iterations.txt" \
  "$base/input/diagnostic_mode.txt"; do
  [[ -f "$path" && ! -L "$path" ]] || die "unsafe sealed base input: $path"
done
[[ "$(<"$base/input/outer_iterations.txt")" == 2 ]] || \
  die "sealed base outer-iteration count is not two"
[[ "$(<"$base/input/diagnostic_mode.txt")" == "$expected_base_mode" ]] || \
  die "sealed base diagnostic mode does not match target ion"
for export_line in \
  'declare -x LUMINA_RADEQ_DIAG_TE_K="10020"' \
  'declare -x LUMINA_A210_SPRODUCER_CAPTURE="1"' \
  'declare -x LUMINA_A210_INDEPENDENT_CAPTURE="1"' \
  'declare -x LUMINA_A210_LINE_SATURATION_DIAG="2"' \
  "declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION=\"$expected_base_target\""; do
  [[ "$(grep -cFx "$export_line" "$base/input/resolved_lumina.exports" || true)" -eq 1 ]] || \
    die "sealed base export mismatch: $export_line"
done
if find "$base/input" -type l -print -quit | grep -q .; then
  die "sealed base input contains a symlink"
fi
for source in \
  "$repo/scripts/run_det_convergence_2026-08-08.slurm" \
  "$repo/scripts/stage_a210_z2a_probe.sh" \
  "$repo/scripts/analyze_a210_z2a_probe.py"; do
  [[ -f "$source" && ! -L "$source" ]] || die "missing safe staged source: $source"
done

mkdir -p "$dest"
# Reflinks are acceptable only because the inode check proves the sealed base
# is not shared.  Hard links are forbidden even if the destination is later
# modified atomically.
cp -R --reflink=auto --preserve=mode,timestamps -- "$base/input" "$dest/input"
verify_no_shared_file_inode "$base/input" "$dest/input" || \
  die "sealed base hard-link invariant failed: $shared_hardlink_detail"

atomic_install_file "$binary" "$dest/input/lumina_cuda" 0750
atomic_install_file "$repo/scripts/run_det_convergence_2026-08-08.slurm" \
  "$dest/input/job.slurm" 0750
atomic_install_file "$repo/scripts/stage_a210_z2a_probe.sh" \
  "$dest/input/stage_a210_z2a_probe.sh" 0750
atomic_install_file "$repo/scripts/analyze_a210_z2a_probe.py" \
  "$dest/input/analyze_a210_z2a_probe.py" 0750

new_env="$(mktemp)"
trap 'rm -f -- "$new_env"' EXIT
rewrite_exports "$base/input/resolved_lumina.exports" "$base" "$dest" \
  "$target_ion" > "$new_env"
expected_export_delta=0
[[ "$target_ion" != "$expected_base_target" ]] && expected_export_delta=1
[[ "$rewritten_exports" -eq "$expected_export_delta" ]] || \
  die "resolved env delta count mismatch: actual=$rewritten_exports expected=$expected_export_delta"
atomic_install_file "$new_env" "$dest/input/resolved_lumina.exports" 0640
verify_export_delta "$base/input/resolved_lumina.exports" \
  "$dest/input/resolved_lumina.exports" "$base" "$dest" "$target_ion" || \
  die "resolved env contains an export delta outside target-ion selection and path relocation"
rm -f -- "$new_env"
trap - EXIT

atomic_text_file "$dest/input/outer_iterations.txt" 0640 2
atomic_text_file "$dest/input/diagnostic_mode.txt" 0640 A210_Z2A_PROBE
atomic_text_file "$dest/input/z2a_target_ion.txt" 0640 "$target_ion"
atomic_text_file "$dest/input/z2a_census_stderr.txt" 0640 "$SEALED_CENSUS_STDERR"
atomic_text_file "$dest/input/git_head.txt" 0640 "$(git -C "$repo" rev-parse HEAD)"
atomic_text_file "$dest/input/git_status.txt" 0640 "$(git -C "$repo" status --short)"
atomic_text_file "$dest/input/git_diff.stat" 0640 "$(git -C "$repo" diff --stat)"
atomic_text_file "$dest/input/binary.sha256" 0640 \
  "$(sha256sum "$dest/input/lumina_cuda" | cut -d' ' -f1)"
atomic_text_file "$dest/input/resolved_lumina.sha256" 0640 \
  "$(sha256sum "$dest/input/resolved_lumina.exports" | cut -d' ' -f1)"
script_hashes="$(mktemp)"
trap 'rm -f -- "$script_hashes"' EXIT
sha256sum "$dest/input/job.slurm" "$dest/input/stage_a210_z2a_probe.sh" \
  "$dest/input/analyze_a210_z2a_probe.py" > "$script_hashes"
atomic_install_file "$script_hashes" "$dest/input/flight_scripts.sha256" 0640
rm -f -- "$script_hashes"
trap - EXIT

[[ "$(<"$dest/input/outer_iterations.txt")" == 2 && \
   "$(<"$dest/input/diagnostic_mode.txt")" == A210_Z2A_PROBE && \
   "$(<"$dest/input/z2a_target_ion.txt")" == "$target_ion" ]] || \
  die "staged execution-mode delta is incomplete"
[[ "$(grep -cFx "declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION=\"$target_ion\"" \
     "$dest/input/resolved_lumina.exports" || true)" -eq 1 ]] || \
  die "staged target-ion export is missing"

atomic_text_file "$dest/OWNER.txt" 0640 \
  'project=LUMINA' \
  "repo=$repo" \
  "purpose=A210_Z2A_PROBE target_ion=$target_ion diagnostic_isolation" \
  'DO_NOT_CANCEL=1'
atomic_text_file "$dest/Z2A_STAGE_SEAL.txt" 0640 \
  'schema=A210_Z2A_PROBE_STAGE_V1' \
  "sealed_base=$base" \
  "sealed_census_stderr=$SEALED_CENSUS_STDERR" \
  'diagnostic_mode=A210_Z2A_PROBE' \
  'outer_iterations=2' \
  "target_ion=$target_ion" \
  "resolved_env_delta_count=$expected_export_delta" \
  "sealed_input_shared_hardlinks=$shared_hardlinks" \
  'physical_values_modified=0' \
  'floor=0' 'cap=0' 'clamp=0' 'jitter=0' 'repair=0'
atomic_text_file "$dest/READY" 0640 READY

printf 'A210_Z2A_PROBE_STAGE_OK run_root=%s base=%s target_ion=%s env_delta=%s shared_hardlinks=%s\n' \
  "$dest" "$base" "$target_ion" "$expected_export_delta" "$shared_hardlinks"
