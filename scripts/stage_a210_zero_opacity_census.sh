#!/usr/bin/env bash
# Stage A210_ZO_CENSUS from the sealed L6C run without mutating or hard-linking it.
set -euo pipefail
umask 027

readonly SEALED_L6C_ROOT=/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/l6c_20260822T154507Z_cover

die() {
  printf 'A210_ZERO_OPACITY_CENSUS_STAGE_FATAL %s\n' "$*" >&2
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
  local source="$1" from_root="$2" to_root="$3"
  local line name
  rewritten_exports=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      'declare -x LUMINA_A210_SPRODUCER_CAPTURE="1"')
        printf '%s\n' 'declare -x LUMINA_A210_SPRODUCER_CAPTURE="0"'
        rewritten_exports=$((rewritten_exports + 1))
        ;;
      'declare -x LUMINA_A210_INDEPENDENT_CAPTURE="1"')
        printf '%s\n' 'declare -x LUMINA_A210_INDEPENDENT_CAPTURE="0"'
        rewritten_exports=$((rewritten_exports + 1))
        ;;
      'declare -x LUMINA_A210_SPRODUCER_CAPTURE='*|\
      'declare -x LUMINA_A210_INDEPENDENT_CAPTURE='*)
        name="${line#declare -x }"
        name="${name%%=*}"
        die "sealed L6C capture value is not 1: $name"
        ;;
      *)
        printf '%s\n' "${line//"$from_root"/"$to_root"}"
        ;;
    esac
  done < "$source"
}

verify_export_delta() {
  local base_env="$1" staged_env="$2" base_root="$3" dest_root="$4"
  local normalized expected rc=0
  normalized="$(mktemp)"
  expected="$(mktemp)"
  sed "s|$dest_root|$base_root|g" "$staged_env" > "$normalized"
  sed \
    -e 's/^declare -x LUMINA_A210_SPRODUCER_CAPTURE="1"$/declare -x LUMINA_A210_SPRODUCER_CAPTURE="0"/' \
    -e 's/^declare -x LUMINA_A210_INDEPENDENT_CAPTURE="1"$/declare -x LUMINA_A210_INDEPENDENT_CAPTURE="0"/' \
    "$base_env" > "$expected"
  cmp -s "$normalized" "$expected" || rc=1
  rm -f -- "$normalized" "$expected"
  return "$rc"
}

verify_no_shared_file_inode() {
  local source_root="$1" copied_root="$2" source rel copied
  while IFS= read -r -d '' source; do
    rel="${source#"$source_root"/}"
    copied="$copied_root/$rel"
    [[ -f "$copied" && ! -L "$copied" ]] || die "copied input missing regular file: $rel"
    [[ "$(stat -c '%d:%i' "$source")" != "$(stat -c '%d:%i' "$copied")" ]] || \
      die "hard-linked staged input is forbidden: $rel"
  done < <(find "$source_root" -type f -print0)
}

stager_selftest() {
  local scratch base dest actual expected
  scratch="$(mktemp -d)"
  base="$scratch/base.exports"
  dest="$scratch/dest.exports"
  actual="$scratch/actual.exports"
  expected="$scratch/expected.exports"
  trap 'rm -rf -- "$scratch"' EXIT
  printf '%s\n' \
    'declare -x LUMINA_MODEL_DIR="/sealed/model"' \
    'declare -x LUMINA_A210_SPRODUCER_CAPTURE="1"' \
    'declare -x LUMINA_A210_INDEPENDENT_CAPTURE="1"' \
    'declare -x LUMINA_A210_LINE_SATURATION_DIAG="2"' \
    'declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION="1"' > "$base"
  rewrite_exports "$base" /sealed /scratch/census > "$actual"
  [[ "$rewritten_exports" -eq 2 ]] || die "selftest did not rewrite exactly two capture exports"
  printf '%s\n' \
    'declare -x LUMINA_MODEL_DIR="/scratch/census/model"' \
    'declare -x LUMINA_A210_SPRODUCER_CAPTURE="0"' \
    'declare -x LUMINA_A210_INDEPENDENT_CAPTURE="0"' \
    'declare -x LUMINA_A210_LINE_SATURATION_DIAG="2"' \
    'declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION="1"' > "$expected"
  cmp -s "$actual" "$expected" || die "selftest export rewrite bytes differ"
  verify_export_delta "$base" "$actual" /sealed /scratch/census || \
    die "selftest exact export delta rejected"
  printf 'NC-ZS0 inject=NONE status=PASS reason=EXPORT_DELTA_EXACTLY_TWO actual=%s\n' \
    "$rewritten_exports"
  sed 's/LUMINA_A210_INDEPENDENT_CAPTURE="0"/LUMINA_A210_INDEPENDENT_CAPTURE="1"/' \
    "$actual" > "$dest"
  if verify_export_delta "$base" "$dest" /sealed /scratch/census 2>/dev/null; then
    die "selftest accepted removed capture delta"
  fi
  printf 'NC-ZS1 inject=REMOVE_INDEPENDENT_CAPTURE_DELTA status=FAIL reason=EXPORT_DELTA_NOT_EXACTLY_TWO\n'
  printf 'NC-ZS1 remove=RESTORE_CAPTURE_DELTA status=PASS reason=EXPORT_DELTA_EXACTLY_TWO\n'
  printf 'A210_ZERO_OPACITY_CENSUS_STAGER_SELFTEST_PASS controls=NC-ZS0,NC-ZS1\n'
  rm -rf -- "$scratch"
  trap - EXIT
}

if [[ $# -eq 1 && "$1" == --selftest ]]; then
  stager_selftest
  exit 0
fi

[[ $# -eq 2 ]] || die "usage: $0 NEW_RUN_ROOT ABSOLUTE_BINARY"
dest="$1"
binary="$2"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
base="$SEALED_L6C_ROOT"

[[ "$dest" = /* && "$dest" != / && "$dest" != /gpfs && ! -e "$dest" ]] || \
  die "destination must be a new safe absolute path: $dest"
[[ "$binary" = /* && -x "$binary" && ! -L "$binary" ]] || \
  die "binary is not a safe executable: $binary"
[[ -d "$base/input" && -f "$base/READY" && ! -L "$base/READY" ]] || \
  die "sealed L6C root is unavailable or unsealed: $base"
for path in \
  "$base/input/resolved_lumina.exports" \
  "$base/input/outer_iterations.txt" \
  "$base/input/diagnostic_mode.txt"; do
  [[ -f "$path" && ! -L "$path" ]] || die "unsafe sealed L6C input: $path"
done
[[ "$(<"$base/input/outer_iterations.txt")" == 2 ]] || \
  die "sealed L6C outer-iteration count is not two"
[[ "$(<"$base/input/diagnostic_mode.txt")" == A210_L6C_PROBE ]] || \
  die "sealed base is not the L6C ion-II diagnostic"
for export_line in \
  'declare -x LUMINA_A210_SPRODUCER_CAPTURE="1"' \
  'declare -x LUMINA_A210_INDEPENDENT_CAPTURE="1"' \
  'declare -x LUMINA_A210_LINE_SATURATION_DIAG="2"' \
  'declare -x LUMINA_A210_LINE_SATURATION_TARGET_ION="1"'; do
  [[ "$(grep -cFx "$export_line" "$base/input/resolved_lumina.exports" || true)" -eq 1 ]] || \
    die "sealed L6C export mismatch: $export_line"
done
if find "$base/input" -type l -print -quit | grep -q .; then
  die "sealed L6C input contains symlink"
fi
for source in \
  "$repo/scripts/run_det_convergence_2026-08-08.slurm" \
  "$repo/scripts/stage_a210_zero_opacity_census.sh" \
  "$repo/scripts/analyze_a210_zero_opacity_census.py"; do
  [[ -f "$source" && ! -L "$source" ]] || die "missing safe staged source: $source"
done

mkdir -p "$dest"
# cp -R does not preserve hard links.  Reflinks, when supported, are distinct
# inodes and copy-on-write; the inode check below proves the sealed root is not shared.
cp -R --reflink=auto --preserve=mode,timestamps -- "$base/input" "$dest/input"
verify_no_shared_file_inode "$base/input" "$dest/input"

atomic_install_file "$binary" "$dest/input/lumina_cuda" 0750
atomic_install_file "$repo/scripts/run_det_convergence_2026-08-08.slurm" \
  "$dest/input/job.slurm" 0750
atomic_install_file "$repo/scripts/stage_a210_zero_opacity_census.sh" \
  "$dest/input/stage_a210_zero_opacity_census.sh" 0750
atomic_install_file "$repo/scripts/analyze_a210_zero_opacity_census.py" \
  "$dest/input/analyze_a210_zero_opacity_census.py" 0750

new_env="$(mktemp)"
trap 'rm -f -- "$new_env"' EXIT
rewrite_exports "$base/input/resolved_lumina.exports" "$base" "$dest" > "$new_env"
[[ "$rewritten_exports" -eq 2 ]] || die "resolved env delta count is not exactly two"
atomic_install_file "$new_env" "$dest/input/resolved_lumina.exports" 0640
verify_export_delta "$base/input/resolved_lumina.exports" \
  "$dest/input/resolved_lumina.exports" "$base" "$dest" || \
  die "resolved env contains a delta other than the two capture toggles or path relocation"
rm -f -- "$new_env"
trap - EXIT

atomic_text_file "$dest/input/outer_iterations.txt" 0640 2
atomic_text_file "$dest/input/diagnostic_mode.txt" 0640 A210_ZO_CENSUS
atomic_text_file "$dest/input/git_head.txt" 0640 "$(git -C "$repo" rev-parse HEAD)"
atomic_text_file "$dest/input/git_status.txt" 0640 "$(git -C "$repo" status --short)"
atomic_text_file "$dest/input/git_diff.stat" 0640 "$(git -C "$repo" diff --stat)"
atomic_text_file "$dest/input/binary.sha256" 0640 "$(sha256sum "$dest/input/lumina_cuda" | cut -d' ' -f1)"
atomic_text_file "$dest/input/resolved_lumina.sha256" 0640 \
  "$(sha256sum "$dest/input/resolved_lumina.exports")"
atomic_capture_tmp="$(mktemp)"
trap 'rm -f -- "$atomic_capture_tmp"' EXIT
sha256sum "$dest/input/job.slurm" \
  "$dest/input/stage_a210_zero_opacity_census.sh" \
  "$dest/input/analyze_a210_zero_opacity_census.py" > "$atomic_capture_tmp"
atomic_install_file "$atomic_capture_tmp" "$dest/input/flight_scripts.sha256" 0640
rm -f -- "$atomic_capture_tmp"
trap - EXIT

atomic_text_file "$dest/OWNER.txt" 0640 \
  'project=LUMINA' \
  "repo=$repo" \
  'purpose=A210_ZO_CENSUS ion=1 capture=0 census' \
  'DO_NOT_CANCEL=1'
atomic_text_file "$dest/ZO_CENSUS_STAGE_SEAL.txt" 0640 \
  'schema=A210_ZERO_OPACITY_CENSUS_STAGE_V1' \
  "sealed_l6c_base=$base" \
  'diagnostic_mode=A210_ZO_CENSUS' \
  'outer_iterations=2' \
  'target_ion=1' \
  'line_saturation_diag=2' \
  'producer_capture=0' \
  'independent_capture=0' \
  'resolved_env_delta_count=2' \
  'sealed_input_shared_hardlinks=0' \
  'physical_values_modified=0' \
  'floor=0' 'cap=0' 'clamp=0' 'jitter=0' 'repair=0'
atomic_text_file "$dest/submit_census.sh" 0750 \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'run_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"' \
  'exec sbatch --partition=a100 --gres=gpu:2 --mem=64G --time=08:00:00 --job-name=LUMINA_ZO_CENSUS_II "$run_root/input/job.slurm" "$run_root"'
atomic_text_file "$dest/READY" 0640 READY

printf 'A210_ZERO_OPACITY_CENSUS_STAGE_OK run_root=%s base=%s env_delta=2 mode=A210_ZO_CENSUS\n' \
  "$dest" "$base"
