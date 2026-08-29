#!/usr/bin/env bash
# SUPERLEVEL_B_PROMOTION: stage B-1/B-2/B-3 from the sealed R-II input.
set -euo pipefail
umask 027

readonly RII_ROOT=/gpfs/kjhan/lumina/a210_z2a_diag_isolation_a100x2_k36/rii_20260826T134337Z_ion1
readonly CENSUS_ROOT=/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1
readonly RII_MANIFEST=RII_RUN_SEAL.sha
readonly CENSUS_MANIFEST=ZO_RUN_SEAL.sha
readonly SLB_PARENT=/gpfs/kjhan/lumina/superlevel_b_a100x2_k36
readonly BINARY_REL=input/lumina_cuda
readonly LEVELS_REL=input/model/levels.csv
readonly LINES_REL=input/model/line_list.csv
readonly PATH_KEYS='LUMINA_CMFGEN_SIGMA_BF LUMINA_DEPOSITION_FILE LUMINA_IONIZATION_REFERENCE_FILE LUMINA_MODEL_DIR LUMINA_PHYSICS_COMPARISON_DIR LUMINA_TOPION_LEVELS_FILE'

die() {
  printf 'SUPERLEVEL_B_STAGE_FATAL reason=%s\n' "$*" >&2
  exit 70
}

new_temporary() {
  local target="$1"
  stage_temporary="${target}.tmp.$$"
  [[ ! -e "$stage_temporary" && ! -L "$stage_temporary" ]] || \
    die "TEMPORARY_PATH_EXISTS path=$stage_temporary"
}

atomic_text_file() {
  local target="$1" mode="$2"
  shift 2
  [[ ! -L "$target" ]] || die "OUTPUT_PATH_SYMLINK path=$target"
  new_temporary "$target"
  printf '%s\n' "$@" > "$stage_temporary"
  chmod "$mode" "$stage_temporary"
  mv -f -- "$stage_temporary" "$target"
}

atomic_install_file() {
  local source="$1" target="$2" mode="$3"
  [[ -f "$source" && ! -L "$source" ]] || die "SOURCE_NOT_REGULAR path=$source"
  [[ ! -L "$target" ]] || die "OUTPUT_PATH_SYMLINK path=$target"
  new_temporary "$target"
  install -m "$mode" -- "$source" "$stage_temporary"
  mv -f -- "$stage_temporary" "$target"
}

require_safe_absolute_new_path() {
  local target="$1" cursor parent
  [[ "$target" = /* && "$target" != / && "$target" != /gpfs && \
     ! -e "$target" && ! -L "$target" ]] || \
    die "DESTINATION_NOT_NEW_SAFE_ABSOLUTE path=$target"
  parent="$(dirname "$target")"
  [[ -d "$parent" && ! -L "$parent" ]] || die "DESTINATION_PARENT_UNSAFE path=$parent"
  cursor="$parent"
  while [[ "$cursor" != / ]]; do
    [[ ! -L "$cursor" ]] || die "DESTINATION_ANCESTOR_SYMLINK path=$cursor"
    cursor="$(dirname "$cursor")"
  done
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

manifest_digest() {
  local manifest="$1" wanted="$2" line digest name count=0 answer=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^([0-9a-f]{64})[[:space:]][\ \*](.+)$ ]] || \
      die "RUN_SEAL_MANIFEST_LINE_INVALID manifest=$manifest"
    digest="${BASH_REMATCH[1]}"
    name="${BASH_REMATCH[2]}"
    name="${name#./}"
    if [[ "$name" == "$wanted" ]]; then
      answer="$digest"
      count=$((count + 1))
    fi
  done < "$manifest"
  [[ "$count" -eq 1 ]] || \
    die "RUN_SEAL_MEMBER_COUNT manifest=$manifest member=$wanted count=$count"
  printf '%s\n' "$answer"
}

verify_run_seal() {
  local root="$1" manifest_name="$2" expected_lines="$3" schema="$4"
  local metadata_name="$5" log lines ok non_ok rc
  local manifest="$root/$manifest_name" metadata="$root/$metadata_name"
  [[ -f "$manifest" && ! -L "$manifest" ]] || \
    die "RUN_SEAL_MANIFEST_UNSAFE path=$manifest"
  [[ -f "$metadata" && ! -L "$metadata" ]] || \
    die "RUN_SEAL_METADATA_UNSAFE path=$metadata"
  lines="$(wc -l < "$manifest")"
  [[ "$lines" -eq "$expected_lines" ]] || \
    die "RUN_SEAL_LINE_COUNT path=$manifest actual=$lines expected=$expected_lines"
  for exact in "schema=$schema" "run_root=$root" 'DO_NOT_MUTATE=1'; do
    [[ "$(grep -cFx "$exact" "$metadata" || true)" -eq 1 ]] || \
      die "RUN_SEAL_METADATA_MISMATCH path=$metadata token=$exact"
  done
  log="$(mktemp)"
  set +e
  (cd "$root" && sha256sum -c "$manifest_name") > "$log" 2>&1
  rc=$?
  set -e
  ok="$(grep -c ': OK$' "$log" || true)"
  non_ok=$(( $(wc -l < "$log") - ok ))
  rm -f -- "$log"
  [[ "$rc" -eq 0 && "$ok" -eq "$expected_lines" && "$non_ok" -eq 0 ]] || \
    die "RUN_SEAL_CHECK_FAILED root=$root rc=$rc ok=$ok non_ok=$non_ok"
  printf 'SUPERLEVEL_B_RUN_SEAL_PASS root=%s entries=%s non_ok=0\n' \
    "$root" "$ok"
}

registered_input_digests() {
  local manifest="$RII_ROOT/$RII_MANIFEST"
  registered_binary="$(manifest_digest "$manifest" "$BINARY_REL")"
  registered_levels="$(manifest_digest "$manifest" "$LEVELS_REL")"
  registered_lines="$(manifest_digest "$manifest" "$LINES_REL")"
  [[ "$registered_binary" == 39d452e4*aa42 ]] || \
    die "REGISTERED_BINARY_DIGEST_MISMATCH digest=$registered_binary"
  [[ "$registered_levels" == b3645052*daf67 ]] || \
    die "REGISTERED_LEVELS_DIGEST_MISMATCH digest=$registered_levels"
  [[ "$registered_lines" == cbd81c4f*54b5 ]] || \
    die "REGISTERED_LINES_DIGEST_MISMATCH digest=$registered_lines"
  [[ "$(sha256sum "$RII_ROOT/$BINARY_REL" | cut -d' ' -f1)" == "$registered_binary" ]] || \
    die "SEALED_BINARY_DIGEST_MISMATCH"
  [[ "$(sha256sum "$RII_ROOT/$LEVELS_REL" | cut -d' ' -f1)" == "$registered_levels" ]] || \
    die "SEALED_LEVELS_DIGEST_MISMATCH"
  [[ "$(sha256sum "$RII_ROOT/$LINES_REL" | cut -d' ' -f1)" == "$registered_lines" ]] || \
    die "SEALED_LINES_DIGEST_MISMATCH"
}

rewrite_exports() {
  local source="$1" base_root="$2" dest_root="$3" mode="$4"
  local line name key value found_path=0 found_capture=0 found_super=0 is_path
  declare -A seen=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^declare\ -x\ ([A-Za-z_][A-Za-z0-9_]*)=\"(.*)\"$ ]]; then
      name="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      [[ -z "${seen[$name]:-}" ]] || die "DUPLICATE_BASE_EXPORT key=$name"
      seen[$name]=1
      [[ "$name" != LUMINA_SUPER_LEVELS ]] || found_super=$((found_super + 1))
      is_path=0
      for key in $PATH_KEYS; do
        [[ "$name" != "$key" ]] || is_path=1
      done
      if [[ "$is_path" -eq 1 ]]; then
        [[ "$value" == *"$base_root"* ]] || \
          die "BASE_PATH_EXPORT_DOES_NOT_NAME_RII key=$name value=$value"
        value="${value//"$base_root"/"$dest_root"}"
        found_path=$((found_path + 1))
      elif [[ "$name" == LUMINA_A210_SPRODUCER_CAPTURE || \
              "$name" == LUMINA_A210_INDEPENDENT_CAPTURE ]]; then
        [[ "$value" == 1 ]] || die "BASE_CAPTURE_NOT_ONE key=$name value=$value"
        [[ "$mode" != B-1 ]] || value=0
        found_capture=$((found_capture + 1))
      fi
      printf 'declare -x %s="%s"\n' "$name" "$value"
    else
      printf '%s\n' "$line"
    fi
  done < "$source"
  [[ "$found_path" -eq 6 ]] || die "PATH_EXPORT_COUNT actual=$found_path expected=6"
  [[ "$found_capture" -eq 2 ]] || die "CAPTURE_EXPORT_COUNT actual=$found_capture expected=2"
  [[ "$found_super" -eq 0 ]] || die "BASE_ALREADY_HAS_SUPERLEVELS"
  printf '%s\n' 'declare -x LUMINA_SUPER_LEVELS="1"'
}

parse_exports_to_file() {
  local source="$1" target="$2" line
  : > "$target"
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^declare\ -x\ ([A-Za-z_][A-Za-z0-9_]*)=\"(.*)\"$ ]]; then
      printf '%s=%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" >> "$target"
    fi
  done < "$source"
  LC_ALL=C sort -o "$target" "$target"
  [[ "$(cut -d= -f1 "$target" | uniq -d | wc -l)" -eq 0 ]] || \
    die "EXPORT_DUPLICATE_AFTER_REWRITE path=$source"
}

verify_export_delta() {
  local base_env="$1" staged_env="$2" base_root="$3" dest_root="$4" mode="$5"
  local scratch base_map run_map diff_file key kind old new expected_count=7 observed=0
  scratch="$(mktemp -d)"
  base_map="$scratch/base"; run_map="$scratch/run"; diff_file="$scratch/diff"
  parse_exports_to_file "$base_env" "$base_map"
  parse_exports_to_file "$staged_env" "$run_map"
  if [[ "$mode" == B-1 ]]; then expected_count=9; fi
  python3 -B - "$base_map" "$run_map" "$base_root" "$dest_root" "$mode" > "$diff_file" <<'PY'
import sys
def load(path):
    out = {}
    with open(path, encoding="utf-8") as stream:
        for raw in stream:
            key, value = raw.rstrip("\n").split("=", 1)
            if key in out:
                raise SystemExit("duplicate export")
            out[key] = value
    return out
base, run = load(sys.argv[1]), load(sys.argv[2])
base_root, dest_root, mode = sys.argv[3:]
path_keys = {
    "LUMINA_CMFGEN_SIGMA_BF", "LUMINA_DEPOSITION_FILE",
    "LUMINA_IONIZATION_REFERENCE_FILE", "LUMINA_MODEL_DIR",
    "LUMINA_PHYSICS_COMPARISON_DIR", "LUMINA_TOPION_LEVELS_FILE",
}
expected = set(path_keys) | {"LUMINA_SUPER_LEVELS"}
if mode == "B-1":
    expected |= {"LUMINA_A210_SPRODUCER_CAPTURE", "LUMINA_A210_INDEPENDENT_CAPTURE"}
actual = {key for key in set(base) | set(run) if base.get(key) != run.get(key)}
if actual != expected:
    raise SystemExit(f"delta keys actual={sorted(actual)} expected={sorted(expected)}")
if run.get("LUMINA_SUPER_LEVELS") != "1" or "LUMINA_SUPER_LEVELS" in base:
    raise SystemExit("super-level promotion delta invalid")
for key in path_keys:
    if base_root not in base[key] or run[key] != base[key].replace(base_root, dest_root):
        raise SystemExit(f"path relocation invalid: {key}")
for key in ("LUMINA_A210_SPRODUCER_CAPTURE", "LUMINA_A210_INDEPENDENT_CAPTURE"):
    wanted = "0" if mode == "B-1" else "1"
    if base.get(key) != "1" or run.get(key) != wanted:
        raise SystemExit(f"capture delta invalid: {key}")
for key in sorted(actual):
    print(f"{key}\t{base.get(key, '<UNSET>')}\t{run.get(key, '<UNSET>')}")
PY
  observed="$(wc -l < "$diff_file")"
  if [[ "$observed" -ne "$expected_count" ]]; then
    rm -rf -- "$scratch"
    die "EXPORT_DELTA_NOT_EXACT mode=$mode actual=$observed expected=$expected_count"
  fi
  expected_export_delta_report="$(<"$diff_file")"
  rm -rf -- "$scratch"
}

rewrite_job_for_offline_reader() {
  local source="$1" target="$2"
  python3 -B - "$source" "$target" <<'PY'
import pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")
needle = ('[[ "${LUMINA_SUPER_CUTOFF:-}" == 100 ]] || \\\n'
          '  die "super-level topology cutoff mismatch"\n')
if text.count(needle) != 1:
    raise SystemExit("SUPER_CUTOFF gate anchor is not unique")
addition = (needle + '[[ "${LUMINA_SUPER_LEVELS:-}" == 1 ]] || \\\n'
            '  die "super-level promotion declaration mismatch"\n')
text = text.replace(needle, addition)
start_token = '\nif [[ "$diagnostic_mode" == A210_Z2A_PROBE ]]; then\n'
landing = text.rindex('\nset +e\n')
start = text.rfind(start_token, 0, landing)
if start < 0:
    raise SystemExit("Z2A landing block not found")
old = text[start:landing]
if old.count('python3 "$z2a_analyzer"') != 3:
    raise SystemExit("Z2A inline analyzer arms changed")
replacement = r'''
if [[ "$diagnostic_mode" == A210_Z2A_PROBE ]]; then
  z2a_complete_summaries="$(grep -c \
    "\\[A2-10\\]\\[LINE-SATURATION-SUMMARY\\].*phase=REQUESTED_TE.*target_ion=$z2a_target_ion.*complete=1" \
    "$run_root/stderr.log" || true)"
  [[ "$z2a_complete_summaries" -eq 2 ]] || \
    die "SLB landing lacks two complete requested-Te summaries"
  [[ "$(grep -c '\[A2-10\]\[LINE-SATURATION-BLOCKED\]' \
       "$run_root/stderr.log" || true)" -eq 0 ]] || \
    die "SLB landing emitted a line-saturation BLOCKED record"
  [[ "$(grep -c '\[A2-10\]\[LINE-SATURATION-ISOLATED\]' \
       "$run_root/stderr.log" || true)" -eq 0 ]] || \
    die "SLB landing emitted a line-saturation ISOLATED record"
  for landed in lumina_ion_pops.csv lumina_plasma_state.csv lumina_spectrum_formal.csv; do
    [[ -f "$work_dir/$landed" && ! -L "$work_dir/$landed" ]] || \
      die "SLB verdict-gate core output missing: $landed"
    [[ ! -e "$run_root/$landed" && ! -L "$run_root/$landed" ]] || \
      die "SLB verdict-gate landing path is not new and safe: $landed"
    cp --reflink=auto -- "$work_dir/$landed" "$run_root/$landed"
  done
  [[ ! -e "$run_root/SUPERLEVEL_B_LANDING.txt" && \
     ! -L "$run_root/SUPERLEVEL_B_LANDING.txt" ]] || \
    die "SLB landing marker path is not new and safe"
  printf 'SUPERLEVEL_B_LANDED status=MODEL_COMPLETE_OFFLINE_READER_REQUIRED job_id=%s run_root=%s\n' \
    "$SLURM_JOB_ID" "$run_root" | tee "$run_root/SUPERLEVEL_B_LANDING.txt"
  exit 0
fi
'''
text = text[:start] + replacement + text[landing:]
if 'python3 "$z2a_analyzer"' in text:
    raise SystemExit("inline Z2A analyzer remains in staged job")
target.write_text(text, encoding="utf-8")
PY
  chmod 0750 "$target"
  verify_staged_job_for_offline_reader "$target"
}

verify_staged_job_for_offline_reader() {
  local target="$1"
  [[ "$(grep -cF 'super-level promotion declaration mismatch' "$target" || true)" -eq 1 ]] || \
    die "STAGED_JOB_SUPERLEVEL_GATE_MISSING"
  [[ "$(grep -cF 'python3 "$z2a_analyzer"' "$target" || true)" -eq 0 ]] || \
    die "STAGED_JOB_INLINE_ANALYZER_PRESENT"
  [[ "$(grep -cF 'SUPERLEVEL_B_LANDED' "$target" || true)" -eq 1 ]] || \
    die "STAGED_JOB_OFFLINE_LANDING_MISSING"
}

stager_selftest() {
  local scratch base copied base_env run_env mode repo staged_job
  scratch="$(mktemp -d)"
  trap 'rm -rf -- "$scratch"' EXIT
  base="$scratch/base"; copied="$scratch/copied"
  mkdir -p "$base/input" "$copied/input"
  printf 'payload\n' > "$base/input/payload"
  cp -R --reflink=auto --preserve=mode,timestamps -- "$base/input/." "$copied/input/"
  verify_no_shared_file_inode "$base/input" "$copied/input" || \
    die "SELFTEST_CLEAN_COPY_SHARED detail=$shared_hardlink_detail"
  printf 'NC-G1 inject=NONE status=PASS reason=SEALED_BASE_SHARED_HARDLINKS actual=0\n'
  rm -f -- "$copied/input/payload"
  ln "$base/input/payload" "$copied/input/payload"
  if verify_no_shared_file_inode "$base/input" "$copied/input"; then
    die "SELFTEST_ACCEPTED_HARDLINK"
  fi
  printf 'NC-G1 inject=SHARED_INODE status=FAIL reason=SEALED_BASE_SHARED_HARDLINKS detail=%s\n' \
    "$shared_hardlink_detail"
  rm -f -- "$copied/input/payload"
  cp -- "$base/input/payload" "$copied/input/payload"
  verify_no_shared_file_inode "$base/input" "$copied/input" || \
    die "SELFTEST_RESTORE_COPY_SHARED"
  printf 'NC-G1 remove=RESTORE_COPY status=PASS reason=SEALED_BASE_SHARED_HARDLINKS actual=0\n'

  base_env="$scratch/base.exports"; run_env="$scratch/run.exports"
  {
    for key in $PATH_KEYS; do
      printf 'declare -x %s="/sealed/run/value/%s"\n' "$key" "$key"
    done
    printf '%s\n' \
      'declare -x LUMINA_A210_SPRODUCER_CAPTURE="1"' \
      'declare -x LUMINA_A210_INDEPENDENT_CAPTURE="1"' \
      'declare -x LUMINA_SUPER_CUTOFF="100"'
  } > "$base_env"
  for mode in B-1 B-2; do
    rewrite_exports "$base_env" /sealed/run "/scratch/$mode" "$mode" > "$run_env"
    verify_export_delta "$base_env" "$run_env" /sealed/run "/scratch/$mode" "$mode"
    printf 'NC-G2-%s inject=NONE status=PASS reason=EXPORT_DELTA_EXACT\n' "$mode"
    printf 'declare -x LUMINA_UNREGISTERED_EXTRA="1"\n' >> "$run_env"
    if (verify_export_delta "$base_env" "$run_env" /sealed/run "/scratch/$mode" "$mode") \
         >/dev/null 2>&1; then
      die "SELFTEST_ACCEPTED_EXTRA_EXPORT mode=$mode"
    fi
    printf 'NC-G2-%s inject=EXTRA_EXPORT status=FAIL reason=EXPORT_DELTA_NOT_EXACT\n' "$mode"
    rewrite_exports "$base_env" /sealed/run "/scratch/$mode" "$mode" > "$run_env"
    verify_export_delta "$base_env" "$run_env" /sealed/run "/scratch/$mode" "$mode"
    printf 'NC-G2-%s remove=RESTORE_EXPORTS status=PASS reason=EXPORT_DELTA_EXACT\n' "$mode"
  done
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  staged_job="$scratch/job.slurm"
  rewrite_job_for_offline_reader "$repo/scripts/run_det_convergence_2026-08-08.slurm" "$staged_job"
  verify_staged_job_for_offline_reader "$staged_job"
  printf 'NC-G3 inject=NONE status=PASS reason=STAGED_JOB_OFFLINE_READER_CONTRACT\n'
  printf '%s\n' 'python3 "$z2a_analyzer" --forbidden-inline-reader' >> "$staged_job"
  set +e
  nc_g3_output="$(verify_staged_job_for_offline_reader "$staged_job" 2>&1)"
  nc_g3_rc=$?
  set -e
  [[ "$nc_g3_rc" -eq 70 && "$nc_g3_output" == *'STAGED_JOB_INLINE_ANALYZER_PRESENT'* ]] || \
    die "SELFTEST_NC_G3_NOT_REJECTED rc=$nc_g3_rc output=$nc_g3_output"
  printf 'NC-G3 inject=INLINE_READER_CALL status=FAIL reason=STAGED_JOB_INLINE_ANALYZER_PRESENT\n'
  rewrite_job_for_offline_reader "$repo/scripts/run_det_convergence_2026-08-08.slurm" "$staged_job"
  verify_staged_job_for_offline_reader "$staged_job"
  printf 'NC-G3 remove=RESTORE_STAGED_JOB status=PASS reason=STAGED_JOB_OFFLINE_READER_CONTRACT\n'
  printf 'SUPERLEVEL_B_STAGER_SELFTEST_PASS controls=NC-G1,NC-G2-B-1,NC-G2-B-2,NC-G3\n'
  rm -rf -- "$scratch"
  trap - EXIT
}

if [[ $# -eq 1 && "$1" == --selftest ]]; then
  stager_selftest
  exit 0
fi

[[ $# -ge 3 && "$1" == --stage ]] || \
  die "USAGE usage='$0 --stage B-1|B-2|B-3 NEW_RUN_ROOT [SLB_F_D_AUTHORITY_FILE]'"
mode="$2"
dest="$3"
authority="${4:-}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
case "$mode" in
  B-1) suffix=census_ion1; job_name=LUMINA_SLB_I ;;
  B-2) suffix=probe_ion1; job_name=LUMINA_SLB_II ;;
  B-3)
    suffix=probe_repeat_ion1; job_name=LUMINA_SLB_III
    [[ "$authority" = /* && -f "$authority" && ! -L "$authority" ]] || \
      die "B3_AUTHORITY_MISSING"
    [[ "$(grep -cFx 'branch=SLB-F-D' "$authority" || true)" -eq 1 && \
       "$(grep -cFx 'B3_AUTHORIZED=1' "$authority" || true)" -eq 1 ]] || \
      die "B3_NOT_AUTHORIZED_BY_SLB_F_D"
    ;;
  *) die "MODE_INVALID mode=$mode" ;;
esac
[[ "$dest" == "$SLB_PARENT"/slb_*Z_"$suffix" ]] || \
  die "RUN_ROOT_NAMING_MISMATCH mode=$mode path=$dest"
require_safe_absolute_new_path "$dest"

for source in "$repo/scripts/stage_superlevel_b.sh" "$repo/scripts/analyze_superlevel_b.py"; do
  [[ -f "$source" && ! -L "$source" ]] || die "IMPLEMENTATION_SOURCE_UNSAFE path=$source"
done
verify_run_seal "$CENSUS_ROOT" "$CENSUS_MANIFEST" 136 \
  A210_ZO_CENSUS_RUN_SEAL_V1 ZO_RUN_SEAL.txt
verify_run_seal "$RII_ROOT" "$RII_MANIFEST" 135 \
  A210_Z2A_RII_RUN_SEAL_V1 RII_RUN_SEAL.txt
registered_input_digests

mkdir -p "$dest"
cp -R --reflink=auto --preserve=mode,timestamps -- "$RII_ROOT/input" "$dest/input"
verify_no_shared_file_inode "$RII_ROOT/input" "$dest/input" || \
  die "SEALED_BASE_SHARED_HARDLINKS detail=$shared_hardlink_detail count=$shared_hardlinks"
if find "$dest/input" -type l -print -quit | grep -q .; then
  die "COPIED_INPUT_CONTAINS_SYMLINK"
fi

for spec in "$BINARY_REL:$registered_binary" "$LEVELS_REL:$registered_levels" \
            "$LINES_REL:$registered_lines"; do
  rel="${spec%%:*}"; expected="${spec#*:}"
  [[ "$(sha256sum "$dest/$rel" | cut -d' ' -f1)" == "$expected" ]] || \
    die "COPIED_REGISTERED_DIGEST_MISMATCH member=$rel"
done

new_env="$(mktemp)"
new_job="$(mktemp)"
trap 'rm -f -- "$new_env" "$new_job"' EXIT
rewrite_exports "$RII_ROOT/input/resolved_lumina.exports" "$RII_ROOT" "$dest" "$mode" > "$new_env"
verify_export_delta "$RII_ROOT/input/resolved_lumina.exports" "$new_env" \
  "$RII_ROOT" "$dest" "$mode"
atomic_install_file "$new_env" "$dest/input/resolved_lumina.exports" 0640
rewrite_job_for_offline_reader "$RII_ROOT/input/job.slurm" "$new_job"
atomic_install_file "$new_job" "$dest/input/job.slurm" 0750
rm -f -- "$new_env" "$new_job"
trap - EXIT

atomic_install_file "$repo/scripts/stage_superlevel_b.sh" \
  "$dest/input/stage_superlevel_b.sh" 0750
atomic_install_file "$repo/scripts/analyze_superlevel_b.py" \
  "$dest/input/analyze_superlevel_b.py" 0750
atomic_text_file "$dest/input/outer_iterations.txt" 0640 2
atomic_text_file "$dest/input/diagnostic_mode.txt" 0640 A210_Z2A_PROBE
atomic_text_file "$dest/input/z2a_target_ion.txt" 0640 1
atomic_text_file "$dest/input/binary.sha256" 0640 "$registered_binary"
atomic_text_file "$dest/input/superlevel_b_mode.txt" 0640 "$mode"
atomic_text_file "$dest/input/SUPERLEVEL_B_EXPECTED_EXPORT_DELTA.txt" 0640 \
  'schema=SUPERLEVEL_B_EXPORT_DELTA_V1' \
  "mode=$mode" \
  "baseline=$RII_ROOT" \
  "physical_effective_delta=LUMINA_SUPER_LEVELS:<UNSET>->1" \
  "$expected_export_delta_report"
atomic_text_file "$dest/input/git_head.txt" 0640 "$(git -C "$repo" rev-parse HEAD)"
atomic_text_file "$dest/input/git_status.txt" 0640 "$(git -C "$repo" status --short)"

capture_value=1
[[ "$mode" != B-1 ]] || capture_value=0
atomic_text_file "$dest/OWNER.txt" 0640 \
  'project=LUMINA' \
  "repo=$repo" \
  "purpose=DET-SUPERLEVEL-B mode=$mode declared_K100_f_consistency" \
  'DO_NOT_CANCEL=1'
atomic_text_file "$dest/SUPERLEVEL_B_STAGE_SEAL.txt" 0640 \
  'schema=SUPERLEVEL_B_STAGE_V1' \
  "mode=$mode" \
  "sealed_base=$RII_ROOT" \
  "sealed_census=$CENSUS_ROOT" \
  'outer_iterations=2' \
  'target_ion=1' \
  'super_cutoff=100' \
  'super_levels=1' \
  "producer_capture=$capture_value" \
  "independent_capture=$capture_value" \
  "registered_binary_sha256=$registered_binary" \
  "registered_levels_sha256=$registered_levels" \
  "registered_line_list_sha256=$registered_lines" \
  'sealed_input_shared_hardlinks=0' \
  'inline_reader_calls=0' \
  'physical_values_modified=0' \
  'floor=0' 'cap=0' 'clamp=0' 'jitter=0' 'repair=0'
atomic_text_file "$dest/submit_superlevel_b.sh" 0750 \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'run_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"' \
  '[[ ! -L "$run_root" && -f "$run_root/READY" && ! -L "$run_root/READY" ]] || { echo "SUPERLEVEL_B_SUBMIT_FATAL unsafe run root" >&2; exit 70; }' \
  '[[ ! -e "$run_root/.run_start" && ! -L "$run_root/.run_start" ]] || { echo "SUPERLEVEL_B_SUBMIT_FATAL already submitted or unsafe start marker" >&2; exit 70; }' \
  'touch "$run_root/.run_start"' \
  "exec sbatch --partition=a100 --gres=gpu:2 --nodes=1 --mem=64G --time=08:00:00 --job-name=$job_name \"\$run_root/input/job.slurm\" \"\$run_root\""
atomic_text_file "$dest/READY" 0640 READY

printf 'SUPERLEVEL_B_STAGE_OK mode=%s run_root=%s env_delta=%s shared_hardlinks=0 submit=NOT_RUN\n' \
  "$mode" "$dest" "$([[ "$mode" == B-1 ]] && printf 9 || printf 7)"
