#!/usr/bin/env bash
# Slurm runner for the single registered DET-CMFGEN-TWIN-MAP observation.
#SBATCH --job-name=cmfgen_twin_map
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail
umask 027

readonly GPU_PROFILE=h200x1

die() {
  printf 'CMFGEN_TWIN_MAP_RUN_FAIL reason=%s\n' "$*" >&2
  exit 70
}

sha256() { sha256sum -- "$1" | awk '{print $1}'; }

run_model() {
  local run_root="$1"
  local input_dir work_dir model_dir binary exports profile
  local expected_sigma actual_sigma actual_binary profile_file_sha te_manifest_sha start elapsed model_rc=125
  local gpu_csv_tmp token row name memory uuid kind first_kind='' first_uuid=''
  local expected_partition expected_gpu_count expected_min_memory expected_kinds ordinal
  local -a visible_tokens

  [[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs ]] || die "UNSAFE_RUN_ROOT path=$run_root"
  [[ -d "$run_root" && ! -L "$run_root" ]] || die "RUN_ROOT_NOT_DIRECTORY"
  input_dir="$run_root/input"; work_dir="$run_root/work"; model_dir="$input_dir/model"
  binary="$input_dir/lumina_cuda"; exports="$input_dir/resolved_lumina.exports"
  profile="$input_dir/fixed_te_profile.txt"
  for path in "$binary" "$exports" "$profile" "$input_dir/binary.sha256" \
    "$input_dir/sigma.sha256" "$input_dir/deck.sha256" "$input_dir/topion.sha256"; do
    [[ -f "$path" && ! -L "$path" ]] || die "STAGED_INPUT_UNSAFE path=$path"
  done
  [[ -d "$model_dir" && ! -L "$model_dir" && -d "$input_dir/global_atomic" && ! -L "$input_dir/global_atomic" ]] || \
    die "STAGED_TREE_INCOMPLETE"
  [[ -x "$binary" ]] || die "BINARY_NOT_EXECUTABLE"
  [[ ! -e "$run_root/RUN_ATTEMPTED" && ! -L "$run_root/RUN_ATTEMPTED" ]] || die "RUN_ALREADY_ATTEMPTED"
  ( set -o noclobber; printf 'job_id=%s\n' "${SLURM_JOB_ID:-UNSET}" > "$run_root/RUN_ATTEMPTED" ) 2>/dev/null || \
    die "RUN_ALREADY_ATTEMPTED"

  case "$GPU_PROFILE" in
    h200x1) expected_partition=h200; expected_gpu_count=1; expected_min_memory=140000; expected_kinds=H200 ;;
    h100-a100x2) expected_partition=h100,a100; expected_gpu_count=2; expected_min_memory=80000; expected_kinds=H100,A100 ;;
    *) die "GPU_PROFILE_INVALID value=$GPU_PROFILE" ;;
  esac
  [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || die "SLURM_ALLOCATION_MISSING"
  [[ ",$expected_partition," == *,"${SLURM_JOB_PARTITION:-}",* ]] || \
    die "PARTITION_NOT_QUALIFIED profile=$GPU_PROFILE value=${SLURM_JOB_PARTITION:-UNSET}"
  [[ "${SLURM_NNODES:-}" == 1 && "${SLURM_NTASKS:-}" == 1 && "${SLURM_CPUS_PER_TASK:-}" == 32 ]] || \
    die "ALLOCATION_SHAPE_MISMATCH nodes=${SLURM_NNODES:-UNSET} tasks=${SLURM_NTASKS:-UNSET} cpus=${SLURM_CPUS_PER_TASK:-UNSET}"
  [[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || die "REQUEUED_EXECUTION_FORBIDDEN"
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die "CUDA_VISIBLE_DEVICES_MISSING"
  IFS=',' read -r -a visible_tokens <<< "$CUDA_VISIBLE_DEVICES"
  [[ "${#visible_tokens[@]}" -eq "$expected_gpu_count" ]] || \
    die "VISIBLE_GPU_MASK_INVALID value=$CUDA_VISIBLE_DEVICES"
  command -v nvidia-smi >/dev/null || die "NVIDIA_SMI_MISSING"

  gpu_csv_tmp="$run_root/.gpu_identity.csv.tmp.$$"
  [[ ! -e "$gpu_csv_tmp" && ! -L "$gpu_csv_tmp" ]] || die "GPU_TEMP_PATH_UNSAFE"
  printf 'visible_ordinal,cuda_visible_token,name,memory_total_mib,uuid,driver_version,compute_capability\n' > "$gpu_csv_tmp"
  for ((ordinal=0; ordinal<expected_gpu_count; ordinal++)); do
    token="${visible_tokens[$ordinal]}"
    [[ "$token" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "VISIBLE_GPU_TOKEN_INVALID token=$token"
    row="$(nvidia-smi --id="$token" --query-gpu=name,memory.total,uuid,driver_version,compute_cap --format=csv,noheader,nounits)" || \
      die "GPU_QUERY_FAILED token=$token"
    [[ -n "$row" && "$(printf '%s\n' "$row" | wc -l)" -eq 1 ]] || die "GPU_QUERY_AMBIGUOUS token=$token"
    IFS=',' read -r name memory uuid _driver _capability <<< "$row"
    name="${name#${name%%[![:space:]]*}}"; name="${name%${name##*[![:space:]]}}"
    memory="${memory//[[:space:]]/}"; uuid="${uuid//[[:space:]]/}"
    case "$name" in *H200*) kind=H200 ;; *H100*) kind=H100 ;; *A100*) kind=A100 ;; *) die "GPU_KIND_NOT_QUALIFIED name=$name" ;; esac
    [[ ",$expected_kinds," == *,"$kind",* ]] || die "GPU_KIND_PROFILE_MISMATCH profile=$GPU_PROFILE kind=$kind"
    [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$expected_min_memory" ]] || \
      die "GPU_MEMORY_BELOW_PROFILE_MINIMUM profile=$GPU_PROFILE token=$token memory=$memory minimum=$expected_min_memory"
    [[ "$uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]] || die "GPU_UUID_INVALID token=$token uuid=$uuid"
    if [[ "$ordinal" -eq 0 ]]; then first_kind="$kind"; first_uuid="$uuid"; else
      [[ "$kind" == "$first_kind" ]] || die "MIXED_GPU_KINDS first=$first_kind second=$kind"
      [[ "$uuid" != "$first_uuid" ]] || die "DUPLICATE_GPU_UUID uuid=$uuid"
    fi
    [[ "${SLURM_JOB_PARTITION}" == "${kind,,}" ]] || die "GPU_PARTITION_MISMATCH partition=$SLURM_JOB_PARTITION kind=$kind"
    printf '%s,%s,%s\n' "$ordinal" "$token" "$row" >> "$gpu_csv_tmp"
  done
  mv -- "$gpu_csv_tmp" "$run_root/gpu_identity.csv"

  actual_binary="$(sha256 "$binary")"
  [[ "$(<"$input_dir/binary.sha256")" == "$actual_binary" ]] || die "RUN_BINARY_IDENTITY_MISMATCH"
  python3 -B - "$input_dir/run_build_identity.json" "$actual_binary" "$run_root/STAGE_IDENTITY.txt" <<'PY' || die "RUN_BUILD_IDENTITY_MISMATCH"
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if record.get("schema") not in ("CMFGEN_TWIN_RUN_BUILD_V1", "CMFGEN_TWIN_RUN_BUILD_V1_SELFTEST"):
    raise SystemExit(1)
if record.get("binary", {}).get("sha256") != sys.argv[2]: raise SystemExit(1)
stage = dict(line.split("=", 1) for line in open(sys.argv[3], encoding="utf-8") if "=" in line)
if stage.get("run_binary_sha256", "").strip() != sys.argv[2]: raise SystemExit(1)
PY
  expected_sigma="$(<"$input_dir/sigma.sha256")"; actual_sigma="$(sha256 "$model_dir/cmfgen_sigma_bf.bin")"
  [[ "$expected_sigma" == "$actual_sigma" ]] || die "SIGMA_IDENTITY_MISMATCH"
  ( cd "$model_dir" && sha256sum -c "$input_dir/deck.sha256" ) > "$run_root/deck_verify.log" 2>&1 || die "DECK_IDENTITY_MISMATCH"
  ( cd "$input_dir/global_atomic" && sha256sum -c "$input_dir/topion.sha256" ) > "$run_root/catalog_verify.log" 2>&1 || die "CATALOG_IDENTITY_MISMATCH"
  [[ -d "$work_dir" && ! -L "$work_dir" && -z "$(find "$work_dir" -mindepth 1 -print -quit)" ]] || die "WORK_NOT_EMPTY"

  env | LC_ALL=C sort > "$run_root/slurm_environment_before_sanitize.txt"
  while IFS='=' read -r name _; do [[ "$name" == LUMINA_* ]] && unset "$name"; done < <(env)
  # shellcheck disable=SC1090
  source "$exports"
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_PLACES=cores OMP_PROC_BIND=close
  [[ "$CUDA_VISIBLE_DEVICES" == "$(IFS=,; printf '%s' "${visible_tokens[*]}")" ]] || die "GPU_MASK_CHANGED"
  [[ "${LUMINA_ENV_STRICT:-}" == 1 && "${LUMINA_PURE_CMFGEN:-}" == 1 && \
     "${LUMINA_DET_TRANSACTIONAL:-}" == 1 && "${LUMINA_PURE_CMFGEN_ITER:-}" == 2 ]] || die "STRICT_TWO_ITERATION_CONTRACT_MISMATCH"
  [[ "${LUMINA_MODEL_DIR:-}" == "$model_dir" && "${LUMINA_FIXED_TE_PROFILE:-}" == "$profile" ]] || die "STAGED_PATH_CONTRACT_MISMATCH"
  [[ "${LUMINA_ION_POP_DUMP:-}" == 1 && "${LUMINA_ION_POP_DUMP_ITER:-}" == 1 && "${LUMINA_NLTE_LEVEL_DUMP:-}" == 1 ]] || die "POPULATION_DUMPS_NOT_ARMED"
  [[ -z "${LUMINA_RADEQ_DIAG_TE_K:-}" && -z "${LUMINA_CMFGEN_THEN_MC:-}" && -z "${LUMINA_MC_COEVOLVE:-}" ]] || die "FORBIDDEN_DIAGNOSTIC_OR_MC_MODE"
  [[ "${LUMINA_CMF_FINE_MGPU_DEVICES:-}" == "$expected_gpu_count" ]] || die "GPU_OWNER_COUNT_MISMATCH profile=$GPU_PROFILE"
  profile_file_sha="$(sha256 "$profile")"

  start="$(date +%s)"
  set +e
  ( ulimit -f 1048576; cd "$work_dir" && "$binary" "$model_dir" 1 2 spectrum nlte ) > "$run_root/stdout.log" 2> "$run_root/stderr.log"
  model_rc=$?
  set -e
  elapsed=$(( $(date +%s) - start ))
  printf '%s\n' "$model_rc" > "$run_root/model.rc"
  {
    printf 'schema=CMFGEN_TWIN_MAP_RUN_V1\njob_id=%s\nnode_list=%s\npartition=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$SLURM_JOB_PARTITION"
    printf 'gpu_profile=%s\nallocated_gpus=%s\nallocated_cpus=%s\nouter_iterations=2\nregistered_runs=1\n' "$GPU_PROFILE" "$expected_gpu_count" "$SLURM_CPUS_PER_TASK"
    printf 'binary_sha256=%s\neffective_exports_sha256=%s\nfixed_te_profile_file_sha256=%s\n' "$actual_binary" "$(sha256 "$exports")" "$profile_file_sha"
    printf 'cuda_visible_devices=%s\ngpu_kind=%s\nmodel_rc=%s\nelapsed_seconds=%s\n' "$CUDA_VISIBLE_DEVICES" "$first_kind" "$model_rc" "$elapsed"
    printf 'rlimit_fsize_kib_blocks=1048576\nrlimit_fsize_bytes=1073741824\n'
    printf 'argv=%q %q 1 2 spectrum nlte\n' "$binary" "$model_dir"
    if [[ "$model_rc" -eq 0 ]]; then printf 'model_phase_state=RC_ZERO_PENDING_LOG_CHECK\n'; else printf 'model_phase_state=FAILED\n'; fi
  } > "$run_root/RUN_FOOTER.txt"
  if [[ "$model_rc" -ne 0 ]]; then printf 'terminal_state=MODEL_FAILED\n' >> "$run_root/RUN_FOOTER.txt"; die "MODEL_EXIT_NONZERO rc=$model_rc"; fi
  if grep -E '\[FATAL\]|CMFGEN_TWIN_MAP_RUN_FAIL' "$run_root/stdout.log" "$run_root/stderr.log" > "$run_root/fatal_scan.log"; then
    printf 'terminal_state=POSTCHECK_FAILED_FATAL_MARKER\n' >> "$run_root/RUN_FOOTER.txt"; die "MODEL_FATAL_MARKER"
  fi
  [[ "$(grep -c '\[DET-TRANSACTIONAL\] status=COMPLETE' "$run_root/stdout.log" || true)" -eq 1 ]] || {
    printf 'terminal_state=POSTCHECK_FAILED_DET_TERMINAL\n' >> "$run_root/RUN_FOOTER.txt"; die "DET_SUCCESS_TERMINAL_MISSING"; }
  te_manifest_sha="$(python3 -B - "$work_dir/cmfgen_final_publication.json" "$work_dir" <<'PY'
import csv, json, re, sys
from pathlib import Path
manifest_path, work = Path(sys.argv[1]), Path(sys.argv[2])
if not manifest_path.is_file() or manifest_path.is_symlink(): raise SystemExit(1)
m = json.loads(manifest_path.read_text(encoding="utf-8"))
if m.get("schema") != "CMFGEN_FINAL_PUBLICATION_V1" or m.get("complete") is not True: raise SystemExit(1)
ints = [m.get(k) for k in ("population_generation", "required_population_generation", "te_generation", "partition_te_generation")]
if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in ints) or ints[0] != ints[1] or ints[2] != ints[3]: raise SystemExit(1)
h = m.get("te_manifest_sha256")
if not isinstance(h, str) or re.fullmatch(r"[0-9a-f]{64}", h) is None: raise SystemExit(1)
expected = {"plasma": "PUBLIC_NE_TE", "saha": "SAHA_STAGE", "nlte": "NLTE_LEVEL"}
for key, authority in expected.items():
    r = m.get("files", {}).get(key, {}); name = r.get("path")
    if not isinstance(name, str) or Path(name).name != name or r.get("authority") != authority: raise SystemExit(1)
    path = work / name
    if not path.is_file() or path.is_symlink(): raise SystemExit(1)
    with path.open(newline="", encoding="utf-8") as stream: count = sum(1 for _ in csv.DictReader(stream))
    if count != r.get("rows"): raise SystemExit(1)
print(h)
PY
)" || { printf 'terminal_state=POSTCHECK_FAILED_FINAL_PUBLICATION\n' >> "$run_root/RUN_FOOTER.txt"; die "FINAL_PUBLICATION_INVALID"; }
  python3 -B - "$profile" "$work_dir/lumina_plasma_state.csv" <<'PY' || {
import csv, math, sys
profile, plasma = sys.argv[1:]
wanted = {}
with open(profile, encoding="utf-8") as stream:
    for line in stream:
        shell, temp = line.split()
        shell = int(shell)
        value = float(temp)
        if shell in wanted or not math.isfinite(value) or value <= 0: raise SystemExit(1)
        wanted[shell] = value
with open(plasma, newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != len(wanted) or {int(r["shell_id"]) for r in rows} != set(wanted): raise SystemExit(1)
for row in rows:
    value = float(row["T_e"])
    if not math.isfinite(value) or value <= 0 or value.hex() != wanted[int(row["shell_id"])].hex(): raise SystemExit(1)
PY
    printf 'terminal_state=POSTCHECK_FAILED_PUBLIC_TE_PROFILE\n' >> "$run_root/RUN_FOOTER.txt"; die "PUBLIC_TE_PROFILE_MISMATCH"; }
  if [[ "$(grep -c "\[A2-10\]\[TE_PUBLICATION\].*te_lane=FIXED_T.*te_profile_sha256=$te_manifest_sha" "$run_root/stderr.log" || true)" -ne 2 ]]; then
    printf 'terminal_state=POSTCHECK_FAILED_FIXED_TE_EVIDENCE\n' >> "$run_root/RUN_FOOTER.txt"; die "FIXED_TE_CONSUMPTION_EVIDENCE_MISSING"
  fi
  printf 'fixed_te_loader_consumed=1\ncanonical_te_manifest_sha256=%s\nterminal_state=SUCCESS\n' "$te_manifest_sha" >> "$run_root/RUN_FOOTER.txt"
  printf 'CMFGEN_TWIN_MAP_RUN_COMPLETE job_id=%s run_root=%s model_rc=0 elapsed_seconds=%s\n' "$SLURM_JOB_ID" "$run_root" "$elapsed"
}

runner_selftest() {
  local tmp script fixture_bin fixture_hash rc log case_name positive=0 negative=0
  tmp="$(mktemp -d /tmp/cmfgen-twin-runner-selftest.XXXXXX)"; trap 'rm -rf -- "$tmp"' EXIT
  script="$(realpath -e -- "$0")"
  printf '[DPROXY][BLOCKED] diagnostic_only=1\n' > "$tmp/blocked-only.log"
  if grep -E '\[FATAL\]|CMFGEN_TWIN_MAP_RUN_FAIL' "$tmp/blocked-only.log" >/dev/null; then
    die "SELFTEST_DPROXY_BLOCKED_CLASSIFIED_FATAL"
  fi
  printf 'RUNNER_SELFTEST case=dproxy_blocked_diagnostic expected=pass rc=0\n'; positive=$((positive+1))
  make_fixture() {
    local root="$1" gpu_mode="$2" model_rc="$3" consumption="$4" mgpu_devices="$5" fixture_profile_sha
    mkdir -p "$root/input/model" "$root/input/global_atomic" "$root/work" "$root/bin"
    printf 'sigma\n' > "$root/input/model/cmfgen_sigma_bf.bin"; printf 'deck\n' > "$root/input/model/deck.dat"; printf 'catalog\n' > "$root/input/global_atomic/topion_levels.csv"
    ( cd "$root/input/model" && sha256sum deck.dat ) > "$root/input/deck.sha256"
    ( cd "$root/input/global_atomic" && sha256sum topion_levels.csv ) > "$root/input/topion.sha256"
    sha256 "$root/input/model/cmfgen_sigma_bf.bin" > "$root/input/sigma.sha256"
    printf '0 1.8900E+04\n1 1.900025E+04\n' > "$root/input/fixed_te_profile.txt"
    fixture_profile_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    printf '%s\n' \
      'declare -x LUMINA_ENV_STRICT="1"' 'declare -x LUMINA_PURE_CMFGEN="1"' 'declare -x LUMINA_DET_TRANSACTIONAL="1"' \
      'declare -x LUMINA_PURE_CMFGEN_ITER="2"' "declare -x LUMINA_MODEL_DIR=\"$root/input/model\"" \
      "declare -x LUMINA_FIXED_TE_PROFILE=\"$root/input/fixed_te_profile.txt\"" "declare -x LUMINA_CMF_FINE_MGPU_DEVICES=\"$mgpu_devices\"" \
      'declare -x LUMINA_ION_POP_DUMP="1"' 'declare -x LUMINA_ION_POP_DUMP_ITER="1"' 'declare -x LUMINA_NLTE_LEVEL_DUMP="1"' > "$root/input/resolved_lumina.exports"
    printf '#!/usr/bin/env bash\n[[ "$(ulimit -f)" == 1048576 ]] || exit 91\nprintf "1048576\\n" > rlimit_fsize.inherited\nfor ((i=0;i<%s;i++)); do printf "[A2-10][TE_PUBLICATION] te_lane=FIXED_T te_profile_sha256=%%s\\n" "%s" >&2; done\nprintf "[DET-TRANSACTIONAL] status=COMPLETE\\n"\nprintf "shell_id,T_e,n_e\\n0,18900,1\\n1,19000.25,2\\n" > lumina_plasma_state.csv\nprintf "shell_id,Z,stage,n_ion\\n0,26,1,1\\n1,26,1,2\\n" > lumina_ion_pops.csv\nprintf "Z,ion,shell,level_idx,global_idx,n_pop,T_e,n_ion_total,population_generation\\n26,1,0,0,0,1,18900,1,7\\n26,1,1,0,0,2,19000.25,2,7\\n" > nlte_levels_final.csv\nprintf %%s "{\\"schema\\":\\"CMFGEN_FINAL_PUBLICATION_V1\\",\\"complete\\":true,\\"population_generation\\":7,\\"required_population_generation\\":7,\\"te_generation\\":11,\\"partition_te_generation\\":11,\\"te_manifest_sha256\\":\\"%s\\",\\"files\\":{\\"plasma\\":{\\"path\\":\\"lumina_plasma_state.csv\\",\\"authority\\":\\"PUBLIC_NE_TE\\",\\"rows\\":2},\\"saha\\":{\\"path\\":\\"lumina_ion_pops.csv\\",\\"authority\\":\\"SAHA_STAGE\\",\\"rows\\":2},\\"nlte\\":{\\"path\\":\\"nlte_levels_final.csv\\",\\"authority\\":\\"NLTE_LEVEL\\",\\"rows\\":2}}}" > cmfgen_final_publication.json\nexit %s\n' "$consumption" "$fixture_profile_sha" "$fixture_profile_sha" "$model_rc" > "$root/input/lumina_cuda"; chmod +x "$root/input/lumina_cuda"
    sha256 "$root/input/lumina_cuda" > "$root/input/binary.sha256"
    printf '{"schema":"CMFGEN_TWIN_RUN_BUILD_V1_SELFTEST","binary":{"sha256":"%s"}}\n' "$(<"$root/input/binary.sha256")" > "$root/input/run_build_identity.json"
    printf 'schema=CMFGEN_TWIN_MAP_STAGE_V2\nrun_binary_sha256=%s\n' "$(<"$root/input/binary.sha256")" > "$root/STAGE_IDENTITY.txt"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$root/bin/module"
    printf '#!/usr/bin/env bash\ntoken=""\nfor a in "$@"; do case "$a" in --id=*) token="${a#--id=}";; esac; done\ncase "%s:$token" in good:GPU-a) echo "NVIDIA H100 80GB HBM3, 81559, GPU-aaaa, 555.1, 9.0";; good:GPU-b) echo "NVIDIA H100 80GB HBM3, 81559, GPU-bbbb, 555.1, 9.0";; h200:GPU-a) echo "NVIDIA H200, 143771, GPU-aaaa, 555.1, 9.0";; h200:GPU-b) echo "NVIDIA H200, 143771, GPU-bbbb, 555.1, 9.0";; h200_lowmem:*) echo "NVIDIA H200, 139999, GPU-aaaa$token, 555.1, 9.0";; lowmem:*) echo "NVIDIA H100, 79999, GPU-aaaa$token, 555.1, 9.0";; a100_lowmem:*) echo "NVIDIA A100-SXM4-40GB, 40960, GPU-aaaa$token, 555.1, 8.0";; duplicate:*) echo "NVIDIA H100 80GB HBM3, 81559, GPU-aaaa, 555.1, 9.0";; a100:*) echo "NVIDIA A100-SXM4-80GB, 81920, GPU-aaaa$token, 555.1, 8.0";; mixed:GPU-a) echo "NVIDIA H100 80GB HBM3, 81559, GPU-aaaa, 555.1, 9.0";; mixed:GPU-b) echo "NVIDIA H200, 143771, GPU-bbbb, 555.1, 9.0";; esac\n' "$gpu_mode" > "$root/bin/nvidia-smi"; chmod +x "$root/bin/nvidia-smi"
  }
  run_case() {
    local name="$1" gpu="$2" model_rc="$3" consumption="$4" mask="$5" partition="$6" profile="$7" mgpu_devices="$8" expect="$9" root
    root="$tmp/$name"
    make_fixture "$root" "$gpu" "$model_rc" "$consumption" "$mgpu_devices"
    if [[ "$name" == public_t_mismatch ]]; then sed -i 's/0,18900,1/0,18901,1/' "$root/input/lumina_cuda"; fi
    if [[ "$name" == duplicate_profile_shell ]]; then printf '0 1.8900E+04\n' >> "$root/input/fixed_te_profile.txt"; fi
    if [[ "$name" == manifest_hash_mismatch ]]; then sed -i '0,/aaaaaaaa/{s/aaaaaaaa/baaaaaaa/}' "$root/input/lumina_cuda"; fi
    sha256 "$root/input/lumina_cuda" > "$root/input/binary.sha256"
    printf '{"schema":"CMFGEN_TWIN_RUN_BUILD_V1_SELFTEST","binary":{"sha256":"%s"}}\n' "$(<"$root/input/binary.sha256")" > "$root/input/run_build_identity.json"
    printf 'schema=CMFGEN_TWIN_MAP_STAGE_V2\nrun_binary_sha256=%s\n' "$(<"$root/input/binary.sha256")" > "$root/STAGE_IDENTITY.txt"
    fixture_hash="$(<"$root/input/binary.sha256")"
    cp -- "$script" "$root/test_runner.sh"
    if [[ "$profile" == h100-a100x2 ]]; then
      sed -i 's/^#SBATCH --partition=h200$/#SBATCH --partition=h100,a100/; s/^#SBATCH --gres=gpu:1$/#SBATCH --gres=gpu:2/; s/^readonly GPU_PROFILE=h200x1$/readonly GPU_PROFILE=h100-a100x2/' "$root/test_runner.sh"
    fi
    set +e
    PATH="$root/bin:$PATH" CUDA_VISIBLE_DEVICES="$mask" SLURM_JOB_ID=123 SLURM_JOB_NODELIST=gpu001 SLURM_JOB_PARTITION="$partition" SLURM_NNODES=1 SLURM_NTASKS=1 SLURM_CPUS_PER_TASK=32 SLURM_RESTART_COUNT=0 \
      bash "$root/test_runner.sh" "$root" > "$root/test.log" 2>&1
    rc=$?; set -e
    if [[ "$expect" == pass ]]; then
      if [[ "$rc" -ne 0 ]]; then sed -n '1,120p' "$root/test.log" >&2; die "SELFTEST_POSITIVE_FAILED case=$name rc=$rc"; fi
      [[ "$(<"$root/work/rlimit_fsize.inherited")" == 1048576 ]] || die "SELFTEST_RLIMIT_NOT_INHERITED"
      positive=$((positive+1))
    else
      [[ "$rc" -ne 0 ]] || die "SELFTEST_NEGATIVE_ACCEPTED case=$name"
      negative=$((negative+1))
    fi
    printf 'RUNNER_SELFTEST case=%s expected=%s rc=%s\n' "$name" "$expect" "$rc"
  }
  run_case h200x1_ok h200 0 2 GPU-a h200 h200x1 1 pass
  run_case h100x2_ok good 0 2 GPU-a,GPU-b h100 h100-a100x2 2 pass
  run_case a100x2_ok a100 0 2 GPU-a,GPU-b a100 h100-a100x2 2 pass
  run_case a100_on_h100_reject a100 0 2 GPU-a,GPU-b h100 h100-a100x2 2 fail
  run_case h100_low_memory lowmem 0 2 GPU-a,GPU-b h100 h100-a100x2 2 fail
  run_case h200_low_memory h200_lowmem 0 2 GPU-a h200 h200x1 1 fail
  run_case a100_40gb_low_memory a100_lowmem 0 2 GPU-a,GPU-b a100 h100-a100x2 2 fail
  run_case duplicate_uuid duplicate 0 2 GPU-a,GPU-b h100 h100-a100x2 2 fail
  run_case mixed_kind mixed 0 2 GPU-a,GPU-b h100 h100-a100x2 2 fail
  run_case h200_two_gpus h200 0 2 GPU-a,GPU-b h200 h200x1 1 fail
  run_case h100_one_gpu good 0 2 GPU-a h100 h100-a100x2 2 fail
  run_case wrong_partition good 0 2 GPU-a,GPU-b h200 h100-a100x2 2 fail
  run_case exports_count_mismatch h200 0 2 GPU-a h200 h200x1 2 fail
  run_case model_failure good 9 2 GPU-a,GPU-b h100 h100-a100x2 2 fail
  run_case missing_consumption good 0 1 GPU-a,GPU-b h100 h100-a100x2 2 fail
  run_case public_t_mismatch good 0 2 GPU-a,GPU-b h100 h100-a100x2 2 fail
  run_case duplicate_profile_shell good 0 2 GPU-a,GPU-b h100 h100-a100x2 2 fail
  run_case manifest_hash_mismatch good 0 2 GPU-a,GPU-b h100 h100-a100x2 2 fail
  set +e
  PATH="$tmp/h100x2_ok/bin:$PATH" CUDA_VISIBLE_DEVICES=GPU-a,GPU-b SLURM_JOB_ID=124 SLURM_JOB_NODELIST=gpu001 SLURM_JOB_PARTITION=h100 SLURM_NNODES=1 SLURM_NTASKS=1 SLURM_CPUS_PER_TASK=32 SLURM_RESTART_COUNT=0 \
    bash "$tmp/h100x2_ok/test_runner.sh" "$tmp/h100x2_ok" >/dev/null 2>&1
  rc=$?; set -e; [[ "$rc" -ne 0 ]] || die "SELFTEST_RERUN_ACCEPTED"; negative=$((negative+1)); printf 'RUNNER_SELFTEST case=rerun_reject expected=fail rc=%s\n' "$rc"
  printf 'CMFGEN_TWIN_MAP_RUNNER_SELFTEST_PASS positive=%d negative=%d\n' "$positive" "$negative"
  rm -rf -- "$tmp"; trap - EXIT
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ $# -eq 1 && "$1" == --selftest ]]; then runner_selftest; exit 0; fi
  [[ $# -eq 1 ]] || die "USAGE expected='ABSOLUTE_RUN_ROOT'"
  run_model "$1"
fi
