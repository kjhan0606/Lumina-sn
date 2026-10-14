#!/usr/bin/env bash
# Independent fail-closed stager for DET-CMFGEN-TWIN-MAP.
set -euo pipefail
umask 027

readonly EXPECTED_BINARY_SHA256=ca721b07cb4fef632b1a0a0b2d3c7b11cdaac0e95f74e6cc56b131a7c5449059
readonly STORAGE_CAP_BYTES=$((16 * 1024 * 1024 * 1024))
readonly DEFAULT_GPU_PROFILE=h200x1

die() {
  printf 'CMFGEN_TWIN_MAP_STAGE_FAIL reason=%s\n' "$*" >&2
  exit 70
}

validate_gpu_profile() {
  [[ "$1" == h200x1 || "$1" == h100-a100x2 ]]
}

is_below() {
  local child="$1" parent="$2"
  [[ "$child" == "$parent" || "$child" == "$parent"/* ]]
}

canonical_no_symlink() {
  local raw="$1" mode="$2" resolved cursor
  [[ "$raw" = /* && "$raw" != / && "$(realpath -m -- "$raw")" == "$raw" ]] || return 1
  if [[ "$mode" == existing ]]; then
    [[ -e "$raw" ]] || return 1
    resolved="$(realpath -e -- "$raw")" || return 1
    [[ "$resolved" == "$raw" ]] || return 1
    cursor="$raw"
  else
    [[ ! -e "$raw" && ! -L "$raw" && -d "$(dirname "$raw")" ]] || return 1
    resolved="$(realpath -e -- "$(dirname "$raw")")/$(basename "$raw")" || return 1
    [[ "$resolved" == "$raw" ]] || return 1
    cursor="$(dirname "$raw")"
  fi
  while [[ "$cursor" != / ]]; do
    [[ ! -L "$cursor" ]] || return 1
    cursor="$(dirname "$cursor")"
  done
}

require_safe_base() {
  local base="$1"
  canonical_no_symlink "$base" existing && [[ -d "$base" ]] || return 1
  [[ -d "$base/input" && ! -L "$base/input" ]] || return 1
  ! find "$base" -type l -print -quit | grep -q .
}

require_new_root() {
  local target="$1" base="$2" evidence="$3"
  canonical_no_symlink "$target" new || return 1
  ! is_below "$target" "$base" || return 1
  ! is_below "$target" "$evidence" || return 1
}

require_evidence_root() {
  local evidence="$1" base="$2" run="$3" rel
  canonical_no_symlink "$evidence" existing && [[ -d "$evidence" ]] || return 1
  ! is_below "$evidence" "$base" && ! is_below "$base" "$evidence" || return 1
  ! is_below "$evidence" "$run" && ! is_below "$run" "$evidence" || return 1
  for rel in inputs offline run result; do
    canonical_no_symlink "$evidence/$rel" existing && [[ -d "$evidence/$rel" ]] || return 1
  done
}

manifest_tree() {
  local root="$1" output="$2"
  python3 -B - "$root" "$output" <<'PY'
import hashlib, os, stat, sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
try:
    for component in os.path.dirname(str(output)).split(os.sep)[1:]:
        next_parent = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        os.close(parent); parent = next_parent
    fd = os.open(output.name, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=parent)
finally:
    os.close(parent)
try:
    meta = os.fstat(fd)
    if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1:
        raise SystemExit(1)
    lines = []
    for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
        rel = path.relative_to(root)
        if path.is_symlink():
            lines.append(f"SYMLINK {os.readlink(path)}  ./{rel}\n")
        elif path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
            lines.append(f"{h.hexdigest()}  ./{rel}\n")
    os.ftruncate(fd, 0); os.write(fd, "".join(lines).encode())
finally:
    os.close(fd)
PY
}

publish_new() {
  python3 -B - "$1" "$2" <<'PY'
import os, stat, sys
source, target = sys.argv[1:]
source_dir, source_name = os.path.dirname(source), os.path.basename(source)
target_dir, target_name = os.path.dirname(target), os.path.basename(target)
def directory_fd(path):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.split(os.sep)[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = next_fd
        return fd
    except OSError:
        os.close(fd); raise
source_fd = directory_fd(source_dir)
target_fd = directory_fd(target_dir)
try:
    meta = os.stat(source_name, dir_fd=source_fd, follow_symlinks=False)
    if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1: raise SystemExit(1)
    os.link(source_name, target_name, src_dir_fd=source_fd, dst_dir_fd=target_fd, follow_symlinks=False)
except FileExistsError:
    raise SystemExit(1)
try:
    published = os.stat(target_name, dir_fd=target_fd, follow_symlinks=False)
    if not stat.S_ISREG(published.st_mode) or published.st_nlink != 2: raise SystemExit(1)
    os.unlink(source_name, dir_fd=source_fd)
    final = os.stat(target_name, dir_fd=target_fd, follow_symlinks=False)
    if final.st_nlink != 1: raise SystemExit(1)
finally:
    os.close(source_fd); os.close(target_fd)
PY
}

replace_private() {
  python3 -B - "$1" "$2" <<'PY'
import os, stat, sys
source, target = sys.argv[1:]
def parent_fd(path):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in os.path.dirname(path).split(os.sep)[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = next_fd
        return fd
    except OSError:
        os.close(fd); raise
source_parent = parent_fd(source)
target_parent = parent_fd(target)
try:
    sm = os.stat(os.path.basename(source), dir_fd=source_parent, follow_symlinks=False)
    try:
        tm = os.stat(os.path.basename(target), dir_fd=target_parent, follow_symlinks=False)
    except FileNotFoundError:
        tm = None
    if not stat.S_ISREG(sm.st_mode) or sm.st_nlink != 1: raise SystemExit(1)
    if tm is not None and (not stat.S_ISREG(tm.st_mode) or tm.st_nlink != 1): raise SystemExit(1)
    os.replace(os.path.basename(source), os.path.basename(target), src_dir_fd=source_parent, dst_dir_fd=target_parent)
finally:
    os.close(source_parent); os.close(target_parent)
PY
}

write_existing_safe() {
  python3 -B - "$1" "$2" <<'PY'
import os, stat, sys
path, text = sys.argv[1:]
parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
try:
    for component in os.path.dirname(path).split(os.sep)[1:]:
        next_parent = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        os.close(parent); parent = next_parent
    fd = os.open(os.path.basename(path), os.O_WRONLY | os.O_NOFOLLOW, dir_fd=parent)
finally:
    os.close(parent)
try:
    meta = os.fstat(fd)
    if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1: raise SystemExit(1)
    os.ftruncate(fd, 0); os.write(fd, text.encode())
finally:
    os.close(fd)
PY
}

copy_existing_safe() {
  python3 -B - "$1" "$2" <<'PY'
import os, shutil, stat, sys
source, target = sys.argv[1:]
parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
try:
    for component in os.path.dirname(target).split(os.sep)[1:]:
        next_parent = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        os.close(parent); parent = next_parent
    fd = os.open(os.path.basename(target), os.O_WRONLY | os.O_NOFOLLOW, dir_fd=parent)
finally:
    os.close(parent)
try:
    meta = os.fstat(fd)
    if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1: raise SystemExit(1)
    os.ftruncate(fd, 0)
    with open(source, "rb") as inp, os.fdopen(fd, "wb", closefd=False) as out: shutil.copyfileobj(inp, out)
finally:
    os.close(fd)
PY
}

verify_no_shared_inode() {
  local source_root="$1" copied_root="$2" source rel copied
  shared_inode_detail=""
  while IFS= read -r -d '' source; do
    rel="${source#"$source_root"/}"
    copied="$copied_root/$rel"
    if [[ ! -f "$copied" || -L "$copied" ]]; then
      shared_inode_detail="MISSING_OR_UNSAFE:$rel"
      return 1
    fi
    if [[ "$(stat -c '%d:%i' "$source")" == "$(stat -c '%d:%i' "$copied")" ]]; then
      shared_inode_detail="$rel"
      return 1
    fi
  done < <(find "$source_root" -type f -print0)
}

rewrite_exports() {
  local source="$1" base="$2" run="$3" output="$4" formal_iteration_cap="${5:-}" gpu_profile="${6:-$DEFAULT_GPU_PROFILE}" nlte_bf_stage_weights="${7:-}"
  validate_gpu_profile "$gpu_profile" || return 1
  python3 -B - "$source" "$base" "$run" "$output" "$formal_iteration_cap" "$gpu_profile" "$nlte_bf_stage_weights" <<'PY'
import os, re, sys
import os
from pathlib import Path
source, base, run, output = map(Path, sys.argv[1:5])
formal_iteration_cap = sys.argv[5]
gpu_profile = sys.argv[6]
nlte_bf_stage_weights = sys.argv[7]
if formal_iteration_cap and formal_iteration_cap not in {"64", "128"}:
    raise SystemExit("invalid formal iteration cap")
if nlte_bf_stage_weights and nlte_bf_stage_weights != "LAST_COMMITTED_NLTE":
    raise SystemExit("invalid NLTE BF stage weights")
pattern = re.compile(r'^declare -x ([A-Za-z_][A-Za-z0-9_]*)="(.*)"$')
values = {}
order = []
for raw in source.read_text(encoding="utf-8").splitlines():
    match = pattern.fullmatch(raw)
    if not match:
        raise SystemExit("invalid export line")
    key, value = match.groups()
    if key in values:
        raise SystemExit("duplicate export")
    values[key] = value
    order.append(key)
base_s, run_s = str(base), str(run)
for key in order:
    values[key] = values[key].replace(base_s, run_s)
values["LUMINA_FIXED_TE_PROFILE"] = f"{run_s}/input/fixed_te_profile.txt"
values.pop("LUMINA_RADEQ_DIAG_TE_K", None)
order = [key for key in order if key != "LUMINA_RADEQ_DIAG_TE_K"]
for key in ("LUMINA_ION_POP_DUMP", "LUMINA_ION_POP_DUMP_ITER", "LUMINA_NLTE_LEVEL_DUMP"):
    if key not in values:
        order.append(key)
    values[key] = "1"
if "LUMINA_CMF_FINE_MGPU_DEVICES" not in values:
    order.append("LUMINA_CMF_FINE_MGPU_DEVICES")
values["LUMINA_CMF_FINE_MGPU_DEVICES"] = {"h200x1": "1", "h100-a100x2": "2"}[gpu_profile]
if nlte_bf_stage_weights:
    if "LUMINA_NLTE_BF_STAGE_WEIGHTS" not in values:
        order.append("LUMINA_NLTE_BF_STAGE_WEIGHTS")
    values["LUMINA_NLTE_BF_STAGE_WEIGHTS"] = nlte_bf_stage_weights
if formal_iteration_cap:
    if "LUMINA_CMF_FINE_ALI" not in values:
        order.append("LUMINA_CMF_FINE_ALI")
    values["LUMINA_CMF_FINE_ALI"] = formal_iteration_cap
parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
try:
    for component in os.path.dirname(str(output)).split(os.sep)[1:]:
        next_parent = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        os.close(parent); parent = next_parent
    fd = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640, dir_fd=parent)
finally:
    os.close(parent)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    stream.write("".join(f'declare -x {key}="{values[key]}"\n' for key in order))
PY
}

verify_export_delta() {
  local source="$1" staged="$2" base="$3" run="$4" formal_iteration_cap="${5:-}" gpu_profile="${6:-$DEFAULT_GPU_PROFILE}" nlte_bf_stage_weights="${7:-}"
  validate_gpu_profile "$gpu_profile" || return 1
  python3 -B - "$source" "$staged" "$base" "$run" "$formal_iteration_cap" "$gpu_profile" "$nlte_bf_stage_weights" <<'PY'
import os, re, sys
from pathlib import Path
formal_iteration_cap = sys.argv[5]
gpu_profile = sys.argv[6]
nlte_bf_stage_weights = sys.argv[7]
if formal_iteration_cap and formal_iteration_cap not in {"64", "128"}:
    raise SystemExit(1)
if nlte_bf_stage_weights and nlte_bf_stage_weights != "LAST_COMMITTED_NLTE":
    raise SystemExit(1)
pattern = re.compile(r'^declare -x ([A-Za-z_][A-Za-z0-9_]*)="(.*)"$')
def load(path):
    result = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.fullmatch(raw)
        if not match or match.group(1) in result:
            raise SystemExit(1)
        result[match.group(1)] = match.group(2)
    return result
old, new = load(sys.argv[1]), load(sys.argv[2])
base, run = sys.argv[3:5]
expected = {key: value.replace(base, run) for key, value in old.items()}
expected["LUMINA_FIXED_TE_PROFILE"] = f"{run}/input/fixed_te_profile.txt"
expected.pop("LUMINA_RADEQ_DIAG_TE_K", None)
expected.update(LUMINA_ION_POP_DUMP="1", LUMINA_ION_POP_DUMP_ITER="1", LUMINA_NLTE_LEVEL_DUMP="1")
expected["LUMINA_CMF_FINE_MGPU_DEVICES"] = {"h200x1": "1", "h100-a100x2": "2"}[gpu_profile]
if nlte_bf_stage_weights:
    expected["LUMINA_NLTE_BF_STAGE_WEIGHTS"] = nlte_bf_stage_weights
if formal_iteration_cap:
    expected["LUMINA_CMF_FINE_ALI"] = formal_iteration_cap
if new != expected:
    raise SystemExit(1)
PY
}

json_field() {
  local path="$1" field="$2"
  python3 -B - "$path" "$field" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split('.'):
    value = value[key]
print(value)
PY
}

registered_formal_iteration_cap() {
  python3 -B - "$1" <<'PY'
import json, sys
expectations = json.load(open(sys.argv[1], encoding="utf-8"))
if "formal_iteration_cap" not in expectations:
    raise SystemExit(0)
value = expectations["formal_iteration_cap"]
if isinstance(value, bool) or not isinstance(value, int) or value not in (64, 128):
    raise SystemExit(1)
print(value)
PY
}

registered_nlte_bf_stage_weights() {
  python3 -B - "$1" <<'PY'
import json, sys
expectations = json.load(open(sys.argv[1], encoding="utf-8"))
if "nlte_bf_stage_weights" not in expectations:
    raise SystemExit(0)
value = expectations["nlte_bf_stage_weights"]
if value != "LAST_COMMITTED_NLTE":
    raise SystemExit(1)
print(value)
PY
}

observation_prerequisites() {
  python3 -B - "$1" "$2" <<'PY'
import json, sys
expectations = json.load(open(sys.argv[1], encoding="utf-8"))
writer = json.load(open(sys.argv[2], encoding="utf-8"))
deferred = {"actual_final_publication_manifest", "final_publication_generation_link"}
if set(expectations.get("pending", [])) - deferred or set(writer.get("pending", [])) - deferred:
    raise SystemExit(1)
if expectations.get("schema") != "CMFGEN_TWIN_MAP_REV4_V1":
    raise SystemExit(1)
shells = expectations.get("shell_count")
base_ez = expectations.get("base_e_z", {})
self_check = expectations.get("base_self_comparison", {}).get("paired_sign_counts", {})
top = expectations.get("reference_top_bin_semantics", {})
if (
    isinstance(shells, bool) or not isinstance(shells, int) or shells <= 0
    or base_ez.get("cell_count") != shells or not isinstance(base_ez.get("M_Z_interval"), list)
    or self_check != {"improved": 0, "worsened": 0, "representation_unresolved": shells}
    or set(top) != {"ca", "co", "fe", "ni", "si", "s"}
    or any(not isinstance(record, dict) or record.get("value") not in {"CUMULATIVE", "INDIVIDUAL"} or not record.get("source") for record in top.values())
    or expectations.get("mass_convention") != {
        "production": "sealed_model_atom_masses.csv",
        "comparison": {"Fe": "56", "Co": "56", "Ni": "56", "other": "sealed_model_atom_masses.csv"},
    }
):
    raise SystemExit(1)
if writer.get("manifest_relative_path") != "work/cmfgen_final_publication.json":
    raise SystemExit(1)
PY
}

resource_check() {
  local card="$1" free_bytes="$2" input_bytes="$3" evidence_bytes="${4:-0}" planned_bytes="${5:-0}"
  python3 -B - "$card" "$free_bytes" "$input_bytes" "$evidence_bytes" "$planned_bytes" "$STORAGE_CAP_BYTES" <<'PY'
import json, sys
try:
    card = json.load(open(sys.argv[1], encoding="utf-8"))
    values = [card["dump_upper_bound_formula_bytes"], *sys.argv[2:]]
    if any(isinstance(value, bool) or not str(value).isdigit() for value in values):
        raise ValueError
    estimate, free_bytes, input_bytes, evidence_bytes, planned_bytes, cap = map(int, values)
    if card.get("pending") or estimate > cap or input_bytes > cap or evidence_bytes > cap or planned_bytes > cap:
        raise ValueError
    if estimate > cap - input_bytes - evidence_bytes - planned_bytes:
        raise ValueError
    total = estimate + input_bytes + evidence_bytes + planned_bytes
    if free_bytes < total:
        raise ValueError
except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

binary_sm90_check() {
  local binary="$1" listing
  command -v cuobjdump >/dev/null || return 1
  listing="$(cuobjdump --list-elf "$binary" 2>/dev/null)" || return 1
  grep -Eq '(^|[^[:alnum:]_])sm_90([^[:alnum:]_]|$)' <<< "$listing" || return 1
  printf 'CMFGEN_TWIN_MAP_BINARY_COMPATIBILITY binary_sha256=%s target=sm_90\n' "$(sha256sum "$binary" | awk '{print $1}')"
}

rewrite_job_resources() {
  local source="$1" output="$2" gpu_profile="${3:-$DEFAULT_GPU_PROFILE}" old_root="${4:-}" new_root="${5:-}"
  validate_gpu_profile "$gpu_profile" || return 1
  python3 -B - "$source" "$output" "$gpu_profile" "$old_root" "$new_root" <<'PY'
import os, re, sys
from pathlib import Path
source, output = map(Path, sys.argv[1:3])
text = source.read_text(encoding="utf-8")
gpu_profile = sys.argv[3]
if sys.argv[4]:
    text = text.replace(sys.argv[4], sys.argv[5])
if "requires A100 hardware" in text or "must use partition a100" in text:
    raise SystemExit("INPUT_PENDING: legacy A100-only runner is not qualified for H200/H100")
profiles = {
    "h200x1": {"partition": "h200", "gres": "gpu:1"},
    "h100-a100x2": {"partition": "h100,a100", "gres": "gpu:2"},
}
required = {
    "partition": f"#SBATCH --partition={profiles[gpu_profile]['partition']}",
    "gres": f"#SBATCH --gres={profiles[gpu_profile]['gres']}",
    "cpus-per-task": "#SBATCH --cpus-per-task=32",
    "time": "#SBATCH --time=04:00:00",
}
lines = text.splitlines()
for key, replacement in required.items():
    pattern = re.compile(rf"^#SBATCH --{re.escape(key)}(?:=|\s).*$")
    hits = [i for i, line in enumerate(lines) if pattern.match(line)]
    if len(hits) != 1:
        raise SystemExit(f"INPUT_PENDING: runner directive {key} count={len(hits)}")
    lines[hits[0]] = replacement
profile_hits = [i for i, line in enumerate(lines) if line.startswith("readonly GPU_PROFILE=")]
if len(profile_hits) != 1:
    raise SystemExit(f"INPUT_PENDING: runner GPU_PROFILE count={len(profile_hits)}")
lines[profile_hits[0]] = f"readonly GPU_PROFILE={gpu_profile}"
parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
try:
    for component in os.path.dirname(str(output)).split(os.sep)[1:]:
        next_parent = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        os.close(parent); parent = next_parent
    fd = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640, dir_fd=parent)
finally:
    os.close(parent)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")
PY
}

stage_transaction() {
  local run_binary="${5:-$1/input/lumina_cuda}" run_build_identity="${6:-}" gpu_profile="${7:-$DEFAULT_GPU_PROFILE}"
  validate_gpu_profile "$gpu_profile" || return 1
  python3 -B - "$1" "$2" "$3" "$4" "$STORAGE_CAP_BYTES" "$run_binary" "$run_build_identity" "$gpu_profile" <<'PY'
import hashlib, json, os, re, stat, sys

base, evidence, run, runner = sys.argv[1:5]
cap = int(sys.argv[5])
run_binary, run_build_identity, gpu_profile = sys.argv[6:9]
profiles = {
    "h200x1": {"partitions": "h200", "count": 1, "kinds": "H200", "minimum_memory_mib": 140000},
    "h100-a100x2": {"partitions": "h100,a100", "count": 2, "kinds": "H100,A100", "minimum_memory_mib": 80000},
}
gpu = profiles[gpu_profile]

def directory_fd(path):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.split(os.sep)[1:]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = next_fd
        return fd
    except BaseException:
        os.close(fd); raise

def scan_tree(root_fd):
    files, directories = {}, []
    def visit(fd, prefix):
        for name in sorted(os.listdir(fd)):
            meta = os.stat(name, dir_fd=fd, follow_symlinks=False)
            rel = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(meta.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                directories.append((rel, stat.S_IMODE(meta.st_mode)))
                try: visit(child, rel)
                finally: os.close(child)
            elif stat.S_ISREG(meta.st_mode):
                source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    digest = hashlib.sha256()
                    while True:
                        block = os.read(source, 1024 * 1024)
                        if not block: break
                        digest.update(block)
                finally: os.close(source)
                files[rel] = (meta.st_size, stat.S_IMODE(meta.st_mode), (meta.st_dev, meta.st_ino), digest.hexdigest())
            else:
                raise RuntimeError(f"unsafe tree entry: {rel}")
    visit(root_fd, "")
    return files, directories

def manifest(files):
    return "".join(f"{value[3]}  ./{rel}\n" for rel, value in sorted(files.items())).encode()

def read_file(root_fd, rel):
    parts = rel.split("/"); parent = open_relative(root_fd, parts[:-1])
    try: fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    finally: os.close(parent)
    try:
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block: break
            chunks.append(block)
        return b"".join(chunks)
    finally: os.close(fd)

def transform_exports(data, formal_iteration_cap, nlte_bf_stage_weights):
    pattern = re.compile(r'^declare -x ([A-Za-z_][A-Za-z0-9_]*)="(.*)"$')
    values, order = {}, []
    for raw in data.decode().splitlines():
        match = pattern.fullmatch(raw)
        if not match or match.group(1) in values: raise RuntimeError("invalid export line")
        key, value = match.groups(); values[key] = value; order.append(key)
    for key in order: values[key] = values[key].replace(base, run)
    values["LUMINA_FIXED_TE_PROFILE"] = f"{run}/input/fixed_te_profile.txt"
    values.pop("LUMINA_RADEQ_DIAG_TE_K", None)
    order = [key for key in order if key != "LUMINA_RADEQ_DIAG_TE_K"]
    for key in ("LUMINA_ION_POP_DUMP", "LUMINA_ION_POP_DUMP_ITER", "LUMINA_NLTE_LEVEL_DUMP"):
        if key not in values: order.append(key)
        values[key] = "1"
    if "LUMINA_CMF_FINE_MGPU_DEVICES" not in values: order.append("LUMINA_CMF_FINE_MGPU_DEVICES")
    values["LUMINA_CMF_FINE_MGPU_DEVICES"] = str(gpu["count"])
    if nlte_bf_stage_weights is not None:
        if "LUMINA_NLTE_BF_STAGE_WEIGHTS" not in values: order.append("LUMINA_NLTE_BF_STAGE_WEIGHTS")
        values["LUMINA_NLTE_BF_STAGE_WEIGHTS"] = nlte_bf_stage_weights
    if formal_iteration_cap is not None:
        if "LUMINA_CMF_FINE_ALI" not in values: order.append("LUMINA_CMF_FINE_ALI")
        values["LUMINA_CMF_FINE_ALI"] = str(formal_iteration_cap)
    return "".join(f'declare -x {key}="{values[key]}"\n' for key in order).encode()

def transform_job(data):
    text = data.decode().replace(base, run)
    if "requires A100 hardware" in text or "must use partition a100" in text:
        raise RuntimeError("legacy A100-only runner is not qualified")
    required = {"partition":f"#SBATCH --partition={gpu['partitions']}", "gres":f"#SBATCH --gres=gpu:{gpu['count']}", "cpus-per-task":"#SBATCH --cpus-per-task=32", "time":"#SBATCH --time=04:00:00"}
    lines = text.splitlines()
    for key, replacement in required.items():
        pattern = re.compile(rf"^#SBATCH --{re.escape(key)}(?:=|\s).*$")
        hits = [i for i, line in enumerate(lines) if pattern.match(line)]
        if len(hits) != 1: raise RuntimeError(f"runner directive {key} count={len(hits)}")
        lines[hits[0]] = replacement
    profile_hits = [i for i, line in enumerate(lines) if line.startswith("readonly GPU_PROFILE=")]
    if len(profile_hits) != 1: raise RuntimeError(f"runner GPU_PROFILE count={len(profile_hits)}")
    lines[profile_hits[0]] = f"readonly GPU_PROFILE={gpu_profile}"
    return ("\n".join(lines) + "\n").encode()

def open_relative(root_fd, parts, create=False):
    fd = os.dup(root_fd)
    try:
        for component in parts:
            if create:
                try: os.mkdir(component, 0o750, dir_fd=fd)
                except FileExistsError: pass
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = next_fd
        return fd
    except BaseException:
        os.close(fd); raise

def create_file(root_fd, rel, data, mode):
    parts = rel.split("/"); parent = open_relative(root_fd, parts[:-1])
    try:
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode, dir_fd=parent)
        try:
            view = memoryview(data)
            while view: view = view[os.write(fd, view):]
            os.fsync(fd)
        finally: os.close(fd)
    finally: os.close(parent)

def copy_file(source_root_fd, target_root_fd, rel, mode):
    parts = rel.split("/")
    source_parent = open_relative(source_root_fd, parts[:-1]); target_parent = open_relative(target_root_fd, parts[:-1])
    source = target = -1
    try:
        source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=source_parent)
        target = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode, dir_fd=target_parent)
        while True:
            block = os.read(source, 1024 * 1024)
            if not block: break
            view = memoryview(block)
            while view: view = view[os.write(target, view):]
        os.fsync(target)
    finally:
        if source >= 0: os.close(source)
        if target >= 0: os.close(target)
        os.close(source_parent); os.close(target_parent)

def clear_dir(fd):
    for name in os.listdir(fd):
        meta = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISDIR(meta.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try: clear_dir(child)
            finally: os.close(child)
            os.rmdir(name, dir_fd=fd)
        else:
            os.unlink(name, dir_fd=fd)

base_fd = directory_fd(base)
evidence_fd = directory_fd(evidence)
input_fd = os.open("input", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=base_fd)
run_fd = -1; run_parent_fd = -1; run_created = False; complete = False; evidence_published = []
try:
    base_files, _base_dirs = scan_tree(base_fd)
    before_manifest = manifest(base_files)
    input_files, input_dirs = scan_tree(input_fd)
    evidence_files, _evidence_dirs = scan_tree(evidence_fd)
    profile = read_file(evidence_fd, "inputs/fixed_te_profile.txt")
    expectations = json.loads(read_file(evidence_fd, "offline/expectations.json"))
    formal_iteration_cap = expectations.get("formal_iteration_cap")
    if "formal_iteration_cap" in expectations and (isinstance(formal_iteration_cap, bool) or not isinstance(formal_iteration_cap, int) or formal_iteration_cap not in (64, 128)):
        raise RuntimeError("invalid registered formal iteration cap")
    nlte_bf_stage_weights = expectations.get("nlte_bf_stage_weights")
    if "nlte_bf_stage_weights" in expectations and nlte_bf_stage_weights != "LAST_COMMITTED_NLTE":
        raise RuntimeError("invalid registered NLTE BF stage weights")
    exports = transform_exports(read_file(input_fd, "resolved_lumina.exports"), formal_iteration_cap, nlte_bf_stage_weights)
    with open(runner, "rb") as stream:
        job = transform_job(stream.read())
    binary_path = os.path.realpath(run_binary)
    if binary_path != run_binary or not os.path.isfile(binary_path) or os.path.islink(run_binary):
        raise RuntimeError("run binary unsafe")
    with open(binary_path, "rb") as stream: binary = stream.read()
    binary_hash = hashlib.sha256(binary).hexdigest()
    if run_build_identity:
        identity_path = os.path.realpath(run_build_identity)
        if identity_path != run_build_identity or not os.path.isfile(identity_path) or os.path.islink(run_build_identity):
            raise RuntimeError("run build identity unsafe")
        with open(identity_path, "rb") as stream: build_identity = stream.read()
        record = json.loads(build_identity)
        if (record.get("schema") != "CMFGEN_TWIN_RUN_BUILD_V1" or
            record.get("binary", {}).get("path") != run_binary or
            record.get("binary", {}).get("sha256") != binary_hash):
            raise RuntimeError("run build identity mismatch")
    else:
        build_identity = json.dumps({"schema":"CMFGEN_TWIN_RUN_BUILD_V1_SELFTEST", "binary":{"path":run_binary,"sha256":binary_hash}}, sort_keys=True).encode() + b"\n"
    text = job.decode()
    required_directives = {
        "partition": gpu["partitions"], "nodes": "1", "ntasks": "1", "gres": f"gpu:{gpu['count']}",
        "cpus-per-task": "32", "mem": "64G", "time": "04:00:00", "no-requeue": None,
    }
    for key, value in required_directives.items():
        expected = f"#SBATCH --{key}" + (f"={value}" if value is not None else "")
        if text.splitlines().count(expected) != 1:
            raise RuntimeError(f"dedicated runner directive mismatch: {key}")
    if "#SBATCH --exclude=grammar" in text or "CUDA_VISIBLE_DEVICES=0,1" in text:
        raise RuntimeError("dedicated runner contains forbidden node or GPU mask override")
    overrides = {"fixed_te_profile.txt": profile, "resolved_lumina.exports": exports,
                 "job.slurm": job, "lumina_cuda": binary,
                 "binary.sha256": (binary_hash + "\n").encode(),
                 "run_build_identity.json": build_identity}
    staged_files = dict(input_files)
    for rel, data in overrides.items():
        mode = 0o750 if rel == "lumina_cuda" else 0o640
        staged_files[rel] = (len(data), mode, (-1, -1), hashlib.sha256(data).hexdigest())
    input_manifest = manifest(staged_files)
    base_binary_hash = input_files["lumina_cuda"][3]
    stage_identity = ("\n".join(("schema=CMFGEN_TWIN_MAP_STAGE_V2", f"base={base}", f"evidence={evidence}", f"run={run}", "registered_runs=1", "outer_iterations=2", f"gpu_profile={gpu_profile}", f"expected_partitions={gpu['partitions']}", f"expected_gpu_count={gpu['count']}", f"expected_gpu_kinds={gpu['kinds']}", f"minimum_gpu_memory_mib={gpu['minimum_memory_mib']}", f"sealed_mgpu_devices={gpu['count']}", f"base_binary_sha256={base_binary_hash}", f"run_binary_sha256={binary_hash}", f"run_build_identity_sha256={hashlib.sha256(build_identity).hexdigest()}", f"effective_exports_sha256={hashlib.sha256(exports).hexdigest()}", f"runner_sha256={hashlib.sha256(job).hexdigest()}", f"fixed_te_profile_file_sha256={hashlib.sha256(profile).hexdigest()}", "state=STAGED_NOT_RUN")) + "\n").encode()
    card = json.loads(read_file(evidence_fd, "offline/resource_card.json"))
    estimate = card.get("dump_upper_bound_formula_bytes")
    future_fields = card.get("future_storage_upper_bounds_bytes")
    required_future = ("run_logs", "result_and_summary", "reproduction_and_audit")
    if estimate is None or card.get("pending") or not isinstance(future_fields, dict) or any(future_fields.get(name) is None for name in required_future):
        raise SystemExit("INPUT_PENDING: complete resource and future-storage upper bounds are required before staging")
    future_values = [future_fields.get(name) for name in required_future]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in [estimate, *future_values]):
        raise RuntimeError("resource card invalid")
    future = sum(future_values)
    input_bytes = sum(value[0] for value in staged_files.values())
    evidence_before = sum(value[0] for value in evidence_files.values())
    directory_reservation = 4096 * (len(input_dirs) + 3)
    planned = input_bytes + len(stage_identity) + len(input_manifest) + 2 * len(before_manifest) + directory_reservation
    run_parent = os.path.dirname(run); run_name = os.path.basename(run)
    run_parent_fd = directory_fd(run_parent)
    forbidden = {(os.fstat(base_fd).st_dev, os.fstat(base_fd).st_ino), (os.fstat(evidence_fd).st_dev, os.fstat(evidence_fd).st_ino)}
    cursor = os.dup(run_parent_fd)
    try:
        while True:
            meta = os.fstat(cursor)
            if (meta.st_dev, meta.st_ino) in forbidden: raise RuntimeError("run parent crossed protected root")
            parent = os.open("..", os.O_RDONLY | os.O_DIRECTORY, dir_fd=cursor)
            pmeta = os.fstat(parent)
            if (pmeta.st_dev, pmeta.st_ino) == (meta.st_dev, meta.st_ino): os.close(parent); break
            os.close(cursor); cursor = parent
    finally: os.close(cursor)
    free = os.statvfs(run_parent_fd).f_bavail * os.statvfs(run_parent_fd).f_frsize
    peak = evidence_before + planned + estimate + future
    if peak > cap or planned + estimate + future > free: raise RuntimeError("resource reservation rejected")
    print(f"CMFGEN_TWIN_MAP_RESERVATION_ESTABLISHED evidence_before={evidence_before} input_copy={input_bytes} metadata_and_directories={planned-input_bytes} temporary=0 dump={estimate} run_logs={future_values[0]} result_summary={future_values[1]} reproduction_audit={future_values[2]} peak={peak} cap={cap} first_write=run_mkdir", flush=True)

    os.mkdir(run_name, 0o750, dir_fd=run_parent_fd); run_created = True
    run_fd = os.open(run_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=run_parent_fd)
    os.mkdir("input", 0o750, dir_fd=run_fd); os.mkdir("work", 0o750, dir_fd=run_fd)
    staged_input_fd = os.open("input", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=run_fd)
    try:
        for rel, mode in sorted(input_dirs):
            parent = open_relative(staged_input_fd, rel.split("/")[:-1])
            try: os.mkdir(rel.split("/")[-1], mode & 0o750 or 0o750, dir_fd=parent)
            finally: os.close(parent)
        for rel, value in sorted(staged_files.items()):
            mode = value[1] & 0o750 or 0o640
            if rel in overrides: create_file(staged_input_fd, rel, overrides[rel], mode)
            else: copy_file(input_fd, staged_input_fd, rel, mode)
        copied_files, _ = scan_tree(staged_input_fd)
        if manifest(copied_files) != input_manifest: raise RuntimeError("input copy mismatch")
    finally: os.close(staged_input_fd)
    create_file(run_fd, "STAGE_IDENTITY.txt", stage_identity, 0o640)
    create_file(run_fd, "input_manifest.sha256", input_manifest, 0o640)
    after_files, _ = scan_tree(base_fd)
    if manifest(after_files) != before_manifest: raise RuntimeError("base seal changed during stage")
    evidence_inputs_fd = os.open("inputs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=evidence_fd)
    try:
        create_file(evidence_inputs_fd, "base_before.sha256", before_manifest, 0o640)
        evidence_published.append("base_before.sha256")
        create_file(evidence_inputs_fd, "base_after_stage.sha256", before_manifest, 0o640)
        evidence_published.append("base_after_stage.sha256")
    finally: os.close(evidence_inputs_fd)
    run_files, _ = scan_tree(run_fd); final_evidence_files, _ = scan_tree(evidence_fd)
    actual = sum(value[0] for value in run_files.values()) + sum(value[0] for value in final_evidence_files.values())
    post_free = os.statvfs(run_fd).f_bavail * os.statvfs(run_fd).f_frsize
    if actual + estimate + future > cap or post_free < estimate + future: raise RuntimeError("post-stage storage check failed")
    print(f"CMFGEN_TWIN_MAP_STAGE_COMPLETE copied_input_bytes={input_bytes} dump_bound_bytes={estimate} actual_stage_bytes={actual} base_unchanged=1 state=STAGED_NOT_RUN")
    complete = True
finally:
    if evidence_published and not complete:
        cleanup_inputs = os.open("inputs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=evidence_fd)
        try:
            for name in evidence_published:
                try: os.unlink(name, dir_fd=cleanup_inputs)
                except FileNotFoundError: pass
        finally: os.close(cleanup_inputs)
    if run_created and not complete and run_fd >= 0:
        clear_dir(run_fd)
        os.rmdir(os.path.basename(run), dir_fd=run_parent_fd)
    if run_fd >= 0: os.close(run_fd)
    if run_parent_fd >= 0: os.close(run_parent_fd)
    os.close(input_fd); os.close(evidence_fd); os.close(base_fd)
PY
}

stager_selftest() {
  local scratch base evidence dest before after exports staged bad cap_staged free input_bytes job staged_job valid_expectations
  local runner_template
  local positive=0 negative=0
  scratch="$(mktemp -d /tmp/cmfgen-twin-stage-selftest.XXXXXX)"
  runner_template="$(dirname "$(realpath -e -- "$0")")/run_cmfgen_twin_map.sh"
  base="$scratch/base"; evidence="$scratch/evidence"; dest="$scratch/run"
  before="$scratch/before.sha256"; after="$scratch/after.sha256"
  exports="$base/input/resolved_lumina.exports"; staged="$scratch/staged.exports"; bad="$scratch/bad.exports"
  trap 'rm -rf -- "$scratch"' EXIT
  mkdir -p "$base/input" "$evidence"/{inputs,offline,run,result}
  : > "$before"; : > "$after"
  printf 'payload\n' > "$base/input/data"
  printf '%s\n' \
    "declare -x LUMINA_MODEL_DIR=\"$base/input/model\"" \
    'declare -x LUMINA_FIXED_TE_PROFILE="/sealed/old.txt"' \
    'declare -x LUMINA_CMF_FINE_TOL="1e-8"' \
    'declare -x LUMINA_RADEQ_DIAG_TE_K="10020"' > "$exports"

  require_safe_base "$base" || die "SELFTEST_POSITIVE_BASE_REJECTED"
  printf 'STAGER_SELFTEST control=base_path observation=accepted\n'; positive=$((positive + 1))
  require_new_root "$dest" "$base" "$evidence" || die "SELFTEST_POSITIVE_ROOT_REJECTED"
  require_evidence_root "$evidence" "$base" "$dest" || die "SELFTEST_POSITIVE_EVIDENCE_REJECTED"
  printf 'STAGER_SELFTEST control=independent_roots observation=accepted\n'; positive=$((positive + 1))
  rewrite_exports "$exports" "$base" "$dest" "$staged"
  verify_export_delta "$exports" "$staged" "$base" "$dest" || die "SELFTEST_POSITIVE_DELTA_REJECTED"
  grep -qx 'declare -x LUMINA_CMF_FINE_MGPU_DEVICES="1"' "$staged" || die "SELFTEST_H200X1_EXPORT_MISSING"
  printf 'STAGER_SELFTEST control=exports_delta observation=accepted\n'; positive=$((positive + 1))
  local alt_staged="$scratch/staged-h100-a100x2.exports"
  rewrite_exports "$exports" "$base" "$dest" "$alt_staged" '' h100-a100x2
  verify_export_delta "$exports" "$alt_staged" "$base" "$dest" '' h100-a100x2 || die "SELFTEST_ALT_PROFILE_DELTA_REJECTED"
  grep -qx 'declare -x LUMINA_CMF_FINE_MGPU_DEVICES="2"' "$alt_staged" || die "SELFTEST_ALT_PROFILE_EXPORT_MISSING"
  printf 'STAGER_SELFTEST control=gpu_profile_exports profiles=h200x1,h100-a100x2 observation=accepted\n'; positive=$((positive + 1))
  local selector_staged="$scratch/staged-bf-selector.exports"
  rewrite_exports "$exports" "$base" "$dest" "$selector_staged" '' h200x1 LAST_COMMITTED_NLTE
  verify_export_delta "$exports" "$selector_staged" "$base" "$dest" '' h200x1 LAST_COMMITTED_NLTE || die "SELFTEST_BF_SELECTOR_ROUNDTRIP_REJECTED"
  grep -qx 'declare -x LUMINA_NLTE_BF_STAGE_WEIGHTS="LAST_COMMITTED_NLTE"' "$selector_staged" || die "SELFTEST_BF_SELECTOR_EXPORT_MISSING"
  printf 'STAGER_SELFTEST control=NLTE_BF_STAGE_WEIGHTS observation=roundtrip_accepted\n'; positive=$((positive + 1))
  if verify_export_delta "$exports" "$selector_staged" "$base" "$dest" '' h200x1; then die "SELFTEST_UNREGISTERED_BF_SELECTOR_ACCEPTED"; fi
  printf 'STAGER_SELFTEST inject=UNREGISTERED_BF_SELECTOR observation=rejected\n'; negative=$((negative + 1))
  sed 's/LAST_COMMITTED_NLTE/UNREGISTERED/' "$selector_staged" > "$bad"
  if verify_export_delta "$exports" "$bad" "$base" "$dest" '' h200x1 LAST_COMMITTED_NLTE; then die "SELFTEST_BF_SELECTOR_MISMATCH_ACCEPTED"; fi
  printf 'STAGER_SELFTEST inject=BF_SELECTOR_CONTRACT_EXPORT_MISMATCH observation=rejected\n'; negative=$((negative + 1))
  printf '{"nlte_bf_stage_weights":"UNKNOWN"}\n' > "$scratch/invalid-selector.json"
  if registered_nlte_bf_stage_weights "$scratch/invalid-selector.json" >/dev/null; then die "SELFTEST_UNKNOWN_BF_SELECTOR_ACCEPTED"; fi
  if rewrite_exports "$exports" "$base" "$dest" "$scratch/invalid-selector.exports" '' h200x1 UNKNOWN >/dev/null 2>&1; then die "SELFTEST_UNKNOWN_BF_SELECTOR_REWRITE_ACCEPTED"; fi
  [[ ! -e "$scratch/invalid-selector.exports" ]] || die "SELFTEST_UNKNOWN_BF_SELECTOR_WROTE_OUTPUT"
  printf 'STAGER_SELFTEST inject=UNKNOWN_BF_SELECTOR observation=rejected_before_write\n'; negative=$((negative + 1))
  cap_staged="$scratch/staged-cap128.exports"
  rewrite_exports "$exports" "$base" "$dest" "$cap_staged" 128
  verify_export_delta "$exports" "$cap_staged" "$base" "$dest" 128 || die "SELFTEST_REGISTERED_CAP128_REJECTED"
  grep -qx 'declare -x LUMINA_CMF_FINE_ALI="128"' "$cap_staged" || die "SELFTEST_REGISTERED_CAP128_MISSING"
  printf 'STAGER_SELFTEST control=REGISTERED_CAP128 observation=accepted\n'; positive=$((positive + 1))
  if verify_export_delta "$exports" "$cap_staged" "$base" "$dest"; then die "SELFTEST_ACCEPTED_UNREGISTERED_CAP128"; fi
  printf 'STAGER_SELFTEST inject=UNREGISTERED_CAP128 observation=rejected\n'; negative=$((negative + 1))
  if verify_export_delta "$exports" "$cap_staged" "$base" "$dest" 64; then die "SELFTEST_ACCEPTED_CAP64_MISMATCH"; fi
  printf 'STAGER_SELFTEST inject=REGISTERED_CAP64_ACTUAL128 observation=rejected\n'; negative=$((negative + 1))
  sed 's/LUMINA_CMF_FINE_TOL="1e-8"/LUMINA_CMF_FINE_TOL="2e-8"/' "$cap_staged" > "$bad"
  if verify_export_delta "$exports" "$bad" "$base" "$dest" 128; then die "SELFTEST_ACCEPTED_TOLERANCE_MUTATION"; fi
  printf 'STAGER_SELFTEST inject=TOLERANCE_MUTATION observation=rejected\n'; negative=$((negative + 1))
  printf '{"formal_iteration_cap":true}\n' > "$scratch/invalid-cap.json"
  if registered_formal_iteration_cap "$scratch/invalid-cap.json" >/dev/null; then die "SELFTEST_ACCEPTED_INVALID_BOOL_CAP"; fi
  if rewrite_exports "$exports" "$base" "$dest" "$scratch/invalid-cap.exports" 65 >/dev/null 2>&1; then die "SELFTEST_ACCEPTED_INVALID_VALUE_CAP"; fi
  printf 'STAGER_SELFTEST inject=INVALID_CAP_BOOL_OR_VALUE observation=rejected\n'; negative=$((negative + 1))
  manifest_tree "$base" "$before"; manifest_tree "$base" "$after"; cmp -s "$before" "$after" || die "SELFTEST_POSITIVE_SEAL_REJECTED"
  printf 'STAGER_SELFTEST control=base_manifest observation=unchanged\n'; positive=$((positive + 1))
  mkdir "$scratch/publish-source" "$scratch/publish-target"
  printf 'reserved\n' > "$scratch/publish-source/item"
  publish_new "$scratch/publish-source/item" "$scratch/publish-target/item" || die "SELFTEST_DIRFD_PUBLISH_REJECTED"
  [[ ! -e "$scratch/publish-source/item" && "$(cat "$scratch/publish-target/item")" == reserved ]] || die "SELFTEST_DIRFD_PUBLISH_WRONG_RESULT"
  printf 'STAGER_SELFTEST control=directory_fd_exclusive_publish observation=accepted\n'; positive=$((positive + 1))
  printf 'new-profile\n' > "$scratch/publish-source/new-profile"
  replace_private "$scratch/publish-source/new-profile" "$scratch/publish-target/new-profile" || die "SELFTEST_NEW_PROFILE_PUBLISH_REJECTED"
  [[ "$(cat "$scratch/publish-target/new-profile")" == new-profile ]] || die "SELFTEST_NEW_PROFILE_PUBLISH_WRONG_RESULT"
  printf 'STAGER_SELFTEST control=NEW_PROFILE_DESTINATION_ABSENT observation=accepted mode=new_publish\n'; positive=$((positive + 1))
  printf 'replacement-profile\n' > "$scratch/publish-source/replacement-profile"
  replace_private "$scratch/publish-source/replacement-profile" "$scratch/publish-target/new-profile" || die "SELFTEST_EXISTING_PROFILE_REPLACE_REJECTED"
  [[ "$(cat "$scratch/publish-target/new-profile")" == replacement-profile ]] || die "SELFTEST_EXISTING_PROFILE_REPLACE_WRONG_RESULT"
  printf 'STAGER_SELFTEST control=PROFILE_DESTINATION_EXISTS observation=accepted mode=replace\n'; positive=$((positive + 1))

  mkdir -p "$scratch/ancestor-route/result" "$scratch/ancestor-held" "$scratch/protected-route/result"
  printf 'replacement\n' > "$scratch/ancestor-source"
  printf 'sealed\n' > "$scratch/protected-route/result/item"
  chmod 0440 "$scratch/protected-route/result/item"
  mv "$scratch/ancestor-route" "$scratch/ancestor-held/original"
  ln -s "$scratch/protected-route" "$scratch/ancestor-route"
  if replace_private "$scratch/ancestor-source" "$scratch/ancestor-route/result/item" 2>/dev/null; then die "SELFTEST_ACCEPTED_ANCESTOR_SWAP"; fi
  [[ "$(cat "$scratch/protected-route/result/item")" == sealed && "$(stat -c '%a' "$scratch/protected-route/result/item")" == 440 ]] || die "SELFTEST_ANCESTOR_SWAP_CHANGED_PROTECTED"
  printf 'STAGER_SELFTEST inject=ANCESTOR_PATH_SWAP observation=rejected protected_content=sealed protected_mode=0440\n'; negative=$((negative + 1))

  mkdir -p "$scratch/nested-route/one" "$scratch/nested-protected/result"
  printf 'replacement\n' > "$scratch/nested-source"
  printf 'sealed-nested\n' > "$scratch/nested-protected/result/item"
  chmod 0440 "$scratch/nested-protected/result/item"
  ln -s "$scratch/nested-protected" "$scratch/nested-route/one/two"
  if replace_private "$scratch/nested-source" "$scratch/nested-route/one/two/result/item" 2>/dev/null; then die "SELFTEST_ACCEPTED_NESTED_ANCESTOR_LINK"; fi
  [[ "$(cat "$scratch/nested-protected/result/item")" == sealed-nested && "$(stat -c '%a' "$scratch/nested-protected/result/item")" == 440 ]] || die "SELFTEST_NESTED_ANCESTOR_CHANGED_PROTECTED"
  printf 'STAGER_SELFTEST inject=NESTED_ANCESTOR_LINK observation=rejected protected_content=sealed-nested protected_mode=0440\n'; negative=$((negative + 1))

  ln -s "$scratch" "$scratch/linkdest"
  if require_new_root "$scratch/linkdest/new" "$base" "$evidence"; then die "SELFTEST_ACCEPTED_SYMLINK_DESTINATION"; fi
  printf 'STAGER_SELFTEST inject=DESTINATION_SYMLINK observation=rejected reason=DESTINATION_NOT_INDEPENDENT\n'; negative=$((negative + 1))
  if require_new_root "$base/inside" "$base" "$evidence"; then die "SELFTEST_ACCEPTED_BASE_DESTINATION"; fi
  printf 'STAGER_SELFTEST inject=DESTINATION_INSIDE_BASE observation=rejected reason=DESTINATION_NOT_INDEPENDENT\n'; negative=$((negative + 1))
  if require_new_root "$base/./inside" "$base" "$evidence"; then die "SELFTEST_ACCEPTED_DOT_ALIAS_BASE_DESTINATION"; fi
  printf 'STAGER_SELFTEST inject=BASE_DOT_ALIAS_DESTINATION observation=rejected reason=PATH_NOT_CANONICAL\n'; negative=$((negative + 1))
  if require_new_root "$scratch/other/../base/inside" "$base" "$evidence"; then die "SELFTEST_ACCEPTED_RELATIVE_COMPONENT_DESTINATION"; fi
  printf 'STAGER_SELFTEST inject=RELATIVE_COMPONENT_DESTINATION observation=rejected reason=PATH_NOT_CANONICAL\n'; negative=$((negative + 1))
  mkdir -p "$base/evidence"/{inputs,offline,run,result}
  if require_evidence_root "$base/evidence" "$base" "$dest"; then die "SELFTEST_ACCEPTED_EVIDENCE_INSIDE_BASE"; fi
  printf 'STAGER_SELFTEST inject=EVIDENCE_INSIDE_BASE observation=rejected reason=OUTPUT_LOCATION_UNSAFE\n'; negative=$((negative + 1))

  cp "$staged" "$bad"
  printf 'declare -x UNREGISTERED_DELTA="1"\n' >> "$bad"
  if verify_export_delta "$exports" "$bad" "$base" "$dest"; then die "SELFTEST_ACCEPTED_EXTRA_EXPORT"; fi
  printf 'STAGER_SELFTEST inject=EXTRA_EXPORT_DELTA observation=rejected reason=EXPORT_DELTA_NOT_EXACT\n'; negative=$((negative + 1))
  sed 's/LUMINA_ION_POP_DUMP="1"/LUMINA_ION_POP_DUMP="0"/' "$staged" > "$bad"
  if verify_export_delta "$exports" "$bad" "$base" "$dest"; then die "SELFTEST_ACCEPTED_STALE_DUMP_EXPORT"; fi
  printf 'STAGER_SELFTEST inject=STALE_EXPORTS_HASH observation=rejected reason=EXPORT_DELTA_NOT_EXACT\n'; negative=$((negative + 1))

  mkdir -p "$dest/input"
  cp -a --reflink=auto "$base/input/." "$dest/input/"
  verify_no_shared_inode "$base/input" "$dest/input" || die "SELFTEST_COPY_SHARED_INODE"
  printf 'STAGER_SELFTEST control=copy_inode observation=independent\n'; positive=$((positive + 1))
  rm -f -- "$dest/input/data"
  ln "$base/input/data" "$dest/input/data"
  if verify_no_shared_inode "$base/input" "$dest/input"; then die "SELFTEST_ACCEPTED_HARDLINK"; fi
  printf 'STAGER_SELFTEST inject=HARDLINKED_BASE_INPUT observation=rejected reason=BASE_INODE_SHARED\n'; negative=$((negative + 1))

  manifest_tree "$base" "$before"
  printf 'mutated\n' >> "$base/input/data"
  manifest_tree "$base" "$after"
  if cmp -s "$before" "$after"; then die "SELFTEST_MISSED_BASE_MUTATION"; fi
  printf 'STAGER_SELFTEST inject=PRIVATE_BASE_FILE_REPLACED observation=detected reason=BASE_SEAL_CHANGED\n'; negative=$((negative + 1))

  printf '{"pending":[],"dump_upper_bound_formula_bytes":17179869185}\n' > "$evidence/offline/resource_card.json"
  free=999999999999; input_bytes=1
  if resource_check "$evidence/offline/resource_card.json" "$free" "$input_bytes"; then die "SELFTEST_ACCEPTED_STORAGE_CAP_EXCESS"; fi
  printf 'STAGER_SELFTEST inject=STORAGE_CAP_EXCESS observation=rejected reason=RESOURCE_LIMIT\n'; negative=$((negative + 1))
  printf '{"pending":[],"dump_upper_bound_formula_bytes":100}\n' > "$evidence/offline/resource_card.json"
  if resource_check "$evidence/offline/resource_card.json" 50 1; then die "SELFTEST_ACCEPTED_FREE_SPACE_SHORTAGE"; fi
  printf 'STAGER_SELFTEST inject=FREE_SPACE_SHORTAGE observation=rejected reason=RESOURCE_LIMIT\n'; negative=$((negative + 1))
  printf '{"pending":[],"dump_upper_bound_formula_bytes":17179869000}\n' > "$evidence/offline/resource_card.json"
  if resource_check "$evidence/offline/resource_card.json" 999999999999 1000; then die "SELFTEST_ACCEPTED_NEW_TOTAL_CAP_EXCESS"; fi
  printf 'STAGER_SELFTEST inject=NEW_TOTAL_STORAGE_CAP_EXCESS observation=rejected reason=RESOURCE_LIMIT\n'; negative=$((negative + 1))
  printf '{"pending":[],"dump_upper_bound_formula_bytes":9223372036854775807}\n' > "$evidence/offline/resource_card.json"
  if resource_check "$evidence/offline/resource_card.json" 999999999999999999999 1; then die "SELFTEST_ACCEPTED_FIXED_WIDTH_OVERFLOW"; fi
  printf 'STAGER_SELFTEST review=R5 inject=FIXED_WIDTH_SUM_OVERFLOW observed=FAIL reason=RESOURCE_VALUE_OUT_OF_RANGE\n'; negative=$((negative + 1))
  printf '{"pending":[],"dump_upper_bound_formula_bytes":999999999999999999999999999999999999999}\n' > "$evidence/offline/resource_card.json"
  if resource_check "$evidence/offline/resource_card.json" 999999999999999999999999999999999999999 1; then die "SELFTEST_ACCEPTED_ARBITRARY_PRECISION_OVERSIZE"; fi
  printf 'STAGER_SELFTEST review=R5 inject=ARBITRARY_PRECISION_OVERSIZE observed=FAIL reason=RESOURCE_VALUE_OUT_OF_RANGE\n'; negative=$((negative + 1))
  printf '{"pending":[],"dump_upper_bound_formula_bytes":-1}\n' > "$evidence/offline/resource_card.json"
  if resource_check "$evidence/offline/resource_card.json" 1000 1; then die "SELFTEST_ACCEPTED_NEGATIVE_ESTIMATE"; fi
  printf 'STAGER_SELFTEST review=R5 inject=NEGATIVE_ESTIMATE observed=FAIL reason=RESOURCE_VALUE_INVALID\n'; negative=$((negative + 1))
  printf '{"pending":[],"dump_upper_bound_formula_bytes":0}\n' > "$evidence/offline/resource_card.json"
  reserved_output="$scratch/must-not-exist"
  if resource_check "$evidence/offline/resource_card.json" 999999999999 0 $((STORAGE_CAP_BYTES - 1)) 2; then
    printf 'created-after-check\n' > "$reserved_output"
  fi
  [[ ! -e "$reserved_output" ]] || die "SELFTEST_OUTPUT_CREATED_AFTER_CAP_REJECTION"
  printf 'STAGER_SELFTEST inject=POST_CHECK_OUTPUT observation=not_created evidence_before=%d planned=2 cap=%d\n' $((STORAGE_CAP_BYTES - 1)) "$STORAGE_CAP_BYTES"; negative=$((negative + 1))
  printf '{"pending":[],"dump_upper_bound_formula_bytes":1}\n' > "$evidence/offline/resource_card.json"
  if resource_check "$evidence/offline/resource_card.json" 999999999999 2 $((STORAGE_CAP_BYTES - 5)) 3; then die "SELFTEST_ACCEPTED_COMBINED_SIMULTANEOUS_OCCUPANCY"; fi
  printf 'STAGER_SELFTEST inject=COMBINED_SIMULTANEOUS_OCCUPANCY observation=rejected estimate=1 input=2 evidence=%d temporary=3 cap=%d\n' $((STORAGE_CAP_BYTES - 5)) "$STORAGE_CAP_BYTES"; negative=$((negative + 1))

  valid_expectations='{"schema":"CMFGEN_TWIN_MAP_REV4_V1","shell_count":2,"pending":["final_publication_generation_link"],"base_e_z":{"cell_count":2,"M_Z_interval":["0","1"]},"base_self_comparison":{"paired_sign_counts":{"improved":0,"worsened":0,"representation_unresolved":2}},"reference_top_bin_semantics":{"ca":{"value":"CUMULATIVE","source":"fixture"},"co":{"value":"CUMULATIVE","source":"fixture"},"fe":{"value":"CUMULATIVE","source":"fixture"},"ni":{"value":"CUMULATIVE","source":"fixture"},"si":{"value":"CUMULATIVE","source":"fixture"},"s":{"value":"CUMULATIVE","source":"fixture"}},"mass_convention":{"production":"sealed_model_atom_masses.csv","comparison":{"Fe":"56","Co":"56","Ni":"56","other":"sealed_model_atom_masses.csv"}}}'
  printf '%s\n' "$valid_expectations" > "$scratch/deferred-expectations.json"
  printf '{"pending":["actual_final_publication_manifest"],"manifest_relative_path":"work/cmfgen_final_publication.json"}\n' > "$scratch/deferred-writer.json"
  observation_prerequisites "$scratch/deferred-expectations.json" "$scratch/deferred-writer.json" || die "SELFTEST_DEFERRED_PUBLICATION_LINK_REJECTED"
  printf 'STAGER_SELFTEST control=RUN_AFTER_PUBLICATION_LINK observation=deferred_not_prerequisite\n'; positive=$((positive + 1))
  printf '%s\n' "${valid_expectations/\"final_publication_generation_link\"/\"source_identity\"}" > "$scratch/deferred-expectations.json"
  if observation_prerequisites "$scratch/deferred-expectations.json" "$scratch/deferred-writer.json"; then die "SELFTEST_ACCEPTED_REAL_PREREQUISITE_GAP"; fi
  printf 'STAGER_SELFTEST inject=EXECUTION_PREREQUISITE_GAP observation=rejected reason=source_identity\n'; negative=$((negative + 1))
  printf '%s\n' "${valid_expectations/\"source\":\"fixture\"/\"source\":null}" > "$scratch/deferred-expectations.json"
  if observation_prerequisites "$scratch/deferred-expectations.json" "$scratch/deferred-writer.json"; then die "SELFTEST_ACCEPTED_UNSOURCED_TOP_BIN_SEMANTICS"; fi
  printf 'STAGER_SELFTEST inject=TOP_BIN_SEMANTICS_SOURCE_MISSING observation=rejected reason=INPUT_PENDING\n'; negative=$((negative + 1))

  job="$scratch/job.slurm"; staged_job="$scratch/job.staged.slurm"
  printf '%s\n' '#!/usr/bin/env bash' '#SBATCH --partition=a100' '#SBATCH --gres=gpu:1' '#SBATCH --cpus-per-task=64' '#SBATCH --time=2-00:00:00' 'readonly GPU_PROFILE=fixture' '[[ "${#dproxy_visible_tokens[@]}" -eq 2 ]]' > "$job"
  rewrite_job_resources "$job" "$staged_job"
  grep -qx '#SBATCH --partition=h200' "$staged_job" && grep -qx '#SBATCH --gres=gpu:1' "$staged_job" && grep -qx 'readonly GPU_PROFILE=h200x1' "$staged_job" && grep -qx '#SBATCH --cpus-per-task=32' "$staged_job" && grep -qx '#SBATCH --time=04:00:00' "$staged_job" || die "SELFTEST_H200_RUNNER_LIMITS_NOT_ENFORCED"
  local alt_job="$scratch/job.alt.staged.slurm"
  rewrite_job_resources "$job" "$alt_job" h100-a100x2
  grep -qx '#SBATCH --partition=h100,a100' "$alt_job" && grep -qx '#SBATCH --gres=gpu:2' "$alt_job" && grep -qx 'readonly GPU_PROFILE=h100-a100x2' "$alt_job" || die "SELFTEST_ALT_RUNNER_LIMITS_NOT_ENFORCED"
  if rewrite_job_resources "$job" "$scratch/must-not-exist-job" invalid-profile >/dev/null 2>&1; then die "SELFTEST_INVALID_GPU_PROFILE_ACCEPTED"; fi
  [[ ! -e "$scratch/must-not-exist-job" ]] || die "SELFTEST_INVALID_GPU_PROFILE_WROTE_OUTPUT"
  printf 'STAGER_SELFTEST control=runner_limits profiles=h200x1,h100-a100x2 observation=enforced\n'; positive=$((positive + 1)); negative=$((negative + 1))
  mkdir "$scratch/tool-bin"
  printf '#!/usr/bin/env bash\nprintf "ELF file: fixture.sm_90.cubin\\n"\n' > "$scratch/tool-bin/cuobjdump"; chmod +x "$scratch/tool-bin/cuobjdump"
  PATH="$scratch/tool-bin:$PATH" binary_sm90_check "$base/input/data" >/dev/null || die "SELFTEST_SM90_BINARY_REJECTED"
  printf 'STAGER_SELFTEST control=binary_sm90_compatibility observation=accepted\n'; positive=$((positive + 1))
  printf '#!/usr/bin/env bash\nprintf "ELF file: fixture.sm_80.cubin\\n"\n' > "$scratch/tool-bin/cuobjdump"
  if PATH="$scratch/tool-bin:$PATH" binary_sm90_check "$base/input/data" >/dev/null; then die "SELFTEST_ACCEPTED_SM80_ONLY_BINARY"; fi
  printf 'STAGER_SELFTEST inject=SM80_ONLY_BINARY observation=rejected reason=SM90_COMPATIBILITY_MISSING\n'; negative=$((negative + 1))

  local tx_base="$scratch/tx-base" tx_evidence="$scratch/tx-evidence" tx_run="$scratch/tx-run"
  mkdir -p "$tx_base/input" "$tx_evidence"/{inputs,offline,run,result}
  printf 'payload\n' > "$tx_base/input/data"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$tx_base/input/lumina_cuda"; chmod +x "$tx_base/input/lumina_cuda"
  sha256sum "$tx_base/input/lumina_cuda" | awk '{print $1}' > "$tx_base/input/binary.sha256"
  printf '%s\n' "declare -x LUMINA_MODEL_DIR=\"$tx_base/input/model\"" 'declare -x LUMINA_FIXED_TE_PROFILE="/sealed/old.txt"' 'declare -x LUMINA_RADEQ_DIAG_TE_K="10020"' > "$tx_base/input/resolved_lumina.exports"
  printf '%s\n' '#!/usr/bin/env bash' '#SBATCH --partition=a100' '#SBATCH --gres=gpu:1' '#SBATCH --cpus-per-task=64' '#SBATCH --time=2-00:00:00' > "$tx_base/input/job.slurm"
  printf '0 18900\n' > "$tx_evidence/inputs/fixed_te_profile.txt"
  printf '{"formal_iteration_cap":128,"nlte_bf_stage_weights":"LAST_COMMITTED_NLTE"}\n' > "$tx_evidence/offline/expectations.json"
  printf '{"pending":[],"dump_upper_bound_formula_bytes":0,"future_storage_upper_bounds_bytes":{"run_logs":0,"result_and_summary":0,"reproduction_and_audit":0}}\n' > "$tx_evidence/offline/resource_card.json"
  stage_transaction "$tx_base" "$tx_evidence" "$tx_run" "$runner_template" || die "SELFTEST_NORMAL_STAGE_REJECTED"
  [[ -f "$tx_run/input/fixed_te_profile.txt" && "$(cat "$tx_run/input/fixed_te_profile.txt")" == '0 18900' ]] || die "SELFTEST_NORMAL_NEW_PROFILE_MISSING"
  [[ -x "$tx_run/input/lumina_cuda" ]] || die "SELFTEST_STAGED_BINARY_NOT_EXECUTABLE"
  grep -qx 'declare -x LUMINA_CMF_FINE_ALI="128"' "$tx_run/input/resolved_lumina.exports" || die "SELFTEST_TRANSACTION_CAP128_MISSING"
  grep -qx 'declare -x LUMINA_NLTE_BF_STAGE_WEIGHTS="LAST_COMMITTED_NLTE"' "$tx_run/input/resolved_lumina.exports" || die "SELFTEST_TRANSACTION_BF_SELECTOR_MISSING"
  grep -qx 'declare -x LUMINA_CMF_FINE_MGPU_DEVICES="1"' "$tx_run/input/resolved_lumina.exports" || die "SELFTEST_TRANSACTION_H200_MGPU_MISSING"
  grep -qx 'gpu_profile=h200x1' "$tx_run/STAGE_IDENTITY.txt" && grep -qx 'expected_gpu_count=1' "$tx_run/STAGE_IDENTITY.txt" || die "SELFTEST_TRANSACTION_GPU_IDENTITY_MISSING"
  grep -qx '#SBATCH --partition=h200' "$tx_run/input/job.slurm" && grep -qx '#SBATCH --gres=gpu:1' "$tx_run/input/job.slurm" || die "SELFTEST_TRANSACTION_H200_JOB_MISMATCH"
  "$tx_run/input/lumina_cuda" || die "SELFTEST_STAGED_BINARY_EXECUTION_FAILED"
  printf 'STAGER_SELFTEST control=NORMAL_STAGING_NEW_PROFILE observation=accepted destination_was_absent=1\n'; positive=$((positive + 1))

  local render_evidence="$scratch/render-evidence" render_run="$scratch/render-run" render_output="$scratch/render-output"
  mkdir -p "$render_evidence"/{inputs,offline,run,result}
  printf '0 18900\n' > "$render_evidence/inputs/fixed_te_profile.txt"
  printf '{"formal_iteration_cap":128,"nlte_bf_stage_weights":"LAST_COMMITTED_NLTE"}\n' > "$render_evidence/offline/expectations.json"
  render_artifacts "$tx_base" "$render_evidence" "$render_run" "$render_output" >/dev/null || die "SELFTEST_RENDER_CAP128_REJECTED"
  grep -qx 'declare -x LUMINA_CMF_FINE_ALI="128"' "$render_output/resolved_lumina.exports" || die "SELFTEST_RENDER_CAP128_MISSING"
  grep -qx 'declare -x LUMINA_NLTE_BF_STAGE_WEIGHTS="LAST_COMMITTED_NLTE"' "$render_output/resolved_lumina.exports" || die "SELFTEST_RENDER_BF_SELECTOR_MISSING"
  printf 'STAGER_SELFTEST control=RENDER_REGISTERED_CAP128 observation=accepted\n'; positive=$((positive + 1))
  local render_alt_run="$scratch/render-alt-run" render_alt_output="$scratch/render-alt-output"
  render_artifacts "$tx_base" "$render_evidence" "$render_alt_run" "$render_alt_output" h100-a100x2 >/dev/null || die "SELFTEST_RENDER_ALT_PROFILE_REJECTED"
  grep -qx '#SBATCH --partition=h100,a100' "$render_alt_output/job.slurm" && grep -qx '#SBATCH --gres=gpu:2' "$render_alt_output/job.slurm" && grep -qx 'readonly GPU_PROFILE=h100-a100x2' "$render_alt_output/job.slurm" || die "SELFTEST_RENDER_ALT_JOB_MISMATCH"
  grep -qx 'declare -x LUMINA_CMF_FINE_MGPU_DEVICES="2"' "$render_alt_output/resolved_lumina.exports" || die "SELFTEST_RENDER_ALT_EXPORT_MISMATCH"
  printf 'STAGER_SELFTEST control=RENDER_GPU_PROFILE profile=h100-a100x2 observation=accepted\n'; positive=$((positive + 1))

  local cap_evidence="$scratch/cap-evidence" cap_run="$scratch/cap-run"
  mkdir -p "$cap_evidence"/{inputs,offline,run,result}
  printf '0 18900\n' > "$cap_evidence/inputs/fixed_te_profile.txt"
  printf '{}\n' > "$cap_evidence/offline/expectations.json"
  printf '{"pending":[],"dump_upper_bound_formula_bytes":17179869184,"future_storage_upper_bounds_bytes":{"run_logs":0,"result_and_summary":0,"reproduction_and_audit":0}}\n' > "$cap_evidence/offline/resource_card.json"
  if stage_transaction "$tx_base" "$cap_evidence" "$cap_run" "$runner_template" >/dev/null 2>&1; then die "SELFTEST_ACCEPTED_PREWRITE_RESERVATION_EXCESS"; fi
  [[ ! -e "$cap_run" ]] || die "SELFTEST_PREWRITE_RESERVATION_CREATED_RUN"
  printf 'STAGER_SELFTEST inject=REVIEW_MANIFEST_BEFORE_RESOURCE_CHECK observation=rejected first_write=none run_entry=absent\n'; negative=$((negative + 1))
  local future_evidence="$scratch/future-evidence" future_run="$scratch/future-run"
  mkdir -p "$future_evidence"/{inputs,offline,run,result}
  printf '0 18900\n' > "$future_evidence/inputs/fixed_te_profile.txt"
  printf '{}\n' > "$future_evidence/offline/expectations.json"
  printf '{"pending":[],"dump_upper_bound_formula_bytes":0,"future_storage_upper_bounds_bytes":{"run_logs":0,"result_and_summary":0,"reproduction_and_audit":17179869184}}\n' > "$future_evidence/offline/resource_card.json"
  if stage_transaction "$tx_base" "$future_evidence" "$future_run" "$runner_template" >/dev/null 2>&1; then die "SELFTEST_ACCEPTED_FUTURE_AUDIT_RESERVATION_EXCESS"; fi
  [[ ! -e "$future_run" ]] || die "SELFTEST_FUTURE_AUDIT_RESERVATION_CREATED_RUN"
  printf 'STAGER_SELFTEST inject=FUTURE_REPRODUCTION_AUDIT_RESERVATION observation=rejected first_write=none run_entry=absent\n'; negative=$((negative + 1))

  local protected_parent="$scratch/protected-stage-parent" swapped_parent="$scratch/swapped-stage-parent"
  mkdir "$protected_parent"; ln -s "$protected_parent" "$swapped_parent"
  if stage_transaction "$tx_base" "$cap_evidence" "$swapped_parent/new-run" "$runner_template" >/dev/null 2>&1; then die "SELFTEST_ACCEPTED_STAGE_PARENT_SWAP"; fi
  [[ ! -e "$protected_parent/new-run" ]] || die "SELFTEST_STAGE_PARENT_SWAP_CREATED_ENTRY"
  printf 'STAGER_SELFTEST inject=RUN_PARENT_SYMLINK_SWAP observation=rejected first_write=none protected_entries=unchanged\n'; negative=$((negative + 1))
  mkdir "$scratch/nested-stage-parent"; ln -s "$protected_parent" "$scratch/nested-stage-parent/one"
  if stage_transaction "$tx_base" "$cap_evidence" "$scratch/nested-stage-parent/one/new-run" "$runner_template" >/dev/null 2>&1; then die "SELFTEST_ACCEPTED_NESTED_STAGE_PARENT_LINK"; fi
  [[ ! -e "$protected_parent/new-run" ]] || die "SELFTEST_NESTED_STAGE_PARENT_LINK_CREATED_ENTRY"
  printf 'STAGER_SELFTEST inject=NESTED_RUN_PARENT_LINK observation=rejected first_write=none protected_entries=unchanged\n'; negative=$((negative + 1))

  printf 'CMFGEN_TWIN_MAP_STAGER_SELFTEST_PASS positive=%d negative=%d\n' "$positive" "$negative"
  rm -rf -- "$scratch"
  trap - EXIT
}

render_artifacts() {
  local base="$1" evidence="$2" run="$3" output="$4" gpu_profile="${5:-$DEFAULT_GPU_PROFILE}" runner_template staged_exports formal_iteration_cap nlte_bf_stage_weights
  validate_gpu_profile "$gpu_profile" || die "GPU_PROFILE_INVALID value=$gpu_profile"
  runner_template="$(dirname "$(realpath -e -- "$0")")/run_cmfgen_twin_map.sh"
  require_safe_base "$base" || die "BASE_NOT_SAFE_OR_SEALED"
  canonical_no_symlink "$evidence" existing && [[ -d "$evidence" ]] || die "EVIDENCE_ROOT_UNSAFE"
  require_new_root "$run" "$base" "$evidence" || die "RUN_NOT_NEW_INDEPENDENT_ROOT"
  canonical_no_symlink "$output" new || die "RENDER_OUTPUT_NOT_NEW_SAFE"
  [[ -f "$evidence/inputs/fixed_te_profile.txt" && ! -L "$evidence/inputs/fixed_te_profile.txt" ]] || die "FIXED_TE_PROFILE_UNSAFE"
  [[ -f "$evidence/offline/expectations.json" && ! -L "$evidence/offline/expectations.json" ]] || die "EXPECTATIONS_UNSAFE"
  [[ -f "$runner_template" && ! -L "$runner_template" ]] || die "DEDICATED_RUNNER_UNSAFE"
  formal_iteration_cap="$(registered_formal_iteration_cap "$evidence/offline/expectations.json")" || die "FORMAL_ITERATION_CAP_INVALID"
  nlte_bf_stage_weights="$(registered_nlte_bf_stage_weights "$evidence/offline/expectations.json")" || die "NLTE_BF_STAGE_WEIGHTS_INVALID"
  mkdir -- "$output"
  staged_exports="$output/resolved_lumina.exports"
  rewrite_exports "$base/input/resolved_lumina.exports" "$base" "$run" "$staged_exports" "$formal_iteration_cap" "$gpu_profile" "$nlte_bf_stage_weights"
  rewrite_job_resources "$runner_template" "$output/job.slurm" "$gpu_profile"
  cp -- "$evidence/inputs/fixed_te_profile.txt" "$output/fixed_te_profile.txt"
  verify_export_delta "$base/input/resolved_lumina.exports" "$staged_exports" "$base" "$run" "$formal_iteration_cap" "$gpu_profile" "$nlte_bf_stage_weights" || die "RENDERED_EXPORT_DELTA_INVALID"
  printf 'CMFGEN_TWIN_MAP_RENDER_COMPLETE job=%s exports=%s profile=%s run=%s\n' \
    "$output/job.slurm" "$staged_exports" "$output/fixed_te_profile.txt" "$run"
}

gpu_profile="$DEFAULT_GPU_PROFILE"
if [[ $# -ge 1 && "$1" == --gpu-profile ]]; then
  [[ $# -ge 2 ]] || die "GPU_PROFILE_VALUE_MISSING"
  gpu_profile="$2"; shift 2
  validate_gpu_profile "$gpu_profile" || die "GPU_PROFILE_INVALID value=$gpu_profile"
fi
if [[ $# -eq 1 && "$1" == --selftest ]]; then
  stager_selftest
  exit 0
fi
if [[ $# -eq 5 && "$1" == --render ]]; then
  render_artifacts "$2" "$3" "$4" "$5" "$gpu_profile"
  exit 0
fi

[[ $# -eq 5 ]] || die "USAGE expected='BASE EVID RUN RUN_BINARY RUN_BUILD_IDENTITY_JSON'"
base="$1"; evidence="$2"; run="$3"; run_binary="$4"; run_build_identity="$5"
[[ "$base" = /* && "$evidence" = /* && "$run" = /* && "$run_binary" = /* && "$run_build_identity" = /* ]] || die "PATHS_MUST_BE_ABSOLUTE"
require_safe_base "$base" || die "BASE_NOT_SAFE_OR_SEALED"
require_new_root "$run" "$base" "$evidence" || die "RUN_NOT_NEW_INDEPENDENT_ROOT"
require_evidence_root "$evidence" "$base" "$run" || die "EVIDENCE_NOT_SAFE_INDEPENDENT_ROOT"
for path in \
  "$base/input/lumina_cuda" \
  "$base/input/resolved_lumina.exports" \
  "$evidence/inputs/identity.json" \
  "$evidence/inputs/fixed_te_profile.txt" \
  "$evidence/offline/expectations.json" \
  "$evidence/offline/writer_contract.json" \
  "$evidence/offline/resource_card.json"; do
  [[ -f "$path" && ! -L "$path" ]] || die "REQUIRED_INPUT_UNSAFE path=$path"
done
[[ -f "$run_binary" && ! -L "$run_binary" && -x "$run_binary" ]] || die "RUN_BINARY_UNSAFE"
[[ -f "$run_build_identity" && ! -L "$run_build_identity" ]] || die "RUN_BUILD_IDENTITY_UNSAFE"
python3 -B - "$evidence/inputs/identity.json" "$run_build_identity" "$run_binary" <<'PY' || die "REGISTERED_RUN_BUILD_IDENTITY_MISMATCH"
import hashlib, json, sys
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()
registered = json.load(open(sys.argv[1], encoding="utf-8")).get("run_build")
provided = json.load(open(sys.argv[2], encoding="utf-8"))
if not isinstance(registered, dict) or registered.get("identity_sha256") != digest(sys.argv[2]): raise SystemExit(1)
if provided.get("binary", {}).get("path") != sys.argv[3] or provided.get("binary", {}).get("sha256") != digest(sys.argv[3]): raise SystemExit(1)
PY
observation_prerequisites "$evidence/offline/expectations.json" "$evidence/offline/writer_contract.json" || die "INPUT_PENDING_OBSERVATIONS"
actual_binary="$(sha256sum "$base/input/lumina_cuda" | awk '{print $1}')"
[[ "$actual_binary" == "$EXPECTED_BINARY_SHA256" ]] || die "BINARY_IDENTITY_MISMATCH"
binary_sm90_check "$base/input/lumina_cuda" || die "INPUT_PENDING_BINARY_SM90_COMPATIBILITY"
registered_binary="$(json_field "$evidence/inputs/identity.json" base_binary.sha256)"
[[ "$registered_binary" == "$actual_binary" ]] || die "STALE_IDENTITY"
registered_exports="$(json_field "$evidence/inputs/identity.json" effective_exports.sha256)"
actual_exports="$(sha256sum "$base/input/resolved_lumina.exports" | awk '{print $1}')"
[[ "$registered_exports" == "$actual_exports" ]] || die "STALE_EXPORTS_HASH"

runner_template="$(dirname "$(realpath -e -- "$0")")/run_cmfgen_twin_map.sh"
[[ -f "$runner_template" && ! -L "$runner_template" ]] || die "DEDICATED_RUNNER_UNSAFE"
stage_transaction "$base" "$evidence" "$run" "$runner_template" "$run_binary" "$run_build_identity" "$gpu_profile" || die "STAGE_TRANSACTION_FAILED"
