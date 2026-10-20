#!/usr/bin/env bash
# Rebuild an isolated Lumina atomic deck from one sealed CMFGEN run's active atoms.
# Submit with EXPECT_* SHA256 values for source and immutable CMFGEN inputs.
#SBATCH --job-name=lumina-r1deck
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/r1-modelspec-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/r1-modelspec-%j.err

set -euo pipefail
umask 027

readonly REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
readonly PYTHON=/home/kjhan/miniconda3/bin/python3
readonly RUNS=/gpfs/kjhan/cmfgen_runs
readonly CMFGEN_RUN="$RUNS/toy06_19p48d_modern"
readonly OUTPUT="$RUNS/lumina_model_spec_${SLURM_JOB_ID}"
readonly HYD_L_DATA="$REPO/data/atomic/cmfgen/HYD/I/5dec96/hyd_l_data.dat"
readonly GBF_N_DATA="$REPO/data/atomic/cmfgen/HYD/I/5dec96/gbf_n_data.dat"
readonly PHOT_SOURCE=/gpfs/kjhan/cmfgen_src/cur_cmf
readonly PHOT_BUILD="$RUNS/lumina_photsmooth_${SLURM_JOB_ID}"

[[ ${SLURM_CPUS_PER_TASK:-} == 10 ]] || {
  echo "expected 10 allocated CPUs" >&2
  exit 20
}
[[ ${CMFGEN_EXACT_HYD:-} == 1 ]] || {
  echo "CMFGEN_EXACT_HYD=1 is required for comparison-deck generation" >&2
  exit 20
}
export CMFGEN_EXACT_HYD=1
[[ -x "$PYTHON" && -d "$CMFGEN_RUN" && -d "$RUNS/slurm_logs" ]] || {
  echo "Python, sealed CMFGEN run, or Slurm log directory is missing" >&2
  exit 21
}
[[ ! -e "$OUTPUT" ]] || { echo "refusing existing output: $OUTPUT" >&2; exit 22; }

module purge
module load gnu13
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

assert_sha() {
  local expected="$1" path="$2" actual
  [[ -n "$expected" && -f "$path" ]] || {
    echo "missing expected checksum or input: $path" >&2
    exit 23
  }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "sealed input changed: $path expected=$expected actual=$actual" >&2
    exit 24
  }
}

assert_sha "${EXPECT_EXPAND:-}" "$REPO/scripts/expand_atomic_data_cmfgen.py"
assert_sha "${EXPECT_PARSER:-}" "$REPO/scripts/cmfgen_parser.py"
assert_sha "${EXPECT_DRIVER:-}" "$REPO/scripts/deck_regen_r1_vintage_driver.py"
assert_sha "${EXPECT_SMOOTHER:-}" "$REPO/scripts/cmfgen_phot_smoothing.py"
assert_sha "${EXPECT_RECOMB_POLICY:-}" "$REPO/scripts/cmfgen_recombination_policy.py"
assert_sha "${EXPECT_MODEL:-}" "$CMFGEN_RUN/MODEL"
assert_sha "${EXPECT_SM_PHOT:-}" "$PHOT_SOURCE/tools/sm_phot_v3.f"
assert_sha "${EXPECT_CUT_POINTS:-}" "$PHOT_SOURCE/newsubs/cut_points_v3.f"
assert_sha "${EXPECT_SET_KIND:-}" "$PHOT_SOURCE/tools/set_kind_module.f"
assert_sha "${EXPECT_CONTROL:-}" "$PHOT_SOURCE/new_main/mod_subs/rd_control_variables.f"
assert_sha "${EXPECT_LINKS:-}" "$CMFGEN_RUN/atomic_links.txt"
assert_sha "${EXPECT_MODEL_SPEC:-}" "$CMFGEN_RUN/MODEL_SPEC"
assert_sha "${EXPECT_LEVEL_MAP:-}" "$CMFGEN_RUN/LEVEL_SL_STEQ_LINKS"
assert_sha "${EXPECT_SIII_PHOT:-}" "$CMFGEN_RUN/atomic_local/SUL/III/19apr23/phot_data_A"
assert_sha "${EXPECT_COII_PHOT:-}" "$CMFGEN_RUN/atomic_local/COB/II/19apr23/phot_data_A"
assert_sha "${EXPECT_HYD_L_DATA:-}" "$HYD_L_DATA"
assert_sha "${EXPECT_GBF_N_DATA:-}" "$GBF_N_DATA"

[[ ! -e "$PHOT_BUILD" ]] || { echo "refusing existing PHOT build" >&2; exit 22; }
mkdir "$PHOT_BUILD"
(
  cd "$PHOT_BUILD"
  gfortran -O2 -fPIC -shared -ffixed-line-length-none \
    "$PHOT_SOURCE/tools/set_kind_module.f" \
    "$PHOT_SOURCE/tools/sm_phot_v3.f" \
    "$PHOT_SOURCE/newsubs/cut_points_v3.f" -o libsm_phot.so
)
export CMFGEN_PHOT_SMOOTH_LIB="$PHOT_BUILD/libsm_phot.so"
export CMFGEN_PHOT_SMOOTH_SOURCE_ROOT="$PHOT_SOURCE"

export CMFGEN_LINKS="$CMFGEN_RUN/atomic_links.txt"
export CMFGEN_MODEL_SPEC="$CMFGEN_RUN/MODEL_SPEC"
export CMFGEN_LINK_FTOS=1
export CMFGEN_VINTAGE_MATCH=0
export CMFGEN_FULL_LEVELS=1
export CMFGEN_SUPER_LEVELS=1
export LUMINA_SIII_PHOT_PATH="$CMFGEN_RUN/atomic_local/SUL/III/19apr23/phot_data_A"
export LUMINA_COII_PHOT_PATH="$CMFGEN_RUN/atomic_local/COB/II/19apr23/phot_data_A"
export LUMINA_CMFGEN_OUTPUT_DECK="$OUTPUT"

"$PYTHON" "$REPO/scripts/deck_regen_r1_vintage_driver.py"

"$PYTHON" - "$OUTPUT/atomic_vintage_manifest.csv" <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

expected = {
    "Si II": (125, 69),
    "Si III": (147, 99),
    "Si IV": (61, 50),
    "S II": (322, 55),
    "S III": (256, 79),
    "S IV": (176, 65),
    "Co II": (2558, 134),
}
with open(sys.argv[1], newline="") as stream:
    rows = {row["ion"]: row for row in csv.DictReader(stream)}
for ion, (nf, ns) in expected.items():
    row = rows.get(ion)
    if row is None:
        raise SystemExit(f"missing active ion in manifest: {ion}")
    got = (int(row["model_spec_nf"]), int(row["active_superlevels"]))
    if got != (nf, ns):
        raise SystemExit(f"active atom mismatch for {ion}: got={got}, want={(nf, ns)}")
for ion, env_path in (("S III", "LUMINA_SIII_PHOT_PATH"),
                      ("Co II", "LUMINA_COII_PHOT_PATH")):
    if rows[ion]["phot_path"] != os.environ[env_path]:
        raise SystemExit(f"run-local PHOT path mismatch for {ion}")
manifest_path = Path(sys.argv[1])
identity_path = manifest_path.with_name("atomic_source_identity.json")
identity = json.loads(identity_path.read_text())
if identity["n_active_ions"] != len(rows):
    raise SystemExit("atomic source identity ion count differs from active manifest")
if any(row["selection_source"] != "links" for row in rows.values()):
    raise SystemExit("non-MODEL_SPEC ion leaked into active manifest")
identity_rows = {row["ion"]: row for row in identity["ions"]}
if set(identity_rows) != set(rows):
    raise SystemExit("atomic source identity ion set differs from active manifest")
for ion, row in rows.items():
    if (int(row["model_spec_nf"]), int(row["active_superlevels"])) != \
            (identity_rows[ion]["model_spec_nf"], identity_rows[ion]["active_superlevels"]):
        raise SystemExit(f"atomic source identity count mismatch for {ion}")
stamp = json.loads(manifest_path.with_name("DECK_PROVENANCE.json").read_text())
if stamp["cmfgen_links_sha256"] != os.environ["EXPECT_LINKS"] or \
        stamp["cmfgen_model_spec_sha256"] != os.environ["EXPECT_MODEL_SPEC"] or \
        stamp["level_sl_steq_links_sha256"] != os.environ["EXPECT_LEVEL_MAP"]:
    raise SystemExit("deck provenance does not bind the submitted CMFGEN inputs")
if stamp["env"].get("CMFGEN_EXACT_HYD") != "1":
    raise SystemExit("comparison deck was not generated with exact hydrogenic fits")
preprocessing = stamp.get("phot_preprocessing") or {}
if preprocessing.get("model", {}).get("sha256") != os.environ["EXPECT_MODEL"]:
    raise SystemExit("PHOT preprocessing does not bind the submitted MODEL")
library_path = manifest_path.with_name("cmfgen_phot_smoothing.so")
if preprocessing.get("library", {}).get("sha256") != \
        hashlib.sha256(library_path.read_bytes()).hexdigest():
    raise SystemExit("PHOT preprocessing library differs from provenance")
hyd_inputs = {Path(row["path"]).name: row["sha256"]
              for row in stamp.get("hydrogenic_fit_data", [])}
if hyd_inputs.get("hyd_l_data.dat") != os.environ["EXPECT_HYD_L_DATA"] or \
        hyd_inputs.get("gbf_n_data.dat") != os.environ["EXPECT_GBF_N_DATA"]:
    raise SystemExit("provenance does not bind the submitted HYD/GBF tables")
identity_sha = hashlib.sha256(identity_path.read_bytes()).hexdigest()
if stamp["atomic_source_identity"]["sha256"] != identity_sha:
    raise SystemExit("deck provenance source-identity checksum mismatch")
if set(stamp["active_ions"]) != set(rows):
    raise SystemExit("deck provenance ion set differs from active manifest")
channel_manifest_path = manifest_path.with_name("cmfgen_bf_channels_manifest.json")
channel_manifest = json.loads(channel_manifest_path.read_text())
channel_path = manifest_path.with_name("cmfgen_bf_channels.bin")
sigma_path = manifest_path.with_name("cmfgen_sigma_bf.bin")
channel_sha = hashlib.sha256(channel_path.read_bytes()).hexdigest()
sigma_sha = hashlib.sha256(sigma_path.read_bytes()).hexdigest()
if channel_manifest.get("schema") != "lumina-cmfgen-bf-channels-v5" or not channel_manifest.get("recombination_policy"):
    raise SystemExit("BF channel asset lacks the v5 source recombination policy")
if channel_manifest.get("channel_bin_sha256") != channel_sha or \
        channel_manifest.get("sigma_bin_sha256") != sigma_sha:
    raise SystemExit("BF channel manifest does not match its binary assets")
with channel_path.open("rb") as stream:
    header = stream.read(36)
if len(header) != 36:
    raise SystemExit("BF channel asset has a truncated header")
magic, version, n_levels, n_bins, n_channels, nu_min, nu_max = \
    struct.unpack("<IIiiidd", header)
if magic != 0x434D4643 or version != 5 or \
        n_channels != channel_manifest.get("n_channels") or \
        n_levels != channel_manifest.get("n_levels") or \
        n_bins != channel_manifest.get("n_freq_bins"):
    raise SystemExit(
        f"BF channel binary/manifest schema mismatch: version={version}, "
        f"levels={n_levels}/{channel_manifest.get('n_levels')}, "
        f"bins={n_bins}/{channel_manifest.get('n_freq_bins')}, "
        f"channels={n_channels}/{channel_manifest.get('n_channels')}"
    )
print("ACTIVE_ATOM_MANIFEST_CHECK PASS")
PY

for name in atomic_data_cmfgen.h5 atomic_vintage_manifest.csv \
            atomic_source_identity.json DECK_PROVENANCE.json \
            cmfgen_sigma_bf.bin cmfgen_bf_channels.bin \
            cmfgen_bf_channels_manifest.json line_list.csv levels.csv; do
  [[ -s "$OUTPUT/$name" ]] || { echo "required output missing/empty: $name" >&2; exit 25; }
done

{
  printf 'bundle=CMFGEN MODEL_SPEC active-atom deck staging\n'
  printf 'job_id=%s\noutput=%s\ninput_run=%s\n' "$SLURM_JOB_ID" "$OUTPUT" "$CMFGEN_RUN"
  printf 'host=%s\nsource_git_head=%s\n' "$(hostname)" "$(git -C "$REPO" rev-parse HEAD)"
  printf 'resources=partition:a10,cpus:10,mem:100G,gpu:0,time:01:00:00,node_unpinned\n'
  printf 'python=%s\n' "$("$PYTHON" --version 2>&1)"
  printf 'CMFGEN_LINKS=%s\nCMFGEN_MODEL_SPEC=%s\n' "$CMFGEN_LINKS" "$CMFGEN_MODEL_SPEC"
  printf 'LEVEL_SL_STEQ_LINKS=%s\n' "$CMFGEN_RUN/LEVEL_SL_STEQ_LINKS"
  printf 'linked_ftos=1\nfull_levels=1\nsuper_levels=1\nvintage_match=0\n'
  printf 'CMFGEN_EXACT_HYD=%s\nHYD_L_DATA=%s\nGBF_N_DATA=%s\n' \
    "$CMFGEN_EXACT_HYD" "$HYD_L_DATA" "$GBF_N_DATA"
} > "$OUTPUT/regen_runtime.txt"

sha256sum \
  "$REPO/scripts/expand_atomic_data_cmfgen.py" \
  "$REPO/scripts/cmfgen_parser.py" \
  "$REPO/scripts/cmfgen_phot_smoothing.py" \
  "$REPO/scripts/deck_regen_r1_vintage_driver.py" \
  "$CMFGEN_RUN/MODEL" \
  "$PHOT_SOURCE/tools/set_kind_module.f" \
  "$PHOT_SOURCE/tools/sm_phot_v3.f" \
  "$PHOT_SOURCE/newsubs/cut_points_v3.f" \
  "$PHOT_SOURCE/new_main/mod_subs/rd_control_variables.f" \
  "$CMFGEN_RUN/atomic_links.txt" \
  "$CMFGEN_RUN/MODEL_SPEC" \
  "$CMFGEN_RUN/LEVEL_SL_STEQ_LINKS" \
  "$CMFGEN_RUN/atomic_local/SUL/III/19apr23/phot_data_A" \
  "$CMFGEN_RUN/atomic_local/COB/II/19apr23/phot_data_A" \
  "$HYD_L_DATA" "$GBF_N_DATA" \
  "$REPO/scripts/submit_cmfgen_model_spec_deck.slurm" \
  > "$OUTPUT/regen_inputs.sha256"
sha256sum \
  "$OUTPUT/atomic_vintage_manifest.csv" \
  "$OUTPUT/atomic_source_identity.json" \
  "$OUTPUT/DECK_PROVENANCE.json" \
  "$OUTPUT/cmfgen_phot_smoothing.so" \
  "$OUTPUT/atomic_data_cmfgen.h5" \
  "$OUTPUT/cmfgen_sigma_bf.bin" \
  "$OUTPUT/cmfgen_bf_channels.bin" \
  "$OUTPUT/cmfgen_bf_channels_manifest.json" \
  "$OUTPUT/line_list.csv" "$OUTPUT/levels.csv" \
  > "$OUTPUT/regen_outputs.sha256"

printf 'DECK_REGEN_COMPLETE output=%s\n' "$OUTPUT"
