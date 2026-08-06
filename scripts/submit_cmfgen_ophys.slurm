#!/usr/bin/env bash
# Submit with OPHYS_MODE=solve|capture|formal|package (default: solve).
# This file is a batch payload, not a login-node runner.
#SBATCH --job-name=cmf-ophys
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/ophys-%x-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/ophys-%x-%j.err
set -euo pipefail
umask 027

readonly RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys
readonly CMFGEN=/gpfs/kjhan/cmfgen_src/cur_cmf/exe/cmfgen_dev.exe
readonly CMF_FLUX=/gpfs/kjhan/cmfgen_src/cur_cmf/exe/cmf_flux.exe
readonly MODE=${OPHYS_MODE:-solve}

[[ ${SLURM_CPUS_PER_TASK:-} == 16 ]] || { echo "OMP contract requires cpus-per-task=16" >&2; exit 20; }
[[ -d "$RUN" ]] || { echo "run directory missing: $RUN" >&2; exit 21; }
[[ -x "$CMFGEN" && -x "$CMF_FLUX" ]] || { echo "CMFGEN executable missing" >&2; exit 22; }
mkdir -p /gpfs/kjhan/cmfgen_runs/slurm_logs "$RUN/seq_logs/captures"

export OMP_NUM_THREADS=16
export OMP_DYNAMIC=FALSE
export OMP_STACKSIZE=512M
export OMP_PROC_BIND=close
export OMP_PLACES=cores
ulimit -s unlimited 2>/dev/null || true
cd "$RUN"

{
  echo "UTC=$(date -u +%FT%TZ)"
  echo "HOST=$(hostname)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
  echo "MODE=${MODE}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  sha256sum "$CMFGEN" "$CMF_FLUX"
  grep -E '\[(FIX_T|FIX_T_AUTO|FIX_NE|FIX_IMP|MAX_LIN|MAX_LAM|MAX_dT|LAM_VAL|NUM_LAM|DO_NG|CHK_NG|WRITE_RATES|WRITE_JH)\]' VADAT
} >"seq_logs/${MODE}_${SLURM_JOB_ID}.info"

assert_key() {
  local value=$1 key=$2 file=$3
  awk -v want="$value" -v key="[$key]" 'index($0,key) && $1 == want {ok++} END {exit ok == 1 ? 0 : 1}' "$file"
}

assert_common() {
  assert_key F FIX_T VADAT
  assert_key T WRITE_RATES VADAT
  assert_key T WRITE_JH VADAT
  [[ ! -e XzV_IN ]] || { echo "XzV_IN is forbidden" >&2; return 1; }
}

archive_formal_outputs() {
  local archive="seq_logs/stale_formal_${SLURM_JOB_ID}"
  local found=0 name
  for name in CHI_DATA CHI_DATA_INFO ETA_DATA ETA_DATA_INFO OBSFLUX OBS_FREQ \
              OUT_FLUX OUT_PARAMS OBSFRAME FLUX_FILE PROVENANCE_FORMAL.json; do
    if [[ -e "$name" ]]; then
      if (( found == 0 )); then mkdir -p "$archive"; found=1; fi
      mv -- "$name" "$archive/"
    fi
  done
}

run_cmfgen() {
  local tag=$1
  "$CMFGEN" < /dev/null >"seq_logs/${tag}.OUTGEN.stdout" 2>"seq_logs/${tag}.OUTGEN.stderr"
  [[ -s OUTGEN ]] || { echo "CMFGEN produced no OUTGEN" >&2; return 1; }
  cp -a -- OUTGEN "seq_logs/${tag}.OUTGEN"
}

capture_current() {
  local dest=$1 name
  mkdir -p "$dest"
  for name in EDDFACTOR EDDFACTOR_INFO RVTJ NETRATE TOTRATE GENCOOL LINEHEAT \
              JH_AT_CURRENT_TIME JH_AT_CURRENT_TIME_INFO OUTGEN POP* *OUT *PRRR; do
    [[ -e "$name" ]] && cp -a -- "$name" "$dest/"
  done
  sha256sum "$dest"/* >"$dest/SHA256SUMS"
}

case "$MODE" in
  solve)
    assert_common
    cat >IN_ITS.tmp <<'EOF'
80           [NUM_ITS]                  ! O-PHYS free-T continuation/extension budget
F            [DO_LAM_IT]                ! request a full coupled solve first
EOF
    mv -- IN_ITS.tmp IN_ITS
    assert_key 80 NUM_ITS IN_ITS
    assert_key F DO_LAM_IT IN_ITS
    archive_formal_outputs
    run_cmfgen "solve_${SLURM_JOB_ID}"
    ;;

  capture)
    assert_common
    archive_formal_outputs
    cat >IN_ITS.tmp <<'EOF'
1            [NUM_ITS]                  ! O-PHYS independently archived capture
F            [DO_LAM_IT]                ! full coupled solve requested
EOF
    mv -- IN_ITS.tmp IN_ITS
    for serial in 1 2 3 4; do
      tag="capture_${SLURM_JOB_ID}_${serial}"
      run_cmfgen "$tag"
      capture_current "seq_logs/captures/$tag"
    done
    ;;

  formal)
    assert_common
    assert_key T WR_ETA CMF_FLUX_PARAM
    assert_key F WR_FLUX CMF_FLUX_PARAM
    assert_key F COMP_F CMF_FLUX_PARAM
    for name in CHI_DATA CHI_DATA_INFO ETA_DATA ETA_DATA_INFO; do
      [[ ! -e "$name" ]] || { echo "refusing to overwrite formal output: $name" >&2; exit 23; }
    done
    "$CMF_FLUX" <CMF_FLUX_STDIN >OUT_FLUX 2>seq_logs/formal_${SLURM_JOB_ID}.stderr
    for name in CHI_DATA CHI_DATA_INFO ETA_DATA ETA_DATA_INFO OBSFLUX OBS_FREQ; do
      [[ -s "$name" ]] || { echo "formal output missing: $name" >&2; exit 24; }
    done
    python3 - "$SLURM_JOB_ID" <<'PY'
import datetime as dt, hashlib, json, pathlib, sys
names = ["CHI_DATA", "CHI_DATA_INFO", "ETA_DATA", "ETA_DATA_INFO", "OBSFLUX", "OBS_FREQ", "RVTJ"]
doc = {
    "schema": "lumina-cmfgen-ophys-formal-provenance-v1",
    "slurm_job_id": sys.argv[1],
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "files": {n: hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest() for n in names},
}
pathlib.Path("PROVENANCE_FORMAL.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
PY
    ;;

  package)
    [[ -n ${OPHYS_EVIDENCE:-} ]] || { echo "set OPHYS_EVIDENCE to reviewed JSON" >&2; exit 25; }
    package_args=()
    [[ ${OPHYS_PACKAGE_FORCE:-0} == 1 ]] && package_args+=(--force)
    python3 ./package_cmfgen_ophys_attestation.py \
      --root "$RUN" --evidence "$OPHYS_EVIDENCE" "${package_args[@]}"
    manifest=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys.manifest.json
    contract=seq_logs/tools/cmfgen_oracle_contract.py
    profile=seq_logs/tools/A2_00_OPHYS_PROFILE.json
    python3 "$contract" write "$RUN" --manifest "$manifest" \
      --profile ophys --profile-file "$profile"
    python3 "$contract" check "$RUN" --manifest "$manifest" \
      --profile ophys --profile-file "$profile"
    ;;

  *)
    echo "OPHYS_MODE must be solve, capture, formal, or package" >&2
    exit 26
    ;;
esac
