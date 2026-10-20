#!/usr/bin/env bash
# One isolated CMFGEN fixed-T continuation from the modern run's iteration 40.
# Submitting this job never edits the parent run or the CMFGEN source tree.
#SBATCH --job-name=cmf-stageA
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/stageA-fixedT-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/stageA-fixedT-%j.err

set -euo pipefail
umask 027

readonly BASE=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern
readonly RUNS=/gpfs/kjhan/cmfgen_runs
readonly SOURCE=/gpfs/kjhan/cmfgen_src/cur_cmf
readonly CMFGEN="$SOURCE/exe/cmfgen_dev.exe"
readonly REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
readonly PYTHON=/home/kjhan/miniconda3/bin/python3
readonly JOB_TAG="toy06_19p48d_stageA_fixedT_${SLURM_JOB_ID}"
readonly RUN="$RUNS/$JOB_TAG"
readonly STAGE="$RUNS/.${JOB_TAG}.staging"

[[ ${SLURM_CPUS_PER_TASK:-} == 16 ]] || { echo "expected 16 allocated CPUs" >&2; exit 20; }
[[ -d "$BASE" && -x "$CMFGEN" && -x "$PYTHON" ]] || { echo "CMFGEN base, executable, or Python is missing" >&2; exit 21; }
[[ ! -e "$RUN" && ! -e "$STAGE" ]] || { echo "refusing existing target: $RUN or $STAGE" >&2; exit 22; }

module purge
module load gnu13
export OMP_NUM_THREADS=16
export OMP_DYNAMIC=FALSE
export OMP_STACKSIZE=512M
export OMP_PROC_BIND=close
export OMP_PLACES=cores
ulimit -s unlimited 2>/dev/null || true

inputs=(
  VADAT IN_ITS MODEL_SPEC SN_HYDRO_DATA MODEL SPECIES_MASSES
  LEVEL_SL_STEQ_LINKS model_spec_isf.txt atomic_links.txt setup_links.sh
  PROVENANCE.txt INTENDED_DIFF_MANIFEST.txt MODEL_SPEC.base_reference
  RUNTIME_ESTIMATE.txt PHOT_PRESCAN.txt PREFLIGHT.txt SIGMA_REPAIR_CHECK.txt
  gen_atomic_base_reference.py gen_atomic_modern.py mk_atomic_local_repairs.py
  mk_sn_hydro.py phot_prescan.py preflight_lib.py preflight_run.py
  sigma_repair_check.py snia_toy06_19.48d.dat run_modern.info
)
restart=(SCRTEMP POINT1 POINT2 EDDFACTOR EDDFACTOR_INFO CONT_FREQ CUR_MODEL_DATA STEQ_VALS RVTJ)

for name in "${inputs[@]}" "${restart[@]}"; do
  [[ -f "$BASE/$name" ]] || { echo "missing/non-regular seed input: $BASE/$name" >&2; exit 23; }
done
for name in "${restart[@]}"; do
  [[ ! -L "$BASE/$name" ]] || { echo "restart must not be an external symlink: $BASE/$name" >&2; exit 23; }
done

mkdir "$STAGE"
mkdir -p "$STAGE/seq_logs"

# Prove the seed's interpreted iteration-40 state and pointer/frequency-depth
# compatibility before creating the continuation copy.
"$PYTHON" "$REPO/scripts/ptf_gated_metrics.py" "$BASE" \
  --from-it 40 --to-it 40 --state-first-it 1 --format json \
  --output "$STAGE/seq_logs/seed-iteration-40-preflight.json"
"$PYTHON" - "$BASE/POINT1" "$BASE/POINT2" "$BASE/EDDFACTOR_INFO" <<'PY'
from pathlib import Path
import sys

for filename in sys.argv[1:3]:
    lines = Path(filename).read_text().splitlines()
    fields = lines[1].split()
    if fields[:4] != ["40", "40", "1", "-1000"] or fields[4] != "F":
        raise SystemExit(f"unexpected restart pointer in {filename}: {fields}")

info = Path(sys.argv[3]).read_text().splitlines()
if int(info[2].split()[0]) != 90:
    raise SystemExit(f"EDDFACTOR depth count does not match ND=90: {info[2]}")
PY

for name in "${inputs[@]}" "${restart[@]}"; do
  cp -a -- "$BASE/$name" "$STAGE/"
done
mkdir -p "$STAGE/atomic_local/SUL/III/19apr23" "$STAGE/atomic_local/COB/II/19apr23" \
  "$STAGE/seq_logs/tools"
cp -a -- "$BASE/atomic_local/SUL/III/19apr23/phot_data_A" "$STAGE/atomic_local/SUL/III/19apr23/"
cp -a -- "$BASE/atomic_local/COB/II/19apr23/phot_data_A" "$STAGE/atomic_local/COB/II/19apr23/"
mkdir -p "$STAGE/seq_logs" "$STAGE/metrics"
cp -a -- "$REPO/scripts/ptf_gated_metrics.py" "$STAGE/seq_logs/tools/"
cp -a -- "$REPO/scripts/cmfgen_extract/cmfgen_convergence.py" "$STAGE/seq_logs/tools/"
cp -a -- "$REPO/scripts/configure_cmfgen_stageA.py" "$STAGE/seq_logs/tools/"

{
  for name in "${inputs[@]}" "${restart[@]}"; do
    sha256sum "$BASE/$name"
  done
} >"$STAGE/seq_logs/seed-inputs.sha256"

"$PYTHON" "$STAGE/seq_logs/tools/configure_cmfgen_stageA.py" "$STAGE/VADAT" "$STAGE/IN_ITS"
cp -a -- "$REPO/docs/BUNDLE_CMFGEN_FIXEDT_STAGEA_2026-10-19.md" "$STAGE/STAGE_A_PLAN.md"
mv -- "$STAGE" "$RUN"

# Retarget only the two run-local repaired photoionization data links and the
# setup script's working directory to this immutable, job-specific run root.
"$PYTHON" - "$RUN/setup_links.sh" "$RUN/atomic_links.txt" "$BASE" "$RUN" <<'PY'
from pathlib import Path
import sys

old, new = sys.argv[3:5]
for name in sys.argv[1:3]:
    path = Path(name)
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"expected parent path not found in {path}")
    path.write_text(text.replace(old, new))
PY
bash "$RUN/setup_links.sh"

# Validate and checksum every input link emitted by setup_links.sh, including
# generic CMFGEN options as well as atomic data. Only the two documented H I
# line-list links are optional.
while IFS= read -r link; do
  if [[ ! -e "$RUN/$link" ]]; then
    case "$link" in
      HI_IS_LINE_LIST|H2_IS_LINE_LIST)
        printf 'OPTIONAL_MISSING\t%s\n' "$link" >>"$RUN/seq_logs/linked-inputs.targets.tsv"
        continue
        ;;
      *) echo "required linked input missing: $link" >&2; exit 24 ;;
    esac
  fi
  printf '%s\t%s\n' "$link" "$(readlink -f "$RUN/$link")" \
    >>"$RUN/seq_logs/linked-inputs.targets.tsv"
  sha256sum "$RUN/$link"
done < <(awk '$1 == "ln" && $2 == "-sf" {print $4}' "$RUN/setup_links.sh" | sort -u) \
  >"$RUN/seq_logs/linked-inputs.sha256"

for link in Sk2_F_OSCDAT Sk2_F_TO_S PHOTSk2_A PHOTSk2_B \
            SkIII_F_OSCDAT SkIII_F_TO_S PHOTSkIII_A \
            S2_F_OSCDAT S2_F_TO_S PHOTS2_A SIII_F_OSCDAT SIII_F_TO_S PHOTSIII_A; do
  [[ -e "$RUN/$link" ]] || { echo "Si/S atomic link missing: $link" >&2; exit 25; }
done

{
  printf 'bundle=%s\n' 'CMFGEN fixed-T Stage A, diagnostic staging only'
  printf 'job_id=%s\nrun_root=%s\nseed_root=%s\n' "$SLURM_JOB_ID" "$RUN" "$BASE"
  printf 'host=%s\nsource_git_head=%s\n' "$(hostname)" "$(git -C "$SOURCE" rev-parse HEAD)"
  printf 'cmfgen_sha256=%s\n' "$(sha256sum "$CMFGEN" | awk '{print $1}')"
  printf 'resources=cpus:16,mem:200G,partition:a10,gpu:0,time:14:00:00\n'
  printf 'seed_status=untrusted_initial_guess; not an anchor or oracle\n'
  printf 'controls=FIX_T:T,FIX_T_AUTO:F,DO_DDT:F,TRT_NON_TE:F,DO_T_AUTO:F,DO_LAM_IT:F,DO_LAM_AUTO:F,DO_NG:F,MAX_LIN:3.0D0,MAX_LAM:10.0D0,NUM_ITS:19\n'
  printf 'linked_input_manifest=seq_logs/linked-inputs.sha256\nseed_manifest=seq_logs/seed-inputs.sha256\n'
  printf 'postprocessor_sha256=%s\n' "$(sha256sum "$RUN/seq_logs/tools/ptf_gated_metrics.py" | awk '{print $1}')"
  printf 'python=%s\n' "$($PYTHON --version 2>&1)"
  printf 'numpy=%s\n' "$($PYTHON -c 'import numpy; print(numpy.__version__)')"
} >"$RUN/seq_logs/stageA-runtime.txt"
{
  sha256sum "$CMFGEN" "$PYTHON" "$REPO/scripts/submit_cmfgen_stageA_fixedT.slurm"
  for name in "${inputs[@]}" "${restart[@]}"; do
    sha256sum "$RUN/$name"
  done
  sha256sum "$RUN/seq_logs/tools/ptf_gated_metrics.py" \
    "$RUN/seq_logs/tools/cmfgen_convergence.py" \
    "$RUN/seq_logs/tools/configure_cmfgen_stageA.py"
} >"$RUN/seq_logs/stageA-pre-run.sha256"

assert_key() {
  local value=$1 key=$2 file=$3
  awk -v want="$value" -v key="[$key]" \
    'index($0,key) && $1 == want {ok++} END {exit ok == 1 ? 0 : 1}' "$file"
}
assert_key T FIX_T "$RUN/VADAT"
assert_key F FIX_T_AUTO "$RUN/VADAT"
assert_key F DO_DDT "$RUN/VADAT"
assert_key F TRT_NON_TE "$RUN/VADAT"
assert_key T INS_DEP "$RUN/VADAT"
assert_key F DO_NG "$RUN/VADAT"
assert_key 3.0D0 MAX_LIN "$RUN/VADAT"
assert_key 10.0D0 MAX_LAM "$RUN/VADAT"
assert_key T WRITE_RATES "$RUN/VADAT"
assert_key T WRITE_JH "$RUN/VADAT"
assert_key 19 NUM_ITS "$RUN/IN_ITS"
assert_key F DO_LAM_IT "$RUN/IN_ITS"
assert_key F DO_LAM_AUTO "$RUN/IN_ITS"
assert_key F DO_T_AUTO "$RUN/IN_ITS"
[[ ! -e "$RUN/XzV_IN" ]] || { echo "external population vector is forbidden" >&2; exit 26; }

cd "$RUN"
run_start_epoch=$(date +%s)
printf 'run_start_epoch=%s\n' "$run_start_epoch" >>"seq_logs/stageA-runtime.txt"
set +e
srun --cpu-bind=cores "$CMFGEN" </dev/null \
  >"seq_logs/cmfgen.stdout" 2>"seq_logs/cmfgen.stderr"
cmfgen_rc=$?
set -e
printf 'cmfgen_exit_code=%s\n' "$cmfgen_rc" >>"seq_logs/stageA-runtime.txt"
printf 'CMFGEN_EXIT=%s\n' "$cmfgen_rc" >batch.log
[[ $cmfgen_rc == 0 ]] || exit "$cmfgen_rc"
[[ -s OUTGEN && -s RVTJ && -s NETRATE && -s TOTRATE && -s LINEHEAT && -s JH_AT_CURRENT_TIME ]] || {
  echo "required final-iteration diagnostic output missing" >&2
  exit 27
}
if rg -n -i '\b(NaN|Infinity)\b' OUTGEN seq_logs/cmfgen.stdout seq_logs/cmfgen.stderr \
  >seq_logs/nonfinite-markers.txt; then
  echo "non-finite marker found in CMFGEN text output" >&2
  exit 28
fi
if rg -n -i 'moment.{0,80}(error|fail)|((error|fail).{0,80}(moment|MOM_JREL))' \
  OUTGEN seq_logs/cmfgen.stdout seq_logs/cmfgen.stderr \
  >seq_logs/moment-error-markers.txt; then
  echo "moment solver error marker found" >&2
  exit 29
fi
rg -n -i 'moment|grey|error|fail|blow|bad initial|unable to allocate' \
  OUTGEN seq_logs/cmfgen.stdout seq_logs/cmfgen.stderr \
  >seq_logs/solver-review-lines.txt || true
"$PYTHON" "$RUN/seq_logs/tools/cmfgen_convergence.py" "$RUN" \
  --run-start "$run_start_epoch" --json >"$RUN/metrics/convergence.json"

# This report is diagnostic only. It does not certify the full v2 active-set
# union or an anchor; contribution-only activity is still evaluated separately.
"$PYTHON" "$RUN/seq_logs/tools/ptf_gated_metrics.py" "$RUN" \
  --state-first-it 1 --format json --output "$RUN/metrics/stageA_metrics.json"
convergence_status=$("$PYTHON" - "$RUN/metrics/convergence.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])
PY
)
printf 'cmfgen_convergence_status=%s\n' "$convergence_status" >>"$RUN/seq_logs/stageA-runtime.txt"
if [[ "$convergence_status" == CRASHED ]]; then
  echo "CMFGEN convergence classifier reports CRASHED" >&2
  exit 29
fi
sha256sum "$RUN/OUTGEN" "$RUN/RVTJ" "$RUN/NETRATE" "$RUN/TOTRATE" \
  "$RUN/LINEHEAT" "$RUN/JH_AT_CURRENT_TIME" "$RUN/metrics/stageA_metrics.json" \
  "$RUN/metrics/convergence.json" "$RUN/seq_logs/nonfinite-markers.txt" \
  "$RUN/seq_logs/moment-error-markers.txt" \
  "$RUN/seq_logs/linked-inputs.sha256" \
  "$RUN/seq_logs/linked-inputs.targets.tsv" \
  "$RUN/seq_logs/solver-review-lines.txt" \
  >"$RUN/seq_logs/stageA-output.sha256"
printf 'stageA_status=completed_diagnostic_staging_only; no physics acceptance implied\n' >>"$RUN/seq_logs/stageA-runtime.txt"
printf 'stageA_run_root=%s\n' "$RUN"
