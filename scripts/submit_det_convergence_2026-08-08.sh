#!/usr/bin/env bash
# Stage immutable inputs and submit one fail-closed accelerator DET flight.
set -euo pipefail
umask 027

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

die() {
  printf 'DET_SUBMIT_FATAL %s\n' "$*" >&2
  exit 70
}

command -v sbatch >/dev/null || die "sbatch unavailable"
command -v sha256sum >/dev/null || die "sha256sum unavailable"
command -v rsync >/dev/null || die "rsync unavailable"

model="$repo_root/data/tardis_reference_toy06_19p48d_sivcaiv_active"
binary="$repo_root/lumina_cuda"
batch="$repo_root/scripts/run_det_convergence_2026-08-08.slurm"
checker="$repo_root/scripts/check_det_convergence.py"
census_summarizer="$repo_root/scripts/summarize_a210_cancellation_census.py"
cancellation_checker="$repo_root/scripts/check_a210_cancellation_witnesses.py"
targeted_gate_checker="$repo_root/scripts/check_a210_targeted_gate.py"
reference_launcher="$repo_root/scripts/run_coevolve_s01.sh"
topion_ground="$repo_root/data/atomic/topion_ground_levels.csv"
topion_levels="$repo_root/data/atomic/topion_levels.csv"
ionization_reference="$repo_root/data/atomic/ionization_reference.csv"
iterations="${DET_OUTER_ITERS:-20}"
run_base="${DET_RUN_BASE:-/gpfs/$USER/lumina/det_convergence}"
slurm_job_name="${DET_SLURM_JOB_NAME:-det_conv_1234}"
slurm_partition="${DET_SLURM_PARTITION:-h200,h100,a100}"
slurm_gres="${DET_SLURM_GRES:-gpu:1}"
slurm_cpus="${DET_SLURM_CPUS_PER_TASK:-32}"
slurm_mem="${DET_SLURM_MEM:-}"
slurm_time="${DET_SLURM_TIME:-2-00:00:00}"
slurm_nodelist="${DET_SLURM_NODELIST:-}"
matdump="${DET_MATDUMP:-0}"
matdump_Z="${DET_MATDUMP_Z:-14}"
matdump_ion="${DET_MATDUMP_ION:-1}"
matdump_shell="${DET_MATDUMP_SHELL:-4}"
single_total="${DET_SINGLE_TOTAL:-0}"
stage4="${DET_NLTE_STAGE4:-0}"
cmf_fine_mgpu_devices="${DET_CMF_FINE_MGPU_DEVICES:-0}"
cmf_fine_envelope_refinements="${DET_CMF_FINE_ENVELOPE_REFINEMENTS:-8}"
stage_only="${DET_STAGE_ONLY:-0}"
targeted_one_iter="${DET_TARGETED_ONE_ITER:-0}"
cancellation_census="${DET_A210_CANCELLATION_CENSUS:-0}"
precore_tau_refresh="${DET_A210_PRECORE_TAU_REFRESH:-0}"
expected_sigma="90d04042c17bcc5f2c7c521b65a9bb0f824179d79493f82ad40deaa7185cc3ad"

[[ "$targeted_one_iter" == 0 || "$targeted_one_iter" == 1 ]] || \
  die "DET_TARGETED_ONE_ITER must be 0 or 1"
[[ "$cancellation_census" == 0 || "$cancellation_census" == 1 ]] || \
  die "DET_A210_CANCELLATION_CENSUS must be 0 or 1"
[[ "$precore_tau_refresh" == 0 || "$precore_tau_refresh" == 1 ]] || \
  die "DET_A210_PRECORE_TAU_REFRESH must be 0 or 1"
if [[ "$targeted_one_iter" == 1 ]]; then
  [[ "$iterations" == 1 ]] || \
    die "DET_TARGETED_ONE_ITER requires DET_OUTER_ITERS=1"
else
  [[ "$iterations" =~ ^[0-9]+$ && "$iterations" -ge 4 ]] || \
    die "DET_OUTER_ITERS must be an integer >=4 outside targeted diagnostics"
  [[ "$cancellation_census" == 0 ]] || \
    die "cancellation census requires DET_TARGETED_ONE_ITER=1"
fi
if [[ "$cancellation_census" == 1 ]]; then
  diagnostic_mode=A210_CANCELLATION_CENSUS
elif [[ "$targeted_one_iter" == 1 ]]; then
  diagnostic_mode=A210_TARGETED_GATE
else
  diagnostic_mode=FLIGHT
fi
[[ "$slurm_job_name" =~ ^[A-Za-z0-9._-]+$ ]] || \
  die "DET_SLURM_JOB_NAME contains unsafe characters"
[[ "$slurm_partition" =~ ^[A-Za-z0-9,_-]+$ ]] || \
  die "DET_SLURM_PARTITION contains unsafe characters"
[[ "$slurm_gres" =~ ^gpu(:[A-Za-z0-9._-]+)?:[1-9][0-9]*$ ]] || \
  die "DET_SLURM_GRES must be gpu[:TYPE]:COUNT"
[[ "$slurm_cpus" =~ ^[1-9][0-9]*$ ]] || \
  die "DET_SLURM_CPUS_PER_TASK must be positive"
[[ -z "$slurm_mem" || "$slurm_mem" =~ ^[1-9][0-9]*[KMGT]?$ ]] || \
  die "DET_SLURM_MEM is invalid"
[[ "$slurm_time" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] || \
  die "DET_SLURM_TIME is invalid"
[[ -z "$slurm_nodelist" || "$slurm_nodelist" =~ ^[A-Za-z0-9,._-]+$ ]] || \
  die "DET_SLURM_NODELIST contains unsafe characters"
[[ "$matdump" == 0 || "$matdump" == 1 ]] || \
  die "DET_MATDUMP must be 0 or 1"
[[ "$single_total" == 0 || "$single_total" == 1 ]] || \
  die "DET_SINGLE_TOTAL must be 0 or 1"
[[ "$stage4" == 0 || "$stage4" == 1 ]] || \
  die "DET_NLTE_STAGE4 must be 0 or 1"
[[ "$cmf_fine_mgpu_devices" =~ ^[0-9]+$ ]] || \
  die "DET_CMF_FINE_MGPU_DEVICES must be a nonnegative integer"
[[ "$cmf_fine_envelope_refinements" =~ ^[1-9][0-9]*$ &&
   "$cmf_fine_envelope_refinements" -le 64 ]] || \
  die "DET_CMF_FINE_ENVELOPE_REFINEMENTS must be an integer in 1..64"
[[ "$stage_only" == 0 || "$stage_only" == 1 ]] || \
  die "DET_STAGE_ONLY must be 0 or 1"
slurm_gres_count="${slurm_gres##*:}"
if [[ "$cmf_fine_mgpu_devices" -gt 0 &&
      "$slurm_gres_count" -ne "$cmf_fine_mgpu_devices" ]]; then
  die "DET_CMF_FINE_MGPU_DEVICES must equal the GPU count in DET_SLURM_GRES"
fi
[[ "$matdump_Z" =~ ^[1-9][0-9]*$ && "$matdump_ion" =~ ^[1-9][0-9]*$ &&
   "$matdump_shell" =~ ^[0-9]+$ ]] || \
  die "DET_MATDUMP_Z/ION/SHELL must be positive/positive/nonnegative integers"
[[ "$run_base" == /gpfs/* && "$run_base" != /gpfs/ && "$run_base" != /gpfs ]] || \
  die "DET_RUN_BASE must be a specific directory below /gpfs"
[[ -d "$model" && -x "$binary" && -x "$checker" && -x "$batch" && \
   -f "$census_summarizer" && -f "$cancellation_checker" && \
   -f "$targeted_gate_checker" && \
   -f "$topion_ground" && -f "$topion_levels" &&
   -f "$ionization_reference" ]] || \
  die "model/binary/checker/batch is missing or not executable"
[[ -f "$model/DECK_PROVENANCE.json" ]] || die "active deck provenance missing"
[[ "$(find "$model" -maxdepth 1 -type f | wc -l)" -eq 55 ]] || \
  die "active deck top-level file count is not the sealed value 55"
actual_sigma="$(sha256sum "$model/cmfgen_sigma_bf.bin" | cut -d' ' -f1)"
[[ "$actual_sigma" == "$expected_sigma" ]] || die "canonical exact-Hyd sigma mismatch"
python3 scripts/check_topion_catalog.py >/dev/null || die "top-ion catalog C1-C8 failed"
rg -q '"LUMINA_PHYSICS_COMPARISON_DIR"' src/env_universe.h || \
  die "comparison dump env is absent from strict universe"

# A queued job must never build or race with another job.  Refuse a stale target;
# the operator performs the one build and tests before calling this helper.
if ! make -q lumina_cuda; then
  die "lumina_cuda is stale; rebuild and test once before submission"
fi

binary_sha="$(sha256sum "$binary" | cut -d' ' -f1)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="det1234_${timestamp}_${binary_sha:0:12}"
run_root="$run_base/$run_id"
[[ ! -e "$run_root" ]] || die "run root already exists: $run_root"
mkdir -p "$run_root/input/model" "$run_root/input/global_atomic"
printf 'STAGING\n' > "$run_root/STAGING"
input_dir="$run_root/input"
staged_model="$input_dir/model"
staged_global_atomic="$input_dir/global_atomic"

printf 'DET_SUBMIT_STAGE run_root=%s active_files=55\n' "$run_root"

# Hash source basenames, copy active top-level files only (never quarantine), and
# verify the staged copy before it is made visible to the batch job.
(
  cd "$model"
  find . -maxdepth 1 -type f -printf '%P\0' | LC_ALL=C sort -z | \
    xargs -0 sha256sum
) > "$input_dir/deck.sha256"
while IFS= read -r -d '' source_file; do
  cp -a "$source_file" "$staged_model/"
done < <(find "$model" -maxdepth 1 -type f -print0)
(
  cd "$staged_model"
  sha256sum -c "$input_dir/deck.sha256"
) > "$run_root/deck_stage_verify.log"

install -m 0644 "$topion_ground" "$staged_global_atomic/topion_ground_levels.csv"
install -m 0644 "$topion_levels" "$staged_global_atomic/topion_levels.csv"
install -m 0644 "$ionization_reference" \
  "$staged_global_atomic/ionization_reference.csv"
(
  cd "$repo_root/data/atomic"
  sha256sum topion_ground_levels.csv topion_levels.csv ionization_reference.csv
) > "$input_dir/topion.sha256"
(
  cd "$staged_global_atomic"
  sha256sum -c "$input_dir/topion.sha256"
) > "$run_root/topion_stage_verify.log"

install -m 0755 "$binary" "$input_dir/lumina_cuda"
install -m 0755 "$checker" "$input_dir/check_det_convergence.py"
install -m 0755 "$census_summarizer" \
  "$input_dir/summarize_a210_cancellation_census.py"
install -m 0755 "$cancellation_checker" \
  "$input_dir/check_a210_cancellation_witnesses.py"
install -m 0755 "$targeted_gate_checker" \
  "$input_dir/check_a210_targeted_gate.py"
install -m 0755 "$batch" "$input_dir/job.slurm"
install -m 0644 "$reference_launcher" "$input_dir/run_coevolve_s01.reference.sh"
printf '%s\n' "$binary_sha" > "$input_dir/binary.sha256"
printf '%s\n' "$expected_sigma" > "$input_dir/sigma.sha256"
printf '%s\n' "$iterations" > "$input_dir/outer_iterations.txt"
printf '%s\n' "$single_total" > "$input_dir/single_total.txt"
printf '%s\n' "$stage4" > "$input_dir/stage4.txt"
printf '%s\n' "$cmf_fine_envelope_refinements" \
  > "$input_dir/envelope_refinements.txt"
printf '%s\n' "$diagnostic_mode" > "$input_dir/diagnostic_mode.txt"
printf '%s\n' "$precore_tau_refresh" > "$input_dir/precore_tau_refresh.txt"
git rev-parse HEAD > "$input_dir/git_head.txt"
git status --short > "$input_dir/git_status.txt"
git diff --stat > "$input_dir/git_diff.stat"
sha256sum "$input_dir/check_det_convergence.py" \
  "$input_dir/summarize_a210_cancellation_census.py" \
  "$input_dir/check_a210_cancellation_witnesses.py" \
  "$input_dir/check_a210_targeted_gate.py" "$input_dir/job.slurm" \
  "$input_dir/run_coevolve_s01.reference.sh" > "$input_dir/flight_scripts.sha256"

# Resolve the existing physical environment mechanically, then remove every
# enforced/retired/dead knob and apply only DET-flight path/diagnostic overrides.
# The batch starts by deleting all inherited LUMINA_* variables and sources this
# sealed file, so submission-shell contamination cannot cross the boundary.
(
  set -euo pipefail
  while IFS='=' read -r name _; do
    [[ "$name" == LUMINA_* ]] && unset "$name"
  done < <(env)
  MODEL="$staged_model"
  NITER="$iterations"
  eval "$(grep -E '^export ' "$reference_launcher")"

  unset_list=$(
    {
      grep -oE 'X\("(LUMINA_[A-Z0-9_]+)", +LK_ENFORCE_FATAL' \
        "$repo_root/src/legacy_knob_registry.h" | grep -oE 'LUMINA_[A-Z0-9_]+'
      sed -n '/retired_scalar_options\[\]/,/};/p' "$repo_root/src/lumina_atomic.c" \
        | grep -oE '"LUMINA_[A-Z0-9_]+"' | tr -d '"'
      echo LUMINA_CMF_EPAY
    } | sort -u | tr '\n' ' '
  )
  unset $unset_list

  universe="$(grep -oE '"LUMINA_[A-Z0-9_]+"' "$repo_root/src/env_universe.h" \
    | tr -d '"' | sort -u)"
  dead="$(comm -23 <(env | grep -oE '^LUMINA_[A-Z0-9_]+' | sort -u) \
    <(printf '%s\n' "$universe"))"
  [[ -z "$dead" ]] || unset $dead

  unset LUMINA_CMFGEN_THEN_MC LUMINA_MC_COEVOLVE \
    LUMINA_MC_COEVOLVE_CONSUME LUMINA_MC_COEVOLVE_INJECT \
    LUMINA_CMF_FINE_MGPU_DEVICES LUMINA_CMF_FINE_MGPU_AB \
    LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS \
    LUMINA_A210_CANCELLATION_CENSUS \
    LUMINA_A210_PRECORE_TAU_REFRESH
  # This flight measures the raw physical producer.  Every historical
  # population/field repair is explicitly disabled after resolving the old
  # reference launcher, so that launcher defaults cannot silently re-arm one.
  export LUMINA_NLTE_LTE_FLOOR=0
  export LUMINA_NLTE_FLOOR_MODE=0
  export LUMINA_NLTE_FLOOR_REG=0
  export LUMINA_NLTE_BK_CEIL=0
  export LUMINA_NLTE_INV_CEIL=0
  export LUMINA_NLTE_COLL_FLOOR=0
  export LUMINA_DR_FLOOR_CMS=0
  export LUMINA_STAGE4_BK_CAP=0
  export LUMINA_HRESP_CLAMP=0
  export LUMINA_TE_STEP_CLAMP=0
  export LUMINA_J_CAP_FACTOR=0
  export LUMINA_J_FLOOR_FACTOR=0
  export LUMINA_RADEQ_LINE_CULL=0
  export LUMINA_NLTE_GREY_TAU=0
  export LUMINA_NLTE_ASSEMBLE_GPU=0
  export LUMINA_NLTE_FALLBACK_TE=0
  export LUMINA_MODEL_DIR="$staged_model"
  export LUMINA_DEPOSITION_FILE="$staged_model/deposition_cmfgen.csv"
  export LUMINA_CMFGEN_SIGMA_BF="$staged_model/cmfgen_sigma_bf.bin"
  export LUMINA_TOPION_LEVELS_FILE="$staged_global_atomic/topion_levels.csv"
  export LUMINA_IONIZATION_REFERENCE_FILE="$staged_global_atomic/ionization_reference.csv"
  export LUMINA_PURE_CMFGEN=1
  export LUMINA_DET_TRANSACTIONAL=1
  export LUMINA_PURE_CMFGEN_ITER="$iterations"
  export LUMINA_CMF_FINE_LAMLO=100
  export LUMINA_CMF_FINE_LAMHI=20000
  export LUMINA_CMF_FINE_VDOP=1000000
  export LUMINA_CMF_FINE_PPD=12
  export LUMINA_CMF_FINE_ALI=64
  export LUMINA_CMF_FINE_TOL=1e-8
  export LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS="$cmf_fine_envelope_refinements"
  if [[ "$cmf_fine_mgpu_devices" -gt 0 ]]; then
    export LUMINA_CMF_FINE_MGPU_DEVICES="$cmf_fine_mgpu_devices"
  fi
  if [[ "$single_total" == 1 ]]; then
    # Sealed A/B: one element-total closure lets the homogeneous SE generator
    # own the ion-stage partition.  Apply after inherited-env sanitization and
    # reference resolution so the staged values cannot silently revert to 1.
    export LUMINA_NLTE_ION_LOCK=0
    export LUMINA_NLTE_PER_ION_RESCALE=0
  fi
  if [[ "$stage4" == 1 ]]; then
    # Sealed ownership A/B: promote the existing adjacent III-IV pairs.  Keep
    # this independent of single-total so the two controls remain explicit in
    # the staged manifest and cannot leak in from the submission shell.
    export LUMINA_NLTE_STAGE4=1
  else
    unset LUMINA_NLTE_STAGE4
  fi
  export LUMINA_PHYSICS_COMPARISON_DIR="$run_root/work/physics_comparison"
  export LUMINA_RADEQ_DIAG=1
  if [[ "$cancellation_census" == 1 ]]; then
    export LUMINA_A210_CANCELLATION_CENSUS=1
  fi
  if [[ "$precore_tau_refresh" == 1 ]]; then
    export LUMINA_A210_PRECORE_TAU_REFRESH=1
  fi
  if [[ "$matdump" == 1 ]]; then
    export LUMINA_NLTE_MATDUMP=1
    export LUMINA_NLTE_MATDUMP_PATH="$run_root/work/nlte_matrix"
    export LUMINA_POP_Z="$matdump_Z"
    export LUMINA_POP_ION="$matdump_ion"
    export LUMINA_POP_SHELL="$matdump_shell"
  fi
  export LUMINA_ENV_STRICT=1
  declare -px | grep -E '^declare -x LUMINA_' | LC_ALL=C sort
) > "$input_dir/resolved_lumina.exports"

grep -q '^declare -x LUMINA_PHYSICS_COMPARISON_DIR=' \
  "$input_dir/resolved_lumina.exports" || die "resolved env lost comparison path"
grep -q '^declare -x LUMINA_PURE_CMFGEN="1"' \
  "$input_dir/resolved_lumina.exports" || die "resolved env lost DET lane"
expected_stage_lock=1
[[ "$single_total" == 1 ]] && expected_stage_lock=0
grep -q "^declare -x LUMINA_NLTE_ION_LOCK=\"$expected_stage_lock\"" \
  "$input_dir/resolved_lumina.exports" || die "resolved ion-lock mode mismatch"
grep -q "^declare -x LUMINA_NLTE_PER_ION_RESCALE=\"$expected_stage_lock\"" \
  "$input_dir/resolved_lumina.exports" || die "resolved per-ion-rescale mode mismatch"
if [[ "$stage4" == 1 ]]; then
  grep -q '^declare -x LUMINA_NLTE_STAGE4="1"' \
    "$input_dir/resolved_lumina.exports" || die "resolved stage4 mode mismatch"
elif grep -q '^declare -x LUMINA_NLTE_STAGE4=' \
  "$input_dir/resolved_lumina.exports"; then
  die "resolved env contains stage4 in the OFF arm"
fi
if grep -Eq '^declare -x LUMINA_(CMFGEN_THEN_MC|MC_COEVOLVE)=' \
    "$input_dir/resolved_lumina.exports"; then
  die "resolved env contains MC feedback"
fi
if [[ "$cmf_fine_mgpu_devices" -gt 0 ]]; then
  grep -q "^declare -x LUMINA_CMF_FINE_MGPU_DEVICES=\"$cmf_fine_mgpu_devices\"" \
    "$input_dir/resolved_lumina.exports" || die "resolved multi-GPU owner mismatch"
elif grep -q '^declare -x LUMINA_CMF_FINE_MGPU_DEVICES=' \
    "$input_dir/resolved_lumina.exports"; then
  die "resolved env unexpectedly contains multi-GPU owner"
fi
if grep -q '^declare -x LUMINA_CMF_FINE_MGPU_AB=' \
    "$input_dir/resolved_lumina.exports"; then
  die "DET flight must not enable CPU/GPU A/B duplication"
fi
grep -q "^declare -x LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=\"$cmf_fine_envelope_refinements\"" \
  "$input_dir/resolved_lumina.exports" || \
  die "resolved envelope refinement count mismatch"
if [[ "$cancellation_census" == 1 ]]; then
  grep -q '^declare -x LUMINA_A210_CANCELLATION_CENSUS="1"' \
    "$input_dir/resolved_lumina.exports" || \
    die "resolved env lost cancellation census"
elif grep -q '^declare -x LUMINA_A210_CANCELLATION_CENSUS=' \
    "$input_dir/resolved_lumina.exports"; then
  die "resolved env unexpectedly contains cancellation census"
fi
if [[ "$precore_tau_refresh" == 1 ]]; then
  grep -q '^declare -x LUMINA_A210_PRECORE_TAU_REFRESH="1"' \
    "$input_dir/resolved_lumina.exports" || \
    die "resolved env lost pre-core tau diagnostic"
elif grep -q '^declare -x LUMINA_A210_PRECORE_TAU_REFRESH=' \
    "$input_dir/resolved_lumina.exports"; then
  die "resolved env unexpectedly contains pre-core tau diagnostic"
fi
for zero_knob in \
  LUMINA_NLTE_LTE_FLOOR LUMINA_NLTE_FLOOR_MODE LUMINA_NLTE_FLOOR_REG \
  LUMINA_NLTE_BK_CEIL LUMINA_NLTE_INV_CEIL LUMINA_NLTE_COLL_FLOOR \
  LUMINA_DR_FLOOR_CMS LUMINA_STAGE4_BK_CAP LUMINA_HRESP_CLAMP \
  LUMINA_TE_STEP_CLAMP LUMINA_J_CAP_FACTOR LUMINA_J_FLOOR_FACTOR \
  LUMINA_RADEQ_LINE_CULL LUMINA_NLTE_GREY_TAU \
  LUMINA_NLTE_ASSEMBLE_GPU LUMINA_NLTE_FALLBACK_TE; do
  grep -q "^declare -x ${zero_knob}=\"0\"" \
    "$input_dir/resolved_lumina.exports" || \
    die "resolved numerical-repair contract mismatch: $zero_knob"
done
sha256sum "$input_dir/resolved_lumina.exports" > "$input_dir/resolved_lumina.sha256"
sha256sum "$input_dir/lumina_cuda" | grep -q "^$binary_sha " || \
  die "staged binary mismatch"

rm "$run_root/STAGING"
printf 'READY\n' > "$run_root/READY"
if [[ "$stage_only" == 1 ]]; then
  printf 'STAGED_ONLY run_root=%s binary_sha256=%s sigma_sha256=%s iterations=%s single_total=%s stage4=%s mgpu_devices=%s refinements=%s cpus=%s diagnostic=%s\n' \
    "$run_root" "$binary_sha" "$expected_sigma" "$iterations" \
    "$single_total" "$stage4" "$cmf_fine_mgpu_devices" \
    "$cmf_fine_envelope_refinements" "$slurm_cpus" \
    "$diagnostic_mode" | \
    tee "$run_root/STAGED_ONLY.txt"
  exit 0
fi
sbatch_args=(
  --parsable
  --job-name="$slurm_job_name"
  --partition="$slurm_partition"
  --gres="$slurm_gres"
  --cpus-per-task="$slurm_cpus"
  --time="$slurm_time"
  --output="$run_root/slurm-%j.out"
  --error="$run_root/slurm-%j.err"
)
[[ -z "$slurm_mem" ]] || sbatch_args+=(--mem="$slurm_mem")
[[ -z "$slurm_nodelist" ]] || sbatch_args+=(--nodelist="$slurm_nodelist")
job_id="$(sbatch "${sbatch_args[@]}" "$input_dir/job.slurm" "$run_root")"
printf '%s\n' "$job_id" > "$run_root/job_id.txt"
printf 'SUBMITTED job_id=%s run_root=%s binary_sha256=%s sigma_sha256=%s iterations=%s single_total=%s stage4=%s refinements=%s diagnostic=%s partition=%s gres=%s cpus=%s mem=%s time=%s nodelist=%s\n' \
  "$job_id" "$run_root" "$binary_sha" "$expected_sigma" "$iterations" \
  "$single_total" "$stage4" "$cmf_fine_envelope_refinements" \
  "$diagnostic_mode" \
  "$slurm_partition" "$slurm_gres" \
  "$slurm_cpus" "${slurm_mem:-default}" "$slurm_time" \
  "${slurm_nodelist:-scheduler}" | \
  tee "$run_root/SUBMITTED.txt"
