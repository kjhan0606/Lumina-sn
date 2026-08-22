#!/usr/bin/env bash
#SBATCH --job-name=det_conv_1234
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00

# A single, sealed DET flight on a qualified sm80/sm90 accelerator.  The submit
# helper stages the binary, active-only deck, resolved environment, checker, and
# hashes before sbatch.
set -euo pipefail
umask 027

die() {
  printf 'DET_FLIGHT_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 1 ]] || die "usage: $0 ABSOLUTE_RUN_ROOT"
run_root="$1"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs ]] || \
  die "unsafe run root: $run_root"
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || \
  die "must run under Slurm on a compute node"

input_dir="$run_root/input"
work_dir="$run_root/work"
model_dir="$input_dir/model"
global_atomic_dir="$input_dir/global_atomic"
binary="$input_dir/lumina_cuda"
checker="$input_dir/check_det_convergence.py"
census_summarizer="$input_dir/summarize_a210_cancellation_census.py"
targeted_gate_checker="$input_dir/check_a210_targeted_gate.py"
l6_stager="$input_dir/stage_det_stage12_l6_probe.sh"
l6_analyzer="$input_dir/analyze_det_stage12_l6.py"
l6c_analyzer="$input_dir/analyze_det_stage12_l6c.py"
l6c_precondition="$input_dir/audit_l6c_cover_precondition.py"
resolved_env="$input_dir/resolved_lumina.exports"
iterations="$(<"$input_dir/outer_iterations.txt")"
diagnostic_mode="$(<"$input_dir/diagnostic_mode.txt")"
expected_refinements="$(<"$input_dir/envelope_refinements.txt")"
expected_sigma="$(<"$input_dir/sigma.sha256")"
expected_binary="$(<"$input_dir/binary.sha256")"

case "$diagnostic_mode" in
  FLIGHT)
    [[ "$iterations" =~ ^[0-9]+$ && "$iterations" -ge 4 ]] || \
      die "invalid flight outer iteration count: $iterations"
    ;;
  A210_CANCELLATION_CENSUS)
    [[ "$iterations" == 1 ]] || \
      die "cancellation census requires exactly one outer iteration"
    ;;
  A210_TARGETED_GATE)
    [[ "$iterations" == 1 ]] || \
      die "A2-10 targeted gate requires exactly one outer iteration"
    ;;
  A210_L6_PROBE)
    [[ "$iterations" == 2 ]] || \
      die "A2-10 L6 probe requires exactly two outer iterations"
    ;;
  A210_L6C_PROBE)
    [[ "$iterations" == 2 ]] || \
      die "A2-10 L6C probe requires exactly two outer iterations"
    ;;
  *) die "invalid diagnostic mode: $diagnostic_mode" ;;
esac
[[ "$expected_sigma" =~ ^[0-9a-f]{64}$ ]] || die "invalid sigma seal"
[[ "$expected_binary" =~ ^[0-9a-f]{64}$ ]] || die "invalid binary seal"
[[ "$expected_refinements" =~ ^[1-9][0-9]*$ &&
   "$expected_refinements" -le 64 ]] || die "invalid envelope refinement seal"
[[ -x "$binary" && -x "$checker" && -x "$census_summarizer" && \
   -x "$targeted_gate_checker" && \
   -d "$model_dir" && \
   -d "$global_atomic_dir" && -f "$resolved_env" ]] || \
  die "incomplete staged inputs"
if [[ "$diagnostic_mode" == A210_L6_PROBE ]]; then
  [[ -x "$l6_stager" && -x "$l6_analyzer" ]] || \
    die "incomplete staged L6 inputs"
fi
if [[ "$diagnostic_mode" == A210_L6C_PROBE ]]; then
  [[ -x "$l6_stager" && -x "$l6c_analyzer" && -x "$l6c_precondition" && \
     -f "$input_dir/l6c_target_ion.txt" && \
     ! -L "$input_dir/l6c_target_ion.txt" ]] || \
    die "incomplete staged L6C inputs"
fi

module load cuda/13.0.2 2>/dev/null || module load cuda 2>/dev/null || \
  die "CUDA module unavailable"
command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
gpu_csv="$(nvidia-smi --query-gpu=name,memory.total,uuid,driver_version,compute_cap \
  --format=csv,noheader,nounits | sed -n '1p')"
[[ -n "$gpu_csv" ]] || die "no visible GPU"
gpu_name="${gpu_csv%%,*}"
gpu_mem="$(printf '%s' "$gpu_csv" | cut -d, -f2 | tr -d ' ')"
case "$gpu_name" in
  *H200*|*H100*|*A100*) ;;
  *) die "unsupported deterministic-flight GPU: $gpu_name" ;;
esac
[[ "$gpu_mem" =~ ^[0-9]+$ && "$gpu_mem" -ge 80000 ]] || \
  die "GPU memory below 80000 MiB: $gpu_mem"

actual_binary="$(sha256sum "$binary" | cut -d' ' -f1)"
[[ "$actual_binary" == "$expected_binary" ]] || die "binary SHA mismatch"
actual_sigma="$(sha256sum "$model_dir/cmfgen_sigma_bf.bin" | cut -d' ' -f1)"
[[ "$actual_sigma" == "$expected_sigma" ]] || die "exact-Hyd sigma SHA mismatch"
(
  cd "$model_dir"
  sha256sum -c "$input_dir/deck.sha256"
) > "$run_root/deck_verify.log" 2>&1 || die "active deck seal mismatch"
(
  cd "$global_atomic_dir"
  sha256sum -c "$input_dir/topion.sha256"
) > "$run_root/topion_verify.log" 2>&1 || die "top-ion catalog seal mismatch"
[[ "$(find "$model_dir" -maxdepth 1 -type f | wc -l)" -eq 55 ]] || \
  die "active deck file count is not 55"

mkdir -p "$work_dir"
[[ -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
  die "work directory is not empty"
mkdir "$work_dir/physics_comparison"
printf '%s\n' "$gpu_csv" > "$run_root/gpu.csv"
env | LC_ALL=C sort > "$run_root/slurm_environment_before_sanitize.txt"

# No inherited LUMINA_* value may enter the flight.  Only the submission-time
# resolved and sealed export set is restored below.
while IFS='=' read -r name _; do
  [[ "$name" == LUMINA_* ]] && unset "$name"
done < <(env)
# shellcheck disable=SC1090
source "$resolved_env"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

[[ "${LUMINA_ENV_STRICT:-0}" == 1 ]] || die "strict env gate is not armed"
[[ "${LUMINA_PURE_CMFGEN:-0}" == 1 ]] || die "DET lane is not armed"
[[ "${LUMINA_DET_TRANSACTIONAL:-0}" == 1 ]] || \
  die "single-owner DET transaction lane is not armed"
[[ "${LUMINA_PURE_CMFGEN_ITER:-0}" == "$iterations" ]] || \
  die "outer iteration env mismatch"
[[ "${LUMINA_CMF_FINE_LAMLO:-}" == 100 &&
   "${LUMINA_CMF_FINE_LAMHI:-}" == 20000 &&
   "${LUMINA_CMF_FINE_VDOP:-}" == 1000000 &&
   "${LUMINA_CMF_FINE_PPD:-}" == 12 &&
   "${LUMINA_CMF_FINE_ALI:-}" == 64 &&
   "${LUMINA_CMF_FINE_TOL:-}" == 1e-8 &&
   "${LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS:-}" == \
     "$expected_refinements" ]] || \
  die "exact fine-Jbar solve contract mismatch"
[[ "${LUMINA_MODEL_DIR:-}" == "$model_dir" ]] || die "model path mismatch"
[[ "${LUMINA_CMFGEN_SIGMA_BF:-}" == "$model_dir/cmfgen_sigma_bf.bin" ]] || \
  die "sigma path mismatch"
[[ "${LUMINA_TOPION_LEVELS_FILE:-}" == "$global_atomic_dir/topion_levels.csv" ]] || \
  die "top-ion catalog path mismatch"
[[ "${LUMINA_IONIZATION_REFERENCE_FILE:-}" == \
   "$global_atomic_dir/ionization_reference.csv" ]] || \
  die "ionization-energy reference path mismatch"
[[ "${LUMINA_PHYSICS_COMPARISON_DIR:-}" == "$work_dir/physics_comparison" ]] || \
  die "comparison dump path mismatch"
[[ -z "${LUMINA_CMFGEN_THEN_MC:-}" && -z "${LUMINA_MC_COEVOLVE:-}" ]] || \
  die "MC feedback contaminated DET lane"
if [[ "$diagnostic_mode" == A210_CANCELLATION_CENSUS ]]; then
  [[ "${LUMINA_A210_CANCELLATION_CENSUS:-0}" == 1 ]] || \
    die "cancellation census mode is not armed"
else
  [[ -z "${LUMINA_A210_CANCELLATION_CENSUS:-}" ]] || \
    die "cancellation census contaminated a flight"
fi
if [[ "$diagnostic_mode" == A210_TARGETED_GATE ||
      "$diagnostic_mode" == A210_L6_PROBE ||
      "$diagnostic_mode" == A210_L6C_PROBE ]]; then
  [[ "${LUMINA_CMF_FINE_MGPU_DEVICES:-0}" == 2 ]] || \
    die "A2-10 diagnostic requires the two-device exact owner"
  case "$gpu_name" in
    *A100*) ;;
    *) die "A2-10 diagnostic requires A100 hardware: $gpu_name" ;;
  esac
fi
if [[ "$diagnostic_mode" == A210_L6_PROBE ]]; then
  [[ "${LUMINA_RADEQ_DIAG_TE_K:-}" == 10020 ]] || \
    die "L6 requested-temperature switch is not 10020 K"
  [[ "${LUMINA_A210_SPRODUCER_CAPTURE:-0}" == 1 ]] || \
    die "L6 raw producer capture is not armed"
  [[ "${LUMINA_A210_INDEPENDENT_CAPTURE:-0}" == 1 ]] || \
    die "L6 independent continuum capture is not armed"
  [[ "${LUMINA_A210_LINE_SATURATION_DIAG:-0}" == 2 ]] || \
    die "L6 line-saturation union diagnostic is not armed"
  [[ "${LUMINA_A210_LINE_SATURATION_TARGET_ION:-}" == 3 ]] || \
    die "L6 target ion is not IV"
fi
if [[ "$diagnostic_mode" == A210_L6C_PROBE ]]; then
  [[ "${LUMINA_RADEQ_DIAG_TE_K:-}" == 10020 ]] || \
    die "L6C requested-temperature switch is not 10020 K"
  [[ "${LUMINA_A210_SPRODUCER_CAPTURE:-0}" == 1 ]] || \
    die "L6C raw producer capture is not armed"
  [[ "${LUMINA_A210_INDEPENDENT_CAPTURE:-0}" == 1 ]] || \
    die "L6C independent continuum capture is not armed"
  [[ "${LUMINA_A210_LINE_SATURATION_DIAG:-0}" == 2 ]] || \
    die "L6C line-saturation union diagnostic is not armed"
  [[ "${LUMINA_A210_LINE_SATURATION_TARGET_ION:-}" == 1 ]] || \
    die "L6C target ion is not II"
  [[ "$(<"$input_dir/l6c_target_ion.txt")" == 1 ]] || \
    die "L6C target-ion provenance mismatch"
fi
for zero_knob in \
  LUMINA_NLTE_LTE_FLOOR LUMINA_NLTE_FLOOR_MODE LUMINA_NLTE_FLOOR_REG \
  LUMINA_NLTE_BK_CEIL LUMINA_NLTE_INV_CEIL LUMINA_NLTE_COLL_FLOOR \
  LUMINA_DR_FLOOR_CMS LUMINA_STAGE4_BK_CAP LUMINA_HRESP_CLAMP \
  LUMINA_TE_STEP_CLAMP LUMINA_J_CAP_FACTOR LUMINA_J_FLOOR_FACTOR \
  LUMINA_RADEQ_LINE_CULL LUMINA_NLTE_GREY_TAU \
  LUMINA_NLTE_ASSEMBLE_GPU LUMINA_NLTE_FALLBACK_TE; do
  [[ "${!zero_knob:-}" == 0 ]] || \
    die "numerical repair is armed: $zero_knob=${!zero_knob:-unset}"
done
# SUPER_CUTOFF is a declared super-level topology (model resolution), not a
# post-solve floor/cap on a physical value.  Seal it separately to keep that
# distinction explicit.
[[ "${LUMINA_SUPER_CUTOFF:-}" == 100 ]] || \
  die "super-level topology cutoff mismatch"

(
  printf 'schema=LUMINA_DET_FLIGHT_V1\n'
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'node_list=%s\n' "$SLURM_JOB_NODELIST"
  printf 'binary_sha256=%s\n' "$actual_binary"
  printf 'sigma_sha256=%s\n' "$actual_sigma"
  printf 'topion_levels_sha256=%s\n' \
    "$(sha256sum "$global_atomic_dir/topion_levels.csv" | cut -d' ' -f1)"
  printf 'ionization_reference_sha256=%s\n' \
    "$(sha256sum "$global_atomic_dir/ionization_reference.csv" | cut -d' ' -f1)"
  printf 'outer_iterations=%s\n' "$iterations"
  printf 'diagnostic_mode=%s\n' "$diagnostic_mode"
  printf 'envelope_refinements=%s\n' "$expected_refinements"
  printf 'argv=%q %q 1 %q spectrum nlte\n' "$binary" "$model_dir" "$iterations"
  if [[ "$diagnostic_mode" == A210_L6_PROBE ]]; then
    printf 'job_slurm_sha256=%s\n' \
      "$(sha256sum "$input_dir/job.slurm" | cut -d' ' -f1)"
    printf 'checker_sha256=%s\n' \
      "$(sha256sum "$targeted_gate_checker" | cut -d' ' -f1)"
    printf 'stager_sha256=%s\n' \
      "$(sha256sum "$l6_stager" | cut -d' ' -f1)"
    printf 'analyzer_sha256=%s\n' \
      "$(sha256sum "$l6_analyzer" | cut -d' ' -f1)"
    printf 'git_head=%s\n' "$(<"$input_dir/git_head.txt")"
    printf 'python3_version=%s\n' "$(python3 --version 2>&1)"
  fi
  if [[ "$diagnostic_mode" == A210_L6C_PROBE ]]; then
    printf 'job_slurm_sha256=%s\n' \
      "$(sha256sum "$input_dir/job.slurm" | cut -d' ' -f1)"
    printf 'checker_sha256=%s\n' \
      "$(sha256sum "$targeted_gate_checker" | cut -d' ' -f1)"
    printf 'stager_sha256=%s\n' \
      "$(sha256sum "$l6_stager" | cut -d' ' -f1)"
    printf 'analyzer_l6c_sha256=%s\n' \
      "$(sha256sum "$l6c_analyzer" | cut -d' ' -f1)"
    printf 'precondition_sha256=%s\n' \
      "$(sha256sum "$l6c_precondition" | cut -d' ' -f1)"
    printf 'git_head=%s\n' "$(<"$input_dir/git_head.txt")"
    printf 'python3_version=%s\n' "$(python3 --version 2>&1)"
  fi
  env | grep -E '^(LUMINA_|OMP_|CUDA_VISIBLE_DEVICES=)' | LC_ALL=C sort
) > "$run_root/RUN_FOOTER.txt"

cd "$work_dir"
set +e
"$binary" "$model_dir" 1 "$iterations" spectrum nlte \
  > "$run_root/stdout.log" 2> "$run_root/stderr.log"
model_rc=$?
set -e
printf '%s\n' "$model_rc" > "$run_root/model.rc"
if [[ "$diagnostic_mode" == A210_CANCELLATION_CENSUS ]]; then
  [[ "$model_rc" -eq 1 ]] || \
    die "census model rc=$model_rc, expected fail-closed rc=1"
  grep -q '\[A2-10\]\[BLOCKED\] event=R7_MATERIAL_UPDATE_BLOCKED' \
    "$run_root/stderr.log" || die "census lacks R7 fail-closed marker"
  grep -hE '\[FATAL\]|DET_FLIGHT_FATAL' \
    "$run_root/stdout.log" "$run_root/stderr.log" \
    > "$run_root/fatal_scan.log" || true
  [[ "$(grep -cFx '[R7][FATAL] lane=DET iter=0 rc=4' \
       "$run_root/fatal_scan.log" || true)" -eq 1 ]] || \
    die "census lacks the exact expected R7 fail-closed fatal marker"
  [[ "$(grep -cFx '[DET-TRANSACTIONAL][FATAL] cmfgen_run rc=4' \
       "$run_root/fatal_scan.log" || true)" -eq 1 ]] || \
    die "census lacks the exact expected transactional fatal marker"
  if grep -vFx -e '[R7][FATAL] lane=DET iter=0 rc=4' \
      -e '[DET-TRANSACTIONAL][FATAL] cmfgen_run rc=4' \
      "$run_root/fatal_scan.log" > "$run_root/unexpected_fatal_scan.log"; then
    die "unexpected fatal marker in cancellation census"
  fi
  python3 "$census_summarizer" \
    --log "$run_root/stderr.log" \
    --csv "$run_root/a210_cancellation_census.csv" \
    --report "$run_root/a210_cancellation_census.json" \
    --expect-phase LOWER --expect-phase UPPER \
    > "$run_root/census.stdout" 2> "$run_root/census.stderr" || \
    die "cancellation census judge failed"
  printf 'A210_CANCELLATION_CENSUS_ACCEPT status=COMPLETE_FAIL_CLOSED job_id=%s run_root=%s\n' \
    "$SLURM_JOB_ID" "$run_root" | tee "$run_root/CENSUS_VERDICT.txt"
  exit 0
fi
[[ "$model_rc" -eq 0 ]] || die "model exited rc=$model_rc"

if grep -E '\[(FATAL|BLOCKED)\]|BLOCKED_|DET_FLIGHT_FATAL' \
    "$run_root/stdout.log" "$run_root/stderr.log" > "$run_root/fatal_scan.log"; then
  die "fatal/blocked marker found in model logs"
fi
committed_count="$(grep -c '\[PHYSICS-COMPARISON\] lane=DET .*status=COMMITTED' \
  "$run_root/stderr.log" || true)"
[[ "$committed_count" -eq "$iterations" ]] || \
  die "comparison commit count $committed_count != $iterations"
[[ -z "$(find "$work_dir/physics_comparison" -name '*.tmp.*' -print -quit)" ]] || \
  die "temporary comparison file remains"

if [[ "$diagnostic_mode" == A210_TARGETED_GATE ]]; then
  python3 "$targeted_gate_checker" \
    --stdout "$run_root/stdout.log" \
    --stderr "$run_root/stderr.log" \
    --expected-devices 2 \
    --expected-refinements "$expected_refinements" \
    --report "$run_root/a210_targeted_gate_report.json" \
    > "$run_root/targeted_gate.stdout" \
    2> "$run_root/targeted_gate.stderr" || \
    die "A2-10 targeted log gate failed"
  python3 "$checker" \
    --dump-dir "$work_dir/physics_comparison" \
    --expected-iterations 1 \
    --tail-transitions 0 \
    --expected-bins 1234 \
    --report "$run_root/a210_targeted_snapshot_report.json" \
    > "$run_root/targeted_snapshot.stdout" \
    2> "$run_root/targeted_snapshot.stderr" || \
    die "A2-10 targeted comparison snapshot gate failed"
  printf 'A210_TARGETED_GATE_ACCEPT status=PASS job_id=%s run_root=%s\n' \
    "$SLURM_JOB_ID" "$run_root" | tee "$run_root/TARGETED_GATE_VERDICT.txt"
  exit 0
fi

if [[ "$diagnostic_mode" == A210_L6_PROBE ]]; then
  python3 "$targeted_gate_checker" \
    --stdout "$run_root/stdout.log" \
    --stderr "$run_root/stderr.log" \
    --expected-devices 2 \
    --expected-outer-iterations 2 \
    --expected-refinements "$expected_refinements" \
    --report "$run_root/a210_l6_targeted_gate_report.json" \
    > "$run_root/l6_targeted_gate.stdout" \
    2> "$run_root/l6_targeted_gate.stderr" || \
    die "A2-10 L6 targeted log gate failed"
  python3 "$checker" \
    --dump-dir "$work_dir/physics_comparison" \
    --expected-iterations 2 \
    --tail-transitions 0 \
    --expected-bins 1234 \
    --report "$run_root/a210_l6_snapshot_report.json" \
    > "$run_root/l6_snapshot.stdout" \
    2> "$run_root/l6_snapshot.stderr" || \
    die "A2-10 L6 comparison snapshot gate failed"
  printf 'A210_L6_PROBE_ACCEPT status=PASS job_id=%s run_root=%s\n' \
    "$SLURM_JOB_ID" "$run_root" | tee "$run_root/L6_PROBE_VERDICT.txt"
  exit 0
fi

if [[ "$diagnostic_mode" == A210_L6C_PROBE ]]; then
  python3 "$targeted_gate_checker" \
    --stdout "$run_root/stdout.log" \
    --stderr "$run_root/stderr.log" \
    --expected-devices 2 \
    --expected-outer-iterations 2 \
    --expected-refinements "$expected_refinements" \
    --report "$run_root/a210_l6c_targeted_gate_report.json" \
    > "$run_root/l6c_targeted_gate.stdout" \
    2> "$run_root/l6c_targeted_gate.stderr" || \
    die "A2-10 L6C targeted log gate failed"
  python3 "$checker" \
    --dump-dir "$work_dir/physics_comparison" \
    --expected-iterations 2 \
    --tail-transitions 0 \
    --expected-bins 1234 \
    --report "$run_root/a210_l6c_snapshot_report.json" \
    > "$run_root/l6c_snapshot.stdout" \
    2> "$run_root/l6c_snapshot.stderr" || \
    die "A2-10 L6C comparison snapshot gate failed"
  printf 'A210_L6C_PROBE_ACCEPT status=PASS job_id=%s run_root=%s\n' \
    "$SLURM_JOB_ID" "$run_root" | tee "$run_root/L6C_PROBE_VERDICT.txt"
  exit 0
fi

set +e
python3 "$checker" \
  --dump-dir "$work_dir/physics_comparison" \
  --expected-iterations "$iterations" \
  --tail-transitions 3 \
  --expected-bins 1234 \
  --report "$run_root/det_convergence_report.json" \
  > "$run_root/convergence.stdout" 2> "$run_root/convergence.stderr"
convergence_rc=$?
set -e
printf '%s\n' "$convergence_rc" > "$run_root/convergence.rc"
[[ "$convergence_rc" -eq 0 ]] || {
  printf 'DET_FLIGHT_NOT_ACCEPTED convergence_rc=%s report=%s\n' \
    "$convergence_rc" "$run_root/det_convergence_report.json" >&2
  exit "$convergence_rc"
}

printf 'DET_FLIGHT_ACCEPT status=CONVERGED job_id=%s run_root=%s\n' \
  "$SLURM_JOB_ID" "$run_root" | tee "$run_root/FLIGHT_VERDICT.txt"
