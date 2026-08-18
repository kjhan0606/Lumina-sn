#!/usr/bin/env bash
# Same sealed production state on one requested GPU type/device count.
set -euo pipefail
umask 027

die() {
  printf 'CMF_EXACT_GPU_FAIR_FLIGHT_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 3 ]] || die "usage: $0 ABSOLUTE_RUN_ROOT DEVICE_COUNT GPU_NAME"
run_root="$1"
device_count="$2"
expected_gpu_name="$3"
[[ "$run_root" = /gpfs/"$USER"/lumina/cmf_exact_gpu_fair/pair_*/fair_* ]] || \
  die "unsafe run root: $run_root"
[[ "$device_count" =~ ^[1-9][0-9]*$ ]] || die "invalid device count"
[[ "$expected_gpu_name" == A40 || "$expected_gpu_name" == A100 ||
   "$expected_gpu_name" == H100 || "$expected_gpu_name" == H200 ]] || \
  die "unexpected GPU name: $expected_gpu_name"
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || \
  die "must run under Slurm"

input_dir="$run_root/input"
binary="$input_dir/lumina_cuda"
checker="$input_dir/check_cmf_exact_cmfgen_ab.py"
sealed_source="$(<"$input_dir/sealed_source.txt")"
model_dir="$sealed_source/input/model"
atomic_dir="$sealed_source/input/global_atomic"
resolved_env="$input_dir/resolved_lumina.exports"
expected_binary="$(<"$input_dir/binary.sha256")"
[[ "$sealed_source" = /gpfs/"$USER"/lumina/det_convergence/det1234_* ]] || \
  die "unexpected sealed source"
[[ -x "$binary" && -x "$checker" && -f "$resolved_env" &&
   -d "$model_dir" && -d "$atomic_dir" ]] || die "incomplete input"

module load cuda/13.0.2 2>/dev/null || module load cuda 2>/dev/null || \
  die "CUDA module unavailable"
(
  cd "$input_dir"
  sha256sum -c manifest.sha256
) > "$run_root/input_verify.log" 2>&1 || die "staged input SHA mismatch"
[[ "$(sha256sum "$binary" | cut -d' ' -f1)" == "$expected_binary" ]] || \
  die "binary SHA mismatch"
(
  cd "$model_dir"
  sha256sum -c "$input_dir/deck.sha256"
) > "$run_root/deck_verify.log" 2>&1 || die "sealed deck SHA mismatch"
(
  cd "$atomic_dir"
  sha256sum -c "$input_dir/topion.sha256"
) > "$run_root/topion_verify.log" 2>&1 || die "sealed atomic SHA mismatch"

nvidia-smi --query-gpu=index,name,memory.total,uuid,driver_version,compute_cap \
  --format=csv,noheader,nounits > "$run_root/gpu.csv"
[[ "$(wc -l < "$run_root/gpu.csv")" -eq "$device_count" ]] || \
  die "visible GPU count mismatch"
[[ "$(grep -c "$expected_gpu_name" "$run_root/gpu.csv")" -eq "$device_count" ]] || \
  die "visible GPU type mismatch"

work_dir="$run_root/work"
mkdir -p "$work_dir/physics_comparison"
[[ "$(find "$work_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1 ]] || \
  die "work directory is not clean"
env | LC_ALL=C sort > "$run_root/slurm_environment_before_sanitize.txt"
while IFS='=' read -r name _; do
  [[ "$name" == LUMINA_* ]] && unset "$name"
done < <(env)
# shellcheck disable=SC1090
source "$resolved_env"

export LUMINA_MODEL_DIR="$model_dir"
export LUMINA_DEPOSITION_FILE="$model_dir/deposition_cmfgen.csv"
export LUMINA_CMFGEN_SIGMA_BF="$model_dir/cmfgen_sigma_bf.bin"
export LUMINA_TOPION_LEVELS_FILE="$atomic_dir/topion_levels.csv"
export LUMINA_IONIZATION_REFERENCE_FILE="$atomic_dir/ionization_reference.csv"
export LUMINA_PHYSICS_COMPARISON_DIR="$work_dir/physics_comparison"
export LUMINA_PURE_CMFGEN_ITER=1
export LUMINA_CMF_FINE_MGPU_DEVICES="$device_count"
export LUMINA_CMF_FINE_MGPU_AB=1
export LUMINA_CMF_FINE_EXTERNAL_FIXTURE=1
export LUMINA_VALIDATION_EXIT_AFTER_R6=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

[[ "${LUMINA_ENV_STRICT:-0}" == 1 && "${LUMINA_PURE_CMFGEN:-0}" == 1 &&
   "${LUMINA_DET_TRANSACTIONAL:-0}" == 1 ]] || \
  die "sealed production lane is not armed"
[[ "${LUMINA_CMF_FINE_ALI:-}" == 64 &&
   "${LUMINA_CMF_FINE_TOL:-}" == 1e-8 &&
   "${LUMINA_CMF_FINE_LAMLO:-}" == 100 &&
   "${LUMINA_CMF_FINE_LAMHI:-}" == 20000 &&
   "${LUMINA_CMF_FINE_VDOP:-}" == 1000000 &&
   "${LUMINA_CMF_FINE_PPD:-}" == 12 ]] || \
  die "fine-grid physical contract drift"
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
env | grep -E '^(LUMINA_|OMP_|CUDA_VISIBLE_DEVICES=)' | LC_ALL=C sort \
  > "$run_root/execution_environment.txt"

nvidia-smi \
  --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem \
  --format=csv,noheader,nounits --loop=5 \
  > "$run_root/gpu_samples.csv" 2> "$run_root/gpu_monitor.stderr" &
monitor_pid=$!
stop_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap stop_monitor EXIT

cd "$work_dir"
set +e
model_start_utc="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
model_start_seconds=$SECONDS
"$binary" "$model_dir" 1 1 spectrum nlte \
  > "$run_root/stdout.log" 2> "$run_root/stderr.log"
model_rc=$?
model_wall_seconds=$((SECONDS - model_start_seconds))
set -e
printf 'model_start_utc=%s\nmodel_wall_seconds=%d\n' \
  "$model_start_utc" "$model_wall_seconds" > "$run_root/resource_usage.txt"
printf '%s\n' "$model_rc" > "$run_root/model.rc"
stop_monitor
trap - EXIT
[[ "$model_rc" -eq 0 ]] || die "model process failed rc=$model_rc"

python3 "$checker" --stderr "$run_root/stderr.log" \
  --expected-devices "$device_count" --require-exit-after-r6 \
  --require-external-fixture \
  --report "$run_root/ab_report.json" \
  > "$run_root/checker.stdout" 2> "$run_root/checker.stderr" || \
  die "CPU/multi-GPU A/B verdict failed"
[[ -f "$work_dir/lumina_cmf_fine_external_fixture.csv" ]] || \
  die "external J_nu fixture missing"

sha256sum "$run_root/gpu.csv" "$run_root/gpu_samples.csv" \
  "$run_root/execution_environment.txt" "$run_root/resource_usage.txt" \
  "$run_root/stdout.log" "$run_root/stderr.log" "$run_root/model.rc" \
  "$run_root/ab_report.json" "$run_root/checker.stdout" \
  "$work_dir/lumina_cmf_fine_external_fixture.csv" \
  "$run_root/checker.stderr" > "$run_root/output.sha256"
printf '%s\n' \
  "CMF_EXACT_GPU_FAIR_FLIGHT_OK job_id=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST" \
  "binary_sha256=$expected_binary" \
  "sealed_source=$sealed_source" \
  "gpu_name=$expected_gpu_name devices=$device_count cells=100655650 bins=2013113" \
  "exit_after_r6=PASS floor=0 clamp=0 jitter=0 repair=0" \
  > "$run_root/RUN_FOOTER.txt"
