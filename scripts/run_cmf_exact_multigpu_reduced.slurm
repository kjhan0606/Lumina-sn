#!/usr/bin/env bash
#SBATCH --job-name=cmf_mgpu_reduced
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A40:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
umask 027

die() {
  printf 'CMF_MGPU_REDUCED_FLIGHT_FATAL %s\n' "$*" >&2
  exit 70
}

[[ $# -eq 1 ]] || die "usage: $0 ABSOLUTE_RUN_ROOT"
run_root="$1"
[[ "$run_root" = /* && "$run_root" != / && "$run_root" != /gpfs ]] || \
  die "unsafe run root: $run_root"
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || \
  die "must run under Slurm"
input_dir="$run_root/input"
binary="$input_dir/bench_cmf_exact_multigpu_reduced"
geometry="$input_dir/geometry.csv"
[[ -x "$binary" && -f "$geometry" && -f "$input_dir/manifest.sha256" ]] || \
  die "incomplete staged input"

module load cuda/13.0.2 2>/dev/null || module load cuda 2>/dev/null || \
  die "CUDA module unavailable"
command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
(
  cd "$input_dir"
  sha256sum -c manifest.sha256
) > "$run_root/input_verify.log" 2>&1 || die "input SHA mismatch"

nvidia-smi --query-gpu=index,name,memory.total,uuid,driver_version,compute_cap \
  --format=csv,noheader,nounits > "$run_root/gpu.csv"
[[ "$(wc -l < "$run_root/gpu.csv")" -ge 4 ]] || \
  die "fewer than four physical GPUs visible"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die "CUDA_VISIBLE_DEVICES is unset"
printf '%s\n' \
  "job_id=$SLURM_JOB_ID" \
  "node=$SLURM_JOB_NODELIST" \
  "partition=${SLURM_JOB_PARTITION:-unknown}" \
  "cuda_visible_devices=$CUDA_VISIBLE_DEVICES" \
  "reduced_bins=${CMF_MGPU_REDUCED_BINS:-8192}" \
  "epoch_frequency_parallel=${CMF_MGPU_REDUCED_EPOCH:-0}" \
  "epoch_block=${CMF_MGPU_EPOCH_BLOCK:-128}" \
  "epoch_batch=${CMF_MGPU_EPOCH_BATCH:-64}" \
  "epoch_replay=${CMF_MGPU_EPOCH_REPLAY:-32}" \
  "coefficient_contract=finite_synthetic_not_cmfgen_physics" \
  > "$run_root/execution_environment.txt"

export CMF_MGPU_REDUCED_BINS="${CMF_MGPU_REDUCED_BINS:-8192}"
export CMF_MGPU_REDUCED_DEVICES=4
export CMF_MGPU_REDUCED_REFINEMENTS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

trace="$run_root/vram_trace.csv"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used \
  --format=csv,noheader,nounits --loop-ms=200 > "$trace" 2>&1 &
sampler_pid=$!
stop_sampler() {
  if kill -0 "$sampler_pid" 2>/dev/null; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
trap stop_sampler EXIT

set +e
CMF_MGPU_REDUCED_OUTPUT="$run_root/result_a.bin" \
  "$binary" "$geometry" > "$run_root/bench_a.log" 2> "$run_root/bench_a.err"
rc_a=$?
CMF_MGPU_REDUCED_OUTPUT="$run_root/result_b.bin" \
  "$binary" "$geometry" > "$run_root/bench_b.log" 2> "$run_root/bench_b.err"
rc_b=$?
set -e
stop_sampler
trap - EXIT
[[ "$rc_a" -eq 0 && "$rc_b" -eq 0 ]] || \
  die "benchmark failed rc_a=$rc_a rc_b=$rc_b"
cmp -s "$run_root/result_a.bin" "$run_root/result_b.bin" || \
  die "binary result determinism mismatch"
rg '^CMF_MGPU_REDUCED_RESULT PASS ' "$run_root/bench_a.log" \
  > "$run_root/result_a.txt"
rg '^CMF_MGPU_REDUCED_RESULT PASS ' "$run_root/bench_b.log" \
  > "$run_root/result_b.txt"
cmp -s "$run_root/result_a.txt" "$run_root/result_b.txt" || \
  die "reported result determinism mismatch"
if rg -q 'FAIL|numerical_repairs=[1-9]' \
    "$run_root/bench_a.log" "$run_root/bench_a.err" \
    "$run_root/bench_b.log" "$run_root/bench_b.err"; then
  die "failure or numerical repair marker present"
fi
python3 "$input_dir/summarize_nvidia_smi_vram.py" "$trace" \
  > "$run_root/vram_summary.txt" || die "VRAM summary failed"

sha256sum "$run_root/gpu.csv" "$run_root/execution_environment.txt" \
  "$run_root/input_verify.log" "$run_root/bench_a.log" \
  "$run_root/bench_a.err" "$run_root/bench_b.log" \
  "$run_root/bench_b.err" "$run_root/result_a.bin" \
  "$run_root/result_b.bin" "$run_root/result_a.txt" \
  "$run_root/result_b.txt" "$trace" "$run_root/vram_summary.txt" \
  > "$run_root/output.sha256"
printf '%s\n' \
  "CMF_MGPU_REDUCED_FLIGHT_OK job_id=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST devices=4" \
  "binary_sha256=$(sha256sum "$binary" | cut -d' ' -f1)" \
  "result_sha256=$(sha256sum "$run_root/result_a.bin" | cut -d' ' -f1)" \
  "coefficient_contract=finite_synthetic_not_cmfgen_physics" \
  "repair=0 floor=0 clamp=0 jitter=0" \
  > "$run_root/RUN_FOOTER.txt"
