#!/usr/bin/env bash
set -euo pipefail
umask 027

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$repo_root/bench_cmf_exact_multigpu_reduced"
driver="$repo_root/scripts/run_cmf_exact_multigpu_reduced_split.slurm"
summary="$repo_root/scripts/summarize_nvidia_smi_vram.py"
compare="$repo_root/scripts/compare_cmf_exact_multigpu_partitions.py"
geometry="$repo_root/data/tardis_reference_toy06_19p48d/geometry.csv"
for path in "$binary" "$driver" "$summary" "$compare" "$geometry"; do
  [[ -f "$path" ]] || { printf 'missing input: %s\n' "$path" >&2; exit 2; }
done
[[ -x "$binary" ]] || { printf 'binary is not executable: %s\n' "$binary" >&2; exit 2; }

binary_sha="$(sha256sum "$binary" | cut -d' ' -f1)"
stamp="$(date -u +%Y%m%dT%H%M%S%NZ)_p$$"
run_root="/gpfs/${USER}/lumina/cmf_multigpu_partition_ab/ab_${stamp}_${binary_sha:0:12}"
[[ "$run_root" = /gpfs/"${USER}"/lumina/cmf_multigpu_partition_ab/ab_* ]] || {
  printf 'unsafe run root: %s\n' "$run_root" >&2
  exit 2
}
mkdir -p "$run_root/input" "$run_root/equal" "$run_root/weighted"
install -m 0750 "$binary" "$run_root/input/bench_cmf_exact_multigpu_reduced"
install -m 0640 "$repo_root/src/cmf_exact_multigpu.cu" \
  "$repo_root/src/cmf_exact_multigpu.h" \
  "$repo_root/src/cmf_exact_sliding.c" \
  "$repo_root/src/cmf_exact_sliding.h" \
  "$repo_root/src/cmf_error_envelope.c" \
  "$repo_root/src/cmf_error_envelope.h" \
  "$repo_root/tests/cmf_exact_multigpu_reduced_bench.cu" \
  "$summary" "$compare" "$geometry" "$run_root/input/"
install -m 0750 "$driver" "$run_root/input/job.slurm"
(
  cd "$run_root/input"
  sha256sum bench_cmf_exact_multigpu_reduced \
    cmf_error_envelope.c cmf_error_envelope.h \
    cmf_exact_multigpu.cu cmf_exact_multigpu.h \
    cmf_exact_sliding.c cmf_exact_sliding.h \
    cmf_exact_multigpu_reduced_bench.cu geometry.csv \
    summarize_nvidia_smi_vram.py compare_cmf_exact_multigpu_partitions.py \
    job.slurm > manifest.sha256
)

node="${CMF_MGPU_NODELIST:-syn07}"
[[ "$node" =~ ^[A-Za-z0-9._-]+$ ]] || {
  printf 'invalid CMF_MGPU_NODELIST: %s\n' "$node" >&2
  exit 2
}
allocated_gpus="${CMF_MGPU_ALLOCATED_GPUS:-5}"
[[ "$allocated_gpus" =~ ^[4-7]$ ]] || {
  printf 'invalid CMF_MGPU_ALLOCATED_GPUS: %s\n' "$allocated_gpus" >&2
  exit 2
}
export CMF_MGPU_DEVICE_ORDER="${CMF_MGPU_DEVICE_ORDER:-0,1,2,3}"

equal_job="$(sbatch --parsable --nodelist="$node" \
  --partition=a40 --nodes=1 --ntasks=1 --gres=gpu:A40:"$allocated_gpus" \
  --cpus-per-task=16 --mem=64G --time=01:00:00 \
  --job-name=cmf_mgpu_equal \
  --output="$run_root/equal/slurm-%j.out" \
  --error="$run_root/equal/slurm-%j.err" \
  "$run_root/input/job.slurm" "$run_root" four equal)"
weighted_job="$(sbatch --parsable --dependency=afterok:"$equal_job" \
  --nodelist="$node" \
  --partition=a40 --nodes=1 --ntasks=1 --gres=gpu:A40:"$allocated_gpus" \
  --cpus-per-task=16 --mem=64G --time=01:00:00 \
  --job-name=cmf_mgpu_weighted \
  --output="$run_root/weighted/slurm-%j.out" \
  --error="$run_root/weighted/slurm-%j.err" \
  "$run_root/input/job.slurm" "$run_root" four weighted)"
printf 'CMF_MGPU_PARTITION_AB_SUBMITTED equal_job=%s weighted_job=%s run_root=%s binary_sha256=%s node=%s allocated_gpus=%s device_order=%s\n' \
  "$equal_job" "$weighted_job" "$run_root" "$binary_sha" "$node" \
  "$allocated_gpus" "$CMF_MGPU_DEVICE_ORDER"
