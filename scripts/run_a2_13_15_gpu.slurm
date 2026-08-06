#!/usr/bin/env bash
set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:?submit sbatch from the repository root}"
cd "$repo_root"
artifact_root="${A2_ARTIFACT_ROOT:-/gpfs/$USER/lumina/a2_13_15}"
out_dir="$artifact_root/${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
mkdir -p "$out_dir/a2_12" "$out_dir/a2_13_15"

gpu_csv="$(nvidia-smi --query-gpu=name,memory.total,uuid,driver_version \
  --format=csv,noheader,nounits | head -n1)"
printf '%s\n' "$gpu_csv" | tee "$out_dir/gpu.csv"
gpu_name="${gpu_csv%%,*}"
gpu_mem="$(printf '%s' "$gpu_csv" | cut -d, -f2 | tr -d ' ')"
if [[ "$gpu_name" != *H200* && "$gpu_name" != *H100* ]]; then
  echo "A2_13_15_INELIGIBLE_GPU name=$gpu_name" >&2
  exit 75
fi
if (( gpu_mem < 80000 )); then
  echo "A2_13_15_INSUFFICIENT_GPU_MEMORY mib=$gpu_mem required=80000" >&2
  exit 75
fi

make selftest_a2_12_gpu_lifecycle selftest_a2_13_15_contract \
  selftest_a2_13_gpu_oracle
sha256sum src/gpu_radiation_field.cu src/gpu_physics_kernels.cu \
  src/gpu_opacity_kernels.cu src/gpu_emissivity_kernels.cu \
  selftest_a2_12_gpu_lifecycle selftest_a2_13_gpu_oracle \
  > "$out_dir/source_binary.sha256"
python3 scripts/run_a2_12_gpu_lifecycle.py \
  --binary ./selftest_a2_12_gpu_lifecycle --out-dir "$out_dir/a2_12"
set +e
python3 scripts/run_a2_13_15_gpu.py \
  --oracle ./selftest_a2_13_gpu_oracle \
  --contract ./selftest_a2_13_15_contract \
  --a2-12-report "$out_dir/a2_12/gpu_lifecycle_report.json" \
  --out-dir "$out_dir/a2_13_15"
rc=$?
set -e
exit "$rc"
