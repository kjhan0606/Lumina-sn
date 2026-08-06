#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p validation/a2_12
gpu_row="$(nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader | head -n1)"
printf '%s\n' "$gpu_row"
gpu_name="${gpu_row%%,*}"
gpu_mem="$(printf '%s' "$gpu_row" | awk -F, '{gsub(/[^0-9]/,"",$2); print $2}')"
if [[ "$gpu_name" == *A40* ]] || (( gpu_mem < 80000 )); then
  printf 'A2_12_GPU_MEMORY_INELIGIBLE name=%s memory_mib=%s\n' "$gpu_name" "$gpu_mem" >&2
  exit 75
fi
printf 'A2_12_FULL_NLTE_DRIVER READY lifecycle_only=1 physical_migrated_kernels=0\n'
printf 'UNVERIFIED_INPUT_DECK driver requires driving-seat full-NLTE deck selection\n'
exit 75
