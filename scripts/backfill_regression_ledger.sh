#!/usr/bin/env bash
set -euo pipefail

# Offline CPU-only backfill.  The guard is intentionally fail-closed because the
# 58 level-population files are roughly 4 GB in aggregate.
if [[ "${REGRESSION_LEDGER_COMPUTE_NODE:-0}" != "1" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "REFUSED: backfill must run inside a compute-node job step (SLURM_JOB_ID absent)." >&2
    exit 2
  fi
  if [[ -z "${SLURMD_NODENAME:-}" && -z "${SLURM_STEP_ID:-}" ]]; then
    echo "REFUSED: allocation variables exist but no slurmd/job-step marker was found; use srun." >&2
    exit 2
  fi
fi

host_name=$(hostname -s)
if [[ "$host_name" == *login* ]]; then
  echo "REFUSED: hostname '$host_name' looks like a login node." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=""
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

shopt -s nullglob
candidates=("$repo_root"/logs/coevolve_consume_a10_kx_* /gpfs/kjhan/lumina_runner2/scratch/*)
runs=()
for candidate in "${candidates[@]}"; do
  [[ -d "$candidate" ]] && runs+=("$candidate")
done
expected=${REGRESSION_LEDGER_EXPECTED_RUNS:-69}
if [[ ${#runs[@]} -ne $expected ]]; then
  echo "REFUSED: discovered ${#runs[@]} run directories, expected $expected; refusing a silent partial backfill." >&2
  exit 2
fi

echo "CPU-only append-only backfill: ${#runs[@]} runs; CUDA_VISIBLE_DEVICES is empty."
python3 "$script_dir/regression_ledger.py" \
  --repo-root "$repo_root" \
  --ledger "$repo_root/validation/regression_ledger/ledger.jsonl" \
  "${runs[@]}"
