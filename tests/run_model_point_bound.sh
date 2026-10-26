#!/usr/bin/env bash
# CPU1/4GiB/최대2시간에서 전ν 율 연결을 시도한다. 수송/물질 실행 아님.
set -euo pipefail
ulimit -c 0
project=$(realpath "${1:?project}")
output=$(realpath "${2:?new-output}")
test "${SLURM_CPUS_PER_TASK:-}" = 1
test "${SLURM_MEM_PER_NODE:-}" = 4096
test ! -e "$output/point.so"
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$project"
gcc -O2 -std=c11 -fPIC -shared -Wl,-z,defs -Isrc -o "$output/point.so" \
    src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c -lm
python3 scripts/run_lumina_model_point_bound.py \
    --base /gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json \
    --j /gpfs/kjhan/cmfgen_runs/lumina_resident_field_20261026-qFP5rJ/run/J.npy \
    --library "$output/point.so" --out "$output/run" --wall-seconds "${LUMINA_BOUND_WALL_SECONDS:-300}"
