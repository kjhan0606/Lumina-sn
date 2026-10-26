#!/usr/bin/env bash
# CPU1/4GiB/300초의 회귀+점값 BB 생산. 수송 실행 없음.
set -euo pipefail
ulimit -c 0
project=$(realpath "${1:?project}")
output=$(realpath "${2:?new-output}")
test "${SLURM_CPUS_PER_TASK:-}" = 1
test "${SLURM_MEM_PER_NODE:-}" = 4096
test ! -e "$output/lines.so"
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$project"
sha256sum src/model_source_kernels.c src/model_gaussian_profile.h \
    scripts/lumina_model_point_lines.py scripts/run_lumina_model_point_lines.py \
    scripts/cmfgen_model_sources.py \
    scripts/run_lumina_model_point_bound.py tests/test_model_point_bound_recovery.py \
    scripts/build_cmfgen_source_kernels.sh tests/test_model_point_lines.py tests/test_cmfgen_model_sources.py \
    > "$output/source-before.sha256"
python3 -m pytest -q -p no:cacheprovider tests/test_model_point_lines.py \
    tests/test_cmfgen_model_sources.py tests/test_model_point_bound_recovery.py --basetemp "$output/pytest-work"
gcc -O2 -std=c11 -fPIC -shared -Wl,-z,defs -Isrc -o "$output/lines.so" src/model_source_kernels.c -lm
python3 scripts/run_lumina_model_point_lines.py \
    --base /gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json \
    --j /gpfs/kjhan/cmfgen_runs/lumina_resident_field_20261026-qFP5rJ/run/J.npy \
    --library "$output/lines.so" --out "$output/run"
sha256sum -c "$output/source-before.sha256" > "$output/source-after-check.txt"
