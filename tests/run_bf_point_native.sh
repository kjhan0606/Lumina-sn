#!/usr/bin/env bash
# 이미 완료된 물질/J의 선택 열만 읽는 CPU1 시험. 수송 실행 없음.
set -euo pipefail
ulimit -c 0
project=$(realpath "${1:?project}")
output=$(realpath "${2:?new-output}")
test "${SLURM_CPUS_PER_TASK:-}" = 1
test "${SLURM_MEM_PER_NODE:-}" = 4096
test ! -e "$output/source-before.sha256"
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$project"
sha256sum src/bf_rate_jnu.c src/bf_rate_jnu.h src/radiation_field.c src/seed_capability.c \
    scripts/cmfgen_model_continuum.py scripts/lumina_native90_rates.py \
    scripts/lumina_native_columns.py tests/bf_point_native_test.py tests/run_bf_point_native.sh \
    > "$output/source-before.sha256"
gcc -O1 -std=c11 -fPIC -shared -Wl,-z,defs -Isrc -o "$output/bf_point.so" \
    src/bf_rate_jnu.c src/radiation_field.c src/seed_capability.c -lm
python3 tests/bf_point_native_test.py \
    --base /gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/gpu-local-plan.json \
    --j /gpfs/kjhan/cmfgen_runs/lumina_resident_field_20261026-qFP5rJ/run/J.npy \
    --library "$output/bf_point.so" --out "$output/native-result.json"
sha256sum -c "$output/source-before.sha256" > "$output/source-after-check.txt"
sha256sum "$output/native-result.json" "$output/bf_point.so"
