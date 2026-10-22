#!/bin/bash
# lageunha의 격리 후보 root에서만 CPU 직접시험과 고정35열을 수행한다.
set -euo pipefail
cd "$1"
test "$(hostname)" = LagEunha
test ! -e cpu-cost.json
test ! -e cpu-probe.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/scripts"
(
    ulimit -v 4194304
    timeout -k 5 180 taskset -c 25 python3 -B -m pytest -p no:cacheprovider -q \
        tests/test_lumina_native_bin_stream.py >host-test.log 2>&1
)
ulimit -v 16777216
timeout -k 5 60 taskset -c 25 c++ -O2 -std=c++17 -ffp-contract=off -Wall -Wextra -Werror \
    -fPIC -shared -DCMF_GPU_ENVELOPE_UPPER -DCMF_GPU_ADAPTIVE_LINEAR_EXP \
    -DCMF_GPU_ENDPOINT_CACHE -DCMF_GPU_NATIVE_BIN_STREAM -Isrc tests/native_cost_probe.cpp \
    -o cpu-probe.so >cpu-compile.log 2>&1
timeout -k 5 300 taskset -c 25 python3 -B scripts/lumina_native_cost_diag.py cpu "$PWD" >cpu-actual.log 2>&1
timeout -k 5 30 taskset -c 25 python3 -B scripts/compare_lumina_native_cost.py \
    /gpfs/kjhan/cmfgen_runs/lumina_endpoint_followup_20261022-0YoWCi/cpu-cost.json \
    cpu-cost.json --output cpu-comparison.json >comparison.log 2>&1
