#!/bin/bash
# 새 격리 root에서만 수행하는 CPU 직접시험과 원 고정 표본 비교다.
source /etc/profile
module purge
module load gnu13
set -euo pipefail
run_root=$1
test "$(hostname | tr '[:upper:]' '[:lower:]')" = lageunha
cd "$run_root"
test ! -e host-test.log
test ! -e cpu-cost.json
test ! -e cpu-probe.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$PWD/scripts:/gpfs/kjhan/cmfgen_runs/lumina_envelope_query_20261022-cwIcgC/scripts:/gpfs/kjhan/cmfgen_runs/lumina_compact_native_doubletime_20261022-nd9sC6/scripts"
(
    ulimit -v 4194304
    timeout -k 5 300 taskset -c 0 python3 -B -m pytest -q -p no:cacheprovider \
        --basetemp=host-work tests/test_lumina_source_measure.py \
        tests/test_lumina_characteristic_linear.py >host-test.log 2>&1
)
ulimit -v 16777216
timeout -k 5 60 taskset -c 0 c++ -O2 -std=c++17 -ffp-contract=off -Wall -Wextra -Werror \
    -fPIC -shared -DCMF_GPU_ENVELOPE_UPPER -DCMF_GPU_ADAPTIVE_LINEAR_EXP -Isrc \
    tests/native_cost_probe.cpp -o cpu-probe.so >cpu-compile.log 2>&1
timeout -k 5 300 taskset -c 0 python3 -B scripts/lumina_native_cost_diag.py cpu "$PWD" >cpu-cost.log 2>&1
