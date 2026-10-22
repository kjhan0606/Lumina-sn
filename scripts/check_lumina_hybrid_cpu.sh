#!/bin/bash
set -euo pipefail
cd "$1"
test "$(hostname)" = LagEunha
test ! -e host-test.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/scripts"
ulimit -v 4194304
timeout -k 5 180 taskset -c 25 python3 -B -m pytest -p no:cacheprovider -q \
    tests/test_lumina_gpu_hybrid.py tests/test_lumina_warp_host.py tests/test_lumina_warp_device.py \
    tests/test_lumina_native_bin_stream.py tests/test_lumina_gpu_native_bin.py \
    tests/test_lumina_gpu_compact.py tests/test_lumina_gpu_measure_local.py \
    tests/test_lumina_characteristic_linear.py tests/test_lumina_linear_native.py \
    tests/test_lumina_source_measure.py tests/test_lumina_endpoint_cache.py \
    tests/test_lumina_gpu_measure.py >host-test.log 2>&1
