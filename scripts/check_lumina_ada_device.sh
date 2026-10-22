#!/bin/bash
# 소형 장치/원입력 대조만 수행한다. 호출자가 부모전체60초로 제한한다.
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -euo pipefail
cd "$1"
test "$(hostname)" = LagEunha
test ! -e verification.started
test ! -e gpu-local.started
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_MODULE_LOADING=EAGER
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/scripts"
export LUMINA_COMPACT_GPU_LIBRARY="$PWD/history_cuda.so"
export LUMINA_COMPACT_NATIVE_LIBRARY=/gpfs/kjhan/cmfgen_runs/lumina_source_projection_GPU_20261022-xQNYu9/native.so
export LUMINA_COMPACT_GPU_RECORD_ROOT="$PWD"
export LUMINA_COMPACT_RANGE_TEST=1 LUMINA_ENDPOINT_CACHE=1 LUMINA_WARP_MEASURE=1
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = GPU-42b3367f-1df7-2b2d-7961-7781c87b022d
test "$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader)" = 8.9
sha256sum --check execution.sha256
set -C
date +%s >verification.started
printf '%s\n' "$$" >verification.pid
set +C
trap 'date +%s >verification.ended' EXIT
set +e
selection=()
if test "${LUMINA_ADA_MISSING_ONLY:-0}" = 1; then
    selection=(-k 'warp_polynomial or warp_positive or successful_primary or limit_fallback')
elif test "${LUMINA_ADA_MISSING_ONLY:-0}" = 2; then
    selection=(-k limit_fallback)
fi
timeout -k 5 20 python3 -B -m pytest -x -p no:cacheprovider -q \
    tests/test_lumina_gpu_compact_device.py tests/test_lumina_gpu_hybrid_device.py "${selection[@]}" >device-test.log 2>&1
code=$?
printf '%s\n' "$code" >device-test.exit
if test "$code" -ne 0; then exit "$code"; fi
timeout -k 5 25 python3 -B scripts/lumina_hybrid_cost_diag.py "$PWD" >gpu-cost.log 2>&1
code=$?
printf '%s\n' "$code" >checks.exit
exit "$code"
