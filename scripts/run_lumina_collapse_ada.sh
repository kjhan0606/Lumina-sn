#!/bin/bash
# 기본 부모 timeout -k5 3410. 세 번째 인자가 있으면 -k5 14695(총14700초).
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -euo pipefail
cd "$1"
test "$(hostname)" = LagEunha
test ! -e parent.started
test ! -e gpu-local.started
exec 9>/tmp/lumina_hybrid_ada_GPU-42b3367f.lock
flock -n 9
plan_sha=$2
extended_args=()
actual_seconds=3350
if test "$#" -eq 3; then
    extended_args=(--extended-after-gpu-seconds "$3")
    actual_seconds=14600
fi
test "$#" -ge 2 && test "$#" -le 3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_MODULE_LOADING=EAGER
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/scripts"
export LUMINA_COMPACT_GPU_LIBRARY="$PWD/history_cuda.so"
export LUMINA_COMPACT_NATIVE_LIBRARY=/gpfs/kjhan/cmfgen_runs/lumina_source_projection_GPU_20261022-xQNYu9/native.so
export LUMINA_COMPACT_GPU_RECORD_ROOT="$PWD"
export LUMINA_COMPACT_RANGE_TEST=1 LUMINA_ENDPOINT_CACHE=1 LUMINA_WARP_MEASURE=1
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = GPU-42b3367f-1df7-2b2d-7961-7781c87b022d
sha256sum --check --status execution.sha256
set -C
date +%s >parent.started
printf '%s\n' "$$" >parent.pid
set +C
trap 'date +%s >parent.ended' EXIT
set +e
timeout -k 5 20 python3 -B -m pytest -x -p no:cacheprovider -q \
    tests/test_lumina_gpu_compact_device.py tests/test_lumina_gpu_hybrid_device.py >device-test.log 2>&1
code=$?;printf '%s\n' "$code" >device-test.exit
if test "$code" -ne 0; then printf '%s\n' "$code" >parent.exit;exit "$code";fi
timeout -k 5 35 python3 -B scripts/lumina_gpu_collapse_actual.py check "$PWD" "${extended_args[@]}" >gpu-cost.log 2>&1
code=$?;printf '%s\n' "$code" >checks.exit
if test "$code" -ne 0; then printf '%s\n' "$code" >parent.exit;exit "$code";fi
timeout -k 10 "$actual_seconds" python3 -B scripts/lumina_gpu_collapse_actual.py run gpu-local-plan.json --sha256 "$plan_sha" "${extended_args[@]}"
code=$?;printf '%s\n' "$code" >parent.exit
exit "$code"
