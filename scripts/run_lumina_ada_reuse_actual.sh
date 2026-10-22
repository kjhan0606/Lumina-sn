#!/bin/bash
# 소형검증 재사용. 외부 부모3580+kill10초와 정리여유를 합쳐3598초 이하다.
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -euo pipefail
cd "$1"
test "$(hostname)" = LagEunha
plan_sha=$2
test ! -e gpu-local.started
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_MODULE_LOADING=EAGER
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/scripts"
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = GPU-42b3367f-1df7-2b2d-7961-7781c87b022d
sha256sum --check --status execution.sha256
timeout -k 10 3500 python3 -B scripts/lumina_gpu_ada_reuse_actual.py run gpu-local-plan.json --sha256 "$plan_sha"
