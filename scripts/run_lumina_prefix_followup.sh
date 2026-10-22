#!/bin/bash
#SBATCH --job-name=lumina-prefix-followup
#SBATCH --partition=a100_pcie
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:58:00
#SBATCH --no-requeue
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -euo pipefail
cd "$1"
plan_sha=$2
test ! -e allocation.started
test ! -e gpu-local.started
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_MODULE_LOADING=EAGER
export PYTHONPATH="$PWD/scripts"
export LUMINA_COMPACT_GPU_LIBRARY="$PWD/history_cuda.so"
export LUMINA_COMPACT_GPU_RECORD_ROOT="$PWD"
export LUMINA_ENDPOINT_CACHE=1
hostname
date -Iseconds
nvidia-smi --query-gpu=name,uuid,memory.free,utilization.gpu --format=csv
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
sha256sum --check execution.sha256
date +%s >allocation.started
trap 'date +%s >allocation.ended' EXIT
set +e
timeout -k 5 20 srun --exact --cpu-bind=cores -n 1 -c 1 python3 -B -m pytest -p no:cacheprovider -q \
    tests/test_lumina_gpu_prefix_device.py >device-test.log 2>&1
code=$?
printf '%s\n' "$code" >device-test.exit
if test "$code" -ne 0; then exit "$code"; fi
timeout -k 10 3420 srun --exact --cpu-bind=cores -n 1 -c 1 python3 -B \
    scripts/lumina_gpu_prefix_actual.py run gpu-local-plan.json --sha256 "$plan_sha"
code=$?
printf '%s\n' "$code" >actual.exit
exit "$code"
