#!/bin/bash
#SBATCH --job-name=lumina-slab-local
#SBATCH --partition=a100,a100_pcie
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -euo pipefail
cd "$1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_MODULE_LOADING=EAGER
export PYTHONPATH="$PWD/scripts"
export LUMINA_COMPACT_GPU_LIBRARY="$PWD/history_cuda.so"
export LUMINA_COMPACT_GPU_RECORD_ROOT="$PWD"
export LUMINA_SLAB_CPU_LIBRARY=/gpfs/kjhan/cmfgen_runs/lumina_extended_4h_20261022-rJGMVR/driver.so
sha256sum --check execution.sha256 >source-check.log
if ! test -s history_cuda.so; then bash compile.sh "$PWD"; fi
hostname
date -Iseconds
nvidia-smi --query-gpu=name,uuid,memory.free,utilization.gpu --format=csv
timeout -k 5 90 srun --exact --cpu-bind=cores -n 1 -c 1 python3 -B -m pytest \
    -x -p no:cacheprovider -q --basetemp="$PWD/device-pytest" \
    tests/test_lumina_gpu_slab_local_device.py >device-test.log 2>&1
slab_interval_seconds=$((840-SECONDS))
if test "$slab_interval_seconds" -gt 720; then slab_interval_seconds=720; fi
test "$slab_interval_seconds" -gt 0
timeout -k 5 800 srun --exact --cpu-bind=cores -n 1 -c 1 python3 -B \
    scripts/lumina_slab_local_actual.py "$PWD" --seconds "$slab_interval_seconds"
