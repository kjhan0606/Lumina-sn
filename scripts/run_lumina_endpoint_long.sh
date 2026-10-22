#!/bin/bash
#SBATCH --job-name=lumina-endpoint-long
#SBATCH --partition=a100_pcie
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
export PYTHONDONTWRITEBYTECODE=1 CUDA_MODULE_LOADING=EAGER
export PYTHONPATH="$PWD/scripts"
hostname
date -Iseconds
nvidia-smi --query-gpu=name,uuid,memory.free,utilization.gpu,clocks.sm,clocks.mem --format=csv
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
sha256sum --check execution.sha256
date +%s >allocation.started
trap 'date +%s >allocation.ended' EXIT
set +e
timeout -k 10 3550 srun --exact --cpu-bind=cores -n 1 -c 1 python3 -B \
    scripts/lumina_gpu_endpoint_long.py run gpu-local-plan.json --sha256 "$plan_sha"
code=$?
printf '%s\n' "$code" >actual.exit
nvidia-smi --query-gpu=name,memory.free,utilization.gpu,clocks.sm,clocks.mem --format=csv
exit "$code"
