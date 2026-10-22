#!/bin/bash
#SBATCH --job-name=lumina-cap-a100
#SBATCH --partition=a100_pcie
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:02:54
#SBATCH --no-requeue
# 위174초요청은Slurm에서3분으로올림된다. 새실행의예약은180초로계상한다.
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -euo pipefail
run_root=$1
cd "$run_root"
test ! -e gpu-window.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_MODULE_LOADING=EAGER
export PYTHONPATH="$PWD/scripts:/gpfs/kjhan/cmfgen_runs/lumina_envelope_query_20261022-cwIcgC/scripts:/gpfs/kjhan/cmfgen_runs/lumina_compact_native_doubletime_20261022-nd9sC6/scripts"
export LUMINA_COMPACT_GPU_LIBRARY="$PWD/diagnostic.so"
export LUMINA_COMPACT_NATIVE_LIBRARY=/gpfs/kjhan/cmfgen_runs/lumina_source_projection_GPU_20261022-xQNYu9/native.so
export LUMINA_COMPACT_GPU_RECORD_ROOT="$PWD"
export LUMINA_COMPACT_SHARED_CORRECTNESS=1 LUMINA_COMPACT_RANGE_TEST=1
hostname
date -Iseconds
nvidia-smi --query-gpu=name,uuid,memory.free,utilization.gpu,mig.mode.current --format=csv
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
sha256sum --check execution.sha256
timeout -k 10 148 srun --exact --cpu-bind=cores -n 1 -c 1 python3 -B gpu-window.py "$PWD"
