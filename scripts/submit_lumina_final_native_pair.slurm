#!/bin/bash -l
# 완료 물질에 대한 native h05/h025 독립 열 평가. 한 할당·재시도0.
#SBATCH --job-name=lumina-final-native-pair
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
set -euo pipefail
umask 027
module purge
module load gnu13/13.2.0
ulimit -s unlimited
ulimit -c 0
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
project=$(realpath "${1:?sealed source}")
exec python3 -B "$project/scripts/run_lumina_final_native_pair.py" --config "${2:?config}" --sha "${3:?config hash}"
