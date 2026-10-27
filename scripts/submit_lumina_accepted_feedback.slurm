#!/bin/bash
# 새 계수가 실제 완성된 뒤에만 제출한다. 같은 할당에서 최대2F·합계12GPUh.
#SBATCH --job-name=lumina-accepted-field
#SBATCH --partition=h100,h200,a100,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
lumina_stage=${1:?새 봉인 GPU root 필요}
lumina_preparation=${2:?완성 새 경계 기록 필요}
lumina_preparation_sha=${3:?완성 새 경계 SHA 필요}
cd "$lumina_stage"
exec bash code/scripts/run_lumina_resident_field.sh "$PWD" 43200 --feedback \
    --cache /gpfs/kjhan/cmfgen_runs/lumina_phase_prepare_20261026-8fMV0m/run/cache \
    --accepted-input "$lumina_preparation" --accepted-input-sha256 "$lumina_preparation_sha" --max-sweeps 2
