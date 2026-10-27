#!/bin/bash
# 기존 GPU 작업 종료 뒤 미완료 수송만 이어간다. 추가 GPU1 최대24시간.
#SBATCH --job-name=lumina-field-continuation
#SBATCH --partition=h100,h200,a100,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -eo pipefail
# 시스템 프로필은 미정의 셸 변수를 읽는다. 모듈 초기화 뒤 nounset을 켠다.
source /etc/profile
module purge
module load gnu13 cuda/13.0.2
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_MODULE_LOADING=EAGER
parent=${1:?이어계산 출력 위치 필요}
commit=${2:?고정 커밋 필요}
repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
mkdir "$parent/control"
git -C "$repo" archive "$commit" scripts/lumina_continue_accepted_field.py | tar -x -C "$parent/control"
export PYTHONPATH=/gpfs/kjhan/cmfgen_runs/lumina_accepted_feedback_1224468_20261027/code/scripts
python3 "$parent/control/scripts/lumina_continue_accepted_field.py" --parent "$parent" \
  --staged /gpfs/kjhan/cmfgen_runs/lumina_accepted_feedback_1224468_20261027 \
  --preparation /gpfs/kjhan/cmfgen_runs/lumina_accepted_sources_20261027-b5eCNv/run/boundary/preparation.json \
  --prep-sha 4707762a819fee7f013fb8655ad5d6a738c9f64cfd1ddc897e5138caa9478887 \
  --cache /gpfs/kjhan/cmfgen_runs/lumina_phase_prepare_20261026-8fMV0m/run/cache --seconds 85800
