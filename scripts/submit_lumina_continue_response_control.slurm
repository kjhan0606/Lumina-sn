#!/bin/bash
# 이어계산 뒤 GPU 없이 물질응답 단회 제출 여부만 처리한다.
#SBATCH --job-name=lumina-response-control
#SBATCH --partition=a10,a40,h100,h200,a100,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --no-requeue
set -euo pipefail
parent=${1:?이어계산 위치 필요}
commit=${2:?고정 코드 필요}
repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
mkdir "$parent/response-control-code"
git -C "$repo" archive "$commit" scripts/lumina_continue_response_dispatch.py | tar -x -C "$parent/response-control-code"
python3 "$parent/response-control-code/scripts/lumina_continue_response_dispatch.py" \
  --parent "$parent" --control /gpfs/kjhan/cmfgen_runs/lumina_bundle_dispatch_1224468_20261027 \
  --address 10.0.190.200 --response-commit 965029583c76d06e13b26a7289ca163eee2874f2
