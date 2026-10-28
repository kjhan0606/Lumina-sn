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
lumina_control_config=''
lumina_control_sha=''
if [[ $# -ge 5 ]]; then
    lumina_control_config=$4
    lumina_control_sha=$5
    shift 5
elif [[ $# == 3 ]]; then
    shift 3
else
    echo '수송 입력3개 또는 전달 구성2개를 함께 지정해야 함' >&2
    exit 2
fi
if [[ $# == 0 ]]; then set -- --max-sweeps 2; fi
cd "$lumina_stage"
lumina_field_status=0
bash code/scripts/run_lumina_resident_field.sh "$PWD" 43200 --feedback \
    --cache /gpfs/kjhan/cmfgen_runs/lumina_phase_prepare_20261026-8fMV0m/run/cache \
    --accepted-input "$lumina_preparation" --accepted-input-sha256 "$lumina_preparation_sha" "$@" || lumina_field_status=$?
if [[ -n "$lumina_control_config" ]]; then
    # 마지막 완성장이 없으면 제출하지 않는다. 확인 실패에도 재시도하지 않는다.
    python3 "$(dirname "$lumina_control_config")/lumina_bundle_dispatch.py" \
        "$lumina_control_config" "$lumina_control_sha" response
fi
exit "$lumina_field_status"
