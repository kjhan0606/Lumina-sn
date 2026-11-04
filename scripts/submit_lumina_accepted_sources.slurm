#!/bin/bash -l
# 새 물질의 전주파수 계수·경계1회. 실제 수송/물질응답은 묶음 내 후속 작업이다.
#SBATCH --job-name=lumina-accepted-sources
#SBATCH --partition=normal
#SBATCH --exclude=grammar072,grammar078,grammar080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
set -euo pipefail
umask 027
ulimit -c 0
module purge
module load gnu13/13.2.0
repository=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
parent=$(realpath "${1:?새 parent 필요}")
source_commit=${2:?봉인 commit 필요}
shift 2
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]]
lumina_bundle_config=${LUMINA_ACCEPTED_BUNDLE_CONFIG:-}
lumina_bundle_digest=${LUMINA_ACCEPTED_BUNDLE_CONFIG_SHA256:-}
if [[ -n "$lumina_bundle_config" || -n "$lumina_bundle_digest" ]]; then
    [[ -n "$lumina_bundle_config" && "$lumina_bundle_digest" =~ ^[0-9a-f]{64}$ ]]
    test -f "$lumina_bundle_config"
    test -f "$(dirname "$lumina_bundle_config")/lumina_bundle_dispatch.py"
    read -r lumina_actual_digest _ < <(sha256sum -- "$lumina_bundle_config")
    [[ "$lumina_actual_digest" = "$lumina_bundle_digest" ]]
    python3 "$(dirname "$lumina_bundle_config")/lumina_bundle_dispatch.py" \
        "$lumina_bundle_config" "$lumina_bundle_digest" preflight-source
fi
test -d "$parent"
test ! -e "$parent/source"
test ! -e "$parent/run"
mkdir "$parent/source"
git -C "$repository" archive "$source_commit" src scripts data | tar -x -C "$parent/source"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$parent/source/scripts"
cd "$parent/source"
timeout -k 30 6900s python3 scripts/run_lumina_accepted_sources.py \
    --out "$parent/run" --source-commit "$source_commit" "$@"
if [[ -n "$lumina_bundle_config" ]]; then
    # 승인된 묶음 구성에서만 새 계수 완성 후 GPU를 한 번 제출한다. 대기/재시도 없음.
    python3 "$(dirname "$lumina_bundle_config")/lumina_bundle_dispatch.py" \
        "$lumina_bundle_config" "$lumina_bundle_digest" field
fi
