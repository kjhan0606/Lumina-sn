#!/bin/bash
#SBATCH --job-name=lumina-f0-sr
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=none
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --exclude=grammar072,grammar078,grammar080

# cfg/hash/새 출력/봉인 소스 루트는 제출자가 명시한다. 노드는 고정하지 않는다.
set -eo pipefail
source /etc/profile
set -u
module purge
module load gnu13
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONDONTWRITEBYTECODE=1
if [[ $# != 4 ]]; then
    echo '사용법: job CONFIG CONFIG_SHA256 NEW_OUTPUT SNAPSHOT_ROOT' >&2
    exit 2
fi
cd "$4"
exec python3 scripts/run_lumina_bridge_field0.py --config "$1" --config-sha256 "$2" --out "$3"
