#!/bin/bash
# 별도 GPU 예산을 정한 뒤에만 제출한다. 실제 관측자 수송 단회다.
#SBATCH --job-name=lumina-resident-observer
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
parent=$(realpath "${1:?새 실행 parent 필요}")
commit=${2:?고정 소스 커밋 필요}
preparation=$(realpath "${3:?수용 물질 준비 기록 필요}")
prep_sha=${4:?물질 준비 기록 SHA 필요}
input_j=$(realpath "${5:?완성 자체장 필요}")
j_sha=${6:?완성 자체장 SHA 필요}
disk_subdivisions=${7:-1}
checkpoint=${8:-}
[[ "$commit" =~ ^[0-9a-f]{40}$ && "$prep_sha" =~ ^[0-9a-f]{64}$ && "$j_sha" =~ ^[0-9a-f]{64}$ ]]
test -d "$parent"
test ! -e "$parent/source"
test ! -e "$parent/staged"
repository=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
mkdir "$parent/source"
git -C "$repository" archive "$commit" scripts src | tar -x -C "$parent/source"
python3 "$parent/source/scripts/stage_lumina_fullsr_sc.py" "$parent/source" "$parent/staged" --observer
base=/gpfs/kjhan/cmfgen_runs/lumina_accepted_feedback_1224468_20261027
test "$(sha256sum "$base/chain_cuda.so" | cut -d ' ' -f 1)" = 531ded6df7c3f72ec2cd17688d6245d5fd291ddfcdfb1705fdd6a435664d4956
test "$(sha256sum "$base/geometry.so" | cut -d ' ' -f 1)" = 3cddc57f07232c92af77d7647f137d6f0f63904024959b34345e487da9bec2d0
cp -- "$base/chain_cuda.so" "$base/geometry.so" "$parent/staged/"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$parent/staged/code/scripts"
args=("$parent/staged" --accepted-input "$preparation" --accepted-input-sha256 "$prep_sha"
      --input-j "$input_j" --input-j-sha256 "$j_sha" --seconds 85800
      --disk-subdivisions "$disk_subdivisions")
if [[ -n "$checkpoint" ]]; then args+=(--resume-checkpoint "$checkpoint"); fi
timeout -k 30 86100s python3 "$parent/staged/code/scripts/run_lumina_resident_observer.py" "${args[@]}"
