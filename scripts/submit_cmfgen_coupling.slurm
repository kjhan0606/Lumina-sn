#!/usr/bin/env bash
# 등록된 유한 계약의 한 실제 outer. 관찰 부산물은 입력하지 않는다.
#SBATCH --job-name=cmf-coupling
#SBATCH --partition=a10,a40,a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/coupling-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/coupling-%j.err
set -euo pipefail
umask 027
module purge
module load gnu13
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ulimit -s unlimited 2>/dev/null || true
readonly COUPLING_RUNNER=${1:?sealed runner required}
readonly COUPLING_BUILD=${2:?field build required}
readonly COUPLING_CONTRACT=${3:?contract path required}
readonly COUPLING_CONTRACT_SHA=${4:?contract sha256 required}
readonly COUPLING_SOURCE=${5:?completed source required}
readonly COUPLING_SOURCE_RECORD=${6:?source record required}
readonly COUPLING_SELECTION=${7:-}
readonly COUPLING_RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_${SLURM_JOB_ID}
[[ ${SLURM_CPUS_PER_TASK:-} == 16 && ${SLURM_MEM_PER_NODE:-} == 32768 && ! -e "$COUPLING_RUN" ]] || exit 23
(cd "$COUPLING_RUNNER" && sha256sum -c runner.sha256)
COUPLING_SELECTION_ARGS=()
if [[ -n "$COUPLING_SELECTION" ]]; then COUPLING_SELECTION_ARGS=(--selection "$COUPLING_SELECTION"); fi
/home/kjhan/miniconda3/bin/python3 "$COUPLING_RUNNER/run_cmfgen_field_feedback.py" \
 --original /gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702 \
 --source "$COUPLING_SOURCE" --source-record "$COUPLING_SOURCE_RECORD" --max-lam 1.0D6 \
 --run "$COUPLING_RUN" --build "$COUPLING_BUILD" --native /gpfs/kjhan/cmfgen_src/cur_cmf \
 --coupling-contract "$COUPLING_CONTRACT" --coupling-contract-sha256 "$COUPLING_CONTRACT_SHA" \
 "${COUPLING_SELECTION_ARGS[@]}"
