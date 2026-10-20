#!/usr/bin/env bash
# 명시한 완료 물질 → 새 장 → 전체 물질. 고정T 교대1회, 전체 자식40분.
#SBATCH --job-name=cmf-field-feedback
#SBATCH --partition=a10,a40,a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:0
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/field-feedback-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/field-feedback-%j.err
set -euo pipefail
umask 027
module purge
module load gnu13
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ulimit -s unlimited 2>/dev/null || true
readonly FEEDBACK_RUNNER_DIR=${1:?sealed runner is required}
readonly FEEDBACK_BUILD_DIR=${2:?sealed build is required}
readonly FEEDBACK_SOURCE=${3:-/gpfs/kjhan/cmfgen_runs/toy06_19p48d_fixed_field_428866}
readonly FEEDBACK_SOURCE_RECORD=${4:-63}
readonly FEEDBACK_MAX_LAM=${5:-2.0D0}
readonly FEEDBACK_SELECTION=${6:-}
readonly FEEDBACK_RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_${SLURM_JOB_ID}
[[ ${SLURM_CPUS_PER_TASK:-} == 16 && ${SLURM_MEM_PER_NODE:-} == 32768 && ! -e "$FEEDBACK_RUN" ]] || exit 23
(cd "$FEEDBACK_RUNNER_DIR" && sha256sum -c runner.sha256)
FEEDBACK_SELECTION_ARGS=()
if [[ -n "$FEEDBACK_SELECTION" ]]; then FEEDBACK_SELECTION_ARGS=(--selection "$FEEDBACK_SELECTION"); fi
/home/kjhan/miniconda3/bin/python3 "$FEEDBACK_RUNNER_DIR/run_cmfgen_field_feedback.py" \
 --original /gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702 \
 --source "$FEEDBACK_SOURCE" --source-record "$FEEDBACK_SOURCE_RECORD" --max-lam "$FEEDBACK_MAX_LAM" \
 --run "$FEEDBACK_RUN" --build "$FEEDBACK_BUILD_DIR" \
 --native /gpfs/kjhan/cmfgen_src/cur_cmf "${FEEDBACK_SELECTION_ARGS[@]}"
