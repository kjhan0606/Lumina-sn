#!/usr/bin/env bash
# 기존 물리/첫60회 phase 유지, 실패 MOM 호출 포획 후 종료. 자동 재시도 없음.
#SBATCH --job-name=cmf-mom-capture
#SBATCH --partition=a10,a40,a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:0
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/mom-capture-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/mom-capture-%j.err
set -euo pipefail
umask 027
module purge
module load gnu13
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
ulimit -s unlimited 2>/dev/null || true
readonly RUNNER_DIR=${1:?sealed runner directory is required}
readonly BUILD=${2:?sealed capture build is required}
readonly MODE=${3:-selected}
[[ $MODE == selected || $MODE == first ]] || exit 22
readonly NATIVE=/gpfs/kjhan/cmfgen_src/cur_cmf
readonly SEED=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702
if [[ $MODE == first ]]; then
  readonly RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_mom_first_${SLURM_JOB_ID}
  [[ ${SLURM_MEM_PER_NODE:-} == 16384 ]] || exit 23
  EXTRA_ARGS=(--first-repair)
else
  readonly RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_mom_capture_${SLURM_JOB_ID}
  EXTRA_ARGS=()
fi
readonly BASELINE=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_lambda4_428468/seq_logs/lambda-probe-inputs.json
readonly PYTHON=/home/kjhan/miniconda3/bin/python3
[[ ${SLURM_CPUS_PER_TASK:-} == 16 && ! -e "$RUN" ]] || exit 20
(cd "$RUNNER_DIR" && sha256sum -c runner.sha256)
"$PYTHON" "$RUNNER_DIR/run_cmfgen_mom_capture.py" --seed "$SEED" --run "$RUN" \
  --native "$NATIVE" --build "$BUILD" --baseline "$BASELINE" "${EXTRA_ARGS[@]}"
