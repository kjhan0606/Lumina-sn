#!/usr/bin/env bash
# 실제 두 주파수의 formal 입력만 포획. 원래 물리·인구 보존, 자동 재시도 없음.
#SBATCH --job-name=cmf-formal-capture
#SBATCH --partition=a10,a40,a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:0
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/formal-capture-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/formal-capture-%j.err
set -euo pipefail
umask 027
module purge
module load gnu13
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
ulimit -s unlimited 2>/dev/null || true
readonly RUNNER_DIR=${1:?sealed runner is required}
readonly BUILD=${2:?sealed build is required}
readonly RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_formal_capture_${SLURM_JOB_ID}
[[ ${SLURM_CPUS_PER_TASK:-} == 16 && ${SLURM_MEM_PER_NODE:-} == 16384 && ! -e "$RUN" ]] || exit 23
(cd "$RUNNER_DIR" && sha256sum -c runner.sha256)
/home/kjhan/miniconda3/bin/python3 "$RUNNER_DIR/run_cmfgen_formal_capture.py" \
 --seed /gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702 --run "$RUN" \
 --native /gpfs/kjhan/cmfgen_src/cur_cmf --build "$BUILD" \
 --baseline /gpfs/kjhan/cmfgen_runs/toy06_19p48d_lambda4_428468/seq_logs/lambda-probe-inputs.json
