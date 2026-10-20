#!/usr/bin/env bash
# StageA59 clone, fixed T, LAMBDA from the first step, max four iterations.
#SBATCH --job-name=cmf-lambda4
#SBATCH --partition=a10,a40,a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/lambda4-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/lambda4-%j.err
set -euo pipefail
umask 027
module purge
module load gnu13
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M OMP_PROC_BIND=close OMP_PLACES=cores
ulimit -s unlimited 2>/dev/null || true
readonly REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
readonly SEED=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702
readonly SOURCE=/gpfs/kjhan/cmfgen_src/cur_cmf
readonly RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_lambda4_${SLURM_JOB_ID}
readonly PYTHON=/home/kjhan/miniconda3/bin/python3
readonly RUNNER_DIR=${1:?sealed runner directory is required}
[[ ${SLURM_CPUS_PER_TASK:-} == 16 && ! -e "$RUN" ]] || exit 20
[[ -f "$RUNNER_DIR/run_cmfgen_lambda_probe.py" && -f "$RUNNER_DIR/runner.sha256" ]] || exit 21
(cd "$RUNNER_DIR" && sha256sum -c runner.sha256)
"$PYTHON" "$RUNNER_DIR/run_cmfgen_lambda_probe.py" --seed "$SEED" --run "$RUN" \
  --executable "$SOURCE/exe/cmfgen_dev.exe" --source "$SOURCE"
