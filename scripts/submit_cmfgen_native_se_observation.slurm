#!/usr/bin/env bash
# 원래67의 관찰 재현과 완료67/J67 원시 SE 평가. 추가 RT/outer는 없다.
#SBATCH --job-name=cmf-native-se-observe
#SBATCH --partition=a10,a40,a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/cmfgen_runs/slurm_logs/native-se-observe-%j.out
#SBATCH --error=/gpfs/kjhan/cmfgen_runs/slurm_logs/native-se-observe-%j.err
set -euo pipefail
umask 027
module purge
module load gnu13
export OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_STACKSIZE=512M
export OMP_PROC_BIND=close OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ulimit -s unlimited 2>/dev/null || true
readonly OBS_RUNNER=${1:?sealed runner required}
readonly OBS_BUILD=${2:?sealed build required}
readonly OBS_RUN=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_native_se_observation_${SLURM_JOB_ID}
[[ ${SLURM_CPUS_PER_TASK:-} == 16 && ${SLURM_MEM_PER_NODE:-} == 32768 && ! -e "$OBS_RUN" ]] || exit 23
(cd "$OBS_RUNNER" && sha256sum -c runner.sha256)
/home/kjhan/miniconda3/bin/python3 "$OBS_RUNNER/run_cmfgen_native_se_observation.py" \
 --original /gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702 \
 --source66 /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_428954/material \
 --source67 /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_428960/material \
 --run "$OBS_RUN" --native /gpfs/kjhan/cmfgen_src/cur_cmf --build "$OBS_BUILD"
