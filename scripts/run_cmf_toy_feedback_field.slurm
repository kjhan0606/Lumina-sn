#!/bin/bash
#SBATCH --job-name=CMF_SIII_FIELD_FEEDBACK
#SBATCH --partition=a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/feedback-field-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/feedback-field-%j.err
set -euo pipefail
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
export CMF_FEEDBACK_TEST_OUT=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/feedback-field-${SLURM_JOB_ID}
exec bash scripts/run_cmf_toy_feedback_field.sh
