#!/bin/bash
# 실제113셸 장 비교용. 짧은 GPU 단위시험의 사용자 지정 장소는 Ada 그대로다.
#SBATCH --job-name=CMF_GF_FROZEN_FIELD
#SBATCH --partition=a100,a100_pcie,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/gf-field-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/gf-field-%j.err
set -euo pipefail
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
export CMF_GF_TEST_OUT=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/gf-field-${SLURM_JOB_ID}
exec bash scripts/run_cmf_toy_gf_field.sh
