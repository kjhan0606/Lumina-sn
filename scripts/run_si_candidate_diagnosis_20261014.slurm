#!/bin/bash
#SBATCH --job-name=LUMINA_SI_CAUSE_CPU
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/candidate-check-%j.json
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/candidate-check-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
python3 -B scripts/diagnose_si_candidates_20261014.py \
    --run /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1
