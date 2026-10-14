#!/bin/bash
#SBATCH --job-name=CMF_REFERENCE_INPUTS
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:0
#SBATCH --time=00:02:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/reference-inputs-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/reference-inputs-%j.err
set -euo pipefail
umask 027
reference_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
reference_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/reference-inputs-${SLURM_JOB_ID}
mkdir "$reference_out"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$reference_src"
git rev-parse HEAD > "$reference_out/commit.txt"
sha256sum scripts/diagnose_cmf_reference_inputs.py scripts/run_cmf_reference_inputs.slurm \
 > "$reference_out/source.sha256"
/home/kjhan/miniconda3/bin/python3 scripts/diagnose_cmf_reference_inputs.py > "$reference_out/summary.json"
sha256sum -c "$reference_out/source.sha256" > "$reference_out/source_verify.log"
