#!/bin/bash
#SBATCH --job-name=CMF_GEOMETRY_PHYSICS
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/boundary-geometry-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/boundary-geometry-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
probe_repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
probe_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/boundary-geometry-${SLURM_JOB_ID}
mkdir "$probe_out"
cd "$probe_repo"
sha256sum tests/cmf_boundary_geometry_selftest.c src/cmf_exact_sliding.c \
 src/cmf_exact_sliding.h src/cmf_error_envelope.c scripts/run_cmf_boundary_geometry_selftest.slurm \
 > "$probe_out/source.sha256"
gcc -O2 -std=gnu11 -fopenmp -Isrc tests/cmf_boundary_geometry_selftest.c \
 src/cmf_exact_sliding.c src/cmf_error_envelope.c -lm -o "$probe_out/test"
"$probe_out/test" "${1:-report}" > "$probe_out/test.log"
sha256sum -c "$probe_out/source.sha256" > "$probe_out/verify.log"
echo "result=$probe_out"
