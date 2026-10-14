#!/bin/bash
#SBATCH --job-name=CMF_STREAM_COST
#SBATCH --partition=a10,a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/stream-cost-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/stream-cost-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cost_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
cost_geo=/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model/geometry.csv
cost_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/stream-cost-${SLURM_JOB_ID}
mkdir "$cost_out"
cd "$cost_src"
sha256sum tests/cmf_streaming_cost.cu src/cmf_cell_geometry.h src/cmf_exact_multigpu.cu \
 src/cmf_exact_multigpu.h src/cmf_exact_sliding.c src/cmf_error_envelope.c \
 src/lumina_frequency_grid.h scripts/run_cmf_streaming_cost.slurm "$cost_geo" \
 > "$cost_out/source.sha256"
nvcc -O2 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
 -gencode arch=compute_90,code=sm_90 -std=c++14 -Xcompiler -fopenmp -Isrc \
 tests/cmf_streaming_cost.cu src/cmf_exact_multigpu.cu src/cmf_exact_sliding.c \
 src/cmf_error_envelope.c -lm -o "$cost_out/cost" > "$cost_out/build.log" 2>&1
"$cost_out/cost" "$cost_geo" > "$cost_out/cost.log" 2> "$cost_out/cost.err"
sha256sum -c "$cost_out/source.sha256" > "$cost_out/verify.log"
