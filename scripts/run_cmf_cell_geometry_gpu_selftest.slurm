#!/bin/bash
#SBATCH --job-name=CMF_CELL_GPU_TEST
#SBATCH --partition=a10,a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/cell-geometry-gpu-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/cell-geometry-gpu-%j.err
set -euo pipefail
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CMF_MGPU_TEST_DEVICES=1
probe_repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-common-20261015
probe_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/cell-geometry-gpu-${SLURM_JOB_ID}
mkdir "$probe_out"
cd "$probe_repo"
sha256sum tests/cmf_boundary_geometry_selftest.c tests/cmf_exact_multigpu_selftest.cu \
 tests/cmf_parallel_prefix_selftest.cu \
 src/cmf_exact_sliding.c src/cmf_exact_sliding.h src/cmf_error_envelope.c \
 src/cmf_error_envelope.h src/cmf_exact_multigpu.cu src/cmf_exact_multigpu.h \
 src/cmf_cell_geometry.h scripts/run_cmf_cell_geometry_gpu_selftest.slurm > "$probe_out/source.sha256"
nvcc -O2 -arch=sm_86 -std=c++14 -DCMF_TEST_CUDA -Xcompiler -fopenmp \
 -Isrc tests/cmf_boundary_geometry_selftest.c src/cmf_exact_sliding.c \
 src/cmf_error_envelope.c src/cmf_exact_multigpu.cu -lm -o "$probe_out/boundary" \
 > "$probe_out/boundary-build.log" 2>&1
"$probe_out/boundary" require-physical > "$probe_out/boundary.log" 2> "$probe_out/boundary.err"
nvcc -O2 -arch=sm_86 -std=c++14 -Xcompiler -fopenmp \
 -Isrc tests/cmf_exact_multigpu_selftest.cu src/cmf_exact_sliding.c \
 src/cmf_error_envelope.c src/cmf_exact_multigpu.cu -lm -o "$probe_out/regression" \
 > "$probe_out/regression-build.log" 2>&1
"$probe_out/regression" > "$probe_out/regression.log" 2> "$probe_out/regression.err"
compute-sanitizer --tool memcheck --error-exitcode=79 "$probe_out/regression" \
 > "$probe_out/memcheck.log" 2> "$probe_out/memcheck.err"
nvcc -O2 -arch=sm_86 -std=c++14 -Xcompiler -fopenmp -Isrc \
 tests/cmf_parallel_prefix_selftest.cu src/cmf_exact_sliding.c \
 src/cmf_error_envelope.c src/cmf_exact_multigpu.cu -lm -o "$probe_out/prefix" \
 > "$probe_out/prefix-build.log" 2>&1
"$probe_out/prefix" > "$probe_out/prefix.log" 2> "$probe_out/prefix.err"
compute-sanitizer --tool memcheck --error-exitcode=79 "$probe_out/prefix" \
 > "$probe_out/prefix-memcheck.log" 2>&1
compute-sanitizer --tool racecheck --error-exitcode=79 "$probe_out/prefix" \
 > "$probe_out/prefix-racecheck.log" 2>&1
sha256sum -c "$probe_out/source.sha256" > "$probe_out/verify.log"
echo "result=$probe_out"
