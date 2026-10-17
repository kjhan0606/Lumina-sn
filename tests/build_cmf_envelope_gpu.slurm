#!/bin/bash
#SBATCH --job-name=CMF_BOUND_TEST_BUILD
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/envelope-gpu-build-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/envelope-gpu-build-%j.err
set -euo pipefail
umask 027
# 컴파일만 한다. GPU 시험은 Ada의 기존 계산 종료 후 별도로 실행한다.
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
eg_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
eg_root=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/envelope-gpu-build-${SLURM_JOB_ID:?}
mkdir "$eg_root"
cd "$eg_src"
eg_files=(src/cmf_exact_multigpu.cu src/cmf_exact_multigpu.h
 src/cmf_exact_sliding.c src/cmf_exact_sliding.h src/cmf_error_envelope.c
 src/cmf_error_envelope.h src/cmf_cell_geometry.h src/cmf_anderson.h
 tests/cmf_exact_multigpu_selftest.cu)
sha256sum "${eg_files[@]}" > "$eg_root/source.sha256"
git rev-parse HEAD > "$eg_root/commit.txt"
git diff --binary -- tests/cmf_exact_multigpu_selftest.cu > "$eg_root/test.patch"
cp tests/build_cmf_envelope_gpu.slurm "$eg_root/runner.slurm"
mkdir "$eg_root/source"
cp --parents "${eg_files[@]}" "$eg_root/source/"
cd "$eg_root/source"
sha256sum -c "$eg_root/source.sha256" > "$eg_root/source_verify.log"
nvcc -O2 -arch=sm_89 -std=c++14 -Xcompiler -fopenmp \
 -DLUMINA_CMF_EXACT_MANAGED_MEMORY=1 -Isrc \
 tests/cmf_exact_multigpu_selftest.cu src/cmf_exact_multigpu.cu \
 src/cmf_exact_sliding.c src/cmf_error_envelope.c -lm \
 -o "$eg_root/selftest" > "$eg_root/build.log" 2> "$eg_root/build.err"
sha256sum -c "$eg_root/source.sha256" > "$eg_root/source_after_verify.log"
sha256sum "$eg_root/selftest" > "$eg_root/binary.sha256"
printf 'build=%s GPU_execution=0 managed=1 sm=89\n' "$eg_root"
