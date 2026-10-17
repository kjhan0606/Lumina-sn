#!/usr/bin/env bash
set -euo pipefail
umask 027
test "$(hostname -s)" = LagEunha
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -v 8388608
aa_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
aa_out=$(mktemp -d /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/acceleration-gpu-build-20261017.XXXXXX)
printf 'GPU_ACCELERATION_BUILD_ROOT=%s\n' "$aa_out"
cd "$aa_out"
for aa_file in cmf_exact_multigpu.cu cmf_exact_multigpu.h cmf_exact_sliding.c cmf_exact_sliding.h cmf_error_envelope.c cmf_error_envelope.h cmf_cell_geometry.h cmf_anderson.h cmf_anderson_multi.h; do
 cp "$aa_src/src/$aa_file" .
done
cp "$aa_src/tests/cmf_acceleration_gpu.cu" "$aa_src/tests/cmf_exact_multigpu_selftest.cu" \
 "$aa_src/tests/build_cmf_acceleration_gpu.sh" .
sha256sum ./*.c ./*.cu ./*.h ./*.sh > source.sha256
date -Is > started.txt
nvcc -O2 -arch=sm_89 -std=c++14 -Xcompiler -fopenmp -DLUMINA_CMF_EXACT_MANAGED_MEMORY=0 -I. \
 -c cmf_exact_multigpu.cu -o gpu.o > build.log 2> build.err
gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -I. -c cmf_exact_sliding.c cmf_error_envelope.c >> build.log 2>> build.err
for aa_test in cmf_acceleration_gpu cmf_exact_multigpu_selftest; do
 nvcc -O2 -arch=sm_89 -std=c++14 -Xcompiler -fopenmp -I. "$aa_test.cu" \
  gpu.o cmf_exact_sliding.o cmf_error_envelope.o -lm -o "$aa_test" >> build.log 2>> build.err
done
sha256sum -c source.sha256 > source_verify.log
sha256sum cmf_acceleration_gpu cmf_exact_multigpu_selftest > binary.sha256
date -Is > finished.txt
printf 'COMPILE_ONLY GPU=0 root=%s\n' "$aa_out"
