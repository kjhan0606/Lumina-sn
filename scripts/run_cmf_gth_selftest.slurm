#!/bin/bash
#SBATCH --job-name=CMF_GTH_PARALLEL
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/gth-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/gth-%j.err
set -euo pipefail
umask 027
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=2 OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
gth_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
gth_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/gth-${SLURM_JOB_ID}
gth_reference=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-build-423710
mkdir "$gth_out" "$gth_out/source"
cd "$gth_src"
git rev-parse HEAD > "$gth_out/commit.txt"
gth_files=(src/population_contract.c src/population_contract.h
 tests/population_gth_parallel_selftest.c tests/a2_07_population_selftest.c
 scripts/run_cmf_gth_selftest.slurm)
cp --parents "${gth_files[@]}" "$gth_out/source/"
sha256sum "${gth_files[@]}" > "$gth_out/source.sha256"
(cd "$gth_reference/source"; sha256sum -c "$gth_reference/source.sha256") > "$gth_out/reference_verify.log"
cp "$gth_reference/source/src/population_contract.c" "$gth_out/reference.c"
cp "$gth_reference/commit.txt" "$gth_out/reference_commit.txt"
sha256sum "$gth_out/reference.c" > "$gth_out/reference.sha256"
cd "$gth_out/source"
cmp src/population_contract.h "$gth_reference/source/src/population_contract.h"
for gth_variant in reference current; do
 gth_unit=src/population_contract.c
 if [[ "$gth_variant" == reference ]]; then gth_unit=$gth_out/reference.c; fi
 gcc -O2 -std=gnu11 -fopenmp -Isrc tests/population_gth_parallel_selftest.c "$gth_unit" \
  -lm -o "$gth_out/$gth_variant" > "$gth_out/$gth_variant-build.log" 2>&1
 "$gth_out/$gth_variant" "$gth_out/$gth_variant.bin" > "$gth_out/$gth_variant.log" 2>&1
done
cmp "$gth_out/reference.bin" "$gth_out/current.bin"
# 생산 빌드와 같은 nvcc의 C 번역 경로도 직접 실행해 확인한다(GPU 사용 없음).
nvcc -O2 -std=c++14 -Xcompiler -fopenmp -Isrc tests/population_gth_parallel_selftest.c \
 src/population_contract.c -lm -o "$gth_out/native" > "$gth_out/native-build.log" 2>&1
"$gth_out/native" "$gth_out/native.bin" > "$gth_out/native.log" 2>&1
cmp "$gth_out/reference.bin" "$gth_out/native.bin"
gcc -O2 -std=gnu11 -fopenmp -Isrc tests/a2_07_population_selftest.c src/population_contract.c \
 -lm -o "$gth_out/regression" > "$gth_out/regression-build.log" 2>&1
"$gth_out/regression" > "$gth_out/regression.log" 2>&1
gcc -O1 -g -std=gnu11 -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
 tests/population_gth_parallel_selftest.c src/population_contract.c -lm -o "$gth_out/sanitized" \
 > "$gth_out/sanitized-build.log" 2>&1
ASAN_OPTIONS=detect_leaks=1 "$gth_out/sanitized" "$gth_out/sanitized.bin" > "$gth_out/sanitized.log" 2>&1
cmp "$gth_out/reference.bin" "$gth_out/sanitized.bin"
sha256sum "$gth_out/reference.bin" "$gth_out/current.bin" "$gth_out/native.bin" "$gth_out/sanitized.bin" > "$gth_out/output.sha256"
sha256sum -c "$gth_out/source.sha256" > "$gth_out/source_verify.log"
echo 'GTH_SELFTEST_COMPLETE: 순수 정상행렬 풀이 대조. 실제 SN 물리해 아님.'
