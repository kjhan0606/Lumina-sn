#!/bin/bash
#SBATCH --job-name=CMF_ANDERSON_TEST
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --exclude=syn102,syn03
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/anderson-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/anderson-%j.err
set -euo pipefail
umask 027
aa_mode=${1:-cpu}
case "$aa_mode" in cpu|gpu) ;; *) exit 2 ;; esac
aa_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
aa_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/anderson-${SLURM_JOB_ID}
mkdir "$aa_out" "$aa_out/source"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset LUMINA_CMF_ANDERSON1
cd "$aa_src"
git rev-parse HEAD > "$aa_out/commit.txt"
aa_files=(src/cmf_anderson.h src/cmf_cell_geometry.h src/cmf_exact_sliding.h
 src/cmf_exact_sliding.c src/cmf_error_envelope.h src/cmf_error_envelope.c
 src/cmf_exact_multigpu.h src/cmf_exact_multigpu.cu
 tests/cmf_anderson_selftest.c tests/cmf_exact_sliding_selftest.c
 tests/cmf_exact_multigpu_selftest.cu scripts/run_cmf_anderson_selftest.slurm)
cp --parents "${aa_files[@]}" "$aa_out/source/"
cd "$aa_out/source"
sha256sum "${aa_files[@]}" > "$aa_out/source.sha256"
aa_units=(src/cmf_exact_sliding.c src/cmf_error_envelope.c)
gcc -O2 -std=gnu11 -fopenmp -Isrc tests/cmf_anderson_selftest.c "${aa_units[@]}" -lm -o "$aa_out/test"
gcc -O2 -std=gnu11 -fopenmp -Isrc tests/cmf_exact_sliding_selftest.c "${aa_units[@]}" -lm -o "$aa_out/exact-test"
"$aa_out/test" > "$aa_out/test.log" 2> "$aa_out/test.err"
"$aa_out/exact-test" > "$aa_out/exact-plain.log" 2> "$aa_out/exact-plain.err"
LUMINA_CMF_ANDERSON1=1 "$aa_out/exact-test" > "$aa_out/exact-fast.log" 2> "$aa_out/exact-fast.err"
gcc -O1 -g -std=gnu11 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
 tests/cmf_anderson_selftest.c "${aa_units[@]}" -lm -o "$aa_out/sanitized"
ASAN_OPTIONS=detect_leaks=1 "$aa_out/sanitized" > "$aa_out/sanitized.log" 2> "$aa_out/sanitized.err"
if [[ "$aa_mode" == gpu ]]; then
 module load cuda/13.0.2
 nvcc -O2 -arch=native -std=c++14 -Xcompiler -fopenmp -Isrc -DCMF_ANDERSON_GPU_TEST \
  tests/cmf_anderson_selftest.c "${aa_units[@]}" src/cmf_exact_multigpu.cu -lm -o "$aa_out/gpu-test"
 nvcc -O2 -arch=native -std=c++14 -Xcompiler -fopenmp -Isrc \
  tests/cmf_exact_multigpu_selftest.cu "${aa_units[@]}" src/cmf_exact_multigpu.cu -lm -o "$aa_out/gpu-exact-test"
 "$aa_out/gpu-test" > "$aa_out/gpu.log" 2> "$aa_out/gpu.err"
 CMF_MGPU_TEST_DEVICES=1 "$aa_out/gpu-exact-test" > "$aa_out/gpu-exact-plain.log" 2> "$aa_out/gpu-exact-plain.err"
 CMF_MGPU_TEST_DEVICES=1 LUMINA_CMF_ANDERSON1=1 "$aa_out/gpu-exact-test" > "$aa_out/gpu-exact-fast.log" 2> "$aa_out/gpu-exact-fast.err"
fi
sha256sum -c "$aa_out/source.sha256" > "$aa_out/source_verify.log"
printf 'CMF_ANDERSON_ALL_TESTS_PASS mode=%s\n' "$aa_mode"
