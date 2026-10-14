#!/bin/bash
#SBATCH --job-name=CMF_FORMAL_MOMENTS_GPU
#SBATCH --partition=a10,a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/formal-moments-gpu-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/formal-moments-gpu-%j.err
set -euo pipefail
umask 027
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CMF_MGPU_TEST_DEVICES=1
moment_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
moment_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/formal-moments-gpu-${SLURM_JOB_ID}
mkdir "$moment_out"
cd "$moment_src"
moment_files=(src/cmf_exact_multigpu.cu src/cmf_exact_multigpu.h src/cmf_exact_sliding.c
 src/cmf_exact_sliding.h src/cmf_error_envelope.c src/cmf_error_envelope.h src/cmf_cell_geometry.h
 tests/cmf_formal_moments_gpu_selftest.cu tests/cmf_exact_multigpu_selftest.cu
 tests/cmf_exact_sliding_selftest.c
 src/cmf_observer.c src/cmf_observer.cu src/cmf_observer.h src/cmf_observer_ray.h
 tests/cmf_observer_selftest.c
 tests/cmf_emissivity_selftest.cu
 scripts/run_cmf_formal_moments_gpu_selftest.slurm)
mkdir "$moment_out/source"
cp --parents "${moment_files[@]}" "$moment_out/source/"
git rev-parse HEAD > "$moment_out/commit.txt"
sha256sum "${moment_files[@]}" > "$moment_out/source.sha256"
cd "$moment_out/source"
gcc -O2 -std=c11 -Isrc -c src/cmf_exact_sliding.c -o "$moment_out/sliding.o" > "$moment_out/c-build.log" 2>&1
gcc -O2 -std=c11 -Isrc -c src/cmf_error_envelope.c -o "$moment_out/envelope.o" >> "$moment_out/c-build.log" 2>&1
nvcc -O2 -arch=sm_86 -std=c++14 -Isrc -c src/cmf_exact_multigpu.cu \
 -o "$moment_out/multigpu.o" > "$moment_out/cuda-build.log" 2>&1
moment_objects=("$moment_out/multigpu.o" "$moment_out/sliding.o" "$moment_out/envelope.o")
nvcc -O2 -arch=sm_86 -std=c++14 -Isrc tests/cmf_emissivity_selftest.cu \
 "${moment_objects[@]}" -lm -o "$moment_out/emissivity" > "$moment_out/emissivity-build.log" 2>&1
"$moment_out/emissivity" > "$moment_out/emissivity.log" 2> "$moment_out/emissivity.err"
compute-sanitizer --tool memcheck --error-exitcode=79 "$moment_out/emissivity" > "$moment_out/emissivity-memcheck.log" 2>&1
compute-sanitizer --tool racecheck --error-exitcode=79 "$moment_out/emissivity" > "$moment_out/emissivity-racecheck.log" 2>&1
nvcc -O2 -arch=sm_86 -std=c++14 -Isrc tests/cmf_formal_moments_gpu_selftest.cu \
 "${moment_objects[@]}" -lm -o "$moment_out/moments" > "$moment_out/test-build.log" 2>&1
"$moment_out/moments" > "$moment_out/test.log" 2> "$moment_out/test.err"
compute-sanitizer --tool memcheck --error-exitcode=79 "$moment_out/moments" \
 > "$moment_out/memcheck.log" 2>&1
compute-sanitizer --tool racecheck --error-exitcode=79 "$moment_out/moments" \
 > "$moment_out/racecheck.log" 2>&1
nvcc -O2 -arch=sm_86 -std=c++14 -Isrc tests/cmf_exact_multigpu_selftest.cu \
 "${moment_objects[@]}" -lm -o "$moment_out/legacy" > "$moment_out/legacy-build.log" 2>&1
"$moment_out/legacy" > "$moment_out/legacy.log" 2> "$moment_out/legacy.err"
gcc -O2 -std=c11 -Isrc tests/cmf_exact_sliding_selftest.c "$moment_out/sliding.o" \
 "$moment_out/envelope.o" -lm -o "$moment_out/cpu-legacy" > "$moment_out/cpu-legacy-build.log" 2>&1
"$moment_out/cpu-legacy" > "$moment_out/cpu-legacy.log" 2> "$moment_out/cpu-legacy.err"
gcc -O2 -std=c11 -Isrc -c src/cmf_observer.c -o "$moment_out/observer.o" > "$moment_out/observer-build.log" 2>&1
gcc -O2 -std=c11 -DOBSERVER_TEST_GPU -Isrc -c tests/cmf_observer_selftest.c \
 -o "$moment_out/observer-test.o" >> "$moment_out/observer-build.log" 2>&1
nvcc -O2 -arch=sm_86 -std=c++14 -Isrc src/cmf_observer.cu "$moment_out/observer.o" \
 "$moment_out/observer-test.o" -lm -o "$moment_out/observer" >> "$moment_out/observer-build.log" 2>&1
"$moment_out/observer" > "$moment_out/observer.log" 2> "$moment_out/observer.err"
compute-sanitizer --tool memcheck --error-exitcode=79 "$moment_out/observer" \
 > "$moment_out/observer-memcheck.log" 2>&1
sha256sum -c "$moment_out/source.sha256" > "$moment_out/source_verify.log"
