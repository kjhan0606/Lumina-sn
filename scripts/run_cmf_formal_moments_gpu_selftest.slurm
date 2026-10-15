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
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CMF_MGPU_TEST_DEVICES=1
while read -r moment_key; do unset "$moment_key"; done < <(compgen -v LUMINA_ || true)
moment_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
 moment_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/formal-moments-gpu-${SLURM_JOB_ID}
else
 # 사용자 지정 짧은 GPU 시험 서버에서만 수동 실행한다. 기존 출력은 거부한다.
 [[ "$(hostname -s)" == LagEunha ]]
 moment_out=${CMF_GPU_TEST_OUT:?new absolute output directory required}
 [[ "$moment_out" == /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/ada-* ]]
fi
mkdir "$moment_out"
if [[ -n "${CMF_TEST_COMMON_HISTORY:-}" ]]; then
 # 이미 통과한 실제 선 조립 시험을 동일 C 객체/소스와 CUDA dispatcher로 실행한다.
 # 형식해 회귀 전체를 중복 수행하지 않는 짧은 지정 시험이다.
 moment_reference=$CMF_TEST_COMMON_HISTORY
 (cd "$moment_reference/source"; sha256sum -c "$moment_reference/source.sha256") > "$moment_out/reference_verify.log"
 test -f "$moment_reference/source/tests/cmf_common_signed_material_selftest.c"
 mkdir "$moment_out/source" "$moment_out/objects"
 cp "$0" "$moment_out/runner.slurm"
 cp "$moment_reference/source.sha256" "$moment_out/reference_source.sha256"
 cd "$moment_reference/source"
 cp --parents src/*.h src/lumina_cmfgen.c tests/cmf_common_signed_material_selftest.c "$moment_out/source/"
 cd "$moment_src"
 moment_cuda_units=(cmf_exact_multigpu lumina_cmf_solve lumina_bf_gemm cmf_observer)
 for moment_unit in "${moment_cuda_units[@]}"; do
  cp --parents "src/$moment_unit.cu" "$moment_out/source/"
 done
 git rev-parse HEAD > "$moment_out/commit.txt"
 cd "$moment_reference/objects"
 for moment_object in *.o; do
  [[ "$moment_object" == lumina_cmfgen.o ]] && continue
  cp "$moment_object" "$moment_out/objects/"
 done
 cd "$moment_out/source"
 sha256sum src/*.h src/*.c src/*.cu tests/*.c > "$moment_out/source.sha256"
 (cd "$moment_out/objects"; sha256sum *.o) > "$moment_out/reference_objects.sha256"
 gcc -O2 -std=gnu11 -D_GNU_SOURCE -DLUMINA_HAS_CUDA_BF_GEMM -DLUMINA_W5_CONT_SOURCE_SELFTEST \
  -fopenmp -ffunction-sections -fdata-sections -Isrc -c src/lumina_cmfgen.c \
  -o "$moment_out/objects/lumina_cmfgen.o" > "$moment_out/build.log" 2>&1
 gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -Isrc -c tests/cmf_common_signed_material_selftest.c \
  -o "$moment_out/test.o" >> "$moment_out/build.log" 2>&1
 for moment_unit in "${moment_cuda_units[@]}"; do
  nvcc -O2 -arch=native -std=c++14 -Isrc -Xcompiler -fopenmp -c "src/$moment_unit.cu" \
   -o "$moment_out/$moment_unit-cuda.o" >> "$moment_out/build.log" 2>&1
 done
 nvcc -O2 -arch=native -Xcompiler -fopenmp "$moment_out/test.o" "$moment_out/objects/"*.o \
  "$moment_out/"*-cuda.o -Xlinker --gc-sections -lm -lcublas -o "$moment_out/common-signed" \
  >> "$moment_out/build.log" 2>&1
 export LUMINA_CMF_FINE_MGPU_DEVICES=1
 "$moment_out/common-signed" > "$moment_out/test.log" 2> "$moment_out/test.err"
 grep -Fx 'CMF_COMMON_SIGNED_MATERIAL_SELFTEST_PASS' "$moment_out/test.log"
 test "$(grep -c 'EXACT-MULTIGPU-EPOCH.*status=OK' "$moment_out/test.err")" -eq 12
 sha256sum -c "$moment_out/source.sha256" > "$moment_out/source_verify.log"
 sha256sum "$moment_out/common-signed" "$moment_out/test.log" "$moment_out/test.err" > "$moment_out/output.sha256"
 echo 'COMMON_SIGNED_GPU_PASS cases=12 devices=1 physical_SN=0'
 exit 0
fi
if [[ -n "${CMF_TEST_MATERIAL_CSV:-}" ]]; then
 cp -- "$CMF_TEST_MATERIAL_CSV" "$moment_out/geometry_input.csv"
 export CMF_TEST_MATERIAL_CSV=$moment_out/geometry_input.csv
 sha256sum "$CMF_TEST_MATERIAL_CSV" > "$moment_out/geometry_input.sha256"
fi
cd "$moment_src"
moment_files=(src/cmf_anderson.h src/cmf_exact_multigpu.cu src/cmf_exact_multigpu.h src/cmf_exact_sliding.c
 src/cmf_time_moment.c src/cmf_time_moment.h src/cmf_time_field.c src/cmf_time_field.h
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
nvcc -O2 -arch=native -std=c++14 -Isrc -c src/cmf_exact_multigpu.cu \
 -o "$moment_out/multigpu.o" > "$moment_out/cuda-build.log" 2>&1
moment_objects=("$moment_out/multigpu.o" "$moment_out/sliding.o" "$moment_out/envelope.o")
gcc -O2 -std=c11 -Isrc -c src/cmf_time_moment.c -o "$moment_out/time-moment.o" >> "$moment_out/c-build.log" 2>&1
gcc -O2 -std=c11 -Isrc -c src/cmf_time_field.c -o "$moment_out/time-field.o" >> "$moment_out/c-build.log" 2>&1
moment_objects+=("$moment_out/time-moment.o" "$moment_out/time-field.o")
nvcc -O2 -arch=native -std=c++14 -Isrc tests/cmf_emissivity_selftest.cu \
 "${moment_objects[@]}" -lm -o "$moment_out/emissivity" > "$moment_out/emissivity-build.log" 2>&1
"$moment_out/emissivity" > "$moment_out/emissivity.log" 2> "$moment_out/emissivity.err"
compute-sanitizer --tool memcheck --error-exitcode=79 "$moment_out/emissivity" > "$moment_out/emissivity-memcheck.log" 2>&1
compute-sanitizer --tool racecheck --error-exitcode=79 "$moment_out/emissivity" > "$moment_out/emissivity-racecheck.log" 2>&1
nvcc -O2 -arch=native -std=c++14 -Isrc tests/cmf_formal_moments_gpu_selftest.cu \
 "${moment_objects[@]}" -lm -o "$moment_out/moments" > "$moment_out/test-build.log" 2>&1
"$moment_out/moments" > "$moment_out/test.log" 2> "$moment_out/test.err"
compute-sanitizer --tool memcheck --error-exitcode=79 "$moment_out/moments" \
 > "$moment_out/memcheck.log" 2>&1
compute-sanitizer --tool racecheck --error-exitcode=79 "$moment_out/moments" \
 > "$moment_out/racecheck.log" 2>&1
nvcc -O2 -arch=native -std=c++14 -Isrc tests/cmf_exact_multigpu_selftest.cu \
 "${moment_objects[@]}" -lm -o "$moment_out/legacy" > "$moment_out/legacy-build.log" 2>&1
"$moment_out/legacy" > "$moment_out/legacy.log" 2> "$moment_out/legacy.err"
gcc -O2 -std=c11 -Isrc tests/cmf_exact_sliding_selftest.c "$moment_out/sliding.o" \
 "$moment_out/envelope.o" -lm -o "$moment_out/cpu-legacy" > "$moment_out/cpu-legacy-build.log" 2>&1
"$moment_out/cpu-legacy" > "$moment_out/cpu-legacy.log" 2> "$moment_out/cpu-legacy.err"
gcc -O2 -std=c11 -Isrc -c src/cmf_observer.c -o "$moment_out/observer.o" > "$moment_out/observer-build.log" 2>&1
gcc -O2 -std=c11 -DOBSERVER_TEST_GPU -Isrc -c tests/cmf_observer_selftest.c \
 -o "$moment_out/observer-test.o" >> "$moment_out/observer-build.log" 2>&1
nvcc -O2 -arch=native -std=c++14 -Isrc src/cmf_observer.cu "$moment_out/observer.o" \
 "$moment_out/observer-test.o" -lm -o "$moment_out/observer" >> "$moment_out/observer-build.log" 2>&1
"$moment_out/observer" > "$moment_out/observer.log" 2> "$moment_out/observer.err"
compute-sanitizer --tool memcheck --error-exitcode=79 "$moment_out/observer" \
 > "$moment_out/observer-memcheck.log" 2>&1
sha256sum -c "$moment_out/source.sha256" > "$moment_out/source_verify.log"
if [[ -n "${CMF_TEST_MATERIAL_CSV:-}" ]]; then
 sha256sum -c "$moment_out/geometry_input.sha256" > "$moment_out/geometry_input_verify.log"
fi
