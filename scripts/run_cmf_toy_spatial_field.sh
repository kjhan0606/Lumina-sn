#!/bin/bash
# 작은 직접시험은 Ada, 실제 전체장만 Slurm에서 실행한다.
set -euo pipefail
umask 027
sp_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
sp_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
sp_ref=$sp_base/toy-material-424656
sp_cuda=$sp_base/ada-signed-20261016-0332
sp_out=${CMF_SPATIAL_TEST_OUT:?new absolute output directory required}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
 [[ "$sp_out" == "$sp_base/spatial-field-$SLURM_JOB_ID" ]]
else
 [[ "$(hostname -s)" == LagEunha && "$sp_out" == "$sp_base"/ada-spatial-* ]]
fi
mkdir "$sp_out" "$sp_out/source" "$sp_out/objects" "$sp_out/work"
sp_start=$SECONDS
trap 'sp_rc=$?; printf "RUN_FOOTER exit=%s elapsed_seconds=%s GPUs=1\n" "$sp_rc" "$((SECONDS-sp_start))"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -z "${SLURM_JOB_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES=0; fi
cd "$sp_src"
test "$(git rev-parse --show-toplevel)" == "$sp_src"
git rev-parse HEAD > "$sp_out/commit.txt"
(cd "$sp_ref/source"; sha256sum -c "$sp_ref/source.sha256") > "$sp_out/reference-source-verify.log"
while read -r sp_hash sp_file;do
 [[ "$sp_file" == src/* ]] || continue
 cmp "$sp_file" "$sp_ref/source/$sp_file"
 cp --parents "$sp_file" "$sp_out/source/"
done < "$sp_ref/source.sha256"
cp --parents tests/cmf_toy_spatial_field_diagnostic.c tests/cmf_toy_bf_iterate_diagnostic.c scripts/run_cmf_toy_spatial_field.sh scripts/run_cmf_toy_spatial_field.slurm "$sp_out/source/"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then scontrol write batch_script "$SLURM_JOB_ID" "$sp_out/runner.slurm"; fi
for sp_unit in cmf_exact_multigpu lumina_cmf_solve lumina_bf_gemm cmf_observer;do
 cmp "src/$sp_unit.cu" "$sp_cuda/source/src/$sp_unit.cu"
 cp --parents "src/$sp_unit.cu" "$sp_out/source/"
 if [[ -z "${SLURM_JOB_ID:-}" ]]; then cp "$sp_cuda/$sp_unit-cuda.o" "$sp_out/objects/"; fi
done
for sp_object in "$sp_ref"/objects/*.o;do
 [[ "$sp_object" == */lumina_cmfgen.o ]] && continue
 cp "$sp_object" "$sp_out/objects/"
done
cd "$sp_out/source"
sha256sum src/*.h src/*.c src/*.cu tests/*.c scripts/*.sh scripts/*.slurm > "$sp_out/source.sha256"
(cd "$sp_out/objects"; sha256sum *.o) > "$sp_out/reused-objects.sha256"
gcc -O2 -std=gnu11 -D_GNU_SOURCE -DLUMINA_HAS_CUDA_BF_GEMM -DLUMINA_W5_CONT_SOURCE_SELFTEST -fopenmp \
 -ffunction-sections -fdata-sections -Isrc -c src/lumina_cmfgen.c \
 -o "$sp_out/objects/lumina_cmfgen.o" > "$sp_out/build.log" 2>&1
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
 for sp_unit in cmf_exact_multigpu lumina_cmf_solve lumina_bf_gemm cmf_observer;do
  nvcc -O2 -arch=native -std=c++14 -Isrc -Xcompiler -fopenmp \
   -DLUMINA_CMF_EXACT_MANAGED_MEMORY=0 -c "src/$sp_unit.cu" \
   -o "$sp_out/objects/$sp_unit-cuda.o" >> "$sp_out/build.log" 2>&1
 done
fi
gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections -Isrc \
 -c tests/cmf_toy_spatial_field_diagnostic.c -o "$sp_out/test.o" >> "$sp_out/build.log" 2>&1
nvcc -O2 -arch=native -Xcompiler -fopenmp "$sp_out/test.o" "$sp_out/objects/"*.o \
 -Xlinker --gc-sections -lm -lcublas -o "$sp_out/spatial-field" >> "$sp_out/build.log" 2>&1
while read -r sp_key;do unset "$sp_key";done < <(compgen -v LUMINA_ || true)
cd "$sp_out/work"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
 export LUMINA_CMF_ANDERSON1=1
 "$sp_out/spatial-field" --selftest > "$sp_out/test.log" 2> "$sp_out/test.err"
 grep '^SPATIAL_SELFTEST_PASS ' "$sp_out/test.log"
else
 (cd "$sp_ref/input"; sha256sum -c "$sp_ref/input.sha256") > "$sp_out/input-verify.log"
 sha256sum -c "$sp_ref/iterate.sha256" > "$sp_out/iterate-verify.log"
 while IFS= read -r sp_entry;do export "$sp_entry";done < "$sp_base/toy-initial-424528/env_lumina_effective.txt"
 export LUMINA_BF_CHANNELS_FILE=$sp_ref/input/bf_channels.bin
 export LUMINA_TOPION_LEVELS_FILE=$sp_ref/input/bf_channels.terminal.csv
 export LUMINA_IONIZATION_REFERENCE_FILE=$sp_ref/input/ionization_reference.csv
 unset CMF_TEST_BF_LTE_LEVELS
 env | LC_ALL=C sort | grep '^LUMINA_' > "$sp_out/env_lumina_effective.txt"
 sp_id=$(sha256sum "$sp_ref/iterate_identity.txt");sp_id=${sp_id%% *}
 "$sp_out/spatial-field" "$sp_ref/input/snia_toy06_2d.dat" "$sp_ref/input" \
  "$sp_ref/iterate_input.bin" "$sp_id" > "$sp_out/test.log" 2> "$sp_out/test.err"
 sha256sum stored_bf.csv baseline_bf.csv refined_bf.csv field_change.csv > "$sp_out/output.sha256"
 sha256sum -c "$sp_ref/iterate.sha256" > "$sp_out/iterate-final-verify.log"
 grep -Fx 'SPATIAL_FIELD_DONE coefficients_byte_identical=1 material_byte_identical=1 full_SE_solved=0 physical_history=0 final_goal=NOT_JUDGED' "$sp_out/test.log"
fi
(cd "$sp_out/source"; sha256sum -c "$sp_out/source.sha256") > "$sp_out/source-verify.log"
sha256sum "$sp_out/spatial-field" "$sp_out/test.log" "$sp_out/test.err" > "$sp_out/executable-output.sha256"
