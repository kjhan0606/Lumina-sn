#!/bin/bash
set -euo pipefail
umask 027
tr_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
tr_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
tr_ref=$tr_base/toy-material-424656
tr_build=$tr_base/common-build-424525
tr_input=$tr_base/toy-initial-424528/input
tr_out=${CMF_TEMPERATURE_TEST_OUT:?new output directory required}
tr_mode=${1:-selftest}
if [[ -n "${SLURM_JOB_ID:-}" ]];then
 [[ "$tr_out" == "$tr_base/temperature-response-$SLURM_JOB_ID" ]]
 [[ "$tr_mode" == constant || "$tr_mode" == reconstructed ]]
else
 [[ "$(hostname -s)" == LagEunha && "$tr_out" == "$tr_base"/ada-temperature-* && "$tr_mode" == selftest ]]
fi
mkdir "$tr_out" "$tr_out/source" "$tr_out/source/tests" "$tr_out/source/scripts" "$tr_out/objects" "$tr_out/work"
tr_start=$SECONDS
trap 'tr_rc=$?; printf "RUN_FOOTER exit=%s elapsed_seconds=%s GPUs=1\n" "$tr_rc" "$((SECONDS-tr_start))"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -z "${SLURM_JOB_ID:-}" ]];then export CUDA_VISIBLE_DEVICES=0;fi
cd "$tr_src"
test "$(git rev-parse --show-toplevel)" == "$tr_src"
git rev-parse HEAD > "$tr_out/commit.txt"
(cd "$tr_ref/source";sha256sum -c "$tr_ref/source.sha256") > "$tr_out/reference-source-verify.log"
while read -r tr_hash tr_file;do
 [[ "$tr_file" == src/* ]] || continue
 cmp "$tr_file" "$tr_ref/source/$tr_file"
 cp --parents "$tr_file" "$tr_out/source/"
done < "$tr_ref/source.sha256"
tr_extra=(cmf_nonthermal cmf_nonthermal_atomic cmf_nonthermal_provider cmf_nonthermal_excitation cmf_nonthermal_binding cmf_gamma_source cmf_gamma_transport)
for tr_unit in "${tr_extra[@]}" cmf_toy_run;do
 cmp "src/$tr_unit.c" "$tr_build/source/src/$tr_unit.c"
 cp --parents "src/$tr_unit.c" "$tr_out/source/"
done
cp --parents tests/cmf_toy_temperature_response.c scripts/run_cmf_toy_temperature_response.sh scripts/run_cmf_toy_temperature_response.slurm "$tr_out/source/"
cp "$tr_base/temperature-map-424675/source/mon_interp.f" "$tr_base/temperature-map-424675/source/set_kind_module.f" "$tr_base/temperature-map-424675/source/error_unit.f" "$tr_out/source/"
cp "$tr_ref/iterate_identity.txt" "$tr_out/donor_identity.txt"
if [[ -n "${SLURM_JOB_ID:-}" ]];then scontrol write batch_script "$SLURM_JOB_ID" "$tr_out/runner.slurm";fi
for tr_object in "$tr_ref"/objects/*.o;do
 [[ "$tr_object" == */cmf_toy_model.o || "$tr_object" == */lumina_cmfgen.o ]] && continue
 cp "$tr_object" "$tr_out/objects/"
done
for tr_unit in cmf_exact_multigpu lumina_cmf_solve lumina_bf_gemm cmf_observer;do
 cmp "src/$tr_unit.cu" "$tr_base/ada-signed-20261016-0332/source/src/$tr_unit.cu"
 cp --parents "src/$tr_unit.cu" "$tr_out/source/"
 if [[ -z "${SLURM_JOB_ID:-}" ]];then cp "$tr_base/ada-signed-20261016-0332/$tr_unit-cuda.o" "$tr_out/objects/";fi
done
cd "$tr_out/source"
sha256sum src/*.h src/*.c src/*.cu *.f tests/*.c scripts/*.sh scripts/*.slurm > "$tr_out/source.sha256"
(cd "$tr_out/objects";sha256sum *.o) > "$tr_out/reused-objects.sha256"
tr_flags=(-O2 -std=gnu11 -D_GNU_SOURCE -DLUMINA_HAS_CUDA_BF_GEMM -fopenmp -ffunction-sections -fdata-sections -Isrc)
gcc "${tr_flags[@]}" -Dcmf_toy_ejecta_read_refined=temperature_ejecta_read_refined -c src/cmf_toy_model.c -o "$tr_out/objects/cmf_toy_model.o" > "$tr_out/build.log" 2>&1
gcc "${tr_flags[@]}" -c src/lumina_cmfgen.c -o "$tr_out/objects/lumina_cmfgen.o" >> "$tr_out/build.log" 2>&1
for tr_unit in "${tr_extra[@]}";do
 gcc "${tr_flags[@]}" -c "src/$tr_unit.c" -o "$tr_out/objects/$tr_unit.o" >> "$tr_out/build.log" 2>&1
done
gfortran -O2 -ffixed-line-length-none -c set_kind_module.f mon_interp.f error_unit.f >> "$tr_out/build.log" 2>&1
cp set_kind_module.o mon_interp.o error_unit.o "$tr_out/objects/"
if [[ -n "${SLURM_JOB_ID:-}" ]];then
 for tr_unit in cmf_exact_multigpu lumina_cmf_solve lumina_bf_gemm cmf_observer;do
  nvcc -O2 -arch=native -std=c++14 -Isrc -Xcompiler -fopenmp -DLUMINA_CMF_EXACT_MANAGED_MEMORY=0 -c "src/$tr_unit.cu" -o "$tr_out/objects/$tr_unit-cuda.o" >> "$tr_out/build.log" 2>&1
 done
fi
gcc "${tr_flags[@]}" -c tests/cmf_toy_temperature_response.c -o "$tr_out/test.o" >> "$tr_out/build.log" 2>&1
nvcc -O2 -arch=native -Xcompiler -fopenmp "$tr_out/test.o" "$tr_out/objects/"*.o -Xlinker --gc-sections -lm -lcublas -lgfortran -o "$tr_out/temperature-response" >> "$tr_out/build.log" 2>&1
while read -r tr_key;do unset "$tr_key";done < <(compgen -v LUMINA_ || true)
(cd "$tr_input";sha256sum -c "$tr_base/toy-initial-424528/input.sha256") > "$tr_out/input-verify.log"
cd "$tr_out/work"
if [[ "$tr_mode" == selftest ]];then
 "$tr_out/temperature-response" --selftest "$tr_input/snia_toy06_2d.dat" > "$tr_out/test.log" 2> "$tr_out/test.err"
 sha256sum -c "$tr_base/ada-spatial-20261016-0619/executable-output.sha256" > "$tr_out/gpu-binary-verify.log"
 LUMINA_CMF_ANDERSON1=1 "$tr_base/ada-spatial-20261016-0619/spatial-field" --selftest >> "$tr_out/test.log" 2>> "$tr_out/test.err"
 grep '^TEMPERATURE_MAPPING_PASS ' "$tr_out/test.log"
 grep '^SPATIAL_SELFTEST_PASS ' "$tr_out/test.log"
else
 sha256sum -c "$tr_ref/iterate.sha256" > "$tr_out/iterate-verify.log"
 while IFS= read -r tr_entry;do export "$tr_entry";done < "$tr_base/toy-initial-424528/env_lumina_effective.txt"
 env | LC_ALL=C sort | grep '^LUMINA_' > "$tr_out/env_lumina_effective.txt"
 tr_id=$(sha256sum "$tr_ref/iterate_identity.txt");tr_id=${tr_id%% *}
 "$tr_out/temperature-response" --verify-donor "$tr_input/snia_toy06_2d.dat" "$tr_input" "$tr_ref/iterate_input.bin" "$tr_id" \
  > "$tr_out/donor.log" 2> "$tr_out/donor.err"
 grep -Fx 'TEMPERATURE_DONOR_VALIDATED source_iteration=4 physical_history=0' "$tr_out/donor.log"
 sha256sum -c "$tr_ref/iterate.sha256" > "$tr_out/iterate-between-verify.log"
 "$tr_out/temperature-response" "$tr_input/snia_toy06_2d.dat" "$tr_input" "$tr_ref/iterate_input.bin" "$tr_id" "$tr_mode" \
  "$tr_input/NUC_DECAY_DATA_ni56_co56_fe56" "$tr_input/arnaud_rothenflug_23jun15" "$tr_input/nonthermal_resolved_bindings.csv" \
  > "$tr_out/test.log" 2> "$tr_out/test.err"
 sha256sum mapped_predictor.csv gamma.csv temperature.csv response_material_0.csv response_material_1.csv response_field_0.csv response_field_1.csv > "$tr_out/output.sha256"
 sha256sum -c "$tr_ref/iterate.sha256" > "$tr_out/iterate-final-verify.log"
 grep '^TEMPERATURE_RESPONSE_DONE ' "$tr_out/test.log"
fi
(cd "$tr_out/source";sha256sum -c "$tr_out/source.sha256") > "$tr_out/source-verify.log"
sha256sum "$tr_out/temperature-response" "$tr_out/test.log" "$tr_out/test.err" > "$tr_out/executable-output.sha256"
