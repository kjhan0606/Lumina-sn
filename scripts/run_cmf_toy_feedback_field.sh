#!/bin/bash
set -euo pipefail
umask 027
fb_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
fb_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
fb_ref=$fb_base/toy-material-424656
fb_cuda=$fb_base/ada-signed-20261016-0332
fb_out=${CMF_FEEDBACK_TEST_OUT:?new absolute output root required}
if [[ -n "${SLURM_JOB_ID:-}" ]];then
 [[ "$fb_out" == "$fb_base/feedback-field-$SLURM_JOB_ID" ]]
else
 [[ "$(hostname -s)" == LagEunha && "$fb_out" == "$fb_base"/ada-feedback-* ]]
fi
mkdir "$fb_out" "$fb_out/source" "$fb_out/objects"
fb_start=$SECONDS
trap 'fb_rc=$?; printf "RUN_FOOTER exit=%s elapsed_seconds=%s GPUs=1\n" "$fb_rc" "$((SECONDS-fb_start))"' EXIT
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -z "${SLURM_JOB_ID:-}" ]];then export CUDA_VISIBLE_DEVICES=0;fi
cd "$fb_src"
[[ "$(git rev-parse --show-toplevel)" == "$fb_src" ]]
git rev-parse HEAD > "$fb_out/commit.txt"
(cd "$fb_ref/source";sha256sum -c "$fb_ref/source.sha256") > "$fb_out/reference-source-verify.log"
while read -r fb_hash fb_file;do
 [[ "$fb_file" == src/* ]] || continue
 cmp "$fb_file" "$fb_ref/source/$fb_file"
 cp --parents "$fb_file" "$fb_out/source/"
done < "$fb_ref/source.sha256"
cp --parents tests/cmf_toy_feedback_field.c tests/cmf_toy_bf_iterate_diagnostic.c \
 scripts/compare_cmf_feedback_field.py scripts/run_cmf_toy_feedback_field.sh scripts/run_cmf_toy_feedback_field.slurm "$fb_out/source/"
for fb_unit in cmf_exact_multigpu lumina_cmf_solve lumina_bf_gemm cmf_observer;do
 cmp "src/$fb_unit.cu" "$fb_cuda/source/src/$fb_unit.cu"
 cp --parents "src/$fb_unit.cu" "$fb_out/source/"
 if [[ -z "${SLURM_JOB_ID:-}" ]];then cp "$fb_cuda/$fb_unit-cuda.o" "$fb_out/objects/";fi
done
for fb_object in "$fb_ref"/objects/*.o;do
 [[ "$fb_object" == */lumina_cmfgen.o ]] && continue
 cp "$fb_object" "$fb_out/objects/"
done
cd "$fb_out/source"
sha256sum src/*.h src/*.c src/*.cu tests/*.c scripts/*.sh scripts/*.slurm scripts/*.py > "$fb_out/source.sha256"
(cd "$fb_out/objects";sha256sum *.o) > "$fb_out/reused-objects.sha256"
fb_flags=(-O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections -Isrc)
gcc "${fb_flags[@]}" -DLUMINA_HAS_CUDA_BF_GEMM -c src/lumina_cmfgen.c -o "$fb_out/objects/lumina_cmfgen.o" > "$fb_out/build.log" 2>&1
if [[ -n "${SLURM_JOB_ID:-}" ]];then
 for fb_unit in cmf_exact_multigpu lumina_cmf_solve lumina_bf_gemm cmf_observer;do
  nvcc -O2 -arch=native -std=c++14 -Isrc -Xcompiler -fopenmp -DLUMINA_CMF_EXACT_MANAGED_MEMORY=0 \
   -c "src/$fb_unit.cu" -o "$fb_out/objects/$fb_unit-cuda.o" >> "$fb_out/build.log" 2>&1
 done
fi
gcc "${fb_flags[@]}" -c tests/cmf_toy_feedback_field.c -o "$fb_out/test.o" >> "$fb_out/build.log" 2>&1
nvcc -O2 -arch=native -Xcompiler -fopenmp "$fb_out/test.o" "$fb_out/objects/"*.o \
 -Xlinker --gc-sections -lm -lcublas -o "$fb_out/feedback-field" >> "$fb_out/build.log" 2>&1
while read -r fb_key;do unset "$fb_key";done < <(compgen -v LUMINA_ || true)
cd "$fb_out"
python3 source/scripts/compare_cmf_feedback_field.py --selftest > analysis-selftest.log
./feedback-field --selftest > selftest.log 2> selftest.err
if [[ -z "${SLURM_JOB_ID:-}" ]];then
 sha256sum -c "$fb_base/ada-spatial-20261016-0619/executable-output.sha256" > gpu-binary-verify.log
 LUMINA_CMF_ANDERSON1=1 "$fb_base/ada-spatial-20261016-0619/spatial-field" --selftest > gpu-selftest.log 2> gpu-selftest.err
 grep '^SPATIAL_SELFTEST_PASS ' gpu-selftest.log
 grep '^FEEDBACK_SHIFT_SELFTEST_PASS ' selftest.log
else
 scontrol write batch_script "$SLURM_JOB_ID" runner.slurm
 (cd "$fb_ref/input";sha256sum -c "$fb_ref/input.sha256") > input-verify.log
 sha256sum -c "$fb_ref/iterate.sha256" > iterate-verify.log
 while IFS= read -r fb_entry;do export "$fb_entry";done < "$fb_base/toy-initial-424528/env_lumina_effective.txt"
 export LUMINA_BF_CHANNELS_FILE=$fb_ref/input/bf_channels.bin
 export LUMINA_TOPION_LEVELS_FILE=$fb_ref/input/bf_channels.terminal.csv
 export LUMINA_IONIZATION_REFERENCE_FILE=$fb_ref/input/ionization_reference.csv
 unset CMF_TEST_BF_LTE_LEVELS
 env | LC_ALL=C sort | grep '^LUMINA_' > env_lumina_effective.txt
 fb_id=$(sha256sum "$fb_ref/iterate_identity.txt");fb_id=${fb_id%% *}
 for fb_mode in zero plus minus;do
  case "$fb_mode" in zero) fb_delta=0 ;; plus) fb_delta=0.01 ;; minus) fb_delta=-0.01 ;; esac
  mkdir "$fb_mode"
  (
   cd "$fb_mode"
   sha256sum ../source.sha256 ../feedback-field ../env_lumina_effective.txt "$fb_ref/iterate_input.bin" > diagnostic_identity.txt
   printf 'purpose=private_sulfur_feedback_no_SE_no_physical_history\ndelta=%s\nshell=39\n' "$fb_delta" >> diagnostic_identity.txt
   fb_diagnostic_id=$(sha256sum diagnostic_identity.txt);fb_diagnostic_id=${fb_diagnostic_id%% *}
   ../feedback-field "$fb_ref/input/snia_toy06_2d.dat" "$fb_ref/input" "$fb_ref/iterate_input.bin" \
    "$fb_id" "$fb_delta" "$fb_diagnostic_id" > test.log 2> test.err
   sha256sum stored_field_bf.csv new_field_bf.csv diagnostic_material.csv diagnostic_state.bin \
    perturbation.json field_change.csv diagnostic_identity.txt test.log test.err > output.sha256
   grep '^FEEDBACK_FIELD_DONE ' test.log
  )
 done
 python3 source/scripts/compare_cmf_feedback_field.py . > comparison.log
 sha256sum feedback.json comparison.log zero/output.sha256 plus/output.sha256 minus/output.sha256 > output.sha256
 sha256sum -c "$fb_ref/iterate.sha256" > iterate-final-verify.log
fi
(cd source;sha256sum -c ../source.sha256) > source-verify.log
sha256sum feedback-field selftest.log selftest.err analysis-selftest.log > executable-output.sha256
