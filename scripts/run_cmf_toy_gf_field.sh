#!/bin/bash
# 단발 고정물질 비교. 수동 Ada는 foreground timeout, 전체장 Slurm은 job 한도다.
set -euo pipefail
umask 027
if [[ -z "${SLURM_JOB_ID:-}" ]]; then [[ "$(hostname -s)" == LagEunha ]]; fi
gf_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
gf_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
gf_ref=$gf_base/toy-material-424593
gf_cuda=$gf_base/ada-signed-20261016-0332
gf_out=${CMF_GF_TEST_OUT:?new absolute output directory required}
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
 [[ "$gf_out" == "$gf_base/gf-field-$SLURM_JOB_ID" ]]
else
 [[ "$gf_out" == "$gf_base"/ada-gf-* ]]
fi
mkdir "$gf_out" "$gf_out/source" "$gf_out/objects" "$gf_out/work"
gf_start=$SECONDS
trap 'gf_rc=$?; printf "RUN_FOOTER exit=%s elapsed_seconds=%s GPUs=1\n" "$gf_rc" "$((SECONDS-gf_start))"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0 cuda/13.0.2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_PLACES=cores OMP_PROC_BIND=close OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -z "${SLURM_JOB_ID:-}" ]]; then export CUDA_VISIBLE_DEVICES=0; fi
cd "$gf_src"
git rev-parse --show-toplevel > "$gf_out/project.txt"
git rev-parse HEAD > "$gf_out/commit.txt"
(cd "$gf_ref/source"; sha256sum -c "$gf_ref/source.sha256") > "$gf_out/reference-source-verify.log"
(cd "$gf_ref/input"; sha256sum -c "$gf_ref/input.sha256") > "$gf_out/input-verify.log"
sha256sum -c "$gf_ref/iterate.sha256" > "$gf_out/iterate-verify.log"
while read -r gf_hash gf_file;do
 [[ "$gf_file" == src/* ]] || continue
 cmp "$gf_file" "$gf_ref/source/$gf_file"
 cp --parents "$gf_file" "$gf_out/source/"
done < "$gf_ref/source.sha256"
cp --parents tests/cmf_toy_gf_field_diagnostic.c tests/cmf_toy_bf_iterate_diagnostic.c scripts/run_cmf_toy_gf_field.sh scripts/run_cmf_toy_gf_field.slurm "$gf_out/source/"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then scontrol write batch_script "$SLURM_JOB_ID" "$gf_out/runner.slurm"; fi
for gf_unit in cmf_exact_multigpu lumina_cmf_solve lumina_bf_gemm cmf_observer;do
 cmp "src/$gf_unit.cu" "$gf_cuda/source/src/$gf_unit.cu"
 cp --parents "src/$gf_unit.cu" "$gf_out/source/"
 if [[ -z "${SLURM_JOB_ID:-}" ]]; then cp "$gf_cuda/$gf_unit-cuda.o" "$gf_out/objects/"; fi
done
for gf_object in "$gf_ref"/objects/*.o;do
 [[ "$gf_object" == */lumina_cmfgen.o ]] && continue
 cp "$gf_object" "$gf_out/objects/"
done
cd "$gf_out/source"
sha256sum src/*.h src/*.c src/*.cu tests/*.c scripts/*.sh scripts/*.slurm > "$gf_out/source.sha256"
(cd "$gf_out/objects"; sha256sum *.o) > "$gf_out/reused-objects.sha256"
gcc -O2 -std=gnu11 -D_GNU_SOURCE -DLUMINA_HAS_CUDA_BF_GEMM -fopenmp \
 -ffunction-sections -fdata-sections -Isrc -c src/lumina_cmfgen.c \
 -o "$gf_out/objects/lumina_cmfgen.o" > "$gf_out/build.log" 2>&1
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
 for gf_unit in cmf_exact_multigpu lumina_cmf_solve lumina_bf_gemm cmf_observer;do
  nvcc -O2 -arch=native -std=c++14 -Isrc -Xcompiler -fopenmp \
   -DLUMINA_CMF_EXACT_MANAGED_MEMORY=0 -c "src/$gf_unit.cu" \
   -o "$gf_out/objects/$gf_unit-cuda.o" >> "$gf_out/build.log" 2>&1
 done
fi
gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections -Isrc \
 -c tests/cmf_toy_gf_field_diagnostic.c -o "$gf_out/test.o" >> "$gf_out/build.log" 2>&1
nvcc -O2 -arch=native -Xcompiler -fopenmp "$gf_out/test.o" "$gf_out/objects/"*.o \
 -Xlinker --gc-sections -lm -lcublas -o "$gf_out/gf-field" >> "$gf_out/build.log" 2>&1
"$gf_out/gf-field" --selftest > "$gf_out/selftest.log" 2>&1
while read -r gf_key;do unset "$gf_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r gf_entry;do export "$gf_entry";done < "$gf_base/toy-initial-424528/env_lumina_effective.txt"
export LUMINA_BF_CHANNELS_FILE=$gf_ref/input/bf_channels.bin
export LUMINA_TOPION_LEVELS_FILE=$gf_ref/input/bf_channels.terminal.csv
export LUMINA_IONIZATION_REFERENCE_FILE=$gf_ref/input/ionization_reference.csv
unset CMF_TEST_BF_LTE_LEVELS
env | LC_ALL=C sort | grep '^LUMINA_' > "$gf_out/env_lumina_effective.txt"
gf_id=$(sha256sum "$gf_ref/iterate_identity.txt");gf_id=${gf_id%% *}
{
 sha256sum "$gf_out/gf-field" "$gf_out/source.sha256" "$gf_ref/iterate_input.bin"
 printf 'diagnostic=FROZEN_MATERIAL_GF_FIELD\nGF_CUT=0.001\nGF_LEV_CUT=9\nMIN_TRANS=10\nAT_CUT=10\npublic_exact_parameters_known=0\n'
} > "$gf_out/diagnostic_identity.txt"
gf_new_id=$(sha256sum "$gf_out/diagnostic_identity.txt");gf_new_id=${gf_new_id%% *}
cd "$gf_out/work"
"$gf_out/gf-field" "$gf_ref/input/snia_toy06_2d.dat" "$gf_ref/input" \
 "$gf_ref/iterate_input.bin" "$gf_id" "$gf_new_id" .001 > "$gf_out/test.log" 2> "$gf_out/test.err"
sha256sum baseline_bf.csv selected_bf.csv selected_iterate.bin field_change.csv > "$gf_out/output.sha256"
(cd "$gf_out/source"; sha256sum -c "$gf_out/source.sha256") > "$gf_out/source-verify.log"
sha256sum -c "$gf_ref/iterate.sha256" > "$gf_out/iterate-final-verify.log"
grep -Fx 'GF_FIELD_DONE frozen_material_byte_identical=1 field_recomputed=1 SE_solved=0 physical_history=0 final_goal=NOT_JUDGED' "$gf_out/test.log"
