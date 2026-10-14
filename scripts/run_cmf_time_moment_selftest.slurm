#!/bin/bash
#SBATCH --job-name=CMF_TIME_MOMENT_TEST
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/time-moment-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/time-moment-%j.err
set -euo pipefail
umask 027
moment_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
moment_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/time-moment-${SLURM_JOB_ID}
mkdir "$moment_out"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$moment_src"
git rev-parse HEAD > "$moment_out/commit.txt"
moment_files=(src/cmf_time_moment.c src/cmf_time_moment.h src/cmf_exact_sliding.c
 src/cmf_time_field.c src/cmf_time_field.h tests/cmf_time_field_selftest.c
 src/cmfgen_adiabatic.c src/cmfgen_adiabatic.h tests/cmf_material_time_selftest.c
 tests/cmfgen_adiabatic_selftest.c
 src/cmf_exact_sliding.h src/cmf_cell_geometry.h src/cmf_error_envelope.c
 src/cmf_error_envelope.h tests/cmf_time_moment_selftest.c
 tests/cmf_formal_moments_selftest.c tests/cmf_exact_sliding_selftest.c
 scripts/run_cmf_time_moment_selftest.slurm)
mkdir "$moment_out/source"
cp --parents "${moment_files[@]}" "$moment_out/source/"
cd "$moment_out/source"
sha256sum "${moment_files[@]}" > "$moment_out/source.sha256"
moment_units=(src/cmf_time_moment.c src/cmf_time_field.c src/cmf_exact_sliding.c src/cmf_error_envelope.c)
gcc -O2 -std=c11 -Wall -Wextra -Werror -Isrc "${moment_units[@]}" \
 tests/cmf_time_field_selftest.c -lm -o "$moment_out/field_test" > "$moment_out/field-build.log" 2>&1
"$moment_out/field_test" > "$moment_out/field.log" 2> "$moment_out/field.err"
gcc -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
 "${moment_units[@]}" tests/cmf_time_field_selftest.c -lm -o "$moment_out/field_san" \
 > "$moment_out/field-san-build.log" 2>&1
ASAN_OPTIONS=detect_leaks=1 "$moment_out/field_san" > "$moment_out/field-san.log" 2> "$moment_out/field-san.err"
gcc -O2 -std=c11 -Wall -Wextra -Werror -Isrc "${moment_units[@]}" \
 tests/cmf_time_moment_selftest.c -lm -o "$moment_out/selftest" > "$moment_out/build.log" 2>&1
"$moment_out/selftest" > "$moment_out/test.log" 2> "$moment_out/test.err"
gcc -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
 "${moment_units[@]}" tests/cmf_time_moment_selftest.c -lm -o "$moment_out/selftest_san" \
 > "$moment_out/build_san.log" 2>&1
ASAN_OPTIONS=detect_leaks=1 "$moment_out/selftest_san" > "$moment_out/san.log" 2> "$moment_out/san.err"
gcc -O2 -std=c11 -Wall -Wextra -Werror -Isrc "${moment_units[@]}" \
 tests/cmf_formal_moments_selftest.c -lm -o "$moment_out/formal_test" > "$moment_out/formal_build.log" 2>&1
"$moment_out/formal_test" > "$moment_out/formal.log" 2> "$moment_out/formal.err"
gcc -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
 "${moment_units[@]}" tests/cmf_formal_moments_selftest.c -lm -o "$moment_out/formal_san" \
 > "$moment_out/formal-san-build.log" 2>&1
ASAN_OPTIONS=detect_leaks=1 "$moment_out/formal_san" > "$moment_out/formal-san.log" 2> "$moment_out/formal-san.err"
gcc -O2 -std=c11 -Wall -Wextra -Isrc "${moment_units[@]}" \
 tests/cmf_exact_sliding_selftest.c -lm -o "$moment_out/legacy_test" > "$moment_out/legacy_build.log" 2>&1
"$moment_out/legacy_test" > "$moment_out/legacy.log" 2> "$moment_out/legacy.err"
sha256sum -c "$moment_out/source.sha256" > "$moment_out/source_verify.log"
gcc -O2 -std=c11 -Wall -Wextra -Werror -Isrc src/cmfgen_adiabatic.c \
 tests/cmf_material_time_selftest.c -lm -o "$moment_out/material_test" > "$moment_out/material-build.log" 2>&1
"$moment_out/material_test" > "$moment_out/material.log" 2> "$moment_out/material.err"
gcc -O1 -g -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc \
 src/cmfgen_adiabatic.c tests/cmf_material_time_selftest.c -lm -o "$moment_out/material_san" \
 > "$moment_out/material-san-build.log" 2>&1
ASAN_OPTIONS=detect_leaks=1 "$moment_out/material_san" > "$moment_out/material-san.log" 2> "$moment_out/material-san.err"
gcc -O2 -std=c11 -Wall -Wextra -Werror -Isrc src/cmfgen_adiabatic.c \
 tests/cmfgen_adiabatic_selftest.c -lm -o "$moment_out/material_legacy" > "$moment_out/material-legacy-build.log" 2>&1
"$moment_out/material_legacy" > "$moment_out/material-legacy.log" 2> "$moment_out/material-legacy.err"
sha256sum -c "$moment_out/source.sha256" > "$moment_out/source_verify.log"
