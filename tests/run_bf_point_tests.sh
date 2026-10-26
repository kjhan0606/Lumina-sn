#!/usr/bin/env bash
# CPU1/4GiB/300초 직접시험. 기존 산출물과 물질은 수정하지 않는다.
set -euo pipefail
ulimit -c 0
project=$(realpath "${1:?project}")
output=$(realpath "${2:?empty-output}")
test "${SLURM_CPUS_PER_TASK:-}" = 1
test "${SLURM_MEM_PER_NODE:-}" = 4096
test ! -e "$output/source-before.sha256"
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$project"
sha256sum src/*.h src/lumina_plasma.c src/lumina_element_wide.c \
    src/bf_rate_jnu.c tests/a2_05_bf_rate_selftest.c \
    tests/bf_point_matrix_selftest.c tests/run_bf_point_tests.sh > "$output/source-before.sha256"
cd "$output"
cc=(gcc -O1 -std=gnu11 -D_GNU_SOURCE -ffunction-sections -fdata-sections -I"$project/src" -Wl,--gc-sections)
"${cc[@]}" -o bf_rate "$project/tests/a2_05_bf_rate_selftest.c" \
    "$project/src/bf_rate_jnu.c" "$project/src/radiation_field.c" "$project/src/seed_capability.c" -lm
./bf_rate
core=(lumina_plasma lumina_element_wide population_contract opacity_publication
      emissivity_publication radeq_publication line_net_rate cmfgen_adiabatic
      bf_rate_jnu radiation_field seed_capability atomic_internal_energy
      nlte_population_candidate line_jbar lumina_atomic)
objects=()
for name in "${core[@]}"; do
    "${cc[@]}" -DLUMINA_EW_SYNTHETIC_TEST -c "$project/src/$name.c" -o "$name.o"
    objects+=("$name.o")
done
"${cc[@]}" -o bf_matrix "$project/tests/bf_point_matrix_selftest.c" "${objects[@]}" -lm
./bf_matrix
"${cc[@]}" -DLUMINA_EW_SYNTHETIC_TEST -o ew_layout \
    "$project/tests/nlte_element_wide_layout_selftest.c" "${objects[@]}" -lm
./ew_layout
"${cc[@]}" -o population_candidate "$project/tests/nlte_population_candidate_selftest.c" "${objects[@]}" -lm
./population_candidate
cd "$project"
sha256sum -c "$output/source-before.sha256" > "$output/source-after-check.txt"
sha256sum "$output/bf_rate" "$output/bf_matrix" "$output/ew_layout" \
    "$output/population_candidate" > "$output/binary.sha256"
printf 'BF_POINT_TEST_BUNDLE PASS\n'
