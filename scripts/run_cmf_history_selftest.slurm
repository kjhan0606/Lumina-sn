#!/bin/bash
#SBATCH --job-name=CMF_HISTORY_INTEGRATION
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/history-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/history-%j.err
set -euo pipefail
umask 027
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while read -r history_key; do unset "$history_key"; done < <(compgen -v LUMINA_ || true)
history_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
history_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/history-${SLURM_JOB_ID}
mkdir "$history_out"
mkdir "$history_out/source" "$history_out/objects"
cd "$history_src"
git rev-parse HEAD > "$history_out/commit.txt"
history_units=(lumina_transport a2_02c_segment_capture radiation_field bf_rate_jnu
 line_jbar seed_capability jnu_seed population_contract atomic_internal_energy
 opacity_publication emissivity_publication radeq_publication line_net_rate
 cmfgen_adiabatic cmf_exact_sliding cmf_error_envelope cmf_observer nlte_population_candidate
 physics_comparison gpu_physics_contract lumina_plasma lumina_element_wide
 lumina_atomic lumina_cmfgen cmfgen_final_writer a210_coupled_newton)
history_tests=(nlte_candidate_adiabatic a2_10_radeq cmf_common_field a2_10_seed_commit physics_comparison population_time population_time_assembly population_time_element a2_07_population)
history_files=(scripts/run_cmf_history_selftest.slurm src/*.h)
for history_unit in "${history_units[@]}"; do history_files+=("src/$history_unit.c"); done
for history_test in "${history_tests[@]}"; do history_files+=("tests/${history_test}_selftest.c"); done
cp --parents "${history_files[@]}" "$history_out/source/"
cd "$history_out/source"
sha256sum "${history_files[@]}" > "$history_out/source.sha256"
history_objects=()
gcc -std=gnu11 -D_GNU_SOURCE -DLUMINA_HAS_CUDA_BF_GEMM -Isrc -fsyntax-only src/lumina_cmfgen.c \
 > "$history_out/cuda-dispatch-build.log" 2>&1
for history_unit in "${history_units[@]}"; do
 gcc -O2 -std=gnu11 -D_GNU_SOURCE -DLUMINA_TIME_PAIR_SELFTEST -fopenmp -ffunction-sections -fdata-sections -Isrc \
  -c "src/$history_unit.c" -o "$history_out/objects/$history_unit.o" >> "$history_out/build.log" 2>&1
 history_objects+=("$history_out/objects/$history_unit.o")
done
for history_test in "${history_tests[@]}"; do
 gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections -Isrc \
  "tests/${history_test}_selftest.c" "${history_objects[@]}" -Wl,--gc-sections -lm \
  -o "$history_out/$history_test" >> "$history_out/build.log" 2>&1
 (cd "$history_out"; "./$history_test" > "$history_test.log" 2> "$history_test.err")
done
history_san_units=(nlte_population_candidate atomic_internal_energy cmfgen_adiabatic population_contract
 opacity_publication emissivity_publication radeq_publication radiation_field seed_capability line_net_rate)
history_san_files=()
for history_unit in "${history_san_units[@]}"; do history_san_files+=("src/$history_unit.c"); done
gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer \
 -ffunction-sections -fdata-sections -Isrc tests/nlte_candidate_adiabatic_selftest.c \
 "${history_san_files[@]}" -Wl,--gc-sections -lm -o "$history_out/history_san" \
 > "$history_out/san-build.log" 2>&1
(cd "$history_out"; ASAN_OPTIONS=detect_leaks=1 ./history_san > san.log 2> san.err)
gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer \
 -ffunction-sections -fdata-sections -Isrc tests/population_time_selftest.c \
 "${history_san_files[@]}" -Wl,--gc-sections -lm -o "$history_out/population_time_san" \
 > "$history_out/pop-san-build.log" 2>&1
(cd "$history_out"; ASAN_OPTIONS=detect_leaks=1 ./population_time_san > pop-san.log 2> pop-san.err)
history_element_rest=()
for history_unit in "${history_units[@]}"; do
 case "$history_unit" in
  lumina_plasma|lumina_element_wide|nlte_population_candidate|population_contract) ;;
  *) history_element_rest+=("$history_out/objects/$history_unit.o") ;;
 esac
done
gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -DLUMINA_TIME_PAIR_SELFTEST -fopenmp \
 -fsanitize=address,undefined -fno-omit-frame-pointer -ffunction-sections -fdata-sections -Isrc \
 tests/population_time_element_selftest.c src/lumina_plasma.c src/lumina_element_wide.c \
 src/nlte_population_candidate.c src/population_contract.c "${history_element_rest[@]}" \
 -Wl,--gc-sections -lm -o "$history_out/element_time_san" > "$history_out/element-san-build.log" 2>&1
(cd "$history_out"; ASAN_OPTIONS=detect_leaks=1 ./element_time_san > element-san.log 2> element-san.err)
sha256sum -c "$history_out/source.sha256" > "$history_out/source_verify.log"
