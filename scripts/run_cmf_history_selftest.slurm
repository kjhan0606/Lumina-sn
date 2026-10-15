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
unset CMF_TEST_CENTER CMF_TEST_INITIAL
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
 cmfgen_adiabatic cmf_decay cmf_ejecta cmf_exact_sliding cmf_error_envelope cmf_observer cmf_time_moment cmf_time_field cmf_time_epoch nlte_population_candidate
 physics_comparison gpu_physics_contract lumina_plasma lumina_element_wide
 lumina_atomic lumina_cmfgen cmfgen_final_writer a210_coupled_newton)
history_tests=(cmf_observer cmf_ejecta cmf_decay cmf_material_time cmf_time_line_energy bf_channels cmf_charge_exchange cmf_recombination nlte_candidate_adiabatic a2_10_radeq cmf_common_field a2_10_seed_commit physics_comparison population_time population_time_assembly population_time_element a2_07_population)
history_mode=${1:-regression}
history_iteration=${2:-plain}
case "$history_iteration" in
 plain) ;;
 anderson) export LUMINA_CMF_ANDERSON1=1 ;;
 *) exit 2 ;;
esac
case "$history_mode" in
 regression) ;;
 channels) history_tests=(bf_channels) ;;
 ce-charge) history_tests=(cmf_charge_exchange) ;;
 recombination) history_tests=(cmf_recombination) ;;
 energy) history_tests=(cmf_material_time nlte_candidate_adiabatic a2_10_radeq physics_comparison) ;;
 coupled) history_tests=(cmf_time_coupled) ;;
 coupled-center) history_tests=(cmf_time_coupled); export CMF_TEST_CENTER=1 ;;
 initial) history_tests=(cmf_time_coupled); export CMF_TEST_CENTER=1 CMF_TEST_INITIAL=1 ;;
 *) exit 2 ;;
esac
history_files=(scripts/run_cmf_history_selftest.slurm src/*.h)
history_files+=(tests/cmf_time_epoch_selftest.c)
if [[ "$history_mode" == regression ]]; then history_files+=(tests/cmf_decay_oracle.f90); fi
for history_unit in "${history_units[@]}"; do history_files+=("src/$history_unit.c"); done
for history_test in "${history_tests[@]}"; do history_files+=("tests/${history_test}_selftest.c"); done
cp --parents "${history_files[@]}" "$history_out/source/"
cd "$history_out/source"
sha256sum "${history_files[@]}" > "$history_out/source.sha256"
if [[ "$history_mode" == regression ]]; then
 mkdir original
 cp /gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/do_species_decays_v2.f \
    /gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/do_lev_pop_decays.f original/
 sha256sum original/do_species_decays_v2.f original/do_lev_pop_decays.f >> "$history_out/source.sha256"
 gfortran -O2 -ffixed-line-length-none tests/cmf_decay_oracle.f90 \
  original/do_species_decays_v2.f original/do_lev_pop_decays.f -o "$history_out/decay_oracle" \
  > "$history_out/oracle-build.log" 2>&1
 (cd "$history_out"; ./decay_oracle > oracle.log 2> oracle.err)
 cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/input_models/snia_toy06_2d.dat "$history_out/"
 (cd "$history_out"; sha256sum snia_toy06_2d.dat > input.sha256)
fi
history_objects=()
gcc -std=gnu11 -D_GNU_SOURCE -DLUMINA_HAS_CUDA_BF_GEMM -Isrc -fsyntax-only src/lumina_cmfgen.c \
 > "$history_out/cuda-dispatch-build.log" 2>&1
for history_unit in "${history_units[@]}"; do
 gcc -O2 -std=gnu11 -D_GNU_SOURCE -DLUMINA_TIME_PAIR_SELFTEST -fopenmp -ffunction-sections -fdata-sections -Isrc \
  -c "src/$history_unit.c" -o "$history_out/objects/$history_unit.o" >> "$history_out/build.log" 2>&1
 history_objects+=("$history_out/objects/$history_unit.o")
done
history_epoch_wrap=(-Wl,--wrap=cmfgen_time_radiation_publish -Wl,--wrap=lumina_r7_publish_and_solve_te
 -Wl,--wrap=gamma_deposition_require)
gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections -Isrc \
 tests/cmf_time_epoch_selftest.c "${history_objects[@]}" "${history_epoch_wrap[@]}" \
 -Wl,--gc-sections -lm -o "$history_out/time_epoch" >> "$history_out/build.log" 2>&1
(cd "$history_out"; ./time_epoch > time-epoch.log 2> time-epoch.err)
for history_test in "${history_tests[@]}"; do
 gcc -O2 -std=gnu11 -D_GNU_SOURCE -DCMF_MATERIAL_ROOT_SELFTEST -fopenmp -ffunction-sections -fdata-sections -Isrc \
  "tests/${history_test}_selftest.c" "${history_objects[@]}" -Wl,--gc-sections -lm \
  -o "$history_out/$history_test" >> "$history_out/build.log" 2>&1
 (cd "$history_out"; "./$history_test" > "$history_test.log" 2> "$history_test.err")
 if [[ "$history_test" == cmf_recombination ]]; then
  (cd "$history_out"; LUMINA_NLTE_BF_JEQB=1 "./$history_test" > recombination-thermal.log 2> recombination-thermal.err)
 fi
done
if [[ "$history_mode" == initial ]]; then
 # 실제 초기/시간해 연결은 그대로, 새 checkpoint 소유 코드와 호출부를 계측한다.
 history_checkpoint_rest=()
 for history_unit in "${history_units[@]}"; do
  if [[ "$history_unit" != cmf_time_epoch ]]; then
   history_checkpoint_rest+=("$history_out/objects/$history_unit.o")
  fi
 done
 gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -fopenmp -fsanitize=address,undefined \
  -fno-omit-frame-pointer -ffunction-sections -fdata-sections -Isrc \
  tests/cmf_time_coupled_selftest.c src/cmf_time_epoch.c "${history_checkpoint_rest[@]}" \
  -Wl,--gc-sections -lm -o "$history_out/checkpoint_san" > "$history_out/checkpoint-san-build.log" 2>&1
 (cd "$history_out"; mkdir checkpoint-san; cd checkpoint-san
  ASAN_OPTIONS=detect_leaks=1 ../checkpoint_san > ../checkpoint-san.log 2> ../checkpoint-san.err)
fi
if [[ "$history_mode" != regression ]]; then
 sha256sum -c "$history_out/source.sha256" > "$history_out/source_verify.log"
 exit 0
fi
history_san_units=(nlte_population_candidate atomic_internal_energy cmfgen_adiabatic cmf_decay cmf_ejecta population_contract
 opacity_publication emissivity_publication radeq_publication radiation_field seed_capability line_net_rate)
history_san_files=()
for history_unit in "${history_san_units[@]}"; do history_san_files+=("src/$history_unit.c"); done
gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer \
 -ffunction-sections -fdata-sections -Isrc tests/cmf_decay_selftest.c \
 "${history_san_files[@]}" -Wl,--gc-sections -lm -o "$history_out/decay_san" \
 > "$history_out/decay-san-build.log" 2>&1
(cd "$history_out"; ASAN_OPTIONS=detect_leaks=1 ./decay_san > decay-san.log 2> decay-san.err)
gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer \
 -ffunction-sections -fdata-sections -Isrc tests/cmf_ejecta_selftest.c \
 "${history_san_files[@]}" -Wl,--gc-sections -lm -o "$history_out/ejecta_san" \
 > "$history_out/ejecta-san-build.log" 2>&1
(cd "$history_out"; ASAN_OPTIONS=detect_leaks=1 ./ejecta_san > ejecta-san.log 2> ejecta-san.err)
gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer \
 -ffunction-sections -fdata-sections -Isrc tests/cmf_time_epoch_selftest.c src/cmf_time_epoch.c src/cmf_time_field.c \
 "${history_san_files[@]}" "${history_epoch_wrap[@]}" -Wl,--gc-sections -lm -o "$history_out/epoch_san" \
 > "$history_out/epoch-san-build.log" 2>&1
(cd "$history_out"; ASAN_OPTIONS=detect_leaks=1 ./epoch_san > epoch-san.log 2> epoch-san.err)
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
gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -DLUMINA_TIME_PAIR_SELFTEST -fopenmp \
 -fsanitize=address,undefined -fno-omit-frame-pointer -ffunction-sections -fdata-sections -Isrc \
 tests/bf_channels_selftest.c src/lumina_plasma.c src/lumina_element_wide.c \
 src/nlte_population_candidate.c src/population_contract.c "${history_element_rest[@]}" \
 -Wl,--gc-sections -lm -o "$history_out/bf_channels_san" > "$history_out/channels-san-build.log" 2>&1
(cd "$history_out"; mkdir channels-san; cd channels-san; ASAN_OPTIONS=detect_leaks=1 ../bf_channels_san > ../channels-san.log 2> ../channels-san.err)
sha256sum -c "$history_out/source.sha256" > "$history_out/source_verify.log"
(cd "$history_out"; sha256sum -c input.sha256 > input_verify.log)
