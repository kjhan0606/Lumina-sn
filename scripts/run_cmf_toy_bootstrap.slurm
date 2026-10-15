#!/bin/bash
#SBATCH --job-name=CMF_TOY_MATERIAL
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --exclude=syn102,syn03
#SBATCH --output=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-material-%j.log
#SBATCH --error=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-material-%j.err
set -euo pipefail
umask 027
toy_shells=${1:-100}
case "$toy_shells" in 100|200) ;; *) exit 2 ;; esac
toy_mode=${2:-material}
case "$toy_mode" in
 material) toy_test=tests/cmf_toy_model_selftest.c ;;
 material-rates|material-rates-components|material-rates-profile|material-rates-cache) toy_test=tests/cmf_toy_rates_selftest.c ;;
 material-epoch) toy_test=tests/cmf_toy_epoch_selftest.c ;;
 nonthermal) toy_test=tests/cmf_nonthermal_actual_selftest.c ;;
 initial-prepare) toy_test=tests/cmf_toy_initial_selftest.c ;;
 *) exit 2 ;;
esac
toy_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
toy_original=${3:-/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/model}
toy_channels=${4:-/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/bf-input-423197}
toy_out=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-material-${SLURM_JOB_ID}
mkdir "$toy_out"
mkdir "$toy_out/source" "$toy_out/objects" "$toy_out/input"
source /opt/ohpc/admin/lmod/lmod/init/bash
module purge
module load gnu13/13.2.0
if [[ "$toy_mode" == nonthermal ]]; then module load python/3.11.14; fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset CMF_TEST_ISOLATE_CE CMF_TEST_PROFILE_CACHED CMF_TEST_RATE_CACHE
if [[ "$toy_mode" == material-rates-cache ]]; then export CMF_TEST_RATE_CACHE=1; fi
toy_profile_flags=()
if [[ "$toy_mode" == material-rates-profile ]]; then
 toy_profile_flags=(-pg);export CMF_TEST_PROFILE_CACHED=1
fi
while read -r toy_key; do unset "$toy_key"; done < <(compgen -v LUMINA_ || true)
export LUMINA_PURE_CMFGEN=1 LUMINA_CMF_COMMON_FIELD=1 LUMINA_CMF_BF_MILNE=2
export LUMINA_SUPER_LEVELS=1 LUMINA_NLTE_COLL_ION=1 LUMINA_OMEGA_CMFGEN=1
if [[ "$toy_mode" == material-rates* ]]; then
 export LUMINA_NLTE_JEQB=1 LUMINA_NLTE_BF_JEQB=1
fi
export LUMINA_BF_CHANNELS_FILE=$toy_out/input/bf_channels.bin
export LUMINA_TOPION_LEVELS_FILE=$toy_out/input/bf_channels.terminal.csv
export LUMINA_IONIZATION_REFERENCE_FILE=$toy_out/input/ionization_reference.csv
toy_units=(lumina_transport a2_02c_segment_capture radiation_field bf_rate_jnu
 line_jbar seed_capability jnu_seed population_contract atomic_internal_energy
 opacity_publication emissivity_publication radeq_publication line_net_rate
 cmfgen_adiabatic cmf_decay cmf_ejecta cmf_toy_model cmf_exact_sliding cmf_error_envelope
 cmf_observer cmf_time_moment cmf_time_field cmf_time_epoch nlte_population_candidate
 physics_comparison gpu_physics_contract lumina_plasma lumina_element_wide
 lumina_atomic lumina_cmfgen cmfgen_final_writer a210_coupled_newton)
if [[ "$toy_mode" == nonthermal ]]; then
 toy_units+=(cmf_nonthermal cmf_nonthermal_atomic cmf_nonthermal_provider cmf_nonthermal_excitation cmf_nonthermal_binding)
fi
if [[ "$toy_mode" == initial-prepare || "$toy_mode" == material-epoch ]]; then
 toy_units+=(cmf_nonthermal cmf_nonthermal_provider cmf_gamma_source cmf_gamma_transport cmf_toy_run)
fi
cd "$toy_src"
git rev-parse HEAD > "$toy_out/commit.txt"
toy_files=(src/*.h scripts/run_cmf_toy_bootstrap.slurm "$toy_test")
if [[ "$toy_mode" == nonthermal ]]; then toy_files+=(scripts/map_cmf_nonthermal.py); fi
for toy_unit in "${toy_units[@]}"; do toy_files+=("src/$toy_unit.c"); done
cp --parents "${toy_files[@]}" "$toy_out/source/"
toy_inputs=(line_list.csv levels.csv ionization_energies.csv zeta_ions.csv zeta_temps.csv
 zeta_data.npy atom_masses.csv coldata_cmfgen_manifest.csv)
while read -r toy_table; do
 [[ "$toy_table" =~ ^ige_col_[0-9]+_[0-9]+_cmfgen.bin$ ]]
 toy_inputs+=("$toy_table")
done < <(awk -F, '{gsub(/\r/, "");if($NF=="OK")print $(NF-1)}' "$toy_original/coldata_cmfgen_manifest.csv")
for toy_input in "${toy_inputs[@]}"; do cp "$toy_original/$toy_input" "$toy_out/input/"; done
cp "$toy_channels/bf_channels.bin" "$toy_channels/bf_channels.terminal.csv" "$toy_out/input/"
cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/input_models/snia_toy06_2d.dat "$toy_out/input/"
cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/ionization_reference.csv "$toy_out/input/"
if [[ "$toy_mode" == initial-prepare || "$toy_mode" == material-epoch ]]; then
 cp /gpfs/kjhan/cmfgen_21jun23/atomic/rad_decay_data/NUC_DECAY_DATA_ni56_co56_fe56 "$toy_out/input/"
fi
if [[ "$toy_mode" == nonthermal ]]; then
 cp /gpfs/kjhan/cmfgen_21jun23/atomic/misc/arnaud_rothenflug_23jun15 "$toy_out/input/"
 (cd "$toy_out"; python3 source/scripts/map_cmf_nonthermal.py > mapping.log 2> mapping.err)
 # 목표준위 처방 비교의 출처다. 실행/링크하지 않는 원본 읽기 자료다.
 mkdir "$toy_out/reference-source"
 cp /gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/non_therm/se_ba_non_therm_v2.f90 \
  /gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/non_therm/se_ba_non_therm_v3.f90 \
  /gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/non_therm/read_arnaud_ion_data.f90 \
  "$toy_out/reference-source/"
 (cd "$toy_out/reference-source"; sha256sum -- * > "$toy_out/reference-source.sha256")
fi
# 기존50셸 조성은 의도적으로 복사하지 않는다.
test ! -e "$toy_out/input/abundances.csv"
(cd "$toy_out/input"; sha256sum -- * > "$toy_out/input.sha256")
env | LC_ALL=C sort | grep '^LUMINA_' > "$toy_out/env_lumina_effective.txt"
cd "$toy_out/source"
sha256sum "${toy_files[@]}" > "$toy_out/source.sha256"
toy_objects=()
for toy_unit in "${toy_units[@]}"; do
 gcc -O2 "${toy_profile_flags[@]}" -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections -Isrc \
  -c "src/$toy_unit.c" -o "$toy_out/objects/$toy_unit.o" >> "$toy_out/build.log" 2>&1
 toy_objects+=("$toy_out/objects/$toy_unit.o")
done
gcc -O2 "${toy_profile_flags[@]}" -std=gnu11 -D_GNU_SOURCE -fopenmp -Isrc "$toy_test" \
 "${toy_objects[@]}" -Wl,--gc-sections -lm -o "$toy_out/prepare" >> "$toy_out/build.log" 2>&1
toy_arguments=(input/snia_toy06_2d.dat input "$toy_shells")
if [[ "$toy_mode" == nonthermal ]]; then toy_arguments+=(nonthermal_resolved_bindings.csv); fi
if [[ "$toy_mode" == material-rates-components ]]; then
 # 동일 binary의 새 프로세스로 정적 env cache도 분리한다. 생산 처방이 아니다.
 for toy_component in without-dr without-ce without-dr-ce; do
  toy_dr=1;toy_noce=0
  case "$toy_component" in
   without-dr) toy_dr=0 ;;
   without-ce) toy_noce=1 ;;
   without-dr-ce) toy_dr=0;toy_noce=1 ;;
  esac
  mkdir "$toy_out/$toy_component"
  (cd "$toy_out/$toy_component";
   env LUMINA_DR_BOOST="$toy_dr" CMF_TEST_ISOLATE_CE="$toy_noce" \
    ../prepare "$toy_out/input/snia_toy06_2d.dat" "$toy_out/input" "$toy_shells" \
    > prepare.log 2> prepare.err)
 done
else
 (cd "$toy_out"; ./prepare "${toy_arguments[@]}" > prepare.log 2> prepare.err)
fi
if [[ "$toy_mode" == material-rates-profile ]]; then
 (cd "$toy_out"; gprof -b -p ./prepare gmon.out > profile.txt)
fi
if [[ "$toy_mode" == material-epoch ]]; then
 # 프로세스당 초기화는1회다. 별도 실제 CLI 프로세스에서 fixture 복원과
 # 감마 준비만 시험하며 cap=0이므로 수렴한 물리 시점으로 저장하지 않는다.
 mkdir "$toy_out/step-prepare"
 (cd "$toy_out/step-prepare"; ../prepare --cmf-toy-step \
  "$toy_out/input/snia_toy06_2d.dat" "$toy_out/input" \
  "$toy_out/input/NUC_DECAY_DATA_ni56_co56_fe56" "$toy_out/fixture_nonconverged_epoch.bin" \
  177120 "$toy_shells" 2000 0 > ../step-prepare.log 2> ../step-prepare.err)
 test ! -e "$toy_out/step-prepare/step_state.bin"
fi
sha256sum -c "$toy_out/source.sha256" > "$toy_out/source_verify.log"
(cd "$toy_out/input"; sha256sum -c "$toy_out/input.sha256" > "$toy_out/input_verify.log")
