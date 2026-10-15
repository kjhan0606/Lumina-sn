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
unset CMF_TEST_CENTER CMF_TEST_INITIAL CMF_TEST_INITIAL_NT CMF_TEST_INITIAL_MATRIX_CACHE CMF_TEST_BF_CACHE_ONLY CMF_TEST_ITERATE CMF_TEST_ITERATE_INPUT
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
 cmfgen_adiabatic cmf_decay cmf_ejecta cmf_toy_model cmf_exact_sliding cmf_error_envelope cmf_observer cmf_time_moment cmf_time_field cmf_time_epoch nlte_population_candidate cmf_nonthermal cmf_nonthermal_provider
 physics_comparison gpu_physics_contract lumina_plasma lumina_element_wide
 lumina_atomic lumina_cmfgen cmfgen_final_writer a210_coupled_newton)
history_tests=(cmf_observer cmf_ejecta cmf_decay cmf_material_time cmf_time_line_energy bf_channels cmf_charge_exchange cmf_recombination nlte_candidate_adiabatic a2_10_radeq cmf_common_field a2_10_seed_commit physics_comparison population_time population_time_assembly population_time_element a2_07_population)
history_mode=${1:-regression}
history_iteration=${2:-plain}
history_warm_ne=${3:-plain}
case "$history_warm_ne" in
 plain) ;;
 anderson) [[ "$history_mode" == initial-ne-warm ]] || exit 2; export LUMINA_NLTE_NE_ANDERSON1=1 ;;
 *) exit 2 ;;
esac
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
 initial-restart) history_tests=(cmf_time_coupled); export CMF_TEST_CENTER=1 CMF_TEST_INITIAL=1 CMF_TEST_ITERATE=1 ;;
 initial-ne-compare) history_tests=(dc1_ne_closure cmf_time_coupled); export CMF_TEST_CENTER=1 CMF_TEST_INITIAL=1 ;;
 initial-nonthermal) history_tests=(cmf_time_coupled); export CMF_TEST_CENTER=1 CMF_TEST_INITIAL=1 CMF_TEST_INITIAL_NT=1 ;;
 initial-matrix-cache) history_tests=(cmf_time_coupled); export CMF_TEST_CENTER=1 CMF_TEST_INITIAL=1 CMF_TEST_INITIAL_NT=1 CMF_TEST_INITIAL_MATRIX_CACHE=1 ;;
 initial-ne-warm) history_tests=(cmf_time_coupled); export CMF_TEST_CENTER=1 CMF_TEST_INITIAL=1 CMF_TEST_INITIAL_NT=1 CMF_TEST_INITIAL_MATRIX_CACHE=1 ;;
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
history_matrix_flags=()
if [[ "$history_mode" == initial-matrix-cache || "$history_mode" == initial-ne-warm ]]; then
 history_matrix_flags=(-DCMF_MATRIX_CACHE_INTEGRATION -Wl,--wrap=nlte_element_initial_steady_solve_cached)
fi
for history_test in "${history_tests[@]}"; do
 gcc -O2 -std=gnu11 -D_GNU_SOURCE -DCMF_MATERIAL_ROOT_SELFTEST -fopenmp -ffunction-sections -fdata-sections -Isrc \
  "tests/${history_test}_selftest.c" "${history_objects[@]}" "${history_matrix_flags[@]}" -Wl,--gc-sections -lm \
  -o "$history_out/$history_test" >> "$history_out/build.log" 2>&1
 (cd "$history_out"; "./$history_test" > "$history_test.log" 2> "$history_test.err")
 if [[ "$history_test" == cmf_recombination ]]; then
  (cd "$history_out"; LUMINA_NLTE_BF_JEQB=1 "./$history_test" > recombination-thermal.log 2> recombination-thermal.err)
 fi
 if [[ "$history_test" == bf_channels ]]; then
  (cd "$history_out"; mkdir cache; cd cache; CMF_TEST_BF_CACHE_ONLY=1 ../bf_channels > ../cache.log 2> ../cache.err)
 fi
done
if [[ "$history_mode" == initial-ne-warm ]]; then
 # 같은 합성 원자의 완료된 구판 산출물을 재사용한다. 기존 실행을 덮어쓰거나 재실행하지 않는다.
 history_reference=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/history-423992
 (cd "$history_reference/source"; sha256sum -c "$history_reference/source.sha256") > "$history_out/reference_verify.log"
 for history_file in "${history_files[@]}"; do
  case "$history_file" in src/lumina_plasma.c|scripts/run_cmf_history_selftest.slurm) continue ;; esac
  cmp "$history_file" "$history_reference/source/$history_file"
 done
 mkdir "$history_out/reference" "$history_out/sanitized"
 if [[ "$history_warm_ne" == plain ]]; then
  for history_file in initial_spectrum.csv initial_material.csv initial_evolved_spectrum.csv initial_evolved_material.csv cmf_time_coupled.log; do
   cp "$history_reference/$history_file" "$history_out/reference/$history_file"
  done
 else
  cp "$history_reference/cmf_time_coupled" "$history_out/reference-binary"
  printf 'eedcb06139078c4d989c37f5da391151c164cc9a905cb120e2b8fd98083486cb  %s\n' "$history_out/reference-binary" > "$history_out/reference_binary.sha256"
  sha256sum -c "$history_out/reference_binary.sha256" > "$history_out/reference_binary_verify.log"
  (cd "$history_out/reference"; ../reference-binary > cmf_time_coupled.log 2> cmf_time_coupled.err)
  (cd "$history_out/reference"; sha256sum initial_spectrum.csv initial_material.csv initial_evolved_spectrum.csv initial_evolved_material.csv) > "$history_out/reference_output.sha256"
 fi
 cp "$history_reference/source.sha256" "$history_out/reference_source.sha256"
 history_warm_rest=()
 for history_unit in "${history_units[@]}"; do
  if [[ "$history_unit" != lumina_plasma ]]; then history_warm_rest+=("$history_out/objects/$history_unit.o"); fi
 done
 gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections -Isrc \
  -fsanitize=address,undefined -fno-omit-frame-pointer tests/cmf_time_coupled_selftest.c \
  src/lumina_plasma.c "${history_warm_rest[@]}" "${history_matrix_flags[@]}" \
  -Wl,--gc-sections -lm -o "$history_out/warm-sanitized" > "$history_out/warm-san-build.log" 2>&1
 (cd "$history_out/sanitized"; ASAN_OPTIONS=detect_leaks=1 ../warm-sanitized > ../sanitized.log 2> ../sanitized.err)
 module load python/3.11.14
 python3 - "$history_out" "$history_warm_ne" > "$history_out/warm-comparison.log" <<'PY'
import csv, hashlib, math, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
sealed = {
    'initial_material.csv': '6059a3b07110a88fcdb26341b686f641b66488c3e073973c282012f5d74d279b',
    'initial_spectrum.csv': '503efba08574e8d2113d29c0616af1d927ae1d2e5931963dc5a6ee628994c0b6',
    'initial_evolved_material.csv': '8515a14280988d246950707b7e2073329b3fefedb12a533b26ecb348a7a92524',
    'initial_evolved_spectrum.csv': '785913647bedcded637327ba61b36723f434e530727caf4772fc040ca35e415d',
}
for name, digest in sealed.items():
    ref = root / 'reference' / name
    if sys.argv[2] == 'plain':
        assert hashlib.sha256(ref.read_bytes()).hexdigest() == digest
    with ref.open() as f:
        reference = list(csv.DictReader(f))
    field = 'flux' if 'spectrum' in name else 'value'
    for variant in (root, root / 'sanitized'):
        with (variant / name).open() as f:
            result = list(csv.DictReader(f))
        assert reference and len(reference) == len(result)
        worst = 0.0
        for a, b in zip(reference, result):
            assert {k:v for k,v in a.items() if k != field} == {k:v for k,v in b.items() if k != field}
            x, y = float(a[field]), float(b[field])
            assert math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0
            worst = max(worst, abs(x-y)/max(x,y) if max(x,y) else 0)
        assert worst < 1e-8, (name, variant.name, worst)
        print(f'INITIAL_WARM_PARITY file={name} variant={variant.name} max_relative={worst:.17g} amplitude_fit=0')
    assert (root/name).read_bytes() == (root/'sanitized'/name).read_bytes(), name
pattern = r'CMF_INITIAL_NONTHERMAL_PASS provider=ACTUAL_SPENCER_FANO calls=(\d+)'
def calls(path):
    match = re.search(pattern, path.read_text())
    assert match, path
    return int(match.group(1))
reference = calls(root/'reference'/'cmf_time_coupled.log')
native, sanitized = calls(root/'cmf_time_coupled.log'), calls(root/'sanitized.log')
assert native == sanitized and native < reference, (reference, native, sanitized)
print(f'INITIAL_WARM_CALLS reference={reference} native={native} sanitized={sanitized} synthetic_atom=1')
if sys.argv[2] == 'anderson':
    for log in (root/'cmf_time_coupled.err', root/'reference'/'cmf_time_coupled.err', root/'sanitized.err'):
        assert '[A2-07][NE-ANDERSON]' in log.read_text(), log
    print('INITIAL_WARM_NE_ANDERSON actual_acceleration_in_all_three_variants=1')
PY
fi
if [[ "$history_mode" == initial-matrix-cache ]]; then
 history_reference=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/common-build-423957
 (cd "$history_reference/source"; sha256sum -c "$history_reference/source.sha256") > "$history_out/reference_verify.log"
 cp "$history_reference/source/src/lumina_plasma.c" "$history_out/reference_plasma.c"
 cp "$history_reference/source/src/lumina_element_wide.c" "$history_out/reference_element.c"
 cp "$history_reference/commit.txt" "$history_out/reference_commit.txt"
 sha256sum "$history_out/reference_plasma.c" "$history_out/reference_element.c" > "$history_out/reference.sha256"
 history_matrix_rest=()
 for history_unit in "${history_units[@]}"; do
  case "$history_unit" in
   lumina_plasma|lumina_element_wide) ;;
   *) history_matrix_rest+=("$history_out/objects/$history_unit.o") ;;
  esac
 done
 gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections -Isrc \
  tests/cmf_time_coupled_selftest.c "$history_out/reference_plasma.c" "$history_out/reference_element.c" \
  "${history_matrix_rest[@]}" -Wl,--gc-sections -lm -o "$history_out/matrix-reference" \
  > "$history_out/reference-build.log" 2>&1
 gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections -Isrc \
  -fsanitize=address,undefined -fno-omit-frame-pointer tests/cmf_time_coupled_selftest.c \
  src/lumina_plasma.c src/lumina_element_wide.c "${history_matrix_rest[@]}" \
  "${history_matrix_flags[@]}" -Wl,--gc-sections -lm -o "$history_out/matrix-sanitized" \
  > "$history_out/matrix-san-build.log" 2>&1
 mkdir "$history_out/reference" "$history_out/sanitized"
 (cd "$history_out/reference"; ../matrix-reference > ../reference.log 2> ../reference.err)
 (cd "$history_out/sanitized"; ASAN_OPTIONS=detect_leaks=1 ../matrix-sanitized > ../sanitized.log 2> ../sanitized.err)
 for history_file in initial_spectrum.csv initial_material.csv initial_evolved_spectrum.csv initial_evolved_material.csv; do
  cmp "$history_out/$history_file" "$history_out/reference/$history_file"
  cmp "$history_out/$history_file" "$history_out/sanitized/$history_file"
  sha256sum "$history_out/$history_file" >> "$history_out/parity.sha256"
 done
 printf 'INITIAL_MATRIX_PARITY_PASS files=4 native/reference/ASAN byte_exact=1\n' > "$history_out/matrix-parity.log"
fi
if [[ "$history_mode" == initial-ne-compare ]]; then
 history_ne_rest=()
 for history_unit in "${history_units[@]}"; do
  case "$history_unit" in
   lumina_plasma|nlte_population_candidate|cmf_time_epoch) ;;
   *) history_ne_rest+=("$history_out/objects/$history_unit.o") ;;
  esac
 done
 gcc -O1 -g -std=gnu11 -D_GNU_SOURCE -DLUMINA_TIME_PAIR_SELFTEST -fopenmp \
  -fsanitize=address,undefined -fno-omit-frame-pointer -ffunction-sections -fdata-sections -Isrc \
  tests/cmf_time_coupled_selftest.c src/lumina_plasma.c src/nlte_population_candidate.c \
  src/cmf_time_epoch.c "${history_ne_rest[@]}" -Wl,--gc-sections -lm \
  -o "$history_out/ne-sanitized" > "$history_out/ne-san-build.log" 2>&1
 mkdir "$history_out/ne-accelerated"
 (cd "$history_out/ne-accelerated"; ASAN_OPTIONS=detect_leaks=1 LUMINA_NLTE_NE_ANDERSON1=1 ../ne-sanitized \
  > ../ne-accelerated.log 2> ../ne-accelerated.err)
 module load python/3.11.14
 python3 - "$history_out" > "$history_out/ne-comparison.log" <<'PY'
import csv, math, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
for name in ('initial_spectrum.csv', 'initial_evolved_spectrum.csv',
             'initial_material.csv', 'initial_evolved_material.csv'):
    with (root / name).open() as f:
        reference = list(csv.DictReader(f))
    with (root / 'ne-accelerated' / name).open() as f:
        accelerated = list(csv.DictReader(f))
    assert reference and len(reference) == len(accelerated)
    field = 'flux' if 'spectrum' in name else 'value'
    worst = 0.0
    for a, b in zip(reference, accelerated):
        assert {k:v for k,v in a.items() if k != field} == {k:v for k,v in b.items() if k != field}
        x, y = float(a[field]), float(b[field])
        assert math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0
        worst = max(worst, abs(x-y)/max(x,y) if max(x,y) else 0)
    assert worst < 1e-6, (name, worst)
    print(f'NE_ACCELERATION_OUTPUT_PASS file={name} max_relative={worst:.17g} amplitude_fit=0 synthetic_atom=1')
plain = (root / 'cmf_time_coupled.err').read_text()
fast = (root / 'ne-accelerated.err').read_text()
pattern = r'\[A2-07\]\[NE-CLOSURE\] commit=R7[^\n]* iterations=(\d+)'
p, a = [list(map(int, re.findall(pattern, log))) for log in (plain, fast)]
assert p and a and sum(a) < sum(p), (p, a)
assert '[A2-07][NE-ANDERSON]' not in plain
assert '[A2-07][NE-ANDERSON]' in fast
print(f'NE_ACCELERATION_ITERATIONS_PASS baseline_calls={len(p)} accelerated_calls={len(a)} baseline_solves={sum(p)} accelerated_solves={sum(a)}')
PY
fi
if [[ "$history_mode" == initial-restart ]]; then
 mkdir "$history_out/restarted"
 (cd "$history_out/restarted"; CMF_TEST_ITERATE_INPUT="$history_out/initial_iterate_test.bin" \
  ../cmf_time_coupled > ../restarted.log 2> ../restarted.err)
 # 초기/시간 관측 CSV의 절대값을 비교한다. 진폭 재조정은 없다.
 module load python/3.11.14
 python3 - "$history_out" > "$history_out/restart-comparison.log" <<'PY'
import csv, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name in ('initial_spectrum.csv', 'initial_evolved_spectrum.csv'):
    with (root / name).open() as f:
        reference = list(csv.DictReader(f))
    with (root / 'restarted' / name).open() as f:
        resumed = list(csv.DictReader(f))
    assert reference and len(reference) == len(resumed)
    worst = 0.0
    for a, b in zip(reference, resumed):
        assert a['wavelength_angstrom'] == b['wavelength_angstrom']
        x, y = float(a['flux']), float(b['flux'])
        assert math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0
        error = abs(x-y)/max(x,y) if max(x,y) else 0
        worst = max(worst, error)
    assert worst < 1e-6, (name, worst)
    print(f'INITIAL_RESTART_SPECTRUM_PASS file={name} max_relative={worst:.17g} amplitude_fit=0 synthetic_atom=1')
PY
fi
if [[ "$history_mode" == initial || "$history_mode" == initial-nonthermal ]]; then
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
(cd "$history_out"; mkdir cache-san; cd cache-san; ASAN_OPTIONS=detect_leaks=1 CMF_TEST_BF_CACHE_ONLY=1 ../bf_channels_san > ../cache-san.log 2> ../cache-san.err)
sha256sum -c "$history_out/source.sha256" > "$history_out/source_verify.log"
(cd "$history_out"; sha256sum -c input.sha256 > input_verify.log)
