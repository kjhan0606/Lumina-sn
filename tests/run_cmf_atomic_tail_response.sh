#!/usr/bin/env bash
set -euo pipefail
umask 027
# Q-GOAL=PASS: 현행 고준위 전이망이 Si/S의 큰 차이를 만드는지 실제 율계로 분리.
# Q-LEAN=APPROPRIATE: 기존 reader/행렬 포착/원자료를 재사용. 새 RTE/GPU/게이트 없음.
# CPU2/12GiB/10분 상한. 고정J 조건부 원인 시험이며 수렴/공개 모형 재현 판정 아님.
test "$(hostname -s)" = LagEunha
ulimit -v 12582912
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
at_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
at_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
at_prior=$at_base/toy-material-425144
at_orig=$at_base/ada-evolution-20261016-1036/initial
at_build=$at_base/common-build-424525
at_root=$(mktemp -d "$at_base/atomic-tail-response-20261017.XXXXXX")
printf 'ATOMIC_TAIL_ROOT=%s\n' "$at_root"
cd "$at_root"
cp "$at_src/tests/cmf_toy_nt_line_response.c" "$at_src/tests/cmf_toy_line_response.c" \
 "$at_src/tests/cmf_toy_superlevel_response.c" "$at_src/tests/cmf_atomic_tail_response.py" \
 "$at_src/tests/run_cmf_atomic_tail_response.sh" .
sha256sum ./*.c ./*.py ./*.sh > source.sha256
sha256sum "$at_prior/objects/"*.o > objects.sha256
sha256sum "$at_orig/work/initial_gamma.csv" "$at_orig/input/arnaud_rothenflug_23jun15" \
 "$at_orig/input/nonthermal_resolved_bindings.csv" "$at_prior/input/levels.csv" > inputs.sha256
sha256sum -c "$at_prior/iterate.sha256" > before_iterate_verify.log
(cd "$at_orig/input";sha256sum -c "$at_orig/input.sha256") > original_input_verify.log
cmp "$at_orig/input/line_list.csv" "$at_prior/input/line_list.csv"
cmp "$at_orig/input/levels.csv" "$at_prior/input/levels.csv"
at_units=()
for at_unit in cmf_nonthermal cmf_nonthermal_atomic cmf_nonthermal_provider cmf_nonthermal_excitation cmf_nonthermal_binding;do
 at_units+=("$at_build/source/src/$at_unit.c")
done
grep -E ' src/cmf_nonthermal[^/]*\.[ch]$' "$at_build/source.sha256" > nonthermal_expected.sha256
(cd "$at_build/source";sha256sum -c "$at_root/nonthermal_expected.sha256") > nonthermal_verify.log
date -Is > started.txt
gcc -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_RESPONSE_FULL_ONLY -fopenmp \
 -I"$at_prior/source/src" cmf_toy_nt_line_response.c "${at_units[@]}" "$at_prior/objects/"*.o \
 -Wl,--wrap=population_generator_stationary_gth -Wl,--gc-sections -lm -o response > build.log 2>&1
while read -r at_key;do unset "$at_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r at_entry;do export "$at_entry";done < "$at_prior/env_lumina_effective.txt"
export LUMINA_DIAG_GAMMA_CSV="$at_orig/work/initial_gamma.csv"
export LUMINA_DIAG_NT_ATOMIC="$at_orig/input/arnaud_rothenflug_23jun15"
export LUMINA_DIAG_NT_BINDINGS="$at_orig/input/nonthermal_resolved_bindings.csv"
at_id=$(sha256sum "$at_prior/iterate_identity.txt");at_id=${at_id%% *}
env | sort | grep '^LUMINA_' > effective.env
./response "$at_prior/input/snia_toy06_2d.dat" "$at_prior/input" "$at_prior/iterate_input.bin" "$at_id" 39 > response.log 2> response.err
python3 -B cmf_atomic_tail_response.py > analysis.log 2> analysis.err
sha256sum -c source.sha256 > source_verify.log
sha256sum -c objects.sha256 > objects_verify.log
sha256sum -c inputs.sha256 > inputs_verify.log
sha256sum -c "$at_prior/iterate.sha256" > after_iterate_verify.log
sha256sum response ./*.bin ./*.csv tail_summary.json response.log response.err analysis.log analysis.err > output.sha256
date -Is > finished.txt
cat tail_summary.json
