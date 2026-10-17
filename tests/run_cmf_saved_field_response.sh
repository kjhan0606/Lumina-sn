#!/usr/bin/env bash
set -euo pipefail
umask 027
# Q-GOAL=PASS: 새 저장장에서 Si/S가 받는 조건부 물질 반응을 실제 율계로 확인.
# Q-LEAN=APPROPRIATE: 기존 C reader/SE/비열 공급자 재사용. 새 RTE/GPU 없음.
# CPU2/12GiB, 호출자는 timeout 5m 사용. 고정J/T/ne이며 공동수렴 판정 아님.
test "$(hostname -s)" = LagEunha
ulimit -v 12582912
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sf_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
sf_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
sf_prior=${1:?verified reader root}
sf_prod=${2:?production root with same identity}
sf_iteration=${3:?expected completed iteration}
sf_shell=${4:-39}
sf_target=${5:-0}
sf_snapshot=${6:-}
[[ $# -le 6 && "$sf_target" =~ ^(0|14|16|20)$ ]]
sf_target_flags=()
if ((sf_target)); then sf_target_flags+=("-DLUMINA_RESPONSE_TARGET_Z=$sf_target"); fi
[[ "$sf_prior" =~ ^$sf_base/toy-material-[0-9]+$ ]]
# Slurm와 수동 Ada 모두 같은 초기해 러너가 작성한 initial 루트만 허용한다.
# 실행 위치만 확장하며 아래 저장점 신원/원자 입력/봉인 빌드 검사는 유지한다.
[[ "$sf_prod" =~ ^$sf_base/toy-initial-[0-9]+$ ||
   "$sf_prod" =~ ^$sf_base/ada-evolution-[0-9]{8}-[0-9]{4}/initial$ ]]
[[ "$sf_iteration" =~ ^[0-9]+$ && "$sf_shell" =~ ^[0-9]+$ ]]
((sf_iteration<=128 && sf_shell<113))
cmp "$sf_prior/iterate_identity.txt" "$sf_prod/iterate_identity.txt"
sf_checkpoint=$sf_prior/iterate_input.bin
if [[ -n "$sf_snapshot" ]]; then
 [[ "$sf_snapshot" == "$sf_base"/baseline-*/work/initial_iterate.bin && -f "$sf_snapshot" ]]
 cmp "$sf_prior/iterate_identity.txt" "${sf_snapshot%/work/initial_iterate.bin}/iterate_identity.txt"
 sf_checkpoint=$sf_snapshot
fi
sf_build=$(sed -n 's/^build=//p' "$sf_prod/identity.txt")
[[ "$sf_build" =~ ^$sf_base/common-build-[0-9]+$ ]]
sf_root=$(mktemp -d "$sf_base/saved-field-response-20261017.XXXXXX")
printf 'SAVED_FIELD_RESPONSE_ROOT=%s\n' "$sf_root"
cd "$sf_root"
cp "$sf_src/tests/cmf_toy_nt_line_response.c" "$sf_src/tests/cmf_toy_line_response.c" \
 "$sf_src/tests/cmf_toy_superlevel_response.c" "$sf_src/tests/run_cmf_saved_field_response.sh" .
printf 'reader=%s\nproduction=%s\nexpected_iteration=%s\nshell=%s\ntarget_Z=%s\ncheckpoint=%s\n' \
 "$sf_prior" "$sf_prod" "$sf_iteration" "$sf_shell" "$sf_target" "$sf_checkpoint" > inputs.txt
sha256sum ./*.c ./*.sh > source.sha256
sha256sum "$sf_prior/objects/"*.o > objects.sha256
sha256sum -c "$sf_prior/iterate.sha256" > before_iterate_verify.log
sha256sum "$sf_checkpoint" > selected_iterate.sha256
(cd "$sf_prior/input"; sha256sum -c "$sf_prior/input.sha256") > input_verify.log
for sf_file in levels.csv line_list.csv bf_channels.bin bf_channels.terminal.csv; do
 cmp "$sf_prior/input/$sf_file" "$sf_prod/input/$sf_file"
done
sf_units=()
for sf_unit in cmf_nonthermal cmf_nonthermal_atomic cmf_nonthermal_provider cmf_nonthermal_excitation cmf_nonthermal_binding; do
 sf_units+=("$sf_build/source/src/$sf_unit.c")
done
grep -E ' src/cmf_nonthermal[^/]*\.[ch]$' "$sf_build/source.sha256" > nonthermal_expected.sha256
(cd "$sf_build/source"; sha256sum -c "$sf_root/nonthermal_expected.sha256") > nonthermal_verify.log
sha256sum "$sf_prod/work/initial_gamma.csv" "$sf_prod/input/arnaud_rothenflug_23jun15" \
 "$sf_prod/input/nonthermal_resolved_bindings.csv" > nonthermal_input.sha256
date -Is > started.txt
gcc -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_RESPONSE_GROUPED_ONLY \
 "${sf_target_flags[@]}" \
 "-DLUMINA_RESPONSE_EXPECTED_ITERATION=$sf_iteration" -fopenmp -I"$sf_prior/source/src" \
 cmf_toy_nt_line_response.c "${sf_units[@]}" "$sf_prior/objects/"*.o \
 -Wl,--wrap=population_generator_stationary_gth -Wl,--gc-sections -lm -o response > build.log 2>&1
while read -r sf_key; do unset "$sf_key"; done < <(compgen -v LUMINA_ || true)
while IFS= read -r sf_entry; do export "$sf_entry"; done < "$sf_prior/env_lumina_effective.txt"
export LUMINA_DIAG_GAMMA_CSV="$sf_prod/work/initial_gamma.csv"
export LUMINA_DIAG_NT_ATOMIC="$sf_prod/input/arnaud_rothenflug_23jun15"
export LUMINA_DIAG_NT_BINDINGS="$sf_prod/input/nonthermal_resolved_bindings.csv"
sf_id=$(sha256sum "$sf_prior/iterate_identity.txt"); sf_id=${sf_id%% *}
env | sort | grep '^LUMINA_' > effective.env
./response "$sf_prior/input/snia_toy06_2d.dat" "$sf_prior/input" "$sf_checkpoint" \
 "$sf_id" "$sf_shell" > response.log 2> response.err
sha256sum -c source.sha256 > source_verify.log
sha256sum -c objects.sha256 > objects_verify.log
sha256sum -c nonthermal_input.sha256 > nonthermal_input_verify.log
sha256sum -c "$sf_prior/iterate.sha256" > after_iterate_verify.log
sha256sum -c selected_iterate.sha256 > selected_iterate_verify.log
sha256sum response ./*.bin ./*.csv response.log response.err > output.sha256
date -Is > finished.txt
grep -E 'SL_RESPONSE_INPUT|NT_RESPONSE_SOURCE|SL_RESPONSE_RESULT|SL_RESPONSE_DONE' response.log
