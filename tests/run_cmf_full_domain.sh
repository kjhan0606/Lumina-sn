#!/usr/bin/env bash
set -euo pipefail
umask 027
# Q-GOAL=PASS: 모형 BB의 순·역과정이 실제 소비 경로에서 같은 장을 쓰는지 확인.
# Q-LEAN=APPROPRIATE: 기존 물질/객체와 한 셸 회귀시험. 새 GPU/RTE/외부 감사 없음.
# lageunha 전용, CPU2/16GiB/15분 상한. 기존 결과에 재실행하지 않는다.
test "$(hostname -s)" = LagEunha
ulimit -v 16777216
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fd_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
fd_prior=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-material-425144
fd_root=$(mktemp -d /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/full-domain-20261017.XXXXXX)
printf 'FULL_DOMAIN_ROOT=%s\n' "$fd_root"
cd "$fd_root"
mkdir source objects
cp "$fd_src"/src/*.h source/
cp "$fd_src/src/line_jbar.c" "$fd_src/src/lumina_plasma.c" "$fd_src/src/lumina_cmfgen.c" "$fd_src/src/opacity_publication.c" source/
cp "$fd_src/tests/a2_06_line_jbar_selftest.c" "$fd_src/tests/cmf_toy_planck_pipeline.c" "$fd_src/tests/cmf_toy_line_response.c" "$fd_src/tests/cmf_toy_superlevel_response.c" .
cp "$fd_src/tests/run_cmf_full_domain.sh" .
sha256sum source/* ./*.c run_cmf_full_domain.sh > source.sha256
sha256sum "$fd_prior/objects/"*.o > prior_objects.sha256
sha256sum -c "$fd_prior/iterate.sha256" > before_iterate_verify.log
date -Is > started.txt
gcc -O2 -std=gnu11 -Isource a2_06_line_jbar_selftest.c source/line_jbar.c -lm -o unit > unit_build.log 2>&1
./unit > unit.log 2>&1
cat unit.log
for fd_unit in line_jbar lumina_plasma lumina_cmfgen opacity_publication;do
 gcc -O2 -std=gnu11 -D_GNU_SOURCE -fopenmp -ffunction-sections -fdata-sections -Isource -c "source/$fd_unit.c" -o "objects/$fd_unit.o" >> build.log 2>&1
done
fd_objects=()
for fd_object in "$fd_prior/objects/"*.o;do
 case "${fd_object##*/}" in line_jbar.o|lumina_plasma.o|lumina_cmfgen.o|opacity_publication.o) ;; *) fd_objects+=("$fd_object");; esac
done
gcc -O2 -std=gnu11 -D_GNU_SOURCE -DLUMINA_RESPONSE_PLANCK_FULL_DOMAIN -fopenmp -Isource cmf_toy_planck_pipeline.c \
 objects/*.o "${fd_objects[@]}" -Wl,--wrap=population_generator_stationary_gth -Wl,--gc-sections -lm -o response >> build.log 2>&1
while read -r fd_key;do unset "$fd_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r fd_entry;do export "$fd_entry";done < "$fd_prior/env_lumina_effective.txt"
fd_id=$(sha256sum "$fd_prior/iterate_identity.txt");fd_id=${fd_id%% *}
env | sort | grep '^LUMINA_' > effective.env
./response "$fd_prior/input/snia_toy06_2d.dat" "$fd_prior/input" "$fd_prior/iterate_input.bin" "$fd_id" 39 0 1e9 1 > response.log 2> response.err
sha256sum -c source.sha256 > after_source_verify.log
sha256sum -c prior_objects.sha256 > after_objects_verify.log
sha256sum -c "$fd_prior/iterate.sha256" > after_iterate_verify.log
sha256sum unit response objects/*.o ./*.bin ./*.csv unit.log response.log response.err > output.sha256
date -Is > finished.txt
grep -E '^(PLANCK_|SL_RESPONSE_RESULT|SL_RESPONSE_DONE)' response.log
