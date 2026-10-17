#!/usr/bin/env bash
set -euo pipefail
umask 027
# 전체 전이망 수정본의 실제 계수 조립만 확인. CPU4/24GiB/10분/GPU0.
# 수송 직전 의도적 중단, 기존 저장점 및 입력 불변. 새 평형해가 아니다.
test "$(hostname -s)" = LagEunha
[[ "$#" = 1 || "$#" = 2 ]]
fm_mode=${2:-native}
case "$fm_mode" in native|bootstrap) ;; *) exit 2;; esac
fm_build=$(realpath "$1")
case "$fm_build" in /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/full-domain-20261017.*) ;; *) exit 2;; esac
test -f "$fm_build/finished.txt"
ulimit -v 25165824
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fm_prior=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/toy-material-425144
fm_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
fm_root=$(mktemp -d /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results/full-domain-material-20261017.XXXXXX)
printf 'FULL_DOMAIN_MATERIAL_ROOT=%s\n' "$fm_root"
cd "$fm_root"
cp "$fm_src/tests/cmf_toy_fine_material.c" "$fm_src/tests/run_cmf_full_domain_material.sh" .
sha256sum ./*.c ./*.sh > source.sha256
(cd "$fm_build"; sha256sum -c source.sha256; sha256sum -c output.sha256) > build_verify.log
fm_objects=()
for fm_object in "$fm_prior/objects/"*.o;do
 case "${fm_object##*/}" in line_jbar.o|lumina_plasma.o|lumina_cmfgen.o) ;;
 opacity_publication.o) if [[ ! -f "$fm_build/objects/opacity_publication.o" ]];then fm_objects+=("$fm_object");fi ;;
 *) fm_objects+=("$fm_object");; esac
done
sha256sum "${fm_objects[@]}" "$fm_build/objects/"*.o > objects.sha256
gcc -O2 -std=gnu11 -D_GNU_SOURCE -DLUMINA_FULL_DOMAIN_MATERIAL -fopenmp -I"$fm_build/source" cmf_toy_fine_material.c \
 "${fm_objects[@]}" "$fm_build/objects/"*.o -Wl,--wrap=cmf_exact_emissivity_solve -Wl,--gc-sections -lm -o capture > build.log 2>&1
while read -r fm_key;do unset "$fm_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r fm_entry;do export "$fm_entry";done < "$fm_prior/env_lumina_effective.txt"
export LUMINA_CMF_FINE_MGPU_DEVICES=0
export LUMINA_CMF_FINE_VDOP=5000000 LUMINA_CMF_FINE_PPD=3 LUMINA_CMF_FINE_PHOTOION=1
export LUMINA_CMF_FINE_ALI=768 LUMINA_CMF_FINE_TOL=1e-8 LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=42
if [[ "$fm_mode" == bootstrap ]];then export LUMINA_DIAG_BOOTSTRAP_LTE=1;fi
env | sort | grep '^LUMINA_' > effective.env
fm_id=$(sha256sum "$fm_prior/iterate_identity.txt");fm_id=${fm_id%% *}
date -Is > started.txt
./capture "$fm_prior/input/snia_toy06_2d.dat" "$fm_prior/input" "$fm_prior/iterate_input.bin" "$fm_id" > capture.log 2> capture.err
sha256sum -c source.sha256 > after_source_verify.log
sha256sum -c objects.sha256 > after_objects_verify.log
sha256sum -c "$fm_prior/iterate.sha256" > after_iterate_verify.log
sha256sum capture ./*.bin ./*.csv capture.log capture.err > output.sha256
date -Is > finished.txt
grep -E '^(FULL_DOMAIN_MATERIAL|FINE_MATERIAL_)' capture.log
