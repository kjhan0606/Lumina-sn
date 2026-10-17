#!/usr/bin/env bash
set -euo pipefail
umask 027
# Q-GOAL=PASS: 전리 기여 대역의 실제 소광/방출 원소를 분리한다.
# Q-LEAN=APPROPRIATE: 기존 CPU reader 객체/수송 직전 중단 시험을 재사용한다.
# lageunha CPU4/24GiB/10분/GPU0. 현재 첫 장의 계수이지 새 공동해가 아니다.
test "$(hostname -s)" = LagEunha
ulimit -v 25165824
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
origin_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
origin_prior=$origin_base/toy-material-425642
origin_repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
origin_root=$(mktemp -d "$origin_base/continuum-origin-20261017.XXXXXX")
printf 'CONTINUUM_ORIGIN_ROOT=%s\n' "$origin_root"
cd "$origin_root"
cp "$origin_repo/tests/cmf_toy_fine_material.c" "$origin_repo/tests/cmf_continuum_origin.h" \
   "$origin_repo/tests/run_cmf_continuum_origin.sh" .
sha256sum ./*.c ./*.h ./*.sh > source.sha256
sha256sum "$origin_prior/objects/"*.o > objects.sha256
sha256sum -c "$origin_prior/iterate.sha256" > before_iterate_verify.log
(cd "$origin_prior/input"; sha256sum -c "$origin_prior/input.sha256") > input_verify.log
gcc -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FINE_ORIGIN \
 -DLUMINA_FINE_EXPECTED_ITERATION=0 -fopenmp -I"$origin_prior/source/src" \
 cmf_toy_fine_material.c "$origin_prior/objects/"*.o \
 -Wl,--wrap=cmf_exact_emissivity_solve -Wl,--gc-sections -lm -o origin > build.log 2>&1
while read -r origin_key; do unset "$origin_key"; done < <(compgen -v LUMINA_ || true)
while IFS= read -r origin_entry; do export "$origin_entry"; done < "$origin_prior/env_lumina_effective.txt"
export LUMINA_CMF_FINE_MGPU_DEVICES=0
export LUMINA_CMF_FINE_VDOP=5000000 LUMINA_CMF_FINE_PPD=3 LUMINA_CMF_FINE_PHOTOION=1
export LUMINA_CMF_FINE_ALI=768 LUMINA_CMF_FINE_TOL=1e-8 LUMINA_CMF_FINE_ENVELOPE_REFINEMENTS=42
env | sort | grep '^LUMINA_' > effective.env
origin_id=$(sha256sum "$origin_prior/iterate_identity.txt"); origin_id=${origin_id%% *}
date -Is > started.txt
timeout 8m ./origin "$origin_prior/input/snia_toy06_2d.dat" "$origin_prior/input" \
 "$origin_prior/iterate_input.bin" "$origin_id" > origin.log 2> origin.err
sha256sum -c source.sha256 > source_verify.log
sha256sum -c objects.sha256 > objects_verify.log
sha256sum -c "$origin_prior/iterate.sha256" > after_iterate_verify.log
sha256sum origin ./*.bin ./*.csv origin.log origin.err > output.sha256
date -Is > finished.txt
grep -E '^ORIGIN_|^FINE_MATERIAL_' origin.log
