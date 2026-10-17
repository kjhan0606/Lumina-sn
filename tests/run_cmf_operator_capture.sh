#!/usr/bin/env bash
set -euo pipefail
umask 027
test "$(hostname -s)" = LagEunha
ulimit -v 25165824
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
op_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
op_src=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
op_prior=$op_base/toy-material-425792
op_prod=$op_base/ada-evolution-20261017-1034/initial
op_saved=$op_base/material-guess-read-425876
op_build=$op_base/common-build-425846
op_root=$(mktemp -d "$op_base/operator-capture-20261017.XXXXXX")
printf 'OPERATOR_CAPTURE_ROOT=%s\n' "$op_root"
cd "$op_root"
cp "$op_src/tests/cmf_toy_operator_capture.c" "$op_src/tests/run_cmf_operator_capture.sh" .
sha256sum ./*.c ./*.sh > source.sha256
(cd "$op_prior/source";sha256sum -c "$op_prior/source.sha256") > reader_source_verify.log
(cd "$op_prod/input";sha256sum -c "$op_prod/input.sha256") > input_verify.log
for op_unit in "$op_prior/objects/"*.o; do
 op_name=${op_unit##*/};op_name=${op_name%.o}
 # 시간 운전의 중간 저장 기능만 변경됐다. 계수 조립/판독 객체는 동일해야 한다.
 if [[ "$op_name" != cmf_time_epoch ]];then
  cmp "$op_prior/source/src/$op_name.c" "$op_build/source/src/$op_name.c"
 fi
done
sha256sum "$op_prior/objects/"*.o > objects.sha256
printf '%s  %s\n' 72e547ba0422983e8bd9ffe730c0bdc71d4cec502083dfaaecf180d3cf76496f "$op_saved/iterate_input.bin" | sha256sum -c - > checkpoint_verify.log
cmp "$op_saved/iterate_identity.txt" "$op_prod/iterate_identity.txt"
gcc -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -fopenmp -I"$op_prior/source/src" \
 cmf_toy_operator_capture.c "$op_prior/objects/"*.o \
 -Wl,--wrap=cmf_exact_emissivity_solve -Wl,--gc-sections -lm -o capture > build.log 2> build.err
while read -r op_key;do unset "$op_key";done < <(compgen -v LUMINA_ || true)
while IFS= read -r op_entry;do export "$op_entry";done < "$op_prod/env_lumina_effective.txt"
export LUMINA_CMF_FINE_MGPU_DEVICES=0
env | LC_ALL=C sort | grep '^LUMINA_' > effective.env
op_identity=$(sha256sum "$op_prod/iterate_identity.txt");op_identity=${op_identity%% *}
date -Is > started.txt
./capture "$op_prod/input/snia_toy06_2d.dat" "$op_prod/input" "$op_saved/iterate_input.bin" "$op_identity" > capture.log 2> capture.err
sha256sum -c source.sha256 > source_verify.log
sha256sum -c objects.sha256 > objects_verify.log
sha256sum capture operator.bin capture.log capture.err > output.sha256
date -Is > finished.txt
cat capture.log
