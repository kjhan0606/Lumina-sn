#!/usr/bin/env bash
set -euo pipefail
umask 027
# 원본 PRRR와 실제 BF 소비의 목적 준위 보정을 비교한다. 새 물리 게이트가 아니다.
test "$(hostname -s)" = LagEunha
ulimit -v 2097152
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
prrr_repo=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015
prrr_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
prrr_root=$(mktemp -d "$prrr_base/prrr-comparison-20261017.XXXXXX")
printf 'PRRR_ROOT=%s\n' "$prrr_root"
cd "$prrr_root"
cp "$prrr_repo/tests/cmf_prrr_bridge.f90" "$prrr_repo/tests/cmf_prrr_c_bridge.c" \
   "$prrr_repo/tests/cmf_prrr_comparison.py" "$prrr_repo/tests/run_cmf_prrr_comparison.sh" .
cp "$prrr_base/common-build-425635/source/src/bf_channels.h" .
cp /gpfs/kjhan/cmfgen_src/cur_cmf/subs/prrr_sl_v6.f .
cmp bf_channels.h "$prrr_repo/src/bf_channels.h"
sha256sum cmf_prrr_bridge.f90 cmf_prrr_c_bridge.c cmf_prrr_comparison.py \
 run_cmf_prrr_comparison.sh bf_channels.h prrr_sl_v6.f > source.sha256
sha256sum "$prrr_base/toy-material-425642/iterate_input.bin" \
 "$prrr_base/toy-material-425642/input/bf_channels.bin" \
 "$prrr_base/toy-material-425642/input/levels.csv" \
 "$prrr_base/saved-field-response-20261017.vNeZQn/matrix_Z14.bin" \
 "$prrr_base/saved-field-response-20261017.vNeZQn/matrix_Z16.bin" \
 "$prrr_base/saved-field-response-20261017.vNeZQn/levels_Z14.csv" \
 "$prrr_base/saved-field-response-20261017.vNeZQn/levels_Z16.csv" > inputs.sha256
date -Is > started.txt
gcc -O2 -fPIC -I. -c cmf_prrr_c_bridge.c -o bridge.o
gfortran -O2 -fPIC -fcheck=all -c cmf_prrr_bridge.f90 -o fortran_bridge.o
gfortran -O2 -fPIC -fcheck=all -ffixed-line-length-none -c prrr_sl_v6.f -o original.o
gfortran -shared bridge.o fortran_bridge.o original.o -o comparison.so
sha256sum comparison.so > binary.sha256
timeout 120s python3 -B cmf_prrr_comparison.py > analysis.log 2> analysis.err
sha256sum -c source.sha256 > source_verify.log
sha256sum -c inputs.sha256 > input_verify.log
sha256sum summary.json analysis.log analysis.err > output.sha256
date -Is > finished.txt
cat summary.json
