#!/usr/bin/env bash
set -euo pipefail
umask 027
# 실제 실패 계수에 봉인 상계 연산자만 적용한다. 새 물리장/생산 계산 없음.
test "$(hostname -s)" = LagEunha
op_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
op_build=${1:-$op_base/operator-build-425943}
[[ $# -le 1 && "$op_build" =~ ^$op_base/(operator|weighted)-build-[0-9]+$ ]]
op_capture=$op_base/operator-capture-20261017.CXKfQS
op_prod=$op_base/common-build-425846
test -f "$op_build/finished.txt"
test -f "$op_capture/finished.txt"
exec 9> "$op_base/.cmf-toy-ada.lock"
flock -n 9 || exit 75
op_active=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
test -z "$op_active" || exit 75
op_root=$(mktemp -d "$op_base/operator-probe-20261017.XXXXXX")
printf 'OPERATOR_PROBE_ROOT=%s\n' "$op_root"
cd "$op_root"
cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015/tests/run_cmf_operator_probe_ada.sh .
(cd "$op_build";sha256sum -c binary.sha256
 if [[ -d source ]];then cd source;fi
 sha256sum -c "$op_build/source.sha256") > build_verify.log
(cd "$op_capture";sha256sum -c output.sha256) > capture_verify.log
(cd "$op_prod/source";sha256sum -c "$op_prod/source.sha256") > production_source_verify.log
printf 'build=%s\ncapture=%s\nproduction_source=%s\n' "$op_build" "$op_capture" "$op_prod" > inputs.txt
ulimit -v 134217728
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
while read -r op_key;do unset "$op_key";done < <(compgen -v LUMINA_ || true)
date -Is > started.txt
op_start=$(date +%s)
set +e
timeout -k 5s 600s bash -c '
 set -e
 if [[ "$1" == */weighted-build-* ]];then
  CMF_MGPU_TEST_DEVICES=1 "$1/cmf_exact_multigpu_selftest" > gpu_selftest.log 2> gpu_selftest.err
  "$1/cmf_emissivity_selftest" > emissivity_selftest.log 2> emissivity_selftest.err
 fi
 "$1/probe" "$2/operator.bin" > probe.log 2> probe.err
' bash "$op_build" "$op_capture"
op_rc=$?
set -e
op_end=$(date +%s)
printf 'exit=%s elapsed_seconds=%s gpu_count=1\n' "$op_rc" "$((op_end-op_start))" > RUN_FOOTER.txt
date -Is > finished.txt
(cd "$op_build";sha256sum -c binary.sha256) > binary_after_verify.log
sha256sum ./*.log ./*.err RUN_FOOTER.txt inputs.txt run_cmf_operator_probe_ada.sh > output.sha256
cat RUN_FOOTER.txt
if [[ -f probe.log ]];then cat probe.log;fi
exit "$op_rc"
