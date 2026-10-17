#!/usr/bin/env bash
set -euo pipefail
umask 027
# 기존 Ada 계산과 GPU 시험을 겹치지 않는다. 기다리는 루프/자동 재시작 없음.
# CPU2/GPU1/전체3분(+kill5초), 새 생산 계산/입력 변경 없음.
test "$(hostname -s)" = LagEunha
eg_base=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results
eg_build=$eg_base/envelope-gpu-build-425576
test -x "$eg_build/selftest"
exec 9> "$eg_base/.cmf-toy-ada.lock"
flock -n 9 || { echo 'GPU_TEST_NOT_STARTED: existing Ada owner holds lock' >&2; exit 75; }
eg_active=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
test -z "$eg_active" || { echo 'GPU_TEST_NOT_STARTED: GPU0 compute process exists' >&2; exit 75; }
sha256sum -c "$eg_build/binary.sha256"
(cd "$eg_build/source";sha256sum -c "$eg_build/source.sha256")
eg_out=$(mktemp -d "$eg_base/envelope-gpu-test-20261017.XXXXXX")
printf 'GPU_ENVELOPE_TEST_ROOT=%s\n' "$eg_out"
cd "$eg_out"
cp "$eg_build/binary.sha256" "$eg_build/source.sha256" .
cp /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn-timedep-20261015/tests/run_cmf_envelope_gpu_ada.sh .
ulimit -v 134217728
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0 CMF_MGPU_TEST_DEVICES=1
while read -r eg_key;do unset "$eg_key";done < <(compgen -v LUMINA_ || true)
date -Is > started.txt
eg_start=$(date +%s)
set +e
timeout -k 5s 3m bash -c '
 set -e
 env -u LUMINA_CMF_ANDERSON1 "$1/selftest" > plain.log 2> plain.err
 env LUMINA_CMF_ANDERSON1=1 "$1/selftest" > anderson.log 2> anderson.err
' bash "$eg_build"
eg_rc=$?
set -e
eg_end=$(date +%s)
printf 'exit=%s elapsed_seconds=%s gpu_count=1\n' "$eg_rc" "$((eg_end-eg_start))" > RUN_FOOTER.txt
date -Is > finished.txt
sha256sum -c "$eg_build/binary.sha256" > binary_verify.log
sha256sum ./*.log ./*.err RUN_FOOTER.txt run_cmf_envelope_gpu_ada.sh > output.sha256
cat RUN_FOOTER.txt
if [[ "$eg_rc" == 0 ]];then
 grep -E 'GPU_ENVELOPE_REFINEMENT_INVARIANCE|SELFTEST PASS' plain.log anderson.log
fi
exit "$eg_rc"
