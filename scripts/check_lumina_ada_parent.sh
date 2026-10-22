#!/bin/bash
# 최대60초 소형 장치 검증의 부모다. 수송 전범위 실행은 포함하지 않는다.
set -euo pipefail
cd "$1"
test "$(hostname)" = LagEunha
test ! -e manual-parent.started
exec 9>/tmp/lumina_hybrid_ada_GPU-42b3367f.lock
flock -n 9
set -C
date +%s >manual-parent.started
set +C
trap 'date +%s >manual-parent.ended' EXIT
set +e
timeout -k 3 50 taskset -c 25 bash scripts/check_lumina_ada_device.sh "$PWD"
code=$?
printf '%s\n' "$code" >verification.exit
exit "$code"
