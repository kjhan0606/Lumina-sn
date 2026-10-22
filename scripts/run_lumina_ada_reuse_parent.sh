#!/bin/bash
set -euo pipefail
cd "$1"
test "$(hostname)" = LagEunha
test ! -e parent.started
exec 9>/tmp/lumina_hybrid_ada_GPU-42b3367f.lock
flock -n 9
set -C
date +%s >parent.started
printf '%s\n' "$$" >parent.pid
set +C
trap 'date +%s >parent.ended' EXIT
set +e
timeout -k 10 3580 taskset -c 25 bash actual.sh "$PWD" "$2" >manual-run.log 2>&1
code=$?
printf '%s\n' "$code" >parent.exit
exit "$code"
