#!/bin/bash
# 실제 추가 계산의 입력·횟수 승인을 받은 뒤 호출한다. 이 파일은 자동 제출하지 않는다.
# 인수는 기존 Slurm 실행기와 같다: 새 parent, 봉인 commit, 입력/해시 옵션.
set -euo pipefail
umask 027
[[ "${HOSTNAME%%.*}" = LagEunha || "${HOSTNAME%%.*}" = lageunha ]]
test -z "${SLURM_JOB_ID:-}"
test "$#" -ge 2
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
test "$(git -C "$repository" rev-parse --show-toplevel)" = "$repository"
test -f "$repository/scripts/lumina_new_field_response.py"
# 이 프로세스에 허용된 CPU 중16개에만 실제 affinity를 고정한다.
cpus=$(python3 -c 'import os; c=sorted(os.sched_getaffinity(0)); assert len(c)>=16; print(",".join(map(str,c[:16])))')
ulimit -c 0
ulimit -Sv 33554432
export LUMINA_RESPONSE_EXECUTOR=manual-lageunha CUDA_VISIBLE_DEVICES=''
# 기본4시간은 유지한다. 명시한 추가 승인 실행만 빌드 포함12시간을 선택한다.
response_seconds=${LUMINA_RESPONSE_SECONDS:-13800}
[[ "$response_seconds" =~ ^[0-9]+$ ]]
(( response_seconds >= 600 && response_seconds <= 42600 ))
export LUMINA_RESPONSE_SECONDS="$response_seconds"
exec taskset -c "$cpus" timeout -k 30 "$((response_seconds+570))s" \
    bash "$repository/scripts/submit_lumina_new_field_response.slurm" "$@"
