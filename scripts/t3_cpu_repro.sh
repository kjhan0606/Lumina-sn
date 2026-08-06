#!/usr/bin/env bash
# T3 사망 지점을 **CPU 바이너리로 오프라인 재현**한다.
#
# lumina_prepare_solver_owned_tau 는 GPU 경로(lumina_cuda.cu:7488)와 CPU 경로
# (lumina_main.c:250) 양쪽에서 호출된다.  따라서 GPU 판정런을 또 태우지 않고
# 여기서 이유를 확정할 수 있다 — offline-first.
#
# env 는 T3 런처와 **같은 방식으로** 만든다(기억으로 다시 쓰지 않는다).
# 패킷·반복은 최소로 — 목표는 물리가 아니라 준비 단계 통과 여부다.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${T3_DECK:?T3_DECK 필요}"
[[ -d "$MODEL" ]] || { echo "MISSING_DECK $MODEL" >&2; exit 64; }

eval "$(grep -E '^export ' scripts/run_coevolve_s01.sh)"

UNSET_LIST=$(
  { grep -oE 'X\("(LUMINA_[A-Z0-9_]+)", +LK_ENFORCE_FATAL' src/legacy_knob_registry.h \
      | grep -oE 'LUMINA_[A-Z0-9_]+'
    sed -n '/retired_scalar_options\[\]/,/};/p' src/lumina_atomic.c \
      | grep -oE '"LUMINA_[A-Z0-9_]+"' | tr -d '"'
    echo LUMINA_CMF_EPAY
  } | sort -u | tr '\n' ' ')
unset $UNSET_LIST

UNIVERSE=$(grep -oE '"LUMINA_[A-Z0-9_]+"' src/env_universe.h | tr -d '"' | sort -u)
DEAD=$(comm -23 <(env | grep -oE '^LUMINA_[A-Z0-9_]+' | sort -u) <(echo "$UNIVERSE"))
if [[ -n "$DEAD" ]]; then unset $DEAD; fi

export LUMINA_MODEL_DIR="$MODEL"
export LUMINA_DEPOSITION_FILE="$MODEL/deposition_cmfgen.csv"
export LUMINA_CMFGEN_SIGMA_BF="$MODEL/cmfgen_sigma_bf.bin"
export LUMINA_ENV_STRICT=1
export OMP_NUM_THREADS="${OMP:-8}"

echo "=== T3_CPU_REPRO deck=$(basename "$MODEL") host=$(hostname) OMP=$OMP_NUM_THREADS ==="
set +e
./lumina "$MODEL" "${PKTS:-2000}" "${NITER:-1}" spectrum nlte 2>&1
rc=$?
set -e
echo "=== EXIT=$rc ==="
exit $rc
