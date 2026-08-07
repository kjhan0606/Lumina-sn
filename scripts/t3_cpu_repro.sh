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

# ★이 하네스는 **GPU 없는 노드**에서 CPU 바이너리를 돌린다.  참조 런처는
# LUMINA_CMF_SOLVE_GPU=1 을 상속시키는데, 그러면 결정론 팔이 GPU 솔버를 요구하고
# 계약이 조용한 CPU 폴백을 옳게 거부한다(BLOCKED_GPU_FALLBACK_FORBIDDEN).
# 하네스이므로 여기서만 내린다 — **숨기지 않고 찍는다**.  물리 설정은 건드리지 않는다.
if [[ -n "${LUMINA_CMF_SOLVE_GPU-}" ]]; then
  echo "[HARNESS] unset LUMINA_CMF_SOLVE_GPU (was '$LUMINA_CMF_SOLVE_GPU') — CPU-only node"
  unset LUMINA_CMF_SOLVE_GPU
fi

# ★2026-08-08: 위 `eval` 이 런처의 export 를 **호출자 env 뒤에** 적용한다.  그래서
# `env LUMINA_PURE_CMFGEN=0 bash t3_cpu_repro.sh` 가 조용히 무시됐다 — run_coevolve_s01.sh:30
# 이 LUMINA_PURE_CMFGEN=1 을 다시 export 한다.  R7 판정에서 **DET 를 두 번 돌리고 하나를
# MC 라 불렀다**(로그의 `lane=DET` 가 잡았다).  팔 선택은 eval **뒤에** 하고, 찍는다.
if [[ -n "${T3_LANE-}" ]]; then
  case "$T3_LANE" in
    MC)  export LUMINA_PURE_CMFGEN=0 ;;
    DET) export LUMINA_PURE_CMFGEN=1 ;;
    *)   echo "[HARNESS][FATAL] T3_LANE=$T3_LANE (MC|DET 만)" >&2; exit 65 ;;
  esac
  echo "[HARNESS] T3_LANE=$T3_LANE -> LUMINA_PURE_CMFGEN=$LUMINA_PURE_CMFGEN"
fi

export LUMINA_MODEL_DIR="$MODEL"
export LUMINA_DEPOSITION_FILE="$MODEL/deposition_cmfgen.csv"
export LUMINA_CMFGEN_SIGMA_BF="$MODEL/cmfgen_sigma_bf.bin"
export LUMINA_ENV_STRICT=1
export OMP_NUM_THREADS="${OMP:-32}"

# ★2026-08-07: 오늘 CPU 런이 전부 **단일 코어**로 돌았다.  Makefile 의 OMP 는 스레드 수가
# 아니라 **빌드 스위치**(`ifdef OMP -> -fopenmp`)인데 `make lumina` 로만 빌드해 OpenMP 가
# 아예 없었다.  그래서 여기 OMP_NUM_THREADS 는 무시됐다.  하네스가 그것을 잡는다.
# ⚠`grep -q` 를 파이프에 쓰면 안 된다: 첫 일치에서 즉시 종료하며 파이프를 닫아
# nm 이 SIGPIPE 로 죽고, `set -o pipefail` 때문에 파이프라인이 비0 이 된다 ⟹ 오판.
# 2026-08-07 에 이 가드가 멀쩡한 바이너리를 세 번 거부했다.  계수로 받는다.
_omp_syms=$(nm ./lumina 2>/dev/null | grep -c 'GOMP\|omp_get' || true)
if [ "${_omp_syms:-0}" -eq 0 ]; then
  echo "[HARNESS][FATAL] ./lumina 에 OpenMP 가 없다 — 'rm -f lumina && make OMP=1 lumina' 로 빌드하라." >&2
  echo "  (Makefile 의 OMP 는 스레드 수가 아니라 빌드 스위치다.  스레드 수는 이 스크립트의 OMP=)" >&2
  exit 70
fi

echo "=== T3_CPU_REPRO deck=$(basename "$MODEL") host=$(hostname) OMP=$OMP_NUM_THREADS ==="
set +e
./lumina "$MODEL" "${PKTS:-2000}" "${NITER:-1}" spectrum nlte 2>&1
rc=$?
set -e
echo "=== EXIT=$rc ==="
exit $rc
