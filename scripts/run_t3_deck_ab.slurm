#!/usr/bin/env bash
#SBATCH --job-name=t3deckab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/kjhan/lumina/t3/%j.out
set -euo pipefail

# T3 — 덱 전환 A/B (docs/DECK_TRANSITION_SCOPING.md).
#
# 목적: 회귀 대장에 **새 덱 축**의 첫 점을 찍는다. 구 축과의 비교가 아니다 —
# 구 생산 덱(bare)은 현 로더가 거부하므로(0-K 형상 30열/50셸) 재현 자체가 불가능하다.
# 두 arm 의 차이는 **수소꼴 정확 평가 하나**로 격리된다:
#     _vac   = jnu4 vintage + I20 수리
#     _jnu4  = jnu4 vintage + I20 수리 + CMFGEN_EXACT_HYD
# 나중에 문제가 생겼을 때 이등분할 수 있는 축을 지금 만들어 둔다.
#
# ★폐기·금지 env 를 설정하지 않는다 — src/legacy_knob_registry.h 의 ENFORCE_FATAL 12종.
#   참조 런처 run_coevolve_s01.sh 는 LUMINA_TE_TRAD_RATIO=1.0 과
#   LUMINA_TRAD_COLOR_FIX=1 을 export 하는데 둘 다 값을 보지 않는 FATAL 이라
#   현 바이너리로는 로드 단계에서 즉사한다(2026-08-06 프리플라이트 적발, 영향 런처 83-166).

REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
cd "$REPO"
module load cuda 2>/dev/null || true

MODEL="${T3_DECK:?T3_DECK must name a deck under data/}"
[[ -d "$MODEL" ]] || { echo "T3_MISSING_DECK $MODEL" >&2; exit 64; }
[[ -f "$MODEL/DECK_PROVENANCE.json" ]] || { echo "T3_DECK_NO_PROVENANCE $MODEL" >&2; exit 65; }

OUT=/gpfs/kjhan/lumina/t3/${SLURM_JOB_ID:-manual}_$(basename "$MODEL")
mkdir -p "$OUT"

# --- 금지 env 프리플라이트: 상속으로 들어온 것도 잡는다 (--export=ALL 대비) ---
FATAL_ENV=$(grep -oE 'X\("(LUMINA_[A-Z0-9_]+)", +LK_ENFORCE_FATAL' src/legacy_knob_registry.h \
            | grep -oE 'LUMINA_[A-Z0-9_]+')
bad=0
for k in $FATAL_ENV; do
  v="${!k-}"
  case "$k" in
    LUMINA_TE_TRAD_RATIO|LUMINA_TRAD_COLOR_FIX)
      [[ -n "$v" ]] && { echo "T3_FORBIDDEN_ENV $k='$v' (비어있지 않으면 FATAL)" >&2; bad=1; } ;;
    *)
      [[ -n "$v" && "$v" != 0 ]] && { echo "T3_FORBIDDEN_ENV $k='$v' (atoi!=0 이면 FATAL)" >&2; bad=1; } ;;
  esac
done
(( bad == 0 )) || { echo "T3_PREFLIGHT_FAIL — 금지 env 가 환경에 있다" >&2; exit 66; }

# --- 물리 설정: 참조 런처에서 **그대로 상속**한다 (기억으로 다시 쓰지 않는다) ---
#
# ★2026-08-07 첫 T3 제출 실패의 원인: env 목록을 참조 런처에서 **기억으로 다시 썼고**
#   50개 중 16개만 옮겨 34개를 빠뜨렸다(KPACKET · MAX_INTERACTIONS · BF_RATE_POPS ·
#   JBAR_MIN 등 런 동작 직결 항목 포함). 그러고서 "금지 2종만 제거했다"고 잘못 보고했다.
#   ⟹ 목록을 손으로 옮기지 않는다. 참조의 export 줄을 그대로 eval 하고,
#     **금지된 것만 이름으로 unset** 한다. 무엇을 뺐는지가 코드에 드러나고,
#     참조가 바뀌면 따라간다.
REF_LAUNCHER=scripts/run_coevolve_s01.sh
[[ -f "$REF_LAUNCHER" ]] || { echo "T3_NO_REF_LAUNCHER $REF_LAUNCHER" >&2; exit 68; }
NITER="${NITER:-12}"
eval "$(grep -E '^export ' "$REF_LAUNCHER")"          # 39 export 줄, 실행줄은 포함 안 됨
N_INHERITED=$(grep -cE '^export ' "$REF_LAUNCHER")

# 금지 env 만 이름으로 제거 (src/legacy_knob_registry.h 의 ENFORCE_FATAL 중 참조가 켜는 것)
UNSET_LIST="LUMINA_TE_TRAD_RATIO LUMINA_TRAD_COLOR_FIX"
unset $UNSET_LIST

# T3 고유 설정만 덮어쓴다
export LUMINA_MODEL_DIR="$MODEL"
export LUMINA_DEPOSITION_FILE="$MODEL/deposition_cmfgen.csv"
export LUMINA_CMFGEN_SIGMA_BF="$MODEL/cmfgen_sigma_bf.bin"
export EVENT_LOG=1 EVENT_LOG_CAP_MB=128        # 캠페인 상설: 전 런 이벤트로그 ON

# ★잡 안에서 빌드하지 않는다.  2026-08-07 첫 제출에서 두 잡이 1초 차로 시작해 같은
# 저장소에서 make 를 동시에 돌렸고, 실행 중인 바이너리가 덮어써져 둘 다 exit 1 로 죽었다
# (오류 메시지 없음 · 바이너리 mtime 이 런 도중).  빌드는 제출 전에 한 번만 한다.
[[ -x ./lumina_cuda ]] || { echo "T3_NO_BINARY — 제출 전에 make lumina_cuda 를 한 번만 돌려라" >&2; exit 67; }
BIN_SHA=$(sha256sum ./lumina_cuda | cut -d" " -f1)
PKTS="${PKTS:-400000}"; NIT="${NITER:-12}"

# --- RUN FOOTER: argv 까지 복제 (feedback_replicate_argv_not_just_env) ---
{
  echo "T3_RUN_FOOTER"
  echo "  git_head=$(git rev-parse HEAD)"
  echo "  binary_sha256=$BIN_SHA"
  echo "  env_inherited_from=$REF_LAUNCHER ($N_INHERITED export lines)"
  echo "  env_unset=$UNSET_LIST"
  echo "  deck=$MODEL"
  echo "  deck_provenance=$(python3 -c "import json;print(json.load(open('$MODEL/DECK_PROVENANCE.json')).get('cmfgen_links','UNKNOWN'))")"
  echo "  argv=./lumina_cuda $MODEL $PKTS $NIT spectrum nlte"
  echo "  node=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  env | grep -E '^(LUMINA_|OMP_|EVENT_LOG)' | LC_ALL=C sort
} > "$OUT/RUN_FOOTER.txt"
cp "$MODEL/DECK_PROVENANCE.json" "$OUT/deck_provenance.json"

./lumina_cuda "$MODEL" "$PKTS" "$NIT" spectrum nlte 2>&1 | tee "$OUT/run.log"
echo "T3_DONE deck=$MODEL out=$OUT"
