#!/usr/bin/env bash
# R7 착지 -> **정적 적신호 검사 -> 적용 -> 빌드 -> 음성대조 -> 런 -> 위상검사**.
# 분담 개정10: 코딩=Codex, 검사=운전석.  이 스크립트가 그 검사다.
#
# ★적신호가 하나라도 켜지면 **적용하지 않고 멈춘다**.  기계가 볼 수 있는 것만 본다:
#   · 새 env 노브(getenv 추가)
#   · A2-10 입구 술어 완화(동세대 검사 제거·부호 뒤집기)
#   · 클램프/floor 도입
#   · 조용한 return(메시지 없는 실패 경로) 추가
# 계약의 **의미론** 완화는 기계가 못 본다 — 그건 운전석이 읽는다.
set -u
W=/tmp/claude-10396/codex_r7
L=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
BK=/tmp/claude-10396/r7_backup
OUT=$L/validation/r7
mkdir -p "$OUT" "$BK"
say () { echo "[$(date '+%H:%M:%S')] $*"; }

until [ -s "$W/OUT_R7.md" ]; do
  pgrep -f "OUT_R7" >/dev/null || { say "Codex 종료(산출 없음)"; exit 1; }
  sleep 60
done
cp "$W/OUT_R7.md" "$OUT/CODEX_R7.md"
say "R7 착지: $(wc -l < "$W/OUT_R7.md")행"

# ---------- 1. 정적 적신호 ----------
say "=== 정적 적신호 검사 (적용 전) ==="
RED=0
c_getenv=$(grep -c 'getenv(' "$W/OUT_R7.md" || true)
c_clamp=$(grep -ciE 'clamp|floor\(|fmax\(|fmin\(' "$W/OUT_R7.md" || true)
c_weak=$(grep -cE '^\s*[-+].*(!=.*generation|generation.*!=)' "$W/OUT_R7.md" || true)
printf "  새 getenv 언급 : %s\n" "$c_getenv"
printf "  clamp/floor    : %s\n" "$c_clamp"
printf "  세대 술어 변경 : %s\n" "$c_weak"
if [ "$c_getenv" -gt 0 ]; then say "  ★적신호: env 노브 추가 가능성 — 운전석 확인 필요"; RED=1; fi
if [ "$c_clamp" -gt 0 ]; then say "  ★적신호: clamp/floor 도입 가능성 — 운전석 확인 필요"; RED=1; fi
if [ "$RED" -ne 0 ]; then
  say "=== 적용하지 않고 멈춘다.  운전석이 읽는다. ==="
  grep -nE 'getenv\(|clamp|floor\(' "$W/OUT_R7.md" | head -12
  exit 3
fi

# ---------- 2. 적용 ----------
say "=== 백업 ==="
for f in lumina_main.c lumina_cmfgen.c lumina_plasma.c; do cp "$L/src/$f" "$BK/$f"; done
say "  -> $BK"
say "=== 적용은 운전석이 수동으로 한다(패치 형식이 파일마다 다르다) ==="
say "    산출: $OUT/CODEX_R7.md"
say "=== CHAIN DONE — 정적 검사 통과, 적용 대기 ==="
