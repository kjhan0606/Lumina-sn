#!/usr/bin/env bash
# R7 판정 체인 — 적용된 발행 위상이 **런에서** 성립하는지 본다.
# 사전등록: validation/r7/CODEX_R7.md §3.  기대 결과는 두 갈래다.
#
#   DET lane : a209 가 **차단**된다(line view 미착륙).  이것은 R7 실패가 아니라
#              R6 경계다 — Codex 가 사전등록했고 판정에 넣지 않는다.
#              ★R7 의 성립 증거는 **차단 지점이 옮겨졌다**는 것:
#                수리 전 = A2-10 blocked_stale(자격 실패, 원인 불명)
#                수리 후 = A2-09 blocked_stale_line(원인 지목)
#   MC  lane : 전 위상이 서야 한다(o=e=r, t:1->2).  ★이쪽이 진짜 판정이다.
#
# user 상설 지시: 런이 시작하면 인계철선을 걸고 종료신호에 다음 스텝 자동 진입.
set -u
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
OUT=$R/validation/r7
mkdir -p "$OUT"
LOG=$OUT/chain.log
: > "$LOG"
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

BIN_SHA=$(sha256sum "$R/lumina" | cut -d" " -f1)
say "바이너리 sha=${BIN_SHA:0:12}"
_omp=$(nm "$R/lumina" | grep -c 'GOMP_parallel' || true)
[ "$_omp" -eq 0 ] && { say "★OMP 없는 바이너리 — 중단"; exit 70; }

run () {  # $1=덱 $2=로그이름 $3=추가env
  ssh lageunha "cd $R && timeout 5400 env T3_DECK=$1 PKTS=800 NITER=1 OMP=32 $3 \
      bash scripts/t3_cpu_repro.sh > /tmp/kjhan_$2.log 2>&1; echo rc=\$?" 2>&1 | tail -1
  ssh lageunha "cp /tmp/kjhan_$2.log $OUT/$2.log" 2>/dev/null
}

DECK=data/tardis_reference_toy06_19p48d_sivcaiv_active

# ---------- 1. DET lane (현행 경로) ----------
say "=== DET lane: 차단 지점이 A2-10 -> A2-09 로 옮겨졌는가 ==="
rc=$(run "$DECK" r7_det "")
say "  종료 $rc"
grep -E '\[R7\]\[PHASE\]|\[A2-0[89]\]\[|\[A2-10\]\[|R7_|EXIT=' "$OUT/r7_det.log" \
     2>/dev/null | head -25 | sed 's/^/    /' | tee -a "$LOG"

# ---------- 2. MC lane (진짜 판정) ----------
say "=== MC lane: 전 위상 성립 (o=e=r, t:1->2) ==="
rc=$(run "$DECK" r7_mc "LUMINA_PURE_CMFGEN=0")
say "  종료 $rc"
grep -E '\[R7\]\[PHASE\]|\[A2-0[89]\]\[|\[A2-10\]\[|R7_|EXIT=' "$OUT/r7_mc.log" \
     2>/dev/null | head -25 | sed 's/^/    /' | tee -a "$LOG"

# ---------- 3. 위상 검사기 ----------
say "=== check_publication_phase.py ==="
python3 "$R/scripts/check_publication_phase.py" "$OUT/r7_det.log" "$OUT/r7_mc.log" \
  2>&1 | sed 's/^/    /' | tee -a "$LOG"

say "=== CHAIN DONE ==="
