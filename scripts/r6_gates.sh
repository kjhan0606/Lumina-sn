#!/usr/bin/env bash
# R6 게이트 — 결정론 정본 line-J̄.  사전등록 docs/RUNG_R6_DETERMINISTIC_LINE_JBAR.md.
#
# 전제: Γ단이 커밋되어 트리가 깨끗하고, ./lumina 가 그 상태의 바이너리다.
# ★N6-4(부분 적용범위)는 **주입이 필요 없다** — 생산자가 UV 창만 채우므로 자연 상태가 곧 대조다.
set -u
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
P=/tmp/claude-10396/codex_r6/R6_DETERMINISTIC_LINE_JBAR.patch
OUT=$R/validation/r6
BK=/tmp/claude-10396/r6_backup
mkdir -p "$OUT" "$BK"
V=$OUT/R6_GATES.md; : > "$V"
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$V"; }
stop () { say ""; say "## ★중단: $*"; exit "${2:-1}"; }
SRCS="lumina.h lumina_cmfgen.c lumina_cmfgen.h lumina_main.c radiation_field.c radiation_field.h"
DECK=data/tardis_reference_toy06_19p48d_sivcaiv_active

run () {  # $1=로그 $2=LANE $3=패킷 $4=바이너리
  ssh lageunha "cd $R && timeout 5400 env T3_DECK=$DECK T3_LANE=$2 T3_BIN=$4 PKTS=$3 NITER=1 OMP=32 \
      bash scripts/t3_cpu_repro.sh > /tmp/kjhan_$1.log 2>&1; echo rc=\$?" 2>&1 | tail -1
  ssh lageunha "cp /tmp/kjhan_$1.log $OUT/$1.log" 2>/dev/null
  local got; got=$(grep -m1 -oE 'lane=(MC|DET)' "$OUT/$1.log" 2>/dev/null | cut -d= -f2)
  [ -n "$got" ] && [ "$got" != "$2" ] && { echo "★팔 불일치 $2 vs $got" | tee -a "$V" >&2; }
  return 0
}

say "# R6 게이트 — $(date '+%Y-%m-%d %H:%M')"
say ""
say "## A. 적용 + 빌드"
cd "$R" || stop cd 1
[ -z "$(git status --porcelain src/)" ] || stop "트리가 깨끗하지 않다 — 앞 단이 미커밋이다" 1
cp lumina lumina.preR6
for f in $SRCS; do cp "src/$f" "$BK/$f" 2>/dev/null; done
git apply --check "$P" || stop "패치 미적용" 2
git apply "$P" || stop "적용 실패" 2
say "  적용 $(git status --short src/ | wc -l) 파일"
rm -f lumina
if ! make OMP=1 lumina > "$OUT/build.log" 2>&1; then
  say '```'; grep -E 'error|undefined' "$OUT/build.log" | head -15 | tee -a "$V"; say '```'
  for f in $SRCS; do cp "$BK/$f" "src/$f" 2>/dev/null; done; cp lumina.preR6 lumina
  stop "★빌드 실패 — 되돌렸다.  Codex 반송" 3
fi
cp lumina lumina.postR6
say "  빌드 OK sha=$(sha256sum lumina | cut -c1-12)"

say ""
say "## B. R6-1 결정론 팔이 a209 를 통과하는가 (+ R6-5 적용범위 · N6-4 부분 적용범위)"
rc=$(run r6_det DET 800 ./lumina.postR6); say "  $rc"
say '```'
grep -E '\[R6\]|\[R7\]\[PHASE\]|\[A2-0[89]\]\[|\[A2-10\]\[|EXIT=' "$OUT/r6_det.log" | head -14 | tee -a "$V"
say '```'
a209=$(grep -c 'lane=DET.*phase=a209' "$OUT/r6_det.log" || true)
pre=$(grep -m1 '\[A2-10\]\[PRE\] lane=DET' "$OUT/r6_det.log" || true)
if [ "$a209" -gt 0 ]; then say "  **R6-1 PASS** — DET 가 a209 를 통과했다"; else say "  **R6-1 FAIL** — a209 미통과"; fi
if [ -n "$pre" ]; then
  vals=$(echo "$pre" | grep -oE '(rad|line|opacity|emissivity)=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ' ')
  say "  동세대: { $vals}  $( [ "$(echo "$vals"|wc -w)" -eq 1 ] && echo '<- 일치' || echo '<- ★불일치')"
fi
cov=$(grep -oE 'valid_cells=[0-9]+ exact_zero_cells=[0-9]+' "$OUT/r6_det.log" | head -1)
say "  **R6-5 적용범위**: ${cov:-관측 없음}"
say "  (N6-4: 창 밖 선이 UNSAMPLED 로 남은 채 a209 가 통과하면 PASS — 위 두 줄로 판정)"

say ""
say "## C. R6-4 MC 팔 바이트-parity (결정론 발행 추가가 MC 를 안 건드린다)"
rc=$(run r6_par_pre  MC 100 ./lumina.preR6);  say "  pre  $rc"
rc=$(run r6_par_post MC 100 ./lumina.postR6); say "  post $rc"
python3 - "$OUT/r6_par_pre.log" "$OUT/r6_par_post.log" <<'PY' | tee -a "$V"
import re, sys
from collections import Counter
def nums(p):
    out=[]
    for ln in open(p, errors='ignore'):
        if ln.startswith('[R6]'): continue
        ln = re.sub(r'\[A2-08\]\[BLOCKED\][^\n]*','',ln)
        ln = re.sub(r'(sha|bin)=\S+','',ln)
        out += re.findall(r'[-+]?\d+\.\d+e[-+]?\d+|[-+]?\d+\.\d+|[-+]?\d+', ln)
    return Counter(out)
a,b = nums(sys.argv[1]), nums(sys.argv[2])
oa, ob = a-b, b-a
print(f"  수치 pre={sum(a.values())} post={sum(b.values())} / pre만={sum(oa.values())} post만={sum(ob.values())}")
print("  **R6-4 " + ("PASS** — MC 수치 불변" if not oa and not ob else f"FAIL** {list(oa.items())[:4]} vs {list(ob.items())[:4]}"))
PY

say ""
say "## 남긴 것"
say "- N6-2(q-hash 변조) · N6-3(센티널 VALID 위장) 은 시험 빌드 필요 — 운전석이 깨어서"
say "- R6-2(두 팔 해시 동일)는 **구조적 보장**(같은 line_qset 객체) + view 의 hash 검사로 강제"
say "- 커밋은 운전석"
say "=== R6 GATES DONE ==="
