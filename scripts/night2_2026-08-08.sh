#!/usr/bin/env bash
# 야간 자율 체인 2 — Γ단 반송 수리 착지 → 검사 → 적용 → 빌드 → 게이트.
# R6 는 같은 시각에 Codex 가 저작 중이나 **적용은 하지 않는다**(src-편집 동시 1개).
#
# 원칙(1차와 동일): 앞 단계 PASS 일 때만 진행 · 커밋하지 않는다 · 실패 시 되돌린다.
# ★게이트 순서를 바꿨다: **NC3 를 맨 앞에** 둔다.  오늘 그것이 유일하게 결함을 잡았다.
set -u
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
W=/tmp/claude-10396/codex_gfix
OUT=$R/validation/gamma
BK=/tmp/claude-10396/gamma_backup2
mkdir -p "$OUT" "$BK"
V=$OUT/NIGHT2_VERDICT.md
: > "$V"
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$V"; }
stop () { say ""; say "## ★중단: $*"; exit "${2:-1}"; }

SRCS="lumina.h lumina_plasma.c lumina_main.c lumina_cmfgen.c lumina_cuda.cu lumina_element_wide.c population_contract.c"
DECK=data/tardis_reference_toy06_19p48d_sivcaiv_active
NCDECK=data/tardis_reference_toy06_19p48d_nicozero

run () {  # $1=덱 $2=로그 $3=LANE $4=패킷 $5=바이너리
  ssh lageunha "cd $R && timeout 5400 env T3_DECK=$1 T3_LANE=$3 T3_BIN=$5 PKTS=$4 NITER=1 OMP=32 \
      bash scripts/t3_cpu_repro.sh > /tmp/kjhan_$2.log 2>&1; echo rc=\$?" 2>&1 | tail -1
  ssh lageunha "cp /tmp/kjhan_$2.log $OUT/$2.log" 2>/dev/null
  local got; got=$(grep -m1 -oE 'lane=(MC|DET)' "$OUT/$2.log" 2>/dev/null | cut -d= -f2)
  [ -n "$got" ] && [ "$got" != "$3" ] && { echo "★팔 불일치 $3 vs $got" | tee -a "$V" >&2; touch "$OUT/.mm"; }
  return 0
}

say "# 야간 체인 2 — $(date '+%Y-%m-%d %H:%M')"
say ""
say "## A. Γ 반송 수리 착지 대기 (Codex pid 292377)"
while kill -0 292377 2>/dev/null; do sleep 60; done
say "  Codex 종료."
P=$(ls -t "$W"/*.patch 2>/dev/null | head -1)
[ -n "$P" ] && [ -s "$P" ] || stop "패치 산출이 없다 — OUT_GFIX.md 를 운전석이 읽는다" 2
cp "$W"/OUT_GFIX.md "$OUT/CODEX_GFIX.md" 2>/dev/null
say "  패치: $(basename "$P") $(wc -l < "$P")행"

# ---- 정적 적신호 ----
say ""
say "## B. 정적 검사"
g=$(grep -c '^+.*getenv(' "$P" || true)
c=$(grep -ciE '^\+.*(clamp|fmax\(|fmin\(|floor\()' "$P" || true)
t=$(grep -c '^+.*population_te_manifest_sha256' "$P" || true)
say "  새 getenv=$g  clamp/floor=$c  te_manifest 재사용=$t"
[ "$g" -gt 0 ] && stop "새 env 노브 — 운전석이 읽는다" 3
[ "$c" -gt 0 ] && stop "clamp/floor 도입 — 운전석이 읽는다" 3
[ "$t" -gt 0 ] && stop "★온도 매니페스트를 또 쓴다 — 반송 사유를 안 고쳤다" 3
cd "$R" || stop "cd" 3
git apply --check "$P" 2>/dev/null || git apply --recount --check "$P" || stop "패치가 적용되지 않는다" 3
say "  git apply --check OK"

# ---- 적용 + 빌드 ----
say ""
say "## C. 적용 + 빌드"
for f in $SRCS; do cp "src/$f" "$BK/$f" 2>/dev/null; done
cp lumina "$BK/lumina.pre"
git apply "$P" 2>/dev/null || git apply --recount "$P" || stop "적용 실패" 3
say "  적용: $(git status --short src/ | wc -l) 파일"
rm -f lumina
if ! make OMP=1 lumina > "$OUT/build2.log" 2>&1; then
  say '```'; grep -E 'error|undefined' "$OUT/build2.log" | head -15 | tee -a "$V"; say '```'
  for f in $SRCS; do cp "$BK/$f" "src/$f" 2>/dev/null; done; cp "$BK/lumina.pre" lumina
  stop "★빌드 실패 — 되돌렸다.  Codex 반송(운전석은 고치지 않는다)" 4
fi
say "  빌드 OK sha=$(sha256sum lumina | cut -c1-12)"
cp lumina lumina.postGamma2
# Γ2-a 재확인
git show HEAD:src/lumina_plasma.c > /tmp/claude-10396/ph.c
awk '/^(static )?void compute_gamma_deposition\(/,/^}/' /tmp/claude-10396/ph.c > /tmp/claude-10396/a.c
awk '/^(static )?void compute_gamma_deposition\(/,/^}/' src/lumina_plasma.c > /tmp/claude-10396/b.c
d=$(diff /tmp/claude-10396/a.c /tmp/claude-10396/b.c | grep -cE '^[<>]' || true)
say "  Γ2-a 수식 diff 줄=$d (static 한정자 2줄까지가 정상)"
[ "$d" -gt 2 ] && say "  ★Γ2-a 의심 — 운전석이 읽는다"

# ---- ★NC3 부터 ----
say ""
say "## D. ★NC3 (정당하게 0) — 오늘 이것만이 결함을 잡았다"
rm -f "$OUT/.mm"; rc=$(run "$NCDECK" n2_nc3 MC 100 ./lumina.postGamma2); say "  $rc"
say '```'; grep -E '\[GAMMA|RADEQ_GAMMA_UNPUB|\[A2-10\]\[|EXIT=' "$OUT/n2_nc3.log" | head -10 | tee -a "$V"; say '```'
pub=$(grep -c 'GAMMA.*PUBLISHED' "$OUT/n2_nc3.log" || true)
unp=$(grep -c 'RADEQ_GAMMA_UNPUBLISHED\|GAMMA_MANIFEST_BUILD_FAILED' "$OUT/n2_nc3.log" || true)
if [ "$pub" -gt 0 ] && [ "$unp" -eq 0 ]; then say "  **NC3 PASS**"; else
  say "  **NC3 FAIL** 발행=$pub 차단=$unp"
  for f in $SRCS; do cp "$BK/$f" "src/$f" 2>/dev/null; done; cp "$BK/lumina.pre" lumina
  stop "NC3 재실패 — 되돌렸다.  Codex 재반송" 5
fi

# ---- Γ2-b parity ----
say ""
say "## E. Γ2-b 바이트-parity (MC 100pkt, preGamma vs postGamma2)"
rm -f "$OUT/.mm"; rc=$(run "$DECK" n2_par_pre  MC 100 ./lumina.preGamma);   say "  pre  $rc"
rc=$(run "$DECK" n2_par_post MC 100 ./lumina.postGamma2); say "  post $rc"
strip () { grep -vE '\[GAMMA|host=|sha=|bin=' "$1" | sed 's/[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}//g'; }
strip "$OUT/n2_par_pre.log"  > /tmp/claude-10396/np.txt
strip "$OUT/n2_par_post.log" > /tmp/claude-10396/nq.txt
if diff -q /tmp/claude-10396/np.txt /tmp/claude-10396/nq.txt >/dev/null; then
  say "  **Γ2-b PASS** — 값이 하나도 안 바뀌었다."
else
  say "  **Γ2-b FAIL** — 차이:"; say '```'
  diff /tmp/claude-10396/np.txt /tmp/claude-10396/nq.txt | head -25 | tee -a "$V"; say '```'
fi
say ""; say "### 새 감마 관측"; say '```'
grep -E '\[GAMMA' "$OUT/n2_par_post.log" | head -8 | tee -a "$V"; say '```'

# ---- Γ3 DET ----
say ""
say "## F. Γ3 발행이 a208 앞에 서는가 (DET)"
rm -f "$OUT/.mm"; rc=$(run "$DECK" n2_det DET 800 ./lumina.postGamma2); say "  $rc"
say '```'; grep -E '\[GAMMA|\[R7\]\[PHASE\]|\[A2-0[89]\]\[|EXIT=' "$OUT/n2_det.log" | head -12 | tee -a "$V"; say '```'
gp=$(grep -n 'GAMMA.*PUBLISHED' "$OUT/n2_det.log" | head -1 | cut -d: -f1)
ap=$(grep -n 'phase=a208'       "$OUT/n2_det.log" | head -1 | cut -d: -f1)
if [ -n "$gp" ] && [ -n "$ap" ] && [ "$gp" -lt "$ap" ]; then say "  **Γ3(위상) PASS** 발행 $gp < a208 $ap"
else say "  **Γ3(위상) FAIL** 발행=${gp:-없음} a208=${ap:-없음}"; fi

say ""
say "## 남긴 것"
say "- 커밋은 운전석이 깨어서 한다(계약1=커밋1)"
say "- R6 패치는 **적용하지 않았다**(src-편집 동시 1개).  산출 /tmp/claude-10396/codex_r6/"
say "- Γ4(M1/M2)는 사건 측도 단(E) 이후 — MC 전송이 죽어 A2-10 에 도달 못 한다"
say "=== NIGHT2 DONE ==="
