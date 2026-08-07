#!/usr/bin/env bash
# 야간 자율주행 — R7 판정 폐합 → Γ단(감마 침착 소유권) 적용·빌드·게이트.
# user 2026-08-08 01:4x: "난, 자러감. 자율주행"
#
# ★설계 원칙
#   · 각 단계는 **앞 단계가 PASS 일 때만** 진행한다.  FAIL 이면 그 자리에서 멈추고 기록한다.
#   · **커밋하지 않는다.**  판정은 운전석이 깨어서 읽고 한다(계약1=커밋1).
#   · 덱 정본을 건드리지 않는다.  NC3 덱은 **심볼릭 링크 사본**에 abundances 만 바꿔 만든다.
#   · 적용 전 바이너리·소스를 백업한다.  실패 시 되돌린다.
#   · 진행은 VERDICT 파일에 **증분 기록** — 중간에 죽어도 어디까지 갔는지 남는다.
set -u
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
W=/tmp/claude-10396/codex_gamma
OUT=$R/validation/gamma
BK=/tmp/claude-10396/gamma_backup
mkdir -p "$OUT" "$BK"
V=$OUT/NIGHT_VERDICT.md
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$V"; }
stop () { say ""; say "## ★중단: $*"; say "다음 행동은 운전석이 깨어서 판단한다."; exit "${2:-1}"; }

: > "$V"
say "# 야간 자율주행 판정 — $(date '+%Y-%m-%d %H:%M')"
say ""

run () {  # $1=덱 $2=로그이름 $3=T3_LANE(MC|DET) $4=패킷
  ssh lageunha "cd $R && timeout 5400 env T3_DECK=$1 T3_LANE=$3 PKTS=${4:-800} NITER=1 OMP=32 \
      bash scripts/t3_cpu_repro.sh > /tmp/kjhan_$2.log 2>&1; echo rc=\$?" 2>&1 | tail -1
  ssh lageunha "cp /tmp/kjhan_$2.log $OUT/$2.log" 2>/dev/null
  # ★팔이 실제로 그 팔인지 확인한다.  01:45 에 DET 를 MC 라 부른 사고의 재발 방지.
  local got; got=$(grep -m1 -oE 'lane=(MC|DET)' "$OUT/$2.log" 2>/dev/null | cut -d= -f2)
  if [ -n "$got" ] && [ "$got" != "$3" ]; then
    { echo "[$(date '+%H:%M:%S')]   ★팔 불일치: 요청=$3 실제=$got — 하니스가 거짓말했다"; } \
      | tee -a "$V" >&2
    touch "$OUT/.lane_mismatch"
    return 9
  fi
}
DECK=data/tardis_reference_toy06_19p48d_sivcaiv_active

# ══════════════════════════════════════════════════════════════════
say "## A. R7 MC 판정 대기"
# ══════════════════════════════════════════════════════════════════
MCL=$OUT/r7_mc_real.log
say "01:45 판정은 **무효**였다 — LUMINA_PURE_CMFGEN=0 이 하니스의 eval 에 덮여"
say "DET 를 두 번 돌리고 하나를 MC 라 불렀다(로그의 lane=DET 가 잡았다)."
say "하니스에 T3_LANE 을 넣고 다시 돌린다."
rm -f "$OUT/.lane_mismatch"
rc=$(run "$DECK" r7_mc_real MC 800); say "  MC 런 종료 $rc"
[ -e "$OUT/.lane_mismatch" ] && stop "요청한 팔로 돌지 않았다" 2
[ -s "$MCL" ] || stop "MC 로그가 없다" 2

a209=$(grep -c 'lane=MC.*phase=a209' "$MCL" || true)
pre=$(grep -m1 '\[A2-10\]\[PRE\] lane=MC' "$MCL" || true)
commit=$(grep -m1 'R7_MATERIAL_PHASE_COMMITTED.*lane=MC' "$MCL" || true)
say ""
say '```'
grep -E '\[R7\]\[PHASE\]|\[A2-0[89]\]\[|\[A2-10\]\[|R7_|EXIT=' "$MCL" | head -12 | tee -a "$V"
say '```'

MCPASS=1
[ "$a209" -gt 0 ] || { say "  ★MC a209 위상 없음"; MCPASS=0; }
[ -n "$pre" ]     || { say "  ★MC A2-10 PRE 없음 — A2-10 미도달"; MCPASS=0; }
[ -n "$commit" ]  || { say "  ★MC MATERIAL_PHASE_COMMITTED 없음"; MCPASS=0; }
if [ -n "$pre" ]; then
  # o=e=r 동세대인가 (기대: rad=line=opacity=emissivity)
  vals=$(echo "$pre" | grep -oE '(rad|line|opacity|emissivity)=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ' ')
  say "  동세대 검사: rad/line/opacity/emissivity = { $vals}"
  [ "$(echo "$vals" | wc -w)" -eq 1 ] || { say "  ★동세대 아님"; MCPASS=0; }
fi
python3 "$R/scripts/check_publication_phase.py" "$MCL" 2>&1 | sed 's/^/    /' | tee -a "$V"

if [ "$MCPASS" -ne 1 ]; then
  say ""
  say "**R7 MC = FAIL.**  사전등록에 없는 실패다 — R7 의 실제 실패다."
  stop "R7 이 닫히지 않았다.  감마를 얹지 않는다(단독 귀속이 처음부터 깨진다)." 2
fi
say ""
say "**R7 MC = PASS.**  DET 는 사전등록대로 A2-09 경계(R6 소관).  R7 폐합."

# ══════════════════════════════════════════════════════════════════
say ""
say "## B. Γ단 적용 + 빌드"
# ══════════════════════════════════════════════════════════════════
cd "$R" || stop "cd 실패" 3
cp lumina "$BK/lumina.preGamma" && cp lumina lumina.preGamma
for f in lumina.h lumina_plasma.c lumina_main.c lumina_cmfgen.c lumina_cuda.cu \
         lumina_element_wide.c radeq_publication.c radeq_publication.h; do
  cp "src/$f" "$BK/$f"
done
say "백업: $BK  (바이너리 sha=$(sha256sum lumina | cut -c1-12))"

git apply --check "$W/gamma_deposition_owner.patch" || stop "패치가 적용되지 않는다" 3
git apply "$W/gamma_deposition_owner.patch" || stop "패치 적용 실패" 3
say "패치 적용: $(git status --short src/ | wc -l) 파일"

rm -f lumina
if ! make OMP=1 lumina > "$OUT/build.log" 2>&1; then
  say ""
  say '```'
  grep -E 'error|Error|undefined' "$OUT/build.log" | head -20 | tee -a "$V"
  say '```'
  say "소스를 되돌린다(검사 결과는 남긴다)."
  for f in lumina.h lumina_plasma.c lumina_main.c lumina_cmfgen.c lumina_cuda.cu \
           lumina_element_wide.c radeq_publication.c radeq_publication.h; do
    cp "$BK/$f" "src/$f"
  done
  cp "$BK/lumina.preGamma" lumina
  stop "★빌드 실패 — Codex 에 되돌린다(운전석은 고치지 않는다, 분담 개정10)" 4
fi
_omp=$(nm ./lumina | grep -c 'GOMP_parallel' || true)
[ "$_omp" -gt 0 ] || stop "빌드는 됐으나 OpenMP 가 없다" 4
say "빌드 OK (OMP 심볼 $_omp, sha=$(sha256sum lumina | cut -c1-12))"

# ---- Γ2-a: 계산식 불변 (정적) ----
say ""
say "### Γ2-a 계산식 불변 (정적 diff)"
FORM='epsilon_gamma|bateman_factor|exp_ni|exp_co|f_dep|KAPPA_GAMMA|column_density|n_ni|n_co|Q_NI56|Q_CO56|LAMBDA_NI56|LAMBDA_CO56|W_ION_EV|ETA_NONTHERMAL'
touched=$(git diff -U0 src/ | grep -E "^[-+]" | grep -vE '^[-+][-+]' | grep -cE "$FORM" || true)
say "  수식 줄 변경: $touched"
if [ "$touched" -ne 0 ]; then
  git diff -U0 src/ | grep -E "^[-+]" | grep -vE '^[-+][-+]' | grep -E "$FORM" | head -10 | sed 's/^/    /' | tee -a "$V"
  say "  ★수식이 바뀌었다 — Γ2 위반.  단독 귀속이 깨진다."
fi

# ══════════════════════════════════════════════════════════════════
say ""
say "## C. Γ2-b 바이트-parity (MC 마이크로픽스처 100pkt×1iter)"
say "계약만 추가했다면 MC 런의 **모든 수치가 동일**해야 한다."
# ══════════════════════════════════════════════════════════════════
cp lumina lumina.postGamma
cp lumina.preGamma lumina
say "  preGamma 런..."
rm -f "$OUT/.lane_mismatch"; rc=$(run "$DECK" gamma_parity_pre MC 100); say "    $rc"; [ -e "$OUT/.lane_mismatch" ] && stop "팔 불일치" 5
cp lumina.postGamma lumina
say "  postGamma 런..."
rm -f "$OUT/.lane_mismatch"; rc=$(run "$DECK" gamma_parity_post MC 100); say "    $rc"; [ -e "$OUT/.lane_mismatch" ] && stop "팔 불일치" 5

# 새 진단 줄과 호스트/시간 의존 줄만 걷어내고 비교
strip () { grep -vE '\[GAMMA|GAMMA-MEASURE|host=|real|user|sys|elapsed|sha=' "$1" | sed 's/[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}//g'; }
strip "$OUT/gamma_parity_pre.log"  > /tmp/claude-10396/gp_pre.txt  2>/dev/null
strip "$OUT/gamma_parity_post.log" > /tmp/claude-10396/gp_post.txt 2>/dev/null
if diff -q /tmp/claude-10396/gp_pre.txt /tmp/claude-10396/gp_post.txt >/dev/null 2>&1; then
  say "  **Γ2-b PASS** — 새 진단을 제외한 전 출력이 동일하다."
  PAR=1
else
  say "  **Γ2-b FAIL** — 수치가 달라졌다.  차이 앞부분:"
  say '```'
  diff /tmp/claude-10396/gp_pre.txt /tmp/claude-10396/gp_post.txt | head -30 | tee -a "$V"
  say '```'
  PAR=0
fi
say ""
say "### 감마 발행 관측 (postGamma MC)"
say '```'
grep -E '\[GAMMA\]|\[GAMMA-MEASURE\]' "$OUT/gamma_parity_post.log" 2>/dev/null | head -8 | tee -a "$V"
say '```'

[ "$PAR" -eq 1 ] || stop "Γ2 실패 — 값이 바뀌었다.  이 단은 값을 고치는 단이 아니다." 5

# ══════════════════════════════════════════════════════════════════
say ""
say "## D. Γ3 결정론 팔 — 발행이 a208 **앞**에 서는가"
say "⚠Γ3 원문은 DET 가 A2-10 까지 가기를 요구하나 그것은 R6 착지 후다."
say "  지금 검증 가능한 것은 **발행 위상**(GAMMA→a208)이다.  나머지는 R6 로 이월."
# ══════════════════════════════════════════════════════════════════
rm -f "$OUT/.lane_mismatch"; rc=$(run "$DECK" gamma_det DET 800); say "  $rc"; [ -e "$OUT/.lane_mismatch" ] && stop "팔 불일치" 5
say '```'
grep -E '\[GAMMA\]|\[R7\]\[PHASE\]|\[A2-0[89]\]\[|\[A2-10\]\[|EXIT=' "$OUT/gamma_det.log" \
  2>/dev/null | head -12 | tee -a "$V"
say '```'
gp=$(grep -n 'GAMMA..PUBLISHED' "$OUT/gamma_det.log" 2>/dev/null | head -1 | cut -d: -f1)
ap=$(grep -n 'phase=a208' "$OUT/gamma_det.log" 2>/dev/null | head -1 | cut -d: -f1)
if [ -n "$gp" ] && [ -n "$ap" ] && [ "$gp" -lt "$ap" ]; then
  say "  **Γ3(부분) PASS** — 발행($gp행)이 a208($ap행) 앞."
else
  say "  **Γ3(부분) FAIL** — 발행=$gp a208=$ap"
fi

# ══════════════════════════════════════════════════════════════════
say ""
say "## E. ★NC3 — 정당하게 0 인 덱은 **통과**해야 한다"
say "이 대조가 없으면 게이트가 '0 이면 무조건 차단' 인 잘못된 게이트가 된다."
# ══════════════════════════════════════════════════════════════════
ND=$R/data/tardis_reference_toy06_19p48d_nicozero
if [ ! -d "$ND" ]; then
  mkdir -p "$ND"
  for f in "$R/$DECK"/*; do
    b=$(basename "$f")
    [ "$b" = "abundances.csv" ] && continue
    ln -sf "$f" "$ND/$b"
  done
  python3 - "$R/$DECK/abundances.csv" "$ND/abundances.csv" <<'PY'
import sys, csv
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.reader(open(src)))
hdr, body = rows[0], rows[1:]
data = {int(r[0]): [float(x) for x in r[1:]] for r in body}
ns = len(hdr) - 1
for s in range(ns):
    removed = sum(data[z][s] for z in (27, 28) if z in data)
    rest = sum(data[z][s] for z in data if z not in (27, 28))
    for z in (27, 28):
        if z in data: data[z][s] = 0.0
    if rest > 0:                      # 남은 원소로 재규격 (합=1 유지)
        f = (rest + removed) / rest
        for z in data:
            if z not in (27, 28): data[z][s] *= f
with open(dst, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(hdr)
    for z in sorted(data): w.writerow([z] + [repr(v) for v in data[z]])
# 검증: 합이 1 인가, Ni/Co 가 0 인가
for s in range(ns):
    tot = sum(data[z][s] for z in data)
    assert abs(tot - 1.0) < 1e-9, f"shell {s} sum={tot}"
    assert data.get(27, [0]*ns)[s] == 0.0 and data.get(28, [0]*ns)[s] == 0.0
print("NC3 deck OK: Ni/Co = 0, 전 셸 합 = 1")
PY
  say "  NC3 덱 생성(심링크 + abundances 만 교체): $(basename "$ND")"
fi
rc=$(run "data/tardis_reference_toy06_19p48d_nicozero" gamma_nc3 MC 100); say "  $rc"
say '```'
grep -E '\[GAMMA\]|RADEQ_GAMMA_UNPUBLISHED|\[A2-10\]\[|EXIT=' "$OUT/gamma_nc3.log" \
  2>/dev/null | head -10 | tee -a "$V"
say '```'
pub=$(grep -c 'GAMMA..PUBLISHED' "$OUT/gamma_nc3.log" 2>/dev/null || true)
unp=$(grep -c 'RADEQ_GAMMA_UNPUBLISHED' "$OUT/gamma_nc3.log" 2>/dev/null || true)
if [ "$pub" -gt 0 ] && [ "$unp" -eq 0 ]; then
  say "  **NC3 PASS** — 정확히 0 인데 발행되고 런이 진행됐다."
else
  say "  **NC3 FAIL** — 발행=$pub 차단=$unp.  게이트가 과잉이다."
fi

say ""
say "## 남긴 것 (운전석 판단)"
say "- NC1/NC2/NC4: 결함 주입 시험 빌드 필요 — 깨어서 한다"
say "- M1/M2 수치를 대장에 옮기기(Γ4)"
say "- ★새 사실 (d): 덱이 deposition_cmfgen.csv 를 싣고 런처가 LUMINA_DEPOSITION_FILE 로"
say "  가리키는데 **CPU 바이너리는 그 변수를 읽지 않는다**(lumina_cuda.cu 에만 있다)."
say "  ⟹ CPU 는 MC·DET 둘 다 내부 Bateman.  의도된 CMFGEN 침착이 읽히지 않는다."
say "  이 단은 그것을 **보이게** 만들 뿐이다(provenance 도장).  소비자를 바꾸면 값이 바뀌므로"
say "  Γ2 위반 — **다음 단**이다."
say ""
say "=== NIGHT DONE ==="
