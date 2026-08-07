#!/usr/bin/env bash
# Γ단 런타임 게이트 — Γ2-b(바이트 parity) · Γ3(발행 위상) · NC3(정당한 0).
# A(R7 판정)·B(적용·빌드)·Γ2-a(수식 불변)는 이미 끝났다.
#
# ★팔은 T3_LANE 으로 지정하고, **로그의 lane= 로 확인**한다(어젯밤 3번 당했다).
set -u
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
OUT=$R/validation/gamma
V=$OUT/GAMMA_GATES.md
mkdir -p "$OUT"; : > "$V"
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$V"; }
DECK=data/tardis_reference_toy06_19p48d_sivcaiv_active

run () {  # $1=덱 $2=로그이름 $3=LANE $4=패킷
  ssh lageunha "cd $R && timeout 5400 env T3_DECK=$1 T3_LANE=$3 PKTS=${4:-800} NITER=1 OMP=32 \
      bash scripts/t3_cpu_repro.sh > /tmp/kjhan_$2.log 2>&1; echo rc=\$?" 2>&1 | tail -1
  ssh lageunha "cp /tmp/kjhan_$2.log $OUT/$2.log" 2>/dev/null
  local got; got=$(grep -m1 -oE 'lane=(MC|DET)' "$OUT/$2.log" 2>/dev/null | cut -d= -f2)
  if [ -n "$got" ] && [ "$got" != "$3" ]; then
    { echo "★팔 불일치: 요청=$3 실제=$got"; } | tee -a "$V" >&2; touch "$OUT/.mismatch"
  fi
}

say "# Γ단 런타임 게이트 — $(date '+%Y-%m-%d %H:%M')"
say ""
say "postGamma sha=$(sha256sum "$R/lumina" | cut -c1-12)  preGamma sha=$(sha256sum "$R/lumina.preGamma" | cut -c1-12)"

# ── Γ2-b 바이트 parity ─────────────────────────────────────────
say ""
say "## Γ2-b 바이트-parity (MC 마이크로픽스처 100pkt)"
say "계약만 추가했다면 새 진단을 뺀 전 출력이 동일해야 한다."
cd "$R"
cp lumina lumina.postGamma
cp lumina.preGamma lumina;  rm -f "$OUT/.mismatch"
say "  preGamma..."; rc=$(run "$DECK" g_par_pre MC 100); say "    $rc"
cp lumina.postGamma lumina
say "  postGamma..."; rc=$(run "$DECK" g_par_post MC 100); say "    $rc"
[ -e "$OUT/.mismatch" ] && { say "★팔 불일치 — 중단"; exit 5; }

strip () { grep -vE '\[GAMMA|host=|sha=' "$1" | sed 's/[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}//g'; }
strip "$OUT/g_par_pre.log"  > /tmp/claude-10396/gpre.txt
strip "$OUT/g_par_post.log" > /tmp/claude-10396/gpost.txt
if diff -q /tmp/claude-10396/gpre.txt /tmp/claude-10396/gpost.txt >/dev/null; then
  say "  **Γ2-b PASS** — 값이 하나도 안 바뀌었다."
else
  say "  **Γ2-b FAIL** — 차이:"
  say '```'; diff /tmp/claude-10396/gpre.txt /tmp/claude-10396/gpost.txt | head -25 | tee -a "$V"; say '```'
fi
say ""
say "### 새로 생긴 감마 관측"
say '```'
grep -E '\[GAMMA' "$OUT/g_par_post.log" | head -8 | tee -a "$V"
say '```'

# ── Γ3 발행 위상 (DET) ─────────────────────────────────────────
say ""
say "## Γ3 발행이 a208 **앞**에 서는가 (DET)"
say "⚠원문은 DET 가 A2-10 까지 가기를 요구하나 그것은 R6 착지 후다 — 여기서는 위상만."
rm -f "$OUT/.mismatch"; rc=$(run "$DECK" g_det DET 800); say "  $rc"
say '```'
grep -E '\[GAMMA|\[R7\]\[PHASE\]|\[A2-0[89]\]\[|\[A2-10\]\[|EXIT=' "$OUT/g_det.log" | head -12 | tee -a "$V"
say '```'
gp=$(grep -n 'GAMMA..PUBLISHED' "$OUT/g_det.log" | head -1 | cut -d: -f1)
ap=$(grep -n 'phase=a208'       "$OUT/g_det.log" | head -1 | cut -d: -f1)
if [ -n "$gp" ] && [ -n "$ap" ] && [ "$gp" -lt "$ap" ]; then
  say "  **Γ3(위상) PASS** — 발행 $gp행 < a208 $ap행"
else say "  **Γ3(위상) FAIL** — 발행=${gp:-없음} a208=${ap:-없음}"; fi

# ── NC3 정당한 0 ───────────────────────────────────────────────
say ""
say "## ★NC3 — Ni·Co 존비 0 인 덱은 **통과**해야 한다"
say "없으면 게이트가 '0 이면 무조건 차단' 인 잘못된 게이트가 된다."
ND=$R/data/tardis_reference_toy06_19p48d_nicozero
if [ ! -d "$ND" ]; then
  mkdir -p "$ND"
  for f in "$R/$DECK"/*; do b=$(basename "$f"); [ "$b" = "abundances.csv" ] && continue; ln -sf "$f" "$ND/$b"; done
  python3 - "$R/$DECK/abundances.csv" "$ND/abundances.csv" <<'PY' | tee -a "$V"
import sys, csv
rows = list(csv.reader(open(sys.argv[1])))
hdr, body = rows[0], rows[1:]
data = {int(r[0]): [float(x) for x in r[1:]] for r in body}
ns = len(hdr) - 1
for s in range(ns):
    removed = sum(data[z][s] for z in (27, 28) if z in data)
    rest = sum(data[z][s] for z in data if z not in (27, 28))
    for z in (27, 28):
        if z in data: data[z][s] = 0.0
    if rest > 0:
        f = (rest + removed) / rest
        for z in data:
            if z not in (27, 28): data[z][s] *= f
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(hdr)
    for z in sorted(data): w.writerow([z] + [repr(v) for v in data[z]])
for s in range(ns):
    tot = sum(data[z][s] for z in data)
    assert abs(tot - 1.0) < 1e-9, f"shell {s} sum={tot}"
    assert data[27][s] == 0.0 and data[28][s] == 0.0
print("  NC3 덱 검증: Ni/Co = 0, 전 셸 합 = 1")
PY
fi
rm -f "$OUT/.mismatch"
rc=$(run "data/tardis_reference_toy06_19p48d_nicozero" g_nc3 MC 100); say "  $rc"
say '```'
grep -E '\[GAMMA|RADEQ_GAMMA_UNPUBLISHED|\[A2-10\]\[|EXIT=' "$OUT/g_nc3.log" | head -10 | tee -a "$V"
say '```'
pub=$(grep -c 'GAMMA..PUBLISHED'        "$OUT/g_nc3.log" || true)
unp=$(grep -c 'RADEQ_GAMMA_UNPUBLISHED' "$OUT/g_nc3.log" || true)
if [ "$pub" -gt 0 ] && [ "$unp" -eq 0 ]; then
  say "  **NC3 PASS** — 정확히 0 인데 발행되고 런이 진행됐다."
else say "  **NC3 FAIL** — 발행=$pub 차단=$unp"; fi

say ""
say "=== GATES DONE ==="
