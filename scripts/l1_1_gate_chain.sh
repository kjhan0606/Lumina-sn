#!/usr/bin/env bash
# L1-1 게이트 자율 체인 — 각 단계가 끝나면 **사람 개입 없이** 다음 단계로 들어간다.
# user 상설 지시(2026-08-07): "런이 시작하면 인계철선을 걸어두고 종료신호가 오면 다음 스텝으로 바로 진입."
#
# 사전등록 docs/RUNG_L1_1_BOOTSTRAP_SUPPLIER.md 의 게이트:
#   G1 iter=0 첫 수송 도달        G4 seed 비유한 주입 -> fail-closed
#   G5 재진입 거부                G6 덱 3종 동일 통과
#   G7 전하합 섭동 -> 잔차 검출   O7 population m=1 provenance
#   + R0 음성대조(Codex 지정): catalog 를 {E=0,g=0} 로 -> POP_INVALID_PARTITION
#
# ★덱 정본을 건드리지 않는다.  G7 은 심볼릭 링크 사본에 abundances 만 바꿔 만든다.
set -u
R=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
OUT=$R/validation/l1_1_gates
mkdir -p "$OUT"
LOG=$OUT/chain.log
: > "$LOG"
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ★리빌드 경합 방지: 체인이 도는 동안 바이너리가 바뀌면 가드(rc=70)가 걸린다.
# 시작 시 sha 를 고정하고, 매 런 전에 같은지 확인한다.
BIN_SHA=$(sha256sum "$R/lumina" | cut -d" " -f1)
say "바이너리 고정 sha=${BIN_SHA:0:12}"

run_deck () {  # $1=덱경로 $2=로그이름 $3=패킷
  cur=$(sha256sum "$R/lumina" | cut -d" " -f1)
  if [ "$cur" != "$BIN_SHA" ]; then
    say "  ⚠ 바이너리가 체인 도중 바뀌었다 (${BIN_SHA:0:12} -> ${cur:0:12}) — 새 sha 로 계속"
    BIN_SHA=$cur
  fi
  ssh lageunha "cd $R && timeout 5400 env T3_DECK=$1 PKTS=${3:-800} NITER=1 OMP=32 \
      bash scripts/t3_cpu_repro.sh > /tmp/kjhan_$2.log 2>&1; echo rc=\$?" 2>&1 | tail -1
  ssh lageunha "cp /tmp/kjhan_$2.log $OUT/$2.log" 2>/dev/null
}

# ---------- G6: 덱 3종 ----------
say "=== G6 덱 3종 ==="
for d in _sivcaiv_active _ophys _jnu4; do
  say "  런 시작: $d"
  rc=$(run_deck "data/tardis_reference_toy06_19p48d$d" "g6$d" 800)
  say "  종료 $d $rc"
  grep -E '\[R0\]|reservoir|BOOTSTRAP|charge-cons|K-FRESH|A2-10..PRE|FATAL|EXIT=' \
       "$OUT/g6$d.log" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
done

# ---------- G4/G5: 발행·창 음성대조 배터리 ----------
say "=== G4/G5 음성대조 배터리 ==="
( cd "$R" && ./selftest_seed_te_publish; echo "seed_rc=$?";
              ./selftest_bootstrap_window; echo "window_rc=$?" ) 2>&1 \
  | grep -E 'PASS|FAIL|rc=' | sed 's/^/    /' | tee -a "$LOG"

# ---------- R0 음성대조: catalog 결함 주입 ----------
say "=== R0 음성대조 (catalog {E=0,g=0}) ==="
CSV=$R/data/atomic/topion_levels.csv
cp "$CSV" /tmp/claude-10396/topion_levels.keep.csv
python3 - <<'PY'
import csv
src='/tmp/claude-10396/topion_levels.keep.csv'
rows=list(csv.DictReader(open(src)))
out=[r for r in rows if r['label']!='Fe VII']
bad=dict(rows[0]); bad.update({'Z':'26','ion_stage_0based':'6','label':'Fe VII',
                               'level_index':'1','E_cm-1':'0.000000','g':'0.0'})
out.append(bad)
p='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/topion_levels.csv'
w=csv.DictWriter(open(p,'w',newline=''),fieldnames=list(rows[0].keys()))
w.writeheader(); w.writerows(out)
PY
rc=$(run_deck "data/tardis_reference_toy06_19p48d_sivcaiv_active" "r0neg" 400)
say "  종료 $rc"
grep -E 'partition build failed|POP_INVALID_PARTITION|POP_ATOMIC_MISSING|reservoir|EXIT=' \
     "$OUT/r0neg.log" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
cp /tmp/claude-10396/topion_levels.keep.csv "$CSV"
say "  catalog 복원"

# ---------- G7: 전하합 섭동 ----------
say "=== G7 전하합 섭동 (덱 정본 불변 — 링크 사본에 abundances 만 교체) ==="
PERT=$R/data/tardis_reference_toy06_19p48d_sivcaiv_active_G7PERT
rm -rf "$PERT"; mkdir -p "$PERT"
SRC=$R/data/tardis_reference_toy06_19p48d_sivcaiv_active
for f in "$SRC"/*; do
  b=$(basename "$f")
  [ "$b" = "abundances.csv" ] && continue
  [ "$b" = "quarantine" ] && continue
  ln -s "$f" "$PERT/$b"
done
python3 - <<PY
import csv
src="$SRC/abundances.csv"; dst="$PERT/abundances.csv"
rows=list(csv.reader(open(src)))
# 첫 원소 행의 값들을 1.5배 — 전하합이 깨지도록
for i,r in enumerate(rows):
    if i==1 and len(r)>1:
        rows[i]=[r[0]]+[str(float(x)*1.5) for x in r[1:]]
csv.writer(open(dst,'w',newline='')).writerows(rows)
print("  abundances 첫 원소 x1.5 주입")
PY
rc=$(run_deck "data/tardis_reference_toy06_19p48d_sivcaiv_active_G7PERT" "g7pert" 400)
say "  종료 $rc"
grep -E 'charge-conservation|n_e\]\[FATAL\]|reservoir|EXIT=' "$OUT/g7pert.log" 2>/dev/null \
  | sed 's/^/    /' | tee -a "$LOG"
rm -rf "$PERT"
say "  섭동 덱 제거"

say "=== CHAIN COMPLETE ==="
