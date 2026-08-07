#!/usr/bin/env python3
"""**코드 검사** — 최상단 이온 catalog 자료가 물리적으로 성립하는가.

분담 개정10(2026-08-07): 코딩은 Codex, **검사는 운전석**.  이 파일이 그 검사다.
Codex 가 낸 추출기의 산출물을 **독립적으로** 검증한다 — 추출기의 코드를 읽는 것이 아니라
**결과가 물리를 만족하는지**만 본다.

오늘 이 검사가 없어서 놓친 것:
· `g=0` 인 행이 3,480 중 1,636 — 분배함수가 물리적으로 불가능해진다
· 전이 자료(f, A)가 준위 자료(g, E)로 기록됨 — Fe VII 1195행 중 1041행
· 그 자료로 계산한 Z 가 런타임에 1e-65 로 나왔고, 결함 주입 음성대조가 **통과**해 버렸다

검사 항목(전부 fail-closed):
  C1 g 는 **양의 정수**다 (g = 2J+1)
  C2 이온마다 바닥 준위가 정확히 하나이고 E=0, g>=1
  C3 E 가 단조 비감소는 아니어도 되지만 **음수는 없다**
  C4 ★Z(T) >= g0 — 수학적 하한.  위반하면 자료가 깨진 것이다
  C5 준위 수가 이온마다 1 이상이고, 두 산출 CSV 의 이온 집합이 일치한다
  C6 등전자 일치: 전자 수가 같으면 g0 도 같다
  C7 g 값이 그 이온의 전자 수로 설명 가능한 범위(보수적 상한)

rc=0 전부 통과 · rc=2 하나라도 위반.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUND = ROOT / "data/atomic/topion_ground_levels.csv"
LEVELS = ROOT / "data/atomic/topion_levels.csv"
K_CM = 0.695034800          # cm^-1 per K
G_MAX = 200.0               # 바닥 준위 하나의 통계중량 보수적 상한


def main() -> int:
    if not (GROUND.is_file() and LEVELS.is_file()):
        print(f"FAIL: 산출 CSV 가 없다 ({GROUND.name} / {LEVELS.name})")
        return 2
    ground = {(int(r["Z"]), int(r["ion_stage_0based"])): r
              for r in csv.DictReader(GROUND.open())}
    lv = defaultdict(list)
    for r in csv.DictReader(LEVELS.open()):
        lv[(int(r["Z"]), int(r["ion_stage_0based"]))].append(
            (float(r["E_cm-1"]), float(r["g"]), r["label"]))

    bad = []

    # C5 이온 집합 일치
    if set(ground) != set(lv):
        bad.append(f"C5 이온 집합 불일치: ground={len(ground)} levels={len(lv)}")

    for key in sorted(set(ground) & set(lv)):
        z, st = key
        rows = lv[key]
        label = rows[0][2]
        n_elec = z - st

        # C1 g 는 양의 정수
        for e, g, _ in rows:
            if not (g >= 1.0 and abs(g - round(g)) < 1e-9):
                bad.append(f"C1 {label}: g={g} — 양의 정수가 아니다 (g=2J+1)")
                break
        # C3 음의 에너지 없음
        if any(e < -1e-6 for e, _, _ in rows):
            bad.append(f"C3 {label}: 음의 E 가 있다")
        # C2 바닥 준위
        e0 = min(e for e, _, _ in rows)
        if abs(e0) > 1e-6:
            bad.append(f"C2 {label}: 최저 E={e0} (0 이어야 한다)")
        g0_rows = [g for e, g, _ in rows if abs(e - e0) < 1e-6]
        g0_decl = float(ground[key]["g0"])
        if not g0_rows or abs(g0_rows[0] - g0_decl) > 1e-9:
            bad.append(f"C2 {label}: 바닥 g 불일치 levels={g0_rows[:1]} ground={g0_decl}")
        # C7 g 상한
        if g0_decl > G_MAX or n_elec <= 0:
            bad.append(f"C7 {label}: g0={g0_decl} n_elec={n_elec} — 범위 밖")

        # C4 ★Z(T) >= g0  (수학적 하한)
        for T in (5000.0, 8000.0, 10000.0, 15000.0, 30000.0):
            Zt = sum(g * math.exp(-(e - e0) / (K_CM * T)) for e, g, _ in rows)
            if Zt < g0_decl - 1e-9:
                bad.append(f"C4 {label}: Z({T:.0f}K)={Zt:.3e} < g0={g0_decl} — 자료가 깨졌다")
                break

    # ★C8 vintage 일관성 — 2026-08-07 추가.
    # 운전석 파서가 `sorted(glob)[0]`(알파벳 첫 번째)로 골라 S VI 만 1999 년판이 섞였다.
    # 덱은 `_pick_latest`/링크 규칙을 쓰므로, catalog 만 다른 vintage 면 **vintage 혼입**이다.
    # C1~C7 은 이것을 못 잡는다 — 값 자체는 물리적으로 정상이기 때문이다.
    vint = defaultdict(set)
    for r in ground.values():
        src = r.get("source_file", "")
        if "/atomic/" in src:                       # CMFGEN 원본만 대상
            parts = Path(src).parts
            vint["CMFGEN"].add(parts[-2])           # 날짜 디렉토리
    for prov, dates in vint.items():
        if len(dates) > 1:
            bad.append(f"C8 {prov} vintage 혼입: {sorted(dates)} — 덱과 같은 규칙으로 골라야 한다")

    # C6 등전자 일치
    seq = defaultdict(set)
    for (z, st), r in ground.items():
        seq[z - st].add(float(r["g0"]))
    for n, gs in sorted(seq.items()):
        if len(gs) > 1:
            bad.append(f"C6 {n}전자: g0 가 갈린다 {sorted(gs)}")

    print(f"이온 {len(ground)} · 준위 {sum(len(v) for v in lv.values())}")
    for b in bad:
        print(f"  **{b}**")
    print(f"\nTOPION_CATALOG violations={len(bad)} verdict={'PASS' if not bad else 'FAIL'}")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
