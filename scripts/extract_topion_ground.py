#!/usr/bin/env python3
"""R0 자료 추출 — 최상단 이온 15종의 바닥준위 (E₀, g₀).

로더는 전리에너지 n 개 → population n+1 개를 만들므로 원소마다 최상단 population 의
**속박준위가 0 개**다(실측 15/74). 기준 배선 ARTIS 는 그 이온에 준위 **1 개**(바닥)를 준다
(`input.cc:1226` SINGLE_LEVEL_TOP_ION) ⟹ Z_top = g₀ 이지 1 이 아니다.

★손으로 옮기지 않는다. 파일에서 기계로 읽고 fail-closed 로 검사한다.

## 읽기 규약 (Codex 확정)

CMFGEN osc: 헤더의 `!Number of energy levels` 뒤 공백줄을 지나 준위 레코드가 온다.
레코드는 `name  g  E(cm^-1)  10^15Hz  eV  Lam(A)  ID  ARAD  C4  C6`.
**첫 레코드의 ID 가 1 이고 E 가 0 인지 검사한 뒤** 그 레코드의 g 를 쓴다.
g₀ 는 **첫 미세구조 준위의 통계중량**이지 바닥 항 전체의 합이 아니다 — 섞으면 틀린다.

⚠파일 제목 줄은 신뢰하지 않는다: `CARB/IV/19apr23/osc_data` 의 제목은 "Ca IV" 인데
실제 자료는 C IV 다(바닥 `2s_2Se` = Li-like 탄소). 제목이 아니라 **경로**가 이온을 정한다.

Cloudy/Stout `.nrg`: `ID<TAB>E(cm^-1)<TAB>g<TAB>"term"`. 첫 행 ID=1, E=0 검사.
출처는 파일이 NIST 를 명시한다.

산출: data/atomic/topion_ground_levels.csv (+ provenance 열)
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMF = Path("/gpfs/kjhan/cmfgen_21jun23/atomic")
STOUT = Path("/gpfs/kjhan/cloudy-master/data/stout")

# (Z, 0-based top stage, 표기, CMFGEN 경로 or None, Stout 경로 or None)
TARGETS = [
    (6,  3, "C IV",   CMF / "CARB/IV",  None),
    (8,  3, "O IV",   CMF / "OXY/IV",   None),
    (12, 3, "Mg IV",  CMF / "MG/IV",    None),
    (13, 4, "Al V",   CMF / "AL/V",     None),
    (14, 5, "Si VI",  CMF / "SIL/VI",   None),
    (16, 5, "S VI",   CMF / "SUL/VI",   None),
    (20, 5, "Ca VI",  CMF / "CA/VI",    None),
    (21, 3, "Sc IV",  None, STOUT / "sc/sc_4/sc_4.nrg"),
    (22, 4, "Ti V",   None, STOUT / "ti/ti_5/ti_5.nrg"),
    (23, 1, "V II",   None, STOUT / "v/v_2/v_2.nrg"),
    (24, 4, "Cr V",   CMF / "CHRO/V",   None),
    (25, 3, "Mn IV",  CMF / "MAN/IV",   None),
    (26, 6, "Fe VII", CMF / "FE/VII",   None),
    (27, 6, "Co VII", CMF / "COB/VII",  None),
    (28, 6, "Ni VII", CMF / "NICK/VII", None),
]

def parse_level_record(line: str):
    """열 개수가 판본마다 다르다(19apr23 는 10열, 08jul99/10apr99 는 6열).
    그래서 열 수에 기대지 않는다: 이름·g·E 는 앞에서, ID 는 **정수 토큰**으로 잡는다.
    반환 (g, E_cm-1, ID) 또는 None."""
    tok = line.split()
    if len(tok) < 4:
        return None
    name = tok[0]
    # 준위명은 `2s_2Se[1/2]` 처럼 숫자로 시작할 수도, `13___` 처럼 **문자가 아예 없을 수도**
    # 있다(CMFGEN 의 병합 준위).  그래서 "문자를 포함한다" 는 틀린 판별식이다 —
    # 2026-08-07 에 그것 때문에 C IV 64개 중 18개를 놓쳤고, 파서가 모자란 수를 채우려
    # **전이 블록**까지 읽어 f 를 g 로, A 를 E 로 기록했다.
    # 옳은 판별식: **실수로 파싱되지 않는다**(헤더 줄은 숫자이므로 여전히 걸러진다).
    try:
        float(name)
        return None
    except ValueError:
        pass
    try:
        g = float(tok[1]); e = float(tok[2])
    except ValueError:
        return None
    lid = None
    for x in tok[3:]:
        # CMFGEN 은 준위 ID 에 **음수**를 쓰기도 한다(O IV: `... 1.781E+03  -7`).
        # 2026-08-07 에 `\d+` 만 받아 324개 중 78개만 파싱했고, 모자란 수를 채우려
        # 파서가 전이 블록으로 넘어갔다.
        if re.fullmatch(r"-?\d+", x):
            lid = int(x); break
    if lid is None:
        return None
    return g, e, lid


def read_cmfgen(dirpath: Path) -> tuple[float, float, str]:
    """CMFGEN osc 에서 첫 준위의 (E₀ cm^-1, g₀) 를 읽는다. fail-closed."""
    cands = sorted(dirpath.glob("*/osc*")) + sorted(dirpath.glob("*/*osc*"))
    if not cands:
        raise SystemExit(f"FAIL {dirpath}: osc 파일 없음")
    f = cands[0]
    n_declared = None
    for line in f.read_text(errors="ignore").splitlines():
        if "!Number of energy levels" in line:
            n_declared = int(line.split()[0])
            continue
        rec = parse_level_record(line) if n_declared is not None else None
        if rec:
            g, e, lid = rec
            # ID 규약이 판본마다 다르므로(양수·음수) 판별의 무게는 **E0=0 과 g>0** 에 둔다.
            if abs(e) > 1e-6:
                raise SystemExit(f"FAIL {f}: 첫 준위 E={e} (0 이어야 한다)")
            if not (g > 0):
                raise SystemExit(f"FAIL {f}: 첫 준위 g={g}")
            return e, g, str(f)
    raise SystemExit(f"FAIL {f}: 준위 레코드를 찾지 못함(선언 {n_declared})")


def read_stout(f: Path) -> tuple[float, float, str]:
    for line in f.read_text(errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not parts[0].strip().isdigit():
            continue
        lid, e, g = int(parts[0]), float(parts[1]), float(parts[2])
        if lid != 1:
            raise SystemExit(f"FAIL {f}: 첫 준위 ID={lid}")
        if abs(e) > 1e-6:
            raise SystemExit(f"FAIL {f}: 첫 준위 E={e}")
        if not (g > 0):
            raise SystemExit(f"FAIL {f}: 첫 준위 g={g}")
        return e, g, str(f)
    raise SystemExit(f"FAIL {f}: 준위 레코드 없음")


def all_levels_cmfgen(dirpath: Path) -> list[tuple[float, float]]:
    """Z(T) 계산용 — 선언된 준위 전체를 (E cm^-1, g) 로 읽는다."""
    cands = sorted(dirpath.glob("*/osc*")) + sorted(dirpath.glob("*/*osc*"))
    f = cands[0]
    out, started, n_declared, in_block = [], False, None, False
    for line in f.read_text(errors="ignore").splitlines():
        if "!Number of energy levels" in line:
            n_declared = int(line.split()[0]); started = True; continue
        if not started:
            continue
        rec = parse_level_record(line)
        if rec:
            in_block = True
            g, e, _ = rec
            out.append((e, g))
            if n_declared and len(out) >= n_declared:
                break
        elif in_block:
            # ★블록 경계.  레코드가 시작된 뒤 처음 파싱 실패하는 줄이 준위 블록의 끝이다.
            # 2026-08-07: 이 경계를 안 잡아서 파서가 **전이 블록**으로 넘어갔고
            # f 를 g 로, A 를 E 로 읽었다(Fe VII 1195행 중 1041행이 전이 자료였다).
            # n_declared 상한은 *파싱된 레코드*를 세므로 경계를 지키지 못한다.
            break
    if n_declared and len(out) != n_declared:
        raise SystemExit(f"FAIL {f}: 준위 {len(out)}개만 파싱(선언 {n_declared}) — "
                         f"블록 경계나 레코드 형식이 다르다")
    return out


def all_levels_stout(f: Path) -> list[tuple[float, float]]:
    out = []
    for line in f.read_text(errors="ignore").splitlines():
        p = line.split("\t")
        if len(p) < 3 or not p[0].strip().isdigit():
            continue
        out.append((float(p[1]), float(p[2])))
    return out


def main() -> int:
    rows = []
    for z, stage, label, cmf_dir, stout_f in TARGETS:
        if cmf_dir is not None:
            e, g, src = read_cmfgen(cmf_dir)
            prov = "CMFGEN_21jun23"
        else:
            e, g, src = read_stout(stout_f)
            prov = "CLOUDY_STOUT_NIST"
        rows.append({"Z": z, "ion_stage_0based": stage, "label": label,
                     "E0_cm-1": f"{e:.6f}", "g0": f"{g:.1f}",
                     "provenance": prov, "source_file": src})
        print(f"  {label:<8} Z={z:<3} stage={stage}  E0={e:<10.4f} g0={g:<5.1f} {prov}")

    out = ROOT / "data/atomic/topion_ground_levels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} 이온 -> {out}")
    if len(rows) != 15:
        print("⚠ 15 이온이 아니다 — fail-closed"); return 2

    # ★준위 전체를 실어 Z(T) 를 계산할 수 있게 한다.
    # 단일 g 대입은 최대 80배 틀린다(V II: g_first=1 vs Z(10kK)=80.9) — 실측.
    lv_out = ROOT / "data/atomic/topion_levels.csv"
    with lv_out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Z", "ion_stage_0based", "label", "level_index",
                    "E_cm-1", "g", "provenance"])
        for z, stage, label, cmf_dir, stout_f in TARGETS:
            if cmf_dir is not None:
                lv = all_levels_cmfgen(cmf_dir); prov = "CMFGEN_21jun23"
            else:
                lv = all_levels_stout(stout_f); prov = "CLOUDY_STOUT_NIST"
            for i, (e, g) in enumerate(lv):
                w.writerow([z, stage, label, i + 1, f"{e:.6f}", f"{g:.1f}", prov])
    print(f"준위 전체 -> {lv_out}")

    # ★등전자 일치 검사 — CMFGEN 을 부르지 않고 얻는 독립 검증.
    # 전자 수가 같으면 바닥 항이 같으므로 g₀ 도 같아야 한다.  값을 잘못 읽으면 깨진다.
    # 특히 22전자(Ti-like)는 V II(Cloudy/Stout-NIST) 와 Mn IV·Ni VII(CMFGEN) 이 섞여 있어
    # **서로 다른 출처의 교차 검증**이 된다.
    import collections
    seq = collections.defaultdict(list)
    for r in rows:
        seq[int(r["Z"]) - int(r["ion_stage_0based"])].append((r["label"], float(r["g0"]),
                                                              r["provenance"]))
    bad = 0
    print("\n등전자 일치 검사")
    for n in sorted(seq):
        items = seq[n]
        gs = {g for _, g, _ in items}
        provs = {p for _, _, p in items}
        ok = len(gs) == 1
        if not ok:
            bad += 1
        print(f"  {n:>3}전자  {' · '.join(l for l, _, _ in items):<40} "
              f"g0={sorted(gs)}  {'OK' if ok else '**불일치**'}"
              f"{'  ★출처 교차' if len(provs) > 1 else ''}")
    if bad:
        print(f"⚠ 등전자 불일치 {bad}건 — fail-closed"); return 2
    print("등전자 불일치 0건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
