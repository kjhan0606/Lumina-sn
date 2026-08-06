#!/usr/bin/env python3
"""노브가 감싸는 블록이 **무엇을 하는지** 로 스크랩 안전성을 가른다.

앞선 삼중 분류는 77건 중 64건이 "모호" 로 떨어졌다 — 블록 추출기가 조잡해서
`if (getenv(...)) { ... }` 형태만 잡고 나머지를 놓쳤기 때문이다.  여기서는
**getenv 가 쓰인 값이 흘러가는 곳**을 보고 판정한다.

부류(스크랩 안전도 순):
  DIAG-ONLY   ON 경로가 출력·덤프만 한다(printf/fprintf/fopen/fwrite) →
              계측 노브.  user 지시상 창고 대상.
  ADDS-PHYS   ON 경로가 물리 상태(배열·항·자료)를 **더한다** → S-PHYS.
              OFF 는 그것을 결여하므로 스위치가 아니라 덧붙임.  스크랩 금지.
  GUARD       ON/설정 시 **거부·차단**한다(BLOCKED/return -1/abort) → S-GUARD.
              스크랩하면 가드가 사라진다.
  PARAM       수치 파라미터를 바꾼다(기본값 존재) → 값 판정 필요.
  UNKNOWN     위 어느 것도 아니다 → 사람이 읽는다.

★★이 도구는 **음성대조를 통과하지 못했다.**  스크랩을 승인하는 데 쓰지 마라.

음성대조(2026-08-07): `lumina_atomic.c` 9건은 정독으로 **전부 스크랩 금지**로 판정된 것들이다.
이 도구를 돌린 결과:

    ADDS-PHYS 3 (맞음) · GUARD 2 (맞음) · UNKNOWN 2 · **DIAG-ONLY 2 (틀림)**

`DIAG-ONLY` 로 나온 둘은 실제로는 물리 수리(`FIX_BF_CONTINUUM_EVENT`)와 그 입력
(`SPINGATE_MULT`)이다 — 이 신호를 믿고 쓸었으면 **지우면 안 되는 것을 지웠다**.
가장 중요한 `TOPSTAGE_ANCHOR`(최상단 이온 바닥준위 주입)는 `UNKNOWN` 으로 빠졌는데,
`if (!(e && atoi(e) != 0)) return;` 라는 **early-return 가드** 관용구를 블록으로 잡지 못했기 때문이다.

⟹ 용도는 **읽는 순서를 정하는 것** 하나뿐이다(ADDS-PHYS·GUARD 부터 읽는다).
   스크랩 승인은 **정독만** 할 수 있다.  기계 분류로 일괄 sweep 하지 않는다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIAG = re.compile(r'\b(printf|fprintf|fopen|fwrite|fputs|fclose|snprintf)\b')
GUARDW = re.compile(r'\b(BLOCKED|abort|exit|return\s+-1|return\s+EXIT_FAILURE|FATAL)\b')
# 물리 상태로 흘러가는 대입: 배열 원소 또는 구조체 필드에 쓰기
PHYSW = re.compile(
    r'\b(atom|plasma|opacity|nlte|geo|bf|em|op)\s*->\s*[A-Za-z_]+\s*(\[|=)'
    r'|\b(malloc|calloc|realloc|memcpy)\b')


def region(lines: list[str], i: int, span: int = 60) -> str:
    """getenv 사이트에서 시작해 중괄호 균형이 닫히거나 span 줄까지."""
    depth, started, out = 0, False, []
    for j in range(i, min(i + span, len(lines))):
        out.append(lines[j])
        depth += lines[j].count("{") - lines[j].count("}")
        if "{" in lines[j]:
            started = True
        if started and depth <= 0:
            break
        if not started and lines[j].rstrip().endswith(";") and j > i:
            break
    return "\n".join(out)


def classify_file(path: Path, names: set[str]) -> dict[str, tuple[str, int]]:
    lines = path.read_text(errors="ignore").splitlines()
    out: dict[str, tuple[str, int]] = {}
    for i, ln in enumerate(lines):
        for m in re.finditer(r'getenv\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)', ln):
            n = m.group(1)
            if names and n not in names:
                continue
            r = region(lines, i)
            if GUARDW.search(r):
                k = "GUARD"
            elif PHYSW.search(r):
                k = "ADDS-PHYS"
            elif DIAG.search(r):
                k = "DIAG-ONLY"
            elif re.search(r'=\s*(atof|atoi|strtod)\s*\(', r):
                k = "PARAM"
            else:
                k = "UNKNOWN"
            prev = out.get(n)
            # 한 노브가 여러 곳에 있으면 **가장 보수적인** 부류를 택한다
            rank = {"ADDS-PHYS": 0, "GUARD": 1, "PARAM": 2, "UNKNOWN": 3, "DIAG-ONLY": 4}
            if prev is None or rank[k] < rank[prev[0]]:
                out[n] = (k, i + 1)
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: knob_block_classify.py <src file> [name ...]")
        return 2
    p = ROOT / argv[1] if not argv[1].startswith("/") else Path(argv[1])
    names = set(argv[2:])
    res = classify_file(p, names)
    buckets: dict[str, list[str]] = {}
    for n, (k, ln) in sorted(res.items()):
        buckets.setdefault(k, []).append(f"{n}  (:{ln})")
    for k in ("ADDS-PHYS", "GUARD", "PARAM", "UNKNOWN", "DIAG-ONLY"):
        v = buckets.get(k, [])
        print(f"\n{k}  {len(v)}")
        for x in v:
            print(f"    {x}")
    print(f"\n합계 {len(res)}  —  ★ADDS-PHYS·GUARD 는 스크랩 금지, 나머지도 정독 확인")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
