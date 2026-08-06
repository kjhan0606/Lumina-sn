#!/usr/bin/env python3
"""Makefile 의 HEADERS 가 src 가 실제로 #include 하는 헤더 전집과 일치하는가.

빠진 헤더는 **바뀌어도 리빌드를 일으키지 않는다**.  증상이 고약하다: 소스를 고치고
`make` 를 돌리면 "up to date" 라 하고, 런은 옛 코드로 돈다 — 오류 없이 옛 결과가 나온다.

2026-08-07 T3 잡 226529 진단 중 발각.  `src/env_universe.h` 를 483→496 으로 재생성한 뒤
리빌드를 시켰는데 make 가 "up to date" 를 냈다.  세어 보니 HEADERS 는 12개인데 실제
include 되는 src 헤더는 21개였고, 빠진 9개에 **계약 헤더 3종**이 있었다
(gpu_radiation_field_contract · gpu_physics_kernels · lumina_cmfgen).

rc=0 일치 · rc=2 불일치.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def declared() -> set[str]:
    mk = (ROOT / "Makefile").read_text()
    # 줄이음(\) 을 편 뒤 HEADERS 대입만 읽는다
    flat = re.sub(r"\\\n\s*", " ", mk)
    m = re.search(r"^HEADERS\s*=\s*(.*)$", flat, re.M)
    if not m:
        raise SystemExit("Makefile 에 HEADERS 대입이 없다")
    return {t for t in m.group(1).split() if t.endswith(".h")}


def included() -> set[str]:
    out: set[str] = set()
    for p in list((ROOT / "src").glob("*.c")) + list((ROOT / "src").glob("*.cu")) \
            + list((ROOT / "src").glob("*.h")):
        for inc in re.findall(r'#\s*include\s*"([^"]+)"', p.read_text(errors="ignore")):
            q = ROOT / "src" / pathlib.Path(inc).name
            if q.is_file():
                out.add(f"src/{q.name}")
    return out


def main() -> int:
    d, i = declared(), included()
    missing, extra = sorted(i - d), sorted(d - i)
    for x in missing:
        print(f"  MISSING  {x}   (바뀌어도 리빌드 안 됨)")
    for x in extra:
        print(f"  STALE    {x}   (아무도 include 하지 않는다)")
    ok = not missing            # STALE 은 무해하므로 실패시키지 않는다
    print(f"\nMAKEFILE_HEADERS declared={len(d)} included={len(i)} "
          f"missing={len(missing)} stale={len(extra)} verdict={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
