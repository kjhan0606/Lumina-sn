#!/usr/bin/env python3
"""C9 anti-drift — `src/legacy_knob_registry.h` 가 실제 사이트와 일치하는지 기계 대조.

레지스트리를 만든 이유는 목록을 합치는 것이 아니라 **불일치를 보이게 만드는 것**이다.
그러려면 레지스트리 자신이 실물과 어긋나지 않아야 한다 — 눈으로 옮겨 적으면 어긋난다.
(실제로 첫 작성에서 element_wide 16종 중 3종을 빠뜨렸다: STAGE4_BK_CAP · SUPER_CUTOFF ·
TIMEDEP_ION.  이 스크립트는 그 실수를 재발 불가로 만든다.)

세 사이트를 소스에서 직접 파싱해 레지스트리와 대조한다:
  P  src/lumina_plasma.c        forbidden_population_knobs[]    (강제 FATAL)
  E  src/lumina_element_wide.c  ew_guard_config_count()         (관측만)
  S  src/seed_capability.c      obsolete[]                      (강제 FATAL)

rc=0 일치 · rc=1 불일치(어느 쪽이 더/덜 가졌는지 출력).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def block(path: Path, start_pat: str, end_pat: str = r"\};") -> set[str]:
    txt = path.read_text()
    m = re.search(start_pat, txt)
    if not m:
        return set()
    tail = txt[m.start():]
    e = re.search(end_pat, tail)
    seg = tail[: e.start()] if e else tail
    return set(re.findall(r'"(LUMINA_[A-Z0-9_]+)"', seg))


def a2_17_obsolete() -> set[str]:
    """A2-17 폐기 스칼라 목록.

    ★문자 창(±N자)으로 잡으면 근처 코드가 바뀔 때마다 **오탐**이 난다.
    2026-08-07 에 실제로 그랬다: 같은 파일에 env 스캔을 넣자 그 안의
    `getenv("LUMINA_ENV_STRICT")` 가 창에 들어와 DRIFT 로 보고됐다.
    오탐은 검증기의 신뢰를 깎으므로 **배열 선언~닫는 중괄호**로 정확히 자른다.
    """
    txt = (ROOT / "src/lumina_atomic.c").read_text()
    out: set[str] = set()
    m = re.search(r"retired_scalar_options\[\]\s*=\s*\{", txt)
    if m:
        seg = txt[m.end():]
        end = seg.find("}")
        out |= set(re.findall(r'"(LUMINA_[A-Z0-9_]+)"', seg[: end if end > 0 else 0]))
    # 배열 밖의 개별 검사(값 조건부)도 사이트의 일부다
    if re.search(r'getenv\("LUMINA_CMF_EPAY"\)', txt):
        out.add("LUMINA_CMF_EPAY")
    return out


def registry() -> dict[str, tuple[str, str]]:
    txt = (ROOT / "src/legacy_knob_registry.h").read_text()
    out = {}
    for name, disp, enf, obs in re.findall(
            r'X\("(LUMINA_[A-Z0-9_]+)",\s*(LK_\w+),\s*"([PSA\-]+)",\s*"([E\-]+)"\)', txt):
        out[name] = (enf, obs)
    return out


def main() -> int:
    reg = registry()
    sites = {
        "P": block(ROOT / "src/lumina_plasma.c", r"forbidden_population_knobs\[\]"),
        "E": block(ROOT / "src/lumina_element_wide.c", r"ew_guard_config_count\(void\)"),
        "S": block(ROOT / "src/seed_capability.c", r"static const char \*obsolete\[\]"),
        # ★A: A2-17 폐기 스칼라. 첫 작성이 이 사이트를 통째로 놓쳤다 (T3 실패로 발견)
        "A": a2_17_obsolete(),
    }
    reg_by_site = {k: {n for n, (e, o) in reg.items() if k in (e + o)}
                   for k in ("P", "E", "S", "A")}

    rc = 0
    print(f"registry entries: {len(reg)}")
    for site, actual in sites.items():
        expect = reg_by_site[site]
        missing = sorted(actual - expect)      # 소스엔 있는데 레지스트리에 없다
        extra = sorted(expect - actual)        # 레지스트리엔 있는데 소스에 없다
        status = "OK" if not missing and not extra else "DRIFT"
        print(f"  site {site}: source={len(actual):2d} registry={len(expect):2d}  {status}")
        if missing:
            print(f"    ★레지스트리 누락: {missing}")
            rc = 1
        if extra:
            print(f"    ★레지스트리 과잉: {extra}")
            rc = 1

    # 부채 자체를 눈에 보이게 출력한다 — 이것이 레지스트리의 목적이다
    obs_only = sorted(n for n, (e, o) in reg.items() if e == "-" and o == "E")
    enf_only = sorted(n for n, (e, o) in reg.items() if e == "P" and o == "-")
    print(f"\n관측만 되고 강제되지 않는 노브 (C8 fail-open): {len(obs_only)}")
    for n in obs_only:
        print(f"    {n}")
    print(f"강제되나 관측 목록에 없는 노브 (element-wide 진단 과소보고): {len(enf_only)}")
    for n in enf_only:
        print(f"    {n}")
    print("\nLEGACY_KNOB_REGISTRY " + ("PASS" if rc == 0 else "DRIFT"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
