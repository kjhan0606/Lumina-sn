#!/usr/bin/env python3
"""노브 스크랩의 원자재: env 노브마다 **어디서 읽히고 무엇을 감싸는지** 기계로 만든다.

user 지시(2026-08-07): "비물리와 계측용 노브들을 소스코드들로부터 스크랩해서 창고에
쳐박아놓고, 판정으로 인정된 물리 배선도와 0층에서 확증된 자산을 바탕으로 1층 검사를 시작."

판별 기준 — **생존이 예외, 스크랩이 기본**:
  S-INPUT    입력·경로·자원 지정(덱/파일/바이너리)      → 노브가 아니라 입력
  S-CONTRACT 0층 계약 10건이 요구하는 게이트            → 확증된 자산
  S-PHYS     판정된 물리 배선도에 등장                   → 단, 분기가 아니라 무조건 경로로
  SCRAP      그 외 전부(비물리·계측·화석)               → 창고

이 스크립트는 **분류하지 않는다**.  분류에 필요한 사실만 모은다:
읽는 위치, 감싸는 함수, 리터럴 기본값, 런처가 설정하는지, 판정런이 실제로 넘겼는지.
분류는 사람이 하고 대장에 남긴다(attic/knobs/KNOB_SCRAP_LEDGER.md).

산출: validation/layer1_replan/KNOB_INVENTORY.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = sorted(list((ROOT / "src").glob("*.c")) + list((ROOT / "src").glob("*.cu")))
UNIVERSE = ROOT / "validation/instrumentation_debt/ENV_UNIVERSE.json"

# 함수 머리 추정: 열 0에서 시작하고 '(' 를 포함하며 ';' 로 끝나지 않는 줄
FUNC_HEAD = re.compile(r"^[A-Za-z_][A-Za-z0-9_ \*\t]*\**\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
GETENV = re.compile(r'getenv\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)')


def enclosing_functions(lines: list[str]) -> list[str]:
    """각 행이 속한 함수 이름(추정).  중괄호 깊이를 세어 top-level 함수만 잡는다."""
    out = [""] * len(lines)
    cur, depth = "", 0
    for i, ln in enumerate(lines):
        if depth == 0:
            m = FUNC_HEAD.match(ln)
            if m and not ln.rstrip().endswith(";") and "=" not in ln.split("(")[0]:
                cur = m.group(1)
        out[i] = cur
        depth += ln.count("{") - ln.count("}")
        if depth < 0:
            depth = 0
    return out


def launcher_setters() -> dict[str, int]:
    pat = re.compile(r"\b(LUMINA_[A-Z0-9_]+)\s*=")
    n: dict[str, int] = {}
    for p in (ROOT / "scripts").rglob("*"):
        if not p.is_file():
            continue
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        for name in {m.group(1) for m in pat.finditer(txt)}:
            n[name] = n.get(name, 0) + 1
    return n


def live_in_last_run() -> set[str]:
    """가장 최근 판정런이 실제로 바이너리에 넘긴 것(RESOLVED CONFIG)."""
    out: set[str] = set()
    for cand in sorted(Path("/gpfs/kjhan/lumina/t3").glob("*.out"), reverse=True):
        try:
            txt = cand.read_text(errors="ignore")
        except OSError:
            continue
        if "RESOLVED CONFIG" not in txt:
            continue
        for m in re.finditer(r"^\s+(LUMINA_[A-Z0-9_]+)=", txt, re.M):
            out.add(m.group(1))
        if out:
            break
    return out


def main() -> int:
    setters = launcher_setters()
    live = live_in_last_run()
    inv: dict[str, dict] = {}

    for f in SRC:
        lines = f.read_text(errors="ignore").splitlines()
        funcs = enclosing_functions(lines)
        for i, ln in enumerate(lines):
            for m in GETENV.finditer(ln):
                name = m.group(1)
                e = inv.setdefault(name, {
                    "sites": [], "files": set(), "functions": set(),
                    "launchers": setters.get(name, 0),
                    "live_last_run": name in live,
                })
                e["sites"].append(f"src/{f.name}:{i+1}")
                e["files"].add(f.name)
                e["functions"].add(funcs[i] or "(file scope)")

    for e in inv.values():
        e["files"] = sorted(e["files"])
        e["functions"] = sorted(e["functions"])
        e["n_sites"] = len(e["sites"])

    universe = set(json.loads(UNIVERSE.read_text())["env"]) if UNIVERSE.is_file() else set()
    not_literal = sorted(universe - set(inv))   # 배열/조립/래퍼로만 읽히는 것

    out = {
        "schema": "lumina-knob-inventory-v1",
        "note": "분류하지 않는다 — 분류에 필요한 사실만 모은다. "
                "생존이 예외이고 스크랩이 기본이라는 기준은 대장(attic/knobs/)에 있다.",
        "n_literal_getenv_names": len(inv),
        "n_universe": len(universe),
        "names_without_literal_getenv": not_literal,
        "live_last_run_count": len(live),
        "knobs": {k: inv[k] for k in sorted(inv)},
    }
    p = ROOT / "validation/layer1_replan/KNOB_INVENTORY.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    per_file: dict[str, int] = {}
    for k, e in inv.items():
        for fn in e["files"]:
            per_file[fn] = per_file.get(fn, 0) + 1
    print(f"리터럴 getenv 노브 {len(inv)} / 전집 {len(universe)}"
          f" (배열·조립·래퍼로만 읽히는 것 {len(not_literal)})")
    print(f"최근 판정런이 넘긴 것 {len(live)}\n")
    print(f"{'파일':<26}{'노브':>6}")
    for fn, n in sorted(per_file.items(), key=lambda kv: -kv[1]):
        print(f"{fn:<26}{n:>6}")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
