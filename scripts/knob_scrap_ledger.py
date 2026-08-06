#!/usr/bin/env python3
"""창고 대장 생성 — 어떤 노브를 스크랩하고 무엇을 남기는가.

user 지시(2026-08-07): "비물리와 계측용 노브들을 소스코드들로부터 스크랩해서 창고에
쳐박아놓고, 판정으로 인정된 물리 배선도와 0층에서 확증된 자산을 바탕으로 1층 검사를 시작."

**생존이 예외, 스크랩이 기본.**  419 개를 하나씩 변호하는 대신 생존 목록을 명시하고
나머지를 규칙으로 창고에 넣는다.  분류 근거는 전부 기계 관측이며, 사람 판정이 들어간
곳은 `by` 필드에 남긴다.

부류
  S-INPUT     값이 경로·자원 지정 → 노브가 아니라 입력
  S-CONTRACT  0층 계약 10건이 요구하는 게이트 → 확증된 자산
  P-VERDICT   판정런이 실제로 넘긴 것 → **물리 판정 대상**(개별 판정 전까지 보류)
  SCRAP-DEAD  아무 런처도 설정하지 않음 → 분기가 밟히지 않는다(구성상 무해)
  SCRAP-FOSSIL 과거 런처만 설정 → 실패의 화석층
  SCRAP-CLAMP  이름·구현이 clamp/floor/ceil → 규약상 물리가 아니다

산출: attic/knobs/KNOB_SCRAP_LEDGER.json (+ .md 요약)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INV = ROOT / "validation/layer1_replan/KNOB_INVENTORY.json"

# 0층 계약 10건이 요구하는 게이트 (커밋 a97d0e1 의 폐합 목록에서 유래).
# 계약이 없어지지 않는 한 이 게이트는 남는다.
S_CONTRACT = {
    "LUMINA_CONFIG_PREC",     # CONFIG-PREC: 경계온도 우선순위 계약
    "LUMINA_ENV_STRICT",      # 노브 표면 동결 4단계: 미등록 env 거부
}
# 값이 경로·자원인 것 (노브가 아니다)
PATHY = re.compile(r"(FILE|DIR|PATH|CSV|_H5|_NPY|_BIN$|DUMP_PATH)")
# 규약상 물리가 아닌 이름 (feedback_clamps_are_not_physics_fix_the_solver)
CLAMPY = re.compile(r"(CLAMP|FLOOR|CEIL|CAP$|_CAP_|MIN$|MAX$)")


def classify(name: str, e: dict) -> tuple[str, str]:
    if name in S_CONTRACT:
        return "S-CONTRACT", "0층 계약이 요구하는 게이트"
    if PATHY.search(name):
        return "S-INPUT", "값이 경로·자원 지정 — 노브가 아니라 입력"
    if CLAMPY.search(name):
        return "SCRAP-CLAMP", "이름이 clamp/floor/ceil — 규약상 물리가 아니다(개별 확인 필요)"
    if e["live_last_run"]:
        return "P-VERDICT", "판정런이 실제로 넘김 — 물리 판정 대상"
    if e["launchers"] == 0:
        return "SCRAP-DEAD", "아무 런처도 설정하지 않음 — 분기가 밟히지 않는다"
    return "SCRAP-FOSSIL", f"과거 런처 {e['launchers']}개만 설정 — 실패의 화석층"


def main() -> int:
    inv = json.loads(INV.read_text())["knobs"]
    rows = {}
    for name, e in inv.items():
        kind, why = classify(name, e)
        rows[name] = {
            "class": kind, "why": why,
            "sites": e["sites"], "files": e["files"], "functions": e["functions"],
            "launchers": e["launchers"], "live_last_run": e["live_last_run"],
        }

    counts: dict[str, int] = {}
    for r in rows.values():
        counts[r["class"]] = counts.get(r["class"], 0) + 1

    out = {
        "schema": "lumina-knob-scrap-ledger-v1",
        "principle": "생존이 예외, 스크랩이 기본. 살아남으려면 "
                     "입력이거나 · 0층 계약이 요구하거나 · 판정된 물리 배선도에 있어야 한다.",
        "counts": counts,
        "total": len(rows),
        "knobs": rows,
    }
    d = ROOT / "attic/knobs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "KNOB_SCRAP_LEDGER.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    order = ["S-CONTRACT", "S-INPUT", "P-VERDICT", "SCRAP-CLAMP", "SCRAP-FOSSIL", "SCRAP-DEAD"]
    md = ["# 창고 대장 — 노브 스크랩", "",
          "생성 `scripts/knob_scrap_ledger.py`. **생존이 예외, 스크랩이 기본.**", "",
          "| 부류 | 수 | 뜻 |", "|---|---|---|"]
    meaning = {
        "S-CONTRACT": "0층 계약이 요구 — 확증된 자산",
        "S-INPUT": "경로·자원 지정 — 노브가 아니라 입력",
        "P-VERDICT": "**판정런이 넘김 — 개별 물리 판정 대상**",
        "SCRAP-CLAMP": "이름이 clamp/floor/ceil — 규약상 물리가 아니다",
        "SCRAP-FOSSIL": "과거 런처만 설정 — 실패의 화석층",
        "SCRAP-DEAD": "아무도 설정 안 함 — 분기가 밟히지 않는다",
    }
    for k in order:
        md.append(f"| {k} | {counts.get(k,0)} | {meaning[k]} |")
    md += ["", f"합계 **{len(rows)}**", "",
           "## P-VERDICT (개별 판정 대상 — 여기만 사람이 판정한다)", "",
           "| 노브 | 사이트 | 파일 |", "|---|---|---|"]
    for n in sorted(k for k, v in rows.items() if v["class"] == "P-VERDICT"):
        r = rows[n]
        md.append(f"| `{n}` | {len(r['sites'])} | {', '.join(r['files'])} |")
    md += ["", "## SCRAP-CLAMP (규약 위반 후보 — 전량 확인)", "",
           "| 노브 | 사이트 | 파일 |", "|---|---|---|"]
    for n in sorted(k for k, v in rows.items() if v["class"] == "SCRAP-CLAMP"):
        r = rows[n]
        md.append(f"| `{n}` | {len(r['sites'])} | {', '.join(r['files'])} |")
    (d / "KNOB_SCRAP_LEDGER.md").write_text("\n".join(md) + "\n")

    for k in order:
        print(f"  {k:<14}{counts.get(k,0):>5}")
    print(f"  {'합계':<14}{len(rows):>5}")
    print(f"\n-> {d}/KNOB_SCRAP_LEDGER.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
