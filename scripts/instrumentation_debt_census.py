#!/usr/bin/env python3
"""계측·배선 부채 census — 재현 가능한 집계기 (docs/INSTRUMENTATION_DEBT_CENSUS.md 의 정본 산출).

Fable L3 검수 Q3-5 대응:
  "157·12·5·5 어느 것도 커밋된 집계 스크립트나 기록된 grep 패턴이 없다.
   98→157 교정 자체가 '패턴 선택이 수치를 60% 흔든다'는 실증인데,
   제3의 패턴이 190 을 내지 않는다는 보장이 없고 검증할 방법도 없다.
   또한 어느 트리 상태에서 셌는지 미기재 — 미추적 ~1,500개의 더러운 트리."

⟹ 이 스크립트는 세 가지를 강제한다:
  1) **패턴을 산출물에 박는다** — 어떤 정규식으로 셌는지 JSON 에 그대로 들어간다
  2) **패턴 민감도를 함께 보고한다** — 한 수치가 아니라 여러 패턴의 스프레드.
     스프레드가 크면 그 수치는 패턴 아티팩트이지 사실이 아니다
  3) **트리 상태를 앵커한다** — git HEAD · dirty 여부 · 미추적 파일 수

단일 수치를 신뢰하지 말고 `spread` 를 보라. 그것이 이 census 의 정직한 단위다.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd: str) -> str:
    return subprocess.run(["bash", "-c", cmd], cwd=ROOT,
                          capture_output=True, text=True).stdout


def tree_anchor() -> dict:
    dirty = sh("git status --porcelain 2>/dev/null")
    return {
        "git_head": sh("git rev-parse HEAD").strip(),
        "git_head_short": sh("git rev-parse --short HEAD").strip(),
        "branch": sh("git rev-parse --abbrev-ref HEAD").strip(),
        "tracked_modified": len([l for l in dirty.splitlines()
                                 if l[:2].strip() and not l.startswith("??")]),
        "untracked": len([l for l in dirty.splitlines() if l.startswith("??")]),
        "note": "미추적이 많은 더러운 트리에서의 집계다. 앵커 없이 인용 금지.",
    }


def multi_pattern(label: str, patterns: dict[str, str]) -> dict:
    """같은 질문을 여러 정규식으로 세고 **스프레드**를 보고한다.

    스프레드가 크면 그 수치는 사실이 아니라 패턴 아티팩트다.
    """
    counts = {}
    for name, cmd in patterns.items():
        out = sh(cmd).strip()
        counts[name] = int(out) if out.isdigit() else out
    nums = [v for v in counts.values() if isinstance(v, int)]
    return {
        "question": label,
        "patterns": patterns,
        "counts": counts,
        "min": min(nums) if nums else None,
        "max": max(nums) if nums else None,
        "spread_ratio": (max(nums) / min(nums)) if nums and min(nums) else None,
    }


# --- 계약 목록 사본 (C9) — 소스에서 직접 읽는다 ---------------------------
def contract_copies() -> dict:
    def arr(path: str, start: int, end: int) -> list[str]:
        txt = sh(f"sed -n '{start},{end}p' {path}")
        return sorted(set(re.findall(r'"(LUMINA_[A-Z0-9_]+)"', txt)))
    plasma = arr("src/lumina_plasma.c", 14950, 14970)
    ew = arr("src/lumina_element_wide.c", 1810, 1835)
    scalar = arr("src/seed_capability.c", 79, 92)
    return {
        "plasma_hard_fatal": {"n": len(plasma), "env": plasma},
        "element_wide_diagnostic_only": {"n": len(ew), "env": ew,
                                         "note": "세기만 하고 막지 않는다 (C8 fail-open)"},
        "seed_capability_obsolete": {"n": len(scalar), "env": scalar},
        "union": sorted(set(plasma) | set(ew) | set(scalar)),
        "plasma_minus_ew": sorted(set(plasma) - set(ew)),
        "ew_minus_plasma": sorted(set(ew) - set(plasma)),
        "divergence_is_the_defect":
            "같은 성격의 계약이 서로 다른 원소로 3사본 존재한다 (C9)",
    }


def main() -> int:
    hard = contract_copies()
    hard_env = sorted(set(hard["plasma_hard_fatal"]["env"])
                      | set(hard["seed_capability_obsolete"]["env"]))
    alt = "|".join(hard_env)

    out = {
        "schema": "lumina-instrumentation-debt-census-v1",
        "doc": "docs/INSTRUMENTATION_DEBT_CENSUS.md",
        "tree_anchor": tree_anchor(),
        "read_this_first":
            "단일 수치가 아니라 spread_ratio 를 보라. 1.0 에서 멀수록 그 수치는 "
            "패턴 아티팩트다. 98->157 교정이 그 실증이었다.",
        "C9_contract_copies": hard,
        "C2_hard_reject_env": {"n": len(hard_env), "env": hard_env},
    }

    # C2 — 영향 런처: 세 가지 패턴으로 세고 스프레드를 본다
    out["C2_launchers_affected"] = multi_pattern(
        "하드 거부 env 를 설정하는 scripts/*.sh 는 몇 개인가",
        {
            "p1_export_prefix_nonzero":
                f"grep -rlE 'export .*({alt})=[^0 ]' scripts/*.sh 2>/dev/null | wc -l",
            "p2_any_assignment":
                f"grep -rlE '({alt})=' scripts/*.sh 2>/dev/null | wc -l",
            "p3_assignment_nonzero_anywhere":
                f"grep -rlE '({alt})=[^0 ]' scripts/*.sh 2>/dev/null | wc -l",
        })
    out["C2_caveats"] = [
        "정적 grep 은 ${VAR-default} 파라미터화와 sbatch --export=ALL 상속을 판정할 수 없다 (Fable Q2)",
        "plasma 사이트는 enable_nlte && iter>=nlte_start_iter 에서만 검사한다 — "
        "비-NLTE 런처는 사망이 아니라 no-op 다 (C10). '발동' 수는 상한이 아니라 후보 수다",
    ]

    # C1 — 고정 기대값: 형태별로 따로 센다 (sha256 만 세면 자기 동기 사건을 못 찾는다)
    out["C1_pinned_expectations"] = multi_pattern(
        "검증에 쓰이는 하드코딩 기대값은 몇 곳인가",
        {
            "sha256_64hex_in_scripts":
                "grep -rlE '\"[0-9a-f]{64}\"' scripts/*.py 2>/dev/null | wc -l",
            "fnv64_16hex_in_c":
                "grep -rlE 'UINT64_C\\(0x[0-9a-f]{16}\\)' tests/*.c src/*.c 2>/dev/null | wc -l",
            "expected_ident_in_c":
                "grep -rlE 'expected_[a-z_]*(hash|fnv|lines|count|shells)' tests/*.c src/*.c 2>/dev/null | wc -l",
        })

    # C5 — 덱 계보: toy06 계열로 좁히지 않는다 (Fable Q2: 모집단이 좁았다)
    decks = [p for p in (ROOT / "data").iterdir()
             if p.is_dir() and p.name.startswith("tardis_reference")]
    out["C5_deck_provenance"] = {
        "deck_families_total": len(decks),
        "with_vintage_manifest": sum((d / "atomic_vintage_manifest.csv").exists() for d in decks),
        "with_provenance_stamp": sum((d / "DECK_PROVENANCE.json").exists() for d in decks),
        "note": "앞선 census 는 toy06_19p48d 계열 9덱만 셌다. 전 계열로 넓힌 수치다",
    }

    # C7 — 재현 경로: schema 있는 것/없는 것 둘 다 센다
    vjson = list((ROOT / "validation").rglob("*.json"))
    no_schema = 0
    orphan = []
    for f in vjson:
        try:
            s = json.loads(f.read_text()).get("schema")
        except Exception:
            continue
        if not s:
            no_schema += 1
            continue
        if not sh(f"grep -rlF '{s}' scripts/*.py 2>/dev/null").strip():
            orphan.append(str(f.relative_to(ROOT)))
    out["C7_reproduction"] = {
        "validation_json_total": len(vjson),
        "without_schema_field": no_schema,
        "with_schema_but_no_producer": len(orphan),
        "orphans": sorted(orphan),
        "caveat": "schema 를 f-string 으로 합성하는 생성자는 리터럴 grep 이 못 찾는다 (Fable Q2). "
                  "또한 CSV·MD·npy 산출물은 이 프레임 밖이다",
    }

    p = ROOT / "validation/instrumentation_debt/CENSUS.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("C9_contract_copies", "C7_reproduction")},
                     indent=1, ensure_ascii=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
