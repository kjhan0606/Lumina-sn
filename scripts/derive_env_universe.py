#!/usr/bin/env python3
"""src 가 실제로 읽는 env 전집을 **기계적으로** 도출한다.

노브 표면 동결 1단계의 전제.  화이트리스트 거부를 안전하게 켜려면 "src 가 읽는 것"의
전집이 정확해야 한다 — 하나라도 빠지면 정상 런이 죽는다.

2026-08-06~07 에 같은 종류의 목록을 아홉 번 손으로 세었고 **아홉 번 다 짧았다.**
그래서 손으로 세지 않는다.  네 갈래를 전부 기계로 모은다:

  (1) getenv("LITERAL")                       — 656건
  (2) env 이름 배열을 getenv 에 먹이는 곳      — retired_scalar_options / obsolete /
                                                forbidden_population_knobs / blockers / guard
  (3) snprintf 로 조립하는 이름               — LUMINA_AUL_SCALE%s_* (9 suffix × 5 필드)
                                                LUMINA_DR_BOOST_%s   (6 소스명)
  (4) env 를 읽는 **래퍼 함수**의 호출부 리터럴 — config_prec_parse_switch /
                                                ew_env_true / banner_gate_off
      ⚠ (4)가 함정이다.  리터럴이 getenv 옆이 아니라 호출부에 있어서
        `grep 'getenv("'` 로는 원리적으로 안 보인다.

산출: validation/instrumentation_debt/ENV_UNIVERSE.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = sorted(list((ROOT / "src").glob("*.c")) + list((ROOT / "src").glob("*.cu"))
             + list((ROOT / "src").glob("*.h")))

# (2) getenv 에 원소가 그대로 들어가는 배열들
ENV_ARRAYS = ("retired_scalar_options", "obsolete", "forbidden_population_knobs",
              "blockers", "guard")
# (4) 첫 인자(또는 지정 위치)가 env 이름인 래퍼
WRAPPERS = {
    "config_prec_parse_switch": 0,
    "ew_env_true": 0,
    "banner_gate_off": 1,      # banner_gate_off(tag, var, ...)
}


def literal_getenv(txt: str) -> set[str]:
    return set(re.findall(r'getenv\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)', txt))


def array_members(txt: str) -> set[str]:
    out: set[str] = set()
    for name in ENV_ARRAYS:
        for m in re.finditer(re.escape(name) + r"\s*\[\s*\]\s*=\s*\{", txt):
            seg = txt[m.end(): m.end() + 4000]
            end = seg.find("};")
            out |= set(re.findall(r'"([A-Z][A-Z0-9_]*)"', seg[: end if end > 0 else 4000]))
    return out


def constructed(txt: str) -> set[str]:
    """snprintf 로 조립되는 이름을 **접미 배열을 읽어서** 전개한다."""
    out: set[str] = set()
    if "LUMINA_AUL_SCALE%s_" in txt:
        m = re.search(r'suffixes\[\]\s*=\s*\{([^}]*)\}', txt)
        sufs = re.findall(r'"([^"]*)"', m.group(1)) if m else [""]
        fields = set(re.findall(r'"LUMINA_AUL_SCALE%s_([A-Z_]+)"', txt))
        out |= {f"LUMINA_AUL_SCALE{s}_{f}" for s in sufs for f in fields}
    if "LUMINA_DR_BOOST_%s" in txt:
        m = re.search(r'names\[\d*\]\s*=\s*\{([^}]*)\}', txt)
        nm = re.findall(r'"([^"]*)"', m.group(1)) if m else []
        out |= {f"LUMINA_DR_BOOST_{n}" for n in nm if n and n != "NONE"}
    return out


def wrapper_calls(txt: str) -> set[str]:
    """★래퍼 호출부의 리터럴.  getenv 옆이 아니라 호출부에 있어 grep 이 못 본다."""
    out: set[str] = set()
    for fn, argidx in WRAPPERS.items():
        for m in re.finditer(re.escape(fn) + r"\s*\(", txt):
            seg = txt[m.end(): m.end() + 400]
            depth, buf = 1, ""
            for ch in seg:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                buf += ch
            args = [a.strip() for a in re.split(r",(?![^(]*\))", buf)]
            if len(args) > argidx:
                lit = re.fullmatch(r'"([A-Z][A-Z0-9_]*)"', args[argidx])
                if lit:
                    out.add(lit.group(1))
    return out


def main() -> int:
    by_source: dict[str, set[str]] = {"literal_getenv": set(), "env_arrays": set(),
                                      "constructed": set(), "wrapper_calls": set()}
    per_file: dict[str, int] = {}
    for f in SRC:
        txt = f.read_text(errors="ignore")
        a = literal_getenv(txt); b = array_members(txt)
        c = constructed(txt);    d = wrapper_calls(txt)
        by_source["literal_getenv"] |= a
        by_source["env_arrays"] |= b
        by_source["constructed"] |= c
        by_source["wrapper_calls"] |= d
        n = len(a | b | c | d)
        if n:
            per_file[f.name] = n

    union = set().union(*by_source.values())
    lumina = sorted(n for n in union if n.startswith("LUMINA_"))
    other = sorted(n for n in union if not n.startswith("LUMINA_"))
    # 래퍼가 없었다면 놓쳤을 것 — 함정의 크기를 실측한다
    only_wrapper = sorted(by_source["wrapper_calls"] - by_source["literal_getenv"])

    out = {
        "schema": "lumina-env-universe-v1",
        "note": "손으로 세지 않는다. 네 갈래를 기계로 모은다. (4)래퍼 호출부가 함정이다.",
        "counts": {k: len(v) for k, v in by_source.items()},
        "union_total": len(union),
        "lumina_prefixed": len(lumina),
        "non_lumina": other,
        "missed_without_wrapper_scan": only_wrapper,
        "per_file_hits": dict(sorted(per_file.items(), key=lambda kv: -kv[1])),
        "env": lumina,
    }
    p = ROOT / "validation/instrumentation_debt/ENV_UNIVERSE.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("env", "per_file_hits")},
                     indent=1, ensure_ascii=False))
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
