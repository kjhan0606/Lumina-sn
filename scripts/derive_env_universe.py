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
  (5) **src 밖 소비자** — 셸/파이썬이 값을 쓰는 곳                 [2026-08-07 추가]

★(5)를 빠뜨린 것이 T3 잡 226529 를 죽였다.  전집을 "src 가 읽는 것"으로 정의했는데,
런처는 **셸이 소비할 변수**도 설정한다(LUMINA_MODEL_DIR 을 12개 스크립트가 쓴다).
STRICT 가 그것을 미등록으로 보고 런을 거부했다 — 게이트의 오탐이었다.
전집의 올바른 정의는 "**어떤 소비자든 읽는 이름**"이다.  읽는 자가 src 인지 셸인지는
소비자 종류일 뿐이고, 아무도 안 읽는 이름만이 오타/죽은 노브다.

  ⚠(5)의 함정: `NAME=${NAME:-1}` 은 **소비가 아니라 설정**이다(자식 env 로 넘길 뿐).
    이것을 소비로 세면 죽은 노브 7종이 전집에 들어가 게이트가 침묵한다 — 세탁이다.
    그래서 자기-기본값 형태는 명시적으로 제외한다.

산출: validation/instrumentation_debt/ENV_UNIVERSE.json  +  src/env_universe.h(생성물)
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


SELF_DEFAULT = re.compile(r'\b(LUMINA_[A-Z0-9_]+)\s*=\s*\$\{\s*\1\b')
SHELL_DEREF = re.compile(r'\$\{?(LUMINA_[A-Z0-9_]+)')
PY_ENV = re.compile(r'''(?:os\.environ(?:\.get)?[\[(]|os\.getenv\()\s*["'](LUMINA_[A-Z0-9_]+)''')


def nonsrc_consumers() -> tuple[set[str], set[str]]:
    """(5) src 밖에서 값을 **쓰는** 곳.  자기-기본값(`NAME=${NAME:-..}`)은 설정이므로 뺀다.

    반환: (실제 소비, 자기-기본값으로만 등장)
    후자는 죽은 노브의 증거다 — 런처가 설정만 하고 아무도 안 읽는다.
    """
    consumed: set[str] = set()
    set_only: set[str] = set()
    files = [p for d in ("scripts", "tests") for p in (ROOT / d).rglob("*") if p.is_file()]
    if (ROOT / "Makefile").is_file():
        files.append(ROOT / "Makefile")
    for f in files:
        try:
            txt = f.read_text(errors="ignore")
        except OSError:
            continue
        for line in txt.splitlines():
            selfs = {m.group(1) for m in SELF_DEFAULT.finditer(line)}
            for m in SHELL_DEREF.finditer(line):
                (set_only if m.group(1) in selfs else consumed).add(m.group(1))
            consumed |= {m.group(1) for m in PY_ENV.finditer(line)}
    return consumed, set_only - consumed


def emit_header(names: list[str], counts: dict[str, int], only_wrapper: list[str],
                nonsrc_only: list[str]) -> Path:
    """★헤더도 여기서 낸다.  2026-08-07 까지 이 헤더는 **생성자 없는 생성물**이었다
    (내가 인라인으로 만들었다) — C7 재현 고아 부류를 내가 새로 하나 만든 셈이다."""
    src_lines = " · ".join(f"{k} {v}" for k, v in counts.items())
    body = "".join(f'    "{n}",\n' for n in names)
    txt = f"""/* GENERATED — do not edit by hand.
 * scripts/derive_env_universe.py 가 기계적으로 도출한다.
 * 손으로 세면 짧아진다 — 2026-08-06~07 에 같은 종류의 목록을 열한 번 세었고 열한 번 다 짧았다.
 *
 * 갈래: {src_lines}  ⟹ 합집합 {len(names)}
 * 래퍼 스캔이 없었으면 놓쳤을 것: {', '.join(only_wrapper) or '(없음)'}
 * ★src 밖 소비자만 아는 이름(5번 갈래가 없었으면 STRICT 가 오탐했을 것): {len(nonsrc_only)}
 *   {', '.join(nonsrc_only) or '(없음)'}
 *
 * 전집의 정의 = "**어떤 소비자든 읽는 이름**".  src 가 안 읽는다는 사실은
 * 죽은 노브 census 의 몫이지 이 목록의 몫이 아니다.
 */
#ifndef LUMINA_ENV_UNIVERSE_H
#define LUMINA_ENV_UNIVERSE_H

#define LUMINA_ENV_UNIVERSE_COUNT {len(names)}

static const char *const LUMINA_ENV_UNIVERSE[] = {{
{body}}};

#endif /* LUMINA_ENV_UNIVERSE_H */
"""
    p = ROOT / "src/env_universe.h"
    p.write_text(txt)
    return p


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

    src_union = set().union(*by_source.values())
    consumed_nonsrc, set_only = nonsrc_consumers()
    by_source["nonsrc_consumers"] = consumed_nonsrc

    union = src_union | consumed_nonsrc
    lumina = sorted(n for n in union if n.startswith("LUMINA_"))
    other = sorted(n for n in union if not n.startswith("LUMINA_"))
    # 각 갈래가 없었다면 놓쳤을 것 — 함정의 크기를 실측한다
    only_wrapper = sorted(by_source["wrapper_calls"] - by_source["literal_getenv"])
    only_nonsrc = sorted(consumed_nonsrc - src_union)
    # src 가 안 읽는데 런처가 설정만 하는 것 = 죽은 노브 (전집에 넣지 않는다)
    dead_evidence = sorted(set_only - src_union)

    out = {
        "schema": "lumina-env-universe-v2",
        "note": "손으로 세지 않는다. 다섯 갈래를 기계로 모은다. "
                "(4)래퍼 호출부와 (5)src 밖 소비자가 함정이다. "
                "전집 = 어떤 소비자든 읽는 이름. 자기-기본값(NAME=${NAME:-..})은 설정이지 소비가 아니다.",
        "counts": {k: len(v) for k, v in by_source.items()},
        "union_total": len(union),
        "lumina_prefixed": len(lumina),
        "non_lumina": other,
        "missed_without_wrapper_scan": only_wrapper,
        "missed_without_nonsrc_scan": only_nonsrc,
        "set_only_never_read": dead_evidence,
        "per_file_hits": dict(sorted(per_file.items(), key=lambda kv: -kv[1])),
        "env": lumina,
    }
    p = ROOT / "validation/instrumentation_debt/ENV_UNIVERSE.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    h = emit_header(lumina, out["counts"], only_wrapper, only_nonsrc)
    print(json.dumps({k: v for k, v in out.items() if k not in ("env", "per_file_hits")},
                     indent=1, ensure_ascii=False))
    print(f"\n-> {p}\n-> {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
