#!/usr/bin/env python3
"""사전등록 기계 프리플라이트 — 계약이 발주되기 전에 스스로를 검사한다.

이 게이트가 존재하는 이유(2026-08-21, DET-SPRIM-L6 4회 발주 실패):
코드에는 게이트·음성대조·배터리·byte-parity 가 있는데 **가장 상류인 사전등록에는
검사가 하나도 없었다.** 모든 하류가 계약을 소비하고 아무도 계약을 생산-검사하지 않는다 —
「고리 밖 감사」가 지목한 그 형태다. 실패한 다섯 건 중 셋은 여기서 잡혔을 것이다.

fail-closed: 사전등록이 변경집합 표나 분기 표를 갖고도 선언 블록이 없으면 **거부**한다.
조용한 건너뛰기는 이 프로젝트에서 금지다.
"""
from __future__ import annotations
import argparse, json, math, random, re, statistics, subprocess, sys, tempfile, os
from pathlib import Path

DECL_RE = re.compile(r"```prereg-preflight\s*\n(.*?)\n```", re.S)
SAFE = {"abs": abs, "len": len, "sum": sum, "min": min, "max": max,
        "median": statistics.median, "sorted": sorted, "any": any, "all": all}


class Fail(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def emit(kind: str, reason: str, detail: str = "") -> None:
    tail = f" detail={detail}" if detail else ""
    print(f"[PREREG-PREFLIGHT][{kind}] reason={reason}{tail}")


# ---------- PF-1: 변경집합 표 <-> 심볼 grep 양방향 차집합 ----------
def pf1_changeset(doc: str, decl: dict, root: Path) -> None:
    spec = decl.get("changeset")
    if spec is None:
        raise Fail("declaration-missing:changeset")
    heading = spec["table_heading"]
    if heading not in doc:
        raise Fail("changeset-heading-absent", heading)
    seg = doc[doc.index(heading):]
    stop = spec.get("table_end")
    if stop and stop in seg:
        seg = seg[:seg.index(stop)]
    listed = set(re.findall(r"`(" + spec["path_pattern"] + r")`", seg))
    sym = spec["symbol"]
    found = set()
    for r in spec["roots"]:
        out = subprocess.run(["grep", "-rl", sym, r], cwd=root,
                             capture_output=True, text=True)
        found |= {p for p in out.stdout.split() if p}
    found |= set(spec.get("expected_extra", []))   # 표에 있으나 심볼이 없는 정당한 행(예: tests/)
    missing = sorted(found - listed)
    extra = sorted(listed - found)
    if missing:
        raise Fail("changeset-missing-file", ",".join(missing))
    if extra:
        raise Fail("changeset-extra-file", ",".join(extra))
    emit("PASS", "changeset-consistent", f"files={len(listed)}")


# ---------- PF-2: 분기 판정식 무작위 스윕 (중복·공백) ----------
def _draw(rng, spec):
    """혼합 가중치를 단체(simplex)에서 뽑아 각 국면의 **비율을 직접 통제**한다.

    순진한 균등 추출은 겹침이 사는 영역에 도달하지 못한다 — 그 사실을 NC-P2a 가
    잡았다(2026-08-21). 잣대가 못 가는 곳은 잣대가 없는 것과 같다.
    """
    regimes = spec.get("regimes", [[0.3, 0.99], [0.999, 1.001], [1.01, 3.0], [10.5, 60.0]])
    w = [rng.random() ** 2 for _ in regimes]          # ^2 로 치우친 혼합도 자주 뽑는다
    tot = sum(w) or 1.0
    n = rng.choice([20, 50, 100, 200])
    v = []
    for (lo, hi), wi in zip(regimes, w):
        for _ in range(int(round(n * wi / tot))):
            v.append(rng.uniform(lo, hi))
    return v or [1.0]


def _evaluate(metrics, rules, v):
    env = dict(SAFE); env["v"] = v; env["__builtins__"] = {}
    vals = {k: eval(e, env) for k, e in metrics.items()}
    env.update(vals)
    hot = [r["name"] for r in rules if eval(r["predicate"], dict(env))]
    return vals, hot


def pf2_branches(decl: dict, trials: int, seed: int) -> None:
    spec = decl.get("branches")
    if spec is None:
        raise Fail("declaration-missing:branches")
    metrics, rules = spec["metrics"], spec["rules"]
    rng = random.Random(seed)
    overlaps = gaps = 0
    worst = ""

    # (a) 회귀 픽스처 — 역사적 반례를 결정론으로 다시 친다
    for fx in spec.get("adversarial_fixtures", []):
        v = [x for val, cnt in fx["mix"] for x in [val] * cnt]
        vals, hot = _evaluate(metrics, rules, v)
        if len(hot) > 1:
            raise Fail("branch-overlap",
                       f"fixture={fx.get('name', '?')} {'+'.join(hot)}")
        if not hot and not spec.get("residual"):
            raise Fail("branch-gap", f"fixture={fx.get('name', '?')}")

    # (b) 혼합 가중치 스윕
    for _ in range(trials):
        vals, hot = _evaluate(metrics, rules, _draw(rng, spec))
        if len(hot) > 1:
            overlaps += 1
            if not worst:
                worst = f"{'+'.join(hot)} @ {json.dumps({k: round(x, 4) for k, x in vals.items()})}"
        elif not hot and not spec.get("residual"):
            gaps += 1
    if overlaps:
        raise Fail("branch-overlap", f"{overlaps}/{trials} first={worst}")
    if gaps:
        raise Fail("branch-gap", f"{gaps}/{trials}")
    emit("PASS", "branch-partition",
         f"trials={trials} rules={len(rules)} fixtures={len(spec.get('adversarial_fixtures', []))} "
         f"residual={spec.get('residual', '-')}")


# ---------- PF-3: 참조된 경로·플래그의 실존 ----------
def pf3_references(decl: dict, root: Path) -> None:
    refs = decl.get("references")
    if refs is None:
        raise Fail("declaration-missing:references")
    for ref in refs:
        p = root / ref["path"]
        if not p.exists():
            raise Fail("reference-missing-path", ref["path"])
        text = p.read_text(errors="replace")
        for fl in ref.get("flags_existing", []):
            if fl not in text:
                raise Fail("reference-unsupported-flag", f"{ref['path']}:{fl}")
        for fl in ref.get("flags_planned", []):
            if fl in text:
                raise Fail("reference-flag-already-present",
                           f"{ref['path']}:{fl}")
    emit("PASS", "references-resolved", f"refs={len(refs)}")


def load_decl(doc: str) -> dict:
    m = DECL_RE.search(doc)
    if not m:
        raise Fail("declaration-block-absent",
                   "사전등록에 ```prereg-preflight 블록이 없다 (fail-closed)")
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise Fail("declaration-unparsable", str(exc)) from exc


def run(path: Path, root: Path, trials: int, seed: int) -> int:
    doc = path.read_text()
    try:
        decl = load_decl(doc)
        pf1_changeset(doc, decl, root)
        pf2_branches(decl, trials, seed)
        pf3_references(decl, root)
    except Fail as f:
        emit("FAIL", f.reason, f.detail)
        return 1
    emit("PASS", "prereg-preflight", path.name)
    return 0


# ---------------- 음성대조 (주입 결함이 이름 있는 사유로 FAIL 해야 한다) ----------------
GOOD_DECL = {
    "changeset": {"table_heading": "### 변경집합", "table_end": "### 끝",
                  "path_pattern": r"src/[a-z_]+\.[ch]|tests/",
                  "symbol": "widget_", "roots": ["src"],
                  "expected_extra": ["tests/"]},
    "branches": {
        "metrics": {"f_super": "sum(1 for x in v if x>10)/len(v)",
                    "dev": "sum(1 for x in v if abs(x-1)>0.01)/len(v)",
                    "q50": "median(v)",
                    "near": "sum(1 for x in v if abs(x-1)<=1e-3)/len(v)"},
        "rules": [{"name": "A'", "predicate": "f_super >= 0.10"},
                  {"name": "A", "predicate": "f_super < 0.10 and dev >= 0.10 and 0.5 <= q50 < 1.0"},
                  {"name": "B", "predicate": "near >= 0.99"}],
        "adversarial_fixtures": [
            {"name": "codex-counterexample-2026-08-21",
             "mix": [[0.8, 60], [1.0, 30], [11.0, 10]]}],
        "residual": "C"},
    "references": [{"path": "scripts/tool.py",
                    "flags_existing": ["--alpha"], "flags_planned": ["--beta"]}],
}
GOOD_DOC = """# 합성 사전등록
### 변경집합
| `src/a.c` | x |
| `src/b.h` | x |
| `tests/` | x |
### 끝
"""


def _fixture(tmp: Path, doc_extra: str, decl: dict) -> Path:
    (tmp / "src").mkdir(parents=True, exist_ok=True)
    (tmp / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp / "src" / "a.c").write_text("int widget_a;\n")
    (tmp / "src" / "b.h").write_text("extern int widget_b;\n")
    (tmp / "scripts" / "tool.py").write_text("p.add_argument('--alpha')\n")
    doc = GOOD_DOC + doc_extra + "\n```prereg-preflight\n" + json.dumps(decl) + "\n```\n"
    f = tmp / "PREREG.md"
    f.write_text(doc)
    return f


def selftest() -> int:
    import copy
    cases = []

    def case(name, reason, mutate_decl=None, mutate_doc=None, mutate_repo=None):
        cases.append((name, reason, mutate_decl, mutate_doc, mutate_repo))

    case("NC-P0", None)                                                # ★성한 계약 -> PASS
    case("NC-P1a", "changeset-missing-file",
         mutate_doc=lambda d: d.replace("| `src/b.h` | x |\n", ""))
    case("NC-P1b", "changeset-extra-file",
         mutate_doc=lambda d: d.replace("### 끝", "| `src/ghost.c` | x |\n### 끝"))
    def overlap(dc):
        dc["branches"]["rules"][1]["predicate"] = \
            "dev >= 0.10 and 0.5 <= q50 < 1.0"          # f_super 배제조건 제거 = 실제 결함 ④
        return dc
    case("NC-P2a", "branch-overlap", mutate_decl=overlap)
    def gap(dc):
        dc["branches"].pop("residual")
        return dc
    case("NC-P2b", "branch-gap", mutate_decl=gap)
    case("NC-P3a", "reference-missing-path",
         mutate_repo=lambda t: (t / "scripts" / "tool.py").unlink())
    case("NC-P3b", "reference-unsupported-flag",
         mutate_repo=lambda t: (t / "scripts" / "tool.py").write_text("nothing\n"))
    case("NC-P3c", "reference-flag-already-present",
         mutate_repo=lambda t: (t / "scripts" / "tool.py").write_text(
             "p.add_argument('--alpha')\np.add_argument('--beta')\n"))
    case("NC-P4", "declaration-block-absent",
         mutate_doc=lambda d: DECL_RE.sub("", d))

    bad = 0
    for name, want, md, mdoc, mrepo in cases:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            decl = md(copy.deepcopy(GOOD_DECL)) if md else copy.deepcopy(GOOD_DECL)
            f = _fixture(tmp, "", decl)
            if mdoc:
                f.write_text(mdoc(f.read_text()))
            if mrepo:
                mrepo(tmp)
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = run(f, tmp, 4000, 11)
            out = buf.getvalue()
            if want is None:
                ok = rc == 0 and "[FAIL]" not in out
                got = "PASS" if ok else out.strip().splitlines()[-1]
            else:
                ok = rc == 1 and f"reason={want}" in out
                got = next((l for l in out.splitlines() if "[FAIL]" in l), "(no FAIL)")
            print(f"[PREREG-PREFLIGHT][NEGATIVE-CONTROL][{'PASS' if ok else 'FAIL'}] "
                  f"name={name} expect={want or 'PASS'}")
            if not ok:
                bad += 1
                print(f"    got: {got}")
    print(f"SELFTEST_NC total={len(cases)} failed={bad}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prereg", nargs="?", help="사전등록 마크다운 경로")
    ap.add_argument("--root", default=".", help="저장소 루트")
    ap.add_argument("--trials", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.prereg:
        ap.error("사전등록 경로가 필요하다 (또는 --selftest)")
    return run(Path(a.prereg), Path(a.root), a.trials, a.seed)


if __name__ == "__main__":
    sys.exit(main())
