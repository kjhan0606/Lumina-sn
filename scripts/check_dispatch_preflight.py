#!/usr/bin/env python3
"""발주 프리플라이트 — 발주서가 계약을 재서술하지 않았는지 기계로 검사한다.

신설 근거(개정15, 2026-08-22): 발주 중단 9건 중 **4건이 운전석 발주서의 결함**이었고
넷 다 같은 기전이었다 — 뒷겹이 계약의 값(파일 수·NC 목록·커밋 해시·요구 항목)을
**재서술**했고, 재서술은 표류했다. 개정14 의 "재서술 금지" 는 **앞겹만** 말했다.

DP-1 재서술 탐지 · DP-2 앞겹 동일성(계약이 HEAD 에 커밋돼 있는가) · DP-3 계약 자체 통과.
fail-closed: 뒷겹 경계(§0 경위 절)를 못 찾으면 거부한다.
"""
from __future__ import annotations
import argparse, re, subprocess, sys, tempfile, hashlib
from pathlib import Path

# 계약 값 토큰 — 이 캠페인에서 실제로 표류한 종류만 (오탐 최소화)
TOKENS = [
    ("path",     re.compile(r'\b(?:src|tests|scripts|docs)/[A-Za-z0-9_.\-]+\.[A-Za-z0-9]+')),
    ("nc-name",  re.compile(r'\bNC-[A-Z]+[0-9]+[a-z]?\b')),
    ("gate-name",re.compile(r'\b(?:PC|G|W|R|P|E)-?[0-9]+[a-z]?\b(?=\s*(?:~|·|,|\)|:|을|를|이|가|와|과|의))')),
    ("hex",      re.compile(r'\b[0-9a-f]{7,40}\b')),
    ("count",    re.compile(r'\b[0-9]+\s*(?:종|파일|건)\b')),
]
EXEMPT_MARK = "이 절의 값은 요구가 아니다"


class Fail(Exception):
    def __init__(self, reason, detail=""):
        super().__init__(reason); self.reason, self.detail = reason, detail


def emit(kind, reason, detail=""):
    print(f"[DISPATCH-PREFLIGHT][{kind}] reason={reason}" + (f" detail={detail}" if detail else ""))


def split_order(text: str):
    """§0 경위 절(면제)과 그 이후(검사 대상)를 가른다."""
    m = re.search(r'^##\s*1\.', text, re.M)
    if not m:
        raise Fail("dispatch-shape-unknown", "'## 1.' 절 경계를 찾지 못했다 (fail-closed)")
    return text[:m.start()], text[m.start():]


def strip_contract_pointer(body: str, contract_rel: str) -> str:
    """계약 문서 자신의 경로는 재서술이 아니다 — 개정14 가 **요구하는** 지목이다.

    오탐 실측(2026-08-22, 게이트 첫 실전): 앞겹이 계약 파일을 지목한 것을
    DISPATCH-RESTATES-CONTRACT 로 잡았다. 재분장 문서 §5-1 이 "DP-1 의 오탐률 — 값 토큰
    추출이 정당한 집행 조건까지 잡을 수 있다. 구현 후 실측한다" 고 예고한 그 자리다.
    면제는 **계약 자신의 경로 하나**로 좁힌다 — 다른 파일 경로는 그대로 잡힌다.
    """
    for form in (f"`{contract_rel}`", contract_rel):
        body = body.replace(form, "<계약>")
    return body


def dp1_restatement(text: str, contract: str, contract_rel: str = "") -> None:
    head, body = split_order(text)
    if contract_rel:
        body = strip_contract_pointer(body, contract_rel)
    if any(t.search(head) for _, t in TOKENS) and EXEMPT_MARK not in head:
        raise Fail("exempt-section-unmarked",
                   f"§0 에 계약 값이 있으나 '{EXEMPT_MARK}' 문구가 없다")
    hits = []
    for name, rx in TOKENS:
        for m in rx.finditer(body):
            tok = m.group(0)
            if tok in contract:                       # 계약에 등장하는 값 = 재서술
                hits.append(f"{name}:{tok}")
    hits = sorted(set(hits))
    if hits:
        raise Fail("DISPATCH-RESTATES-CONTRACT", ",".join(hits[:8]) + (f",+{len(hits)-8}" if len(hits) > 8 else ""))
    emit("PASS", "no-restatement", f"tokens_checked={len(TOKENS)}")


def dp2_frontpage(contract_path: Path, root: Path) -> None:
    rel = contract_path.resolve().relative_to(root.resolve()).as_posix()
    r = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{rel}"],
                       capture_output=True)
    if r.returncode != 0:
        raise Fail("contract-not-committed", rel)
    if hashlib.sha256(r.stdout).hexdigest() != \
       hashlib.sha256(contract_path.read_bytes()).hexdigest():
        raise Fail("contract-worktree-drift", rel)
    emit("PASS", "frontpage-sealed", rel)


def dp3_contract(contract_path: Path, root: Path) -> None:
    tool = root / "scripts" / "check_prereg_preflight.py"
    if not tool.exists():
        raise Fail("prereg-preflight-missing", str(tool))
    r = subprocess.run([sys.executable, str(tool), str(contract_path), "--root", str(root),
                        "--trials", "20000"], capture_output=True, text=True)
    if r.returncode != 0:
        raise Fail("contract-preflight-failed",
                   next((l for l in r.stdout.splitlines() if "[FAIL]" in l), "?"))
    emit("PASS", "contract-preflight", "rc=0")


def run(order: Path, contract: Path, root: Path) -> int:
    try:
        try:
            crel = contract.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            crel = contract.name
        dp1_restatement(order.read_text(), contract.read_text(), crel)
        dp2_frontpage(contract, root)
        dp3_contract(contract, root)
    except Fail as f:
        emit("FAIL", f.reason, f.detail); return 1
    emit("PASS", "dispatch-preflight", order.name); return 0


# ---------------- 음성대조 (게이트 자신도 주입 결함으로 FAIL 을 시연해야 한다) ----------------
GOOD_CONTRACT = """# 계약
### scripts/ — 도구
| `scripts/tool_alpha.py` | NC-Q1 · NC-Q2 를 내장 |
### 변경집합 끝
| **PC-1** | 요구 | 증거 | NC-Q1 주입 → FAIL |
"""
GOOD_ORDER = """# 발주
## 0. 경위
지난 회차는 NC-Q2 누락으로 멈췄다. 이 절의 값은 요구가 아니다.
## 1. 앞겹
`docs/CONTRACT.md` 를 전문 읽어라. 어긋나면 멈추고 보고하라.
## 2. 뒷겹
- 구현 대상 = §4 표의 행 전부. 이 발주서는 파일을 열거하지 않는다 — 의도적이다.
- §5 가 요구하는 음성대조 전부를 구현하라.
- blob 기준은 §4 가 지정한 값을 쓰라.
- commit 금지. 네트워크 금지. cwd = clean worktree.
## 3. 보고
§5 의 음성대조 전건의 실제 출력을 보고하라.
"""


def selftest() -> int:
    import io, contextlib, os
    cases = []

    def case(name, want, order=None, contract=None, commit=True, mangle=None, decl=True):
        cases.append((name, want, order, contract, commit, mangle, decl))

    case("NC-D0", None)                                              # ★성한 발주서 -> PASS
    case("NC-D1a", "DISPATCH-RESTATES-CONTRACT",
         order=GOOD_ORDER.replace("§5 가 요구하는 음성대조 전부를",
                                  "NC-Q1 · NC-Q2 를"))              # 실제 사고 9 의 형태
    case("NC-D1b", "DISPATCH-RESTATES-CONTRACT",
         order=GOOD_ORDER.replace("§4 표의 행 전부",
                                  "`scripts/tool_alpha.py` 1파일"))  # 실제 사고 1 의 형태
    case("NC-D1c", "exempt-section-unmarked",
         order=GOOD_ORDER.replace("이 절의 값은 요구가 아니다.", ""))
    case("NC-D1d", "dispatch-shape-unknown",
         order=GOOD_ORDER.replace("## 1. 앞겹", "앞겹"))              # fail-closed
    case("NC-D0b", None,                                              # ★오탐 회귀: 계약 지목은 면제
         order=GOOD_ORDER.replace("`docs/CONTRACT.md` 를 전문 읽어라.",
                                  "`docs/CONTRACT.md` 를 전문 읽어라. docs/CONTRACT.md 가 유일 권위다."))
    case("NC-D1e", "DISPATCH-RESTATES-CONTRACT",                      # 계약 아닌 경로는 여전히 잡힌다
         order=GOOD_ORDER.replace("§4 표의 행 전부", "`scripts/tool_alpha.py` 의 행 전부"))
    case("NC-D2", "contract-not-committed", commit=False)
    case("NC-D2b", "contract-worktree-drift", mangle="drift")
    case("NC-D3", "contract-preflight-failed", decl=False)   # 선언 블록 없는 계약

    bad = 0
    for name, want, order, contract, commit, mangle, decl_on in cases:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "docs").mkdir(); (root / "scripts").mkdir()
            for cmd in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
                subprocess.run(["git", "-C", str(root)] + cmd, capture_output=True)
            # 계약이 프리플라이트를 통과하도록 선언 블록을 붙인다
            decl = ('\n```prereg-preflight\n{"changeset":{"table_heading":"### scripts/ — 도구",'
                    '"table_end":"### 변경집합 끝","path_pattern":"scripts/[a-z0-9_]+\\\\.py",'
                    '"symbol":"NC-Q1","roots":["scripts"],'
                    '"expected_extra":["scripts/tool_alpha.py"]},'
                    '"branches":{"metrics":{"f":"sum(1 for x in v if x>0.5)/len(v)"},'
                    '"rules":[{"name":"A","predicate":"f == 0.0"},{"name":"B","predicate":"f > 0.0"}],'
                    '"residual":"C"},'
                    '"references":[{"path":"scripts/tool_alpha.py","flags_existing":[]}]}\n```\n')
            (root / "scripts" / "tool_alpha.py").write_text("# NC-Q1\n")
            cpath = root / "docs" / "CONTRACT.md"
            cpath.write_text((contract or GOOD_CONTRACT) + (decl if decl_on else ""))
            (root / "scripts" / "check_prereg_preflight.py").write_text(
                (Path(__file__).parent / "check_prereg_preflight.py").read_text())
            if commit:
                subprocess.run(["git", "-C", str(root), "add", "-A"], capture_output=True)
                subprocess.run(["git", "-C", str(root), "commit", "-qm", "x"], capture_output=True)
            if mangle == "drift":
                cpath.write_text(cpath.read_text() + "\n작업트리에서만 바뀐 줄\n")
            opath = root / "ORDER.md"; opath.write_text(order or GOOD_ORDER)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = run(opath, cpath, root)
            out = buf.getvalue()
            ok = (rc == 0 and "[FAIL]" not in out) if want is None else \
                 (rc == 1 and f"reason={want}" in out)
            print(f"[DISPATCH-PREFLIGHT][NEGATIVE-CONTROL][{'PASS' if ok else 'FAIL'}] "
                  f"name={name} expect={want or 'PASS'}")
            if not ok:
                bad += 1
                print("    got:", (out.strip().splitlines() or ["(무출력)"])[-1])
    print(f"SELFTEST_NC total={len(cases)} failed={bad}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("order", nargs="?", help="발주서 경로")
    ap.add_argument("contract", nargs="?", help="사전등록(계약) 경로")
    ap.add_argument("--root", default=".")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.order and a.contract):
        ap.error("발주서와 계약 경로가 필요하다 (또는 --selftest)")
    return run(Path(a.order), Path(a.contract), Path(a.root))


if __name__ == "__main__":
    sys.exit(main())
