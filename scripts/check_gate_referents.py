#!/usr/bin/env python3
"""사전등록 게이트의 **참조 대상이 실물로 확정되는가** (fail-closed).

기존 `check_prereg_preflight.py` 는 **정적 참조**를 본다. 이 게이트는 그것이 원리적으로 못 잡는
계급 — **참조는 성한데 뜻이 비어 있는 것** — 을 본다.

실사고 2건 (DET-SUPERLEVEL-B, 2026-08-29, 발주 2회 연속 중단. 두 번 다 프리플라이트 rc=0):
  1차: 「최저 FPC 행 **4필드** hex」가 어느 4필드인지 미열거 · 「**등록 산출물 sha**」의 대상
       부재 · 「sha **4종**」 대 §7-1 표의 3종 · 「BL∩FPC 95행(NEG 46 포함)」 축 혼동
  2차: 「**봉인 4곳** `sha256sum -c`」인데 L6·L6C 의 manifest 가 실물에 없다

검사:
  R-A COUNT_VS_ENUM  게이트/기대치 행의 「N곳·N종·N파일·N필드·N개」가 문서의 열거와 어긋난다
  R-B PATH_ABSENT    문서가 인용한 구체 경로(백틱)가 실물에 없다 — 변경집합 신설분은 면제
  R-C OPERAND_VAGUE  기계 연산(`sha256sum -c`·sha 대조·byte-동일·cmp)을 요구하면서 피연산자를
                     구체 경로로도 절 지목(§N)으로도 특정하지 않는다

usage: check_gate_referents.py <사전등록.md> [--root <repo>]   |   --selftest
rc=0 clean · rc=1 위반 · rc=2 사용법
"""
import re, sys, os, pathlib, tempfile

GATE_ROW = re.compile(r'^\|\s*\*{0,2}(G-[A-Z0-9]+|E-\d+)\b')
QUANT    = re.compile(r'(?<![0-9])([0-9]{1,3})\s*(곳|종|파일|필드|개)\b')
OPS      = re.compile(r'sha256sum\s+-c|SHA-?256|\bsha\b|byte-?동일|\bcmp\b', re.I)
PATHISH  = re.compile(r'`([^`\s]*/[^`\s]*|[A-Za-z0-9_.-]+\.(?:sha|json|csv|py|sh|bin|txt))`')
SECREF   = re.compile(r'§\s*[0-9]|\b(?:G-[A-Z0-9]+|E-[0-9]+)\b')   # 절·게이트 상호참조 둘 다 정당한 지목

def enumerations(text, noun):
    """문서가 그 명사를 몇 개로 열거하는가 — 괄호 목록(A·B·C) 또는 백틱 토큰 개수."""
    best = 0
    pat = (re.escape(noun) + r'\s*(?:[0-9]{1,3}\s*(?:곳|종|파일|필드|개))?\s*'
           r'(?:\(([^)]*)\)|:\s*([^\n|]*))')
    for m in re.finditer(pat, text):
        body = m.group(1) or m.group(2) or ""
        if re.fullmatch(r'\s*§\s*[0-9][^)]*', body):
            return -1                                   # 정의가 다른 절에 있다 — 계수 검사 면제
        parts = [p for p in re.split(r'[·,、/]| 및 ', body) if p.strip()]
        best = max(best, len(parts))
    return best

def scan(text, root=None, changeset=()):
    hits = []
    lines = text.split("\n")
    for i, ln in enumerate(lines, 1):
        if not GATE_ROW.match(ln.strip()):
            continue
        for n, unit in QUANT.findall(ln):
            n = int(n)
            noun_m = re.search(r'([가-힣A-Za-z_]+)\s*' + n.__str__() + r'\s*' + unit, ln)
            noun = noun_m.group(1) if noun_m else ""
            if not noun:
                continue
            got = enumerations(text, noun)
            if got == -1:
                continue
            tail = ln[ln.find(str(n)) :]
            if len(re.findall(r'`[^`]+`', tail)) >= n:     # 같은 행에 백틱으로 인라인 열거
                continue
            if got and got != n:
                hits.append(("R-A", "COUNT_VS_ENUM", i, f"{noun}{n}{unit} but enumerated={got}"))
            elif got == 0:
                hits.append(("R-A", "COUNT_VS_ENUM", i, f"{noun}{n}{unit} but enumerated=0 (미열거)"))
        if OPS.search(ln):
            own = GATE_ROW.match(ln.strip()).group(1)
            rest = ln.replace(own, "", 1)
            if not PATHISH.search(rest) and not SECREF.search(rest):
                hits.append(("R-C", "OPERAND_VAGUE", i, "기계 연산에 피연산자 특정 없음(경로·§지목 둘 다 부재)"))
    if root:
        declared_new = set()                      # 문서가 어디서든 «신설» 로 선언한 경로는 전역 면제
        for l in text.split("\n"):
            if "신설" in l:
                declared_new |= {x for x in PATHISH.findall(l)}
        for m in PATHISH.finditer(text):
            p = m.group(1)
            if p in changeset or p.startswith("<") or "…" in p or "*" in p:
                continue
            p = re.sub(r':[0-9]+(-[0-9]+)?$', '', p)          # 소스 인용의 행번호 접미
            if not re.match(r'^(src|scripts|docs|tests|validation)/', p):
                continue                                       # 저장소 상대 경로만 (런 상대·수식 제외)
            lno = text[:m.start()].count("\n") + 1
            if m.group(1) in declared_new:
                continue
            if not os.path.exists(os.path.join(root, p)):
                hits.append(("R-B", "PATH_ABSENT", lno, p))
    return hits

def run(md, root, quiet=False):
    t = pathlib.Path(md).read_text(errors="replace")
    cs = set(re.findall(r'^\|\s*`([^`]+)`\s*\|', t, re.M))
    hits = scan(t, root, cs)
    for cid, name, ln, det in hits:
        print(f"[GATE-REF][FAIL] {name} {md}:{ln}  {det}  ({cid})")
    if not hits:
        print(f"[GATE-REF][PASS] {md}  changeset_exempt={len(cs)}")
    return 1 if hits else 0

CLEAN = """
## §9 게이트
| **G-1** sha 대조 | 덱 2 + 바이너리 1 = sha 3종 일치 (`docs/a.md`) | 유 |
| **G-7** 봉인 사후 | 봉인 2곳 `sha256sum -c` non-OK 0 — §7-2 목록 | 유 |
| E-1 | 스테이징: sha 3종 일치 (§7-1) | 1 |
본문: sha 3종(덱 levels·덱 line_list·바이너리) · 봉인 2곳(census·R-II)
"""
INJ = {
 "2차 재현 — 봉인 4곳인데 열거 2":       ("봉인 2곳 `sha256sum -c`", "봉인 4곳 `sha256sum -c`"),
 "1차③ 재현 — sha 4종 대 열거 3":        ("E-1 | 스테이징: sha 3종", "E-1 | 스테이징: sha 4종"),
 "1차① 재현 — 4필드 미열거":            ("| E-1 | 스테이징: sha 3종 일치 (§7-1) | 1 |",
                                          "| E-1 | 앵커 4필드 hex 등록 (§7-1) | 1 |"),
 "1차② 재현 — 연산 피연산자 부재":      ("| **G-7** 봉인 사후 | 봉인 2곳 `sha256sum -c` non-OK 0 — §7-2 목록 | 유 |",
                                          "| **G-7** 미러 | 등록 산출물 sha 동일 | 유 |"),
}

def selftest():
    f = 0
    ok = not scan(CLEAN)
    print(f"  NC-F0 성한 계약 PASS      : {'OK' if ok else 'FAIL ' + str(scan(CLEAN))}")
    f += 0 if ok else 1
    for label, (old, new) in INJ.items():
        assert old in CLEAN, label
        h = scan(CLEAN.replace(old, new, 1))
        want = "OPERAND_VAGUE" if "피연산자" in label else "COUNT_VS_ENUM"
        ok = any(x[1] == want for x in h)
        print(f"  {label:34s}: {'OK ' + want if ok else 'FAIL got=' + str([x[1] for x in h])}")
        f += 0 if ok else 1
        f += 0 if not scan(CLEAN) else 1
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "c.md"; p.write_text(CLEAN)
        h = scan(CLEAN, root=d)
        ok = any(x[1] == "PATH_ABSENT" for x in h)
        print(f"  {'NC-F5 경로 부재 검출':34s}: {'OK' if ok else 'FAIL'}")
        f += 0 if ok else 1
    print(f"[GATE-REF][{'PASS' if not f else 'FAIL'}] selftest failed={f}")
    return 1 if f else 0

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--selftest"]: sys.exit(selftest())
    if not a: print(__doc__); sys.exit(2)
    root = a[a.index("--root") + 1] if "--root" in a else None
    sys.exit(run(a[0], root))
