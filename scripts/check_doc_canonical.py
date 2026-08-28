#!/usr/bin/env python3
"""인용하려는 문서가 그 계열의 정본인지 검사한다 (fail-closed).

`X.md` 와 `X.md.rev1`·`.rev2`… 가 공존할 때 어느 것이 정본인지는 이름으로 알 수 없다.
이 캠페인의 규약은 「base = 최신, .revN = 보존본」이지만, 발주서가 그것을 거꾸로 지정한
실사고가 있었다(2026-08-28, A210-Z2A R-III/IV 판정 1차 발주 폐기).

usage: check_doc_canonical.py <인용하려는 경로>...
rc=0 전건 정본 · rc=1 정본 아닌 것을 인용 · rc=2 사용법/부재
"""
import subprocess, sys, pathlib

def newest(sibs):
    def key(p):
        try:
            t = subprocess.run(["git","log","-1","--format=%ct","--",str(p)],
                               capture_output=True, text=True).stdout.strip()
        except Exception:
            t = ""
        return (int(t) if t.isdigit() else 0, p.stat().st_mtime, len(p.read_bytes()))
    return max(sibs, key=key)

def main(argv):
    if not argv: print(__doc__); return 2
    bad = 0
    for a in argv:
        p = pathlib.Path(a)
        if not p.exists(): print(f"[DOC-CANON][FAIL] absent {a}"); bad = 1; continue
        base = pathlib.Path(str(p).split(".md")[0] + ".md")
        sibs = sorted({base, *base.parent.glob(base.name + ".rev*")})
        sibs = [s for s in sibs if s.exists()]
        if len(sibs) < 2:
            print(f"[DOC-CANON][PASS] single {p.name}"); continue
        n = newest(sibs)
        others = " ".join(s.name for s in sibs if s != n)
        if n.resolve() == p.resolve():
            print(f"[DOC-CANON][PASS] canonical={p.name}  (보존본: {others})")
        else:
            print(f"[DOC-CANON][FAIL] cited={p.name} but canonical={n.name}  (계열: {' '.join(s.name for s in sibs)})")
            bad = 1
    return bad

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
