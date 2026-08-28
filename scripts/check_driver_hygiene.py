#!/usr/bin/env python3
"""운전석 위생 린트 — 실사고 4건을 기계로 막는다 (fail-closed).

NC 내장: --selftest 는 성한 입력 PASS + 결함 주입 FAIL 을 전건 시연한다.

검사 (전부 2026-08-28 실사고):
  H1 RC_AFTER_PIPE  `... | tail` 뒤 `$?`/`RC=$?` — pipefail 없으면 파이프 말단의 rc다
  H2 SACCT_NO_M     `sacct -j` 에 `-M <cluster>` 부재 — sacct 는 클러스터 범위다
  H3 TMP_CROSSNODE  ssh/sbatch 로 타 노드에 넘기는 명령 안의 `/tmp` — /tmp 는 노드-로컬
  H4 PY_NO_B        저장소 스크립트를 `python3 -B` 없이 실행 — .pyc 가 프리플라이트를 죽인다

usage: check_driver_hygiene.py <파일>...   |   --selftest
rc=0 clean · rc=1 위반 · rc=2 사용법
"""
import re, sys, pathlib, tempfile, os

CHECKS = [
    ("H1", "RC_AFTER_PIPE",
     lambda t: [i for i, l in enumerate(t, 1)
                if re.search(r'(RC=)?\$\?', l)
                and any(re.search(r'\|\s*(tail|head|grep|sed|awk|tee)\b', p) for p in t[max(0, i-3):i-1])
                and not any(re.search(r'set\s+-o\s+pipefail|set\s+-\w*o?\s*pipefail', p) for p in t[:i])]),
    ("H2", "SACCT_NO_M",
     lambda t: [i for i, l in enumerate(t, 1)
                if re.search(r'\bsacct\b', l) and re.search(r'-j\s*\S', l) and not re.search(r'-M\s*\S', l)]),
    ("H3", "TMP_CROSSNODE",
     lambda t: [i for i, l in enumerate(t, 1)
                if re.search(r'\b(ssh|sbatch|srun)\b', l) and re.search(r'(?<![\w/])/tmp/', l)]),
    ("H4", "PY_NO_B",
     lambda t: [i for i, l in enumerate(t, 1)
                if re.search(r'\bpython3?\s+(?!-B\b)(?!-c\b)(?!-m\b)\S*scripts/\S+\.py', l)]),
]

def scan(text):
    lines = text.split("\n")
    return [(cid, name, n) for cid, name, fn in CHECKS for n in fn(lines)]

def run(paths):
    bad = 0
    for p in paths:
        f = pathlib.Path(p)
        if not f.exists():
            print(f"[DRV-HYG][FAIL] absent {p}"); bad = 1; continue
        hits = scan(f.read_text(errors="replace"))
        if hits:
            bad = 1
            for cid, name, n in hits:
                print(f"[DRV-HYG][FAIL] {name} {p}:{n}  ({cid})")
        else:
            print(f"[DRV-HYG][PASS] {p}")
    return bad

CLEAN = """#!/usr/bin/env bash
set -o pipefail
cd /repo
python3 -B scripts/analyze_x.py --out /gpfs/o.json 2>&1 | tail -1
echo "RC=$?"
sacct -M syntax -j 325913 -X -n -P -o State
ssh grammar "ssh grammar-debug 'cd /repo && python3 -B scripts/y.py'"
cp /gpfs/a /gpfs/b
"""
INJECT = {
    "H1": ("set -o pipefail\n", "\n"),
    "H2": ("sacct -M syntax -j 325913", "sacct -j 325913"),
    "H3": ("ssh grammar \"ssh grammar-debug 'cd /repo && python3 -B scripts/y.py'\"",
           "ssh grammar 'bash /tmp/run.sh'"),
    "H4": ("python3 -B scripts/analyze_x.py", "python3 scripts/analyze_x.py"),
}

def selftest():
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        base = pathlib.Path(d) / "clean.sh"; base.write_text(CLEAN)
        hits = scan(CLEAN)
        ok = not hits
        print(f"  NC-H0 성한 입력 PASS : {'OK' if ok else 'FAIL ' + str(hits)}")
        fails += 0 if ok else 1
        for cid, (old, new) in INJECT.items():
            assert old in CLEAN, cid
            hits = scan(CLEAN.replace(old, new, 1))
            got = {h[0] for h in hits}
            ok = cid in got
            print(f"  NC-{cid} 주입 → {cid} 발화 : {'OK' if ok else 'FAIL got=' + str(got)}")
            fails += 0 if ok else 1
            hits2 = scan(CLEAN)
            ok2 = not hits2
            print(f"  NC-{cid} 복원 → PASS      : {'OK' if ok2 else 'FAIL'}")
            fails += 0 if ok2 else 1
    print(f"[DRV-HYG][{'PASS' if not fails else 'FAIL'}] selftest failed={fails}")
    return 1 if fails else 0

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit(2)
    sys.exit(selftest() if a[0] == "--selftest" else run(a))
