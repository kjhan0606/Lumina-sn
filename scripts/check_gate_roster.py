#!/usr/bin/env python3
"""계약 게이트 표 ↔ 집행 기록 대조 (fail-closed).

판정문 DET-INSTRUMENT-REPAIR §10 이 지목한 공백을 메운다:
  "«계약 §N 게이트 표의 전 행 ↔ 집행 산출물» 대조를 검사하는 게이트는 없다 — 막는 기계는 0"
실사고: 운전석이 9행 중 5행만 돌리고 "게이트 전부 통과"로 보고(→ 분기 IR-X).

usage: check_gate_roster.py <사전등록.md> <집행기록.json>   |   --selftest

집행기록.json = {"<게이트ID>": {"cmd": "...", "evidence": "<경로>", "result": "PASS|FAIL|..."}, ...}
  · evidence 경로가 있으면 실존을 확인한다(없으면 FAIL — 주장만으로는 집행이 아니다)
  · result 누락 → FAIL

rc=0 전건 대응 · rc=1 누락/불일치 · rc=2 사용법
"""
import re, sys, json, pathlib, tempfile

GATE_RE = re.compile(r'^\|\s*\**\s*(G-[A-Z0-9]+[a-z]?)\b')   # ★숫자 없는 게이트(G-RC·G-S)도 잡는다

def roster(md_text):
    ids = []
    for ln in md_text.split("\n"):
        m = GATE_RE.match(ln.strip())
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ids

def check(md_path, rec_path):
    md = pathlib.Path(md_path); rec = pathlib.Path(rec_path)
    if not md.exists(): print(f"[GATE-ROSTER][FAIL] absent {md_path}"); return 1
    if not rec.exists(): print(f"[GATE-ROSTER][FAIL] absent {rec_path}"); return 1
    ids = roster(md.read_text(errors="replace"))
    if not ids:
        print(f"[GATE-ROSTER][FAIL] no-gate-rows {md_path}"); return 1
    try: data = json.loads(rec.read_text())
    except Exception as e: print(f"[GATE-ROSTER][FAIL] bad-json {e}"); return 1
    bad = 0
    for g in ids:
        e = data.get(g)
        if e is None:
            print(f"[GATE-ROSTER][FAIL] MISSING_EXECUTION {g}"); bad = 1; continue
        if not e.get("result"):
            print(f"[GATE-ROSTER][FAIL] NO_RESULT {g}"); bad = 1; continue
        ev = e.get("evidence")
        if ev and not pathlib.Path(ev).exists():
            print(f"[GATE-ROSTER][FAIL] EVIDENCE_ABSENT {g} -> {ev}"); bad = 1; continue
        print(f"[GATE-ROSTER][ok] {g} = {e['result']}" + (f"  [{ev}]" if ev else "  [증거경로 미기재]"))
    extra = [k for k in data if k not in ids]
    if extra:
        print(f"[GATE-ROSTER][FAIL] UNREGISTERED_GATE {','.join(extra)}"); bad = 1
    print(f"[GATE-ROSTER][{'PASS' if not bad else 'FAIL'}] roster={len(ids)} recorded={len(data)}")
    return bad

MD = """
| **G-A1** selftest | 요구 | 담당 | 실측 | 판정 |
| **G-A2** census | 요구 | 담당 | 실측 | 판정 |
| **G-B1** l6c | 요구 | 담당 | 실측 | 판정 |
| 산문 행 — 게이트 아님 | ... |
"""

def selftest():
    f = 0
    with tempfile.TemporaryDirectory() as d:
        md = pathlib.Path(d)/"c.md"; md.write_text(MD)
        ev = pathlib.Path(d)/"e.json"; ev.write_text("{}")
        full = {g: {"result": "PASS", "evidence": str(ev)} for g in ("G-A1","G-A2","G-B1")}
        cases = [
            ("NC-R0 성한 기록 → PASS", full, 0),
            ("NC-R1 1행 누락 → FAIL",  {k: v for k, v in full.items() if k != "G-B1"}, 1),
            ("NC-R2 result 없음 → FAIL", {**full, "G-A2": {"evidence": str(ev)}}, 1),
            ("NC-R3 증거 부재 → FAIL", {**full, "G-A1": {"result":"PASS","evidence":str(pathlib.Path(d)/"nope")}}, 1),
            ("NC-R4 미등록 게이트 → FAIL", {**full, "G-Z9": {"result":"PASS"}}, 1),
        ]
        assert roster(MD) == ["G-A1","G-A2","G-B1"], roster(MD)
        print("  roster 추출 = ['G-A1','G-A2','G-B1'] : OK (산문 행 배제)")
        for label, rec, want in cases:
            p = pathlib.Path(d)/"r.json"; p.write_text(json.dumps(rec))
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): got = check(str(md), str(p))
            ok = got == want
            print(f"  {label} : {'OK' if ok else 'FAIL rc=' + str(got)}")
            f += 0 if ok else 1
    print(f"[GATE-ROSTER][{'PASS' if not f else 'FAIL'}] selftest failed={f}")
    return 1 if f else 0

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--selftest"]: sys.exit(selftest())
    if len(a) != 2: print(__doc__); sys.exit(2)
    sys.exit(check(a[0], a[1]))
