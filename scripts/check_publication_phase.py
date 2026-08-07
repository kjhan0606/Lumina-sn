#!/usr/bin/env python3
"""**코드 검사** — 발행 위상(R7)이 실제로 맞는가.  런 로그만으로 판정한다.

분담 개정10(2026-08-07): 코딩=Codex, **검사=운전석**.
이 검사는 Codex 의 패치를 읽지 않는다 — **런이 낸 이벤트 순서**만 본다.
그래야 "코드는 그럴듯한데 런은 다르다" 를 잡는다(오늘 그런 일이 여러 번 있었다).

## 배선도가 정한 한 반복의 순서 (OUT_F R7)

    commit(r) → view(r) → a208(o=r) → a209(e=r) → A2-10(t→t+1) → 물질 갱신

즉 **A2-10 호출 시점에 opacity·emissivity·radiation 이 같은 세대**여야 한다.
현재(수리 전)는:
  · MC lane   : a208 → **T_e** → a209   (a209 가 T_e 뒤)
  · pure lane : a208 만, **a209 없음**  ⟹ emissivity com=0

## 판정

`[A2-10][PRE]` 줄을 읽는다(운전석이 계측으로 심어 둔 것):
    te_gen · radfield status/gen · line status/gen · opacity req/com/rad/pop · emissivity com

  P1 emissivity com > 0                      — pure lane 에 a209 가 생겼는가
  P2 opacity com == emissivity com           — 동세대인가
  P3 opacity rad == radfield gen             — opacity 가 현 복사장에 결박됐는가
  P4 A2-10 이 blocked_stale 로 죽지 않는가
  P5 line status/gen                         — R6 소관(기대 결과로 분리 보고)

★P5 는 **이 단의 실패가 아니다**.  R7 만으로 line-J̄ 가 생기지 않으므로
`line: status=-1 gen=0` 은 사전등록된 기대 결과다.  섞어서 판정하지 않는다.

usage: check_publication_phase.py <run.log> [...]
rc=0 P1~P4 통과 · rc=2 하나라도 위반 · rc=3 PRE 줄이 없음
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# 형식 1 — 수리 전(운전석 계측).  a209 가 아예 없던 시절의 진단.
PRE_OLD = re.compile(
    r"\[A2-10\]\[PRE\]\s+iter=(\d+)\s+te_gen=(\d+)\s+\|\s+radfield:\s+status=(-?\d+)\s+gen=(\d+)"
    r"\s+\|\s+line:\s+status=(-?\d+)\s+gen=(\d+)"
    r"\s+\|\s+opacity:\s+req=(\d+)\s+com=(\d+)\s+rad=(\d+)\s+pop=(\d+)"
    r"\s+\|\s+emissivity:\s+com=(\d+)"
    r"\s+\|\s+A2-10\s+blocked_stale=(\d+)\s+missing_term=(\d+)\s+schema=(\d+)")

# 형식 2 — R7 수리 후.  헬퍼가 lane 을 붙이고, A2-10 에 **도달한 경우에만** 찍는다.
PRE_R7 = re.compile(
    r"\[A2-10\]\[PRE\]\s+lane=(\w+)\s+iter=(\d+)\s+te_gen=(\d+)"
    r"\s+rad=(\d+)\s+line=(\d+)\s+opacity=(\d+)\s+emissivity=(\d+)\s+population=(\d+)")

# ★R7 이후 판정에 반드시 필요한 것 — A2-10 에 **도달하지 못한** 경우의 위상 기록.
# PRE 줄이 없다고 rc=3(계측 누락)으로 읽으면 안 된다.  차단도 관측이다.
PHASE = re.compile(
    r"\[R7\]\[PHASE\]\s+lane=(\w+)\s+iter=(\d+)\s+phase=(\w+)(.*)")
BLOCK = re.compile(
    r"\[(A2-0[89]|A2-10)\]\[(BLOCKED|FATAL)\]\s+event=(\w+)\s+lane=(\w+)\s+iter=(\d+)(.*)")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1]); return 3
    total_bad, seen = 0, 0
    for p in argv[1:]:
        name = Path(p).name
        txt = Path(p).read_text(errors="ignore")

        # ---- 위상 궤적 (R7 이후) : 어디까지 갔는지가 판정의 절반이다 ----
        phases = [(m.group(1), int(m.group(2)), m.group(3), m.group(4).strip())
                  for m in PHASE.finditer(txt)]
        blocks = [(m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5)),
                   m.group(6).strip()) for m in BLOCK.finditer(txt)]
        if phases or blocks:
            reached = {}
            for lane, it, ph, _ in phases:
                reached.setdefault((lane, it), []).append(ph)
            for (lane, it), phs in sorted(reached.items()):
                print(f"  {name} lane={lane} iter={it}: 위상 {' -> '.join(phs)}")
            for site, kind, ev, lane, it, rest in blocks:
                print(f"      [{site}][{kind}] {ev} lane={lane} iter={it}  {rest}")

        # ---- 형식 1 (수리 전) ----
        for m in PRE_OLD.finditer(txt):
            seen += 1
            (it, te, rst, rgen, lst, lgen,
             oreq, ocom, orad, opop, ecom, bstale, bmiss, bsch) = (int(x) for x in m.groups())
            bad = []
            if ecom == 0:
                bad.append("P1 emissivity com=0 — a209 발행이 없다(pure lane)")
            if ecom and ocom != ecom:
                bad.append(f"P2 동세대 아님: opacity com={ocom} vs emissivity com={ecom}")
            if orad != rgen:
                bad.append(f"P3 opacity 가 현 복사장에 결박 안 됨: rad={orad} vs radfield gen={rgen}")
            if bstale:
                bad.append(f"P4 A2-10 blocked_stale={bstale}")
            tag = "PASS" if not bad else "FAIL"
            print(f"  {name} iter={it} [수리전형식]: {tag}"
                  f"  [te_gen={te} rad={rst}/{rgen} opac={ocom}/{orad} emiss={ecom}]")
            for b in bad:
                print(f"      **{b}**")
            if lst != 0 or lgen == 0:
                print(f"      (R6 소관·기대 결과) line: status={lst} gen={lgen}")
            total_bad += len(bad)

        # ---- 형식 2 (R7 이후) : A2-10 에 도달한 반복만 찍힌다 ----
        for m in PRE_R7.finditer(txt):
            seen += 1
            lane = m.group(1)
            it, te, rgen, lgen, ocom, ecom, mgen = (int(x) for x in m.groups()[1:])
            bad = []
            if ecom == 0:
                bad.append("P1 emissivity com=0 — a209 발행이 없다")
            if ocom != rgen:
                bad.append(f"P2/P3 opacity({ocom}) != radiation({rgen})")
            if ecom != rgen:
                bad.append(f"P2 emissivity({ecom}) != radiation({rgen})")
            if lgen != rgen:
                bad.append(f"P5 line view({lgen}) != radiation({rgen})")
            tag = "PASS" if not bad else "FAIL"
            print(f"  {name} lane={lane} iter={it}: {tag}"
                  f"  [te={te} r={rgen} line={lgen} o={ocom} e={ecom} m={mgen}]")
            for b in bad:
                print(f"      **{b}**")
            total_bad += len(bad)

    if not seen:
        # ★차단도 관측이다.  위상 궤적이 있으면 "계측 누락"이 아니다.
        print("  A2-10 에 도달한 반복이 없다 — 위상 궤적으로 판정하라(위 줄).")
        return 4
    print(f"\nPUBLICATION_PHASE records={seen} violations={total_bad} "
          f"verdict={'PASS' if not total_bad else 'FAIL'}")
    return 0 if not total_bad else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
