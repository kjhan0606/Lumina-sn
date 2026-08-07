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

PRE = re.compile(
    r"\[A2-10\]\[PRE\]\s+iter=(\d+)\s+te_gen=(\d+)\s+\|\s+radfield:\s+status=(-?\d+)\s+gen=(\d+)"
    r"\s+\|\s+line:\s+status=(-?\d+)\s+gen=(\d+)"
    r"\s+\|\s+opacity:\s+req=(\d+)\s+com=(\d+)\s+rad=(\d+)\s+pop=(\d+)"
    r"\s+\|\s+emissivity:\s+com=(\d+)"
    r"\s+\|\s+A2-10\s+blocked_stale=(\d+)\s+missing_term=(\d+)\s+schema=(\d+)")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1]); return 3
    total_bad, seen = 0, 0
    for p in argv[1:]:
        txt = Path(p).read_text(errors="ignore")
        for m in PRE.finditer(txt):
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
            print(f"  {Path(p).name} iter={it}: {tag}"
                  f"  [te_gen={te} rad={rst}/{rgen} opac={ocom}/{orad} emiss={ecom}]")
            for b in bad:
                print(f"      **{b}**")
            # R6 소관 — 분리 보고, 판정에 넣지 않는다
            if lst != 0 or lgen == 0:
                print(f"      (R6 소관·기대 결과) line: status={lst} gen={lgen}")
            total_bad += len(bad)
    if not seen:
        print("  [A2-10][PRE] 줄이 없다 — 계측이 빠졌거나 그 지점 이전에 죽었다")
        return 3
    print(f"\nPUBLICATION_PHASE records={seen} violations={total_bad} "
          f"verdict={'PASS' if not total_bad else 'FAIL'}")
    return 0 if not total_bad else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
