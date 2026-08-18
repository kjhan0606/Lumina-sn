# Fable 총괄계획 정정 요청 — diagnostic completion과 final non-census PASS 분리

선행 응답은
`docs/FABLE_PLAN_ANALYSIS_TRANSFER_A210_2026-08-17.md`에 원문 봉인했다.
Stage 1--4의 방향과 V4/J/O 분기 규칙은 유지할 수 있다. 그러나 Stage 5 및
FINAL GATE EVIDENCE 3/5가 실제 completion auditor 계약과 충돌한다.

## 실제 코드 계약

`scripts/finalize_a210_nonoverlap_gate.py`의 `audit_run`은 final PASS에 다음을 모두
강제한다.

- line 416: `model.rc == 0`
- line 418--420: `TARGETED_GATE_VERDICT.txt`에
  `A210_TARGETED_GATE_ACCEPT status=PASS`
- line 422--439: targeted report `status=PASS`, exact publications 2,
  radiation generations `[1,2]`, seed commit, 109,014,300 Sobolev Jbar cells,
  **R7 material commit와 physics comparison commit 둘 다 true**, census absent,
  repair false
- line 458 이후: R1 sealed-reference comparison PASS
- line 502--515: tripwire `COMPLETED`, `child.rc=0`, FAILED/YIELDED 없음

positive selftest도 `model.rc=0`, child rc 0, targeted verdict PASS를 만든다.
`docs/CURRENT_PLAN.md:3121--3124`는 K24/K30이 `model.rc=1`과 R7 no-bracket 때문에
completion auditor 요건을 못 채웠고 숨은 completion artifact가 없다고 이미 명시한다.

현재 K36 line-saturation run은 **원인 진단 run**이다. 의도된 자연 결과는
`model.rc=1 + REQUESTED_TE RADEQ_NO_BRACKET`이며, 이는 V2/V3/V4를 허용하는
diagnostic completion이지 final non-census PASS가 아니다. 이 run이 같은 binary SHA를
사용한다는 사실도 R7 commit을 대신하지 않는다.

## 정정 질문

총괄·계획 및 분석·평가 담당자로서 아래 네 항목만 간결하게 정정하라.

1. K36 diagnostic rc=1을 final gate PASS에서 명시적으로 제외하라.
2. Stage 4 J/O 결과가 귀속 또는 미결일 때 각각 어떤 추가 read-only/localization 또는
   물리 구현 단계를 거쳐야 `model.rc=0` non-census 재실행이 정당화되는지 정하라.
3. 미결 상태가 final PASS를 막지 않는다는 선행 문장을 철회할지 판정하라. 실제 auditor를
   바꾸지 말고 현재 목표와 계약을 기준으로 답하라.
4. 수정된 Stage 5 PASS 조건과 FINAL GATE EVIDENCE를 실제 auditor 필드에 맞춰 제시하라.

절대 제약은 동일하다: 물리값 floor/cap/clamp/jitter/repair/scaling/deletion 금지,
pre-core tau refresh 부활 금지, coevolution barrier 보존, tripwire 충돌 회피.
파일 수정·코딩·process 신호·작업 제출은 하지 말고 계획/판정만 하라.

응답 형식: `CORRECTION VERDICT`, `REVISED STAGE 4`, `REVISED STAGE 5`,
`REVISED FINAL GATE EVIDENCE`.
