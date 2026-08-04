# 판정 규약 (VERDICT PROTOCOL) — 2026-07-29 제정, user 지시

**목적**: 판정 오류의 실증된 3대 원인 — ①입력 사슬 미확인(사례 19-21) ②잣대 미검증(사례 22-27) ③귀속을 diff 없이 서사로 수행(사례 29) — 을 절차로 차단한다.
**적용**: 원장(project_artis_parity_campaign.md)에 기재되는 **모든 판정런 판정과 인과 귀속**. 이 절차를 거치지 않은 판정은 원장에 `[미인증]`으로 표시한다.
**분담**: V0-V2 = 운전석(Fable). V1 = 기계(스크립트). V3 = **Opus 오프라인 적대검증**. V4-V5 = 운전석.

## V0. 사전등록 확인
판정 항목(F/W/M)이 **런 시작 전에** 런처 헤더 또는 judge 스크립트 docstring에 등록되어 있는가.
사후 추가 지표는 반드시 `특성화(사후)`로 표시 — 사후 지표로 PASS/FAIL 선언 금지.

## V1. 기계 관문 — `python3 scripts/verdict_gate.py <run> [--baseline <run>] [--intended-diff K=V,K,...]`
fail-closed. 비-0 종료면 **원장 기재 금지**. 검사 항목:
1. 완주(END RUN FOOTER) + 판정-핵심 산출물 신선도(mtime > .run_start).
2. **RESOLVED CONFIG 실측 diff**(바이너리 자신의 environ 보고 기준, 런처 아님): baseline 대비 **전체 게이트 diff를 출력**하고, `--intended-diff`와 정확히 일치하는지 판정. 초과·누락 1건이라도 있으면 FAIL. 단일변수 주장은 이 관문 통과 없이는 금지.
3. 바이너리(LUMINA_BIN)·모델 디렉토리(argv) 동일성 — 다르면 intended-diff에 명시된 경우만 통과.
4. 결과를 `logs/coevolve_consume_<run>/VERDICT_PREFLIGHT.md`로 저장(판정문에 첨부).

## V2. 운전석 판정 초안
- 수치는 **보존된 산출물에서만** 계산(repo-root 파일 금지 — 다음 런이 덮어씀).
- 모든 인과 귀속 문장은 V1의 diff 블록을 인용해야 한다.
- 이전 결론을 뒤집는 경우 `[정정]`을 명시하고 무엇이 왜 틀렸는지 1-2문장.
- 초안을 `logs/coevolve_consume_<run>/VERDICT_DRAFT.md`로 저장.

## V3. 적대검증 — 수행자 개정(2026-07-31 user 지시): **Codex CLI**(read-only exec; 호출법 /home/kjhan/how_to_call_codex_from_claude-code.md), Opus 에이전트는 폴백. 운전석은 V3 결과를 검수해 반영 (원 제정 07-29: "운전석의 판정을 opus에게 오프라인으로 추가 검증")
운전석이 Opus 에이전트를 발주한다. 에이전트 프롬프트 필수 요소:
- VERDICT_DRAFT.md 경로 + 원시 산출물 경로 + **기지 항목 목록**(재발견 방지).
- 지시: "각 주장을 **반증하라**. 주장별로 CONFIRMED / REFUTED / UNVERIFIABLE + 재현 명령. 잠정 판단을 확정처럼 쓰지 말 것. 초안에 없는 신규 발견은 별도 절로."
- 에이전트는 초안 작성에 관여하지 않은 컨텍스트에서 출발(독립성).
판정 처리 규칙: **CONFIRMED만 확정으로 원장 기재.** REFUTED는 초안 수정 후 재검증 또는 폐기. UNVERIFIABLE은 `[잠정]` 표시로만 기재 가능.

## V4. 원장 기재
판정문 + V1 preflight 요약 + **V3 스탬프**(에이전트 판정 요약: n건 CONFIRMED / n건 REFUTED / n건 잠정)를 함께 기재. 불일치는 그대로 기록(합의 조작 금지).

## V5. 대장 갱신
판정에서 나온 **모든 신규 결정**을 RESUME.md의 "결정했으나 미실행" 절에 즉시 추가. 실행되면 삭제.

## 근거 사례 (왜 각 단계가 있는가)
- V1.2: parity31→32를 "DB_FB 단독"으로 오귀속 — 실제 diff는 5게이트(사례 29). parity34는 런처 unset으로 게이트 미도달(83 GPU-분 손실).
- V1.1: 화석 CSV 판정 사고(사례 17).
- V2: "입력 출처→코드 분기→값" 사슬 미확인으로 4회 번복(07-27).
- V3: g-가중 b_k 잣대를 Fable 적대검증이 뒤집음(66×→1.6×) — 독립 반증이 실제로 작동한 전례. 운전석 자기판정 오류(오늘 귀속 2건)는 user 개입으로만 잡혔음 — 그 관문의 상설화가 V3.
