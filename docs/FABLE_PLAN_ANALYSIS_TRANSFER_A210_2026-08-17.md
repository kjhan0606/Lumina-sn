# Fable 총괄·계획 및 분석·평가 이관 — 1차 판정

- 요청 모델: `fable`
- CLI: Claude Code `2.1.233`
- 질의: `docs/QUERY_FABLE_A210_PLAN_ANALYSIS_TRANSFER_2026-08-17.md`
- 권한: Read/Grep/Glob only; 코드·파일·process·작업 변경 없음

아래는 Fable 응답 원문이다.

# VERDICT

- **K18 cancellation census와 K12→K18 검증은 폐합 가능하다.** unresolved 19건 전원 refinement-only로 해소, 물리값 변형 0, bound non-increasing — census closure의 정본 요건을 충족한다.
- **현재 K36 A100×2 run은 변경 없이 완주시킨다.** R1 exact/R6/R2 seed barrier/material census가 owner 정본과 bit-exact이므로 R2 exact solve 완료가 유일한 미완 요소다.
- **물리 코드 변경은 현재 어떤 증거로도 허가되지 않는다.** IV-stage absorption 330–400× 결핍은 우선 가설(Jbar)의 근거일 뿐 확정이 아니며, aggregate deficit 단독 확정 금지 제약이 그대로 적용된다.
- **K24/K30은 final non-census completion artifact가 없으므로 이번 gate 주장 범위에서 제외한다.** gate PASS 주장은 K36 단독으로 한다.

# CURRENT ASSESSMENT

- **Census 축**: K18 JSON이 fail-closed 요건(물리 mutation 0, cap/clamp/floor/jitter 0)을 기계적으로 증명한다. 잔여 리스크 없음.
- **K36 run 축**: R1 45 iterations, residual 9.666×10⁻⁹ < 10⁻⁸, refinements 36, valid lines 2,180,286, partial/unsampled 0 — 전부 owner 정본과 bit-exact. generation barrier(te 1→1, pop 1→2) 보존, raw-negative census 4,246,581건은 raw 보존 상태로 기록됨(repair 0). 남은 리스크는 R2 exact solve의 자연 종료와 tripwire 무발동뿐이다.
- **물리 가설 축**: IV-stage emission 차이 1.38–1.49× 대비 absorption 결핍 330–400×는 순수 population normalization 설명을 배제하지만, Sobolev opacity/lower-population 결함도 배제하지 못한다. III-stage(0.897–0.984)가 내장 null control이다.
- **함정 하나를 명시한다**: combined 90% prefix는 Co IV 82.87% 지배 때문에 Fe IV(10.47%)/Ni IV(6.65%)의 개별 90%를 보장하지 않는다. V4 undercoverage는 순수 표본추출 문제이며 물리 신호가 아니다 — 이미 사전등록된 해석이고 이를 유지한다.

# NEXT FIVE STAGES

**Stage 1 — K36 R2 exact solve 완주와 자연 종료 확인**
- 입력: 진행 중 diag dir(`…_f9c2d1b826d5`), owner 정본 R2 기록.
- PASS: R2 exact가 owner 정본과 bit-exact, model이 자연 rc=1 + `REQUESTED_TE RADEQ_NO_BRACKET`로 종료, tripwire 무발동, pre-core refresh 0 유지.
- fail-closed: bit-exact 불일치·비유한값·비자연 종료 → 후처리 전면 중단, diag 봉인, 원인 규명 전 재실행 금지.
- Codex: tripwire/monitoring 유지, 종료 즉시 전 아티팩트 SHA-256 봉인.

**Stage 2 — V2/V3 후처리 순차 실행**
- 입력: Stage 1 sealed diag.
- PASS: V2 combined 90% summary/match 산출 + V3 roundoff-aware 분류 완결. V3 산출물에는 각 항목의 **산술 증거상한**(roundoff로 설명 가능한 최대 차이)을 물리 판단 없이 기재.
- fail-closed: match 실패나 분류 불능 → 해당 항목 UNCLASSIFIED 기재, V4 진행 여부를 Fable 판정으로 회부.
- Codex: 후처리 실행, 산출물 SHA 봉인.

**Stage 3 — V4 ion별 exact 90% coverage 검사와 분기 실행**
- 입력: V2/V3 산출물, ion별 emission 비중.
- PASS: Fe/Co/Ni IV 각각 exact 90% 커버 확정 → Stage 4 직행.
- fail-closed(=UNDERCOVERED): 사전등록대로 **같은 sealed 진단의 ion별 minimal 90% prefix union만** 재실행. 동일 binary SHA·동일 state 필수. 물리 원인 기재 금지.
- Codex: 재실행 발주(운전석 승인 경로), 재실행분도 SHA 봉인.

**Stage 4 — 승인된 read-only finite-component 분기 판정**
- 입력: shell-0 90% carrier lines의 Lumina tau / Jbar-S_line / beta(tau), CMFGEN depths 67/68 `1-ZNET`.
- PASS: BRANCH RULES의 규칙 J/O가 per-line 대조에서 단일 경로를 지목하고 III-stage null control이 무결. 증거 불충분이면 **미결로 명시 기재**하는 것도 PASS다(강제 결론 금지).
- fail-closed: J/O 신호 혼재 또는 III-stage가 결함으로 오분류(잣대 오염) → 판정 보류, 잣대 감사 선행, 코드 변경 불허 유지.
- Codex: 대조표 추출(read-only), 판정 감리(read-only) — 판정 자체는 Fable.

**Stage 5 — A100×2 non-census gate 폐합**
- 입력: Stage 1–4 전 아티팩트.
- PASS: FINAL GATE EVIDENCE 목록 전 항목 충족 시 gate PASS를 V0–V5 규약으로 원장 기재(단 폐합 전 Codex read-only 판정 감리 스탬프 포함, 커밋 1건 = 계약 1건).
- fail-closed: 목록 중 1건이라도 결손 → PASS 주장 금지, 결손 항목 명시.
- Codex: 문서화·커밋. 물리 코드 변경은 이 gate와 **별도 트랙**이며 아래 허가 요건을 통과해야만 개시된다.

# BRANCH RULES

**V4 분기(각각 정확히 하나의 행동)**
- **PASS** → 추가 실행 없이 Stage 4 판정으로 진행한다.
- **UNDERCOVERED** → 같은 진단의 ion별 minimal 90% prefix union만 재실행한다. 이 사실을 물리 원인으로 기재하는 것을 금지한다.

**경로 판정 규칙(per-line, shell 0 — aggregate ratio만으로는 어떤 규칙도 발동 불가)**
- **규칙 J(radiation/Jbar 경로)**: Lumina tau가 optically thick이고 CMFGEN `1-ZNET`이 saturation을 지시하는데, Lumina `Jbar/S`가 그 선 자신의 트래핑 기대 `1-beta(tau)`보다 **V3 산술 증거상한을 초과하여** 낮으면 → FUV Jbar self/local coupling 결함 우선.
- **규칙 O(opacity/lower-population/line-universe 경로)**: Lumina tau가 optically thin인데 CMFGEN `1-ZNET`이 saturation → opacity/lower-population/line-universe 우선. tau는 Jbar에 독립이므로 이 판정은 Jbar 가설과 논리적으로 분리된다.
- **혼동 금지 조항**: V3 산술 증거상한은 "차이가 수치적으로 실재하는가"만 답한다. "얼마나 어긋나야 물리 결함인가"의 물리 tolerance는 별도 사전등록 없이는 적용 금지 — 산술상 실재하나 tolerance 미등록인 차이는 미결로 기재한다.
- **null control 조항**: Fe/Co/Ni III이 같은 규칙에서 결함으로 분류되면 규칙 자체가 오염된 것이다. 판정을 철회하고 잣대 감사를 선행한다.

**물리 코드 변경 허가 요건(전부 충족 전 개시 금지)**
1. Jbar 또는 opacity 경로의 **특정 잘못된 물리 표현식**을 파일:라인 수준으로 국소화.
2. sealed K36 state 위 **offline recomputation**으로 수정식이 IV-stage tau/Jbar 정합을 복원하면서 **III-stage null control이 안정**함을 시연.
3. 기대치 사전등록 + 판정런 1회에서 **preregistered negative control**(주입 결함으로 FAIL 시연 — 게이트 PASS 자격 요건).
4. floor/cap/scaling/삭제/수치 repair 전무, pre-core tau refresh 부활 금지, coevolution generation barrier 보존.

# FINAL GATE EVIDENCE

gate PASS 주장에 필요한 최소 정본 artifact와 불변조건:

1. **K18 comparison JSON**(SHA `b6a0be0b…`) — census closure 증명.
2. **K36 diag dir + binary SHA `f9c2d1b8…`** — R1 exact/R6/R2 seed/material census bit-exact 기록과 R2 exact solve 완료 기록.
3. **자연 종료 증거** — model rc=1 + `REQUESTED_TE RADEQ_NO_BRACKET` 이벤트 로그(EVENT_LOG=1).
4. **V2/V3/V4 산출물 각각의 SHA** — V4는 PASS 또는 UNDERCOVERED→union 재실행 후 PASS.
5. **Stage 4 branch 대조표와 귀속 문서**(귀속 또는 명시적 미결 — 미결도 gate를 막지 않는다. gate는 non-census 실행 완결성을 폐합하는 것이지 물리 결함 해소를 주장하는 것이 아니다).
6. **불변조건 스탬프**: floor/cap/clamp/jitter/repair = 0, physical_values_modified = false, pre-core refresh 0, stage4 repair 0, generation barrier(te 1→1, pop 1→2), tripwire 무발동, 외부 PID/Slurm 충돌 0.
7. **Codex read-only 판정 감리 스탬프**와 V0–V5 원장 기재, K24/K30 제외 범위 명시.

## Codex 보류점

위 1차 응답의 Stage 5와 FINAL GATE EVIDENCE 3/5는 현재
`scripts/finalize_a210_nonoverlap_gate.py`가 요구하는 성공 모델 rc와 R7/publication
commit 계약에 정면으로 부합하는지 재검토가 필요하다. 이 원문은 수정하지 않고, 실제
auditor 계약을 제시한 Fable 정정 판정을 별도로 봉인한다.
