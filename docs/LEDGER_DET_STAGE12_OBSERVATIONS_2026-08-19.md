# 조용한 대장 기재 — DET-STAGE12 부수 관찰 (2026-08-19)

독립 감사 2회전이 **차단은 아니나 기록해야 한다**고 남긴 항목들.
프로젝트 규약 「틀린 값은 조용히 대장 기재 — 튜닝 금지」에 따라 **수리하지 않고 적는다.**
출처: 감사 의견서 §F "기재 권고", 그리고 §D 의 경미 주의.

## 계약 필드가 레인에 따라 의미가 갈린다 (2건) — 가장 무거운 축

| # | 필드 | 문제 |
|---|---|---|
| **S1** | `residual_status[s]` | 고정레인이 무조건 `RADEQ_OK` 를 세운다(`lumina_plasma.c` 고정 경로). 이것 없이는 `candidate_bundle_commit_preflight` 와 `physics_comparison.c:185` 가 발행을 막으므로 **레인 정의상 불가피**하다. 그러나 `RADEQ_OK` 의 의미가 "근 품질 통과"(자유) 대 "근을 안 푼다"(고정)로 갈린다 |
| **S2** | `producer_equation = A210_RE_INTEGRAL` | 고정레인도 이 값을 세운다. T 의 출처는 **파일**인데 이 필드는 "RE 적분이 이 T 를 생산했다" 는 주장이다. `tests/a2_10_radeq_selftest.c:87` 이 이 필드를 문자 그대로 "root value/provenance" 라 부른다. 커밋 preflight 가 RE_INTEGRAL 을 요구해 불가피 |

⟹ **`te_lane` 을 보지 않는 소비자는 두 레인을 구별하지 못한다.**
이 둘이 이 단이 남기는 가장 큰 부채다. 향후 소비자를 추가할 때 반드시 `te_lane` 을 함께 읽어야 한다.

## 표기·명명 불일치 (3건)

- **S5** 사전등록은 `RE_RESIDUAL_AT_PINNED_T`(밑줄), 구현은 `RE-RESIDUAL-AT-PINNED-T`(하이픈).
  **밑줄 형은 저장소에 0건** — L5 를 밑줄로 grep 하면 히트 0이다. 판정 시 하이픈으로 읽을 것.
- **S6** 같은 필드가 세 표기: 공시 `re_root_required=0` · TE_PUBLICATION `false` · manifest JSON `false`.
  파서 하나로 못 잡는다.
- **S3** 공시 위반의 반환코드가 `PHYSICS_COMPARISON_IO_ERROR` 다 — **I/O 오류가 아니다.**
  형제 음성대조는 `STALE_GENERATION`/`INVALID_VALUE` 전용 코드를 쓴다.
  FATAL 줄에 `status=IO_ERROR` 로만 남으면 사후 오귀속 위험(진짜 사유는 바로 위
  `[PHYSICS_COMPARISON][BLOCKED] reason=` 줄에 있다).

## 카운터·진단 (3건)

- **S4** 자유레인 신설 검증 실패 시 `ct->te_context_mismatch++` — **컨텍스트 해시 불일치가 아니라
  레인 스키마 위반**이다. `blocked_schema` 가 맞다. 이 오용 때문에 `r7_a210_block_reason` 이
  `RADEQ_TE_CONTEXT_MISMATCH` 를 보고한다. (해당 분기는 도달 불가로 확인됐으므로 현재 무해)
- **D-경미** 셸별 `[RE-RESIDUAL-AT-PINNED-T]` 줄은 **절대 잔차**, 카운터 `max_heat_residual` 은
  **상대 잔차**(`|res|/(ΣH+ΣC)`)다. 같은 보고서 안의 두 다른 양 — 판독 시 혼동 주의.
  `den==0 && residual!=0` 이면 `e_balance=INFINITY` → `inf` 인쇄(판정 무관).
- **S8** `LUMINA_RADEQ_DIAG=1` 의 `[A2-10][ENDPOINT-FINITE]` 는 셸별 `T_e_K`·전 장부를
  **레인 표지 없이** 찍는다. 핀이 LOWER/UPPER/GEOMETRIC_MID/REQUESTED_TE 중 하나로 **균일**할 때만
  발화한다(`a210_uniform_endpoint_phase` 가 비균일에 NULL) ⟹ 실무 발생 가능성 낮음 [추정]

## 그 밖

- **S7** `te_source`(env 값)가 **무살균**으로 stderr 공시에 들어간다. 개행을 넣으면 A2-10 공시
  스트림에 위조 레코드를 주입할 수 있다. 공격 시나리오가 아니라 **"공시가 변조증거성을 갖는가"**
  의 문제. 프로파일 내용 자체는 sha256 으로 봉인되므로 실질 위험은 낮다.
- **S9** `nlte_reject_numeric_repairs("A2-10-PRODUCTION")` 가드가 기준선에서는 함수 **첫 줄**이었는데
  지금은 고정-T 로드·공시 **뒤**로 밀렸다. 물질 갱신 전에 여전히 막으므로 안전하나,
  수치수리가 켜진 런도 FIXED_T 공시를 **먼저 뱉는다**.
- **S10** `selftest_det_stage12` 가 `.PHONY` 에만 있고 `clean` 목록·`run_gate_battery.py` 에는 없다.
  형제 selftest 들도 배터리에 없으므로 **관행 위반은 아니다**(회귀 아님).
- **M1 부수** 수락 경로 공시의 `publication_authority=NONE` — 저장소의 **유일한 어휘**라
  관례를 따른 것이나, 고정-T 레인은 실제로 **발행한다**. "이 진단 줄에는 발행 권한이 없다" 는
  뜻으로 읽어야 하며, 레인 자체의 발행 권한과 혼동하지 말 것.

## 처분

전부 **기재만** 한다. 수리는 국면 규칙별 별도 결정이며, 특히 S1·S2 는
계약 구조 변경이라 단을 따로 세워야 한다.
