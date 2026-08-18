# Fable 감리 판정 — SH-RADEQ 구현 반영 검토 (2026-08-08)

- model: `claude-fable-5`
- Claude Code CLI session: `c208bf1a-7779-4bb1-ba0a-ff86197f9849`
- 질의: `docs/QUERY_FABLE_SH_RADEQ_IMPLEMENTATION_REVIEW_2026-08-08.md`
- 판정: `REVISE`

아래는 Fable 응답 원문이다.

---

제출 증거만으로 판정한다. 코드는 열지 않았고 작성하지 않는다.

## Q1. 직접 방출 A/B — 선행 판정 2·3 충족 여부

**판정 2(방출식)는 충족한다.** `-expm1(-tau)/tau`는 `(1-e^{-τ})/τ`의 수치적으로 올바른 형태이고(작은 τ에서 소거 오차 없음), `tau==0` 분기는 극한값 1과 매끄럽게 접속한다. signed τ 허용, `n_upper==0 || A_ul==0`만 exact zero, `chi*S` 상대 closure ≤1e-12, τ=0/음수 τ/영·음수 population 자가검사, legacy `line_source_S` read 0개 — 모두 판정 2의 요구와 정합한다. SKIP_Z가 흡수·방출을 짝으로 끄는 것도 옳다(한쪽만 끄면 에너지 장부가 깨진다).

**판정 3(immutable generation-bound view)은 절반 충족이다.** raw slab + 세대 토큰이 이번 단(rung)에서 허용 가능한가에 대한 답: **조건부로 허용한다. compact 사본/해시를 이번 단에 강제하지는 않는다.** 1.25억 cell 복제 회피는 정당한 비용 판단이고, 사다리 규율상 "한 단 = 계약 1개"에 view 자료구조 개편을 얹는 것은 오히려 위반이다. 단, 현 상태의 토큰 결박에는 검출 불가능한 구멍이 있다:

- **토큰은 규율을 지키는 writer만 감시한다.** 세대를 올리지 않고 slab을 건드리는 writer(버그)는 원리적으로 검출되지 않는다. 현재의 5개 음성대조 어느 것도 이 모드를 덮지 않는다.
- **검사 시점이 publication 직후 1회다.** 방출 평가는 그 뒤 flight 중에 일어난다. publication과 소비 사이의 stale-write는 검사 창 밖이다. 이 프로젝트에서 침묵 경로가 여섯 번을 헤매게 한 전력을 감안하면 이 창을 열어둔 채 ACCEPT할 수 없다.

허용 조건(아래 Q4 항목 1)은: 소비 구간 양끝 bracket 재검사 + line slab writer 전수 census 등재. 이 둘이 있으면 토큰 방식은 이번 단의 generation-bound view로 인정한다.

추가 결함: reader가 NLTE writer의 덮어쓰기 조건(두 준위 NLTE mapped, pair-owned/element-wide, not NLTE_SKIP_Z)을 **재유도**하고, LTE `n_u`를 **재계산**한다. 술어와 재계산 루틴이 writer와 별도 사본이면, writer 쪽이 바뀌는 순간 reader가 조용히 갈라진다 — 그때 τ와 `n_u`는 서로 다른 population에서 온 것이 되어 판정 3을 정확히 그 지점에서 위반한다. 1e-12 closure가 이를 부분적으로 묶지만 자가검사 cell 표본에서만이다.

## Q2. RE_INTEGRAL 유일 producer / EHB diagnostic 분리

**충분하다.** kind가 ledger와 publication 양쪽에 각인되고, transaction 층에서 EHB producer 주입이 rc=5로 거부되며(finalize 자체는 허용 — diagnostic으로서 옳다), 음성대조가 public T_e/세대 불변을 시연한다. 선행 판정 1의 요구를 구조(절차 강제)로 구현했다. publication에 kind가 실려 하류 소비자가 provenance를 검증할 수 있는 점도 옳다.

## Q3. 불완전 단열항 fail-closed

**선행 판정 4를 정확히 구현했다.** 값 보존(diagnostic) + `A210_INCOMPLETE` status + `RADEQ_INCOMPLETE_ADIABATIC` finalize + rc=3 + publication 0, 그리고 `LUMINA_FIXED_TE_PROFILE` 경로까지 같은 **이름 있는** 사유로 차단 — fixed/free-T 양쪽 금지라는 판정문 그대로다. 사유가 이름으로 출력되는 것이 특히 중요하다(K-FRESH 침묵 차단의 재발 방지). 유일한 요구: 이 차단은 env opt-in이 아니라 기본값으로 fail-closed여야 하고, 사유 문자열이 이벤트 로그에 남아야 한다. 제출문 상 충족으로 읽히나, 폐합 전 게이트에서 기본값 경로로 1회 시연하라.

## Q4. flight 전 필수 수리 (중요도순)

1. **[구조] raw tau slab 소비 계약의 폐쇄.** (a) `OpacityState` line slab을 쓰는 writer 전수 census를 검증 대장에 등재하고 각각이 세대를 올림을 명시, (b) 방출 소비 구간의 **양끝**(진입 직전·종료 직후)에서 B의 세대 등식 전체를 재검사하고 불일치 시 abort, (c) 이 bracket이 실제로 무는지 음성대조 1건(소비 구간 중 세대를 올리며 slab 변조 → abort 시연). 이것 없이는 "immutable view"가 불변식이 아니라 관례다.
2. **[구조] NLTE 권한 술어의 단일화.** writer가 per-line(또는 per Z,ion) authority/provenance 비트를 publication에 실어 reader는 소비만 하게 하거나, 최소한 술어를 단일 공유 함수로 묶어라. 음성대조: 한쪽 사본만 술어를 뒤집었을 때 검출됨을 시연.
3. **[물리] LTE `n_u` 재계산 = bulk tau writer와 동일 루틴.** 별도 구현이면 Boltzmann/partition 관례 미세 차이가 τ↔`n_u` 비정합으로 조용히 들어온다. 공유 함수로 강제하고, chi*S closure를 NLTE-committed 가지와 LTE-재계산 가지 **양쪽을 덮는** cell 집합에서 통과시켜라.
4. **[등재] 음수 τ 영역 거동.** clamp 없음은 옳으나 β가 |τ| 증가에 지수 성장함을 대장에 기재하고, CMFGEN의 동일 상황 처리와 대조를 pre-register하라(발자국 계승 원칙). 유한성 abort는 침묵이 아니라 이름 있는 오류면이어야 한다.
5. **[등재] D의 소비 계약.** 707개 sub-ν_min BF edge를 재개방 SH-GRID가 어떻게 적분하는지(CMFGEN 동종 처리 여부)를 명시하고 기대 효과를 사전등록하라. 현재 D는 사실 진술이지 소비 계약이 아니다.

항목 1–3이 flight 전 필수이고, 4–5는 대장·사전등록 의무다. A/C/D의 방향 자체는 선행 판정과 정합하므로 BLOCKED는 아니다.

```text
IMPLEMENTATION_VERDICT = REVISE
```

## Codex의 판정 및 반영

Codex도 `REVISE`에 동의한다. 항목 1–3은 flight 전 코드 불변식으로 구현했고,
항목 4–5는 각각 signed-tau 및 SH-GRID 사전등록 문서에 고정했다. 구현 후 최종
재심을 별도로 요청한다.
