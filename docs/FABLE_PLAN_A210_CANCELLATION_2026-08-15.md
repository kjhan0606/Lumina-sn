# Fable 세부계획 — A2-10 cancellation 폐합 (2026-08-15)

- requested model: `fable`
- canonical CLI alias: `claude-fable-5`
- mode: read/write tools disabled, compact supplied context
- result: `VERDICT: REVISE`
- note: 두 번의 선행 호출은 각각 `$1` 비용 상한에서 응답 없이 종료했다. 아래 판정은
  세 번째 압축 호출(`effort=low`, `$2` 상한)에서 수신한 유일한 판정이다.

## Fable 응답 원문

VERDICT: REVISE

Codex 후보 목록(census/refinement/tolerance/1-iter/4-iter)은 순서가 없고, **lower witness 가 대칭 envelope 로는 원리적으로 닫히지 않는다**는 점을 놓쳤다. 아래 검산부터.

### 검산

**Upper witness (line1279130 s18)**: `chi*Jbar = 1.91950e-37`이고
`eta-chi*Jbar = 8.7062e-42`; `4pi*net = 1.09406e-40`이다.
`uncertainty = 4pi*chi*Jbar_bound = 1.6427e-40`이다. 부호 확정 조건은
`chi*Jbar_bound < |net|`이므로 필요한 `Jbar_bound = 8.94e-10`,
현재/필요 비는 `1.50`이다.

**Lower witness (line15 s10)**: `chi*Jbar = 7.7866e-59`이고 eta는 무시할 수
있을 만큼 작지만 값 자체를 절단해서는 안 된다. 대칭 bound로는
`Jbar_bound < Jbar-eta/chi ~= 2.961e-51`가 필요하고 현재/필요 비는 `7.07`이다.
그러나 필요한 것은 `Jbar_lower > eta/chi = 1.295e-154`라는 one-sided 증명이라는
것이 Fable의 핵심 제안이다. 이것이 유도 가능하면 physical value repair가 아니라
증명 정밀도 개선이다.

## Fable 단계 계획

1. **검산 고정물**: rate/uncertainty의 `4pi*deck_scale` 적용과 두 witness 산술을
   기계 재현한다. 상대 `1e-12` 불일치면 상위 단계를 동결하고 단위·scale 경로부터 감사한다.
2. **opt-in 전 셀 census**: 기본 fail-closed 판정은 유지하되 진단 모드에서 모든
   unresolved의 `(line,shell,r,eta,chi*Jbar,bound)`와 NaN/Inf를 기록한다.
   `r<=2` 다수면 refinement 후보, `r>>10` 꼬리가 있으면 one-sided 증명 후보로 분류한다.
3. **refinement-only**: tolerance를 `1e-8`로 고정하고 envelope refinement 궤적과
   unresolved 잔여 수를 기록한다. 증가하면 envelope 유효성 문제로 즉시 중단하고,
   residual 바닥에서 plateau면 tolerance 실험으로 넘긴다.
4. **one-sided envelope 투영**: lower형에 대해 Gaussian profile의 검증된 `Jbar_lower`
   를 유도한다. 전수 부호 확정이면 통과. 현재 cellwise 증명이 one-sided 유도를 지원하지
   않으면 억지로 만들지 않고 tolerance-only로 넘기며, 끝내 미달이면 unresolved를 유지한다.
5. **tolerance-only**: refinement 설정을 고정한 채 `1e-8 -> 1e-10`으로 residual 바닥을
   낮춘다. 선행 envelope 궤적이 residual 지배임을 보여야 발주한다.
6. **A100x2 1-iteration gate**: unresolved 0 또는 one-sided 증명 첨부, repair/negative 0,
   witness 결정론 재현, R6 전수 valid를 모두 요구한다.
7. **CMFGEN finite 비교**: 1-iteration gate 직후, 4-iteration flight 전에 동일한
   `4pi*(eta-chi*Jbar)`를 같은 단위·셀/선 매핑으로 대조하기 시작한다.
8. **4-iteration flight**: 1-iteration PASS와 iteration-0 CMFGEN 대조의 비파국적 결과가
   전제다. envelope는 매 iteration의 잔차로 다시 유도하고 이월하지 않는다.

## DO NOT DO

- `Jbar`, eta, net에 floor/cap/clamp/jitter 또는 작은 값 threshold 절단을 하지 않는다.
- envelope 없이 raw double 부호를 채택하지 않는다.
- refinement와 tolerance를 동시에 바꾸지 않는다.
- flight를 진단 계산으로 사용하지 않는다.
- multi-GPU 자기일치를 독립 물리 검증으로 계상하지 않는다.

## 값비싼 계산 전 Fable 필수 증거

1. rate/uncertainty 단위·scale 검산기 PASS.
2. 전 셀 unresolved census와 non-finite 0 증거.
3. refinement별 envelope 단조 궤적과 residual 지배/K 지배 판별.

## Codex 채택 보류점

Fable의 one-sided 제안은 중요한 방향이지만 현재 구현이 제공하는 것은
`|J_exact-J_approx| <= error_upper` 형태의 대칭 절대오차 envelope다. 이 정보만으로는
`Jbar-error_upper < 0`인 lower witness에서 양의 `Jbar_lower`를 얻을 수 없다.
따라서 exact transport의 positivity, source의 엄밀한 양의 하한, 또는 방향성 있는 residual
bound를 코드에서 실제로 증명할 수 있는지를 별도 확인하기 전에는 4단을 채택하지 않는다.
