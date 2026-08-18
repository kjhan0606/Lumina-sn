# Fable 정정 질의 — FIX_T와 CMFGEN adiabatic RE 항

날짜: 2026-08-08

응답 수신: `docs/FABLE_VERDICT_SH_RADEQ_2026-08-08.md`의 정정 판정 절 —
`FIX_T_RE_ADIABATIC=INCLUDED`, `PRIOR_VERDICT=REVISE 유지`.

선행 판정 `docs/FABLE_VERDICT_SH_RADEQ_2026-08-08.md`의 Q3에는 다음 문장이 있습니다.

> 기준 덱이 FIX_T=T이므로 Q1의 RE_INTEGRAL 잔차 대조에는 단열항이 진입하지 않아
> 현 단계 gate 대상은 free-T 경로에 한정된다.

그런데 CMFGEN 원문을 다시 확인한 결과 다음과 충돌합니다.

## 확인된 원문

1. 기준 덱은 동시에 다음을 설정합니다.
   - `T [INC_AD]` — `VADAT:119`
   - `T [FIX_T]` — `VADAT:622`
2. `new_main/cmfgen_sub.f:2939-2962`는 `RD_FIX_T`를 검사하지 않고
   `EVAL_ADIABATIC_V3(..., INCL_ADIABATIC, ...)`를 호출합니다.
3. `new_main/subs/eval_adiabatic_v3.f:237-250`은 `INCL_ADIABATIC`이면 모든 depth에서
   `NEW_STEQ_T(I)=NEW_STEQ_T(I)-WORK(I)`를 수행합니다.
4. `new_main/mod_subs/solve_for_pops.f:77-90`의 `FIX_T` 처리는 계산된 RE 항을 없애는 것이
   아니라, 온도를 고정하기 위해 행렬의 온도 행을 교체하는 후단 solve 제약입니다.

따라서 `FIX_T=T`여도 diagnostic `STEQ_T`에는 adiabatic cooling이 포함되는 것으로
읽힙니다. 이는 선행 판정의 Q3 근거 한 문장을 반박합니다.

## 요청

다음 형식으로 정정 판정을 내려 주십시오.

```text
FIX_T_RE_ADIABATIC = INCLUDED | EXCLUDED | 판단불가 — 근거
Q3_CORRECTION = A | B | C — fixed-T RE 대조와 free-T solve 각각의 gate
PRIOR_VERDICT = REVISE 유지 | 다른 판정 — 근거
```

- `INCLUDED`라면 fixed-T CMFGEN `STEQ_T` 동종 대조에도 CMFGEN과 같은 adiabatic 항이
  필요하다고 명시하십시오.
- 현재 Lumina `3*n_e*k*T/t`만으로 fixed-T RE 대조를 PASS시킬 수 있는지도 판정하십시오.
- 모르는 내용은 추측하지 말고 `판단불가`로 두십시오.
