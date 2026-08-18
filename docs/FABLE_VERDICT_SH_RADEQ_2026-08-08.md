# Fable 판정 — SH-RADEQ 기준 방정식과 선 방출

날짜: 2026-08-08

## 호출 provenance

- CLI: Claude Code `2.1.226`
- 요청 모델: `--model fable`
- 응답 canonical model: `claude-fable-5`
- session id: `b6d8a0a3-d00a-407e-a4dc-16b9b2fafd76`
- 입력: `docs/QUERY_FABLE_SH_RADEQ_2026-08-08.md` 원문을 stdin으로 전달
- tools: disabled
- CLI status: success, `stop_reason=end_turn`, permission denial 0

## Fable 원문 응답

```text
Q1 = C — 근거: producer = RE_INTEGRAL. 측정된 사실은 기준 덱이 FIX_T=T이며 EHB 경로(USE_EHB/COMP_EHB)를 전혀 실행하지 않고, 실제로 계산·기록되는 유일한 에너지 잔차가 STEQ_T = ∫(χ_noscat·J − η_noscat)dν라는 점이다. 따라서 기준 산출물과 직접 대조 가능한 canonical solve(온도 producer)는 RE_INTEGRAL이어야 한다. A2-10 사전등록의 EHB형 초과에너지 장부는 기준 덱에서 검증 앵커가 없으므로 producer가 될 수 없고, independent closure diagnostic으로 유지한다(두 식은 수렴 시 물리적으로 일치해야 하므로 유효한 교차검증이 됨). CMFGEN 측 COMP_EHB=T, USE_EHB=F 진단 실행으로 EHB 항별 대조를 수행하는 것은 허용되나, 이는 별도 진단 산출물이며 기준 truth를 대체하지 못한다. USE_EHB=T free-T truth 생산은 기준 덱과 다른 설정이므로 별도 사전등록 없이는 truth로 승격 금지.

Q2 = A — canonical bin 적분은 직접식 η_bin = n_upper·A_ul·h·ν·β_esc(τ)/(4π·Δν), β_esc(τ)=(1−e^{−τ})/τ. A2-08 흡수가 이미 (1−e^{−τ}) effective operator를 쓰므로, 방출에서 β_esc를 빼면(선지 B) 공명 내 자기흡수가 이중 계상된다. 정상 cell에서 χ_eff·S_line과 대수적으로 동등함은 사실이나, canonical 구현은 division 표현(χ_eff·S)이 아니라 위 직접 곱 형태로 고정해야 한다. τ=0 cancellation 처리: τ→0에서 β_esc→1이 해석적 극한이므로, population cancellation(유도방출 상쇄로 net τ=0)이면서 n_upper>0인 cell은 exact-zero emission으로 둘 수 없다 — 자발방출 η = n_upper·A_ul·h·ν/(4π·Δν)로 유한 방출을 보존해야 한다(흡수 측이 0이 되는 것과 물리적으로 모순 없음). τ ownership: β_esc의 τ는 A2-08 흡수가 사용하는 것과 동일한 per-line Sobolev τ(유도방출 보정 포함)여야 하며, n_upper·A_ul·τ 세 입력 모두 동일 generation의 immutable generation-bound view에서 읽어야 한다. 가변 OpacityState 재독(현 A2-09 방식)은 흡수/방출 간 generation 불일치로 에너지수지 오염 가능성이 있으므로 금지. n_lines=0 slab은 흡수-방출 τ 공유 계약이 view로 성립한 뒤에만 유지 가능.

Q3 = C — 현재 전자 병진항 3·n_e·k·T/t만으로는 CMFGEN(원자+전자 병진, 전자분율 구배, 여기·전리 내부에너지 구배)을 재현할 수 없다는 것이 확정 사실이다. 그러나 기준 덱이 FIX_T=T이므로 Q1의 RE_INTEGRAL 잔차 대조에는 단열항이 진입하지 않아 현 단계 gate 대상은 free-T 경로에 한정된다. 허용되는 임시 상태: 전자 병진항을 유지하되 (i) 최소한 병진항을 3·(n_e+Σn_ion/atom)·k·T/t로 즉시 확장하는 것은 허용(엄밀·자명한 homologous 축약), (ii) 여기·전리 내부에너지 및 전자분율 항이 없는 동안 모든 free-T 산출물의 gate를 BLOCKED_INCOMPLETE_ADIABATIC으로 고정, (iii) fixed-T RE 잔차 대조는 이 gate와 무관하게 진행 가능. 선지 B의 완전 축약식(내부에너지/구배 항 포함)은 준위·전리 시간미분 상태를 요구하므로 본 질의의 증거만으로는 즉시 확정 불가 — 별도 도출·검증 후 gate 해제.

VERDICT = REVISE
```

Fable 부가 요약:

> 사실 — 기준 덱은 FIX_T=T·EHB 미사용이며 STEQ_T만 계산; Lumina 단열항은 전자
> 병진항뿐; 현 CPU opacity는 n_lines=0 slab에 가변 state 재독. 추론 — RE_INTEGRAL
> producer 지정, β_esc 직접식의 τ→0 극한 처리, generation-bound τ 소유권 계약.
> 사전등록(EHB producer)에서 producer 변경, 방출식 직접형 고정, 단열 gate 신설이
> 필요하므로 PROCEED가 아닌 REVISE.

## 즉시 효력

- A2-10 온도 producer: `RE_INTEGRAL`.
- `EHB_THERMAL`: independent closure diagnostic.
- line eta: direct `n_upper A_ul h nu beta_esc/(4 pi dnu)`.
- cancellation `tau=0`, `n_upper>0`: finite spontaneous emission, exact zero 금지.
- `n_upper`, `A_ul`, `tau`: 동일 immutable generation-bound view.
- fixed-T diagnostic RE와 free-T solve 모두 CMFGEN 동종 단열항 폐합 전
  `BLOCKED_INCOMPLETE_ADIABATIC`.

## 정정 판정 — FIX_T와 adiabatic RE

선행 응답의 “`FIX_T=T`이므로 fixed-T RE 잔차에는 단열항이 진입하지 않는다”는 문장은
CMFGEN 호출부와 모순되어 후속 질의를 보냈다.

- 정정 질의: `docs/QUERY_FABLE_SH_RADEQ_ADIABATIC_CORRECTION_2026-08-08.md`
- 응답 canonical model: `claude-fable-5`
- session id: `cfb08109-babc-465b-9162-6fbb068e8a71`
- CLI status: success, `stop_reason=end_turn`, permission denial 0

Fable 정정 원문:

```text
FIX_T_RE_ADIABATIC = INCLUDED — VADAT:119(`T [INC_AD]`)와 VADAT:622(`T [FIX_T]`)가 동시에 설정된 상태에서, `new_main/cmfgen_sub.f:2939-2962`는 RD_FIX_T를 조건으로 걸지 않고 `EVAL_ADIABATIC_V3(..., INCL_ADIABATIC, ...)`를 무조건 호출하고, `eval_adiabatic_v3.f:237-250`은 INCL_ADIABATIC=T이면 전 depth에서 `NEW_STEQ_T(I)=NEW_STEQ_T(I)-WORK(I)`를 수행함. `solve_for_pops.f:77-90`의 FIX_T 처리는 solve 단계에서 온도 행을 교체하는 후단 제약일 뿐 이미 적산된 STEQ_T의 단열항을 제거하지 않음. 따라서 FIX_T=T여도 diagnostic STEQ_T(RE 잔차)에는 adiabatic cooling이 포함되며, fixed-T CMFGEN STEQ_T와의 동종 대조에도 CMFGEN과 동일한 adiabatic 항이 필요함.

Q3_CORRECTION = 판단불가 — 선택지 A/B/C의 정의가 본 질의에 제시되지 않아 문자 판정은 불가. 실질 정정 내용은 다음과 같음: (i) fixed-T RE_INTEGRAL/STEQ_T 대조 gate — adiabatic 항 포함 필수(단열항이 diagnostic 잔차에 진입하므로 "fixed-T 경로는 gate 대상 아님"이라는 선행 문장은 오류). (ii) free-T solve gate — 기존대로 adiabatic 항 포함 필수(온도 행이 교체되지 않으므로 STEQ_T가 그대로 solve에 진입).

PRIOR_VERDICT = REVISE 유지 — 선행 판정의 REVISE 결론 자체는 유지하되, Q3의 근거 문장("FIX_T=T이므로 단열항이 진입하지 않는다")은 제출된 소스 라인 증거에 의해 반박되므로 해당 문장을 삭제하고 fixed-T 대조 gate에도 adiabatic 항 요구를 추가하는 방향으로 정정 범위를 확대해야 함.
```

정정 후 효력: fixed-T라고 단열항을 생략할 수 없다. 현 Lumina의
`3*n_e*k*T/t`만으로 CMFGEN fixed-T `STEQ_T` 대조를 PASS시킬 근거도 없다.
