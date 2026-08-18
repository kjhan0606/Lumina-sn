# Fable 감리 질의 — SH-RADEQ 구현 반영 검토 (2026-08-08)

역할: 물리/구조 감리자. 아래 제출 증거만 사용해 `ACCEPT`, `REVISE`, `BLOCKED` 중
하나를 고르고, 반드시 결함을 구체적인 불변식 단위로 적어라. 코드를 작성하지 말라.

## 선행 판정

1. 온도 producer는 `RE_INTEGRAL`; `EHB_THERMAL`은 독립 diagnostic.
2. 선 방출은
   `eta_nu=n_u*A_ul*h*nu*beta_esc(tau)/(4*pi*Delta_nu)`,
   `beta=(1-exp(-tau))/tau`, `tau->0 => beta->1`.
3. `n_u`, `A_ul`, signed `tau`는 같은 immutable generation-bound view.
4. 기준 CMFGEN은 `FIX_T=T, INC_AD=T`; fixed/free-T 모두 완전 CMFGEN 단열항 없이는
   통과 금지.

## 구현 제출

### A. 직접 선 방출

순수 함수 `a209_sobolev_line_eta`:

```c
beta = (tau == 0.0) ? 1.0 : -expm1(-tau)/tau;
eta = n_upper*A_ul*h*nu*beta/(4*pi*delta_nu);
```

- 유한성/부호 검사 후 `n_upper==0 || A_ul==0`만 exact zero.
- A2-09 production의 `opacity->line_source_S`와
  `opacity->line_source_validity` read는 0개.
- 정상 cell에서 직접식과 `chi*S` 상대 closure `<=1e-12`, `tau=0,n_u>0`,
  작은 양의 tau, 음의 tau, 0/음수 population 자가검사 통과.
- `LUMINA_OPACITY_SKIP_Z`와 Z-inert는 방출도 skip한다.

### B. population/tau view 선택과 세대

- A2-08 publication 직후 A2-09가 다음을 모두 검사한다.

```text
raw tau computed == raw tau required != 0
A2-08 tau_generation == raw tau computed
A2-08 population_generation == atom population committed
A2-08 te_generation == plasma T_e generation
A2-08 epoch == A2-09 epoch
NLTE population committed is either 0 or == atom population committed
```

- bulk tau writer와 똑같이 line `(Z,ion,lower,upper)`를 resolve한다.
- NLTE writer가 실제로 덮은 조건(두 준위 모두 NLTE mapped, pair-owned 또는 현재
  element-wide authority, not `NLTE_SKIP_Z`)에서만 committed NLTE `n_u`를 사용한다.
  나머지는 bulk tau writer와 동일한 committed ion density, partition, T_e로 LTE
  `n_u`를 재계산한다.
- 단, 1.25억 cell의 두 번째 복사를 피하려고 `tau_sobolev[]/tau_validity[]` 숫자 자체는
  여전히 raw `OpacityState` slab을 읽는다. 해당 slab을 바꾸는 정상 writer는 generation을
  올리지만 C type상 const/물리적 불변 메모리는 아니다. A2-08 publication에는 세대 토큰만
  있고 line slab 사본/해시는 없다.
- 5개 정적 음성대조가 abort 무력화, abort return 제거, 직접식 제거, population generation
  결박 제거, tau=0 극한 훼손을 모두 검출한다.

### C. RE/EHB 및 단열

- `A210EquationKind={NONE,RE_INTEGRAL,EHB_THERMAL}`를 ledger와 publication에 추가.
- `a210_solve_transaction`은 residual ledger가 `RE_INTEGRAL`일 때만 T_e publication을
  허용한다. EHB ledger 자체의 finalize는 허용하되 producer 주입은 rc=5로 거부한다.
- production residual은 `equation_kind=RE_INTEGRAL`.
- 현재 전자 병진 단열 진단값은 보존하지만
  `adiabatic_model=ELECTRON_TRANSLATIONAL_ONLY`, term status=`A210_INCOMPLETE`다.
  finalize 결과는 `RADEQ_INCOMPLETE_ADIABATIC`; solve는 rc=3, publication 0.
- `LUMINA_FIXED_TE_PROFILE` 경로도 같은 사유를 이름으로 출력하고 publication/material
  update를 차단한다.
- 자가검사에서 완전 단열 synthetic RE만 T_e producer가 되고, incomplete RE와 EHB
  producer 주입은 public T_e/세대를 바꾸지 못한다.

### D. MC-EVT 선결 결과

실제 MC-EVT 덱에서 `NLTE_NU_MIN=1.5e14 Hz` 이하 양의 BF edge가 707개이고 모두
기본 활성+CMFGEN sigma다. 따라서 OOG exact-zero 정책을 적용하지 않고 SH-GRID를
재개방했다.

## 질문

Q1. A/B의 직접 방출 구현이 선행 판정 2·3을 충족하는가? 특히 raw tau slab+generation
token은 이번 단계에서 허용 가능한 generation-bound view인가, 아니면 publication 전에
반드시 compact immutable tau view/해시/별도 population-generation stamp가 필요한가?

Q2. C의 `RE_INTEGRAL` 유일 producer와 `EHB_THERMAL` diagnostic 분리가 충분한가?

Q3. 불완전 단열항을 값은 진단으로 남기되 fixed/free-T publication을 모두 차단한 것이
선행 판정을 정확히 구현하는가?

Q4. 다음 flight 전에 반드시 고쳐야 할 물리/구조 결함을 중요도순으로 적어라.

마지막 줄은 정확히 다음 형식으로 써라.

```text
IMPLEMENTATION_VERDICT = ACCEPT|REVISE|BLOCKED
```
