# CMFGEN `EVAL_ADIABATIC_V3` → Lumina 상태 대응 — 2026-08-08

상태: 공식 2025-06-18 CMFGEN 소스의 **적분형 복사평형(`STEQ_T`) 경로**를
대상으로 수식 대응 완료. 아직 production 온도 solve에는 연결하지 않았다.

Fable 중요 판정은
`docs/FABLE_VERDICT_SH_RADEQ_COMPLETE_ADIABATIC_2026-08-08.md`에 보존했다.

공식 배포본:

- `https://sites.pitt.edu/~hillier/web/CMFGEN.htm`
- `https://sites.pitt.edu/~hillier/cmfgen_files/cur_cmf_18jun25.tar.gz`
- 대상 routine: `cur_cmf/new_main/subs/eval_adiabatic_v3.f`

## 1. CMFGEN이 적분형 RE에 넣는 네 항

`UPDATE_BA_ST(BA_T, STEQ_T, INT_EN, TOT_ENERGY)` 호출과
`NEW_STEQ_T(I)=NEW_STEQ_T(I)-WORK(I)`를 cgs로 풀면, depth point `i`의 signed
adiabatic cooling은 다음과 같다.

```text
q_ad = q_dT + q_divv + q_dgamma + q_dUint

q_dT     = (3/2) (n_atom + n_e) k_B v dT/dr
q_divv   =       (n_atom + n_e) k_B T div(v)
q_dgamma = (3/2) n_atom k_B T v d(n_e/n_atom)/dr
q_dUint  =       n_atom v d(u_int/atom)/dr
```

CMFGEN 기호와의 대응은 다음과 같다.

| CMFGEN | 의미 | Lumina 정본 후보 |
|---|---|---|
| `POP_ATOM` | 총 원자핵 수밀도 | trial `ion_number_density`의 전 ion 합 |
| `ED` | 전자 수밀도 | trial `plasma->n_electron` |
| `T` | 전자온도, CMFGEN은 `10^4 K` 단위 | trial `T_e [K]` |
| `R`, `V` | radius/velocity depth point | `Geometry` shell-center `r,v` |
| `SIGMA` | `dlnV/dlnR - 1` | homologous geometry이면 정확히 0 |
| `GAMMA` | `ED/POP_ATOM` | `n_e/n_atom` |
| `INT_EN` | 원자당 여기+전리 내부에너지 | 아래의 `u_int/atom [erg]` |

homologous expansion에서는 `v=r/t`, `div(v)=3/t`이므로
`q_divv=3(n_atom+n_e)k_BT/t`다. 현 Lumina 항
`3 n_e k_BT/t`는 이 네 항 중 전자 병진 velocity 항 하나뿐이다.

## 2. 내부에너지 기준점

CMFGEN `TOT_ENERGY` 생성(`eval_adiabatic_v3.f:89-112`)은 각 원소의 neutral ground를
0으로 둔다. ion stage `z`, level `l`의 에너지는

```text
E_total(z,l) = sum_{j=0}^{z-1} chi(j) + E_exc(z,l)
```

이다. 따라서 Lumina 후보는 다음과 같다.

```text
u_int/atom = sum_{z,l} n(z,l) E_total(z,l) / n_atom
```

- `chi(j)`: `ioniz_energy_eV(Z,j)`의 누적합.
- `E_exc`: 해당 ion의 최저 catalog energy를 뺀 `level_energy_eV`.
- NLTE 권한 ion/level: 같은 trial의 reconstructed full-level population.
- 그 밖의 ion: 같은 trial `T_e`, partition, ion density로 만든 LTE level population.
- 공용 level catalog가 없는 top ion: `topion_E_cm/topion_g` LTE 평균을 사용한다.

level population의 합이 ion population과 닫히지 않거나, ionization ladder가 중간에서
끊기거나, top-ion partition과 energy catalog가 대응하지 않으면 `CMFGEN_COMPLETE`로
승격하지 않고 candidate 전체를 폐기한다.

## 3. 공간 차분과 배열 방향

CMFGEN 배열은 바깥→안쪽으로 `R`이 감소하고, interior와 inner boundary 모두 인접한
안/바깥 point의 단순 선형 차를 쓴다(`:224-245`). Lumina는 안쪽→바깥쪽 shell 순서다.
같은 stencil을 뒤집으면:

```text
neighbor(s) = 1       , s = 0 (inner boundary one-sided)
              s - 1   , s > 0 (current outer cell minus adjacent inner cell)

dX/dr = (X[s] - X[neighbor]) / (r[s] - r[neighbor])
```

`s=0`과 `s=1`은 CMFGEN inner-boundary 처리처럼 같은 첫 간격의 gradient를 공유한다.
shell-center는 일단 `0.5*(inner+outer)`로 정의하되, 이 수치 배치는 Fable의 중요
구조 판정 뒤 고정한다.

## 4. 현재 scalar residual이 부족한 이유

현재 A2-10 callback은 `residual(shell, trial_T)`이며, committed `n_e`, opacity,
emissivity를 읽는다. 그러나 위 식은 한 평가마다 최소한 전 shell의
`T_e`, `n_e/n_atom`, `u_int/atom` 후보가 필요하다. 또한 A2-10 명세는 trial마다
population/opacity/emissivity를 같은 private token에서 다시 만들도록 요구한다.

따라서 scalar callback 안에 committed neighbor를 섞어 완전 단열항이라고 부르는 것은
금지한다. 다음 경계로 구현한다.

1. 전 shell 상태를 입력받아 네 signed component를 내는 순수 vector producer.
2. known-answer/부호/단위/배열방향 음성대조.
3. 전 shell private atomic trial을 생성하는 transaction callback.
4. 그 candidate에서 opacity/emissivity와 vector adiabatic을 함께 평가.
5. 모든 shell 수렴 뒤 `T_e/population/n_e/opacity/emissivity` 단일 commit.

## 5. signed ledger 요구

gradient 항 때문에 `q_ad`는 음수가 될 수 있다. CMFGEN은 signed `WORK`를 그대로
`STEQ_T`에서 빼므로, Lumina도 음수를 오류나 0으로 세탁하면 안 된다. 순수 producer는
네 component와 total을 signed로 보존한다. A2-10 ledger 연결 때는

```text
adiabatic_cooling = max(q_ad, 0)
adiabatic_heating = max(-q_ad, 0)
```

으로 분리하고 동일한 signed raw total/component를 진단 필드에 남겨야 한다. 현
heating enum에는 adiabatic slot이 없으므로 schema version 갱신이 필요하다.

## 6. 공식 소스의 진단 경로 주의점

2025-06-18 파일은 `STEQ_T` 주경로와 별개인 diagnostic `AD_CR_DT`에서
`COL_EN`을 사용하는 줄이 있고, 내부 contained routine의 EHB 호출 interior에는 host
`INT_EN`을 읽는 줄이 있다. 이번 대응은 오직
`UPDATE_BA_ST(BA_T,STEQ_T,INT_EN,TOT_ENERGY)`의 적분형 RE 주경로를 기준으로 한다.
EHB 또는 `COOLGEN` 진단 출력의 혼재를 복제하지 않는다.

## 7. steady V3와 time-dependent DDT 경계

`SN_MODEL .AND. DO_CO_MOV_DDT`이면 CMFGEN은 이 V3가 아니라
`EVAL_TEMP_DDT_V2`를 호출한다. 현재 비교 기준
`/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/VADAT`은 `T [INC_AD]`,
`F [DO_DDT]`임을 실측했으므로 V3가 동종 잣대다. 다른 덱에서 `DO_DDT=T`이면 이
producer를 production에 연결하지 않고 별도 DDT 계약을 요구한다.
