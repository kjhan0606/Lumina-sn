# CMFGEN/ARTIS 물리량 비교 준비도 — 2026-08-08

## 결론

첫 정량 비교는 **SH-GRID 이전에도 고정 snapshot kernel oracle**로 가능하다. 다만 그것은
수렴 solution 비교가 아니다. `T_e`, `n_e`, 이온분율의 solution 비교는 SH-GRID migration과
DET 수렴 뒤, spectrum과 packet-energy 비교는 MC 수렴 뒤에만 판정한다.

## 비교 단계

| 단계 | 비교량 | 가능한 시점 | 판정 범위 |
|---|---|---|---|
| K0 | 단위, 부호, analytic known answer | 현재 | 구현 구조 |
| K1 | `u_atom`; 단열 네 항; bin별 `chi`, `eta`; A210 `H`, `C`, `H-C` | 동일 고정 snapshot dump 연결 뒤 | 물리 kernel |
| S1 | shell별 `T_e`, `n_e`, 원소/이온분율, population, 잔차 | SH-GRID + 수렴 DET 뒤 | material solution |
| R1 | `J_nu`, heating estimator, deposition, packet energy | 수렴 MC/ARTIS epoch 정렬 뒤 | radiation-energy coupling |
| P1 | observer-frame spectrum과 band luminosity | R1 뒤 | 최종 관측량 |

## K1 canonical 열

공통 identity:

```text
epoch_s, shell_id, v_inner_cm_s, v_outer_cm_s, r_center_cm,
atomic_model_sha256, geometry_sha256, radiation_generation,
population_generation, te_generation, opacity_generation, emissivity_generation
```

material과 단열:

```text
T_e_K, n_e_cm3, n_atom_cm3, u_atom_erg,
q_ad_temperature_gradient, q_ad_velocity_divergence,
q_ad_electron_fraction_gradient, q_ad_internal_energy_gradient,
q_ad_signed_total, q_ad_heating, q_ad_cooling
```

주파수 셀:

```text
nu_lo_Hz, nu_hi_Hz, J_nu,
chi_es_cm1, chi_bb_cm1, chi_bf_cm1, chi_ff_cm1, chi_total_cm1,
eta_bb, eta_bf, eta_ff, eta_true_total
```

A210 적분 ledger:

```text
photo_heat, line_abs_heat, ff_abs_heat, compton_heat, gamma_heat,
recomb_cool, line_emit_cool, coll_line_cool, ff_emit_cool, compton_cool,
adiabatic_heat, adiabatic_cool, sum_heating, sum_cooling, residual
```

`chi`는 `cm^-1`, `eta`는 `erg s^-1 cm^-3 Hz^-1 sr^-1`, 모든 체적률은
`erg s^-1 cm^-3`로 고정한다. 복사 적분은 공통 convention으로
`H_nu = 4 pi chi_nu J_nu`, `C_nu = 4 pi eta_nu`를 사용한다. 단열 signed total은
양수가 cooling, 음수가 heating이다.

## 필수 정렬 gate

- 같은 explosion epoch, composition/atomic coverage, homologous shell 구간만 비교한다.
- shell은 겹치는 체적에 보존적으로 투영하며 외삽하지 않는다.
- 주파수는 공통 edge에 적분 보존 재격자한다. center interpolation만으로 비교하지 않는다.
- comoving/observer frame, `per sr`와 `4 pi`, frequency/wavelength Jacobian을 manifest에 쓴다.
- A208/A209/A210과 material의 generation/provenance가 한 transaction인지 먼저 검사한다.
- DET/MC solution 비교에는 수렴 기준과 마지막 반복 변화량을 함께 기록한다.

## oracle별 한계

- CMFGEN: `RVTJ`, population/temperature 출력과 `EVAL_ADIABATIC_V3` diagnostic을 K1/S1
  oracle로 쓸 수 있다. bin별 `chi/eta`는 동일 depth-frequency dump가 있어야 한다.
- ARTIS: `T_e`, `n_e`, ion population, estimators, deposition과 packet/spectrum은 비교 가능하다.
  상세 BF/line estimator는 compile-time 출력 옵션이 필요하다.
- 조사한 ARTIS 출력에는 exact adiabatic-loss accumulator가 없다. input−final residual을
  adiabatic loss라고 부르지 않으며, 직접 비교하려면 ARTIS 계측을 추가한다.

따라서 가장 먼저 만들 실측 표는 K1의
**CMFGEN 단열 네 항 + chi/eta + A210 H−C**이고, 그 다음이 S1의
**Te/ne/ion fraction**이다.
