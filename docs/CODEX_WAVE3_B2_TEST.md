최종 판정은 `EW_VALID_P_ELEM_SCOPE_FAIL`입니다. II–IV 내부 구조·수치 게이트는 통과했지만, s8 S·Fe의 pair-wise 대비 결과는 모두 **“무개선/악화”**이며 25% 문턱을 넘지 못했습니다. Wave 3 Stage 2A acceptance, `map recovery`, production 확장은 성립하지 않습니다.

### 1. §4.1 구조·수치 절대 게이트

| 항목 | s8 S | s8 Fe | 판정 |
|---|---:|---:|---|
| target coverage | 574/574 | 4076/4076 | PASS |
| rank | 303/303 | 303/303 | PASS |
| κ₂ | `3.8971e7` | `1.7859e7` | PASS |
| pivot growth | `0.9996` | `1.3932` | PASS |
| scaled residual | `1.1171e-15` | `1.1378e-15` | PASS |
| conservation | `0` | `6.72e-16` | PASS |
| permutation Δ | `1.22e-11` | `6.75e-14` | PASS |
| negative/nonfinite/guard | 0/0/0 | 0/0/0 | PASS |
| boundary fraction | `1.958e-5` | `7.923e-5` | **FAIL** |
| boundary producer coverage | 0 | 0 | **FAIL** |

근거: [S diagnostics](/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z16_s008_diagnostics.csv), [Fe diagnostics](/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z26_s008_diagnostics.csv).

### 2. §4.3 s8 pair-wise 대비

명세의 조건부 `p_ref`와 parity59 frozen pair-wise 결과를 사용했습니다. `p_ref`가 문서에 약 3자리로 제시되어 아래 백분율은 그 정밀도 범위의 값입니다.

| 원소 | `D(pair)` | `D(elem)` | improvement | 25% 판정 |
|---|---:|---:|---:|---|
| S | 0.75779 | 1.81723 | **−139.8%** | **무개선/악화** |
| Fe | 0.64057 | 1.23735 | **−93.2%** | **무개선/악화** |

Stage별 `d_k`도 모두 악화했습니다.

- S II/III/IV: `0.1045/0.00208/2.1668 → 2.2934/0.03130/3.1270`
- Fe II/III/IV: `0.5563/0.02560/1.3398 → 1.3395/0.21505/2.1575`

지배 ion 절대오차 역시 S `0.00466→0.06794`, Fe `0.05707→0.38937`로 증가하여 §4.3(3)도 실패합니다. 산출 해는 [S solution](/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z16_s008_solution.csv), [Fe solution](/tmp/w31_on_a.JuCpDY/lumina_ew_iter0011_z26_s008_solution.csv)입니다.

`b_k`의 명세상 full-active median은 산정 불가입니다. pair-wise dump에는 II/III만 있고 IV active-level 기준선이 없어 303/303 SL 조인이 되지 않습니다. Gate B에서 직접 조인되는 대표 2행/원소만 보면:

| 원소 | median log-error pair→elem | 감소율 | 분류 |
|---|---:|---:|---|
| S | `0.04732→0.04633 dex` | `2.11%` | **방향 일치/효과 부족** |
| Fe | `0.04626→0.04626 dex` | `0%` | **무개선/악화** |

대표행만의 후보 median/p95는 S `0.0463/0.0831 dex`, Fe `0.0463/0.0587 dex`지만, 이는 각각 2/303 SL에 불과하므로 **“absolute provisional pass”로 표기할 수 없습니다**. 또한 ON-shadow oracle은 OFF와 동일하여 `>=1%`인 `oracle-measured` 표적도 0건입니다.

### 3. §4.4 s0 Fe 역방향 축

| 양 | pair `d_k` | elem `d_k` | 방향 |
|---|---:|---:|---|
| Fe II | 4.8875 | 1.5528 | 개선 |
| Fe III | 1.1385 | 0.9670 | 개선 |
| Fe IV | 0.00306 | 0.00479 | **악화** |

`D` 자체는 `2.0097→0.8415`, 즉 **58.13% 감소**했지만, 명세는 II/III/IV 모두 `d_k(elem)<d_k(pair)`를 요구합니다. Fe IV가 악화했으므로 **s0 Fe 축 회복 실패**입니다. s0도 boundary fraction `1.3749e-2`, producer coverage 0으로 `SCOPE_FAIL`입니다. [s0 diagnostics](/tmp/w31_s0_fe.C6wf6v/lumina_ew_iter0011_z26_s000_diagnostics.csv)

s20 S `p_elem`은 이번 Wave 3.1 산출물에 없으므로 §4.4 전체의 `map recovery`는 성립하지 않습니다.

### 4. SCOPE_FAIL 영향 상한

경계 population 질량만 직접 재분배한다고 제한하면:

- 절대 fraction 영향 상한: S `1.958e-5`(0.001958%), Fe `7.923e-5`(0.007923%).
- 각 stage를 독립적으로 `±boundary_fraction`까지 유리하게 이동시키는, 보존식보다도 관대한 최선 상한에서도 improvement는 S **−136.8%**, Fe **−23.45%**입니다. 따라서 경계 질량 bookkeeping만으로 25% PASS로 뒤집힐 수 없습니다.

다만 `boundary_process_coverage=0`이므로 누락된 I/V rate·heating feedback의 물리적 영향에는 유한 상한을 줄 수 없습니다. 작은 경계 population이 큰 drain/source rate를 가질 가능성까지 population fraction으로 제한할 수 없기 때문에 이것이 acceptance를 계속 막는 이유입니다.

### 5. §4.2·§4.5 및 재현성/OFF

- 동일 identity/frozen state의 ARTIS matrix dump가 없어 §4.2 PASS는 성립하지 않습니다.
- released-T 수렴 CMFGEN 앵커가 없으므로 §4.5 최종 CMFGEN acceptance는 **판정 금지**입니다.
- ON shadow 반복: CSV 17/17 byte-identical.
- OFF 미설정 대 명시 `0`: s0·s8 모두 `cmp=0`.
- ON-shadow oracle도 OFF와 byte-identical.
- SHA-256:
  - s8: `2ee175c6be4a0ab6ae07034722180c6312f0a6e55e1009fcdda23b72310693eb`
  - s0: `7a79f4f345d5c8500b48530a7f859bbe05a57771f43e52f983639ef9a0839381`
- 두 OFF 산출물은 저장된 parity59 oracle과도 byte-identical입니다.
- s43 OFF는 이번 Wave 3.1 산출물에서 새로 재실측되지 않아 §6.3의 3셀 전체 배터리는 부분 완료입니다.

소스 수정, GPU 실행, git 명령은 수행하지 않았습니다.