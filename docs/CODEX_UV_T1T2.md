# T1·T2 — 결맞음 가정과 선 조립의 단일-인자 판별

판정일: 2026-08-02 (Asia/Seoul)  
범위: 동결 parity59 payload와 기존 stage31 CPU formal solver를 사용한 오프라인 시험. 생산 코드 수정, 신규 모델/GPU run, clamp/floor/fallback, commit 없음.

## 0. 결론

| 시험 | 측정 상태 | 핵심 관측 | 사전등록 판독 |
|---|---|---|---|
| **T1 균일 R** | **RESOLVED** | B0 `8.29055→25.90856×CMFGEN` 악화, B1 `4.91614→2.25291×` 개선; B0/B1 동시 개선 gate 실패 | **결맞음/재주입 가정 자체가 문제** (`SAME_BIN_COHERENCE_ASSUMPTION_IS_THE_PROBLEM`) |
| **T2 native χ+η** | **UNRESOLVED** | B2는 η만 바꾸고 χ는 A와 bitwise 동일. exact iter-10 하위준위 population과 선별 `χ_line/χ_line_th/χ_abs`가 payload에 없음 | BALL 붕괴를 측정하지 못했으므로 **진폭 원인 최종 확정 불가** |

T1은 균일 R가 B1·B3·B4를 개선했지만 판별의 핵심인 B0를 E10/E12와 같은 수준으로 크게 악화했다. 따라서 MC R의 세부 branch 형상만 바꾸면 해결된다는 가설은 지지되지 않는다. 선 흡수 에너지를 다시 복사장에 넣는 재순환 가정이 남아 있는 한 B0 과잉이 유지된다.

T2는 실패값을 수치로 위장하지 않았다. 현재 입력으로 native η는 B2에서 재사용할 수 있지만, 같은 population epoch의 native χ를 전 50 shell에 조립할 수 없다. 다른 epoch population, `chi_coherent-min(chi_coherent)` proxy, B2의 η-only trip, clamp를 대신 쓰면 차터의 단일 인자를 깨므로 결과는 `UNRESOLVED`다.

## 1. 사전등록과 동결 좌표

측정 전에 `validation/uv_t1t2/preregistration.json`을 작성해 SHA-256
`b516c32267c265a0e7060e228b6f9a4e68a3d10eea913a0da86653364f3c75ae`로 고정했다.

- T1 예측: same-bin reinforcement를 제거하면 B0/B1의 `|log10(J/CMFGEN)|`가 둘 다 E9보다 작아질 것으로 예측했다. 둘 다 개선하면 `R_SHAPE_IS_THE_PROBLEM`, 하나라도 실패하면 `SAME_BIN_COHERENCE_ASSUMPTION_IS_THE_PROBLEM`으로 판독하도록 고정했다.
- T2 예측: native χ와 η를 함께 쓰면 A-lane BALL `11.70379136×CMFGEN`이 O(1)로 붕괴할 것으로 예측했다. O(1)은 사전에 `[1/3,3]`으로 정의했다. 붕괴 뒤 B0/B1 중 하나가 이 구간 밖이면 잔여 형상 문제로 분리하도록 등록했다.
- 공통 좌표: shell 8, `nmu=16`, `T_inner=10020 K`, `bb_scale=1`, iteration/generation 10 post-damping, 같은 canonical 50×1000 격자와 같은 CMFGEN jnu4.

T1 입력은 `fluormat_capture_188828`의 A/chieta SHA
`894a4ee8...863d5dc`, full matrix SHA `08ff3312...735af6`과 같은-capture E9 SHA
`0bd71b41...6d5e0`이다. T2 감사 입력은 `emiss_ab2_capture_188766`의 A/chieta SHA
`ac62eae5...a34011`, B2 SHA `775c9e84...89f5a`다.

## 2. T1 — 균일 R

### 2.1 단일 인자 구성

E12와 같은 동시대 E9 frozen source의 shell 8에서 E10/E12와 동일한 분해를 사용했다.

```text
chi_e(s)       = min_nu chi_coherent,A(s,nu)
chi_line       = chi_coherent,A - chi_e
eta_line,ret   = (1-eps_MC) chi_line J_E9
eps_MC         = 0.0024368222433042742

P_removed      = sum_i eta_line,ret(i) Delta_nu_i
R[j,i]         = 1/1000
eta_uniform(j) = P_removed / (1000 Delta_nu_j)

chi_coherent,T1 = chi_e
eta_T1           = eta_fixed + chi_e J_E9 + eta_uniform
```

이는 **출력 빈당 같은 에너지 확률**을 주는 명시적 단순 분포다. `chi_total`, population,
continuum, radial/frequency grid, EPAY 결과, 경계, stage31 solver와 나머지 49 shell은
동결했다. stage31 driver는 frozen `eta_total`을 한 번 푸는 `SCAT_NONE` 계약이므로 새로운
산란 재수렴은 없다. 직렬화된 `chi_coherent=chi_e`는 source 분해의 provenance이며,
extinction `chi_total`은 바꾸지 않았다.

| 구성 장부 | 값 |
|---|---:|
| 제거한 동일-빈 비열적 선 반환 에너지 | `6.7651957989972485e-3` |
| 1000 빈 균일 재주입 에너지 | `6.7651957989972485e-3` |
| 적용 에너지 상대오차 | **0.0** |
| R 열 합 | **1.0** |
| E9 source 재구성 최대 상대오차 | `1.6102603843547383e-16` |
| negative / nonfinite / clamp / fallback | `0 / 0 / 0 / 0` |

### 2.2 대역표와 E10/E12 대조군

모든 값은 `J_det/CMFGEN`이다. E10은 iteration-11 prefix MC R, E12는 동시대 full
LFMAT001 MC R의 기존 측정값을 재인용했다.

| 대역 [Å] | E9 same-bin | **T1 균일 R** | T1/E9 | E10 MC-prefix R | E12 MC-full R | T1 판독 |
|---|---:|---:|---:|---:|---:|---|
| B0 600–1000 | 8.29055106 | **25.90856042** | 3.12507097 | 20.90950168 | 26.43249460 | **악화** |
| B1 1000–1500 | 4.91614286 | **2.25290853** | 0.45826751 | 3.58482097 | 5.65886463 | 개선 |
| B2 1500–2000 | 1.83988084 | 0.22120096 | 0.12022570 | 1.49060727 | 1.69129789 | 악화(과소) |
| B3 2000–2500 | 0.20836087 | 0.26726326 | 1.28269414 | 0.86053383 | 0.60935745 | 개선 |
| B4 2500–3000 | 0.33680469 | 0.84276193 | 2.50222744 | 0.74457002 | 1.48272165 | 개선 |
| **BALL 600–3000** | **0.93228813** | **0.63487645** | **0.68098737** | 1.17171673 | 1.52867739 | 진폭보다 형상 실패 |
| optical 3000–10000 | 6.92103893 | **9.89007391** | 1.42898689 | 7.28493778 | 6.37416008 | 악화 |

균일 R의 source 자체는 B0에서 E9의 `3.03678×`, B1에서 `0.38256×`, BALL에서
`0.34435×`, optical에서 `2.04124×`였다. formal transport는 이 방향을 유지해 B0를
`25.91×`까지 올렸다. 따라서 E10의 B2→B0 branch 하나만 없애도 B0가 회복된다는 설명은
성립하지 않는다. 균일 분포도 B0에 충분한 에너지를 배정해 같은 실패 부호를 낸다.

사전등록 gate는 B0와 B1의 log-distance가 **모두** 줄어야 통과한다. B1은 통과했지만 B0는
`|log10 ratio|=0.91858→1.41344`로 악화했으므로 gate는 FAIL이다. 차터 문법에 따른 T1
판독은 **“R의 형상만의 문제가 아니라 결맞음/재주입 가정 자체가 문제”**다.

### 2.3 Gamma

같은 sigma, threshold, route, within-SL quadrature로 Fe III idx201과 S II SL4를 재생했다.

| target | Gamma_CMFGEN [s^-1] | Gamma_E9 | **Gamma_T1** | T1/E9 | T1/CMFGEN |
|---|---:|---:|---:|---:|---:|
| Fe III C48 lump, idx201 | 28.07174861 | 579.35771180 | **1640.27057378** | 2.83118795 | 58.43136444 |
| S II SL4, idx4 | 0.474807695 | 1304.04046467 | **1875.11751129** | 1.43792893 | 3949.21466292 |

균일 R는 BALL 평균을 낮췄지만 고에너지 빈에 동일 확률을 주므로 photoionization Γ는 둘 다
악화했다. 이는 BALL 하나만으로 형상 성공을 선언할 수 없다는 독립 확인이다.

### 2.4 solver·결정성 장부

| 항목 | 값 |
|---|---:|
| transport residual | `8.1805706575466152e-7` |
| source residual / iterations | `0 / 1` |
| clamp / solution-negative-excess / sign-uncertain / nonfinite | `0 / 0 / 0 / 0` |
| raw negative J_det bins | `0` |
| raw minimum J_det | `3.2075086566449516e-10` |
| stage31 3회 SHA-256 | `1bdbedb8...ae91` 3/3 동일 |

`bdf_eta_negative=364053`, solution-subtruncation `124996`, sign-indeterminate
subtruncation `973099`, enclosure restart `1624`는 숨기지 않았다. 인증 guard를 넘는 음수는
0이고 완전한 1000-row 표가 생성됐다.

## 3. T2 — population-native χ+η

### 3.1 입력 감사 결과

T2가 요구하는 단일 인자는 선 조립 전체를

```text
chi_l = (pi e^2 / m_e c) f_lu n_l (1 - g_l n_u/(g_u n_l)) / Delta_nu_b
eta_l = (h nu / 4pi) A_ul n_u / Delta_nu_b
```

로 함께 바꾸는 것이다. 그러나 제공된 LCMFCE01 payload는 다음 아홉 배열만 가진다.

```text
r_edge, nu, dnu, chi_total, chi_coherent,
eta_fixed, eta_coherent, eta_total, J_producer
```

`chi_line`, `chi_line_th`, `chi_abs` 또는 per-line χ가 직렬화되지 않았다. 따라서 기존 선
opacity만 정확히 빼고 continuum을 유지하는 연산을 payload에서 역산할 수 없다.

B2가 T2를 대신하지 못한다는 bitwise 감사 결과는 다음과 같다.

| A 대 B2 배열 | 다른 cell | 최대 절대차 | 판독 |
|---|---:|---:|---|
| `chi_total` | **0** | 0 | χ 미교체 |
| `chi_coherent` | **0** | 0 | χ 미교체 |
| `eta_fixed` | 6,646 | `4.7095022617963364e-12` | native η 개입 |
| `eta_total` | 6,646 | `4.7095022617963364e-12` | native η 개입 |
| `eta_coherent`, grid, J | 0 | 0 | 동결 |

B2 manifest의 공식은
`covered:hnu-over-4pi-times-Aul-times-n_upper-over-dnu;undefined:production-A-retained`이고
common assembly-state SHA는 `302a64e...bf4b044`다. 즉 B2는 차터가 지적한 그대로 η-only
자산이다.

추가 입력도 exact T2를 닫지 못한다.

- `cmf_fine_linedump_s{8,45,49}.csv`는 세 shell만 덮는다. stage31은 전 50 shell의 χ를 소비한다.
- line dump 열에는 `tau_sob`가 있지만 `n_l,n_u,g_l,g_u,f_lu`가 없고 나머지 47 shell의 per-line τ도 없다.
- `lumina_levelpop.csv`는 stdout에서 iteration-10 A/B2 쓰기 **뒤**, `final pure-CMFGEN it=11` resolve 직전에 작성됐다. iteration-10 population payload로 인증되지 않는다.
- B2 undefined 전이는 population 미추적으로 명시돼 있다. exact lower population도 없으므로 stimulated-opacity 항을 전 active 집합에 조립할 수 없다.

### 3.2 판독

T2의 BALL, B0, B1과 Γ는 모두 **UNRESOLVED**다. 기존 B2 η-only solve가 1208.743 Å에서
certified-negative로 중단된 사실은 참고 가능한 E5 결과지만, χ까지 교체한 T2 결과가
아니므로 재사용하지 않았다. final-iteration population, χ proxy, 누락 전이 zeroing,
guard 완화 중 하나라도 사용하면 population/continuum/solver 동결 또는 단일 인자 규율을
깬다.

따라서 사전 예측 “BALL이 `[1/3,3]`으로 붕괴”는 **측정되지 않은 채 유지**된다. 진폭 원인
최종 확정과 붕괴 후 B0/B1 잔여 형상 분리는 둘 다 `UNRESOLVED`다.

exact 재판정에 필요한 최소 추가 frozen artifact는 같은 iteration/generation 10의 전 50
shell per-line `(line_id,n_l,n_u,g_l,g_u,f_lu,A_ul,nu)` 또는 동등하게 검증된 native
`chi_line`와, 기존 `chi_abs/chi_line/chi_line_th` 분해다. 이를 얻는 재캡처는 신규 모델/GPU
run 금지 범위 밖이므로 이번 시험에서 수행하지 않았다.

## 4. 공통 규율 준수

| 규율 | 결과 |
|---|---|
| 시험당 단일 인자 | T1: R만 uniform으로 교체, 나머지 동결. T2: exact 단일 인자 입력 부재 시 중단 |
| 사전등록 후 측정 | PASS, prereg SHA `b516c322...c75ae` |
| clamp/floor/fallback | 0 |
| 생산 코드 수정 | 0; 신규 파일은 오프라인 판독기·산출물·본 보고서뿐 |
| 신규 모델/GPU run | 0; stage31 CPU formal solve만 실행 |
| commit | 0 |
| 판단 불가 | T2를 명시적으로 `UNRESOLVED` 처리 |

## 5. 전 수치 재현 명령

repository root에서 실행한다.

```bash
python3 -m py_compile scripts/uv_t1t2_offline.py
python3 scripts/uv_t1t2_offline.py

python3 scripts/cmf_chieta_check.py \
  validation/uv_t1t2/t1_uniform_iter10

sha256sum \
  validation/uv_t1t2/preregistration.json \
  validation/uv_t1t2/t1_uniform_iter10 \
  validation/uv_t1t2/t1_jdet_s8.tsv \
  validation/uv_t1t2/t1_jdet_s8_repeat2.tsv \
  validation/uv_t1t2/t1_jdet_s8_repeat3.tsv \
  validation/uv_t1t2/t1_summary.json \
  validation/uv_t1t2/t2_availability.json

cmp validation/uv_t1t2/t1_jdet_s8.tsv \
  validation/uv_t1t2/t1_jdet_s8_repeat2.tsv
cmp validation/uv_t1t2/t1_jdet_s8.tsv \
  validation/uv_t1t2/t1_jdet_s8_repeat3.tsv

cat validation/uv_t1t2/t1_band_table.csv
cat validation/uv_t1t2/t1_gamma_table.csv
python3 -m json.tool validation/uv_t1t2/t1_construction.json
python3 -m json.tool validation/uv_t1t2/t1_summary.json
python3 -m json.tool validation/uv_t1t2/t2_availability.json
python3 -m json.tool validation/uv_t1t2/combined_summary.json
```

핵심 산출물 SHA-256은 T1 payload `9457508d...c690b`, stage31 표
`1bdbedb8...ae91`, T1 summary `e26dc1ee...ae91`, T2 availability
`ef20bcbd...5261`이다.
