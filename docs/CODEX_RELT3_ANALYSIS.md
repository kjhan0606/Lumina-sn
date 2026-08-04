# relT3 실패 심층 독립 분석 보고서

## 결론

판정은 **(c) 혼합형**이다.

- `MAXCH`와 “% correction”의 \(10^5\)–\(10^7\%\)는 대부분 거의 비어 있는 고준위·terminal closure 변수에서 생기는 **조건화/잣대 아티팩트**다. 지배 이온과 질량가중 상태가 그 크기로 발산하는 증거는 없다.
- 그러나 물리 상태가 매끈하게 수렴하는 것도 아니다. it41–50에는 작은 비감쇠 진동·drift가 있고, it51 full BA 뒤에는 외곽 radiative luminosity가 실제로 약 2배가 되는 큰 복사장 이동이 발생했다.
- 따라서 “거대 correction = 전역 물리 발산”은 틀리지만, “잣대만 나쁘고 물리적으로 완전 수렴”도 틀리다.
- 현재 파일만으로 **두 개의 안정한 물리 해가 존재하는 쌍안정성**까지 입증할 수는 없다. 관측된 것은 한 번의 복사장 branch 이동과 이후 고 luminosity 상태의 지속이다.

금지된 `docs/FABLE_RELT3_*`는 열람하지 않았고, 신규 런이나 파일 변경 없이 기존 산출물만 사용했다.

---

## 1. it41→55 물리 가중 수렴 재측정

### 측정 방법

`POINT1`은 두 SCRTEMP가 각각 it50과 it55까지 저장됐음을 보인다: [relT3 POINT1](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/POINT1:2), [probe1 POINT1](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/POINT1:2).

SCRTEMP는 direct-access population history다. 소스상 배열 크기는 `NT×ND`, iteration pointer는 실제 iteration 번호이며, population vector가 매 iteration 기록된다: [scr_read_v2.f](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/scr_read_v2.f:80), [기록부](/gpfs/kjhan/cmfgen_src/cur_cmf/subs/scr_read_v2.f:400). `NT=1800`, `ND=90`이며 변수 1799, 1800은 각각 \(n_e,T\)다: [sum_steq_sol.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/sum_steq_sol.f:31).

사용 지표는 다음과 같다.

\[
E_{n_e,\max}=\max_d\left|n_{e,k}/n_{e,k-1}-1\right|
\]

\[
E_{n_e,\rm rms}=
\sqrt{\frac{\sum_d \rho_d\Delta V_d(\Delta n_e/n_e)^2}
{\sum_d\rho_d\Delta V_d}}
\]

\[
E_{\rm pop,L1}=
\frac{\sum_{Z,i,d} A_Z\Delta V_d|n_{Zi,k}-n_{Zi,k-1}|}
{\sum_{Z,i,d}A_Z\Delta V_dn_{Zi,k-1}}
\]

- `pop L1`은 modeled superlevel 1–1797만 사용했다. terminal closure 714·1126·1521·1798은 별도 분석했다.
- `dln-pop q99`는 \(|\ln(n_k/n_{k-1})|\)의 population-mass-weighted 99 percentile이다. 거의 비어 있는 trace level은 자동으로 거의 무게를 받지 않는다.
- `ion TV max`는 Si, S, Ca, Fe, Co, Ni 각각의 질량가중 이온분율에 대해
  \(\frac12\sum_q|F_{Zq,k}-F_{Zq,k-1}|\)를 구한 뒤 원소 간 최대값이다.
- \(T\)는 `FIX_T=T`이므로 변화가 0인 것이 수렴 증거는 아니다: [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/MODEL:308).

### 전 구간 결과

모든 값은 percent다.

| it | \(n_e\) max | \(n_e\) 질량 RMS | population L1 | dln-pop q99 | ion TV max | 질량가중 \(\langle q\rangle_{\rm S}\) |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 0.6492 | 0.0369 | 0.0328 | 0.5448 | 0.0571 | 2.037039 |
| 42 | 0.5781 | 0.0359 | 0.0352 | 0.6631 | 0.0675 | 2.036330 |
| 43 | 0.5798 | 0.0353 | 0.0353 | 0.6746 | 0.0735 | 2.035669 |
| 44 | 0.2946 | 0.0270 | 0.0317 | 0.6294 | 0.0771 | 2.034982 |
| 45 | 0.3203 | 0.0299 | 0.0347 | 0.6350 | 0.1031 | 2.034156 |
| 46 | 0.5567 | 0.0370 | 0.0376 | 0.7074 | 0.0848 | 2.033394 |
| 47 | 0.6879 | 0.0458 | 0.0439 | 0.8221 | 0.0907 | 2.032605 |
| 48 | 0.5792 | 0.0356 | 0.0416 | 0.8003 | 0.0909 | 2.031970 |
| 49 | 0.7251 | 0.0416 | 0.0394 | 0.6618 | 0.0776 | 2.031367 |
| 50 | 0.6847 | 0.0414 | 0.0450 | 0.7699 | 0.0952 | 2.030640 |
| **51 full** | **0.2217** | **0.0720** | **0.2424** | **2.0206** | **0.1089** | **2.030713** |
| 52 | 0.2561 | 0.0316 | 0.0385 | 0.9432 | 0.0915 | 2.031731 |
| 53 | 0.2767 | 0.0370 | 0.0427 | 1.1132 | 0.1074 | 2.032909 |
| 54 | 0.2934 | 0.0394 | 0.0426 | 1.2359 | 0.1160 | 2.034132 |
| 55 | 0.3155 | 0.0431 | 0.0453 | 1.3721 | 0.1264 | 2.035443 |

핵심 해석:

1. it41–50의 population L1은 0.032–0.045%, \(n_e\) 질량 RMS는 0.027–0.046%다. 즉 \(10^5\%\) correction에 대응하는 물질 발산은 없다.
2. 하지만 지표가 0으로 감쇠하지 않고 비슷한 폭으로 흔들린다. S 평균 전하는 2.0370→2.0306으로 한쪽 drift를 보인다.
3. full BA인 it51은 물질 상태를 실제로 움직였다. population L1은 직전 LAMBDA의 약 5.4배, q99는 2.02%다. 그래도 raw \(3.84\times10^7\%\)와는 약 7–8자릿수 괴리다.
4. it52–55에서 bulk L1은 다시 약 0.04%로 돌아오지만, q99와 ion TV는 다시 증가한다. 이는 “실수렴”보다는 작은 진동/재배치다.

### \(n_e(d)\) 직접값

단위는 \({\rm cm^{-3}}\)다. 최종 RVTJ의 \(n_e\)도 SCRTEMP it55와 일치한다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/RVTJ:52).

| it | d1 | d9 | d21 | d31 | d55 | d90 |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 2.00439e5 | 2.16881e5 | 1.58247e6 | 9.34041e6 | 8.34934e8 | 1.86864e10 |
| 50 | 1.99886e5 | 2.16341e5 | 1.58288e6 | 9.22062e6 | 8.34934e8 | 1.86863e10 |
| 51 | 2.00107e5 | 2.16033e5 | 1.58289e6 | 9.22405e6 | 8.34892e8 | 1.86863e10 |
| 55 | 2.00212e5 | 2.16204e5 | 1.58438e6 | 9.25673e6 | 8.34889e8 | 1.86705e10 |

깊은 층은 사실상 정지해 있고, 최대 변화는 외곽 저밀도 층이 지배한다.

### 질량가중 주요 이온분율

각 셀은 주된 이온들의 percent다.

| 원소 | 표시 순서 | it50 | it51 | it55 |
|---|---|---:|---:|---:|
| Si | III / IV | 96.5577 / 3.3953 | 96.5767 / 3.3767 | 96.5265 / 3.4254 |
| S | III / IV | 94.6476 / 4.1803 | 94.6622 / 4.1767 | 94.2525 / 4.6176 |
| Ca | III / IV | 99.9890 / 0.0110 | 99.9892 / 0.0108 | 99.9892 / 0.0107 |
| Fe | III / IV / V | 17.9929 / 74.8993 / 7.1075 | 18.0718 / 74.9104 / 7.0176 | 18.0732 / 74.9109 / 7.0156 |
| Co | III / IV / V | 23.5529 / 70.2929 / 6.1538 | 23.6504 / 70.2788 / 6.0705 | 23.6558 / 70.2745 / 6.0694 |
| Ni | III / IV / V | 27.4833 / 68.9243 / 3.5918 | 27.5921 / 68.8790 / 3.5283 | 27.5946 / 68.8777 / 3.5271 |

full step은 Fe/Co/Ni의 이온화 균형을 약 \(10^{-3}\) 절대분율만큼 실제 이동시켰고, 이후 그 이동은 대부분 유지된다. 반면 S는 full 전후 방향을 반전해 다시 higher-ion 쪽으로 움직인다. 이것이 약한 실체 진동의 가장 선명한 예다.

### 스펙트럼·복사량

저장된 it50와 it55 `OBSFLUX`의 동일 173,491-point observed-intensity grid를 직접 적분했다. 주파수 grid와 intensity block은 [relT3 OBSFLUX](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/OBSFLUX:3), [intensity](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/OBSFLUX:21694)에 있다.

| band | 적분 flux 비 \(F_{55}/F_{50}\) | \(\int|\Delta F|/\int F_{50}\) |
|---|---:|---:|
| 전체 | 1.03722 | 3.8036% |
| optical, 약 3500–9000 Å | 0.99599 | 0.5229% |
| UV, 약 1000–3500 Å | 1.03831 | 3.8533% |
| EUV, \(<1000\) Å | 1.01205 | 1.7558% |

다만 CMFGEN 내부 luminosity는 훨씬 크게 바뀐다.

- `Luminosity(d=1)`은 \(6.1930\times10^{10}\to1.3209\times10^{11}\), 2.133배다: [it50](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/OBSFLUX:39048), [it55](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OBSFLUX:39048).
- `Total Radiative Luminosity(d=1)`도 \(7.6166\times10^{10}\to1.6703\times10^{11}\), 2.193배다: [it50](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/OBSFLUX:39100), [it55](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OBSFLUX:39100).

따라서 synthetic observer-intensity shape는 수%만 바뀌지만 내부 radiative-flux solution은 다른 상태로 이동했다. 둘을 같은 “bolometric convergence” 지표로 취급하면 안 된다.

**질문 1 판정:** 거대 `%`는 잣대 아티팩트가 압도적으로 지배한다. 다만 물리 상태도 작은 진동·drift와 full-step 복사장 이동을 보이므로 “잣대만 진동하고 물리는 완전 수렴”은 아니다.

---

## 2. it51 \(+3.84\times10^7\%\) 변수와 BA 성격

### 변수 동정

it51의 실제 최소 raw correction은

\[
c_{713,d21}=-3.8393\times10^5,
\]

즉 “increase” 표기로 \(3.8393\times10^7\%\)다: [STEQ_VALS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/STEQ_VALS:37196). 소스상 correction은 음수일 때 population 증가 방향이다: [solveba_v13.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:155).

변수 713은 **Si III가 아니라 Ca V SL70**이다. 대표 상세 준위는 `3s2_3p3(2Do)8g_3Ho`이며 여러 \(n=8,9\) 고준위가 같은 superlevel에 묶여 있다: [LEVEL_SL_STEQ_LINKS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/LEVEL_SL_STEQ_LINKS:2650). 비교 대상 Si III 고준위 `3s10g3Ge`는 eq166이고, 이때 극값이 아니었다: [Si III 링크](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/LEVEL_SL_STEQ_LINKS:271).

it50, d21에서:

- \(n_{713}=5.3205\times10^{-41}\ {\rm cm^{-3}}\)
- \(n_{713}/n_e=3.36\times10^{-47}\)
- \(n_{713}/n_{\rm CaV}=7.34\times10^{-31}\)
- Ca V 자체도 전체 Ca의 \(1.76\times10^{-15}\)

따라서 \(3.84\times10^7\%\)는 물리적으로 유의한 population 이동량이 아니라, 거의 0인 분모에 대한 Newton fractional correction이다. 실제 적용은 +5% cap뿐이어서 \(n_{713}\)은 \(5.3205\to5.5865\times10^{-41}\)가 됐다.

### MAJOR scale과 상호작용

`SCALE_OPT=MAJOR`, `MAX_LIN=1.05`가 설정돼 있다: [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/MODEL:313).

MAJOR scale 계산에는

\[
n_j>10^{-10}n_e
\]

인 변수만 참여한다: [fiddle_pop_corrections_v2.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:155). 따라서 eq713은 공통 scale을 정하는 변수조차 아니다.

실제 minimum scale을 정한 것은 d23의 **Ca IV ground superlevel eq603**이다.

- d23 raw \(c_{603}=-299.04\)
- `MAX_LIN=1.05`에서 증가 한계는 \(LIT\_LIM=1-1.05=-0.05\)
- 따라서
  \[
  s_{\min}=0.05/299.04=1.6720\times10^{-4},
  \]
  로그값과 정확히 일치한다: [probe1 OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN:147).
- eq603은 Ca IV ground 묶음이다: [LEVEL_SL_STEQ_LINKS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/LEVEL_SL_STEQ_LINKS:1770).

d21에서도 eq603은 \(n=0.1376\), \(n/n_e=8.70\times10^{-8}\)라 코드의 “major” threshold는 통과하지만, 그 깊이 전체 Ca 중 Ca IV 분율은 \(3.35\times10^{-6}\)에 불과하다. 즉 코드 기준으로는 major지만 물리적 ion-stage 가중치는 trace다.

MAJOR는 이렇게 구한 scale을 모든 변수에 곱한 뒤 개별 cap을 적용한다: [fiddle_pop_corrections_v2.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fiddle_pop_corrections_v2.f:184). 결과적으로:

- 거대 Ca V 고준위 correction은 scale 계산에서 제외되지만,
- Ca IV trace ion의 ground가 depth 공통 scale을 정하고,
- 모든 변수는 최종 ±5% 안에 clip된다.

**성격 판정:** BA가 찾은 방향은 Ca IV→Ca V→Ca VI 고이온 꼬리 전체에 걸친 일관된 선형대수 방향이다. 완전히 임의의 단일 비트 오류는 아니다. 그러나 그 방향의 물리 가중치가 \(10^{-15}\) 이하이고, fractional variable scaling 때문에 수치적으로 증폭되므로 \(3.84\times10^7\%\) 자체는 조건화 아티팩트다. 이 자료만으로 실제 쌍안정 eigenmode라고 부를 근거는 없다.

---

## 3. full step 뒤 LAMBDA 제안 악화 기전

기준 it50 LAMBDA 제안은 \(5.13\times10^3\%\)였다: [relT3 OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3/OUTGEN:272).

full BA 이후:

- it52: \(3.13\times10^5\%\), it50 대비 61.0배
- it53: \(2.08\times10^5\%\), 40.5배
- it54: \(2.82\times10^5\%\), 55.0배
- it55: \(7.48\times10^5\%\), 145.8배

로그는 각각 [it52](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN:169), [it53](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN:201), [it54](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN:238), [it55](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN:277)에 있다. 따라서 “100×”는 정확히는 40–146배 범위다.

### 상태가 실제 이동했는가?

그렇다. it50→51에서:

- 적용된 개별 비율 범위: \(0.952381\)–\(1.05\)
- population L1: 0.2424%
- dominant-population q99: 2.0206%
- ion TV max: 0.1089%
- Fe/Co/Ni 평균 전하: 각각 약 0.00169/0.00181/0.00172 감소

즉 full step이 상태를 실제 이동시켰다. 다만 bulk 이동은 0.2%대이지 수백만 percent가 아니다.

더 중요한 변화는 다음 iteration에서 계산된 복사장이다. it51 luminosity는 correction 적용 전 상태에 대한 평가이고, it52 luminosity가 post-full SCRTEMP51 상태를 처음 평가한 값이다.

\[
\frac{1.33296303\times10^{11}}
     {6.64087280\times10^{10}}
=2.0072,
\]

즉 **+100.72%**다: [it51](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN:137), [it52](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN:169). 이후 it55까지 \(1.32\times10^{11}\) 근방에 남는다.

### 제안잣대만 악화했는가?

부분적으로 그렇다.

- it52 최대 increase는 Co IV SL50 고준위 eq1430, d31이다.
- it54–55는 다시 Ca V trace 고준위 eq705, d21이 지배한다.
- 즉 \(2\)–\(7.5\times10^5\%\) 대부분은 여전히 trace-family fractional norm이다.
- 그러나 outer luminosity의 2배 이동과 iron-group ion fraction의 영구적 소폭 이동은 실제다.
- it51과 it54에는 MOM solver excessive-iteration 메시지도 있다: [it51](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN:129), [it54](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/OUTGEN:230).

따라서 기전은 **“±5% full BA가 작은 물질 상태 변화를 만들었고, 복사장은 그 변화에 매우 민감하게 다른 수치/물리 branch로 이동했으며, 그 뒤의 raw 악화량은 다시 trace-level 잣대가 과장했다”**이다.

---

## 4. terminal \(100\)–\(110\%\) 감소 행의 실체

최종 it55의 decrease 극값은 d27의 **eq1798 = NkSEV SL1**이며 \(c=1.0978\)이다: [CORRECTION_LINK](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/CORRECTION_LINK:20).

같은 깊이 상위 네 행은:

- NkSEV eq1798: 1.0978
- CaSIX eq714: 1.0967
- CoSEV eq1521: 1.0937
- FeSEV eq1126: 1.0871

즉 Ni 하나의 물리 불안정이 아니라 **각 원소의 마지막 modeled ion 다음에 붙는 closure-variable 가족**이다.

CMFGEN은 다음 이온이 명시적으로 없을 때 `DI`를 그 다음 이온의 ground state로 해석해 population conservation과 charge conservation에 넣는다: [steq_multi_v10.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/steq_multi_v10.f:300). 출력 label도 각 이온 block의 마지막 변수를 다음 이온 SL1로 매핑한다: [sum_steq_sol.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/sum_steq_sol.f:37).

따라서 이 행은 제거 가능한 장식이 아니라:

- 원소 총 population conservation
- 전자 charge conservation
- 미모델링 다음 이온의 ground population

을 닫는 실제 수치 변수다.

### 왜 100–110%에 고착되는가?

raw decrease \(c>1\)이면 Newton 제안은 음의 새 population을 뜻한다. `solveba`는 \(c\ge0.99999\)를 반환할 때 물리적 크기 대신 `MAXCH=10^7%` sentinel을 준다: [solveba_v13.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:201).

LAMBDA `LIMIT`는 \(c>1.1\)일 때만 0.999로 교체한다: [solveba_v13.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:129). 따라서 1.087–1.098은:

- 음의 population을 제안할 만큼 크지만,
- `LIMIT`가 작동하는 1.1보다 작아서,
- 매번 `MAXCH=10^7`을 만드는 사각지대다.

실제 population은 전혀 100%씩 변하지 않는다. d27의 it54→55 값은:

| closure | it54 population | it55 population | 실제 비율 | 원소 내 분율(it54) |
|---|---:|---:|---:|---:|
| CaSIX 714 | 1.6011e-31 | 1.5508e-31 | 0.96861 | 1.12e-36 |
| FeSEV 1126 | 1.0375e-54 | 1.0052e-54 | 0.96889 | 1.01e-50 |
| CoSEV 1521 | 2.4798e-55 | 2.4022e-55 | 0.96870 | 2.42e-51 |
| NkSEV 1798 | 2.6411e-59 | 2.5582e-59 | 0.96858 | 2.57e-55 |

즉 실제 적용은 약 −3.1%이고 물리적 질량은 사실상 0이다.

### 제거·재정식화 옵션

1. **행을 그냥 삭제하면 안 된다.** 원소수·전하 보존 closure를 훼손한다.

2. 가장 직접적인 CMFGEN 내 진단 옵션은 원소별 highest-stage fix다.

   - `FIX_CAL`
   - `FIX_IRON`
   - `FIX_COB`
   - `FIX_NICK`

   이 키워드는 현재 모두 0이다: [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT3_probe1/MODEL:276). 소스는 해당 species closure equation을 identity row로 만들고 correction을 0으로 둔다: [fixpop_in_ba_v3.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/fixpop_in_ba_v3.f:115).

   `FIX_NkSEV` 같은 level 옵션과 `FIX_NICK`은 다르다. terminal DI를 확실히 겨냥하는 것은 species-level highest-stage 옵션인 `FIX_NICK`이다.

3. 다음 고이온을 명시적으로 atom model에 포함하는 것도 가능하다. 그러면 현재 closure population은 실제 modeled ground가 되지만 closure가 한 stage 위로 이동할 뿐, trace conditioning 자체가 자동으로 사라지는 것은 아니다.

4. `LAM_SCALE_OPT=LIMIT`만으로는 해결되지 않는다. 현재 문제가 정확히 1.1 아래에 있기 때문이다.

5. 보고/종료 잣대는 terminal closure 및 \(n_i/n_e\ll10^{-10}\) 변수를 제외한 population-weighted norm으로 별도 판단해야 한다. 이는 방정식을 바꾸지 않고 잘못된 종료판정을 막는 가장 안전한 조치다.

---

## 5. 최종 판정과 다음 프로브 1개

### 최종 판정

- **전역 물질 발산:** 부정. \(n_e\), 질량가중 이온분율, dominant population은 \(10^{-4}\)–\(10^{-3}\) 수준에서 움직인다.
- **완전한 실수렴:** 부정. it41–50의 물리 지표가 감쇠하지 않고, S ionization drift가 있으며, it52–55의 q99와 ion TV가 다시 증가한다.
- **거대 MAXCH의 물리성:** 대부분 부정. Ca V 고준위와 terminal closure의 거의 0인 population이 지배한다.
- **full BA의 물리 영향:** 긍정. bulk population은 0.24% 이동했고 radiative luminosity는 다음 평가에서 2.007배가 됐다.
- **쌍안정성:** 미확정. 두 안정 branch의 재현성과 되돌림 이력이 없다.

따라서 **“trace/closure 잣대 아티팩트가 지배하는 가운데, 작은 실체 진동과 강한 복사장 branch 민감도가 공존하는 혼합 실패”**로 판정한다.

### 사전등록된 단일 후속 프로브

**목적:** it51 radiative jump가 smooth한 full-step 응답인지, 작은 상태 변화에도 발생하는 불연속 branch/solver 전이인지 한 인자만 바꿔 판별한다.

**설계:**

- exact it50 checkpoint에서 시작
- `FIX_T=T`, `SCALE_OPT=MAJOR`, BA 및 atom set 모두 probe1과 동일
- closure `FIX_*`도 이번에는 변경하지 않음
- **유일한 개입:** `MAX_LIN=1.05 → 1.01`
- 1회 full BA를 적용한 뒤, 이어서 1회 강제 LAMBDA 평가를 수행해 post-step luminosity와 physical norm을 측정
- 신규 판단 지표는 `population L1`, `dln-pop q99`, ion TV, \(n_e\) RMS, `Total Radiative Luminosity`, observed-spectrum L1로 사전 고정

1.05에서 1.01로 바꾸면 허용 증가폭은 정확히 0.2배, 감소 correction 한계는

\[
\frac{(1.01-1)/1.01}{(1.05-1)/1.05}=0.2079
\]

배가 된다. smooth response라면:

- post-full population L1 기대값: 약 \(0.2424\%\times0.2=0.0485\%\)
- q99 기대값: 약 \(2.0206\%\times0.2=0.40\%\)
- 다음 radiative evaluation의 luminosity 증가는 약 \(100.72\%\times0.2\approx20\%\)
- \(6.6409\times10^{10}\) 기준 예상값은 약 \(8.0\times10^{10}\)

**사전등록 판정선:**

- 다음 luminosity가 \(8.64\times10^{10}\) 이하, population L1 \(<0.08\%\)이면: smooth response. 기존 5% step이 복사장 민감도를 과도하게 자극했고, 거대 raw MAXCH는 여전히 잣대 아티팩트라는 쪽.
- population L1 \(<0.08\%\)인데도 luminosity가 \(1.0\times10^{11}\) 이상으로 다시 점프하면: 작은 상태 변화에 대한 radiative branch/solver 불연속 증거.
- population L1 자체가 \(0.08\%\) 이상이거나 luminosity가 두 기준 사이면: 혼합/비선형 결과로 판정하고 쌍안정성 주장은 보류.

이 프로브는 `SCALE_OPT`, closure fixing, atom model을 동시에 바꾸지 않으므로 it51 점프의 step-size 의존성을 가장 직접적으로 검정한다.