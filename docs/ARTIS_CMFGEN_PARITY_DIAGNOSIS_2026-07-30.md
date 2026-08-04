# ARTIS/CMFGEN parity 실패 진단

작성일: 2026-07-30  
대상 저장소: Lumina-sn  
분석 기준 실행: `logs/coevolve_consume_parity54` 및 루트의 대응 co-evolve 산출물  
ARTIS 기준 소스: `/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref`, commit `36f86476d870cec55bcbe9ab80c1b24ada692eb4`

## 결론

현재 parity 실패는 하나의 계수나 수렴 파라미터를 조정해서 해결할 문제가 아니다. 두 종류의 문제가 겹쳐 있다.

1. **비교 기준선 일부가 잘못되어 있다.** 특히 departure coefficient 비교가 19.48일의 Lumina 결과를 ARTIS의 약 5일 또는 11.24일 결과와 비교했다. 일부 비교는 ARTIS의 Monte Carlo emergent spectrum과 Lumina의 deterministic formal spectrum을 같은 관측량으로 취급한다. 따라서 기존 parity 방향의 일부는 유효하지 않다.
2. **그 기준선 오류를 제거해도 실제 물리 차이는 매우 크다.** 최신 실행은 ARTIS에 비해 radiation field가 셀·파장별로 크게 다르고, Si/S/Fe/Co/Ni가 심하게 저이온화되어 있으며, 2500–3000 Å flux가 과도하게 탈출한다. ARTIS의 element-wide NLTE rate matrix와 level-resolved bound-free/macro-atom 처리도 Lumina의 현재 구현과 구조적으로 다르다.

가장 먼저 해야 할 일은 새로운 보정 계수 추가가 아니라, **동일 timestep·동일 상태·동일 관측량을 강제하는 작은 oracle test**를 만드는 것이다. 그 다음 수정 순서는 대체로 다음과 같다.

1. 비교 도구와 provenance 정정
2. radiation-field 소비 경로를 하나로 통일
3. ARTIS와 같은 element-wide ionization/excitation matrix 검증
4. level-resolved bound-free 및 ion-changing macro-atom 검증
5. 마지막에 전체 transport와 emergent spectrum 비교

현 단계에서 “CMFGEN과 다르므로 특정 departure coefficient를 더 키운다”와 같은 조정은 원인 규명보다 오차 상쇄에 가깝다.

## 1. 분석 범위와 주의점

### 사용한 로컬 자료

- Lumina 최신 비교 실행: `logs/coevolve_consume_parity54`
- Lumina plasma state: `logs/coevolve_consume_parity54/plasma_state.csv`
- Lumina formal spectrum: `logs/coevolve_consume_parity54/lumina_spectrum_formal.csv`
- Lumina wavelength-dependent field: 루트의 `lumina_coevolve_field.csv`
  - 실행 디렉터리에는 이 파일이 보존되지 않았지만 수정시각과 실행 계보가 parity54에 대응한다.
  - 이후에는 반드시 실행 디렉터리 안에 복사하고 checksum을 manifest에 기록해야 한다.
- ARTIS 테스트:
  - `artis-ref/tests/toy06_nlte_bk`
  - `artis-ref/tests/toy06_whitebox_run`
- ARTIS 시간 좌표: `timesteps.out`
- ARTIS plasma/field/population 산출물:
  - `estimators_0000.out`
  - `radfield_0000.out`
  - `ion_frac_0000.out`
  - `binned_j_nu_0000.out`
  - `b_k_0000.out`

### 해석상의 제한

ARTIS 테스트는 time-dependent radioactive-deposition Monte Carlo 계산이고, 현재 Lumina 실행은 inner-boundary/lamp, deposition, deterministic CMF 및 Monte Carlo estimator가 혼합된 경로를 사용한다. 따라서 두 실행의 radiation-field **절대값** 차이를 바로 특정 microphysics 버그로 단정할 수는 없다.

그러나 다음 차이는 너무 크고 구조적이어서 단순한 luminosity normalization 차이로 설명되지 않는다.

- wavelength와 shell에 따라 field ratio가 수십~백 배로 변함
- ARTIS의 높은 IV-stage population이 Lumina에서 거의 사라짐
- Lumina의 `T_rad`가 넓은 shell 구간에서 동일한 값으로 고정됨
- emergent shape가 2500–3000 Å에 집중되고 optical redistribution이 사라짐

정확한 방법 parity 판정은 뒤에서 제안하는 frozen-cell 및 frozen-state 실험으로 해야 한다.

## 2. P0: 기존 ARTIS 비교 기준선에 timestep 오류가 있다

### 실제 ARTIS 시간 좌표

`artis-ref/tests/toy06_nlte_bk/timesteps.out`를 직접 읽으면 다음과 같다.

| ARTIS timestep | 시작일 | 중간일 | 의미 |
|---:|---:|---:|---|
| 20 | 10.7722 d | 11.2353 d | 19.48 d가 아님 |
| 26 | - | 18.6195 d | 목표보다 이른 epoch |
| 27 | 19.4200 d | 20.2549 d | 19.48 d를 포함하는 bin |

따라서 19.48일 Lumina/CMFGEN 결과와 비교할 ARTIS population/field timestep은 일반적으로 **27**이어야 한다. 시간 보간을 하지 않는다면 “ARTIS timestep 27이 19.48일을 포함한다”고 명시해야 한다.

### 잘못된 기존 사용

- `docs/ARTIS_PARITY_GAP_AUDIT.md:5`
  - timestep 20을 19.4945일이라고 기록한다.
- `scripts/artis_baseline_bk.py:6`
  - 기본값이 `TS=20`이고 이를 19.49일 기준선처럼 출력한다.
- `scripts/compare_bk_artis.py`
  - 동일 epoch를 선택하지 않는다.
  - ARTIS 전체 timestep 중 저준위 S II population이 큰 시점을 찾아 자동 선택한다.
  - 실제로는 약 5일 부근의 초기 상태를 19.48일 Lumina와 비교할 수 있다.

이 방식은 “같은 물리 조건에서 방법을 비교”하는 parity가 아니라 “한쪽 결과와 비슷해 보이는 다른 시점을 찾는” 비교다. 이 스크립트로 얻은 큰 ARTIS `b_k`, 특히 초기 Si II/S II의 매우 큰 값은 19.48일 목표값으로 사용하면 안 된다.

추가로 이 스크립트 계열에는 다음 표시 문제도 있다.

- 실제 필터와 출력 레벨 범위가 일치하지 않는다.
- mean/median 표기가 구현과 일치하지 않는 부분이 있다.
- 일부 원소 이름과 ion-stage 출력이 일반화되어 있지 않다.

### 영향

이 오류는 단순한 라벨 문제가 아니다. 지금까지의 population tuning 방향을 바꿀 수 있다.

- 올바른 timestep 27에서 Si II와 S II의 여러 excited level은 초기 epoch만큼 큰 superthermal population을 보이지 않는다.
- 따라서 “ARTIS parity를 위해 해당 `b_k`를 수십~수백으로 끌어올려야 한다”는 결론은 성립하지 않는다.
- 기존 `b_k` 기반 변경은 timestep 27로 다시 검증하기 전까지 parity 증거로 간주하면 안 된다.

## 3. P0: 서로 다른 스펙트럼 산출물을 직접 비교하고 있다

`scripts/cmp3_artis.py`가 사용하는 주요 Lumina 산출물은 `lumina_spectrum_formal.csv`다. 이것은 deterministic formal solution이다. 반면 ARTIS `spec.out`은 Monte Carlo packet의 emergent spectrum이다.

두 결과는 같은 atomic state를 사용하더라도 다음 이유로 달라질 수 있다.

- packet history와 formal ray integration의 estimator가 다름
- macro-atom 재분배가 반영되는 위치가 다름
- source function과 opacity의 snapshot 시점이 다를 수 있음
- time-bin averaging 방식이 다름
- boundary/deposition power source가 다름

따라서 비교 lane을 분리해야 한다.

| 목적 | 올바른 비교 |
|---|---|
| ARTIS transport parity | ARTIS MC emergent spectrum ↔ Lumina MC emergent spectrum |
| CMFGEN formal-solver parity | CMFGEN formal spectrum ↔ Lumina formal spectrum |
| atomic-state parity | 같은 cell의 rate, opacity, emissivity, population을 직접 비교 |

현재의 ARTIS ↔ Lumina formal 비교는 전체 결과의 이상을 발견하는 smoke test로는 쓸 수 있지만, 어느 transport 모듈이 틀렸는지를 판정하는 oracle로는 부족하다.

## 4. 최신 실행에서 확인한 실제 스펙트럼 실패

`python3 scripts/cmp3_artis.py logs/coevolve_consume_parity54`의 shape-normalized 결과는 다음과 같다.

### Peak 및 상관

| 결과 | Peak wavelength |
|---|---:|
| Lumina | 2830 Å |
| CMFGEN | 3922 Å |
| ARTIS | 4102 Å |

| 비교 | Pearson correlation |
|---|---:|
| Lumina–CMFGEN | 0.346 |
| Lumina–ARTIS | 0.069 |
| CMFGEN–ARTIS | 0.849 |

CMFGEN과 ARTIS는 세부 물리 차이에도 불구하고 전체 optical shape는 서로 훨씬 가깝다. 최신 Lumina 결과는 두 코드와 모두 다른 별도의 실패 모드에 있다.

### 정규화된 파장대별 flux

| 파장대 | Lumina | CMFGEN | ARTIS |
|---|---:|---:|---:|
| 2500–3000 Å | 42.7% | 8.5% | 5.5% |
| 3000–3500 Å | 18.0% | 15.0% | 9.2% |
| 3500–5000 Å | 28.7% | 41.6% | 37.2% |
| 5000–6500 Å | 3.2% | 22.7% | 29.2% |
| 6500–9000 Å | 5.5% | 9.7% | 15.9% |

더 좁게 보면 Lumina는 2500–2750 Å에 21.4%, 2750–3000 Å에 20.8%를 내보낸다. CMFGEN과 ARTIS의 대응 값은 각각 약 2.2/6.1%, 1.7/3.7%다.

즉 현재 문제를 “너무 붉다” 또는 “line blanketing이 너무 강하다”로 설명하면 반대 방향이다. 최신 실행의 직접적인 증상은 다음이다.

> UV packet/energy가 optical로 충분히 재분배되지 않고 2500–3000 Å로 과도하게 탈출한다.

shape normalization만 사용했으므로 이 표는 bolometric energy conservation을 검증하지 않는다. 후속 실험에서는 절대 luminosity와 packet-energy census도 함께 기록해야 한다.

## 5. C2/C5: radiation field와 thermal state가 이미 크게 다르다

ARTIS timestep 27의 scalar estimator와 parity54 Lumina plasma state를 비교했다.

### ARTIS 상태 예시

| Shell | `T_R` [K] | `T_e` [K] | `W` | `n_e` [cm⁻³] |
|---:|---:|---:|---:|---:|
| 8 | 18291 | 16397 | 0.07345 | 9.363e8 |
| 10 | 15323 | 12718 | 0.05807 | 4.873e8 |
| 12 | 13709 | 11610 | 0.04737 | 2.893e8 |
| 15 | 13323 | 11427 | 0.03234 | 1.375e8 |
| 20 | 12838 | 11914 | 0.02034 | 4.035e7 |
| 25 | 12395 | 12700 | 0.01473 | 1.242e7 |
| 30 | 11895 | 14353 | 0.01208 | 3.897e6 |

### Lumina와의 차이

- Lumina의 `W`는 대체로 ARTIS의 약 0.53–0.63배다.
- Lumina의 `T_rad`는 위 shell 범위에서 거의 모두 **10470.093 K**로 동일하다.
- Lumina의 `T_e / T_e,ARTIS`는 대략 다음과 같다.
  - shell 8: 0.73
  - shell 10: 0.91
  - shell 12: 0.90
  - shell 15: 0.82
  - shell 20: 0.74
  - shell 25: 0.67
  - shell 30: 0.58
- 반면 `n_e`는 많은 shell에서 ARTIS의 약 0.8–1.0배로 비교적 가깝다.

특히 `T_rad`가 shell과 무관하게 같은 값으로 유지되는 것은 ARTIS의 per-bin field fit과 동등한 결과가 plasma consumer에 전달되고 있지 않다는 강한 신호다.

### wavelength-dependent field

`lumina_coevolve_field.csv`의 Monte Carlo field를 ARTIS timestep 27의 binned field와 비교한 비율 `J_Lumina/J_ARTIS` 예시는 다음과 같다.

| Shell | 1100 Å | 1200 Å | 1300 Å | 1500 Å | 2000 Å | 3000 Å | 5000 Å |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.48 | 0.51 | 0.44 | 0.34 | 0.48 | 1.32 | 1.20 |
| 10 | 5.66 | 2.53 | 3.61 | 2.22 | 1.84 | 2.23 | 1.25 |
| 12 | 9.10 | 4.40 | 1.89 | 0.35 | 0.61 | 0.51 | 0.31 |
| 20 | 110.42 | 96.44 | 12.99 | 0.99 | 0.24 | 0.52 | 0.75 |
| 25 | 139.53 | 29.33 | 16.14 | 0.80 | 1.10 | 0.24 | 0.39 |
| 30 | 50.82 | 30.04 | 21.94 | 1.92 | 1.34 | 0.19 | 0.23 |

1000 Å 부근은 ARTIS bin이 매우 넓어 직접 해석하지 않았다. 위의 큰 비율도 서로 다른 power source의 영향이 포함되므로 그 자체가 구현 오류의 크기는 아니다. 그러나 shell과 wavelength에 따라 ratio가 0.2에서 100 이상으로 바뀌는 형태는 단일 dilution/luminosity 보정으로 맞출 수 없다.

현재 코드에는 여러 radiation-field 표현이 동시에 존재한다.

- continuum-solver `J`
- raw Monte Carlo `J`
- EMA/damped `J`
- scalar `W`, `T_rad`
- line-rate용 `Jbar`
- blue/photoionization 전용 consumer

`src/lumina_cuda.cu:9331-9351`의 binned-J download/fit도 gate에 의존한다. 최신 resolved environment에서는 이 경로가 ARTIS식 field를 모든 plasma consumer에 일관되게 제공했다고 확인되지 않는다. parity54에서 특정 Jbar generation/consumption 설정을 동시에 켠 뒤 iteration 9부터 결과가 갈라지는 현상도 같은 field의 세대와 consumer가 나뉘어 있음을 시사한다.

따라서 C2의 핵심 질문은 “어떤 `J`가 더 그럴듯한가”가 아니라 다음이어야 한다.

> 각 rate, heating/cooling, source function이 정확히 어느 iteration의 어느 field 배열을 읽는가?

이 mapping을 하나의 표와 runtime provenance로 출력하지 않는 한 field parity는 검증되지 않는다.

## 6. C1: Lumina는 ARTIS보다 심하게 저이온화되어 있다

ARTIS timestep 27과 Lumina parity54의 동일 shell ion fractions를 비교했다. ARTIS stage 표기는 1-based, Lumina 내부 stage 표기는 0-based이므로 원소의 물리적 spectroscopic stage로 맞춰 비교했다.

### Shell 8 예시

| 원소 | ARTIS 주요 분포 | Lumina 주요 분포 |
|---|---|---|
| Si | Si III 0.152, Si IV 0.808 | Si III 0.985, Si IV 0.015 |
| S | S III 0.132, S IV 0.849 | S III 0.967, S IV 0.010 |
| Fe | Fe III 0.016, Fe IV 0.970 | Fe III 0.948, Fe IV 0.052 |
| Co | Co III 0.029, Co IV 0.969 | Co III 0.977, Co IV 0.023 |
| Ni | Ni III 0.069, Ni IV 0.926 | Ni III 0.966, Ni IV 0.034 |

### 추가 예시

- shell 10 Fe:
  - ARTIS Fe III/IV = 0.633/0.361
  - Lumina Fe III/IV = 0.933/0.067
- shell 10 S:
  - ARTIS S II/III/IV = 0.00017/0.659/0.334
  - Lumina S II/III/IV = 0.0205/0.973/0.0067
- shell 12 S:
  - ARTIS S II/III/IV = 0.0017/0.773/0.222
  - Lumina S II/III/IV = 0.498/0.498/0.0036
- shell 20 S:
  - ARTIS S II/III/IV = 0.044/0.619/0.325
  - Lumina S II/III/IV = 0.683/0.311/0.0058
- shell 25 S:
  - ARTIS S II/III/IV = 0.032/0.450/0.483
  - Lumina S II/III/IV = 0.784/0.206/0.0094

이는 작은 population correction 문제가 아니다. ARTIS에서 우세한 IV stage가 Lumina에서 거의 사라진다.

ARTIS-NLTE 결과가 더 높은 ionization을 보이는 주된 원인으로 bluer radiation field와 photoionization의 중요성이 보고되어 있다. 참고: [ARTIS 공식 저장소](https://github.com/artis-mcrt/artis), [ARTIS-NLTE 방법 및 CMFGEN 비교 논문](https://academic.oup.com/mnras/article/538/3/1289/8071984).

다만 여기에는 중요한 관찰이 하나 더 있다.

- Lumina는 ARTIS보다 저이온화되어 있다.
- 그런데 Lumina의 emergent spectrum은 ARTIS/CMFGEN보다 훨씬 많은 UV를 방출한다.

저이온화만으로 이 UV leakage를 설명하기 어렵다. 즉 최소 두 문제가 병렬로 존재할 가능성이 높다.

1. field/rate matrix 문제로 인한 잘못된 ionization state
2. opacity 적용, fluorescence/macro-atom 재분배 또는 formal-output 경로의 문제

ion fraction 하나를 맞추면 spectrum도 자동으로 맞을 것이라고 가정하면 안 된다.

## 7. C1/C3: NLTE matrix의 구조가 ARTIS와 다르다

### ARTIS

로컬 ARTIS의 `nltepop.cc:1200-1289`는 한 원소의 모든 ion stage와 level을 하나의 statistical-equilibrium matrix에 넣고 푼다.

- 모든 관련 ion stage가 같은 matrix에 들어감
- bound-bound 및 bound-free rate가 같은 system에 연결됨
- 한 개의 element-total normalization을 사용함
- 테스트 설정의 `FORCE_SAHA_ION_BALANCE=false`
- 따라서 각 adjacent pair의 총량을 따로 고정하지 않음

photoionization/recombination은 `nltepop.cc:563-619`에서 level-resolved photoionization target과 thermal collisional ionization, radiative recombination, three-body/collisional recombination을 포함한다.

### Lumina

현재 Lumina의 중심 경로는 `src/lumina_plasma.c:7242-7277`, `src/lumina_plasma.c:15391-15397`, `src/lumina_plasma.c:15933-16043`에서 확인할 수 있다.

- adjacent ion pair `(lo, hi=lo+1)` 단위로 matrix를 구성함
- 각 pair solve에서 pair-total conservation row를 삽입함
- 여러 pair를 순차적으로 반복해 일부 overlap을 보정함
- parity mode는 반복 횟수와 damping을 강화하지만 system 자체를 element-wide matrix로 바꾸지 않음

`LUMINA_NLTE_NO_ML_LOCK`를 켜면 normalization에 전체 element density를 사용할 수 있지만, 각 pair에 독립적으로 conservation constraint를 넣는 구조는 그대로다. 이는 ARTIS의 단일 element-wide normalization과 수학적으로 같지 않다.

예를 들어 `II↔III`와 `III↔IV`를 순차적으로 풀면 공유 stage III가 앞 solve의 해와 뒤 solve의 해에서 서로 다른 역할을 한다. outer iteration으로 수렴시키더라도, 모든 cross-stage coupling과 normalization을 한 번에 푼 ARTIS matrix와 같은 해를 보장하지 않는다.

### 판정

이 차이는 구현 세부가 아니라 **방정식의 구조 차이**다. 다음을 직접 비교하기 전에는 C1/C3 parity가 성립했다고 볼 수 없다.

- 동일 cell의 matrix dimension
- 각 row/column의 level identity
- bound-free target mapping
- 각 rate coefficient와 단위
- conservation row
- matrix residual
- 최종 stage population과 level population

## 8. C6: bound-free/macro-atom 구현은 아직 ARTIS와 동등하지 않다

Lumina에는 최근 bound-free 및 radiative-recombination 관련 경로가 추가되어 있지만, 현재 상태는 ARTIS의 level-resolved topology와 다르다.

### 확인한 구조 차이

- `src/lumina_plasma.c:6316-6324`
  - 여러 source level이 실제로는 동일한 upper-ion ground state로 매핑되는 제한이 주석에 명시되어 있다.
- `src/lumina_cuda.cu:4403-4422`
  - source마다 upper-ion ground로 가는 단일 up-jump 형태다.
- ARTIS `nltepop.cc:581-605`
  - photoionization cross-section target을 이용해 여러 upper target level로 연결한다.
- ARTIS `macroatom.cc`
  - ion-changing macro-atom action과 k-packet collisional-ionization 경로를 포함한다.

따라서 Lumina의 현재 `LUMINA_MA_RADRECOMB=1`은 “기능이 켜졌다”는 뜻이지 ARTIS method parity가 증명되었다는 뜻이 아니다.

### 현재 설정의 혼합성

parity54 설정은 다음 종류의 경로를 함께 사용한다.

- `LUMINA_PURE_CMFGEN=1`
- Monte Carlo field estimator/Jbar 관련 gate
- NLTE collisional/radiative recombination 보정
- macro-atom radiative recombination
- `LUMINA_LINE_THERM=1`
- deterministic formal spectrum

특히 `LUMINA_LINE_THERM=1`은 ARTIS-faithful macro-atom 비교에서는 분리해야 한다. `src/lumina_cuda.cu:4907-4919`에서 cascade cap으로 unresolved된 packet을 thermal re-emission으로 넘길 수 있기 때문이다. 이것은 결과를 안정화하는 별도 closure일 수 있지만 ARTIS의 packet fate와 같은지 검증되지 않았다.

현재는 “ARTIS 방법”, “CMFGEN 방식의 deterministic closure”, “Lumina 전용 fallback”이 하나의 실행에 섞여 있다. 이 상태에서는 최종 spectrum 차이를 어느 방법의 실패라고 부를 수 없다.

## 9. C0–C7 현재 판정

| 단계 | 대상 | 판정 | 근거 |
|---|---|---|---|
| C0 | 입력 모델·epoch·atomic mapping | **부분 실패** | model/composition은 유사하지만 기존 ARTIS epoch 선택이 틀렸고 atomic target mapping 동등성이 미검증 |
| C1 | ionization balance | **실패** | Si/S/Fe/Co/Ni IV fraction이 order-of-magnitude 이상 다름; pairwise matrix 구조 차이 |
| C2 | radiation field/photoionization input | **최초의 강한 실패 지점** | `T_rad` 고정, shell/파장별 `J` ratio 큰 변동, field consumer 분열 |
| C3 | level population/departure coefficient | **기존 판정 무효, 구조적 실패 가능성 큼** | 잘못된 timestep comparator; element-wide SE가 아님 |
| C4 | opacity/emissivity cell oracle | **미검증** | 동일 state를 강제한 monochromatic opacity/source 비교가 없음 |
| C5 | thermal balance | **실패** | 외곽 `T_e`가 ARTIS보다 20–40% 이상 낮음 |
| C6 | redistribution/macro-atom/packet fate | **실패 또는 미분리** | UV leakage, level-resolved ion-changing topology 부족, thermal fallback 혼재 |
| C7 | emergent spectrum | **실패** | peak, band flux, correlation 모두 ARTIS/CMFGEN과 크게 다름 |

C2를 “최초의 강한 실패 지점”으로 적은 이유는 transport가 만든 field가 plasma state에 들어가는 시점에서 이미 큰 차이가 보이기 때문이다. 다만 C0의 power-source/time-averaging까지 완전히 동일하게 하지 않았으므로, 엄밀한 최초 divergence는 frozen-cell 실험에서 확정해야 한다.

## 10. 가장 가능성 높은 인과 구조

현재 증거로는 단일 선형 원인보다 다음 두 갈래가 합류하는 형태가 더 타당하다.

```text
field 생성/세대/consumer 불일치
              │
              ├─> photoionization rate 오류
              ├─> element ionization 저하
              ├─> departure coefficient 오류
              └─> thermal balance 오류

pairwise NLTE matrix ───────────────┘

level-resolved bf/MA 부족
thermal fallback 혼재
opacity/source/formal 경로 불일치
              │
              └─> UV fluorescence/redistribution 부족
                          │
                          └─> 2500–3000 Å 과잉, optical flux 부족
```

두 갈래는 서로 영향을 주지만 독립적으로 검증해야 한다. 현재 ionization이 낮다는 사실과 UV가 과잉이라는 사실을 하나의 보정 계수로 동시에 고치려고 하면 잘못된 cancellation을 만들 가능성이 높다.

## 11. 권장 parity 작업 순서

### Gate 0 — benchmark 무결성부터 고정

하나의 canonical manifest를 만들어 다음을 모두 기록한다.

- ARTIS git commit
- Lumina git commit 및 dirty diff hash
- model/composition/options/input checksum
- ARTIS timestep index, start/mid/end day
- Lumina epoch
- luminosity/deposition/boundary normalization
- spectrum 종류: MC packet 또는 formal solution
- field 종류: raw MC, EMA, continuum, scalar fit
- 모든 parity 관련 environment variable
- 각 output file checksum

비교 스크립트는 `timesteps.out`에서 시간을 읽어야 하며, population 크기를 보고 timestep을 자동 선택해서는 안 된다. 19.48일 비교의 기본 ARTIS timestep은 27로 정정한다.

### Gate 1 — frozen-cell atomic oracle

shell 8, 10, 12 세 개만 선택한다. ARTIS에서 다음 상태를 추출해 Lumina에 그대로 주입한다.

- `T_e`
- `n_e`
- `J_ν` 또는 ARTIS의 per-bin radiation field
- density/composition
- atomic levels 및 photoionization target

각 코드에서 spectrum을 만들지 말고 다음을 행 단위로 출력한다.

- bound-bound upward/downward rate
- level별 photoionization rate `Γ_bf`
- radiative recombination
- collisional ionization
- three-body/collisional recombination
- matrix coefficient
- normalization row
- matrix residual
- final level/stage populations

첫 번째 불일치 rate가 실제 microphysics parity의 출발점이다. 이 test가 통과하기 전에는 전체 transport 결과로 population을 튜닝하지 않는다.

### Gate 2 — element-wide matrix

Lumina의 adjacent-pair solve를 유지한 채 damping만 조정하는 실험이 아니라, 적어도 한 원소와 한 cell에 대해 ARTIS와 같은 element-wide matrix를 구성한다.

권장 최소 대상:

1. S II–IV
2. Fe II–IV

두 원소는 현재 차이가 크고 spectrum 영향도 크다. full production 전환 전에 matrix dump를 ARTIS와 대조한다.

### Gate 3 — frozen-state transport oracle

ARTIS의 population, `T_e`, `n_e`, level population을 Lumina opacity/emissivity 계산에 고정한다. ionization solve는 끈다.

다음 항목을 wavelength와 shell별로 비교한다.

- bound-bound opacity
- bound-free opacity
- free-free opacity
- emissivity/source function
- Sobolev optical depth
- packet interaction count
- absorption 뒤 macro-atom activation
- fluorescence wavelength transition
- thermalization/k-packet 전환
- escape wavelength

이 실험에서도 2500–3000 Å가 과잉이면 원인은 ionization이 아니라 C4/C6 transport 또는 output 경로다.

### Gate 4 — packet energy/fate census

스펙트럼을 보기 전에 energy conservation을 확인한다.

```text
injected/deposited
= escaped
+ stored
+ adiabatic loss
+ thermal pool
+ numerical loss
```

그리고 escaped energy를 다음 fate별로 나눈다.

- no interaction
- electron scattering
- resonance scattering
- bound-bound macro-atom
- bound-free absorption/recombination
- free-free
- thermal/k-packet re-emission
- cascade-cap fallback

ARTIS와 Lumina의 UV excess가 어느 fate에서 시작되는지 바로 확인할 수 있다.

### Gate 5 — 비교 lane 분리

- ARTIS lane:
  - MC field
  - MC packet spectrum
  - ARTIS-faithful macro-atom
  - Lumina 전용 thermal fallback은 off
- CMFGEN lane:
  - deterministic radiation/SE/thermal iteration
  - formal solution
  - CMFGEN과 동일한 snapshot/normalization

두 lane의 결과를 마지막에만 연결한다. ARTIS parity용 gate와 CMFGEN parity용 closure를 한 실행에서 동시에 켜지 않는다.

## 12. 당분간 parity 판정에 사용하면 안 되는 것

다음 결과는 보조 진단으로만 사용하고 합격/실패 oracle로 쓰지 않는다.

- `scripts/compare_bk_artis.py`의 자동 timestep 선택 결과
- ARTIS timestep 20을 19.48일이라고 간주한 표
- power source와 time averaging이 다른 상태의 absolute `J_ν` ratio
- ARTIS MC `spec.out`과 Lumina formal spectrum의 차이를 특정 macro-atom 버그로 바로 귀속
- shape-normalized spectrum만으로 energy conservation을 판단
- 여러 새 physics gate를 동시에 켠 A/B 결과
- iteration count 또는 damping만 늘려 matrix 구조 parity가 달성되었다는 주장

## 13. 테스트 인프라의 공백

현재 `tests/`에는 Planck 함수와 Sobolev escape probability 같은 단위 테스트가 있지만, ARTIS/CMFGEN oracle을 고정하는 end-to-end regression이 없다.

최소한 다음 regression fixture가 필요하다.

1. ARTIS timestep parser test
2. stage-index mapping test
3. 한 cell/한 원소 rate-matrix golden test
4. level-resolved photoionization target test
5. fixed opacity의 packet-fate histogram test
6. MC spectrum과 formal spectrum을 섞지 않는 output-type test
7. manifest/checksum/provenance test

Golden data는 크고 가변적인 full run 전체가 아니라, 작은 CSV/JSON fixture로 저장하는 편이 낫다. 각 값에는 source commit, timestep, cell, unit을 명시한다.

## 14. 우선 수정 후보

실제 수정은 oracle을 만든 뒤 해야 하지만, 현재 증거가 가리키는 우선순위는 명확하다.

### P0

- ARTIS timestep 선택과 라벨 수정
- `compare_bk_artis.py`의 data-dependent epoch 선택 제거
- 실행별 `lumina_coevolve_field.csv` 보존
- spectrum output type을 파일 metadata에 기록
- 모든 비교에 manifest/checksum 추가

### P1

- radiation-field array의 authoritative source를 iteration별 하나로 정함
- 각 plasma consumer가 읽은 field generation ID를 출력
- 한 cell에서 element-wide NLTE matrix 구현 및 ARTIS matrix dump와 대조
- stage/level/photoionization-target identity table 생성

### P2

- level-resolved ion-changing macro-atom transition
- bound-free activation 및 recombination-continuum packet fate 검증
- ARTIS lane에서 `LINE_THERM` 및 Lumina 전용 fallback 분리
- fixed-state opacity/emissivity/energy census

### P3

- atomic-state와 transport oracle이 통과한 뒤 full ejecta spectrum 재비교
- 그 후에만 CMFGEN thermal/SE/formal lane의 차이를 조정

## 15. 이번 진단의 최종 판단

현재 Lumina가 ARTIS 및 CMFGEN과 다른 첫 번째 이유를 “departure coefficient가 부족하다”로 요약하는 것은 부정확하다.

보다 정확한 판단은 다음과 같다.

1. 기존 ARTIS population 기준선의 epoch가 잘못되어 일부 parity 목표가 오염되었다.
2. 최신 run에서는 radiation field가 plasma solve에 전달되는 단계에서 이미 ARTIS와 큰 차이가 난다.
3. Lumina의 pairwise NLTE system은 ARTIS의 element-wide SE system과 구조적으로 다르다.
4. ion-changing bound-free/macro-atom 처리는 아직 ARTIS의 level-resolved topology와 동등하지 않다.
5. ARTIS MC spectrum과 Lumina formal spectrum을 섞은 비교 때문에 transport 원인 귀속이 불가능하다.
6. 그 결과 Lumina는 심하게 저이온화되어 있으면서도 동시에 UV가 과도하게 탈출하는 복합 실패를 보인다.

따라서 다음 한 단계는 full spectrum을 다시 돌리는 것이 아니라, **ARTIS timestep 27의 shell 8/10/12를 사용한 frozen-cell rate-matrix oracle**을 만드는 것이다. 이 작은 test에서 field 입력, rate, matrix, population을 순서대로 맞추면 C1–C3의 원인을 분리할 수 있다. 이후 fixed-state packet census로 C4–C6를 분리해야 한다.

이 두 oracle이 없으면 다음 full run도 결과가 달라졌다는 사실만 보여줄 가능성이 높고, ARTIS 또는 CMFGEN 방법과의 실제 parity가 개선되었는지는 판단하기 어렵다.

---

## 부록 A. 2026-07-30 조사 작업 메모

이 절은 결론만이 아니라 이번 조사에서 실제로 확인한 경로를 남기기 위한 worklog다. 다음 작업자는 여기서 바로 이어서 진행할 수 있다.

### A.1 작업 시작 상태

- 작업 디렉터리:
  - `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn`
- 확인 당시 branch:
  - `thenmc-macroatom-fluorescence`
- 확인 당시 HEAD:
  - `47bfa20`
- worktree:
  - 기존 수정 및 미추적 파일이 매우 많은 dirty 상태
  - 이들은 사용자 작업으로 간주해 수정·정리·reset하지 않음
- 이번 조사에서 변경한 파일:
  - `docs/ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-30.md` 한 개만 새로 작성
- 소스 코드 및 기존 결과 파일:
  - 변경하지 않음

### A.2 먼저 읽은 기존 문서

- `RESUME.md`
- `docs/ARTIS_PARITY_GAP_AUDIT.md`
- `docs/ARTIS_COMPARISON_LADDER.md`
- `docs/VERIFICATION_REGISTERS.md`
- `logs/coevolve_consume_parity54/VERDICT_DRAFT.md`

기존 문서에서 C0–C7 비교 틀, parity54의 Y6+N3 동시 변경, iteration 9 이후의 분기, 남아 있는 macro-atom/bound-free gap을 파악했다.

### A.3 ARTIS 기준 소스 확인

- 로컬 checkout:
  - `/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref`
- commit:
  - `36f86476d870cec55bcbe9ab80c1b24ada692eb4`
- remote:
  - `https://github.com/artis-mcrt/artis.git`
- 확인한 주요 파일:
  - `nltepop.cc:563-619`
    - level-resolved photoionization target
    - thermal collisional ionization
    - radiative 및 three-body/collisional recombination
  - `nltepop.cc:1200-1289`
    - 한 원소 전체 ion stage를 포함하는 statistical-equilibrium matrix
    - element-total normalization
  - `nltepop.cc:1415` 부근
    - superlevel excitation 온도 선택
  - `radfield.cc`
    - per-bin radiation-field estimator와 bound-free estimator
  - `macroatom.cc`
    - ion-changing macro-atom action 및 k-packet 관련 경로

ARTIS 테스트 옵션에서 `FORCE_SAHA_ION_BALANCE=false`임을 확인했다. 따라서 ARTIS 기준은 stage population을 Saha 값에 고정하는 비교가 아니다.

### A.4 timestep 오류를 찾은 과정

확인한 ARTIS 테스트 디렉터리:

- `artis-ref/tests/toy06_nlte_bk`
- `artis-ref/tests/toy06_whitebox_run`

`timesteps.out`의 timestep 20, 26, 27을 직접 확인했다. 그 결과 timestep 20의 midpoint가 11.2353일이며, 19.48일을 포함하는 bin은 timestep 27임을 확인했다.

그 다음 아래 비교 코드와 문서의 timestep 처리 방식을 대조했다.

- `scripts/artis_baseline_bk.py`
- `scripts/compare_bk_artis.py`
- `scripts/radfield_3way.py`
- `docs/ARTIS_PARITY_GAP_AUDIT.md`

핵심 발견:

- `artis_baseline_bk.py` 기본값 `TS=20`은 19.48일 기준으로 틀림
- `compare_bk_artis.py`는 epoch를 시간으로 고정하지 않고 low-level S II population이 큰 timestep을 선택함
- 이 때문에 초기 epoch의 큰 `b_k`를 19.48일 Lumina와 비교할 수 있음
- `radfield_3way.py`의 `TS=27` 선택은 시간상 더 적절함

### A.5 최신 spectrum 비교

실행한 진단 명령:

```bash
python3 scripts/cmp3_artis.py logs/coevolve_consume_parity54
```

기록한 핵심 결과:

- peak:
  - Lumina 2830 Å
  - CMFGEN 3922 Å
  - ARTIS 4102 Å
- correlation:
  - Lumina–CMFGEN 0.346
  - Lumina–ARTIS 0.069
  - CMFGEN–ARTIS 0.849
- Lumina의 2500–3000 Å 정규화 flux:
  - 42.7%
- CMFGEN:
  - 8.5%
- ARTIS:
  - 5.5%

이 결과를 근거로 최신 failure mode를 “red excess”가 아니라 “UV escape/optical redistribution failure”로 기록했다.

주의:

- 이 스크립트는 shape-normalized 비교이므로 absolute luminosity 또는 energy conservation은 판정하지 못한다.
- Lumina 입력은 formal spectrum이고 ARTIS 입력은 MC emergent spectrum이므로 원인 귀속용 oracle은 아니다.

### A.6 radiation-field 비교

실행한 진단 명령:

```bash
python3 scripts/radfield_3way.py . 27
```

사용한 Lumina field:

- 루트의 `lumina_coevolve_field.csv`

주의:

- parity54 실행 디렉터리 안에는 해당 파일이 보존되지 않았다.
- 수정시각과 실행 계보로 parity54 대응 파일임을 판단했으나, checksum manifest가 없으므로 완전한 provenance는 아니다.
- 다음 실행부터 runner가 실행 디렉터리에 이 파일을 복사해야 한다.

확인한 현상:

- shell과 wavelength에 따라 Lumina MC `J`/ARTIS `J`가 약 0.2에서 100 이상까지 변함
- scalar `W`는 대체로 ARTIS보다 낮음
- Lumina `T_rad`가 많은 shell에서 10470.093 K로 동일
- `n_e`는 상대적으로 가까우나 `T_e`는 특히 외곽에서 낮음

### A.7 ion fraction 비교

ARTIS timestep 27의 `ion_frac_0000.out`과 Lumina parity54 `plasma_state.csv`를 동일 shell에서 직접 대조했다.

확인한 대표 현상:

- shell 8에서 ARTIS는 Si/S/Fe/Co/Ni IV가 우세
- Lumina는 같은 원소의 III stage가 거의 전부를 차지
- shell 10–25의 S도 Lumina에서 IV stage가 심하게 부족
- stage index는 ARTIS의 1-based 표기와 Lumina의 0-based 내부 표기를 물리적 ion stage로 변환해 비교

이를 근거로 “Lumina가 ARTIS보다 심하게 저이온화되어 있다”고 판정했다.

### A.8 Lumina 소스 구조 확인

확인한 주요 위치:

- `src/lumina_plasma.c:7242-7277`
  - adjacent ion pair 구성
- `src/lumina_plasma.c:1421-1443`
  - pair/element density normalization 선택
- `src/lumina_plasma.c:15391-15397`
  - pair-total conservation row
- `src/lumina_plasma.c:15933-16043`
  - 여러 ion pair의 반복 solve와 damping
- `src/lumina_plasma.c:6316-6324`
  - upper-ion ground mapping 제한
- `src/lumina_cuda.cu:4403-4422`
  - ion-changing up-jump 구성
- `src/lumina_cuda.cu:4907-4919`
  - cascade cap 이후 thermal re-emission fallback
- `src/lumina_cuda.cu:9331-9351`
  - binned Monte Carlo `J` download 및 fit gate

이 검토에서 다음을 구분했다.

- 기능 gate가 존재한다는 사실
- ARTIS와 같은 방정식·topology·consumer wiring임이 검증되었다는 사실

현재는 전자는 다수 확인되지만 후자는 아직 확인되지 않았다.

### A.9 parity54 설정에서 확인한 혼합 경로

resolved environment와 실행 로그에서 다음 종류의 설정이 동시에 사용된 것을 확인했다.

- pure-CMFGEN/deterministic 경로
- MC field/Jbar estimator 경로
- NLTE collision/recombination correction
- macro-atom radiative recombination
- line thermalization fallback
- formal spectrum 출력

따라서 parity54는 하나의 깨끗한 ARTIS method clone이라기보다 여러 closure를 함께 시험한 hybrid run으로 기록했다.

### A.10 이번에 하지 않은 일

- Lumina 소스 수정
- 비교 스크립트 수정
- 새 full transport 실행
- compiler/build 변경
- 기존 dirty worktree 정리
- 기존 결과 삭제 또는 이동
- ARTIS/CMFGEN의 full rerun
- frozen-cell oracle 구현
- 수치 계수 tuning

이 문서는 진단과 다음 실험 설계까지만 수행한 결과다.

### A.11 다음 작업 재개 지점

다음 세션에서는 아래 순서로 시작하는 것이 가장 안전하다.

1. `scripts/compare_bk_artis.py`에서 data-dependent timestep 선택 제거
2. 모든 ARTIS parser가 `timesteps.out`을 읽도록 공통 함수 작성
3. 19.48일 기본값을 timestep 27로 설정하고 bin start/mid/end를 함께 출력
4. shell 8/10/12 frozen-cell fixture 생성
5. S II–IV 한 원소의 ARTIS/Lumina rate 및 matrix dump 비교
6. first mismatching matrix entry를 찾은 뒤에만 물리 구현 수정
7. 별도로 fixed-state packet fate/energy census 작성

재개 시 먼저 확인할 질문:

- parity54의 루트 `lumina_coevolve_field.csv`를 실행과 확실히 묶을 checksum 또는 로그가 남아 있는가?
- Lumina MC emergent spectrum 파일이 별도로 존재하는가?
- CMFGEN 비교 snapshot의 absolute luminosity와 Lumina boundary/deposition normalization이 같은가?
- ARTIS atomic photoionization target index를 Lumina atomic model에 손실 없이 매핑할 수 있는가?

