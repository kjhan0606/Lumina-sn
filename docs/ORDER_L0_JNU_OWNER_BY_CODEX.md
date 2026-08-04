# 발주서 — A-2 `J_ν` 단일 소유권 전환 및 CMFGEN 다층 검증

- 발주일: 2026-08-04
- 상태: 구현 전 확정 발주
- 저작: Codex
- 검수·실행·회귀 원장: 운전석
- 구현: Codex
- 변경 범위: 본 답변은 발주서이며 저장소를 변경하지 않는다.
- 선행 자료: [ORDER_L0_TRADFIX_TSEED_BY_CODEX.md](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/ORDER_L0_TRADFIX_TSEED_BY_CODEX.md:1>), [CODEX_IMPL_L0_TRADFIX.md](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_IMPL_L0_TRADFIX.md:1>), [verify_trad_fix.py](</home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_trad_fix.py:1>), [cmfgen.json](</home/kjhan/.lumina_scratch/tradfix_out/cmfgen.json>)

---

## 1. 발주 판정

A-2로 간다.

단일 희석 Planck `(W,T_rad)`는 CMFGEN `J_ν`를 표현하지 못한다. s0에서 등록된 다섯 대역인 450–918 Å, 918–1290 Å, 1290–2000 Å, 2000–10000 Å, 10000–25000 Å가 모두 10% 기준을 벗어난다. Planck 적합 잔차와 전 셸 복사 에너지 차이도 값 조정으로 해결할 수준이 아니다.

따라서 다음은 발주 대상이 아니다.

- `T_rad`만 보정하는 A-1
- `W` 재조정
- 색온도 fitting 범위 변경
- UV용 보정계수
- `J_ν`를 다시 빈별 `(W,T_R)`로 압축하는 ARTIS C1 모사
- TRAD-FIX의 임계값 완화

초안의 L-a∼L-g 방향은 맞지만 충분하지 않다. 다음을 고쳐야 한다.

- 광이온율과 bound-bound 복사율을 분리한다.
- CMFGEN NLTE 준위인구를 Boltzmann 분배함수와 직접 비교하지 않는다.
- 이온인구와 준위인구를 분리한다.
- macro-atom 전이·재분배 층을 추가한다.
- `T_e`뿐 아니라 가열·냉각 항과 복사평형 잔차를 비교한다.
- 방출률 뒤와 창발 스펙트럼 앞에 내부 수송 모멘트 `H_ν` 층을 추가한다.

최종 과학 게이트는 다음 열한 층이다.

1. L-0 `J_ν`
2. L-1bf bound-free 복사율
3. L-1bb bound-bound 복사율
4. L-2ion 이온화 상태
5. L-2level 준위인구
6. L-3 전이·재분배 커널
7. L-4 불투명도 `χ_ν`
8. L-5 방출률 `η_ν`
9. L-6 전자온도·열수지
10. L-7 내부 수송 `H_ν`
11. L-8 창발 스펙트럼

---

## 2. 최상위 계약: 복사장 주인은 하나다

### 2.1 정본

런타임 복사장의 유일한 정본은 다음이다.

```text
RadiationField {
    shell_boundaries
    frequency_bin_edges
    J_nu[shell][bin]
    units
    frame
    epoch
    generation
    provenance
    validity
    estimator_count_or_variance
}
```

`J_nu[s][b]`는 빈 중심 표본이 아니라 다음 빈 평균이다.

\[
J_{s,b}={1\over\Delta\nu_b}\int_{\nu_{b,-}}^{\nu_{b,+}}J_\nu(s)\,d\nu
\]

단위는 `erg s^-1 cm^-2 Hz^-1 sr^-1`, 좌표계는 셸 공이동계로 고정한다. observer-frame 양은 이 객체에 넣지 않는다.

### 2.2 소유권과 허용 객체

허용되는 비정본 객체는 다음뿐이다.

- MC raw path-length estimator
- 통계량 count·variance
- 다음 generation을 계산하는 미공개 작업 버퍼
- 출력 전용 진단량
- 정본 generation에 결박된 읽기 전용 파생 캐시

이 객체들은 소비자 API가 될 수 없다. 작업 버퍼나 `CMFGENState.J`가 존재하더라도 commit 전 임시 상태여야 하며, commit 뒤에는 어떤 rate·population·opacity·emissivity·transfer 소비자도 직접 읽지 못한다.

`J_ν` 갱신 damping은 고정점을 바꾸지 않는 한 허용한다. 계수와 이전·새 generation은 기록한다.

### 2.3 제거할 대체 소유권

최종 상태에서 다음 경로는 계산 소유권을 갖지 않는다.

- `plasma->T_rad`
- `plasma->W`
- GPU의 `d_T_rad`, `d_W`
- `nlte_build_perbin_dilute_field()`가 만든 빈별 `(W_c,T_R,c)`
- 직접 소비되는 `bf_rate_estimator`
- 직접 소비되는 `jbar_line`, `j_blue`
- 직접 소비되는 `j_estimator`, `nu_bar`
- 순수 CMFGEN 내부 경로의 공개 `cs.J`
- seed용 스칼라 배열
- fixed-radiation-profile 환경변수
- `LUMINA_TRAD_COLOR_FIX`

line `\bar J`는 별도 복사장이 아니라 정본으로부터 명시적으로 계산한다.

\[
\bar J_{lu}=\int\phi_{lu}(\nu)J_\nu\,d\nu
\]

### 2.4 현재 census의 처리

현재 157개 항목은 전부 disposition 원장에 남겨야 한다. 역할별 내역은 다음과 같다.

- rate 24
- comparator 14
- GPU_opacity_rate 13
- GPU_transport 11
- opacity_rate 9
- formal_transfer 9
- GPU_lifecycle 8
- GPU_rate 8
- owner_validation 7
- owner_update 7
- seed_radeq 6
- input_owner 5
- diagnostic 4
- GPU_emissivity 4
- rate_diagnostic 3
- GPU_transfer 3
- opacity 3
- seed_rate 3
- lifecycle 2
- output 2
- Boltzmann_partition 2
- transition_probability 2
- rate_Boltzmann 2
- rate_radeq 2
- Boltzmann_diagnostic 2
- seed 1
- emissivity 1
- 미분류 0

각 행에는 `파일:행`, 심볼, 현재 공급원, 물리 의미, 새 공급원, 이행 단계, 최종 상태를 기록한다. 진단 항목이라는 이유로 생략하지 않는다.

---

## 3. 온도 파생 계약

범용 `T_rad`는 만들지 않는다. 필요한 온도는 caller와 수식을 이름에 포함한 순수 함수로만 유도한다.

### 3.1 허용되는 파생량

복사 에너지 밀도 온도:

\[
u_{\rm rad}={4\pi\over c}\int J_\nu d\nu,\qquad
T_E[J]=\left({u_{\rm rad}\over a}\right)^{1/4}
\]

이는 에너지 밀도 진단에만 쓴다. 유한 대역이면 `T_E[ν_min,ν_max]`로 표기하며 bolometric 온도라고 부르지 않는다.

Compton 온도:

\[
T_C[J]={h\over4k}
{\int\nu J_\nu d\nu\over\int J_\nu d\nu}
\]

이는 해당 Compton 교환식에만 쓴다.

특정 주파수 brightness temperature가 필요하면 `T_b(ν;J_ν)`로 명시한다. 이것도 rate나 population의 일반 대체 입력이 아니다.

### 3.2 금지되는 사용

- 분배함수와 LTE/Boltzmann 인구는 물질온도 `T_e`를 쓴다.
- 광이온율은 `J_ν`와 단면적을 직접 적분한다.
- bound-bound율은 `\bar J`를 직접 쓴다.
- 복사평형은 항별 가열·냉각으로 `T_e`를 푼다.
- 재방출 주파수는 `η_ν/\int η_νdν`에서 표본한다.
- formal transfer는 `η_ν`, `χ_ν`, 경계조건을 직접 쓴다.
- `W`는 파생량으로도 복원하지 않는다.
- 파생온도를 캐시할 경우 정본 generation 불일치는 즉시 실패한다.

---

## 4. 주파수 범위와 해상도

현재 `J_ν` 격자는 100–20000 Å이지만 검증 대역은 25000 Å까지다. 현재처럼 격자 밖 질의를 `1e-30`으로 돌려주는 것은 무증상 물리 변경이다.

구현 전에 모든 소비자의 주파수 합집합을 작성한다.

- bound-free threshold와 단면적
- bound-bound 선 주파수와 profile 폭
- opacity·emissivity 격자
- packet emission·absorption
- formal transfer
- observer spectrum
- 등록 검증 대역

정본 격자는 이 합집합을 포함해야 하며 적어도 25000 Å까지 덮어야 한다.

빈 수 1000은 계약값이 아니다. lageunha에서 1000, 2000, 4000, 8000, 16000 빈의 오프라인 해상도 사다리를 실행한다. 가장 작은 후보와 바로 다음 두 배 격자 사이에서 다음 변화가 모두 만족되는 가장 작은 후보를 선택한다.

- 등록 대역별 `∫J_νdν`: 최대 1%, 중앙값 0.2% 이하
- 매칭된 `Γ`: 최대 1%, 중앙값 0.2% 이하
- 매칭된 `\bar J`: 최대 1%, 중앙값 0.2% 이하
- `χ_ν`, `η_ν` 대역 적분: 최대 1%, 중앙값 0.2% 이하

8000 대 16000도 실패하면 격자 결정은 `BLOCKED`다. 1000을 관성적으로 유지하지 않는다.

---

## 5. CMFGEN 원장 자격

### 5.1 현재 스냅샷의 자격

현재 참조 디렉터리는 다음이다.

```text
/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/
```

현재 `EDDFACTOR`는 90개 depth, 196185개 유효 주파수 레코드이며 파일 종료 레코드는 정상이다. s0∼s43은 속도 범위 안이고 s44∼s49는 밖이다.

그러나 `run_jnu4.info`에는 다음이 명시돼 있다.

- `FIX_T=T`
- restart record 62
- 순수 LAMBDA
- 반복 4회
- iteration 67의 NaN 전에 중단

따라서 상태를 분리한다.

```text
CMFGEN_FILE_INTEGRITY       = PASS
CMFGEN_SNAPSHOT_REPLAY      = ELIGIBLE
CMFGEN_NONLINEAR_CONVERGENCE= FAIL
CMFGEN_PHYSICAL_ORACLE      = INELIGIBLE
```

`FINISH_REC=1`은 파일 완결이지 비선형 수렴이 아니다.

현재 스냅샷은 L-0 extractor, wiring replay, 음성 대조에 사용할 수 있다. 최종 물리 PASS를 부여할 수는 없다.

현재 스냅샷에는 다음 최종 게이트 자료가 없다.

- `CHI_DATA`, `CHI_DATA_INFO`
- `ETA_DATA`, `ETA_DATA_INFO`
- `NETRATE`
- `TOTRATE`
- `LINEHEAT`

`MEANOPAC`, `NEG_OPAC`, `GENCOOL`을 빠진 스펙트럼 자료의 대용품으로 쓰지 않는다.

### 5.2 최종 물리 원장 `O-PHYS`

운전석은 최종 인수 전에 동일 epoch·hydro·composition의 새 CMFGEN 원장을 제공한다. 원장에는 다음이 들어가야 한다.

- `EDDFACTOR`, `EDDFACTOR_INFO`
- `RVTJ`
- 매칭 이온의 `*PRRR`
- `NETRATE`, `TOTRATE`, 필요한 경우 상·하향 성분을 분리한 rate audit
- `POPCAL`, `POPCOB`, `POPIRON`, `POPNICK`, `POPSIL`, `POPSUL`
- 매칭용 `*OUT`과 level·superlevel 식별 자료
- `CHI_DATA`, `CHI_DATA_INFO`
- `ETA_DATA`, `ETA_DATA_INFO`
- `GENCOOL`, `LINEHEAT`
- `JH_AT_CURRENT_TIME`, `JH_AT_CURRENT_TIME_INFO`
- `OBSFLUX`, `OBS_FREQ`
- `OUTGEN`
- 코드 revision, 입력 파일, 원자자료 hash, 파일별 SHA-256, 단위·frame·record schema를 담은 manifest

온도는 풀어야 하며 `FIX_T=T`인 실행은 L-6 원장이 아니다. 고정한 이온이나 준위가 있으면 이름과 전체 population·rate·opacity·emissivity 기여율을 공개한다. 공개되지 않은 freeze가 하나라도 있으면 원장 자격은 실패다.

수렴 자격은 다음을 모두 요구한다.

- NaN·Inf 없음
- 마지막 세 반복에서 등록 대역 `J_ν`, `T_e`, 주요 이온분율의 변화가 각각 최대 1% 이하
- 활성 population의 최대 보정이 1% 이하
- 각 depth의 정규화 열수지 잔차가 `10^-3` 이하
- 마지막 반복 이후 작성된 모든 산출물이 동일 generation임을 hash manifest로 증명

---

## 6. 공통 비교 규약

### 6.1 공간 좌표

- Lumina 셸 중점 속도를 공통 좌표로 쓴다.
- CMFGEN depth는 `RVTJ`의 속도로 매핑한다.
- 양수인 국소량은 속도에 대해 로그 보간한다.
- 부호가 있는 `H_ν`, 순가열, 음의 opacity는 선형 보간하며 부호를 별도 검증한다.
- 체적 적분량은 셸 경계 사이에서 체적 보존 평균한다.
- s44∼s49에는 현재 CMFGEN 값을 hold·외삽·복사하지 않는다. 상태는 `OUT_OF_ORACLE`이다.
- 50개 셸 최종 PASS에는 50개 셸을 덮는 `O-PHYS`가 필요하다.

### 6.2 주파수 좌표

CMFGEN fine grid를 정본 빈 경계로 보존 적분한다.

\[
\bar q_b={1\over\Delta\nu_b}\int_{\nu_{b,-}}^{\nu_{b,+}}q_\nu\,d\nu
\]

- point sample 비교는 금지한다.
- rate는 가능하면 CMFGEN fine grid에서 먼저 적분한다.
- line은 주파수 근접성만으로 연결하지 않는다. 원소, 이온, 하위·상위 level label, 에너지, 통계가중치로 연결한다.
- observer spectrum은 공통 wavelength edge에 보존 재빈한다.

### 6.3 분모와 0 처리

비음수 스펙트럼의 기본 오차는 다음이다.

\[
E_1={\sum_b\Delta\nu_b|L_b-C_b|
\over\sum_b\Delta\nu_b C_b}
\]

대역 에너지 오차는 다음이다.

\[
E_B={|\sum_{b\in B}\Delta\nu_b(L_b-C_b)|
\over\sum_{b\in B}\Delta\nu_b C_b}
\]

부호가 있는 양은 분모를 `Σw|C|`로 정의한다.

스칼라 대칭오차는 다음이다.

\[
E_{\rm sym}={2|L-C|\over |L|+|C|}
\]

양쪽이 정확히 0이면 0이다. CMFGEN이 0이고 Lumina가 0이 아니면 false positive로 별도 FAIL한다. 로그 오차는 양쪽이 양수인 항에서만 계산하며 epsilon을 더하지 않는다.

매칭 범위는 다음으로 기록한다.

\[
f_{\rm cov}={\hbox{매칭된 CMFGEN 기여}}
{\hbox{전체 CMFGEN 기여}}
\]

작은 항을 임의 숫자 floor로 제외하지 않는다. 누적 CMFGEN 기여 99.9%를 이루는 활성 집합을 사용한다.

MC 비교는 95% 신뢰구간 반폭이 해당 합격선의 3분의 1 이하일 때만 판정한다. 통계오차를 물리오차에서 빼지 않는다.

### 6.4 층별 원인 분리

각 층은 두 실행을 가진다.

- `CHAIN`: Lumina가 자체 상류 결과를 사용한다.
- `ORACLE_INPUT`: 해당 층의 바로 위 경계에 CMFGEN 대응량을 주입한다.

판정은 다음과 같다.

| CHAIN | ORACLE_INPUT | 판정 |
|---|---|---|
| PASS | PASS | 해당 층 PASS 가능 |
| FAIL | PASS | 상류층 원인 |
| FAIL | FAIL | 해당 층 자체에도 결함 |
| PASS | FAIL | 상쇄에 의한 거짓 PASS, 해당 층 FAIL |

여기에 음성 대조가 실제로 거부되어야 최종 PASS다. 하류 PASS가 상류 FAIL을 덮지 못한다.

---

## 7. 층별 게이트

### L-0 — 복사장 표현

- CMFGEN 파일: `EDDFACTOR`, `EDDFACTOR_INFO`, 공간 좌표용 `RVTJ`
- 추출: record 크기·endian·metadata·`FINISH_REC`를 검증한 뒤 `FL`의 `10^15 Hz` 단위를 복원하고 `J_ν`를 보존 재빈한다.
- 좌표: 공이동 `v,ν`; 현재는 s0∼s43만 판정한다.
- 분모: `E_1`, 대역별 `E_B`, 활성 빈의 `P95(|log10(L/C)|)`
- 합격선:
  - 각 공통 셸 `E_1 ≤ 0.10`
  - 다섯 등록 대역 각각 `E_B ≤ 0.10`
  - 활성 빈 로그오차 P95 `≤ 0.15 dex`
- 음성 대조: 기존 deck `W B_ν(T_rad)`를 넣는다. s0의 다섯 대역이 모두 FAIL해야 한다.
- 추가 조건: 격자 밖 조회, invalid bin, generation 불일치는 값 반환이 아니라 명시적 오류다.

### L-1bf — bound-free 복사율

비교 대상은 다음과 같이 채널을 분리한다.

\[
\Gamma_i=4\pi\int_{\nu_{0,i}}^\infty
{J_\nu\sigma_i(\nu)\over h\nu}\,d\nu
\]

- CMFGEN 파일: 매칭 이온의 `*PRRR`
- 추출: depth별 photoionization rate, collisional ionization, spontaneous·stimulated recombination을 파일 header 의미대로 분리한다.
- 보조 replay: CMFGEN `J_ν`와 Lumina 단면적으로 계산한 `Γ[J,\sigma_L]`. 이는 wiring 검증이며 CMFGEN PRRR의 대체 원장이 아니다.
- 좌표: 공통 셸, 정확히 매칭된 threshold·level
- 분모: CMFGEN rate-flow 가중 `E_1`; 셀별 `E_sym`
- 합격선:
  - 기여 coverage `f_cov ≥ 0.95`
  - 전체 채널 가중 `E_1 ≤ 0.10`
  - 활성 level-shell `E_sym` P95 `≤ 0.25`
  - photoionization·stimulated recombination·spontaneous recombination 각각의 합계 오차 `≤ 0.10`
- 음성 대조: 기존 `W B_ν`, threshold를 한 빈 이동한 단면적, stimulated 항 제거를 각각 넣어 FAIL을 확인한다.
- 금지: `bf_rate_estimator`를 `J_ν`와 동급의 직접 rate 소스로 사용.

### L-1bb — bound-bound 복사율

- 비교량: `\bar J`, `R_lu=B_lu\bar J`, `R_ul^stim=B_ul\bar J`, `R_ul^sp=A_ul`
- CMFGEN 파일: `WRITE_RATES`로 생성한 `NETRATE`, `TOTRATE`, level 연결 자료
- 추출: writer source와 함께 schema를 증명한다. net rate만 있고 상·하향 성분을 분리할 수 없으면 전용 CMFGEN rate audit를 추가한다. 큰 두 수의 차인 net rate로 두 성분을 역추정하지 않는다.
- 좌표: transition ID와 profile-integrated 공이동 `J_ν`
- 분모: 전체 매칭 radiative flow; net rate가 아니라 상·하향 절대 flow
- 합격선:
  - flow coverage `≥ 0.95`
  - `\bar J`, 상향률, stimulated 하향률 각각 `E_1 ≤ 0.10`
  - 활성 transition-shell `E_sym` P95 `≤ 0.25`
  - spontaneous `A_ul`은 atomic crosswalk 상대오차 `≤10^-10`
- 음성 대조: line frequency 한 빈 이동, stimulated 항 제거, 상·하위 level 교환을 각각 거부한다.
- 현재 스냅샷 상태: `BLOCKED_MISSING_RATE_EXPORT`

### L-2ion — 이온화 상태

- CMFGEN 파일: `POPCAL`, `POPCOB`, `POPIRON`, `POPNICK`, `POPSIL`, `POPSUL`, `RVTJ`
- 추출: level population을 이온별로 합산하고 `n_e`와 원소 총밀도를 함께 읽는다.
- 좌표: 공통 셸, 원소·이온 stage
- 분모: 각 원소의 총 number density
- 합격선:
  - 이온분율 total variation `0.5Σ|f_L-f_C| ≤ 0.10`
  - 지배 이온 stage 일치
  - `n_e` 대칭오차 중앙값 `≤0.10`, P95 `≤0.20`
  - 원소별 population closure 상대오차 `≤10^-10`
- 음성 대조: 인접 이온 stage를 바꿔 연결하거나 `n_e`를 인접 depth에서 가져오면 FAIL해야 한다.
- 현재 스냅샷은 고정된 Si VI, S VI, Ca VI, Fe VI, Fe VII, Ni VI, Ni VII, Co VI, Co VII의 영향을 명시하지 않으면 물리 PASS에 사용할 수 없다.

### L-2level — 준위인구·분배함수

CMFGEN NLTE level population은 Boltzmann 분배함수의 정답이 아니다. 두 검사를 분리한다.

- 물리 비교 파일: 위 `POP*`, 이온별 `*OUT`, level·superlevel 연결 자료
- 매칭: label, excitation energy, `g`, parent ion, superlevel membership
- 물리 분모: 이온 총 population
- 물리 합격선:
  - 매칭 population coverage `≥0.95`
  - 매칭 level/superlevel 합계 오차 `≤0.10`
  - `n_i/n_ion` 로그오차 P95 `≤0.30 dex`
  - 매칭 불가능한 level을 0으로 치환하지 않음
- 내부 분배함수 검사:
  \[
  Z(T_e)=\sum_i g_i e^{-(E_i-E_0)/(kT_e)}
  \]
  CPU 상대오차 `≤10^-10`, GPU 상대오차 `≤5×10^-5`
- 음성 대조: `T_e` 대신 옛 `T_rad`를 사용하거나 level 순서를 섞으면 FAIL해야 한다.

### L-3 — 전이·재분배 커널

CMFGEN에는 Lumina packet branching probability와 일대일인 객체가 없다. 따라서 이 층은 내부 보존과 CMFGEN `η_ν`로 닫는다.

- Lumina 비교량: 활성 상태별 전이확률, 흡수 에너지에서 예측된 채널별 방출 에너지
- CMFGEN 대응: `ETA_DATA`의 채널별 또는 총 방출률
- 합격선:
  - 각 활성 상태에서 확률 합 `1±10^-10` CPU, `1±5×10^-5` GPU
  - 음수·NaN 확률 0개
  - 흡수·내부전환·방출 에너지 closure `≤10^-8` CPU
  - 예측 `η_ν`는 L-5 합격선을 만족
- 음성 대조: 전이 목적지를 순열하되 확률 합은 보존한다. 단순 합계 검사는 통과하더라도 `η_ν` 검사가 FAIL해야 한다.
- 현재 스냅샷 상태: `BLOCKED_MISSING_ETA_DATA`

### L-4 — 불투명도 `χ_ν`

- CMFGEN 파일: `CHI_DATA`, `CHI_DATA_INFO`; `NEG_OPAC`은 부호 진단만 담당
- 추출: writer schema에서 total, electron scattering, bound-bound, bound-free, free-free의 포함 의미와 단위를 확정한다.
- 좌표: 공통 셸과 공이동 주파수; 부호를 보존한다.
- 분모: `ΣΔν|χ_C|`; 성분별 CMFGEN 절대 기여
- 합격선:
  - 매칭 기여 coverage `≥0.95`
  - 셸별 total `E_1 ≤0.15`
  - 다섯 등록 대역 각각 `E_B ≤0.15`
  - 성분합과 total closure `≤10^-10` CPU
  - CMFGEN이 음수인 활성 구간의 부호 일치
- 음성 대조: stimulated-emission 보정 제거, 한 bound-free edge 이동, 한 opacity 채널 제거를 각각 거부한다.
- 현재 스냅샷 상태: `BLOCKED_MISSING_CHI_DATA`

### L-5 — 방출률 `η_ν`

- CMFGEN 파일: `ETA_DATA`, `ETA_DATA_INFO`
- 추출: true emission과 scattering source의 포함 의미를 writer source로 확정한다. 정의가 다른 성분을 이름만 같다고 비교하지 않는다.
- 좌표: 공통 셸과 공이동 주파수
- 분모: `ΣΔν η_C`; 채널별 CMFGEN 방출 에너지
- 합격선:
  - 매칭 기여 coverage `≥0.95`
  - 셸별 total `E_1 ≤0.15`
  - 다섯 등록 대역 각각 `E_B ≤0.15`
  - 성분합과 total closure `≤10^-10` CPU
  - L-3 표본 방출과 analytic `η_ν`가 MC 95% 신뢰구간 안에서 일치
- 음성 대조: line, free-bound, free-free 채널을 하나씩 제거하여 모두 FAIL을 확인한다.
- 현재 스냅샷 상태: `BLOCKED_MISSING_ETA_DATA`

### L-6 — 전자온도·복사평형

- CMFGEN 파일: `RVTJ`, `GENCOOL`, `LINEHEAT`, `OUTGEN`
- 추출: `T_e`, 항별 heating, 항별 cooling, net residual을 같은 부호 규약과 체적 단위로 변환한다.
- 좌표: 공통 셸의 체적 평균
- 분모:
  \[
  E_{\rm balance}={|H-C|\over H+C}
  \]
  항별 비교는 `Σ|Q_C|`
- 합격선:
  - `T_e` 대칭오차 중앙값 `≤0.10`, P95 `≤0.20`
  - 양 코드의 각 셸 `E_balance ≤10^-3`
  - 항별 heating·cooling vector `E_1 ≤0.20`
  - photoheating, line heating/cooling, free-free, recombination을 상쇄 전 별도 보고
- 음성 대조: photoheating 제거, 인접 depth의 `T_e` 주입, 순가열만 맞도록 두 큰 항을 상쇄시킨 fixture를 각각 거부한다.
- 현재 스냅샷 상태: `BLOCKED_FIXED_T_AND_MISSING_LINEHEAT`

### L-7 — 내부 수송 모멘트

- CMFGEN 파일: `JH_AT_CURRENT_TIME`, `JH_AT_CURRENT_TIME_INFO`
- 추출: record의 `R²J`, `R²H`, 경계 `H`를 읽고 반지름 규약을 복원한다.
- 비교량:
  \[
  F_\nu=4\pi H_\nu,\qquad
  L_\nu=16\pi^2r^2H_\nu
  \]
- 좌표: 공통 셸 경계, 공이동 주파수
- 분모: `ΣΔν|H_C|`; bolometric luminosity는 CMFGEN luminosity
- 합격선:
  - 셸 경계별 `H_ν E_1 ≤0.10`
  - 등록 대역별 flux 오차 `≤0.10`
  - bolometric luminosity 오차 `≤0.05`
  - 활성 빈의 부호 불일치 0개
  - 셸간 luminosity divergence와 물질 에너지 교환 closure `≤0.01`
- 음성 대조: `R²` 정규화 누락, `H` 부호 반전, depth 순서 반전을 각각 거부한다.

### L-8 — 창발 스펙트럼

- CMFGEN 파일: `OBSFLUX`, `OBS_FREQ`
- 추출: CMFGEN의 정식 reader schema를 따르고 flux·luminosity·거리 정규화를 manifest에 고정한다.
- 좌표: observer-frame 공통 wavelength edge; 사전 등록한 하나의 해상도 kernel만 적용
- 분모: CMFGEN absolute observer flux
- 합격선:
  - bolometric flux 오차 `≤0.05`
  - 다섯 등록 대역 각각 적분 오차 `≤0.10`
  - 전체 스펙트럼 `E_1 ≤0.15`
  - 활성 구간 로그오차 P95 `≤0.15 dex`
- 음성 대조: 공이동 스펙트럼을 observer-frame으로 가장한 입력, 한 해상도 요소의 wavelength 이동, 거리 정규화 제거를 각각 거부한다.

합격선은 구현 결과를 본 뒤 완화하지 않는다. 변경이 필요하면 원장 근거와 별도 발주가 필요하다.

---

## 8. 덱 seed 및 TRAD-FIX 처분

### 8.1 결정

네이티브 덱 seed를 `J_ν`로 바꾼다.

ARTIS의 단일 흑체 bootstrap은 이후 명시적인 multibin 복사장으로 전환되기 때문에 허용된다. Lumina에는 그 전환 계약이 없었고, 이번 A-2의 0층 검증은 시작점부터 `J_ν` 좌표를 요구한다. 그러므로 런타임 입력에 `(W,T_rad)`를 남겨서는 안 된다.

새 seed에는 다음을 직렬화한다.

- 셸 경계 또는 셸 식별자
- 주파수 bin edge
- `J_ν` bin average
- 단위와 frame
- epoch
- 생성 방법
- 원본 파일 hash
- 유효 셸·주파수 mask

### 8.2 레거시 덱

레거시 `(W,T_rad)`는 런타임 입력이 아니라 오프라인 변환기의 입력으로만 허용한다.

변환기는 `W B_ν(T_rad)`를 실제 배열로 만들어 새 덱에 기록하고 provenance를 `DILUTE_PLANCK_LEGACY_APPROXIMATION`으로 남긴다. 런타임은 변환 뒤의 `J_ν`만 본다.

이 변환 결과가 CMFGEN L-0을 통과할 것이라고 기대하지 않는다. toy06의 기존 스칼라 seed는 의도된 L-0 음성 대조다.

현재 CMFGEN 자료에서 s0∼s43은 CMFGEN-rebinned seed를 만들 수 있다. s44∼s49는 `valid=false`로 남기며 hold하지 않는다. 50개 셸 CMFGEN seed를 요구하는 실행은 원장 범위가 확장될 때까지 `BLOCKED`다.

seed는 generation 0에서만 사용한다. 첫 정식 field commit 뒤 seed 파일을 변경하는 poison test가 generation 1 이후 결과를 바꾸면 실패다.

최종 수렴점의 seed 독립성은 최소한 다음 두 seed로 확인한다.

- 레거시 dilute-Planck를 오프라인 변환한 명시적 `J_ν`
- CMFGEN `EDDFACTOR`에서 보존 재빈한 명시적 `J_ν`

공통 44개 셸에서 두 실행의 최종 L-0∼L-8 차이는 각 층 합격선의 10분의 1 이하이고 MC 신뢰구간과 양립해야 한다. 그렇지 않으면 seed는 bootstrap이 아니라 숨은 물리 매개변수이므로 FAIL이다.

### 8.3 TRAD-FIX

- `LUMINA_TRAD_COLOR_FIX`는 폐기한다.
- 환경변수가 설정되면 무시하지 말고 obsolete-option 오류로 종료한다.
- 기존 `verify_trad_fix.py` 결과는 역사적 증거와 음성 대조로 동결한다.
- 기존 verifier의 `PASS`는 parser·census 실행 성공일 뿐 TRAD-FIX 물리 성공으로 해석하지 않는다.
- 새 최종 gate는 J-owner와 L-0∼L-8을 대상으로 한다.
- TRAD-FIX를 T-SEED 또는 다음 단계의 선행 PASS로 요구하지 않는다.

---

## 9. clamp·floor 판별 계약

판별식은 다음이다.

> 계약상 유효한 유한 정확해가 해당 guard 조건을 만족할 수 있고, guard가 그 값을 다른 물리값으로 바꾸는가?

그렇다면 금지다.

금지 대상에는 다음이 포함된다.

- raw estimator가 0일 때 `J_ν=1e-30`
- 격자 밖 조회에 `1e-30`
- `J_ν ≤ W B_ν(T_rad)` UV cap
- `W` 상한
- Planck fitting bound를 실제 복사장으로 대입
- 패킷 0개 빈을 인접 빈으로 채우기
- s44∼s49 hold
- 음의 opacity를 0으로 자르기
- 유효한 `T_e`를 임의 최소·최대로 자르기

허용 대상은 다음뿐이다.

- NaN·Inf·잘못된 단위·잘못된 frame의 명시적 거부
- 배열 범위 밖 접근의 명시적 실패
- 값을 보존하는 수치식 재배열
- 고정점을 바꾸지 않는 반복 damping
- 물리값을 바꾸지 않는 overflow 사전 판정

정확한 0은 유효값이다. “missing”, “unsampled”, “out of range”, “exactly zero”는 서로 다른 상태로 저장한다.

---

## 10. 단계별 구현 발주

모든 단계는 다음 공통 인수조건을 가진다.

- 한 시점에 `src` 편집 태스크는 하나뿐이다.
- 단계 하나가 계약 하나만 수리한다.
- 해당 단계의 새 gate가 PASS한다.
- 앞 단계에서 PASS한 모든 gate가 계속 PASS한다.
- 아직 이행하지 않은 층은 `NOT_RUN` 또는 `BLOCKED`로 남기며 거짓 PASS를 만들지 않는다.
- 고정 RNG에서 허용 목록 밖 출력 변화가 없다.
- 새 guard·fallback hit가 0이다.
- 단계마다 회귀 대장 한 행을 추가한다.

| 단계 | 단일 계약 | 단계 인수조건 |
|---|---|---|
| A2-00 | 원장 자격 | `O-WIRE`, `O-PHYS` 상태와 파일 hash manifest 확정 |
| A2-01 | 소유권 census | 157개 행 전부 disposition, 미분류 0, 런타임 read trace 확보 |
| A2-02 | 좌표·격자 | 소비 주파수 합집합과 해상도 사다리 PASS |
| A2-03 | 정본 자료형 | generation·frame·unit·validity를 가진 `RadiationField` shadow 도입; 기존 결과 불변 |
| A2-04 | 생산자 commit | MC estimator와 pure-CMFGEN 경로가 같은 commit API로 `J_ν`를 생산; Planck 재구성 overwrite 제거; L-0 replay PASS |
| A2-05 | CPU bound-free rate | 직접 `J_ν` 적분, `bf_rate_estimator` 소비 제거; L-1bf PASS |
| A2-06 | CPU bound-bound rate | `\bar J`, 상·하향률 직접 계산; L-1bb PASS |
| A2-07 | 물질 population | Boltzmann·partition은 `T_e`, ion·level solver는 새 rate 사용; L-2ion·L-2level PASS |
| A2-08 | CPU opacity | `χ_ν` 경로의 스칼라 의존 제거; L-4 PASS |
| A2-09 | CPU emissivity | `η_ν`와 재분배 경로의 Planck 표본 제거; L-3·L-5 PASS |
| A2-10 | 복사평형 | 항별 `J_ν` heating/cooling으로 `T_e` 해결; L-6 PASS |
| A2-11 | CPU formal transfer | transfer·diagnostic·output의 스칼라 의존 제거; L-7·L-8 CPU PASS |
| A2-12 | GPU 소유권·lifecycle | 정본 generation 업로드·reset·동기화와 GPU transfer 이행 |
| A2-13 | GPU rate | GPU bound-free·bound-bound rate가 CPU oracle replay와 일치 |
| A2-14 | GPU opacity | GPU `χ_ν`가 CPU와 L-4 기준으로 일치 |
| A2-15 | GPU emissivity | GPU `η_ν`·packet sampling이 CPU와 L-3·L-5 기준으로 일치 |
| A2-16 | 네이티브 J seed | 새 덱 loader와 오프라인 legacy converter; seed generation 규칙 PASS |
| A2-17 | 스칼라 제거 | CPU·GPU 구조체, 입력, update, lifecycle, output, 환경변수의 `(W,T_rad)` 제거 |
| A2-18 | 통합 인수 | `O-PHYS`에서 L-0∼L-8 전부 PASS, seed 독립성과 CPU/GPU 동등성 PASS |

A2-00∼A2-11은 offline-first로 수행한다.

- grammar-debug: parser, schema, 단위·수식 unit test, 경량 빌드
- lageunha: CMFGEN 추출, 해상도 사다리, 대형 CPU replay, CPU formal spectrum
- syn: A2-12 이후 GPU 실행만, 운전석 승인 후 사용

어느 노드에서도 `/usr/bin/time`을 사용하지 않는다.

---

## 11. 단계 회귀 대장

각 단계는 정확히 한 행을 남긴다. 필드는 다음과 같다.

```text
stage_id
contract
source_tree_hash
input_manifest_hash
oracle_id
node
command
exit_status
new_layer_status
all_previous_layer_statuses
negative_control_status
coverage
metric_values
changed_output_allowlist
guard_hits
fallback_hits
rng_seed
mc_confidence
artifact_paths
driver_signoff
```

물리적으로 바뀌어야 하는 출력은 단계 시작 전에 allowlist로 등록한다. “전체 스펙트럼이 바뀌었다”는 허용 목록이 될 수 없다. 변화는 해당 단계가 담당한 층의 observable로 귀속되어야 한다.

---

## 12. 운전석 검수 항목

운전석은 다음을 독립 검수한다.

1. 현재 CMFGEN 스냅샷을 수렴 원장으로 승격하지 않았는가.
2. s44∼s49를 hold·외삽·복사하지 않았는가.
3. 157개 census 행이 모두 최종 disposition을 가졌는가.
4. rate나 transfer 소비자가 raw estimator·`cs.J`·`j_blue`를 직접 읽지 않는가.
5. `J_ν`가 빈 평균이고 모든 producer가 동일한 `Δν`, 체적, 시간, `4π` 정규화를 쓰는가.
6. CPU와 GPU가 같은 bin edge, frame, generation을 쓰는가.
7. 격자 범위가 소비자 합집합과 25000 Å를 덮는가.
8. CMFGEN 파일들이 한 generation인지 hash manifest로 증명됐는가.
9. PRRR의 rate와 coefficient를 혼동하지 않았는가.
10. atomic-data coverage가 95% 미만인데 매칭된 일부만으로 PASS하지 않았는가.
11. CMFGEN NLTE population을 Boltzmann 정답으로 사용하지 않았는가.
12. `CHI_DATA`, `ETA_DATA` 부재를 평균 opacity나 cooling file로 대체하지 않았는가.
13. L-0∼L-8의 `CHAIN`, `ORACLE_INPUT`, 음성 대조가 각각 기록됐는가.
14. 하류 PASS로 상류 FAIL을 가리지 않았는가.
15. seed가 generation 1 이후 다시 소비되지 않는가.
16. TRAD-FIX 환경변수가 사라졌고 설정 시 명시적 오류가 나는가.
17. 금지된 clamp·floor가 0개인가.
18. 앞 단계 PASS가 다음 단계마다 재실행됐는가.
19. GPU 실행 전에 대응 CPU·offline gate가 PASS했는가.
20. 각 단계 회귀 대장이 정확히 한 행이며 운전석 서명이 있는가.

---

## 13. 무증상 실패 경로 원장

다음 경로를 구현 전에 failure injection 대상으로 등록한다.

1. raw MC `J_ν`를 정규화한 뒤 빈별 `(W,T_R)` fit이 다시 덮어쓴다.
2. `bf_rate_estimator`가 새 `J_ν` 경로를 우회한다.
3. `jbar_line` 또는 `j_blue`가 이전 iteration 값을 유지한다.
4. pure-CMFGEN의 `cs.J`와 공개 `J_ν`가 서로 다른 generation이다.
5. CPU `J_ν` 갱신 뒤 GPU 배열 업로드가 누락된다.
6. GPU reset이 CPU보다 한 iteration 빠르거나 늦다.
7. raw estimator를 셸 체적·시간·`4π`·`Δν` 중 하나로 두 번 나눈다.
8. 로그 bin의 중심값을 bin average로 가장한다.
9. 공이동 주파수와 observer 주파수를 혼합한다.
10. `J_ν=0`을 missing으로 취급하거나 missing을 0으로 취급한다.
11. 패킷이 없는 빈을 `1e-30` 또는 이웃 값으로 채운다.
12. 격자 밖 rate 질의가 작은 값으로 조용히 계속된다.
13. UV cap이 강한 CMFGEN UV장을 다시 잘라낸다.
14. legacy fixed-profile 환경변수가 새 정본을 덮어쓴다.
15. 진단용 `T_E`나 `T_C`가 Boltzmann·opacity·rate 입력으로 재사용된다.
16. LTE partition에서 `T_e`가 없을 때 옛 `T_rad`로 fallback한다.
17. re-emission이 `η_ν` 대신 Planck 분포를 표본한다.
18. seed가 첫 commit 뒤에도 rate를 먹인다.
19. 서로 다른 seed가 서로 다른 고정점에 도달하지만 한 seed만 보고 PASS한다.
20. EDDFACTOR, RVTJ, POP, OBSFLUX가 서로 다른 CMFGEN iteration이다.
21. `FINISH_REC=1`을 물리 수렴으로 오인한다.
22. 고정된 CMFGEN 이온·준위의 기여를 숨긴다.
23. `*PRRR`의 재결합 coefficient에 밀도를 잘못 곱하거나 두 번 곱한다.
24. spontaneous와 stimulated recombination을 중복 합산한다.
25. line net rate만 비교해 큰 상·하향률의 오류가 상쇄된다.
26. level index만으로 Lumina와 CMFGEN level을 연결한다.
27. CMFGEN superlevel population과 Lumina 개별 level을 그대로 비교한다.
28. 음의 opacity를 절댓값·0으로 바꿔 적분한다.
29. CMFGEN total `η_ν`와 Lumina true-emission-only `η_ν`를 의미 확인 없이 비교한다.
30. `H_ν`의 부호, `R²`, `4π` 중 하나가 빠진다.
31. `F_ν`, `F_λ`, `L_ν`, 거리 정규화가 섞인다.
32. spectrum convolution 뒤 적분 비보존 interpolation을 한다.
33. GPU FP32·TF32 오차를 물리 차이로 숨기거나 반대로 무제한 허용한다.
34. static `getenv()` 캐시 때문에 같은 프로세스의 음성 대조가 실제로 lane을 바꾸지 못한다.
35. 진단 dump가 RNG 소비 순서를 바꾼다.
36. 출력 반올림이 작은 population·rate의 부호나 0 상태를 숨긴다.
37. 모든 층을 하나의 총점으로 평균하여 특정 층 FAIL을 상쇄한다.
38. 원자자료 불일치를 `J_ν` wiring 오류로 오진한다.
39. 격자 해상도 부족을 단면적·rate 보정계수로 흡수한다.
40. scalar 심볼은 삭제됐지만 같은 압축이 이름만 바뀐 `color_temperature`, `dilution`, `radiation_fit`으로 재등장한다.

---

## 14. 최종 인수조건

다음 조건을 모두 만족해야 A-2 완료다.

- `J_ν`가 CPU·GPU·순수 CMFGEN 경로를 포함한 유일한 런타임 복사장 정본이다.
- `(T_rad,W)`는 런타임 입력·상태·소비·갱신·GPU 메모리에서 제거됐다.
- 레거시 스칼라 덱은 오프라인 변환기로만 처리된다.
- TRAD-FIX가 폐기됐다.
- 주파수 격자가 소비 범위를 완전히 덮고 해상도 수렴을 통과했다.
- 157개 census 항목의 disposition이 완료됐다.
- 금지 clamp·floor가 0개다.
- `O-PHYS`가 물리 원장 자격을 통과했다.
- L-0, L-1bf, L-1bb, L-2ion, L-2level, L-3, L-4, L-5, L-6, L-7, L-8이 각각 독립 PASS다.
- 모든 층의 음성 대조가 해당 오류를 실제로 거부한다.
- s0∼s49 전체에 CMFGEN 공간 coverage가 있다.
- CPU/GPU 결과가 각 층의 수치 허용오차 안에서 일치한다.
- 두 명시적 `J_ν` seed가 동일한 최종 해로 수렴한다.
- A2-00∼A2-18의 회귀 대장이 한 행씩 존재하고 운전석이 서명했다.

`BLOCKED`, `OUT_OF_ORACLE`, `NOT_RUN`은 PASS가 아니다. 현재 `toy06_19.48d_jnu4`만으로는 L-0 wiring 검증을 시작할 수 있지만 A-2 최종 인수는 할 수 없다.
---

## 10. 운전석 검수 결과 (2026-08-04, 자율)

**판정: 조건부 수용.** 반박 0건, 확인 4건, **운전석 신규 발견 1건(발주서에 반영 요구)**.

### 확인 (운전석 독립 실측)

| 항목 | 실측 |
|---|---|
| `J_ν` 격자 100–20000 Å | `src/lumina.h:512-514` `NLTE_N_FREQ_BINS 1000` · `NLTE_NU_MIN 1.5e14`(=c/20000Å) · `NLTE_NU_MAX 3.0e16`(=c/100Å). 검증 대역 25000 Å 를 못 덮는다는 §4 주장 **확인** |
| 격자 밖 `1e-30` 반환 | `lumina_plasma.c:9636` 주석 *"same log-grid bin, same out-of-range 1e-30"* **확인** |
| ARTIS C1 모사 배제 | §1 의 *"J_ν 를 다시 빈별 (W,T_R) 로 압축하는 ARTIS C1 모사"* 제외는 **운전석 독립 측정과 일치** (아래) |
| 11층 분해 | 운전석 초안 7층보다 세밀하고 각 분리가 타당(bf/bb · 이온/준위 · 재분배 커널 · 내부 모멘트). **수용** |

### 운전석 독립 측정 — `W(ν)` shim 을 배제하는 수치 근거

user 가 "편의를 위해 `J_ν` 와 `W(ν)` 를 함께 들고 가면?" 을 제안해 실측했다
(lageunha, EDDFACTOR 196,185 주파수 × s0 브래킷):

```
T_gauge = 10470.093 (gate 값)      W(ν) 1.237e-01 … 2.741e+142   동적범위 143.3 자릿수
                                    float32 오버플로 후보 2,184점
T_gauge = 14172.549 (덱 내재 color) W(ν) 7.469e-02 … 5.184e+78    동적범위 79.8 자릿수
                                    float32 오버플로 후보 1,122점
작업 대역만 (14172.549 게이지):
  EUV 450-918Å 1.09e1…1.62e2 · FUV 5.36…1.16e1 · UV 1.85…5.53 · OPT 0.19…2.02
```
**꼬리(λ→3 Å)에서 `B_ν` 가 언더플로하고 `J_ν` 는 안 하므로 나눗셈이 80–143 자릿수를
제조한다.** `J_ν` 자체에는 없는 병리다. 그리고 `W_ν · B_ν ≡ J_ν` 이므로 shim 이 주는
것은 텍스트 유사성뿐이다. ⟹ **`W(ν)` 를 배열로 두지 않는다는 §1·§2 결정을 지지.**

부수: 게이지를 gate 값(10470)이 아니라 덱 내재 color(14172.549)로 잡으면 동적범위가
143→80 자릿수로 준다. **gate 의 10470 이 수치적으로도 더 나쁜 선택**이라는 독립 증거.

### ★운전석 신규 발견 — `J_ν` 자체에 floor 가 있다 (발주서 미포함)

```c
src/lumina_plasma.c:14683-14688
  if (raw > 0.0 && volume[s] > 0.0 && delta_nu > 0.0)
      nlte->J_nu[idx] = raw / (4π V t Δν);
  else
      nlte->J_nu[idx] = 1e-30;   /* floor */
```
`raw` 는 MC 추정자 누적이다. **그 (셸,빈)에 패킷이 0개면 `raw=0` 이고 물리적 정답은
정확히 0 인데 1e-30 을 넣는다.** 규약 판별식("정확해가 위반 가능한 가드")에 걸린다.

⚠ **tau 의 `1e-100` 보다 해롭다**: 이 floor 는 **"표본 없음"과 "측정된 작은 값"을
구분 불가능하게** 만든다. 캠페인의 `s12+ FUV 기근 13-20×` 는 정확히 패킷 기근 대역이며,
거기서 `1e-30` 이 측정값처럼 보고되고 있다.

**발주서 수정 요구**: L-0 층 계약에 다음을 추가하라.
- `J_ν` 배열에 **`unsampled` 상태를 값과 분리해 표현**한다(별도 mask 또는 sentinel-free 카운터)
- `raw==0` 을 `1e-30` 으로 덮지 않는다
- **패킷 수 per (셸,빈) 을 함께 산출**해 L-0 비교에서 `unsampled` 를 분모에서 제외한다
- 음성 대조: 인위적으로 특정 빈의 패킷을 0으로 만들면 L-0 이 `unsampled` 로 잡아야 하고,
  `1e-30` 을 측정값으로 통과시키면 FAIL

### 수용하는 설계 판단

- **§4 해상도 사다리**(1000/2000/4000/8000/16000, 1% max·0.2% median, 실패 시 `BLOCKED`) —
  I7(Lumina 1,000 vs CMFGEN continuum 15,662)을 오프라인으로 결판내는 올바른 방법.
  *"1000 을 관성적으로 유지하지 않는다"* 수용
- **§8.2 레거시 seed 를 L-0 의 의도된 음성 대조로 규정** — 우아하다
- **§8.2 poison test**(첫 field commit 뒤 seed 파일 변경이 generation≥1 을 바꾸면 FAIL) —
  **T-SEED 를 A-2 안으로 흡수**하며 더 엄밀하다
- **§8.2 s44–s49 `valid=false`, hold 금지, 50셸 CMFGEN seed 요구 실행은 `BLOCKED`** —
  F(격자 범위)의 정직한 처리
- **§8.3 `LUMINA_TRAD_COLOR_FIX` 폐기 + 설정 시 obsolete-option 오류 종료**(무시 아님) — fail-closed
