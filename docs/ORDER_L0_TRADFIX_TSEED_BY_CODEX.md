# 발주서 TRAD-FIX/T-SEED-R3 — 복사장 입력 계약 및 온도 seed 의존성 폐합

발주일: 2026-08-04  
범위: 검증기·실험 구성·판정·증거화. 본 발주서 자체는 read-only이며 구현·제출·파일 수정은 하지 않는다.  
선후관계: `TRAD-FIX PASS → 사전등록 동결 → user 승인 → T-SEED GPU 배치 1회`.

## 1. 즉시 정정하는 사실

1. 환경변수의 정확한 이름은 `LUMINA_TRAD_COLOR_FIX`다. `T_RAD_COLOR_FIX`는 존재하는 계약명으로 쓰지 않는다.
2. 기존 §3.6의 CMF hydro `T_e` lane은 제거한다. `LUMINA_TE_TABLE`과 `LUMINA_FIXED_TE_PROFILE`은 seed가 아니라 최종 \(T_e\)와 하류 상태를 대체하는 pin이다.
3. lane 1의 ratio `1.0`과 lane 2의 ratio `0.9`는 유지한다.
4. 다만 활성 call-chain은 추가 정정한다.

   - 초기 \(T_e\)는 [lumina_cuda.cu](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:7092)의 `compute_electron_temperature(..., self_consistent=0)`에서 `ratio × T_rad`로 생성된다.
   - 캡처 설정은 `LUMINA_RADEQ_SIMUL=1`이다. 이때 [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:11600)의 SIMUL dispatch가 `nlte/J_nu == NULL` fallback보다 먼저 실행된다.
   - 따라서 이 설정에서 ratio의 확정 경로는 “iter 0·1마다 fallback 재대입”이 아니라 `초기 ratio seed → persisted T_old → RADEQ_DAMP 및 step path`다. `LUMINA_NLTE_START_ITER=2`는 NLTE solve 시작을 늦추지만 SIMUL RADEQ를 늦추지 않는다.
   - lane 1·2의 유효성은 유지되지만, GPU 전에 위 경로의 런타임 liveness를 증명해야 한다.

## 2. TRAD-FIX 선행 계약

### 2.1 현재 변환의 정량

대상 덱의 원본은 `T_rad=10470.093240 → 3133.594393 K`이고, gate는 s1–s49를 모두 s0 값으로 덮는다. W는 적재 시 변경하지 않는다.

오프라인 검증기가 반드시 재현할 사전 기대값은 다음과 같다.

- 변경 셸: s1–s49. s0은 불변.
- 최대 절대 변화: s49의 `7336.498847 K`.
- 최대 배율: s49의 `3.341240737`.
- 최대 상대 증가: `234.124074%`.
- 50셸 상대 증가 중앙값: `148.205192%`.
- `Jν=W·Bν(T_rad)` 해석의 bolometric 변화는 \((T_{\rm on}/T_{\rm off})^4\)다.

  - s10: `10.131996배`
  - s25: `39.228813배`
  - s40: `87.001843배`
  - s49: `124.632432배`

- H의 `1.2e-5` 영향과 비교하면 최대 상대변환 `2.34124`는 약 `1.95e5배` 크다. 이것은 허용 근거가 아니라 변환 규모의 비교값이다.

추가로 다음 모순을 독립 검증한다.

- 현재 파일은 모든 셸에서 `T_rad/W^0.25 = 14172.549003 K`로 일정하다.
- 같은 덱 `config.json`의 `T_inner_K`는 `10020 K`다.
- gate가 복사하는 값은 `10470.093240 K`다.
- 따라서 복사값은 파일에서 역산한 color `14172.549 K`도, 설정의 boundary color `10020 K`도 아니다.
- 현재 builder 식 `T_inner·W^0.25`에 `10020 K`를 넣으면 s0은 약 `7402.362 K`여야 하므로 현 파일과 `29.3%` 어긋난다.

이 불일치는 deck lineage 또는 의미 계약 결함으로 판정한다. “대략 같은 온도”로 봉합하지 않는다.

### 2.2 오프라인 검증 방법

H 검증기와 같은 독립 재구성 방식을 사용한다. 생산 builder나 Lumina solver를 import하거나 실행해 자기 코드를 자기 자신으로 검증하지 않는다.

신규 TRAD-FIX 검증기는 다음 네 상태를 산출한다.

1. 덱 원본 `W,T_rad`
2. 현재 gate를 독립 재구성한 `W,T_rad`
3. `T_energy/W^0.25`로 역산한 color 후보
4. 선택된 최종 계약의 `W,T_color`

각 상태에서 다음을 50셸 전부 계산한다.

- `ΔT`, 온도비, 상대차, profile의 unique count
- ratio `1.0`과 `0.9`의 실제 초기 \(T_e\)
- `W·Bν(T)`와 원본 대비 비
- bolometric 적분
- `450–918 Å`, `918–1290 Å`, `1290–2000 Å`, `2000–10000 Å`, `10000–25000 Å` 적분
- `T_rad`와 W의 모든 지속 소비처 및 소비 의미: seed, Boltzmann/partition, rate, opacity, transition probability, 비교자

NaN·Inf·비양수 온도·비양수 W는 대체하지 않고 실패한다. floor나 hold-last를 사용하지 않는다.

이 단계는 GPU 없이 결판 가능하다. 정적 해시·소규모 fixture는 grammar-debug, 143 MB급 CMFGEN field 처리는 lageunha에서 수행한다.

### 2.3 CMFGEN 독립 대조

CMFGEN에는 Lumina와 동형인 `T_rad` 변수가 없으므로 동명 열 비교를 금지한다. `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/`의 다음 실물을 사용해 대응량을 유도한다.

- `EDDFACTOR` SHA-256: `83acc14a35999aaf39cf728ce783308be31fa52676b1c5410b5e84f4cc009705`
- `EDDFACTOR_INFO` SHA-256: `2c032445a9483d5154c15cdac5c0f14dfbb3f45dbb628e2d4936c29b3efabd42`
- `RVTJ` SHA-256: `a042fd49c726dc1c2b710c997fa3d27780189e98edebc28f9d77c06ffe034f78`

검증 순서는 다음과 같다.

1. `ND=90`, 유효 주파수 record `196185`, `FINISH=1`, 주파수·단위 round-trip을 검사한다.
2. Lumina 셸 midpoint velocity를 RVTJ에 대응한다.
3. 동일 주파수에서 velocity 방향 log-J 보간을 수행한다.
4. \(u=(4π/c)\int J_\nu dν\)와 \(T_{\rm energy}=(u/a)^{1/4}\)를 구한다.
5. \(\int \nu J_\nu dν/\int J_\nu dν\)로 독립 moment color를 구한다.
6. 별도의 amplitude-free Planck-shape fit으로 \(T_{\rm color},W\)를 구하고 moment 결과와 fit residual을 함께 기록한다.
7. 원 덱과 gate-on의 `W·Bν(T)`를 CMFGEN Jν와 같은 등록 밴드에서 직접 비교한다.

CMFGEN 피복은 s0–s43이다. s44–s49는 이 snapshot에서 독립 대조 불가능하다. 외곽값을 최근접 depth로 hold하거나 clamp해서 50셸 대조로 꾸미지 않는다.

독립 대조로 다음은 결판할 수 있다.

- 원 덱 값이 energy-equivalent인지
- 현재 gate 값이 실제 color 대응량인지
- 단일 dilute-Planck scalar가 CMFGEN field를 대표할 수 있는지

다만 gate 변경의 nonlinear 최종 스펙트럼 효과는 오프라인으로 결판할 수 없다. 그 효과는 TRAD 입력 계약의 합법성을 판단하는 근거로 사용하지 않는다.

### 2.4 메모리 정본과 캡처의 모순 처분

메모리 정본의 “전셸 10470 K pin은 잣대 결함” 판정을 유지한다. 캡처 188932의 gate-on은 그 판정을 뒤집은 증거가 아니라, 정본 계약을 위반한 역사 specimen이다.

- 캡처 자체와 환경 manifest는 보존한다.
- `W·B(T_rad)`, color ratio 또는 flat-10470을 yardstick로 사용한 판정은 `INVALID_YARDSTICK`로 강등한다.
- flat `T_rad`와 무관한 결론은 자동 폐기하지 않고 의존성을 개별 재검수한다.
- 문서 한 줄 추가만으로 현 gate를 소급 합법화하지 않는다.

### 2.5 처분 후보와 귀결

1. **gate OFF만 수행**

   - 숨은 전셸 overwrite와 정본 모순은 제거된다.
   - 그러나 energy-equivalent 값을 W와 함께 color 소비처에 전달하면 dilution을 이중 적용할 수 있다.
   - 따라서 현재 코드에서 gate OFF만 한 상태는 진단 대조군일 뿐 production 폐합안이 아니다.
   - raw `T_rad`를 seed 전용으로 재정의하려면 모든 지속 color 소비처가 제거되거나 별도 필드로 분리되어야 한다.

2. **덱의 `T_rad` 계약 수정 — 우선안**

   - `T_energy`와 `T_color`를 이름·schema·provenance에서 분리한다.
   - `T_color`와 W는 반드시 같은 radiation-field 정의에서 함께 유도한다.
   - point-source 초기장이라면 `config.T_inner`, 파일에서 역산한 `14172.549 K`, 현재 gate의 `10470.093 K` 중 무엇이 정본인지 먼저 판정한다.
   - CMFGEN Jν를 정본으로 택했는데 단일 `(W,T_color)` fit이 부적합하면 억지 scalar나 outer hold를 만들지 않는다. Jν 기반 입력 또는 명시적인 근사 계약으로 승인을 다시 받는다.
   - 수정 후 runtime gate는 OFF여야 한다. 같은 companion symlink를 공유하는 `_sivcaiv`와 `_ftos`의 full-deck manifest를 각각 다시 발급한다.

3. **현행 유지 후 문서화**

   - 숨은 whole-profile pin, `10020/10470/14172 K` 불일치, CMFGEN 독립 대조 부재가 그대로 남는다.
   - 현 증거에서는 `REJECTED`다.
   - 채택하려면 메모리 정본을 명시적으로 supersede하는 독립 증거와 새 승인 기록이 필요하다. 단순 문서화는 처분이 아니다.
   - 이 경로에서 얻은 T-SEED 결과는 오직 “flat-10470 위의 seed 민감도”이며 raw deck이나 CMFGEN color로 일반화할 수 없다.

“효과가 작다”는 논거를 어느 후보에도 사용하지 않는다.

## 3. T-SEED 재설계

### 3.1 lane 정의

구 lane 3인 CMF \(T_e\) pin은 폐기한다. 새 lane 3은 측정계의 무증상 비결정성을 검출하는 baseline exact replay로 재정의한다.

두 생산 덱을 모두 배포 후보로 유지하므로 한 GPU batch 안에서 다음 여섯 run cell을 각각 별도 프로세스로 순차 실행한다.

1. `_sivcaiv`, lane A: `LUMINA_TE_TRAD_RATIO=1.0`
2. `_sivcaiv`, lane B: `LUMINA_TE_TRAD_RATIO=0.9`
3. `_sivcaiv`, lane R: lane A의 binary·deck·env·RNG exact replay
4. `_ftos`, lane A: `LUMINA_TE_TRAD_RATIO=1.0`
5. `_ftos`, lane B: `LUMINA_TE_TRAD_RATIO=0.9`
6. `_ftos`, lane R: lane A의 binary·deck·env·RNG exact replay

A/B만 seed 대조다. R은 seed lane이 아니라 재현성 음성대조다. 서로 다른 덱 간 결과에는 seed 합격선을 적용하지 않는다.

TRAD-FIX 처분에 따른 정의는 다음과 같다.

- 덱 수정안 채택: A/B/R 모두 수정된 동일 `T_color,W`, gate OFF.
- gate OFF 대조안 채택: A/B/R 모두 raw deck profile, 결과 라벨은 `RAW-ENERGY-PROFILE`.
- 현 gate 유지안: 선행 계약이 현재 증거로 FAIL이므로 GPU 제출 금지.

`LUMINA_TE_TABLE`, `LUMINA_FIXED_TE_PROFILE`, diagnostic \(T_e\) pin은 여섯 cell 모두 미설정이어야 한다.

### 3.2 실행 길이와 liveness

캡처 188932의 마지막 두 \(T_e\) commit은 s0–s43에서 최대 상대변화 약 `4.93%`, 중앙값 약 `0.281%`다. 따라서 12회 trajectory는 기존의 최대 `1%` 수렴 조건조차 충족하지 못한다.

이번 판정런은 각 cell을 고정 24 iteration으로 실행한다.

- argv iterations: `24`
- `LUMINA_PURE_CMFGEN_ITER=24`
- packets: `100000`
- `LUMINA_NLTE_START_ITER=2`
- `LUMINA_RADEQ_DAMP=0.5`
- RNG seed: `23111963`

초기 liveness 증거로 50셸 전부에서 첫 \(T_e\)가 `ratio × effective T_rad`인지 기록한다. 이후 SIMUL의 `T_old`, root status, committed \(T_e\)를 iteration별로 보존한다. “pre-NLTE fallback이 두 번 실행됐다”는 문구는 실제 branch counter가 없으면 쓰지 않는다.

기존 `LUMINA_ION_POP_DUMP_ITER` 경로를 사용해 iteration별 ion population을 남기고, charge sum으로 \(n_e\) trajectory를 독립 재구성한다. 내부 최종 `n_e`와 round-trip이 맞지 않으면 판정하지 않는다.

### 3.3 합격선 재산정

기존 `최대 1% / 중앙값 0.2%`는 최종 A/B 물질적 동등성 기준으로 유지한다. 다만 수렴 판정으로는 불충분하므로 단독 사용을 금지한다.

상대차는 모든 양수 \(x,y\)에 대해 다음 symmetric 정의를 쓴다.

\[
\delta(x,y)=\frac{2|x-y|}{|x|+|y|}.
\]

각 덱별 합격 조건은 모두 충족해야 한다.

1. **측정 재현성 A 대 R**

   - 최종 \(T_e\), \(n_e\): 50셸 최대 `≤0.1%`, 중앙값 `≤0.02%`
   - discrete manifest, root-status 배열, pin/floor counter는 동일
   - 이를 넘으면 stochastic 또는 GPU 비결정성과 seed 효과를 분리할 수 없으므로 `UNRESOLVED`

2. **개별 lane 수렴**

   - 마지막 세 commit에서 연속한 두 step 모두 검사
   - \(T_e\), 재구성 \(n_e\): 각 step의 50셸 최대 `≤0.5%`, 중앙값 `≤0.1%`
   - 마지막 세 commit에서 모든 셸 `root-found`
   - HOLD, pin-low, pin-high, 1000 K floor, NaN, Inf는 0건

3. **seed 독립성 A 대 B**

   - 최종 \(T_e\): 50셸 최대 `≤1%`, 중앙값 `≤0.2%`
   - 최종 \(n_e\): 50셸 최대 `≤1%`, 중앙값 `≤0.2%`
   - root-status와 금지 guard hit 분포가 동일
   - s44–s49도 seed 통계에는 포함한다. 이 셸들이 제외되는 것은 CMFGEN 절대대조뿐이다.

4. **CMFGEN oracle 보고**

   - s0–s43에서 각 lane의 최종 \(T_e,n_e\)와 CMFGEN 차이를 별도 보고한다.
   - A/B가 서로 같아도 둘 다 CMFGEN과 다를 수 있다. 이 경우 `T_SEED_INDEPENDENT=yes`, `PHYSICAL_T_ACCURACY=UNRESOLVED`로 분리한다.
   - CMFGEN 절대 일치를 T-SEED 합격 조건으로 몰래 추가하지 않는다.

### 3.4 clamp/floor 규율

GPU 전에 활성 call-chain의 모든 clamp·floor·hold를 전수 분류한다.

- 정확해가 위반할 수 있으면 수치 guard가 아니라 해 변경이므로 금지한다.
- `LUMINA_HRESP_CLAMP`처럼 trial physical response 자체를 자를 수 있는 guard는 비활성화하거나 제거해야 한다.
- `LUMINA_NLTE_INV_CEIL`처럼 정확한 population ratio가 넘을 수 있는 ceiling도 같은 판정을 적용한다.
- `LUMINA_SIMUL_CAP_TOPION`은 선언된 state-space에서 top rung을 제외하는 모델 경계인지, 계산된 해를 사후 절단하는 cap인지 판정한다. 후자면 금지한다.
- `LUMINA_TE_STEP_CLAMP`는 fixed point에서 \(\Delta T=0\)이므로 고정점을 옮기지 않는 iteration limiter로 둘 수 있다. 다만 hit 수와 셸을 lane별 기록한다.
- 금지 guard를 끈 뒤 불안정해지는 것은 실패 결과이지 재활성화 사유가 아니다.

## 4. GPU 배치 1회가 결판내는 것

단일 Slurm GPU job에서 여섯 별도 프로세스를 실행한다. 한 프로세스 안에서 env만 바꾸며 lane을 재사용하지 않는다. static-cached `getenv`가 앞 lane 값을 보존할 수 있기 때문이다.

이 배치로 결판나는 것은 다음 세 가지다.

1. `_sivcaiv`에서 ratio `1.0` 대 `0.9`의 seed 의존성
2. `_ftos`에서 ratio `1.0` 대 `0.9`의 seed 의존성
3. 각 덱에서 exact replay 비결정성이 seed 합격선보다 충분히 작은지

다음은 이 배치로 결판내지 않는다.

- pin 대 RADEQ solve
- TRAD-FIX의 의미적 정당성
- `_sivcaiv`와 `_ftos`의 원자데이터 동등성
- CMFGEN과의 절대 물리 정확성
- 실패 후 기준을 바꾼 후속런

## 5. 사전등록

GPU 제출 전 단일 preregistration record를 동결한다.

### 5.1 binary·source

- 사용 binary SHA-256: `fc622412c032ab84e776187dd0ca063e386913c8c61ab4de97f820bf94031a4d`
- 루트 구 binary `1c11daee…a2f18`은 사용 금지. 음성 계보로만 기록한다.
- binary의 절대경로는 해시로 실재 확인한 뒤 등록한다. 존재하지 않는 binary 이름을 추정하지 않는다.
- source HEAD: `47bfa200`
- 미커밋 diff: `20327행`, seal `bb1d753a6d6dd3da`
- 위 짧은 seal의 생성 알고리즘과 full digest를 함께 기록한다.
- binary가 이 source seal에서 빌드됐다는 build attestation이 없으면 GPU 제출을 막는다.

### 5.2 deck

다음 두 덱을 별도 specimen으로 등록한다.

1. `data/tardis_reference_toy06_19p48d_sivcaiv`
2. `data/tardis_reference_toy06_19p48d_sivcaiv_ftos`

각 specimen에는 symlink 자체와 resolved target을 분리한 full file manifest SHA를 붙인다. 최소한 다음을 포함한다.

- `config.json`
- `geometry.csv`
- `plasma_state.csv`
- `electron_densities.csv`
- `abundances.csv`
- deposition
- line list·levels·sigma
- tau·transition probabilities
- `kshape_contract.txt`

현재 공통 companion anchor는 다음과 같다.

- `plasma_state.csv`: `45ccb86b…a336a6f4`
- `config.json`: `cf61ab7c…81293b`
- `geometry.csv`: `21bb9349…3ce3d6`

TRAD-FIX로 companion을 수정하면 위 SHA를 재사용하지 않는다.

### 5.3 환경 manifest

기준 env는 `docs/PARITY59_INSTR.env`, SHA-256 `3ee3a446…392594a`다. 다음 차이만 허용하고 전부 명시한다.

1. lane별 `LUMINA_TE_TRAD_RATIO`
2. 선행 계약이 정한 `LUMINA_TRAD_COLOR_FIX` 상태
3. `LUMINA_PURE_CMFGEN_ITER=24`
4. iteration별 ion-pop observer
5. lane별 유일 출력경로
6. clamp/floor 감사에서 금지 판정된 guard의 비활성화

나머지 물리 env 차이는 제출 차단이다. 미설정 변수도 manifest의 일부다.

### 5.4 실패 시 행동

- TRAD 오프라인 검증 실패: GPU 미제출, `TRAD_FIX=UNRESOLVED`
- A/R 재현성 실패: `T_SEED=UNRESOLVED_NONDETERMINISTIC`
- lane 수렴 실패: `T_SEED=UNRESOLVED_NOT_CONVERGED`
- A/B 모두 수렴하고 합격선 초과: `T_SEED=DEPENDENT`
- A/B 합격: 해당 덱에 한해 `T_SEED=INDEPENDENT`
- 한 덱만 실패: 실패 덱은 production 후보에서 제외하거나 별도 승인 전까지 배포 차단
- 결과 확인 후 threshold·셸·iteration 수·CMF 피복을 변경하지 않는다.
- 자동 재제출은 하지 않는다.

## 6. 실행 자산 규율

- grammar-debug: 해시, 정적 consumer/guard census, fixture, prereg 검수
- lageunha: CMFGEN Jν 오프라인 파싱·적분
- syn Slurm: 승인된 GPU batch 1회
- 전용 T-SEED batch artifact는 제출 전에 실제 파일로 생성·검수한다. 존재하지 않는 wrapper 이름을 발주서에서 추정하지 않는다.
- `/usr/bin/time`은 사용하지 않는다.
- elapsed/resource 정보가 필요하면 Slurm accounting과 프로그램 자체 footer만 사용한다.
- user의 명시적 GPU 승인 전에는 제출하지 않는다.

## 7. 운전석 검수 항목

1. 변수명이 정확히 `LUMINA_TRAD_COLOR_FIX`인가.
2. SIMUL 선행 dispatch 때문에 fallback 설명이 조건부라는 점이 반영됐는가.
3. ratio seed의 초기화와 persisted-state 전파가 런타임으로 증명됐는가.
4. `10470`, `14172.549`, `10020 K`의 모순이 처분됐는가.
5. gate가 s1–s49만 바꾸고 적재 시 W를 바꾸지 않는다는 것이 독립 재현됐는가.
6. 이후 iteration에서 W가 바뀌는 현상과 load-time 변환을 섞지 않았는가.
7. CMFGEN 대조가 동명 `T_rad`가 아니라 Jν 유도량을 사용했는가.
8. s44–s49를 CMFGEN 최근접 depth로 hold하지 않았는가.
9. 메모리 정본을 캡처가 조용히 supersede하지 않았는가.
10. 구 CMF-\(T_e\) pin lane이 완전히 제거됐는가.
11. 새 lane R이 exact replay이며 다른 물리변수를 바꾸지 않았는가.
12. 여섯 run cell이 각각 독립 프로세스·독립 출력경로를 쓰는가.
13. binary가 `fc622…31a4d`이고 루트 구 binary가 아닌가.
14. 두 deck의 full manifest와 K-shape 계약이 모두 일치하는가.
15. env 차이가 사전등록된 항목에 한정되는가.
16. 정확해가 위반 가능한 clamp/floor가 비활성화됐는가.
17. 12회 캡처를 수렴한 T-SEED 증거로 재사용하지 않았는가.
18. A/R 재현성, 개별 수렴, A/B seed 차이를 순서대로 판정했는가.
19. CMFGEN 절대오차와 seed 독립성 결과가 별도 필드인가.
20. 실패 뒤 자동 재제출·기준 변경이 없는가.
21. 존재하지 않는 binary·검증기·wrapper를 실행 명령에 넣지 않았는가.
22. `/usr/bin/time` 호출이 없는가.

## 8. 놓칠 수 있는 무증상 실패 경로

- `T_RAD_COLOR_FIX` 오기로 gate가 실제로는 꺼져 있는데 켜졌다고 기록하는 경우
- config·env·argv 우선순위 때문에 manifest 값과 effective 값이 다른 경우
- deck 수정 후 gate도 계속 ON이라 새 profile을 다시 평탄화하는 경우
- `_sivcaiv`와 `_ftos`가 공유하는 symlink target 하나를 바꿔 양쪽이 함께 변하는 경우
- `plasma_state.csv`만 해시하고 resolved symlink target을 누락하는 경우
- W는 load-time에 불변이지만 후속 owner가 갱신한 W를 변환 결과로 오인하는 경우
- CMFGEN `T_e` 또는 RVTJ gas temperature를 color temperature로 오인하는 경우
- line-blanketed Jν에 부적합한 단일 Planck fit을 강제로 성공 처리하는 경우
- CMFGEN 외곽 미피복 셸을 hold-last로 조용히 충전하는 경우
- 순차 lane을 한 프로세스에서 실행해 static-cached env가 앞 lane 값을 유지하는 경우
- `LUMINA_BIN` 문자열만 새 binary이고 실제 실행 ELF는 구 binary인 경우
- lane 출력이 같은 파일명을 사용해 뒤 lane이 앞 lane 증거를 덮는 경우
- stdout의 반올림 \(T_e\)만으로 정밀 threshold를 계산하는 경우
- iteration별 ion population과 \(T_e\)가 서로 다른 generation인데 같은 상태로 조인하는 경우
- A/B의 RNG seed 또는 GPU가 달라 seed 효과와 MC 분산이 섞이는 경우
- 동일 RNG라도 CUDA atomic 순서가 달라지는 비결정성을 lane B 효과로 오인하는 경우
- `tee`나 후처리 성공코드가 solver 비영 종료를 가리는 경우
- lane A가 미수렴하고 lane B와 우연히 가까운 것을 seed 독립으로 판정하는 경우
- 두 lane이 같은 잘못된 attractor에 수렴한 것을 CMFGEN 정확성으로 확대하는 경우
- exact-solution-violable guard가 양 lane에서 같은 횟수 hit했다는 이유로 합법화되는 경우
- `LUMINA_TE_TABLE` 또는 fixed profile이 빈 경로·불완전 파일로 “ignored”된 것을 정상 negative control로 세는 경우
- 24회 실패 후 결과를 보고 iteration 수나 threshold를 바꾸어 같은 계약의 후속런으로 부르는 경우

최종 대장에는 `TRAD_FIX`, `_sivcaiv:T_SEED`, `_ftos:T_SEED`, `PHYSICAL_T_ACCURACY`를 각각 별도 필드로 기록한다. TRAD-FIX가 먼저 PASS하지 않으면 T-SEED GPU batch는 제출할 수 없다.
---

## 9. 운전석 검수 결과 (2026-08-04, 자율)

**판정: 조건부 수용.** 반박 0건, **미검증 1건**, 확인 6건. 운전석 자기정정 1건.

### 확인 (전부 운전석 독립 실측)

| 항목 | 실측 |
|---|---|
| **캡처 `LUMINA_RADEQ_SIMUL=1`** | 확인. `lumina_plasma.c:11607-11611`의 `if (g_simul_on) { radeq_simul_all(...); return; }` 가 `:11627` fallback 앞에서 return ⟹ **운전석이 근거로 삼은 pre-NLTE fallback 은 이 설정에서 죽은 가지** |
| **진짜 seed 경로** | `lumina_cuda.cu:7092` `compute_electron_temperature(&plasma, NULL, t_exp, n_shells, 0)` → `:2996-2999` `!self_consistent` 분기 → `T_e[s] = ratio × T_rad[s]`. **초기화 1회 seed + damped 진화**(`radeq_simul_all:9875` `radeq_damp=0.5`) |
| **TRAD-FIX 선행 사유** | seed 가 `ratio × T_rad` 이고 그 `T_rad` 를 gate 가 대체하므로 **선행 확정** |
| **3중 모순** | `T_rad/W^0.25 = 14172.549003 K` 전 셸 일정(max−min<1e-6) · gate 복사값 `10470.093240 K` · `config.json T_inner=10020.0` → `T_inner·W[0]^0.25 = 7402.362 K` vs 파일 `10470.093` = **차 29.3%**. 셋 중 어느 것도 아니다 |
| **gate 발화 실물** | `[TRAD-COLOR-FIX] T_rad[s>=1] := T_rad[0]=10470 K (W unchanged)` |
| **★산출물 확증** | 캡처 최종 `lumina_plasma_state.csv` 의 `T_rad` 가 **전 50셸 10470.093, flat=True**. 덱의 10470→3134 프로파일이 실제로 소멸했다 |

### ★미검증 1건 — §3.2 의 런 길이 근거

발주서 §3.2: *"캡처의 마지막 두 T_e commit 은 s0–s43 에서 최대 상대변화 약 4.93%,
중앙값 약 0.281%"* ⟹ **운전석이 재현하지 못했다.**

```
캡처 디렉터리 63파일 중 iteration 별 per-shell T_e 덤프 없음
  lumina_plasma_state.csv = 최종 상태 1개뿐
  chieta_iter10.manifest.json = χ/η (iteration:10), T_e 아님
stdout 배너는 3셸만: iter10→11
  T_e[0]  21226→21228  +0.009%
  T_e[25]  8410→ 8476  +0.785%
  T_e[49] 13023→13052  +0.223%
  ⟹ 배너 기준 최대 0.785%
```
**수정 요구**: 4.93%의 산출 근거(파일·명령)를 명시하거나, 재현 불가하면
**런 길이를 그 수치로 정하지 말 것.** 근거 없는 수치로 GPU 배치 길이를 정하면
비용이 그 오차에 비례한다. 대안: 배너 3셸 기준 + 판정런에서 per-shell 이터 덤프를
켜서 수렴을 **런 자체가 증명**하게 한다.

### 운전석 자기정정

앞서 운전석은 *"`LUMINA_TE_TRAD_RATIO` 는 `:11631` 의 pre-NLTE seed"* 라고 보고했다.
**틀렸다.** `RADEQ_SIMUL=1` 이면 그 줄에 도달하지 않는다. 함수를 위에서 아래로 읽으며
**앞선 dispatch 가 short-circuit 하는지 확인하지 않았다** — 어젯밤 하니스 오류
(계측기가 코드에 닿는지 미확인)와 같은 부류. Codex 가 잡았다.

lane 1/2 의 유효성 결론 자체는 유지된다(기전이 다를 뿐 seed 는 실재).

### 수용하는 설계 판단

- **§2.4 모순 처분**: 메모리 정본의 "flat 10470 = 잣대 결함" 유지, 캡처를 **위반 specimen**
  으로 규정, 해당 잣대 기반 판정을 `INVALID_YARDSTICK` 로 강등 — 소거 단조성에 부합
- **§2.5 후보 3**(현행 유지+문서화) `REJECTED` — H 의 `floor=REMOVE` 선례와 일관
- **§3.1 6 run cell**(2덱 × lane A/B/R): lane R 재현성 음성대조 신설은 **운전석 3-lane 안보다 낫다.**
  측정계 자체의 비결정성을 잡는다
