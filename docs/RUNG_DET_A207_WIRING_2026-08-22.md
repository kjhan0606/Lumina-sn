# 단 사전등록 — DET-A207-WIRING: A2-07 population 공급 배선 감사 = L6 분기 B null 의 귀속 (판정런 0회) (2026-08-22)

저자 = Fable (분담 개정14: 사전등록=Fable). 발주서 앞겹 = **이 문서 원문 그대로**(재서술 금지).
갈림길 평가·후보 기각 사유 포함. HEAD `8c2aace`, branch `thenmc-macroatom-fluorescence`.
본 문서의 소스 인용은 전부 **HEAD `8c2aace` blob 기준**이며(작업트리 dirty 는
`validation/a2_09,10/*.json` 2건뿐 — 저자 실측 `git status`), 봉인 런루트
(`sprim_l6_20260821T054111Z_probe`·`idseal_20260820T044703Z_a209`)는 **read-only 로만** 접근했다.

## 0. 갈림길 판단 (겹 1 — 사전등록 저자 요약)

1. **표적의 실질은 받되, 질문을 정정한다.** L6 판정(`docs/VERDICT_DET_SPRIM_L6_2026-08-22.md`
   §8-1)이 등록한 후보는 "A2-07 배선 감사 — candidate solve 가 선 복사장(J̄)을 소비하는
   배선인가"다. 저자의 오프라인 실측(§3)은 이 질문이 **잘못된 전제** 위에 서 있음을 보였다:
   297행(전부 Z∈{26,27,28}·ion=3 = 분광 IV)은 **애초에 A2-07 solve 의 권한 밖**이다 — 봉인
   구성(기본 31슬롯 NLTE 테이블)에 ion IV 슬롯이 없어, 이 행들의 물질은 solve 가 아니라
   **LTE-at-핀-T_e 경로가 구성상 공급**한다. "solve 가 J̄ 를 소비하는가"를 이 행들로 물으면
   답은 solve 와 무관하게 항상 "미반영"이다.
2. **채택: `DET-A207-WIRING`** — 계약을 "L6 분기 B null 의 물질 경로 귀속"으로 잡는다.
   기전은 이미 오프라인으로 특정됐고(§3, 전부 파일:행 실측), 남은 일은 그 특정을 기계
   게이트+음성대조로 폐합하는 것이다. **판정런 0회** — FINDING_NOBRACKET 전례(런 0회·오라클
   무의존)와 같은 계급이다. src 접촉 0 ⟹ **V5 원상 유지, 권한 요청 없음.**
3. 기각한 대안은 §2-2. 요지: 즉시 STAGE4 승격 런(귀속 전 구성 변경 = 원인 규명 원칙 위반),
   커버 이온 재선정 런(이번 null 이 미귀속으로 남음), 문자 그대로의 (a) 검사(틀린 전제를
   계약에 박음) — 전부 이 단 **뒤**의 갈림길이다.

## 1. 계약 (하나)

> **DET-SPRIM-L6 분기 B 의 null("iter1 생산자 297선 전 행 S=B(10020 K), 준위비 기계정밀도
> 동결")이 어느 물질 공급 경로의 귀결인지를 봉인 산출물+HEAD 소스만으로 귀속하고,
> 그 귀속을 기계 게이트(음성대조 포함)로 폐합한다.**

수리 아님·런 0회·src 접촉 0·물리값 무접촉·봉인 무변조(read-only). 귀속이 폐합되면 L6 §5-2
후보 가설 (a)/(b)/(d)의 처분이 결정되고, 다음 갈림길(§7-1)의 입력이 확정된다.

## 2. 왜 이 형상인가

### 2-1. 판정런 0회 판단

offline-first 3요건 중 ①(기전 특정)은 §3 이 이 세션에서 이미 실측으로 마쳤다. ②(수리안
검증)는 해당 없음 — 이 단은 수리가 아니라 감사다. ③(기대치 사전등록)은 §6. **필요한 모든
증거가 이미 존재한다**: 봉인 stderr/stdout/RUN_FOOTER(다시 만들 수 없는 증거, 변조 금지)와
HEAD 소스. 새 런은 새 정보를 0 만큼 더한다 — FINDING_NOBRACKET 이 같은 방식으로 귀속을
닫은 전례가 있다(런 0회). 게이트의 음성대조 의무는 **스크래치 사본 주입**으로 이행한다
(봉인 원본 불접촉).

### 2-2. 기각한 후보

| 후보 | 기각 사유 |
|---|---|
| **문자 그대로의 (a) 검사** ("solve 가 J̄ 미소비인가") | 전제 오류: 297행은 solve 권한 밖(§3-2·§3-3). 게다가 solve 의 생산 bb 조립은 A2-06 checked line view 를 **소비한다**(`src/lumina_plasma.c:17401-17425` production split, `nlte_bb_jbar_canonical` `:583`) — 커버된 이온에 한해. 이 질문을 이 데이터로 물으면 참·거짓 어느 쪽도 solve 에 대해 증명하지 못한다 |
| **즉시 STAGE4 승격 런** (`LUMINA_NLTE_STAGE4=1` 로 ion IV 를 NLTE 화) | 귀속 폐합 전 구성 변경 = 수정-런-관찰 루프. null 의 원인을 대장에 못 박기 전에 표적을 바꾸면 이번 297행 계측이 미귀속 고아가 된다. 승격 런의 타당성·비용은 이 단 착지 후 갈림길(§7-1) |
| **커버 이온(Fe II-III 등)으로 L6 재선정 런** | 같은 이유. 또한 재선정은 A2-10 선정 핀(`TARGET_ION`)의 계약 변경이라 별도 사전등록 사안 |
| **L6 재판정(분기 B 문면 파기)** | 불요 — 분기 B 문면("population 공급이 복사장 미반영")은 이 행들에 대해 **문자 그대로 참**이다. 틀린 것은 후보 가설 (a)의 일반화("candidate solve 가 미소비")이지 판정이 아니다. L6 판정·감리는 후보를 후보로만 등록했다(§3 "함의의 지위") — 골대는 서 있다 |

### 2-3. 커밋 규율

사전등록 커밋(본 문서) → 검증기 커밋 **1개**(`scripts/` 1파일 + `validation/` 산출) →
판정문 커밋. 검수·판정은 커밋 접촉 파일과 §4 표의 1:1 을 확인한다.

## 3. 기전 오프라인 특정 (전부 이 세션 저자 실측; 추측은 [추정] 표기)

### 3-1. 선정은 처음부터 분광 IV 로 핀되어 있다

봉인 env `LUMINA_A210_LINE_SATURATION_TARGET_ION=3`(RUN_FOOTER, IDSEAL 계승) ⟹ A2-10
후보/선정 행은 전부 ion=3(분광 IV). L6 판정 G3 실측: 297행 전건 Z∈{26,27,28}·ion=3·shell 0.
행 예: `[A2-10][LINE-SATURATION-ROW] … line=233521 Z=27 ion=3 ion_label=4 …`(봉인 stderr).

### 3-2. 봉인 구성에는 ion IV 의 NLTE 슬롯이 없다 (env → 소스 → 화석 3중 실측)

- **env**: 봉인 RUN_FOOTER 의 `LUMINA_` 90행 전수에 `LUMINA_NLTE_STAGE4`·
  `LUMINA_NLTE_ELEMENT_WIDE`·`LUMINA_SUPER_LEVELS` **0건**(부재).
- **소스**: `nlte_stage4_enabled()`(`src/lumina_plasma.c:10312-10320`)는 env 미설정 시 0;
  슬롯 테이블 선택(`:16482-16490`)은 기본 `NLTE_TARGET_Z/ION`(`:9791-9798`) — ion 값 집합
  {0,1,2}, **3 부재**. ion=3 을 가진 테이블은 `NLTE_TARGET_ION4`(`:9807-9814`, STAGE4 전용)와
  `NLTE_TARGET_ION_EW`(`:9820-9827`, EW 전용)뿐인데 둘 다 이 런에서 비활성.
- **화석**: 봉인 stdout:204+ 슬롯 목록 — `Z=26 ion=1: 2599 / ion=2: 1500`,
  `Z=27 ion=1: 2558 / ion=2: 3214`, `Z=28 ion=1: 1000 / ion=2: 1000` — **ion=3 슬롯 0건**;
  `Super-levels: off (identity)`; `Lines mapped to NLTE ions: 1777859 / 2588798`
  (미매핑 810,939선).

⟹ **297행의 이온은 이 런의 NLTE 해 집합 밖이다.**

### 3-3. unmapped 선의 물질 공급 경로 — J̄ 가 입력될 자리가 없다

**(A) 후보(candidate)의 τ/S 생산** — `nlte_population_candidate_produce_tau_source`
(`src/lumina_plasma.c:20776`)의 3단:

1. `nlte_writeback_ion_stage`(`:20797`; 정의 `:3381`) — 이온화 급 갱신.
   **행별 공통모드 인자의 유일한 이동원**이 여기다.
2. `compute_tau_sobolev`(`:20809`; 정의 `:3219`) — **전 선**의 τ 를
   `build_lte_level_density_cache`(`:3226-3228`) = **LTE 준위밀도(공시 T_e) 캐시**로 재생성
   (폴백 경로도 `POP_LINE_VIEW_LTE_TE`, `:3333-3343`). 주석이 명시: 사유는 "unmapped 활성
   선이 낡은 T 에 남는 것"의 방지 — 즉 unmapped 선은 **설계상 이 LTE 경로 소유**다.
3. `nlte_update_tau_sobolev_with_authority`(`:20821`; 정의 `:19499`) — NLTE 값 덮어쓰기는
   `if (element_inactive || !authority.mapped) continue;`(`:19568`) — **unmapped 선 셀
   불접촉**. 단 그 전에 전 선의 `line_source_S` 를 0·`A208_UNSAMPLED` 로 리셋(`:19556-19566`)
   ⟹ 소비자는 B(T_e) 를 대체 사용(같은 함수 주석 `:19577-19588` 이 소비자 3곳을 명시).
   `authority.mapped` 는 `nlte_tau_line_authority`(`:8902-8931`)가 `global_to_nlte_level<0`
   (ion IV 준위는 projection 밖)에서 0 으로 둔다.

**(B) 생산자(R6 fine)의 n_upper 소비** — 캡처된 `producer_eta = n_u·A_ul·hν/4π` 의 n_u:
`sobolev_upper_population_cache`(`src/lumina_cmfgen.c:6258-6268` 구축, `:6373` 소비) ←
콜백 `lumina_line_upper_population_fill_for_tau`(`src/lumina_plasma.c:9019`; 바인딩
`src/lumina_cuda.cu:8006`·`src/lumina_main.c:387`) → `a209_upper_population_for_tau`
(`:8953`): `use_nlte` 판별(`:8963-8968`, `nlte_tau_line_uses_nlte_by` `:8945-8951`)이
unmapped ⟹ 0 이고, 분기 `if(!use_nlte&&lte_level_density)`(`:8973`) — **LTE 캐시 값 반환**.
A2-10 행 조립(`:9243`)·radeq 프로브(`:14945`) 등 n_upper 소비자 전부가 같은 함수로 수렴한다.

⟹ 297행의 물질 입력은 **T_e(핀 10020)·n_ion(세대 의존)·partition 뿐**이다. J̄ 는 이 경로의
어느 인자도 아니다.

### 3-4. J̄ 소비는 실존한다 — 다만 딴 곳에

SE 조립의 production split(`src/lumina_plasma.c:17401-17425`)은 legacy 모드 산술을 덮어쓰고
**A2-06 checked line view 만** 소비한다(`nlte_bb_jbar_canonical` `:583-…`;
`nlte_solve_all_impl` 전제조건 `:19777-19790` 이 line view OK·세대 정합을 강제). 그러나 이
행렬은 **NLTE 슬롯 쌍에만** 조립된다 — ion IV 는 슬롯이 없으므로 이 행렬에 그 준위가 없다.
참고: `LUMINA_NLTE_JBAR_POPS`(legacy MC-jbar 모드)는 봉인 env 에 **부재**(mode 0)이며,
발주서 증거 B 대로 `LUMINA_NLTE_START_ITER` 는 DET 경로를 게이트하지 않는다 — (c) 소거 유지.

### 3-5. 관측된 4서명 전부가 이 경로의 예측이다

| L6 판정 실측 (§5-1) | 이 경로의 예측 |
|---|---|
| (iii) 갱신 실재 — 297행 전부 η·τ 변화(최대 +4.05e-3), gen 2→3 커밋 | `nlte_writeback_ion_stage` 가 n_ion 을 갱신한다 — LTE 캐시의 곱 인자 |
| (iv) 갱신은 공통모드뿐 — \|(η₁/η₀)/(τ₁/τ₀)−1\| ≤ 6.7e-16, 준위비 Boltzmann(10020) 동결 | η∝n_u, τ∝(n_l−gf·n_u) 둘 다 같은 n_ion 인자 × 고정-T Boltzmann 인자 ⟹ 비는 대수적 항등으로 불변 |
| (v) 신호 실재·응답 부재 — J̄/B 최저 0.607 에도 S/B=1−7e-6 | J̄ 는 이 경로의 입력이 아니다 — 응답할 배선이 없다 |
| iter1 S/B 행별 값 = iter0 LTE 시드 서명과 ulp 일치 | 두 반복 모두 같은 핀 T=10020 의 같은 LTE 산법 ⟹ 행별 서명(준위에너지 vs hν 불일치의 함수 [추정 — §5 W5 관측]) 동일 |

또한 FINDING_NOBRACKET(런 0회)의 사슬이 확장된다: LTE 시드의 S=B 는 **반복 0 의 성질**이
아니라, 이 구성의 ion IV 행에서는 **모든 반복의 고정물**이다 — 반복을 아무리 돌려도 이
행들로는 이탈이 관측될 수 없다.

### 3-6. L6 §5-2 후보 가설의 처분 (이 단이 폐합 시)

- **(a) 배선**: 문면("candidate solve 가 선 복사장을 소비하지 않음")은 **기각** — solve 는
  소비한다(§3-4). 실체는 **(a′) 커버리지**: 측정 표적 이온이 solve 의 권한 밖.
- **(b) 충돌지배**: 이 297행에 대해 **도달하지 않음(moot)** — 준위비는 어떤 rate 방정식도
  거치지 않고 산법(LTE 캐시)이 결정한다. 감리가 정정한 "압착"(반복 간 미분 중앙 3.2e-14)의
  정체는 물리적 ε 이 아니라 §3-5 (iv) 의 대수적 공통모드 불변이다. shell 0 물리에서 ε≈1
  인지 여부는 **이 단이 판정하지 않는다**(커버 이온에서 별도 질문).
- **(c) 반복 게이트**: 발주서 증거 B(운전석 실측)로 이미 소거 — 이 단은 결과만 인용한다.
- **(d) 미지**: (a′) 폐합 시 소멸.

### 3-7. 교차 화석 (게이트 하중 0 — 정합 관측)

- 봉인 stderr `[A2-09][LINE-OWNER-FORENSIC] phase=REQUESTED_TE shell=0 …
  nlte_se_fraction=0.99999998 … lte_unmapped_emit=7.0599862124704041e-08
  lte_unmapped_fraction=1.5030075833239562e-08` — 코드 자신의 소유권 장부가 shell 0 선
  방출의 사실상 전부를 NLTE-SE 소유(Fe/Co/Ni II-III 등)로, 극소 잔여를 **unmapped-LTE
  버킷**으로 분류한다. ion IV 는 그 극소 버킷이다(stdout tau census: Z=26/27/28 ion=3
  tau_sum 1.65e-4/1.33e-4/6.7e-6 — 점유 0.0%).
- [추정 — 미검증] L6 SUMMARY 총가중 1.4417e-7 과 lte_unmapped_emit 의 정확한 산술 관계
  (rate_factor 곱·4π 규약)는 확인하지 않았다. 관측으로만 병기한다.

### 3-8. 계약 결함 계급의 기재 (대장 후보 — 조용한 기재)

L6 사전등록 §6 분기 A("STAGE-1 목적 달성 실증")는 이 선정(TARGET_ION=3)+이 구성(base
테이블)에서 **구성상 도달 불능**이었다 — ion IV 행은 어떤 NLTE 응답도 보일 수 없으므로
분기 B 가 유일한 실질 착지였다. 게이트·계측·판정은 전부 유효하나, **분기 설계가 측정
표적의 권한 범위를 검사하지 않았다**(잣대 교훈: "측정 행이 피측정 기전의 권한 안에 있는가"
를 분기 전제에 넣을 것). G7 문면 결함(부록 B)과 같은 "사전등록 위생" 계급으로 기재한다 —
판정 파기 사유 아님.

## 4. 기대 변경집합 (이 목록 밖 변경 = 실패) + V5

**src/ 접촉 0 · tests/ 0 · env 0 · 런처 0 · 덱/`/gpfs` 정본 불변 · 봉인 런루트 무변조.**
⟹ **V5 원상 유지 — source edit·K-final 권한을 요청하지 않는다.** 감사는 계측·판독으로
족하다는 원칙 그대로다.

### scripts/ — 검증기 1파일 (신설)

| 파일 | 변경 |
|---|---|
| `scripts/audit_a207_wiring_l6.py` | 단일 검증기(신설). 고유 표지 심볼 `A207_WIRING_L6_ATTRIBUTION` 을 파일 내 정의(변경집합 게이트의 앵커). 기능: ① 봉인 RUN_FOOTER/stdout/stderr 파서(read-only; 행 브래킷 규칙은 L6 사전등록 §3-3 의 R7 마커 규칙 그대로) ② `git show 8c2aace:<path>` 로 HEAD blob 에서 배선 앵커·테이블 파스(작업트리 불신) ③ W1~W4 기계 게이트 + W5 관측 ④ verdict JSON 산출 ⑤ `--selftest` 에 NC-W1~W4 내장(전부 스크래치 사본 주입 — 봉인 원본 불접촉) |

### 변경집합 끝

- 기타 산출(패턴 밖): `docs/RUNG_DET_A207_WIRING_2026-08-22.md`(본 문서),
  `validation/det_stage12/a207_wiring/`(verdict JSON·보고서), 판정문.
- 실행 환경: **grammar-debug**(nested ssh) — 로그인 노드 연산 금지. h5 덱 판독(W5 관측)은
  grammar-debug 의 h5py 3.15.1(저자 실측 실재)로.

## 5. 게이트 표 (각 행: 요구 / 증거 / ★음성대조 — 전부 오프라인, 런 0회)

| # | 요구 (기계 판정식) | 증거 | ★음성대조 |
|---|---|---|---|
| **W1 구성 화석** | 봉인 RUN_FOOTER: `LUMINA_NLTE_STAGE4`·`LUMINA_NLTE_ELEMENT_WIDE`·`LUMINA_SUPER_LEVELS` **0건** ∧ `LUMINA_A210_LINE_SATURATION_TARGET_ION=3` **1건**; 봉인 stdout: Z∈{26,27,28} NLTE 슬롯의 ion 집합 = **{1,2} 정확**(ion=3 0건) ∧ `Super-levels: off (identity)` ∧ `Lines mapped to NLTE ions: 1777859 / 2588798` | 검증기 보고서(필드 축자 기재) | **NC-W1**: stdout 스크래치 사본에 `    Z=26 ion=3: 100 levels` 주입 → `COVERAGE_SLOT_PRESENT` 이름 있는 FAIL·제거 시 PASS |
| **W2 소스 배선 앵커** | `git show 8c2aace:` blob 에서 기계 확인: (i) `src/lumina_plasma.c:19568` 의 unmapped-skip 문면 (ii) `:8973` LTE 분기 문면 (iii) `:20809` 의 produce_tau_source 내 `compute_tau_sobolev` 호출 (iv) `NLTE_TARGET_ION[]` 파스 → ion 값 집합에 **3 부재** (v) `NLTE_TARGET_ION4[]` 파스 → **3 실재**(비교 판별력 전제) | 검증기(정규식+파서; 행 번호 이동 시 문면 탐색으로 강등하되 부재는 FAIL) | **NC-W2a**: 같은 사상 검사를 `NLTE_TARGET_ION4` 테이블로 실행 → 297행이 **mapped 로 분류됨**을 시연(검사기가 커버리지에 맹목이 아님) / **NC-W2b**: 앵커 (i) 문면을 제거한 소스 스크래치 사본 → `ANCHOR_MISSING` FAIL |
| **W3 행 전수 커버리지** | 봉인 stderr 의 `LINE-SATURATION-ROW` **594행 정확**(iter0/iter1 각 297, R7 마커 브래킷 귀속) 전수: `Z∈{26,27,28} ∧ ion=3` ∧ W2-(iv) 소스-추출 base 테이블 기준 **f_mapped = (mapped 행수)/594 = 0 정확** — 두 독립 경로(ROW 필드 / 소스-추출 테이블 사상) | 검증기 census + verdict JSON | **NC-W3**: 행 1개의 `ion=3`→`ion=2` 위조 사본 → f_mapped>0 → **분기 W-B 발화** 시연(제거 시 W-A) |
| **W4 서명 항등 (기전 예측의 정량 검증)** | 봉인 행 재계산(line id 짝맞춤 297쌍): ① 전 행 \|(η₁/η₀)/(τ₁/τ₀)−1\| ≤ **1e-14** ② 행별 S/B(10020) iter0↔iter1 상대차 ≤ **1e-12** ③ 전 594행 \|S/B−1\| ≤ **2e-5** (B_ν 는 물리상수 함수 — 오라클 인용 0) | 검증기(독립 파서·독립 산법 — L6 판정자/감리자 산법의 3번째 경로) | **NC-W4**: η 1개를 1e-9 상대 섭동한 사본 → ①·② FAIL·제거 시 PASS |
| **W5 관측 (게이트 하중 0)** | 덱 준위에너지(봉인 `input/model/atomic_data_cmfgen.h5`, read-only)로 행별 예측 `S/B_pred=(e^{hν/kT}−1)/(e^{ΔE/kT}−1)`, T=10020 K — 실측과 대조표. h5 파스 실패 시 `TOOLING_LIMIT` 로 한계 기재(판정 하중 없음) | 보고서 | (관측 — NC 의무 없음. 단 성공 시 예측-실측 대조 자체가 판별력이다) |

clamp/floor/cap 0. 검증기의 분류(파스 불능·필드 결손)는 전량 census 보고 — 조용한 탈락 금지.
음성대조는 전부 **스크래치 사본**에서 — 봉인 파일 자체는 어떤 게이트도 쓰기 접촉하지 않는다.

## 6. 기대치 사전등록 (빗나가면 그것이 정보다) + 분기

이 단은 런이 없다 — 기대식이 자기 delta 의 영향을 받을 구조가 없다(L6 E1 계급의 재발 여지
0). 그래도 아래가 빗나가면 그것은 파서 결함 또는 봉인/소스 이해의 결함이며 분기 W-D 가 받는다.

| # | 기대 | 수치·범위 |
|---|---|---|
| **E1** | W3 행 수 = **594 정확**(297×2), 전 행 `producer_raw_defined=1` | 봉인 실측 그대로(저자 사전 grep: 594/594) |
| **E2** | **f_mapped = 0/594 정확** | W3 |
| **E3** | W1 필드 전부 §5 문면 그대로 | 저자 사전 실측 그대로 |
| **E4** | W4: ① 실측 최대 ≈6.7e-16 (여유 ~150×) ② 실측 ≤6 ulp (여유 ~10³) ③ 실측 최대 ≈9.6e-6 (여유 ~2×) | 판정문·감리 실측과 3중 일치 예상 |
| **E5** | W5 관측: [추정] \|S/B_pred − S/B_meas\| ≲ 1e-6 등급(행 출력 자릿수·h5 에너지 정밀도 지배). 빗나가면 그 잔차 구조가 정보(준위에너지 대 hν 불일치 가설의 시험) | 보고서 |
| **E6** | 자원: grammar-debug 에서 수 분 [추정]·RAM 수백 MB [추정](stderr 2.04MB + h5 부분 판독) | — |

**분기 (상호 배타·전수 — 어느 쪽이든 단은 착지)**

정의: `f_mapped` := W3 의 594행 중 소스-추출 base 테이블로 (Z,ion) 사상 가능한 행의 비율.
평가 순서: W-D → W-B → W-A, 첫 일치가 verdict (아래 판정식은 순서 없이도 상호배타).

| 분기 | 판정식 | 함의 |
|---|---|---|
| **W-A** | f_mapped **= 0** ∧ W1·W2·W4 전부 PASS | **귀속 폐합**: L6 분기 B null = NLTE 커버리지 구성(base 31슬롯의 ion IV 부재)의 필연. 처분 = §3-6 (a′ 채택·(a)문면 기각·(b) moot·(d) 소멸) + §7-1 갈림길을 user 에게 |
| **W-B** | f_mapped **> 0** ∨ W2 앵커 모순(이름 있는 반례 계급 FAIL) | **귀속 실패** — 반례 행/앵커 자체가 발견이며 다음 단의 입력. §3 의 특정은 그 범위에서 기각된다 |
| **W-D** | 이름 있는 차단(행 수 ≠594·파스 불능·봉인 접근 불능·게이트 도구 실패) | 차단 사유·자리 = 발견. 미결 기재, 폐합 금지 |

분기는 라벨이지 필터가 아니다 — verdict JSON 은 f_mapped·census·게이트별 수치 전량을 보고한다.

## 7. 이 단이 모르는 것 (추측으로 메우지 않는다)

1. **A2-07 solve 가 커버된 이온(Fe/Co/Ni II-III·Si·S·Ca…)에서 J̄ 에 실제로 응답하는가** —
   이 단은 판정하지 않는다. §3-4 는 배선의 실존만 말하고 옳음을 말하지 않는다.
2. **다음 갈림길의 답** — (i) STAGE4 승격 런(ion IV 를 NLTE 화해 L6 재시도) (ii) 커버 이온
   재선정 L6 (iii) 둘 다 아님. 이 단은 입력만 확정하고 선택은 user/별도 Fable 갈림길 평가로.
3. **STAGE4/EW 승격이 DET 레인에서 실행 가능한가**(빌드·게이트·메모리 호환) — 미측정.
4. **봉인 런에 `[A2-06][BB-VIEW]` 카운터 줄이 없는 확정 원인** — [추정] 후보(candidate)의
   사본 카운터가 커밋 없이 사멸(`nlte_population_candidate_free`)하고 공용 카운터는 0 —
   DET 레인의 bb-view 관측 결핍은 **계측 부채**로 대장 기재(이 단은 수리하지 않는다, V5).
5. **W5 의 h5 덱 스키마** — 파스 가능성 미확인(실패 시 한계 기재).
6. **E1 가중 5.0e9 배의 전량 귀속**(L6 §8-4) — 여전히 모른다. 이 단 계약 밖.
7. **lte_unmapped_emit 과 후보 총가중의 정확 산술 관계**(§3-7) — 미검증 관측.

## 8. 이 단이 하지 않는 것

- src/tests/env/런처/덱 접촉 일체(V5 원상 유지 — 권한 요청 0). STAGE4·EW·SUPER 승격 시도.
- L6 판정·감리의 재판정(분기 B 문면은 유효). A2-10 선정 계약 변경. Z-1/Z-O 접촉(존속, user
  보류). 수렴·CMFGEN 정량 대조(오라클 INELIGIBLE — 이 단의 게이트·기대치는 CMFGEN 수치 인용 0).
- 봉인 런루트 쓰기 접촉 일체(음성대조는 스크래치 사본에서만).

## 9. 분장 장부 (집행 후 운전석이 "실제"·"위반"을 채운다 — 규약상 담당만 적는 것 금지)

| 단계 | 규약상 담당 (개정14) | **실제** | 위반 |
|---|---|---|---|
| 갈림길 평가·사전등록(본 문서) | Fable | | |
| 발주(앞겹=본 문서 원문 첨부, 재서술 금지) | 운전석 | | |
| 코딩(§4 검증기) | Codex | | |
| 코드 검수(고정질문에 *"발주서가 사전등록의 범위를 좁혔는가"* 포함) | Fable | | |
| 검증기 실행(grammar-debug)·산출 수집 | 운전석 | | |
| 판정(판정문 저작) | Fable (fresh) | | |
| 판정 감리 | Fable (★판정과 **다른** fresh 컨텍스트·고정질문 4) | | |
| 감리 반영·대장·커밋 | 운전석 | | |

## 10. 판정 절차

- 판정 = Fable **fresh 컨텍스트**(본 사전등록 + 봉인 산출물 경로 + 검증기 산출 제공; 판정
  하중 항목은 판정자가 직접 재실측 — L6 전례. 특히 W3 의 f_mapped 와 W2 의 테이블 파스는
  판정자 자체 경로로 재현할 것).
- 감리 = **또 다른 fresh Fable**(자기 채점 금지), 고정질문 4.
- 폐합 전 감리 필수. 판정문은 §6 분기 중 어느 것이 발화했는지 축자 기재하고, 분기 밖 결과는
  폐합 금지·미결 기재. §3-8 의 계약 결함 계급 기재(대장)와 §7-4 의 계측 부채 기재를 판정문이
  수행했는지를 감리 고정질문에 포함하기를 권고한다.
- 오라클 규율: 게이트·기대치의 CMFGEN 런 수치 인용 0(B_ν 는 물리상수 함수).

## 11. 기계 프리플라이트 선언 — 계약이 스스로를 검사한다

`scripts/check_prereg_preflight.py` 가 이 블록을 읽어 발주 **전에** 계약을 검사한다.
이 단은 런이 없지만 변경집합(검증기 1파일)과 분기 표(W-A/W-B(/W-D 는 차단 계급))를
가지므로 **선언을 전부 유지**한다: PF-1 은 검증기 파일의 단일-소유(고유 심볼)를, PF-2 는
f_mapped 1차원 분할(설계상 1차원임을 명시 — 이 단의 분기 판별량이 실제로 하나다)을,
PF-3 은 §3 인용 앵커의 실존을 강제한다. W-D(차단)는 L6 의 D1 과 같은 계급으로 PF-2 분할
밖의 이름 있는 차단이다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### scripts/ — 검증기 1파일 (신설)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "scripts/[a-z0-9_]+\\.py",
    "symbol": "A207_WIRING_L6_ATTRIBUTION",
    "roots": ["scripts"],
    "expected_extra": ["scripts/audit_a207_wiring_l6.py"]
  },
  "branches": {
    "regimes": [[0.0, 0.4], [0.6, 1.0]],
    "metrics": {
      "f_mapped": "sum(1 for x in v if x > 0.5)/len(v)"
    },
    "rules": [
      {"name": "W-A", "predicate": "f_mapped == 0.0"},
      {"name": "W-B", "predicate": "f_mapped > 0.0"}
    ],
    "adversarial_fixtures": [
      {"name": "single-forged-mapped-row", "mix": [[0.0, 296], [1.0, 1]]},
      {"name": "all-unmapped-297", "mix": [[0.0, 297]]}
    ],
    "residual": "W-D"
  },
  "references": [
    {"path": "src/lumina_plasma.c",
     "flags_existing": ["nlte_update_tau_sobolev_with_authority",
                        "a209_upper_population_for_tau",
                        "build_lte_level_density_cache",
                        "NLTE_TARGET_ION4",
                        "nlte_tau_line_authority"]},
    {"path": "src/lumina_cmfgen.c",
     "flags_existing": ["g_fine_upper_population_fill"]},
    {"path": "src/nlte_population_candidate.c",
     "flags_existing": ["nlte_population_candidate_begin"]},
    {"path": "scripts/analyze_det_stage12_l6.py", "flags_existing": []},
    {"path": "docs/VERDICT_DET_SPRIM_L6_2026-08-22.md",
     "flags_existing": ["A2-07"]},
    {"path": "docs/FINDING_NOBRACKET_LTE_SEED_2026-08-19.md",
     "flags_existing": []}
  ]
}
```

**PF-3 의 정직한 한계**: W2 의 앵커 **문면**(:19568 skip 등)은 PF-3 이 함수명 실존까지만
검사한다 — 문면 자체는 검증기 W2 가 blob 에서 검사하고, 그 검증기의 판별력은 NC-W2b 가
시연한다. 과대 주장하지 않는다.

---

## 12. 집행 기록 (운전석, 2026-08-22)

### 12-1. 발주 1차 — 중단. ★운전석이 계약 상수를 바꿨다

Codex 가 **한 줄도 쓰지 않고** 중단했다: *"발주 뒷겹이 blob 기준을 `85c7ba6` 으로 요구하는데
정본 §4·W2 는 `8c2aace` 를 못박는다."* **옳다.** 운전석 실측으로 관련 4파일 blob 이 전부
동일하고 `8c2aace` 가 `85c7ba6` 의 조상이라 **결과는 같았을** 것이나, **계약이 지정한 상수를
발주가 바꾼 것**이고 임의 동치 처리 거부가 규약대로다. 정정 후 2차에서 착지.

★**계급 반복**: 이 캠페인에서 운전석 발주 결함이 세 번째다 — DET-SPRIM-L6 1차(`tests/` 탈락)·
5차(계약 밖 요구)·이번(계약 상수 변경). 공통 기전은 **뒷겹에 집행 조건을 적다가 계약을
재서술하게 되는 것**이다. 개정14 가 "앞겹=원문 그대로, 재서술 금지" 라 한 이유가 이것이다.

### 12-2. 변경집합 1:1 · 봉인 무변조

§4 표와 **미달 0 · 초과 0**(검증기 1파일). **`src/`·`tests/` 접촉 0** — V5 원상 유지.
검증기 `scripts/audit_a207_wiring_l6.py` sha256 `8cc0c960…`, 1,229행.

★**봉인 무변조 실측 (2중)**:
- 운전석: 런루트 전체 digest 실행 전후 **`2ffebf8ea3de749ca972ed950fb46bd6` 동일**, `03:00`
  이후 수정 파일 0.
- 검수자 독립: 두 봉인 루트 전 파일(133+123개) mtime·size 목록 실행 전후 **완전 동일**.
  코드 낭독으로 쓰기 경로는 `--output` 뿐이며 run_root 하위면 `SEALED_WRITE_FORBIDDEN` 거부
  (`:1202-1204`) 확인.

직전 단에서 stager 가 하드링크로 봉인 base 를 침묵 파괴할 뻔한 전례 때문에 지문을 떴다.

### 12-3. 본실행 (운전석)

```
W1 status=PASS   W2 status=PASS   W3 status=PASS   W4 status=PASS
W5 status=TOOLING_LIMIT gate_weight=0 predicted=0/594
DET_A207_WIRING branch=W-A f_mapped=0.0 reasons=ATTRIBUTION_CLOSED     rc=0
```
W2 상세: `blob=8c2aace:src/lumina_plasma.c` · **`base_contains_ion3=False` ∧
`stage4_contains_ion3=True`** · `missing_anchors=[]`.
W3 상세: 594행 · R7 마커 브래킷(`iter 0: te_generation 1->2` / `iter 1: 2->3`) ·
`target_counterexamples=[]`.
W5: h5 에 `LEVEL_ENERGY_DATASET_NOT_FOUND`(594행 전부) — **설계대로 판정 하중 0**.

### 12-4. 검수 판정 — **조건부 인정 (차단 0)**

검수자가 봉인 read-only·스크래치 사본으로 재현했고 **주입 12종**(selftest 5 + 자체 10 중복
포함)을 end-to-end 로 시연했다. 4번째 독립 경로 재계산이 운전석·검증기와 **최종 자리까지
일치**: `common=6.6613381477509392e-16` · `|S/B−1|max=9.5777180635359116e-06`.

★**NC-W2a 의 성질은 실물로 성립한다** — 검수자가 **봉인 실물 594행 × 실물 `8c2aace` 테이블**로
독립 사상: **base 0/594 · ION4 594/594 mapped**. 검사기는 커버리지 맹목이 아니다.

**수정 필요 1건(비차단) — 해소 = (b) 판정문 명기 채택**:
검증기 자신의 NC-W2a 는 **합성 픽스처**(`synthetic_source`/`discriminator_rows`,
`:988-1007`·`:1098-1109`)로만 시연하고, **실물 594행 × ION4 사상은 검증기 안에도 본실행
산출에도 없다**(`run_audit` 은 stage4_pairs 를 파스만 하고 사상하지 않는다). 합성 행과 합성
테이블은 같은 저자가 같은 관례로 만든 것이라 **실데이터 관례-정합 시연이 아니다.**
⟹ 검증기를 고치지 않고 **이 편차와 검수자의 실물 시연을 판정문에 명기**하는 쪽을 택한다
(§10 이 판정자에게 f_mapped·테이블 파스 자체 재현을 이미 의무화했다). 검증기 보완은
후속 단 후보로 등재.

★**부수 결함 (운전석 보고 정정 사유)**: selftest 의 NC 출력줄이 **하드코딩 문자열**이다
(`:1089`·`:1105-1106`·`:1130-1131` 등). `require()` 단정은 실물이라 **NC 는 진짜로 발화**하나
**인쇄된 수치·분기 라벨은 도출값이 아니다**(W1 주입은 사유 2개가 나는데 1개만 주장;
`branch=W-B/W-A` 는 `determine_branch` 미호출). 발주서의 *"출력을 꾸미지 마라"* 취지 위반.
⟹ **운전석이 그 줄들을 실측으로 인용해 보고했다 — 정정하고 여기 기재한다.** 실물 사상의
근거는 검수자 재측정이지 이 인쇄줄이 아니다. 인쇄를 실측 도출로 교체하는 것도 후속 후보.

### 12-5. 검수 발견 3건 (조용한 대장 기재)

| # | 발견 | 처분 |
|---|---|---|
| 1 | **§6 분기표의 구멍** — `Z∉{26,27,28} ∧ ion=3` 위조는 어느 판정식에도 안 걸려 **W-A 가 참이 될 뻔**했다. 구현은 `W-D UNCLASSIFIED_GATE_STATE`(`:921`)로 막는다 — **코드가 계약보다 엄격** | §3-8 과 같은 **사전등록 위생** 계급. 차기 사전등록에 반영 |
| 2 | **W5 는 계약이 소스를 잘못 지정했다** — 준위에너지는 봉인 h5 가 아니라 같은 덱의 `input/model/levels.csv` 에 있다. 검증기는 계약 문면을 지켰다. 검수자가 계약 외 관측 수행: **594/594 예측, `max|S/B_pred−S/B_meas|=8.2e-6`**(중앙 7.3e-6) — E5 의 [추정] 등급과 정합, §3-5(iv) 서명 구조 지지 | 관측으로 기재. 게이트 하중 0 |
| 3 | W1 FAIL 이 W-D 로 라우팅(계약 분기표가 W1 반례 미배정) | 라벨 문제·JSON 전량 보고라 무해. 기재만 |

### 12-6. 분장 장부 (검수 완료 확인 후 기입)

| 단계 | 규약상 담당 | **실제** | 위반 |
|---|---|---|---|
| 갈림길 평가·사전등록 | Fable | Fable — 표적의 **질문 자체를 정정** | — |
| 발주 | 운전석 | 운전석 | ★1차 **계약 상수 변경**(§12-1) |
| 코딩 | Codex | Codex (1차 중단은 정당) | — |
| 코드 검수 | Fable | **Fable — 조건부 인정**(주입 12종·실물 사상 독립 시연) | — |
| 검증기 실행·산출 수집 | 운전석 | 운전석 (봉인 지문 2중) | ★selftest 인쇄줄을 실측으로 오인용(§12-4) |
| 판정 | Fable (fresh) | | |
| 판정 감리 | Fable (판정과 다른 fresh) | | |
| 감리 반영·대장·커밋 | 운전석 | | |
