# A2-05 구현 명세 v2 — CPU bound-free rate 소유권 이관

저작 운전석(개정 8) · 검수 Codex(`docs/CODEX_REVIEW_A2_05.md`, 반박 7 + 공백 6 전부 반영) ·
기준 `HEAD=bafd2bb`.

## 1. 단일 계약

CPU 생산 물리의 bound-free 광이온율이 `bf_rate_estimator` 대신 **정본
`RadiationField.J_nu` 의 checked read-view 를 직접 적분**한다.

## 2. 검수 반박의 계약화 (R1~R7)

### R1 — PRRR 채널 자격 (반박 1)

- **판정 채널 = photoionization Γ 하나** (PRRR `<ion> Photoionization Rates`).
- 총 RR coefficient α(cm³ s⁻¹)는 **차원 등록 비교**(아래 R7)로만.
- spontaneous/stimulated 분리·collisional ionization 판정 =
  **`BLOCKED_MISSING_RATE_EXPORT`** (O-PHYS §5.2 rate audit 요구에 이미 등재).
  총 α 를 임의 분해하지 않는다.

### R2 — 안전대 = s0–s8 ∩ ion별 bandmask (반박 2)

`rates_certification/run_log.txt` 의 ion별 self-consistency 를 그대로 소비:
S II 는 s8 제외(`['s8']` 실측), Co III·Fe III·S III 는 s0–s8 전체. 판정 산출물에
ion×shell 제외 목록과 제외 전후 f_cov 를 기록.

### R3 — census 재배치 (반박 3)

- **A2-05 실 이관 대상** (원장 ADDENDUM 신설 행):
  `src/lumina_plasma.c:2277-2299, 2342-2344, 5045-5067, 16063-16079` ·
  `src/lumina_element_wide.c:597-613, 1114-1137` — bf_rate_estimator CPU rate 소비 6지점.
- 원장 기존 "BF" 행 중 `:9160/:9162, :11943, :13672` = **population 경로 → A2-07 재배치**.
  `:11976, :12034` = 진단 유지. BB/line 행 = A2-06 재배치.
- 원장 갱신은 diff 로 산출(ADDENDUM + 재배치 표).

### R4 — zero-consumer 게이트 범위 (반박 4)

인수조건 = **"CPU 생산 물리 소비자 0"**. 허용 잔류를 명시 목록+카운터로:
(a) raw 통계 생산자/lifecycle (b) 출력 전용 진단 (c) GPU 경로 전부(A2-13 까지).
legacy estimator 를 rate fallback 으로 두는 것은 **금지** (ORDER §2.2) —
비교 전용 shadow(소비 안 됨)만 허용.

### R5 — checked read-view API (반박 5)

`radiation_field.h` 에 신설 (writer 아님):
```c
int radiation_field_read_view(const RadiationFieldOwner *owner,
                              uint64_t expected_generation,
                              RadiationFieldView *out);
```
성공 조건 전부: enabled · units/frame 정합 · epoch/n_shells 기대치 ·
`required==computed==expected_generation` · canonical edge hash/shape.
실패 시 view 를 주지 않고 오류코드 반환 — **rate 0 반환 금지**. commit 외 writer 추가 금지.

### R6 — rate 결과의 validity 계약 (반박 6)

rate 결과형 = 값 + 상태 {VALID, EXACT_ZERO, UNSAMPLED, OUT_OF_GRID, STALE}.
- 결합 우선순위(적분 구간 [ν_th, ν_max] 내): STALE > UNSAMPLED > OUT_OF_GRID.
  (STALE=세대 불일치 즉시 실패, UNSAMPLED=통계 공백, OOG=구조 공백)
- **가중 규칙**: 구간 내 비-VALID 빈의 σ-가중 기여율 w_miss 를 계산.
  w_miss ≤ 1e-3 → VALID (missing 기여 무시 가능, w_miss 기록);
  그 외 → 해당 상태. 작은 값 대입 금지·process abort 금지.
- 비-VALID rate 의 downstream: SE solve 는 그 (level,shell) 항을
  `BLOCKED_INSUFFICIENT_SAMPLING` 로 표시하고 legacy 값도 0 도 넣지 않는다 —
  이관 게이트에서 그 항 수를 카운터로 보고, 판정 분모에서 사유 제외.

### R7 — 차원 등록 산식 (반박 7)

- Lumina Γ_i = 4π ∫_{ν_i}^{ν_max} J_ν σ_i(ν)/(hν) dν  [s⁻¹]
- PRRR Γ 대조: PRRR 값 = rate/ion-density 이미 [s⁻¹] — **밀도 곱 0회** (기존 검증된
  해석 `oracle_compare_cmfgen.py:127-133` 그대로).
- α 대조: PRRR RR = coefficient [cm³ s⁻¹] — Lumina 쪽도 coefficient 로 환산해 비교
  (n_e·n_ion 나눗셈 1회, 위치 명시). §13 경로 23 음성대조 = 밀도 1회 추가 곱 주입
  → α 비교 FAIL 확인.

## 3. 게이트 계약 (공백 6 반영)

1. **CHAIN / ORACLE_INPUT**: 두 lane 모두 population·T_e·n_e·σ = 동일 스냅샷 고정
   (A2-04 L-0 replay 의 덱 상태). 다른 것은 J 뿐 — CHAIN=MC commit 정본(고정 seed
   capture 의 generation), ORACLE_INPUT=CMFGEN EDDFACTOR 를 deterministic commit 으로.
2. **§6.3 CI 자격**: MC lane 의 Γ 는 count/variance 로 CI 반폭 산출, 반폭 > 합격선/3
   인 (level,shell) 은 UNDERPOWERED 기록·판정 제외 (A2-02C gate2 선례 재사용).
3. **matching universe 사전등록**: `rates_certification` 의 기존 ion·level crosswalk
   를 그대로 인용(신규 발명 금지), f_cov 분모 = ORDER §6.3 누적 기여 99.9% 활성 집합.
4. **부분 빈 산식**: 빈 평균 J_b 는 빈 내 상수. threshold 빈은
   ∫_{max(ν_th,ν_lo)}^{ν_hi} σ(ν)/(hν) dν 를 σ 표점의 구간별 선형(현행 tabulation
   해석 유지)으로 적분 후 J_b 곱. σ 커널 사전계산은 A2-02C builder 의 bf_kernel
   규약(4π σ/(hν), threshold 아래 0)과 동일식.
5. **음성대조 witness 사전등록**: 민감도 여유가 실측된 ion 사용 —
   witness = Fe III·Co III (안전대 전셸 NONE + 강한 edge; `rates_certification`
   실측 기반). 각 poison 의 기대 FAIL metric 명시:
   (a) `W B_ν(14172.549)` 주입 → Γ 가중 E_1 FAIL (A2-04 음성대조와 동일 주입원)
   (b) threshold 한 빈 이동 → witness edge 의 E_sym FAIL
   (c) α 밀도 1회 추가 곱 → α 대조 FAIL (R7)
   물리 FAIL 관측 시 runner rc=0 (검증기 정상 판정).
6. **legacy fallback 없음**: 이관이 유일 경로. 비교 전용 shadow 는 게이트 산출물로만.

## 4. 회귀

배터리(65초) + A2-01 census(ADDENDUM 반영 후) + A2-03 shadow parity(픽스처) +
A2-04 commit selftest + L-0 replay. HEAD artifact 기준(낡은 PENDING 문구 아님 — 검수 확인 6).

## 5. 구현 순서 (운전석)

1. `radiation_field_read_view` (R5) + selftest
2. Γ 적분기 `src/bf_rate_jnu.c` (R7 산식·R6 validity·부분 빈) + 단위 selftest
   (해석해 대조: 수소형 σ~ν⁻³ + 멱법칙 J 의 닫힌형)
3. 소비 6지점 이관 (R3) — census ADDENDUM diff 동반
4. L-1bf 게이트 스크립트 (R1·R2 채널/안전대, 게이트 계약 1~6)
5. 음성대조 3종 + 회귀 → 커밋
