# A2-06 구현 명세 v3 — CPU bound-bound rate: J̄ selective estimator + LineJbarCache

저작·구현 운전석(개정 8) · 검수 Codex 2라운드(V1 BLOCK 7 → V2 BLOCK: B2/B3/B5-6/B7)
전건 계약화 · 기준 HEAD=d8b9870. V2 의 B1(selective estimator 노선)·훅 위치(CPU 적립
단일점 `lumina_transport.c:118`)는 RESOLVED — §1·§3 은 V2 를 승계한다
(`docs/SPEC_A2_06_V2.md` §1, §2-B1, §2-B4). 아래는 V2 대비 신규 확정분만 적는다.

## 1. B2 완결 — commit transaction·view API (구현 계약)

### 1.1 Commit 입력 확장 (`RadiationFieldCommitRequest` 에 line 블록 추가)

```c
/* MC 형: transport 가 적립한 thread-reduce 결과 */
uint64_t    line_q_set_hash;      /* Q_g 사전 고정 해시 (누적 전 계산) */
size_t      line_n;               /* Q_g 라인 수 */
const uint64_t *line_id;          /* [line_n] */
uint64_t    line_profile_id;      /* 등록 프로파일 (census 단일: Gauss v_D=10) */
const char *line_profile_hash;
const double   *line_raw;         /* [line_n*n_shells] Σ∫εφdℓ (정규화 전) */
const uint64_t *line_count;       /* [line_n*n_shells] 기여 세그먼트 수 */
const double   *line_m2;          /* [line_n*n_shells] Welford M2 (필수 — CI 원료) */
/* 결정론(replay) 형: line_jbar/line_validity 직접 (provenance=CMFGEN_REPLAY) */
const double *line_jbar; const int32_t *line_validity;
```

- **staging→publish 순서**: ① J_ν 후보 검증 ② line 후보 검증(validity 산출:
  count==0→UNSAMPLED, raw==0&&count>0→EXACT_ZERO, raw>0→MEASURED; Q_g 밖=항목 없음)
  ③ 둘 다 성공했을 때만 **한 번의 publish** 로 J_ν·cache·generation 동시 전이.
  어느 한쪽 실패 = **public 상태 전부 불변**(generation 포함). 정규화
  `Ĵ̄=raw/(4π V_s Δt)` 는 publish 단계에서.
- line 블록이 없으면(=BB rate graph disabled 런) cache 는 전 항목 없음으로 publish —
  active 소비가 즉시 MISS 오류가 되는 fail-closed.
- **적립 오류 전파**: line accumulator add 는 rc 반환, thread-local 실패 플래그 →
  reduce → commit 거부(전역 accumulator 와 동일 규약; 현 segment-add 반환 무시 관행
  금지).

### 1.2 Checked view

```c
typedef enum { LINE_JBAR_VIEW_OK=0, _DISABLED=-1, _UNITS_FRAME=-2,
  _EPOCH_SHELLS=-3, _STALE_GENERATION=-4, _QHASH=-5, _PROFILE=-6 } …;
int radiation_field_line_jbar_view(const RadiationFieldOwner*, double epoch,
    size_t n_shells, uint64_t generation, uint64_t q_set_hash,
    uint64_t profile_id, LineJbarView *out);
/* 조회: line_jbar_lookup(view, shell, line_id) →
   {jbar, validity, count, se}  |  MISS = 구별 오류 (음수 rc) */
```

- 성공 조건 전부: enabled·units·frame(comoving)·epoch·n_shells·
  `required==computed==expected_generation`·q_set_hash 일치·profile_id/hash 일치.
  **실패 시 out 전체 memset 무효화** + 구별 오류코드 (A2-05 read_view 선례).
- 반환에 SE 포함: `se = sqrt(M2)/count` (count≥2), count<2 → SE 정의 불가 표기.
  단위·frame·provenance 는 view 성공 조건에서 이미 검증되므로 조회 반환은 4-튜플.
- 소비자는 cache 배열·setter 직접 접근 금지 — view 경유만 (§3.2 개정).

### 1.3 Partial-commit 주입 시험 (selftest 필수)

J_ν 후보 성공 + line 후보 강제 실패(음수 raw 주입) → commit rc!=0 이고
J_ν·cache·generation **모두** 이전 public 상태와 memcmp 동일함을 검사. 역방향
(line 성공+J_ν 실패)도 동일.

## 2. B3 완결 — census 16행 1:1 처분표 (구현 시 diff 로 재실측 확정)

| census 행(bafd2bb) | 현행 위치 | 실체 | 처분 |
|---|---|---|---|
| 4556 W/T_rad ×2 | ~4633 그룹 | MA 상향률 J̄ 선택 (jbar_line_det/jbar_line/coarse) | **A2-06 이관** |
| 4596 W/T_rad ×2 | ~4661 | jblue_line 직접 소비 | **A2-06 이관** |
| 4701 W/T_rad ×2 | ~4731 | B_lu−B_ul 구성 W/T_rad | **A2-06 이관** |
| 4879/4880 ×2 | 4934/4935 | lower/upper **population fallback** | **A2-07 재배치** (rate 아님 — R_lu 교체 시 population solver 침범) |
| 11908 ×2·11915 ×2 | 11960/11967 | line-source/blanketed heating field 구성 | **A2-08 재배치** (source/emissivity) |
| 12093/12100 ×2 | 12145/12152 | level population fallback | **A2-07 재배치** |
| 13739/13743 ×2 | 13789/13795 | level population fallback | **A2-07 재배치** |

- 추가 이관(원장 밖, 1차 검수 전수): 10827(simul ETLA) · 12182(RADEQ ETLA) ·
  13823(coupled ETLA) · 15238(dilute rate source) · 15292(jbar_line 직접) ·
  15361(R_absorb/R_stim/R_spont 분기) — ADDENDUM 신설 행.
- 잔류 허용목록(카운터·귀속 명기): 15457·15591(진단 shadow read — 원장 행 신설,
  KEEP_DIAGNOSTIC_READ), cmfgen 3153/3159(A2-08), 진단 3행(:13920/:13940/:14080),
  GPU 전부(A2-12/13), JEQB 류 falsifier.
- 원장 갱신 = ADDENDUM + 재배치 diff (A2-05 형식).

## 3. B5/B6 완결 — validity·게이트 사전등록

### 3.1 카운터·상태

`bb_view_rate_terms` + `bb_view_blocked_{stale,unsampled,oog,profile,qhash,miss,
disabled}` — `nlte_free` 종료 보고 라인(A2-05 패턴). cache MISS(active 인데 Q_g 에
없음)는 **전용 카운터+구별 오류**다(§3.2 개정: Q_g 밖 active 요청 = 명시적 오류).

### 3.2 same-measure 게이트 해시

비교 대상 = {raw segment ledger sha, generation, frame enum, V_s 배열 sha,
Δt(time_simulation), packet normalization 상수} 전부의 결합 해시 — J_ν 와 cache 가
동일값 보유를 게이트가 대조.

### 3.3 MC CI (§6.3)

variance **필수**(1.1 의 M2). CI 반폭 = 1.96·SE, 판정 단위 = (line, shell),
자격 = CI 반폭 ≤ 사전등록 한계/3 **AND** truth-측 f_cov ≥ 0.999, 미달 사유
UNDERPOWERED / BLOCKED_INSUFFICIENT_SAMPLING 구분 (A2-05 CHAIN 확립 형식).

### 3.4 음성대조 — 9종 사전등록 (각각 marker `A2_06_NEG_<n>` + runner rc=0 는 전건
기대 FAIL 관측 시)

개정 §3.5 의 7종 전부: (1) cache generation=J_ν−1 → 조회·commit 둘 다 FAIL
(2) line ID/profile hash 교환 → 결박 FAIL (3) 4π/V_s/Δt/frame 누락 fixture →
projection closure FAIL (4) UNSAMPLED→0 치환·coarse fallback 주입 → FAIL
(5) 작은 A_ul/직전 estimator 기준 Q_g 솎기 → selection census FAIL
(6) cache 독립 setter/독립 generation/긴 lifecycle → owner checker FAIL
(7) φ 정규화≠1·observer-frame ν 주입 → 정합 게이트 FAIL.
개정 §5.4 에서 2종: (8) production ΣφJΔν·coarse fallback·독립 jbar_line owner 잔존 →
**static read-trace FAIL** (구현: grep 기반 zero-consumer 검사 스크립트 — rate 경로의
jbar_line/jblue_line/W·T_rad 소비 0 을 정적 검사) (9) partial-commit 공개 → FAIL
(1.3 의 주입 시험).

### 3.5 L-1bb 판정 스키마 (사전등록; 현재 상태 BLOCKED_MISSING_RATE_EXPORT)

O-PHYS NETRATE/TOTRATE 도착 시 판정: flow coverage ≥0.95 · E_1(J̄)≤0.10 ·
E_1(R_lu)≤0.10 · E_1(R_ul^stim)≤0.10 각각(net 단독 금지) · E_sym P95 ≤0.25 ·
A_ul crosswalk ≤1e-10. 두 lane 고정: population·T_e·n_e·원자자료 = toy06 스냅샷,
**유일 변경 = J̄ 공급자**(CHAIN=MC estimator, ORACLE_INPUT=CMFGEN_REPLAY cache).
지금 실행분 = A_ul crosswalk + wiring replay + BLOCKED 상태 assertion.

### 3.6 truth-측 f_cov (A2-05 원칙 승계)

분모·활성집합 = **truth-측 절대 radiative flow** `n_l(POP)·B_lu·J̄_truth`
(J̄_truth = EDDFACTOR fine 격자에서 φ 직적분 — 오프라인 진단 산출, 런타임 아님)
의 내림차순 누적 99.9% (line,shell) 집합, 가중 freeze = truth 값. 분자 = 그 중
view MEASURED 기여. 상태 나쁜 항을 분모에서 빼는 구성 금지(A2-05 순환 적발 선례).

### 3.7 A_ul crosswalk 완전 정의

매칭 = (Z, ion, E_lower±1e-6eV, g_lower, E_upper±1e-6eV, g_upper) 6-튜플, 중복 후보
= 에너지 최근접 유일 배정·잔여 UNMATCHED 기재(삭제 금지 — 고아 금지 규약). 판정 =
|A_lumina−A_cmfgen|/max ≤1e-10; 양쪽 0 = 일치, 한쪽 0 = FAIL 행. coverage 합격선 =
truth-측(A_ul·g_u 가중) f_cov ≥0.999 + UNMATCHED 목록 보고.

### 3.8 closure cohort 사전등록

projection closure: 제어 프로파일 φ^(N) = canonical 빈-상수, 대역 8개(로그 균등)
사전등록. fine closure: cohort = `A2_02C_LINE_CENSUS.json` 감사 라인 집합 재사용,
fine 히스토그램 해상도 수렴 먼저 증명(연속 2배 세분 변화 <0.2%). 공통: 한계 최대
1%·중앙값 0.2%, UNSAMPLED (line,shell) 은 분모 제외+사유 카운트, zero 는 E_sym 규약
(양쪽 0=0, 한쪽 0=FAIL 표기).

## 4. B7 완결 — 회귀 전판

배터리 36 · L-0 replay · L-1bf 게이트+selftest(A2-05) · A2-03/04 selftest ·
`make lumina` 실빌드 · **A2-01 census checker(ADDENDUM 반영)** · **A2-02C gate2
replay(기존 9종 음성대조 포함)** · zero-consumer static read-trace(3.4-8) ·
partial-commit 주입(1.3) · Q pruning/normalization/lifecycle 음성대조(3.4-5/6/7) ·
L-1bb BLOCKED 상태 assertion. 신규 selftest: line accumulator 합성 세그먼트 해석해
(상수 ε·선형 ν′ 의 φ Gauss 적분 = erf 닫힌형 대조) + view 오류코드 전수.

## 5. 구현 순서 (운전석)

1. Q_g 구축+hash, LineJbar accumulator(Welford M2 포함)+transport 훅 시그니처 확장
   (ν′ 양 끝점 전달; `lumina_transport.c:118` 인접 단일점)+rc 전파
2. commit dual-view 원자화(1.1)+view API(1.2)+partial-commit selftest(1.3)
3. 소비 이관(§2 처분표)+원장 ADDENDUM/재배치 diff+카운터(3.1)
4. 게이트: 정합 3종+음성대조 9종+A_ul crosswalk+wiring replay+BLOCKED assertion
5. 회귀 전판(§4) → Codex diff 검수 → 커밋

제약: V2 §4 승계 (CPU 만·population/이온화 솔버 불변·덱 불변·lageunha 대형).
