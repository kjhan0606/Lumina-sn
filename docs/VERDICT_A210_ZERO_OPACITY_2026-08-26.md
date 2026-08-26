# 판정문 — A210-ZERO-OPACITY Z-1r census (런 324388, 2026-08-26)

- **판정자**: Fable (fresh 컨텍스트, 분담 개정14 ③). 구현·검수·집행과 별개 컨텍스트.
- **분장 장부 = 사전등록 §10** (`docs/RUNG_A210_ZERO_OPACITY_2026-08-19.md` — 정본, 개정16 ①).
  이 문서는 장부를 복제하지 않는다. 판정 칸 기입분: **판정 = Fable fresh (2026-08-26) · 위반 없음.**
- **계약(사전등록 「계약 (하나)」)**: *진단 경로의 실패는 진단만 무효화하고, 물리 트랜잭션을
  중단시키지 않는다.* 이 단은 그 계약 위반(차단)의 **측정 단**이다(§7 처분 원칙).
- 모든 수치는 판정자가 직접 실측·재계산했다(부록 A·B — 전 명령 실행 확인 후 기재).
  봉인 런루트에는 아무것도 쓰지 않았다.

**런 식별**: job **324388** `LUMINA_ZO_CENSUS_II` — COMPLETED · Elapsed 02:42:56 ·
MaxRSS 32195676K(≈30.7 GiB) · syn102 · partition a100 · ExitCode 0:0
[명령: `sacct -j 324388 -o JobID,State,Elapsed,MaxRSS,ExitCode`].
런루트 `/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1`
(이하 `<RR>`). `model.rc`=0 [`cat <RR>/model.rc`]. stderr 151,834,294 B [`ls -la <RR>`].
바이너리 커밋 `e500e29` [`<RR>/ZO_RUN_SEAL.txt` `git_head=` 행]. 소스 인용 행 번호는 현 HEAD
`b7accae` 기준이며, `git diff --stat e500e29 b7accae -- src/` 출력 공백(= src 불변)으로
e500e29 과 동일함을 확인했다.

---

## 1. 증거와 재계산 방법

| 증거 | 접근 |
|---|---|
| `<RR>/stderr.log` (봉인 산출, **행-계측 정본** — 사전등록 §3 "행-계측이 census 의 정본") | ROW/SUMMARY 전행 스트리밍 파스 (부록 B) |
| `<RR>/ZO_CENSUS_REPORT.json` (판독기 산출, 574,320,152 B) | raw 블록 스트리밍 교차 (부록 B 주석) |
| `<RR>/input/model/levels.csv`·`line_list.csv` (런이 해시 검증한 활성 덱 — `<RR>/deck_verify.log` 전건 OK, 검증 주체는 `<RR>/input/job.slurm:115-118`) | (Z,ion,level)→g 전수 대조 (부록 B) |
| 봉인 L6C `/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/l6c_20260822T154507Z_cover/stderr.log` | Z5 ON-팔 grep (부록 A-6) |
| 봉인 L6 `/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe/stderr.log` | Z2 IV 소급 grep (부록 A-7) |
| `src/lumina_plasma.c`·`src/opacity_publication.{c,h}`·`src/lumina.h` | 행 번호 직독 (본문 인용마다 병기) |

판독기 재계산 관계: 판독기 JSON 의 행 데이터는 같은 stderr 에서 나온다. 따라서 **stderr 재계산이
더 상류의 독립 검증**이고, JSON 교차는 판독기 직렬화의 무결성 검사다. 두 경로가 전 항목 일치했다
(부록 B 출력: `JSON raw n_upper vs stderr mismatches = 0`).

---

## 2. §8 기대치 판정 — E-Z1 ~ E-Z5

| # | 등급 | 판정 | 근거 실측 |
|---|---|---|---|
| **E-Z1** 완주 | **게이트** | **PASS** | `model.rc`=0 · SUMMARY 정확 2줄(stderr:225081 iter0 / :449539 iter1) 양쪽 `complete=1` [부록 A-2] · sacct COMPLETED 0:0 · 런처 수용 `A210_ZO_CENSUS_ACCEPT status=PASS` [`<RR>/ZERO_OPACITY_CENSUS_VERDICT.txt`:1 — **정정(감리-B NOTE#12)**: 초판이 `zero_opacity_census.stdout` 로 적었으나 그 파일은 판독기 결과 줄(`A210_ZERO_OPACITY_CENSUS_ANALYZER_RESULT status=PASS branch=ZC-A fa=1.0 …`)만 담는다 — 판정자 양 파일 재확인] |
| **E-Z2** iter0 rows ≥1 ∧ == ROW 줄 수 | **게이트** | **PASS** | iter0 SUMMARY `zero_opacity_emitting_rows=221382` == iter0 ROW 줄 수 **221,382** [부록 A-3]. (요구 밖 보강: iter1 도 221,382==221,382) |
| **E-Z3** 앵커 bit-재현 | **게이트** | **PASS** | iter0 ROW 에 `line=669992` 존재 — **스캔 첫 ROW**(stderr:1242, L6C "스캔 순서상 첫 hazard" 재현) · 4필드 hex 동일: `tau_validity=2` · `n_upper`=0x1.d1534dba20c26p-12 · `A_ul`=0x1.0b6516044e436p-1 · `emission_per_sr`=0x1.3adcec9caadc0p-52 [부록 B 출력 `anchor_bit_ok=True`; 판독기 JSON `anchor.matches=true` 와 일치] |
| **E-Z4** candidate_rows > 66110 | 관측(비게이트) | **충족** | `candidate_rows=749644`(양 반복 동일, SUMMARY 실측) > 66,110. 관측 기재: 사전등록 추정 ≈1.03M 대비 실측 749,644(−27%, 창-절사 추정의 오차 — 비게이트라 판정 무영향). per-ion 분해 Z=26: 356,479 · Z=27: 353,694 · Z=28: 39,471 (합 = 749,644, `[A2-10][LINE-SATURATION-UNION-ION-SUMMARY]` 실측). 준위쌍 목록: **221,382 고유쌍 확보**(행수=고유 line=고유쌍, 중복 0 — 부록 B) |
| **E-Z5** 경과·메모리 등급 | 관측(비게이트) | **정합** | 02:42:56 (기대 ≈04:34 등급보다 짧다 — 캡처 2종 OFF 의 부하 감소로 개연, 등급 판정에 무영향) · MaxRSS 30.7 GiB (≈35.4 GiB 등급 내) |

---

## 3. 분기 판정 — **ZC-A** (판독기 라벨을 §8 판정식에 대조해 재검증)

§8 판정식: **ZC-A ⟺ 완주 ∧ hazard>0 ∧ fa ≥ 1.0** (fa ≡ iter0 hazard 행 중
`tau_validity==A208_EXACT_ZERO` 비율).

- **완주**: §2 E-Z1 의 독립 증거 5종(model rc·SUMMARY 2×complete=1·sacct·수용문·차단 마커 0 —
  아래). 추가로 마지막 SUMMARY 이후 ROW 0줄 [부록 A-3 넷째 명령] — 중도절단 아님.
- **hazard>0**: 221,382 [부록 A-3].
- **fa**: 판정자 재계산 **221,382 / 221,382 = 1.0** [부록 B; `A208_EXACT_ZERO`=2 는
  `src/opacity_publication.h:8-9` enum(`A208_VALID=1` 다음 자리) 직독으로 확정].
- 차단 부재: `LINE-SATURATION-BLOCKED` **0** · `INDEPENDENT_SPROBE_UNDEFINED` **0** ·
  `DET_FLIGHT_FATAL` **0** [부록 A-4].

⟹ **판독기 `branch=ZC-A fa=1.0` 은 옳다.** 처분(§8): Z-2A 발주 준비 + B 병행 검토 +
준위쌍 목록의 생산자 귀속 인계 — §9 에 구체화.

**★ZC-D 라벨에 관한 명시** (검수 인계 Q2-1): 판독기는 §8 의 ZC-D(미완주) 라벨을 **부여하지
않는다** — 미완주는 이름 있는 FAIL 로 fail-closed 되지만 `branch` 필드는 `UNRESOLVED` 로
남는다. 따라서 **"branch 필드가 ZC-D 가 아님"은 완주의 증명이 아니다.** 이 판정문의 완주
판정은 라벨 부재가 아니라 위의 독립 증거(rc·SUMMARY×2·수용문·마커 0·절단 후 ROW 0)에서
나온다. 이번 런은 그 증거로 완주가 확립됐으므로 ZC-D 는 해당 없음.

**ZC-B 미발동**: fa=1.0 이므로 (b) 혼재 분기 아님 — coverage 결손 별도 단 개설 의무는
발생하지 않는다(스코프 한정은 §5-2 ②).

---

## 4. 게이트 Z1–Z6 현황 (§4)

| 게이트 | 판정 | 근거 |
|---|---|---|
| **Z1** | **PASS** | 전 hazard 행 ROW 인쇄 + ROW 수==카운터(양 반복, §2 E-Z2) [판정자 재계산]. "판정 로직 불변" word-diff 는 검수 몫(장부=사전등록 §10: 검수 차단 0건) — 판정자 보강: hazard 분기 블록(`lumina_plasma.c:14075-14089`) 직독 결과 카운터 증가와 `fprintf` 뿐, return·상태 변경 없음 |
| **Z2** | **2/3 충족** | IV=소급 충족 — 봉인 L6 stderr `:1893`·`:3581` `target_ion=3 complete=1 zero_opacity_emitting_rows=0` [판정자 재실행, 부록 A-7] · II=이 런 221,382 · **III=Z-2A 후 Z3 재현런 겸측 — 미결** |
| **Z3** | **미결** (수리 후 게이트) | 이 런의 범위 밖 |
| **Z4** | **미결** (수리 후 게이트) | 이 런의 범위 밖 |
| **Z5** | **PASS** | **(in-vivo 토글 페어)** 봉인 L6C(ion=1·capture=1): `LINE-SATURATION-BLOCKED` 정확 1줄, `reason=INDEPENDENT_SPROBE_UNDEFINED … zero_opacity_emitting_rows=1 complete=0`, job 322449 [부록 A-6] ↔ census(ion=1·capture=0): BLOCKED 0·완주·rows=221,382. 같은 봉인 base(`<RR>/ZO_CENSUS_STAGE_SEAL.txt` `sealed_l6c_base=` 동일 경로)·env delta 정확 2(캡처 2종 1→0) ⟹ **차단이 캡처 진단 탓임이 주입 없이 시연됨** · **(계측 NC)** 판독기 `--selftest` **판정자 재실행 rc=0**: NC-Z0~Z4 5종, 주입마다 FAIL 시연(`ROW_COUNTER_MISMATCH`·`ANCHOR_FIELD_MISMATCH`·`INCOMPLETE_SUMMARY` 등)·복원마다 PASS, `failed=0` [부록 A-5] |
| **Z6** | **PASS** | §2 E-Z3 와 동일 실측(앵커 존재+4필드 bit-재현). 함의 범위는 §8 에 한정 서술 |

---

## 5. ★핵심 판정 — 무엇이 확정됐고 무엇이 미결인가

### 5-1. 확정 (이 단이 계약대로 준 것)

이 단의 계약은 **측정**이었다: §2 미확정 ①(hazard 목록·총수)과 ②(1행 밖 (a)/(b) 판별).
census 는 **스캔 범위 내에서 그 둘을 완결**했다:

1. **총수·목록 확정**: hazard(=`tau==0.0 && n_upper>0.0`, `lumina_plasma.c:14075`) 총수
   **221,382**(양 반복 동일), 전 행의 전 필드·준위쌍 식별자(lower/upper_global) 목록 확보.
   L6C 의 중도절단 하한 ≥1 이 실제로는 **5개 자릿수 위**였다. Z 분해: Fe 106,502 ·
   Co 105,685 · Ni 9,195 [부록 B].
2. **(a)/(b) 판별 확정 — 전건 (a)**: fa=1.0, 즉 전 hazard 행이 출판 플래그
   `A208_EXACT_ZERO`. coverage 결손 `(b)` 는 **스캔 범위 내 0** — 이중 근거:
   (i) fa=1.0 실측, (ii) coverage 류 validity(`A208_UNSAMPLED/MISS/STALE_GENERATION` 등)는
   침묵 skip 되지 않고 **fail-closed** 다 — 소비 루프 상류 필터가
   `status=RADEQ_STALE_OPACITY` 로 트랜잭션을 죽이고(`lumina_plasma.c:14944-14950`),
   진단 입구 가드는 `INVALID_ROW_PROVENANCE` 로 차단한다(`:14049`·`:14069`). 완주(BLOCKED 0)
   자체가 "스캔된 셀 전건 validity ∈ {VALID, EXACT_ZERO}" 의 증명이다.
3. **차단의 소유자 재확인(주입 없는 in-vivo 음성대조)**: 같은 봉인 base·같은 이온에서 캡처
   토글만으로 사망(L6C)↔완주(census)가 갈렸다(§4 Z5). 차단은 물리가 아니라 **opt-in 진단
   경로의 소산**이라는 §1 기전 특정이 세 런 대조(S4-III·L6·L6C)에 이어 **토글 페어로도**
   확증됐다.
4. **Z-2A 검증 기준선 확보**: Z-2A 의 게이트(Z3·Z4)가 소비할 기준선 — hazard 221,382 ·
   fa=1.0 · 앵커 행 · SUMMARY 총계(selected_rows 1201/992, total_scaled_emission
   0.544556800593765592605 / 0.633250617162941479828 [stderr:225081/:449539]) — 가 봉인됐다.

### 5-2. `fa=1.0` 이 의미하지 **않는** 것

- ① **출판·물질 정합의 표지이지 물리 인증이 아니다.** `A208_EXACT_ZERO` 는 "출판이 정당한
  정확 0 으로 등록했고 그 등록이 물질과 일관됨"이다. 그 0 을 만든 **bit-동일 준위쌍 인구
  자체가 물리적으로 옳은가**는 별개 문제이고, 오히려 §6 의 실측은 그것이 구조적(의심스러운)
  생산자의 소산임을 가리킨다 — 승계 미결.
- ② **스코프 밖 (b)=0 을 말하지 않는다.** 사전등록 §3 카운터 스코프 문구 그대로: 이 수는
  shell 0 · REQUESTED_TE · Z∈{26,27,28} · ion=1 · 뷰 창 안이다. shell≠0, 타 이온, 창 밖,
  타 위상의 coverage 는 이 census 가 보지 않았다.
  보완(감리-A 지적 4 반영): 사전등록 §3 목록 중 `UNRESOLVED_CANCELLATION`/`INVALID_INPUT`
  셀은 cancellation-census 노브 ON 이면 합산 전 `continue` 로 스캔을 우회할 수 있다
  (`lumina_plasma.c:15044-15077`; 노브 `LUMINA_A210_CANCELLATION_CENSUS`, `:13669-13672`·
  `:14889-14890`). 단 **이 런에서는 실측 공허**: 봉인 stderr 의 우회 마커
  `[A2-10][LINE-NET-CELL-BLOCKED]`(인쇄 `:15013`) **0건** · exports 에 CANCELLATION 노브
  부재 [판정자 실행: `grep -c` 양쪽 0, 부록 A-10b] ⟹ OFF 경로는 fail-closed
  (`:15078-15079` `RADEQ_SIGN_MISMATCH`/`RADEQ_NONFINITE`) — **총수 221,382 주장에 실손 없음.**
- ③ **★본선 물리 계약(A2-07 solve 가 J̄ 에 응답하는가)을 풀지 않았다.** 이 단은 그 측정을
  가로막던 차단을 **계측·분류**한 단이다. L6C 가 "계약 본체는 미결"로 폐합했고, 이 census
  후에도 그 미결은 **그대로 미결**이다 — 바뀐 것은 "수리(Z-2A)의 선택이 census 무의존으로
  이미 확정돼 있고(사전등록 §3 개정 1), 이제 그 발주 전제(Z-1 착지)가 충족됐다"는 점뿐이다.
  J̄ 응답의 측정은 Z-2A 가 착지해 L6C 형 런이 완주한 **뒤에야** 가능해진다.

### 5-3. census 가 Z-2 설계에 주는 것

- **A(진단 격리)의 유력 전건 "물리적으로 정당한 행"이 표본 1행에서 전 행으로 확장**됐다
  (전건 (a) + §6 의 덱 g 전수 실측). C(차단 유지)의 재등장 조건인 (b) 행은 스캔 범위 내
  0 — **C 는 이 데이터에서 근거를 얻지 못한다**(사전등록의 "전면 정책 기각" 유지).
- **B(행 건너뛰기) 병행 검토의 입력**: skip 대상이 될 행은 전부 (a)이고 수(221,382)·목록이
  확보됐다 — "건너뛴 수 보고"(NC3 정신)의 기대값이 사전에 고정된다. 병행 여부 결정은 Z-2
  발주 설계 몫(판정 범위 밖).
- **hazard 행 전건 `source_defined=0`** [부록 B: 221,382/221,382] — §1-1 의
  χ_eff==0 ⟺ tau==0 정확 관계와 정합. Z-2A 가 격리해야 할 대상의 형상이 균질하다.

---

## 6. ★사전등록 밖 실측 — 독립 재확인·대수 함의·처분

사전등록 §8 은 iter=2 의 이유로 *"iter1 SUMMARY 가 무료 관측을 준다 — bit-동일 준위쌍이 한
세대 solve 뒤에도 유지되는지(생산자 귀속의 입력)"* 를 **예고**했다. 예상되지 않았던 것은
관측 슬롯이 아니라 **그 내용**이다: 유지가 "전건" 이면서 동시에 인구는 "전건 이동"했다.

### 6-1. 실측 (판정자 독립 재계산 — 부록 B, stderr 정본·JSON 교차 일치)

- 두 반복의 hazard 행 집합: line 대칭차 **0** · 준위쌍 대칭차 **0** (완전 동일).
- 같은 행의 `n_upper` bit-동일: **0 / 221,382**. 상대변화 |n₁/n₀−1|:
  min **1.1686e-2** · 중앙값 **1.4416e-2** · max **1.9996e-2**; >1e-12 인 행 = 전건.
- **덱 실측 (전수, 가정 아님)**: 221,382 준위쌍 **전건 g_l == g_u**
  [`<RR>/input/model/line_list.csv` 의 (Z,ion,level_lower/upper) → `levels.csv` g 전수 대조;
  ROW 의 lower/upper_global → levels.csv 파일 순서 인덱스 매핑 불일치 **0**; 교차 확인
  `f_ul == −f_lu` 정확 성립 전건]. g 값 분포 상위: g=6 31,101쌍 · g=8 28,457 · g=7 27,051 ·
  g=5 24,467 · g=9 21,776 · g=4 21,481 · g=10 17,459 · g=11 12,604.
  보조 실측: `f_lu==0` **0건** · `f_lu<0` **0건** · min f_lu = 5.4985e-12 ·
  λ_cm ∈ [1.3415e-5, 2.0000e-4].
- 앵커 매핑 검증: global 2771→(26,1,100) g=8 · 5213→(26,1,2542) g=8 — L6C 판정문 §5-1 의
  실측과 일치.

### 6-2. 대체 생산자 배제 (τ=0·EXACT_ZERO 를 만들 수 있는 다른 경로들 — 전건 실측 배제)

소스 전수로 `tau_sobolev` 기록자는 `lumina_plasma.c` 의 다음뿐이다(타 파일 기록자 없음 —
`lumina_cmfgen.c:5453` 은 비교, `:7288` 은 지역 구조체):

| 기록자 | 내용 | 이 런에서의 배제 근거 |
|---|---|---|
| `:3243-3253`·`:19576-19580` | 원소 비활성 → 그 원소 **전 라인·전 셸** τ=0·EXACT_ZERO | **in-vivo 배제**: 같은 위상·같은 셸의 선정 행 인쇄에 Z=26 `tau_raw=57.35 tau_validity=1`, Z=27 `tau_raw=836.55`, Z=28 `tau_raw=1536.86` 실존 [부록 A-8] — 세 원소 모두 τ≠0 행이 있으므로 전-라인 fiat-zero 아님 |
| `:3257-3264` | `LUMINA_OPACITY_SKIP_Z` 마스크 → τ=0·EXACT_ZERO | env **부재**: `grep -c LUMINA_OPACITY_SKIP_Z <RR>/input/resolved_lumina.exports` = 0, 활성 배너(`:3151` "LUMINA_OPACITY_SKIP_Z active") stdout 부재 [부록 A-8]. (`LUMINA_NLTE_SKIP_Z="14"` 는 별개 노브·Si 뿐) |
| `:3266-3274`·`:3290-3296` | ion/level 미발견 → τ=0·**A208_MISS** | MISS 는 fail-closed(§5-1 ②) — 완주가 배제 |
| `:3309-3311`·`:3342-3344` (LTE 시드 뷰) | T_e 불량/인구 상태 불량 → τ=**NaN**·`A208_INVALID_TE`/`A208_INVALID_POPULATION` | (감리-A 지적 5 반영 — 초판 표 누락 정정, 행 판정자 직독) NaN 은 τ==0 을 만들 수 없고 validity 도 EXACT_ZERO 가 아니라 fail-closed(§5-1 ②) ⟹ **정확-영 생산자 소거에 무영향** |
| `:3347-3354` (LTE 시드 뷰) · **`:19611-19631` (NLTE ladder 재출판)** | `a208_signed_sobolev(_counted)` 산법 | **남는 유일한 생산자.** LTE(10020) 뷰는 hazard 를 만들 수 없다(n_l/n_u≈4.4e4 — L6C §2-② 의 PC-2 in-vivo 반증 승계) ⟹ hazard 셀의 유효 기록자는 `:19624` — 인구를 `nlte->nlte_level_populations`(커밋된 ladder, `:19621-19622`)에서, g 를 `atom->level_g`(덱, `:19611-19612`)에서 읽는다 [직독] |

**별칭 배제 (감리-A 지적 1 반영 — 초판에 없던 배제, 인용 행 전건 판정자 직독 재확인)**:
"n_l ≡ n_u 는 두 인덱스가 같은 배열 원소를 가리킨 별칭(self-aliasing)의 아티팩트다"라는
대안은 §6-3 추론을 무효화할 수 있으므로 배제가 필요하다 — 재출판은 인구를
`nlte_lo`/`nlte_up` 슬롯으로 읽는데(`:19621-19622`) 이는 ROW 가 찍는
`lower_global`/`upper_global` 과 **다른 인덱스 공간**이고, 부록 B 의 매핑 검사는 global
쪽만 덮기 때문이다. 배제 사슬: ① 두 슬롯은 `global_to_nlte_level[]` 로 유도되며(`:8925-8926`,
미매핑 음수는 `:8927` fail-closed) ② 그 맵의 구축(`:16427-16442`)은
`g = nlte_ion_level_offset[i] + cursor[i]++`(`:16431`) — 같은 ion 안에서 커서가 단조
증가라 서로 다른 global 준위는 서로 다른 nlte 슬롯을 받는다(**global→nlte 단사**) ③ hazard
쌍은 전건 같은 (Z,ion) 이고(§6-1 덱 실측) `lower_global≠upper_global` 은 봉인 stderr 전
ROW 실측이다(442,764행 중 `lower_global==upper_global` **0건** [판정자 실행, 부록 A-10b]).
⟹ `nlte_lo ≠ nlte_up` — **별칭 불능**, §6-3 의 n_l ≡ n_u 는 서로 다른 두 원소의 값 일치다.

### 6-3. 대수 함의 (여기부터 추론 — 실측/추론을 문장 단위로 구분)

- [소스 직독] 출판 산법: `value = coefficient·f_lu·λ_cm·t_exp·(n_l − (g_l/g_u)·n_u)`,
  `EXACT_ZERO ⟺ value==0.0` (`opacity_publication.c:26-29`·`:11-14`;
  `coefficient=SOBOLEV_COEFF=2.6540281e-02`, `lumina.h:36`).
- [실측] 전 hazard 쌍 g_l==g_u ⟹ [대수] `g_l/g_u == 1.0` 정확 ⟹ 괄호항 = `n_l − n_u`.
- [대수 — 언더플로 배제] `n_l − n_u ≠ 0` 이면 그 크기는 해당 등급 double 의 간격 이상
  (~1e-19, n_upper 실측 등급 4.4e-4~3.2e-3). 곱이 0.0 으로 언더플로하려면 전치 계수
  `coefficient·f_lu·λ·t` < ~5e-305 이어야 하는데, 실측 하한(f_lu ≥ 5.4985e-12 ·
  λ ≥ 1.3415e-5 · coefficient=2.654e-2 · t_exp>0 물리값)으로 **자릿수 ~280 이상 여유로
  배제**된다. ⟹ `value==0.0 ⟺ n_l == n_u` (double 정확 동일).
- [추론 = 위 실측+대수의 결합] 따라서 **양 반복 공히, 221,382쌍 전부에서 출판 시점의
  n_lower ≡ n_upper (bit-동일)** 이다. n_upper 는 반복 간 전건 ~1.4% 이동했으므로(실측),
  **n_lower 도 각 쌍에서 정확히 같은 bit 로 이동**했다.
  ⟹ **bit-겹침의 구조적 생산자는 NLTE solve 커밋 한 세대를 넘어 작동한다** — 겹침은 시드의
  화석이 아니라 solve 가 **관측된 세대 전이(iter0→iter1)에서 재생산한 성질**이다.
  (감리-A 지적 3 반영: 초판의 "매 세대 재생산" 은 관측 전이 1회(n=1)로부터의 외삽이라
  관측 범위로 낮췄다 — "전 세대에서 항상" 은 이 데이터가 지지하지 않는다.)
- [한정] n_lower 는 ROW 에 인쇄되지 않는다 — 위 문장은 실측이 아니라 출판 산법 경유
  추론이며, 전제는 "판정된 τ(=`opacity->tau_sobolev[cell]`, `:14945`)가 같은 세대의 출판값"
  이라는 것이다. **(감리-A 지적 2 반영 — 인용 기전 정정, 각 행 판정자 직독 재확인)** 초판은
  이 전제의 보증을 `:14944-14950` 의 "세대 가드"로 돌렸으나 **그 행의 실물은
  validity-계급 검사**(validity ∈ {`A208_VALID`,`A208_EXACT_ZERO`} ∧ `isfinite`)이지 세대
  비교가 아니다. 실제 보증 사슬은 셋이다:
  ① **배열 수준 K-FRESH** — 재출판 진입이 `tau_sobolev_require_refresh`(`:19519`)로 요구
  세대를 올리고 완료가 `tau_sobolev_mark_computed`(`:19655`)로 기록한다(정의 `:6893-6911`,
  불량 생산자 상태는 abort). 커밋 preflight 가 `computed==required` 를 fail-closed 로
  강제한다(`:20840-20847` 불일치 시 `NLTE_CANDIDATE_OPACITY_FAILED`; 번들 preflight
  `:21224-21226` 동일 검사 return 0).
  ② **재출판이 mapped·authorized 셀을 호출마다 재기록한다**(`:19583-19631` 직독 — 셀별
  메모·건너뛰기 없음, τ 기록 `:19629`·validity 기록 `:19631`; authorization 은 정적
  매핑(`:8905-8931`)·env 노브·호출자 전달 authority 배열에만 의존해 런 내 정적).
  ③ **시드(LTE 뷰)는 τ=0 을 만들 수 없다** — L6C 판정문 PC-2 오프라인 census(ion=1 후보
  1,048,375 전건 hazard=0, 도구·판정자·감리자 3자 일치) 승계.
  또 ROW 의 n_upper(판독 경로 `a209_upper_population_for_tau`, `:14959-14963`)와
  출판이 쓴 ladder 값의 동일성은 L6C 판정문이 WITNESS 행에서 전 필드로 확증한 사슬을
  승계한다(전 행 개별 재확증은 하지 않았다).
- [시사 — 주장 아님] hazard 가 **전건 등g 쌍**이라는 실측은 생산자 기전과 정합한다:
  bit-동일 인구 쌍이 있어도 g_l≠g_u 면 τ≠0 이라 hazard 에 안 잡힌다 — 즉 hazard =
  {bit-동일 인구 쌍} ∩ {g_l=g_u} 라는 그림과 모순이 없다. 등g 가 아닌 bit-동일 쌍의 존재
  여부는 이 census 가 **측정하지 않았다**(τ≠0 이라 스캔 밖).

### 6-4. 처분 (사전등록 밖 발견의 처분 — 규약: 조용한 대장 기재)

- **별건 단 신설 불요.** 이 발견의 자리는 이미 존재한다 — L6C 폐합이 남긴 승계 미결
  **«bit-동일의 구조적 생산자 귀속»** (L6C 판정문 §5 말미, 사전등록 §2-①). 본 census 가 그
  귀속의 입력을 완성했다: **준위쌍 221,382 목록 + "한 세대 보존·인구는 이동" 실측 + 전건
  등g 관측.** ZC-A 처분 문언("준위쌍 목록을 bit-동일 생산자 귀속에 인계") 그대로 인계한다.
- **조용한 대장 기재 1건**: "NLTE solve 커밋이 등g 준위쌍 221,382개의 인구를 bit-lockstep
  으로 갱신한다(상대이동 ~1.4%에서 겹침 유지)" — 수리·튜닝 제안 없음, Z-2 와 독립.
- 이 발견은 **Z-2A 의 선택을 바꾸지 않는다**(선택은 census 무의존으로 기확정 — 사전등록 §3).

---

## 7. 두 층 계량 — 배경 관찰만 (§10.2 준수)

사전등록 §10.2 가 판독 전에 확정한 대로, 같은 런 stderr 초반의 `cmf_fine` 센서스
(`[cmf_fine][SIGNED-MATERIAL-CENSUS]`, stderr:10 — `exact_zero_tau=81265890`,
`line_shells=27748410`(=signed_cells), INIT_SHARED_GAUSSIAN 위상)는 이 단의
`zero_opacity_emitting_rows` 와 **비교 불가**다(셸·이온·판별식·위상·단위 전부 상이,
포함관계 불성립). 본 판정의 §8 분기·게이트는 **A2-10 계량 단독**으로 판정했으며, 이 문단은
배경 기재다. cmf_fine 층의 정확-영 비율 74.55% 는 §10.2 의 산술을 그대로 인용하는 Z-2 설계
배경 입력이다. 두 수치를 비율로 엮은 문장은 이 판정문에 없다.

---

## 8. E-Z3/Z6 앵커 bit-재현의 함의 — 어디까지 주장 가능한가

**주장 가능**: ① K36 결정론 레인에서, 캡처 2종을 끈 채로도 iter0 REQUESTED_TE 의 **첫 hazard
행**이 L6C 실측과 **4필드 bit-동일 + 스캔 순서 동일**(첫 ROW, stderr:1242)로 재현됐다.
② 이는 Z6 의 설계 목적 — 캡처 토글(진단)이 물리를 건드리지 않았음의 **실측 검출기** — 를
통과한 것이다: 섭동이 있었다면 이 검출기가 울릴 기회가 있었고, 울리지 않았다.

**주장 불가**: ① 이것은 **스팟체크**다(1행×4필드) — "토글이 어떤 물리량도 어디서도 바꾸지
않았다"의 증명이 아니다. ② iter1 물리의 L6C 대조는 **원리적으로 불가**하다 — L6C 는 iter0
첫 hazard 에서 사망했으므로 비교 대상이 없다. ③ `candidate_rows` 는 L6C 가 중도절단 값
(66,110)만 남겨 총량 동등성 검증이 불가하다(749,644 ≥ 66,110 의 정합 확인이 전부 — E-Z4).
④ 결정론 일반("전 필드 bit-불변")은 Z4 급 byte-대조의 몫이며 그 게이트는 수리 후로
사전등록돼 있다.

---

## 9. 남은 미결과 다음 표적

| # | 미결 | 측정 가능해지는 조건 |
|---|---|---|
| 1 | **Z-2A 구현·발주** (진단 격리 — 선택 기확정, 사전등록 §3) | Z-1 착지 완료(이 판정) ⟹ 발주 전제 충족. §5 기대 변경집합 재등록 + 프리플라이트 재실행 의무. B 병행 여부는 §5-3 입력으로 설계 |
| 2 | **Z3** (III 가 Stage-4 row) + **Z2 제3원**(행 262210 전 필드 — Z-1r 이 capture 무관이므로 재현런이 자동 기록) | Z-2A 착지 후 III 재현런 1회 |
| 3 | **Z4** (IV byte 불변, 값 파일 한정 Tier1) | Z-2A 착지 후 IV 재현런 |
| 4 | **★본선 물리 계약 — A2-07 solve 의 J̄ 응답** (L6C 가 남긴 계약 본체) | Z-2A 로 차단 제거 → L6C 형 런(ion=1 커버 이온)이 REQUESTED_TE 를 **캡처 켠 채** 완주 → 반복 간 준위 응답 계측. 이 단은 그 길을 열었을 뿐 **아무것도 측정하지 않았다** |
| 5 | **bit-동일 구조적 생산자 귀속** (승계 미결) | 입력 완비(§6-4) — 준위쌍 221,382 목록·한세대 보존 실측·등g 관측. 별도 단 설계 필요(solve 내부 소스 감사 예상 — 런 불요 가능성) |
| 6 | 스캔 범위 밖 coverage (shell≠0·타 이온·창 밖) | ZC-B 미발동이라 의무 없음. 필요 시 스코프 확장 단 — 이 단의 범위 밖(사전등록 §3 말미) |

---

## 부록 A — 재현 명령 (전건 판정자 실행 확인, 2026-08-26)

`RR=/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1`

```bash
# A-1 job 계정
sacct -j 324388 -o JobID,State,Elapsed,MaxRSS,ExitCode
#  -> 324388 COMPLETED 02:42:56 (batch MaxRSS 32195676K) 0:0
#  주(감리-B NOTE#2): 이 조의 재현에는 Slurm DB 접근 권한이 필요하다 — Codex 샌드박스는
#  소켓 차단으로 재현 불능이었고, 운전석 대리 실행으로 일치 확인됨(324388 LUMINA_ZO_CENSUS_II
#  COMPLETED 02:42:56 / batch MaxRSS 32195676K ExitCode 0:0 syn102 a100).
cat $RR/model.rc                       # -> 0

# A-2 SUMMARY 2줄과 행 번호
grep -n "LINE-SATURATION-SUMMARY\]" $RR/stderr.log
#  -> 225081:(iter0, selected_rows=1201) / 449539:(iter1, selected_rows=992)
#     양쪽 complete=1 candidate_rows=749644 zero_opacity_emitting_rows=221382
#     physical_values_modified=0 clamp=0 floor=0 cap=0 jitter=0 repair=0

# A-3 ROW 계수 (전체/반복별/절단후)
grep -c "\[A2-10\]\[LINE-SATURATION-ZERO-OPACITY-ROW\]" $RR/stderr.log   # -> 442764
awk 'NR<225081' $RR/stderr.log | grep -c "LINE-SATURATION-ZERO-OPACITY-ROW"          # -> 221382
awk 'NR>225081 && NR<449539' $RR/stderr.log | grep -c "LINE-SATURATION-ZERO-OPACITY-ROW"  # -> 221382
awk 'NR>449539' $RR/stderr.log | grep -c "LINE-SATURATION-ZERO-OPACITY-ROW"          # -> 0 (grep rc=1)

# A-4 차단 마커 0 (각 명령 grep rc=1 = 0건)
grep -c "LINE-SATURATION-BLOCKED" $RR/stderr.log        # -> 0
grep -c "INDEPENDENT_SPROBE_UNDEFINED" $RR/stderr.log   # -> 0
grep -c "DET_FLIGHT_FATAL" $RR/stderr.log               # -> 0
# (감리-B NOTE#4 반영: 초판 cut -c1-80 은 line= 값이 잘려 보이지 않았다 — 폭 확장 후 재실행)
grep -n -m1 "LINE-SATURATION-ZERO-OPACITY-ROW" $RR/stderr.log | cut -c1-120
#  -> 1242:[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW] phase=REQUESTED_TE shell=0 line=669992 Z=26 ion=1 lower_global=2771 upper

# A-5 판독기 음성대조 (grammar-debug, repo 루트에서)
python3 scripts/analyze_a210_zero_opacity_census.py --selftest; echo rc=$?
#  -> A210_ZERO_OPACITY_CENSUS_ANALYZER_SELFTEST PASS controls=NC-Z0,NC-Z1,NC-Z2,NC-Z3,NC-Z4 failed=0 / rc=0

# A-6 Z5 ON-팔 (봉인 L6C — 읽기만)
L6C=/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/l6c_20260822T154507Z_cover
grep -c "INDEPENDENT_SPROBE_UNDEFINED" $L6C/stderr.log   # -> 1
grep -m1 "INDEPENDENT_SPROBE_UNDEFINED" $L6C/stderr.log
#  -> [A2-10][LINE-SATURATION-BLOCKED] reason=INDEPENDENT_SPROBE_UNDEFINED ... candidate_rows=66110
#     ... complete=0 ... zero_opacity_emitting_rows=1
# (감리-B NOTE#6 반영: 초판 `grep slurm` 은 slurm_environment_before_sanitize.txt 도 출력했다
#  — 패턴을 좁혀 재실행)
ls $L6C | grep '^slurm-'                                 # -> slurm-322449.out (그 한 줄뿐)

# A-7 Z2 IV 소급 (봉인 L6 — 읽기만)
L6=/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/sprim_l6_20260821T054111Z_probe
sed -n '1893p;3581p' $L6/stderr.log | grep -o "zero_opacity_emitting_rows=[0-9]*\|complete=[0-9]*\|target_ion=[0-9]*"
#  -> target_ion=3 / complete=1 / zero_opacity_emitting_rows=0  (두 줄 모두)

# A-8 대체 생산자 배제
grep -c "LUMINA_OPACITY_SKIP_Z" $RR/input/resolved_lumina.exports   # -> 0 (rc=1)
grep -c "LUMINA_OPACITY_SKIP_Z active" $RR/stdout.log               # -> 0 (rc=1)
grep -m2 "\[A2-10\]\[LINE-SATURATION-ROW\] .*Z=26" $RR/stderr.log | cut -c1-420
#  -> ... Z=26 ion=1 ... tau_raw=57.353449590990671 ... tau_validity=1 ...
# (동일 명령 Z=27 -> tau_raw=836.5490056711709, Z=28 -> tau_raw=1536.8554822066822, 둘 다 tau_validity=1)

# A-9 per-ion 후보 분해
grep -m3 "UNION-ION-SUMMARY" $RR/stderr.log | grep -o "Z=[0-9]* ion=1 candidate_rows=[0-9]*"
#  -> Z=26 356479 / Z=27 353694 / Z=28 39471  (합 749644)

# A-10 소스 인용 행 검증 (HEAD b7accae; src 는 e500e29 와 diff 공백)
git diff --stat e500e29 b7accae -- src/          # -> (공백)
grep -n "if(tau==0.0&&n_upper>0.0){" src/lumina_plasma.c              # -> 14075
grep -n "status = RADEQ_STALE_OPACITY" src/lumina_plasma.c            # -> 14948
grep -n "INVALID_ROW_PROVENANCE" src/lumina_plasma.c | head -1        # -> 14069
sed -n '8,9p' src/opacity_publication.h    # -> A208_VALID = 1, / A208_EXACT_ZERO,
sed -n '26,29p' src/opacity_publication.c  # -> difference = n_lower - (g_lower/g_upper)*n_upper; ...
grep -n "SOBOLEV_COEFF" src/lumina.h       # -> 36: 2.6540281e-02

# A-10b 감리 반영분 검증 (2026-08-26 반영 시 전건 판정자 실행)
# (지적 1 — 별칭 배제 인용 행)
grep -n "nlte_ion_level_offset\[i\] + cursor\[i\]++" src/lumina_plasma.c   # -> 16431
grep -n "a.nlte_lo=nlte->global_to_nlte_level" src/lumina_plasma.c        # -> 8925
awk '/\[A2-10\]\[LINE-SATURATION-ZERO-OPACITY-ROW\]/{lo="";up="";for(i=1;i<=NF;i++){ \
  if($i~/^lower_global=/)lo=substr($i,14);else if($i~/^upper_global=/)up=substr($i,14)} \
  n++;if(lo==up)e++}END{printf "rows=%d lower_eq_upper=%d\n",n,e+0}' $RR/stderr.log
#  -> rows=442764 lower_eq_upper=0
# (지적 2 — K-FRESH 사슬 인용 행)
sed -n '6893,6911p' src/lumina_plasma.c    # -> require_refresh/mark_computed 정의(불량 시 abort)
grep -n 'tau_sobolev_require_refresh(opacity, "nlte_update_tau_sobolev")' src/lumina_plasma.c  # -> 19519
grep -n 'tau_sobolev_mark_computed(opacity, "nlte_update_tau_sobolev")' src/lumina_plasma.c    # -> 19655
sed -n '20840,20847p' src/lumina_plasma.c  # -> computed!=required 시 NLTE_CANDIDATE_OPACITY_FAILED
sed -n '21224,21226p' src/lumina_plasma.c  # -> 번들 preflight 동일 검사 return 0
grep -n "tau_view.value\|tau_view.validity" src/lumina_plasma.c            # -> 19629 / 19631
# (지적 4 — cancellation-census 우회 공허 실측)
grep -n "LUMINA_A210_CANCELLATION_CENSUS" src/lumina_plasma.c              # -> 13670
grep -c "LINE-NET-CELL-BLOCKED" $RR/stderr.log                             # -> 0 (rc=1)
grep -c "CANCELLATION" $RR/input/resolved_lumina.exports                   # -> 0 (rc=1)
# (지적 5 — NaN 기록자 행)
grep -n "A208_INVALID_TE\|A208_INVALID_POPULATION" src/lumina_plasma.c     # -> 3311 / 3344
```

## 부록 B — 판정자 재계산 스크립트 (grammar-debug 에서 실행 확인; 읽기 전용)

실행: `ssh grammar "ssh grammar-debug 'python3 -u -'" < <이 스크립트>` (경과 수 분).

```python
import statistics
RR="/gpfs/kjhan/lumina/a210_zero_opacity_z1r_a100x2_k36/zo_20260826T050021Z_census_ion1"
ROW="[A2-10][LINE-SATURATION-ZERO-OPACITY-ROW]"; SUM="[A2-10][LINE-SATURATION-SUMMARY]"
it=0; R=[{},{}]
for ln in open(RR+"/stderr.log",errors="replace"):
    if ln.startswith(ROW):
        d=dict(t.split("=",1) for t in ln[len(ROW):].split() if "=" in t)
        R[it][int(d["line"])]=d
    elif ln.startswith(SUM): it+=1
n0,n1=len(R[0]),len(R[1])
fa=sum(1 for d in R[0].values() if d["tau_validity"]=="2")/n0
a=R[0][669992]
anchor_ok=(a["tau_validity"]=="2" and float(a["n_upper"]).hex()=="0x1.d1534dba20c26p-12"
 and float(a["A_ul"]).hex()=="0x1.0b6516044e436p-1"
 and float(a["emission_per_sr"]).hex()=="0x1.3adcec9caadc0p-52")
sd=len(set(R[0])^set(R[1]))
pd=len({(d["lower_global"],d["upper_global"]) for d in R[0].values()}
      ^{(d["lower_global"],d["upper_global"]) for d in R[1].values()})
bit=sum(1 for k in R[0] if float(R[0][k]["n_upper"]).hex()==float(R[1][k]["n_upper"]).hex())
rel=sorted(abs(float(R[1][k]["n_upper"])/float(R[0][k]["n_upper"])-1) for k in R[0])
print(f"iter0={n0} iter1={n1} fa={fa} anchor_bit_ok={anchor_ok}")
print(f"line_symdiff={sd} pair_symdiff={pd} nupper_bit_identical={bit}")
print(f"rel min={rel[0]:.4e} med={statistics.median(rel):.4e} max={rel[-1]:.4e}")
g={}; order=[]
f=open(RR+"/input/model/levels.csv"); h=f.readline().strip().split(",")
iZ,iI,iL,iG=h.index("atomic_number"),h.index("ion_number"),h.index("level_number"),h.index("g")
for ln in f:
    p=ln.split(","); k=(int(p[iZ]),int(p[iI]),int(p[iL])); order.append(k); g[k]=float(p[iG])
f=open(RR+"/input/model/line_list.csv"); h=f.readline().strip().split(",")
ix={k:h.index(k) for k in("atomic_number","ion_number","level_number_lower",
                          "level_number_upper","line_id","f_ul","f_lu")}
eq=neq=fs=fz=mm=fnd=0; flu_min=None
for ln in f:
    p=ln.split(","); lid=int(p[ix["line_id"]])
    if lid not in R[0]: continue
    fnd+=1; Z=int(p[ix["atomic_number"]]); io=int(p[ix["ion_number"]])
    ll=int(p[ix["level_number_lower"]]); ul=int(p[ix["level_number_upper"]])
    gl,gu=g[(Z,io,ll)],g[(Z,io,ul)]
    eq+=gl==gu; neq+=gl!=gu
    d=R[0][lid]
    if order[int(d["lower_global"])]!=(Z,io,ll) or order[int(d["upper_global"])]!=(Z,io,ul): mm+=1
    ful,flu=float(p[ix["f_ul"]]),float(p[ix["f_lu"]])
    fs+=ful==-flu; fz+=flu==0.0
    if flu>0.0 and (flu_min is None or flu<flu_min): flu_min=flu
print(f"deck: found={fnd} g_eq={eq} g_neq={neq} map_mismatch={mm} "
      f"ful_eq_neg_flu={fs} flu_zero={fz} flu_min_pos={flu_min:.6e}")
```

실행 확인 출력 (2026-08-26, grammar-debug):

```
iter0=221382 iter1=221382 fa=1.0 anchor_bit_ok=True
line_symdiff=0 pair_symdiff=0 nupper_bit_identical=0
rel min=1.1686e-02 med=1.4416e-02 max=1.9996e-02
deck: found=221382 g_eq=221382 g_neq=0 map_mismatch=0 ful_eq_neg_flu=221382 flu_zero=0 flu_min_pos=5.498500e-12
```

보조 실측(부록 B 확장판으로 확인, 본문 인용분): Z 분포 {26: 106502, 27: 105685, 28: 9195} ·
srce_chk/exact_zero_provenance/source_defined 전건 0 · tau_raw/effective_tau 비영 0 ·
n_upper==0 0건 · n_upper 범위 iter0 [4.437689e-04, 3.129367e-03] / iter1
[4.526424e-04, 3.174480e-03] · 판독기 JSON raw 블록 스트리밍 교차: 행수·tau_validity 동일,
n_upper 대 stderr 불일치 0, JSON 단독 재계산 bit-동일 0/221,382.

---

## 부록 C — 감리 스탬프 (개정17 두 겹, 2026-08-26)

**감리-A — Fable fresh** (판정과 별개 컨텍스트, 개정14 ④): **통과** · **차단 0** ·
**지적 5**(전건 비차단 — 본판에 전건 반영, 판정·분기 불변).

| 지적 | 요지 | 반영 위치 |
|---|---|---|
| 1 | 같은-배열-원소 별칭 대안의 배제 논증이 §6-2·§6-3 에 없었다 | §6-2 말미 「별칭 배제」 신설 + 부록 A-10b 실측(442,764행 중 `lower_global==upper_global` 0건) |
| 2 | §6-3 `[한정]` 의 세대 보증 인용 부정확 — `:14944-14950` 실물은 validity-계급 검사 | §6-3 `[한정]` 을 실제 보증 사슬 3항(K-FRESH+preflight · 호출마다 재기록 · PC-2 시드 배제)으로 교체 + 부록 A-10b |
| 3 | "solve 가 매 세대 재생산" 은 관측 전이 1회(n=1)의 외삽 | §6-3 결론 문장을 관측 범위(iter0→iter1)로 하향 |
| 4 | §5-2 ② 가 cancellation-census 노브의 셀 우회 가능성을 누락 | §5-2 ② 보완 1항 — 이 런 실측 공허(마커 0·노브 부재) 확인, 총수 주장 실손 없음 |
| 5 | §6-2 기록자 전수 주장이 시드-뷰 NaN 기록자 2곳을 누락 | §6-2 표에 `:3309-3311`·`:3342-3344` 행 추가(정확-영 소거 무영향 명시) |

**감리-A 후속 권고 (명기)**: 생산자 귀속 단(§9 #5)의 사전등록에 ① 지적 1 의 global→nlte
**단사성 실측**(부록 A-10b awk — nlte 인덱스 공간 별칭 배제)과 ② 지적 2 의 **K-FRESH 보증
사슬**(`:6893-6911`·`:19519`/`:19655`·`:20840-20847`·`:21224-21226`)을 **승계 기재**할 것.

**감리-B — Codex read-only** (계열 분리, 개정17): **통과** · **차단 0** · **NOTE 4**(전건
비차단 — 본판에 전건 반영) · 재현 대상 **13건 중 재현됨 9 · 불일치 3 · 재현불능 1**
(불일치 3 = NOTE#4·#6·#12, 전건 본판 정정으로 해소; 재현불능 1 = NOTE#2 sacct —
샌드박스 Slurm DB 소켓 차단, 운전석 대리 실행으로 일치 확인).

| NOTE | 요지 | 반영 위치 |
|---|---|---|
| #2 | sacct 재현에 Slurm DB 접근 권한 필요(해소됨 — 대리 실행 일치) | 부록 A-1 주석 |
| #4 | `cut -c1-80` 실제 출력이 `line=6` 에서 절단 — 주석의 `line=669992` 미표시 | 부록 A-4 — 폭 `-c1-120` 확장 후 재실행, 실제 출력으로 교체 |
| #6 | `ls … grep slurm` 이 `slurm_environment_before_sanitize.txt` 도 출력 | 부록 A-6 — 패턴 `^slurm-` 로 협착 후 재실행 |
| #12 | `A210_ZO_CENSUS_ACCEPT` 의 거처를 `zero_opacity_census.stdout` 으로 오기(실제 = `ZERO_OPACITY_CENSUS_VERDICT.txt`) | §2 E-Z1 행 정정(양 파일 내용 재확인 병기) |

**폐합 선언**: 개정17 폐합 조건 — 감리-A·감리-B **두 겹 모두 통과** — **충족**.
분기 `ZC-A` · E-Z 판정 · §5 확정/미결 분류는 감리 전후 **불변**이다.

---

## 한 줄 결론

**분기 ZC-A(fa=1.0 — 판정자 재계산으로 재검증)** · **확정**: Z-1r census 완주로 hazard
총수·목록(221,382행)과 전건-(a) 판별이 스캔 범위 내 완결됐고, 캡처 토글 페어가 차단의
소유자=진단임을 주입 없이 시연했으며, 사전등록 밖 실측(인구 ~1.4% 이동에도 정확-영 전건
유지)은 덱 g_l=g_u 전수 실측과 출판 대수로 «bit-겹침이 solve 한 세대를 넘어 재생산됨»을
확립해 생산자 귀속의 입력을 완성했다 · **미결**: 본선 물리 계약(A2-07 solve 의 J̄ 응답)은
이 단이 풀지 않았다 — Z-2A(선택 기확정)의 구현·착지, Z3·Z4 게이트, 그리고 bit-동일 구조적
생산자 귀속이 열려 있다.
