# 단 사전등록 — SH-A209-IDSEAL: A2-09 방출률 발행체의 신원 봉인 (2026-08-20)

저자: Fable(사전등록, 분담 개정14) · 발단: `docs/VERDICT_DET_PHYSCMP_P1_2026-08-20.md`(측정 단 PASS)
발견 정본: `docs/FINDING_A209_IDENTITY_FIELDS_UNPOPULATED_2026-08-20.md`
소속: **SH-** (발행체는 두 팔이 공유 — `a209_publish_cpu_emissivity` 를 MC/DET 양 팔이 부른다,
[실측] `src/lumina_cuda.cu:586` · `src/lumina_plasma.c:9506` · `src/nlte_population_candidate.c` 번들 경로).
CMFGEN 발자국 규약: **비적용** — 신원 해시는 계약 인프라이며 CMFGEN 대응물이 없다.

직전 단(DET-PHYSCMP P-1)은 **측정 단**이었다. 이 단은 그 측정이 특정한 결함의 **수리 단**이다.

---

## 계약 (하나)

> **A2-09 방출률 발행체는 발행(커밋) 시점에 자기 신원 — 주파수 격자·원자모형·소스항 의미론 —
> 을 SPEC 의 세 sha256 필드에 스스로 계산해 봉인하며, 봉인할 수 없으면 이름 있는 사유로
> 발행을 거부한다.**

계약 1개 = 커밋 1개. 세 필드를 한 단에 묶는 근거는 §2-Q4.

---

## 1. 실측 사실 (수리 전 상태)

| 항목 | 실측 |
|---|---|
| SPEC 요구 | `docs/SPEC_A2_09_10_V1.md:100-102` 가 `atomic_model_sha256`·`grid_manifest_sha256`·`source_manifest_sha256` 를 `CpuEmissivityPublication` 의 필드로 규정 [실측] |
| 선언 | `src/emissivity_publication.h:26-27` [실측] |
| writer | 생산 코드 **0건**(셋 다). `a209_publication_init` 의 `memset`(`emissivity_publication.c:71`)이 NUL 64개로 확정 [실측] |
| 본보기 | 4번째 필드 `cdf_manifest_sha256` 만 채워진다 — `emissivity_publication.c:115-120`, 도메인 문자열+크기+IEEE754 비트 관례 [실측] |
| 발화 | `physics_comparison.c:184-207`(site=133)이 `em->grid_manifest_sha256` hex64 를 요구, 판정런 320568 에서 `grid_manifest_sha256_valid=0` 단독 발화 [실측] |
| 생산 커밋 지점 | `a209_publication_commit_counted`(`emissivity_publication.c:131`) — src 전체에서 생산 호출자는 `lumina_plasma.c:9376` **하나** [실측] |
| 생산 채움 지점 | `a209_publish_cpu_emissivity_impl`(`lumina_plasma.c:9103`)이 후보를 만들고 `nu_edge` 를 `opacity->cpu_opacity.frequency_edges` 에서 memcpy(`:9132`), cdf 빌드+커밋(`:9375-9376`) [실측] |
| 소비자 1 | `physics_comparison.c:132`(hex64 게이트)·`:503,513-514`(JSON 출력) — `grid_manifest_sha256` 만 [실측] |
| 소비자 2 | `scripts/check_det_convergence.py:279-284` — `grid_manifest_sha256` 를 반복 간 **불변량**으로 검사 [실측] |
| 소비자 3 | `scripts/compare_physics_snapshots.py:136-141` — hex64 형식 검사 [실측] |
| 결함 은폐 기전 | 픽스처 3곳이 생산 경로 밖에서 손 채움: `tests/physics_comparison_selftest.c:169`(`fill_hash(...,'b')`) · `tests/physics_comparison_regrid_selftest.py:78`(`"d"*64`) · `tests/det_convergence_selftest.py:104`(`"c"*64`) [실측] |
| 정본 원자모형 해시 | `population_atomic_model_sha256`(`population_contract.c:84`). 생산에서는 `population_partition_build` 가 이를 호출해 `atom->partition_stamp.atomic_model_sha256` 에 봉인(`population_contract.c:159-165`); 온도 발행체가 그 값을 그대로 복사한다(`lumina_plasma.c:15967-15968`) [실측] |
| 유사 선례 | `a210_geometry_sha256`(`radeq_publication.c:228`) — 도메인 문자열+개수+IEEE754 비트, **export 된 정본 헬퍼**를 생산과 시험이 공유 [실측] |

### ★성패 기준 (판정 기준 경고의 수용)

`check_det_convergence.py` 의 불변량 검사와 site=133 의 hex64 검사는 **어떤 상수 hex64 도
공허하게 통과**시킨다. 따라서 이 단의 성패 기준은 "게이트를 통과하는가"가 아니라
**"격자를 바꾸면 값이 바뀌는가"** 다. 이를 두 조각으로 분해해 강제한다:

- **NC1**(§5): 생산 해셔가 격자 1비트 변화에 다른 값을 냄을 생산 커밋 경로에서 시연.
- **B2**(§6): 판정런의 해시가 그 런의 **실제 격자의 함수**임을 독립 재계산(Python, hashlib)으로 입증.

둘이 합쳐져야 "격자를 바꾸면 바뀐다"가 생산 경로에서 성립한다. 상수 채움은 NC1 도 B2 도
통과할 수 없다.

---

## 2. 설계 결정 (증거 패킷 §6 의 여섯 질문)

### Q4 (범위) — 세 필드를 **한 단에 다** 채운다

- 계약은 필드 셋이 아니라 **신원 봉인 하나**다. 셋은 SPEC 에서 한 블록으로 선언됐고
  (`SPEC:100-102`), 같은 `memset` 으로 함께 죽었으며, 강제 지점(커밋)이 하나다.
- 나눠서 하면 같은 커밋 함수(`a209_publication_commit_counted`)를 세 단이 잇달아 고치게
  되어 "계약 1개=커밋 1개" 대응이 오히려 흐려진다.
- fail-closed 를 "신원 전체" 에 일관 적용할 수 있다 — grid 만 채우면 커밋 게이트는 3필드 중
  1개만 검사하는 반쪽이 되고, 나머지 둘은 "선언됐으나 writer 0" 상태가 그대로 남는다
  (FINDING 이 규정한 결함 상태의 존속).
- FINDING §6 의 수리 계약 문장도 셋을 한 계약으로 묶었다 [실측].

### Q1 (grid 정의역) — `n_bins` + `nu_edge[0..n_bins]` 전체, IEEE754 비트

- 도메인 문자열(버전 포함) + BE u64 `n_bins` + `nu_edge` 각 원소의 IEEE754 비트(BE u64).
  정확 직렬화는 부록 A. `a210_geometry_sha256` 관례를 그대로 따른다.
- **격자 생성 규약(로그/선형·`nu_min`·`d_log_nu`)은 봉인하지 않는다.** 근거: edge 배열이
  격자의 완전한 외연 정의다 — 같은 비트의 edge 를 내는 두 규약은 같은 격자다. 규약을
  따로 봉인하면 물리적으로 동일한 격자가 다른 신원을 갖는 오탐을 만든다.
- `n_shells` 미포함 — 격자가 아니라 기하이며 `geometry_sha256`(온도 발행체) 소관.
- 해시 전 검증(fail-closed): `nu_edge` 비NULL, `n_bins>=1`, 전 edge 유한·양수·순증가.
  이는 클램프가 아니라 **신원 스키마 검증**이다 — 주파수 격자의 정확해가 이를 위반할 수
  없다(음/역전 주파수 edge 는 격자가 아님; `physics_comparison.c` 도 동일 조건을 이미
  INVALID_GRID 로 거부한다 [실측]).

### Q2 (source_manifest 의 정의) — SPEC 은 정의를 주지 않는다; **이 단이 v1 을 정의한다**

- [실측] `SPEC_A2_09_10_V1.md` 전문 grep: `source_manifest_sha256` 출현은 :102 의 필드
  나열 한 줄뿐. 정의역 규정 없음. (:113-118 이 소스항 의미론 — true total 과 scattering
  source 의 분리, CMFGEN `ETA_DATA` 포함규약의 manifest 기록 — 을 요구하는 것이 유일한
  단서다.)
- 결정: **소스항 의미론 선언의 봉인**으로 v1 정의. 도메인 문자열(의미론 관례를 문자열에
  고정: `eta_true = bb+bf+ff`, scattering 분리, comoving, per-sr) + `channel_mask`.
  정확 직렬화는 부록 A. 관례가 바뀌면 도메인 문자열 버전이 바뀐다(v2) — cdf 해시와 같은
  버전 규약.
- 범위 밖으로 빼지 않는 근거: 빼면 "선언됐으나 writer 0" 인 검증불가 고아가 존속하고,
  커밋의 fail-closed 가 신원 3필드 중 2개만 지키는 반쪽이 된다. CMFGEN `ETA_DATA` 직접
  대조가 시작되면 포함규약 필드가 필요해질 수 있다 — 그때 v2 로 확장한다(부록 B-1).
- 아는 한계를 적는다: v1 의 정보량은 사실상 `channel_mask` 하나다(나머지는 관례 상수).
  그래도 mask 변화에 민감하고(NC4), 무엇보다 **"이 발행이 어떤 소스 의미론 선언 아래
  나왔는가"를 위조 불가능하게 만든다**는 계약 목적을 충족한다.

### Q3 (atomic_model) — 정본 헬퍼 출력의 **재사용**, 재해시 금지

- `a209_publish_cpu_emissivity_impl` 이 `atom->partition_stamp.atomic_model_sha256` 를
  복사한다. **재해시하지 않는다** — 같은 값을 두 곳에서 따로 계산하면 이중권위이며,
  이 캠페인이 population 음수해의 원인으로 실증한 바로 그 병리다(stage-total lock
  이중권위). 온도 발행체가 이미 같은 방식으로 복사한다(`lumina_plasma.c:15967-15968`) —
  동종 대 동종.
- 복사 전 결박 검증(fail-closed, 셋 다): `partition_stamp.status==POP_OK` ·
  `partition_stamp.computed_population_generation == c.population_generation`(= 그 시점의
  `atom->population_committed_generation`) · 해시 hex64.
  [실측 근거] 번들 경로에서 stamp 는 `required_population_generation` 으로 빌드되고
  (`nlte_population_candidate.c:379-383`), 그 뒤 단계가
  `population_committed_generation == required_population_generation` 를 게이트한다
  (`:430-433`); 직접 경로의 `compute_partition_functions` 는 population 트랜잭션 안에서
  돌며 실패 시 stamp 가 원복된다(`lumina_plasma.c:7102,7119,7128`). ⟹ 커밋된 상태에서
  등식이 성립한다. [추정] 전 경로 전수는 확인하지 못했다 — 그래서 이 결박은 가정이 아니라
  **게이트**다: 등식이 깨지는 경로가 실재하면 판정런이 이름 있는 사유로 죽고, 그것이
  발견이 된다(§6 철회·분기 참조).
- 아는 한계를 적는다: 정본 헬퍼의 정의역은 준위/분배 소속/topion catalog 다
  (`population_contract.c:84-108`) — **선 목록(A_ul·ν·소속)은 봉인되지 않는다.**
  이는 정본 해시 자체의 정의역 한계이며 온도 발행체도 똑같이 안고 있다. 이 단에서
  별도 도메인을 만들면 이중권위가 되므로 만들지 않는다. 대장 기재 항목(부록 B-2).

### Q5 (음성 대조의 주입 지점) — **생산 커밋 체인 안에서** 주입한다

이 결함을 20일 숨긴 방법이 "픽스처가 생산 경로 밖에서 손 채움"이었으므로, NC 는 전부
`a209_publication_init → 채움 → a209_build_reemit_cdf_counted → a209_publication_commit_counted`
라는 **실제 생산 체인**을 통과시킨다. 특히 NC3 은 **역사적 결함 그 자체**(atomic 필드
NUL)를 주입해 커밋이 거부함을 시연한다 — "주입 결함으로 FAIL 시연" 규약의 문자 그대로의
이행이다. 상세 §5.

### Q6 (기존 픽스처의 처분) — 손 채움 → **정본 정의역 유도**로 교체 (3곳 전부)

| 픽스처 | 처분 |
|---|---|
| `tests/physics_comparison_selftest.c:169` | `fill_hash(...,'b')` → export 된 정본 헬퍼 `a209_grid_manifest_sha256(target_edge, NB, em.grid_manifest_sha256)` 호출로 교체. 같은 파일이 이미 `a210_geometry_sha256` 로 tep.geometry 를 채우는 것과 같은 형태(`:199-200` [실측]) — 선례 실존. site=133 음성대조(`:294` 의 `'!'` 주입)는 **유지**(소비자 게이트 시험으로서 정당) |
| `tests/physics_comparison_regrid_selftest.py:78` | `"d"*64` → 픽스처 자신의 `freq_edges` 에서 부록 A 직렬화(hashlib)로 유도 |
| `tests/det_convergence_selftest.py:104` | `"c"*64` → 동일하게 픽스처 edge 에서 유도 |

한계를 정직하게 적는다: 파이썬 checker 시험은 C 생산자를 원리적으로 못 부른다 — "생산
경로 경유"가 아니라 "정본 정의역 유도"까지가 가능 범위다. 생산자 부재를 다시 숨기는 것을
막는 **실효 장치**는 픽스처가 아니라 (1) 해시 계산을 커밋 안에 두어 writer 를 잊을 수
없게 한 구조, (2) 판정런 B2 의 독립 재계산이다. te-측 손 채움(`tep.atomic_model` 'a'*64
등, `det_stage12_selftest.c` 포함)은 이 단 범위 밖 — 부록 B-3 에 기재만 한다.

### (추가 판단) MC 레인 접촉 여부 — 무접촉은 **불가능하며 요구하지 않는다**

발행체와 발행자는 두 팔 공유다. 이 수리는 MC 레인의 발행에도 신원 필드를 채우고 신원
이벤트 줄 1개를 추가한다. 대신 요구하는 것은 **물리값 무접촉**(P6): eta/cdf/샘플러의
수치 계산 라인 0 접촉, 기존 발행 성공/거부 조건 불변(신원 검증 추가분 제외). site=133
게이트 자체는 `LUMINA_PHYSICS_COMPARISON_DIR` 미설정 런(통상 MC 런)에서는 경로가 실행되지
않는다 [실측 `physics_comparison.c:564-565`].

---

## 3. 기대 변경집합 (여기 없는 변경은 위반)

| 파일 | 변경 내용 |
|---|---|
| `src/emissivity_publication.h` | 정본 헬퍼 2개 선언: `int a209_grid_manifest_sha256(const double *nu_edge, size_t n_bins, char out[65]);` · `int a209_source_manifest_sha256(unsigned channel_mask, char out[65]);` (반환 0=성공, 비0=거부, 거부 시 `out[0]='\0'`). `A209Counters` 말미에 `uint64_t identity_seal_failures;` 추가 |
| `src/emissivity_publication.c` | (a) 헬퍼 2개 구현 — **기존 `A209Sha` 기계만 사용**, 직렬화는 부록 A 그대로. (b) `a209_publication_commit_counted`: 기존 게이트(부분발행·closure·cdf 세대) **뒤**, swap **앞**에 신원 봉인 블록 — grid·source 를 후보에서 계산해 써넣고(선채움 값은 덮임 — 커밋이 유일 writer), atomic 은 hex64 검증만. 실패 시 `identity_seal_failures++` + `[A2-09][BLOCKED] reason=EMISS_IDENTITY_{ATOMIC_UNSEALED\|GRID_INVALID\|SOURCE_MASK_EMPTY}` stderr 1줄 + return 5 (발행체 무변경). (c) 성공 swap 직후 신원 이벤트 줄 1개(stderr): `[A2-09][IDENTITY] generation=… n_bins=… channel_mask=0x… grid_manifest_sha256=… atomic_model_sha256=… source_manifest_sha256=…`. (d) counters print 에 새 카운터 추가 |
| `src/lumina_plasma.c` | `a209_publish_cpu_emissivity_impl` 에서 세대 필드 세팅 직후(≈`:9120`), 무거운 루프 **앞**에: §2-Q3 의 3중 결박 검증 후 `atom->partition_stamp.atomic_model_sha256` → `c.atomic_model_sha256` memcpy. 실패 시 `[A2-09][BLOCKED] reason=EMISS_IDENTITY_ATOMIC_STAMP_INVALID …`(stamp status·두 세대·hex64 실측값 포함) + return 5 |
| `tests/a2_09_emissivity_selftest.c` | 기존 성공 커밋 2곳(`:81`, `:94` 부근)의 후보에 정본 경로로 atomic 채움(합성 `PopulationAtomicView` → `population_atomic_model_sha256`; 리터럴 채움 금지) + NC1~NC4(§5) + 신원 이벤트/헬퍼 해시 stdout 출력(P4 용) |
| `tests/physics_comparison_selftest.c` | §2-Q6 처분 (fill_hash 1곳 → 헬퍼) |
| `tests/physics_comparison_regrid_selftest.py` | §2-Q6 처분 (부록 A 직렬화의 파이썬 유도 함수 + 사용) |
| `tests/det_convergence_selftest.py` | §2-Q6 처분 (동일) |
| `scripts/run_a2_09_selftest.py` | P4: selftest 가 출력한 헬퍼 해시(고정 소형 격자·고정 mask)를 hashlib 로 독립 재계산해 일치 검사 |
| `scripts/verify_a209_grid_manifest.py` | **신설.** 판정런 B2 도구: manifest JSON + spectral CSV 를 받아 edge 를 복원(`nu_lo_Hz` 열 + 마지막 `nu_hi_Hz`; 셸 간 동일성·연속성 검사), 부록 A 직렬화를 hashlib 로 재계산, manifest 의 `grid_manifest_sha256` 와 대조. `%.17g` 는 double 왕복 무손실 [실측: printf 정밀도 규약] |
| `Makefile` | `selftest_a2_09_emissivity` 타깃에 `src/population_contract.c` 링크 추가(정본 atomic 헬퍼 사용을 위해; `selftest_physics_comparison` 은 이미 링크됨 [실측 Makefile:501-509]) |
| `docs/RUNG_A209_IDENTITY_SEAL_2026-08-20.md` | 이 문서(집행 기록 절 추가) |

**접촉 금지**: `src/physics_comparison.c`(소비자 불변 — 게이트 문언·site 번호 판정 잣대
유지), `scripts/check_det_convergence.py`, `scripts/compare_physics_snapshots.py`, 모든
`.cu`(SPEC: A2-09 단계 `.cu` diff 0), 덱·`/gpfs` 정본, env 노브 표면(신설 0).

---

## 4. 게이트 표 (판정문은 이 번호를 **그대로** 대조한다)

| # | 조건 | 판정 자료 |
|---|---|---|
| **P1** | 빌드 CPU(`make lumina`)+GPU(`make lumina_cuda`) 두 타깃 에러 0, format 경고 0. **두 타깃 모두 빌드 로그를 파일로 보존**(직전 단 P1 이 "운전석 주장" ⚠부분에 그친 원인 제거) | 보존된 빌드 로그 |
| **P2** | 오프라인 회귀(변경파일 선별): `selftest_a2_09_emissivity` · `selftest_physics_comparison` · `selftest_physics_comparison_regrid` · `tests/det_convergence_selftest.py` · `selftest_det_stage12` · `selftest_a2_10_radeq` 전부 PASS (grammar-debug) | 실행 로그 |
| **P3** | NC1~NC4(§5) 전부 시연 — FAIL 시연 2건(NC2·NC3)과 감도 시연 2건(NC1·NC4) | selftest 출력 |
| **P4** | C↔Python 정의역 합치: `run_a2_09_selftest.py` 가 C 헬퍼 출력(고정 소형 격자 grid 해시 + 고정 mask source 해시)을 hashlib 독립 재계산과 대조해 일치 | selftest 러너 출력 |
| **P5** | 판정런 기대치 **B1·B2·B3** 적중(§6; B4 는 확인용 0-증거) | 판정런 stderr + 산출물 + verifier |
| **P6** | 물리값 무접촉: (a) diff 검수 — eta/cdf/샘플러 수치 계산 라인 0 접촉, (b) `a2_09` selftest 의 기존 수치 케이스(closure·히스토그램 95% 봉투) 문언 불변 PASS | diff + selftest |
| **P7** | 변경집합 준수: 커밋 접촉 파일 = §3 목록과 정확히 일치 | `git show --stat` |

★P6 이 이 단의 자기 규율이다 — 신원 봉인이 물리를 바꾸면 수리 단이 아니다.
★P3 이 NC 의무다 — 주입 지점은 전부 생산 커밋 체인 안이다.

---

## 5. 음성 대조 (전부 생산 커밋 체인 경유: init → 채움 → cdf 빌드 → 커밋)

| # | 주입 | 기대 | 성격 |
|---|---|---|---|
| **NC1** | 유효 후보 A 와, `nu_edge` 원소 **1개의 1 ULP 이웃값**만 다른 후보 B 를 각각 커밋 | 두 committed `grid_manifest_sha256` 이 **다르다**; 각각이 헬퍼 재호출값과 일치 | 격자 감도(성패 기준의 반쪽) |
| **NC2** | `nu_edge` 에 비유한값(NaN) 또는 역전(감소) 주입 후 커밋 | 커밋 **거부**(return 5), reason=`EMISS_IDENTITY_GRID_INVALID` 정확 문자열, `identity_seal_failures` 증가, 발행체(pub) 무변경 | fail-closed FAIL 시연 |
| **NC3** | **역사적 결함 그대로**: atomic 필드를 NUL(init 직후 상태)로 둔 채 커밋 | 커밋 **거부**, reason=`EMISS_IDENTITY_ATOMIC_UNSEALED` 정확 문자열, pub 무변경 | 이 단이 잡는 결함의 재주입 — 수리 전 코드라면 이 커밋이 **성공**했을 것 |
| **NC4** | 동일 후보를 `channel_mask` 0x7 과 0x3 으로 각각 cdf 빌드+커밋 | 두 `source_manifest_sha256` 이 다르다 | 소스 의미론 감도 |

판정식은 전부 **정확 문자열 대조**(OR 술어 금지 — 직전 단 P4 관례 계승).

**커버리지 경계 (사전 선언 — 직전 단 감리 R3 의 교훈: 재해석을 명시하라):**
`lumina_plasma.c` 의 writer 측 거부(`EMISS_IDENTITY_ATOMIC_STAMP_INVALID`: stamp 상태·세대
결박 3중 검증)는 **단위 시험으로 주입하지 않는다** — `a209_publish_cpu_emissivity_impl`
은 static 이고 완전한 `OpacityState/BFOpacity/AtomicData/PlasmaState/NLTEConfig` 픽스처를
요구한다. 이 경로의 시연 부재는 커버리지 구멍으로 **여기 기재**하며, 판정런에서는 정상
경로(결박 성립)만 관측된다. 거부 경로가 실제로 발화하면 그것이 곧 발견이다(§6 분기).

---

## 6. 판정런과 기대치 사전등록

**판정런**: DET-STAGE12 고정-T 구성(판정런 320568 과 동일 구성 — a100 · `--gres=gpu:2` ·
`--mem` 명시 · `LUMINA_PHYSICS_COMPARISON_DIR` 설정) + 수리 바이너리, slurm 1회.
작업명 `LUMINA_` 접두 + 런 루트에 `OWNER.txt`(320567 오인취소 재발방지). 직전 단과 달리
**이 런은 site=133 에서 죽지 않도록 되어 있다** — 단 133 너머는 아무도 본 적 없는
미답이다(직전 판정문 §2 "확정하지 않은 것" 3항). 그 미답을 기대치 분기에 반영한다.

각 기대치에 **"이 관측이 다른 가설 아래서도 같은 값을 내는가"** 의 자문과 답을 붙인다
(직전 단 B3 이 증거력 0 이 된 실패의 재발 방지).

| # | 기대 | 자문: 다른 가설도 같은 값을 내는가 | 증거력 |
|---|---|---|---|
| **B1** | `reason=COMPARISON_HASH_INVALID site=133` **0줄**, iter=0 에서 `PHYSICS-COMPARISON][FATAL]` 없음 | **낸다** — hex64 아무 상수를 채우는 가짜 수리도 B1 적중 | 낮음(차단 해소 확인만; B2 와 결합해서만 의미) |
| **B2** | `physics_DET_iter0000.manifest.json` 의 `grid_manifest_sha256` == `scripts/verify_a209_grid_manifest.py` 가 그 런의 spectral CSV edge 에서 독립 재계산한 값 | **못 낸다** — 상수 채움·타 해시 복사(cdf/geometry)·직렬화 오류 전부 불일치. 잔여 위험: verifier 가 C 코드의 재서술이면 공허 — **검수 항목**: verifier 는 부록 A 명세에서만 작성됐는지 대조 | **핵심 증거** |
| **B3** | `[A2-09][IDENTITY]` 줄 존재(발행마다), 세 해시 전부 hex64, 그 `atomic_model_sha256` == manifest JSON 의 `atomic_model_sha256`(온도측 — 같은 partition stamp 에서 나왔으므로) | **부분적으로 낸다** — 결정론 런이므로 이전 런 로그에서 베낀 상수도 적중 가능. 보강: NC3(생산 체인 거부)+검수(복사원이 stamp 임을 diff 로 확인) | 중간 |
| **B4** | (꼬리 반복이 존재할 만큼 진행 시) `check_det_convergence.py` 불변량 검사에서 `grid_manifest_sha256` 반복 간 동일 | **낸다** — 어떤 상수도 통과 | **0 (성립 확인용으로만 기재; 증거로 세지 않는다)** |

### 철회·분기 (사전 확정)

| 관측 | 처분 |
|---|---|
| site=133 이 여전히 `grid_manifest_sha256_valid=0` 으로 발화 | **사전등록 철회** — writer 가 커밋에 도달하지 않았거나 설계가 틀림 |
| site=133 이 **다른** 필드(온도측 3개 중 하나) 불량으로 발화 | 이 단 판정 **보류** — 온도측 신규 결함, 별도 FINDING 기재 후 재판정 |
| 정상 경로에서 `EMISS_IDENTITY_ATOMIC_STAMP_INVALID` 또는 `EMISS_IDENTITY_*` 거부 발화 | §2-Q3 의 결박 등식이 생산에서 깨진다는 **발견** — 이름 있는 사유가 지점을 특정한다. 이 단은 미폐합, 결박 조항 재설계. (fail-closed 가 의도대로 작동한 것이므로 봉인 기계 자체의 실패가 아니다) |
| B2 불일치 (manifest 는 생성됐는데 재계산과 다름) | **폐합 금지** — 직렬화 divergence 또는 writer 결함; C↔Py 어느 쪽이 부록 A 를 어겼는지 특정 후 재발주 |
| 133 통과 후 **후속 게이트**(INVALID_GRID/INVALID_VALUE/STALE 등)에서 죽어 산출물 0 (CSV·manifest 는 tmp 삭제로 전부 부재 [실측: snapshot_write 는 3파일을 끝에서 일괄 rename]) | 이 단은 **부분폐합** — 봉인 자체는 B1+B3+P3/P4 로 입증되나 B2 미회수. 다음 측정 단(후속 게이트 사유 계측)을 개설하고 B2 는 그 단의 판정런에서 회수한다. "부분"을 "완전"으로 쓰지 않는다 |

---

## 7. 분장 장부 (실제 열은 집행 후 채운다 — 명목을 실제인 양 적는 것 금지)

| 단계 | 규약상 담당 (개정13/14) | **실제** | 위반 |
|---|---|---|---|
| 사전등록 (이 문서) | Fable | Fable | ✅ |
| 발주 | 운전석 | | |
| 코딩 | Codex (clean worktree) | | |
| 코드 검수 | Fable | | |
| 빌드·오프라인 게이트·제출 | 운전석 | | |
| 판정 | Fable | | |
| 판정 감리 | Fable (fresh 컨텍스트) | | |
| 감리 반영·대장·커밋 | 운전석 | | |

직전 단은 Fable 몫 4건 중 3건이 위반이었고, 그 본체는 "충돌을 보고하지 않은 것"이었다.
이번 단에서 하네스 제약과 규약이 충돌하면 **말없이 해소하지 말고 위 표의 실제 열에
그대로 적는다.**

---

## 8. Codex 가 지켜야 할 계약 (발주서에 이 절을 그대로 첨부)

1. **클램프·floor·cap·새 env 노브 금지.** 이 단의 검증들은 신원 스키마 검증이지 수치
   가드가 아니다 — 물리 수치를 만지는 코드는 한 줄도 없어야 한다.
2. **이미 있는 자산만 사용**: `A209Sha`(+`sha_init/sha_up/sha_u64/sha_f64/sha_done`,
   `emissivity_publication.c:8-19`) · `population_atomic_model_sha256` 의 **출력 재사용**
   (partition stamp 경유, 재해시 금지) · 관례 본보기 `cdf_manifest_sha256`(`:115-120`) ·
   `a210_geometry_sha256`(`radeq_publication.c:228`). 새 해시 라이브러리·재구현 금지.
   (개정10 의 교훈: 검증된 판독기를 무시한 새 파서가 세 번 연속 패치를 낳았다.)
3. **fail-closed**: 해시를 만들 수 없으면 이름 있는 사유(§3 의 reason 토큰 정확 문자열)로
   발행을 거부한다. 빈 값·부분 값 발행 금지. 거부 시 발행체(pub)와 후보의 물리 배열은
   무변경이어야 한다.
4. **직렬화는 부록 A 를 byte 단위로 따른다.** 임의 변경 금지 — Python 독립 재계산(P4·B2)이
   같은 명세에서 별도로 작성된다.
5. 덱·`/gpfs` 정본 불변. `src/physics_comparison.c`·checker 스크립트 2종·모든 `.cu` 무접촉.
6. **변경집합은 §3 이 전부다.** 필요해 보이는 추가 변경이 있으면 고치지 말고 보고하라.
7. 계약 1개 = 커밋 1개. `git add -A` 금지.

---

## 부록 A — 직렬화 명세 (정본; C 와 Python 이 각각 이것에서 작성된다)

공통: SHA-256. 정수는 **big-endian u64**(기존 `sha_u64` 관례 [실측: `b[7-i]=x>>(8i)`]),
double 은 **IEEE754 비트를 u64 로 재해석 후 big-endian**(기존 `sha_f64` 관례). 도메인
문자열은 ASCII, 종결 NUL 제외. 출력은 소문자 hex 64자 + NUL.

### A-1. grid_manifest_sha256

```
H = SHA256( "A2-09:grid-manifest:Hz:bin-edges:IEEE754:v1"
            || u64_be(n_bins)
            || f64_bits_be(nu_edge[0]) || … || f64_bits_be(nu_edge[n_bins]) )
```
사전 검증(위반 시 거부, 해시 미산출): `nu_edge != NULL` · `n_bins >= 1` · 모든 원소
유한·양수 · 엄격 순증가.

### A-2. source_manifest_sha256

```
H = SHA256( "A2-09:source-manifest:eta-true=bb+bf+ff:scattering-separate:comoving:per-sr:v1"
            || u64_be(channel_mask) )
```
사전 검증: `channel_mask != 0`.

### A-3. atomic_model_sha256 (em)

계산하지 않는다. `atom->partition_stamp.atomic_model_sha256` 의 65바이트 복사.
결박 검증(§2-Q3)과 커밋 시 hex64 검증만 한다. 도메인은 정본 헬퍼의 것
(`"A2-07:atomic-partition-membership:v2:topion-bound"` [실측 `population_contract.c:88`])
그대로이며 이 단은 그것을 소유하지 않는다.

### A-4. 세대 무포함 원칙

세 신원 해시 모두 **세대(generation)·epoch 를 포함하지 않는다** — `check_det_convergence.py`
가 반복 간 불변량으로 검사하기 때문이다 [실측 `:279-284`]. (반면 `cdf_manifest_sha256` 은
세대를 포함하는 것이 정의다 — 혼동 금지.)

---

## 부록 B — 열린 질문 / 이 단이 모르는 것 (추측으로 메우지 않는다)

1. **source_manifest 의 SPEC 원 의도** — 모른다. v1 정의는 이 단의 것이다. CMFGEN
   `ETA_DATA` 직접 대조(SPEC:113-118 의 포함규약 기록)가 시작되면 v2 확장을 재방문한다.
2. **정본 원자모형 해시의 정의역 한계** — 선 목록(A_ul·ν) 미봉인. 온도 발행체와 공유하는
   한계다. 처분: `docs/CLASSIC_DEBT_CENSUS.md` 계열 대장 기재 후보(조용히 기재; 이 단에서
   수리하지 않는다).
3. **te-측 손 채움 픽스처 잔존** — `physics_comparison_selftest.c` 의 tep.atomic/te_manifest,
   `det_stage12_selftest.c` 의 고정 hex 문자열. 생산 writer 가 실존하므로(온도측) 은폐
   위험은 em 측과 다르나, 같은 무늬다. 범위 밖 — 기재만.
4. **em↔te atomic 동일성의 소비자측 강제** — `physics_comparison.c` 에
   `em->atomic_model_sha256 == te->atomic_model_sha256` 게이트를 추가하면 B3 이 관측에서
   강제로 승격된다. 이 단은 소비자 무접촉이므로 하지 않는다 — 후속 단 후보.
5. **`src/jnu_seed.h` 의 유사 미기재 짝**(`source_geometry_sha256`·`source_payload_sha256`,
   FINDING §5) — 미확인 그대로. 이 단의 주장 아님.
6. **결박 등식의 전 경로 전수**(§2-Q3) — [추정] 표기대로 전수 확인이 아니다. fail-closed
   게이트가 검출기이며, 발화하면 §6 분기대로 처리한다.
7. **stderr 신규 줄이 기존 게이트 배터리와 충돌하는가** — [추정] 이벤트 grep 은 태그
   기반이라 추가 줄은 무해할 것. 검수에서 `[A2-09][IDENTITY]`·신규 reason 토큰이 기존
   파서와 충돌하지 않음을 확인할 것.

---

## 집행 기록 (운전석 실측 — 검수 R2 로 Codex 선기입을 교체)

⚠초판의 이 표는 **Codex 가 채웠다**. 표제가 "운전석 — 집행 후 채운다"인데도 자기 게이트
결과를 적어 넣었고, 근거 로그는 `/tmp/build_*.log`(휘발)였다. 검수 R2 지적을 받아
**운전석이 전부 다시 돌리고 영속 경로에 보존한 실측으로 교체**한다.

**증거 보존 경로**: `/gpfs/kjhan/lumina/gates/sh_a209_idseal_20260820T044201Z/`

| 항목 | 담당 | 실측 |
|---|---|---|
| 사전등록 | Fable | `docs/RUNG_A209_IDENTITY_SEAL_2026-08-20.md` (커밋 `7917cd3`) |
| 발주 | 운전석 | worktree `wt_a209` @ `7917cd3`, `PREREG.md` 그대로 첨부(재서술 0) |
| 코딩 | Codex | §3 변경집합 10수정+1신설 |
| **코드 검수** | **Fable** | **판정 인정.** R1·R2 운전석 집행 몫, R3~R5 낮음 |
| **P1** 빌드 | 운전석 | `make lumina` rc=0 · `make cuda` rc=0, **error 0 · format 경고 0**. 로그 `P1_build_cpu.log`(21.5KB) · `P1_build_gpu.log`(3.8KB), 바이너리 해시 `binaries.sha256` |
| **P2** 회귀 6종 | 운전석 | **6/6 PASS** (grammar-debug). `P2_regression.log`. `a2_09`: `N1_N8=8/8 NC1_NC4=4/4 P4_APPENDIX_A=PASS` |
| **P3** NC1~NC4 | 운전석 | 4/4 (P2 로그에 포함) |
| **P4** C↔Py 부록 A 합치 | 운전석 | PASS + ★**아래 독립성 실증** |
| P5·P6·P7 | — | 판정런 대기 / 검수가 P6 diff 무접촉 확인 / P7 은 폐합 커밋에서 |
| 판정런 · 판정 · 감리 | — | 미실시 |

### ★운전석이 추가한 실증 — B2 독립성은 문언이 아니라 실험이다

검수는 코드 판독으로 "파이썬에 C 산출 상수가 없다"까지 확인했으나, *"Codex 가 C 를 읽었는지"*
는 **코드만으로 증명 불가하며 모른다**고 정직하게 적었다. 운전석이 실험으로 메운다:

**C 의 도메인 문자열만** `…IEEE754:v1` → `…IEEE754:vX` 로 바꾸고 시험을 돌렸다
(`NC_driver_demo.log`):
```
주입:  FAIL A2_09_SELFTEST N1_N8=8/8 NC1_NC4=4/4 P4_APPENDIX_A=FAIL   (make Error 4)
복구:  PASS A2_09_SELFTEST N1_N8=8/8 NC1_NC4=4/4 P4_APPENDIX_A=PASS
```
파이썬이 C 의 번역이었다면 **둘 다 바뀌어 통과**했을 것이다. 갈렸다 ⟹ 두 구현이 각자
부록 A 사본을 들고 대조한다. **B2 의 증거력이 실험으로 지지된다.**

부수 관찰: 주입 중에도 `NC1_NC4=4/4` 였다 — NC 는 감도·거부를 보는 시험이라 도메인
문자열 변경에 무감해야 맞다. 관심사 분리가 의도대로 작동한다.

### 검수 지적의 처분

| # | 지적 | 처분 |
|---|---|---|
| R1 | 신설 verifier 가 untracked | 폐합 커밋에 **명시적 `git add`**(`-A` 금지 규약 유지), 조율 파일 5종 제외 |
| R2 | 집행 기록 Codex 선기입 + P1 로그 휘발 | **이 절로 교체**, 로그를 `/gpfs` 영속 경로에 보존 |
| R3 | NC2/NC3 의 pub 불변 검사가 얕은 `memcmp` | 한계 인정·기재. 거부 경로가 pub 를 역참조하지 않음은 검수가 코드로 확인 |
| R4 | 커밋 거부 시 후보의 신원 필드 부분 변경 | 계약 위반 아님(계약은 **물리 배열** 무변경). **실패한 커밋 후 후보의 신원 필드를 읽지 말 것** |
| R5 | `check_a209_source_failclosed.py` 가 stale | ★**이 단과 무관한 선재 결함.** [실측] impl/래퍼 분리는 `f6c2eb6`(08-18)로 이 단보다 이틀 앞서고, 이 단의 `lumina_plasma.c` diff 는 `@@ +9103,7` 와 `@@ +9129,19` 두 곳뿐으로 래퍼(`:9408`)를 건드리지 않는다. **조용한 대장 기재** — 별도 단 |
