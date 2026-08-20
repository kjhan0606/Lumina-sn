# 단 사전등록 — SH-GATEREPAIR 사다리: 회귀 등록부 신설 + 죽은 잣대 4개의 수리 (2026-08-20)

저자: Fable(사전등록, 분담 개정14) · 발단: `docs/GATE_RECOVERY_INVENTORY_2026-08-18.md` §F·§G·§H·§I
+ 운전석 증거 패킷(`/tmp/claude-10396/gaterepair/EVIDENCE.md`).

**개정 이력**: 초판 = 커밋 `92f9fa9`. **개정 1(2026-08-20, 저자 Fable)** — GR-1 집행 중
발견 3건(배터리 Z-a2-09 빌드 사망 · collective 배선 교차검증 공허 · preflight 의 미추적
픽스처 의존)의 반영. GR-0·GR-7·GR-8 신설, GR-1 재발주(GR-1′), 캠페인 폐합 조건 확장.
원문은 지우지 않고 보존하며, 대체되는 문안에는 `[개정 1]` 표지를 달아 §0-A 를 가리킨다.
판정문은 개정판 게이트 표(§0-A)와 대조한다. 전문 = §0-A.
**개정 2(2026-08-20, 저자 Fable)** — GR-3 게이트 **P3 실패**(외부 주입이 커버리지 구멍을
노출)의 처분: P3 표적(upper 스팬)은 유지, 금지-읽기 스캔을 upper·formula 로 확대,
GR-3 은 **GR-3′ 로 재발주**(착지 구현 보존, delta 만). 전문 = §5-A;
GR-3 판정문은 §5-A 의 개정판 게이트 표와 대조한다.
소속: **SH-** — 이 잣대들이 지키는 계약 표면(a208/a209 발행체·tau 슬랩·event-measure)은 두 팔 공유다.
CMFGEN 발자국 규약: **비적용** — 게이트·회귀 인프라이며 CMFGEN 대응물이 없다.

**캠페인 성격: 잣대 수리.** 판정 §H·§I 가 "코드는 정당, 게이트가 틀렸다"로 확정했다.
따라서 이 사다리 전체에서 **`src/` 접촉은 0줄**이다 — A2-10 귀속 동결
(`docs/CURRENT_PLAN.md:3406` "이 두 입력이 봉인되기 전 source edit 및 최종 non-census gate는
금지" [실측])과 정면으로 양립한다. `src/` 를 고쳐 게이트를 통과시키는 순간 이 캠페인은 실패다.

실행 티어: 전 단 오프라인. 빌드·게이트·배터리 = **grammar-debug**(nested ssh) 또는 slurm.
로그인 노드 실행 금지. GPU 불요(단 하나의 GPU 게이트는 run-dependent 로 등재만 한다). 판정런 없음.

---

## 0. 캠페인 구조 결정 — 증거 패킷 §5-1 의 딜레마에 대한 답

### 질문

> 구조적 처방 (e)(회귀 편입 + 메타 게이트)를 먼저 하면 아직 빨간 게이트들을 배터리에 넣게
> 되어 배터리가 상시 FAIL 이 된다. 나중에 하면 개별 수리 (a)-(d)가 다음 리팩터에 또 죽는다.

### 결정: **(e) 먼저 — 단, "지명 적자 등록부(known-red register)"를 함께 신설한다.**

배터리 상시 FAIL 딜레마는 거짓 이분법이다. 세 번째 길: 등록부가 빨간 게이트 4개를
**서명 고정(known-red pin)** 으로 등재한다 — 각 행은 (실패 명령 위치, rc, 첫 FAIL 줄
정확 문자열, FAIL 줄 수, 등재일, **수리를 담당할 사전등록 단 번호**)를 가진다.
배터리 preflight 의 sweep 이 매번 그 게이트들을 **실행**하고:

- 고정된 서명 그대로 죽으면 → 배터리 green (죽어 있음이 **관측·인증된** 상태)
- 서명이 달라지면 → 배터리 red, 사유 `known-red-signature-drift` (눈먼 기간에 무언가 변했다)
- 예기치 않게 PASS 하면 → 배터리 red, 사유 `known-red-unexpected-pass` (행을 소거하라는 강제)

즉 등록부는 스스로 만료된다 — 수리가 착지하는 순간 red 가 되어 행 소거를 강제하고,
행 소거 커밋이 곧 그 단의 회귀 편입이다.

### (e)-먼저의 근거 4개

1. **[실측] 처방 기구는 이미 있고, 사고의 절반은 비편입이었다.** 2026-08-07 에 정확히 이
   병("존재하지만 실행 안 됨")을 진단하고 `PREFLIGHTS`+`run_preflights()` 를 만들었는데
   (`scripts/run_gate_battery.py:44-72`), 08-18 신설 게이트들이 등재되지 않았다(증거 패킷 §2).
   수리를 먼저 하면 "수리 후 편입"이라는 같은 약속을 4번 반복하게 되고, 그 약속이 지켜지지
   않는다는 것이 바로 오늘의 사고다.
2. **수리-후-편입의 창이 곧 사망 창이다.** 4개 단이 순차 진행되는 며칠 동안, 편입 안 된
   수리본은 `f6c2eb6` 계급의 리팩터 하나에 다시 조용히 죽는다(§G-2 실증). (e)-먼저면
   그 창이 0이 된다.
3. **known-red 행은 죽은 게이트를 수리 대기 중에도 관측 아래 둔다.** 예: sh-radeq 의
   19건이 20건으로 변하면(눈먼 기간에 새 위반 유입) 수리 시점이 아니라 **당일** 드러난다.
   green-only 편입(대안)은 빨간 4개를 기계가 안 보는 TODO 로 남긴다 — 같은 병의 반복.
4. **수리 단의 폐합 자격 자체가 편입을 요구한다**(§H-5-2·§I-7-5). 편입 기구를 먼저 세우면
   각 수리 단의 편입이 "등록부 행 1개 소거 + preflight 행 1개 추가"라는 기계 검증 가능한
   행위가 된다.

### known-red 가 은폐 창고가 되지 않게 하는 안전핀 (사전 확정)

- 메타 게이트는 (첫 FAIL 줄·FAIL 줄 수·등재일·**수리 단 번호**) 없는 known-red 행을
  `known-red-row-incomplete` 로 **거부**한다. 수리 계획 없는 적자 등재는 원리적으로 불가.
- 캠페인 폐합 조건 = 이 4행이 **전부 소거**된 상태. "known-red 로 정리했으니 폐합"은 없다.
  **[개정 1]** 폐합 조건이 확장됐다(unwired 처분 기입 + GR-8 편입) — §0-A-8 이 정본.
- known-red 가 인증하는 것은 **"이 서명으로 죽어 있음"뿐**이다. 게이트의 측정면(계약 준수)은
  무인증 — sh-radeq 첫 FAIL 줄은 래퍼 스캔 산물이라 impl 의 실위반에 무감하다.
  측정면의 창-내 무결은 §I-6 의 판정자 3점 diff 가 이미 별도로 답했다. 이 한계를 안다.
- 이것은 클램프가 아니다(판별식: 물리 수치를 만지지 않고, 정확해가 위반할 가드가 아니다).
  「틀린 값은 조용히 대장 기재」 원칙의 대장을 **기계가 매번 읽는 형태**로 만든 것이다.

### 단 분해 (계약 1개 = 커밋 1개)

| 단 | 계약 (1줄) | 소속 | 의존 | 커밋 |
|---|---|---|---|---|
| **GR-1** | 게이트 성격 make 타깃 62개 전원이 기계가 매번 검사하는 등록부에 배속되고, 죽은 4개는 서명 고정으로 관측된다 | SH | 없음 | 1 |
| **GR-2** | (판정 발주 — 트리 무변조) `:21271`·`:21379` 벌크 이식 writer 의 등록부 지위 판정 (§H-5-3-①) | SH | 없음 | 1 (판정문) |
| **GR-3** | `selftest-sh-radeq-source` 가 HEAD 의 실제 계약 표면을 다시 재고, 8개 주입 전부가 이름 있는 사유로 검출된다 (§I-7 전체) | SH | GR-1 | 1 |
| **GR-4** | `selftest-tau-writer-census` 의 닫힌 writer 집합이 실제 writer 집합(원소 대입 + 벌크 memcpy + 별칭)과 다시 일치한다 (§H-5-1) | SH | GR-1·GR-2 | 1 |
| **GR-5** | `event-measure-check` 의 실패 전수가 계측·판정되고, 판정대로 앵커가 실물로 복원된다 | MC | GR-1 | 2 (판정문+수리) |
| **GR-6** | `selftest-a2-10-line-saturation` 픽스처가 생산자 스키마 v1 을 다시 싣는다 | DET | GR-1 | 1 |

**[개정 1]** 위 표는 GR-0·GR-7·GR-8 신설과 GR-1 재발주로 §0-A-8 의 개정판 표가 대체한다
(원문 보존 — 초판 발주분의 기록).

**순서**: GR-1 → (GR-2 발주, 트리 무변조라 GR-3 코딩과 병행 가능) → GR-3 → GR-4 → GR-5 → GR-6.
트리-변조 태스크는 상시 1개(규약). GR-3 을 수리 선두에 두는 근거: 수리 명세가 §I-7 로
이미 확정돼 있고, 이 게이트가 지키는 계약(선방출 직접형·tau 세대 괄호)이 **진행 중인
Stage-4 캠페인의 측정면**을 지킨다. GR-6 은 자명하고 독립이라 말미.

**이 캠페인에 넣지 않는 것(§5-5 승격 경로의 답)**: §2-Q5 에 근거와 함께.

---

## 0-A. 개정 1 (2026-08-20, 저자 Fable) — GR-1 집행 중 발견 3건의 반영

근거 증거: 운전석 증거 패킷(`/tmp/claude-10396/gr1_amend/EVIDENCE.md`, 발견 A·B·C) +
Codex 자기 보고(`/tmp/claude-10396/wt_gr1/NOTES.md` — 특히 :77-81 이 발견 B 의 약점을
스스로 공개한 자리) + **이 개정의 독립 재측정**(§0-A-1 — 개정 저자가 HEAD 작업트리에서
전 항목을 직접 다시 쟀고, 한 곳은 증거 패킷을 정정한다).

**무엇을 왜 바꾸는가 (요약)**:

| # | 바꾼 것 | 왜 |
|---|---|---|
| 1 | GR-0 신설 — 배터리 Z-a2-09 빌드 재정합 (§0-A-6) | 발견 A. GR-1 기대 변경집합 밖의 선행 결함이라 GR-1 에 실을 수 없고(P6), GR-1 P4 의 전제다 |
| 2 | GR-1 재발주(GR-1′) — collective 파서 수리 + `unwired` 범주 신설 + 픽스처 커밋 (§0-A-5) | 발견 B·C. 현행 P1 은 27건의 거짓 배선을 매판 인증한다 — 계약문("배선 주장 교차검증") 자체가 미이행이라 폐합 자격이 없다 |
| 3 | GR-7 신설 — 고아 27개의 계측·처분 판정 (§0-A-7) | unwired 는 정직한 표기일 뿐 처분이 아니다. 처분 단이 강제되지 않는 범주는 은폐 창고가 된다(known-red 안전핀과 동형) |
| 4 | GR-8 신설 — 중복 빌드 명세 정합 검사 (§0-A-7) | 공통 근원(3중 기재)의 갈라짐을 상설 감시. 근원 제거 자체는 파급 실측 결과 미룬다(§0-A-3) |
| 5 | 검수 R3=문안 처분 · R4=known-red 한정 채택 (§0-A-4) | — |
| 6 | 캠페인 폐합 조건 확장 (§0-A-8) | unwired 처분 미기입 행이 남으면 폐합 불가 |

### 0-A-1. 독립 재측정 — 증거 패킷 수치의 재검과 정정 1건

[실측 — 전부 이 개정 저자가 HEAD 작업트리에서 직접 확인]

**발견 A (확인)**: Makefile 의 `selftest_a2_09_emissivity` 규칙(:367-370)은
`src/population_contract.c` 를 링크하고, 배터리의 `Build("Z-a2-09", ...)` 는
`tests/a2_09_emissivity_selftest.c`+`src/emissivity_publication.c`+`-lm` 뿐이다.
셀프테스트 `:24` 가 `population_atomic_model_sha256`(정의 `src/population_contract.c:84`)를
부른다 — 링크 실패의 재현 조건 성립. Makefile 링크 목록이 정확히 그 1항만 더 가지므로
수리도 그 1항 추가다(§0-A-6).

**발견 B (확인 + ★정정)**: 수리된 잣대(논리행 접합 · 자기 rule recipe 제외 · `clean` recipe
제외)로 재측정 — collective 41 전원이 Makefile 무참조. 증거 패킷과 일치. 그러나 패킷이
고아 26(= 호출자 보유 15)으로 센 그 판별은 **어휘 계수**(grep 토큰 실존)였다. 15개
"호출자" 의 호출부를 전수 판독한 결과 **14개만 실호출**(make/바이너리 실행 또는 slurm
제출용 install)이고,
`selftest_a2_08_signed_opacity` 의 참조 2건(`scripts/a2_08_finalize_artifacts.py:25`·`:77`)은
**원장 메타데이터 문자열과 수동 명령 문서**다 — 실행이 아니다.

⟹ **진성 고아 27 · 실호출 collective 14.** 오분류 채널 **3호**가 추가로 열려 있었다:
자기 recipe(1호) · clean 목록(2호)에 이어 **기록 문자열 속 어휘 언급**(3호). 운전석 첫 측정
4 → 패킷 재측정 26 → 본 개정 27 — 잣대를 잴 때마다 채널이 하나씩 더 나왔다.
1·2호는 GR-1′ 파서가 기계로 닫고(§0-A-5-a), 3호는 기계로 못 닫는다(셸/문자열 의미론) —
collective 의 wiring 을 **판독 확인된 실호출 파일**로만 기재하는 규정(§0-A-5-b)으로 닫는다.

⟹ **고아율의 정정**: §G-1 의 "62개 중 21개" 는 과소였다. 21(§G) + 27(본 개정) = **48/62 가
아무에게도 불리지 않았다** — 이 캠페인의 병의 실제 크기다.

27개의 하위분류 [실측 — recipe 전수 판독]:

| 부류 | 수 | 타깃 |
|---|---|---|
| 배터리-그림자 (make 타깃은 고아이나 **같은 테스트 소스**를 배터리가 자체 명세로 매판 빌드·실행) | 5 | `selftest_a2_08_signed_opacity`(Z-a2-08) · `selftest_a2_09_emissivity`(Z-a2-09) · `selftest_a2_10_radeq`(Z-a2-10) · `selftest_a2_12_contract`(Z-a2-12) · `selftest_a2_13_15_contract`(Z-a2-13-15) |
| CUDA/NVCC 빌드 | 3 | `selftest_nlte_assemble` · `selftest_a2_12_gpu_lifecycle` · `selftest_a2_13_gpu_oracle` |
| 순수 python | 1 | `selftest_physics_comparison_regrid` |
| CPU C 빌드 | 18 | `selftest_wave32_{ew_rc, ew_io, within_sl_oom, boundary_q, counter_atomic}` · `selftest_emiss_ab_insitu` · `selftest_a2_03_producer_parity_fixture` · `selftest_a2_04_replay_commit` · `selftest_det_stage12` · `selftest_line_net_rate` · `selftest_cmfgen_adiabatic` · `selftest_atomic_internal_energy` · `selftest_nlte_population_candidate` · `selftest_nlte_candidate_{adiabatic, tau}` · `selftest_a2_10_seed_commit` · `selftest_physics_comparison` · `selftest_a2_16_seed` |

collective 로 남는 14 [실측 — 각 호출부 판독 완료]: `selftest_cmf_exact_multigpu` ·
`selftest_cmf_exact_epoch_scan` · `selftest_ioniz_saha` · `selftest_seed_te_publish` ·
`selftest_bootstrap_window` · `selftest_a2_03_radiation_field` · `selftest_a2_04_commit` ·
`selftest_a2_05_bf_rate` · `selftest_a2_06_line_jbar` · `selftest_a2_06_dual_commit` ·
`selftest_a2_07_population`(이상 11 — 기존 wiring 정확) ·
`selftest_cmf_chieta_dump`(→ `scripts/cmf_chieta_roundtrip_selftest.py:72,76` make+실행) ·
`selftest_cmf_linepop_dump`(→ `scripts/cmf_linepop_roundtrip_selftest.py:39` make, `:24` 실행) ·
`selftest_wave32_matrix_debit`(→ `tests/test_wave32_seeded_defects.py:16,104` make+실행)
(이상 3 — wiring 이 "Makefile" 로 잘못 기재돼 있어 재기입 대상).

**배터리-그림자의 한계** [실측+미확인]: make recipe 는 빌드+python driver 실행의 결합이다
(예: a2_09 recipe 2번째 명령 = `run_a2_09_selftest.py`). 배터리 Z 러너
(`run_zinert_selftest.py`)가 그 driver 검사를 재현하는지는 **모른다** — GR-7 계측 항목.

**발견 C (확인 + 원인 심화)**: 픽스처가 미추적인 직접 원인은 **`.gitignore:30` 의 `*.log`
광역 규칙**이다. `tests/fixtures/` 의 유일 파일이고(추적 0건), 소비자는
`tests/a2_10_cancellation_census_selftest.py:14` 하나 [실측]. 단순 `git add` 로는 안 들어간다 —
.gitignore 처분이 변경집합에 필요하다(§0-A-5-d). provenance 는 **모른다**(내용 = A2-10
`LINE-NET-CELL-BLOCKED`/`CANCELLATION-CENSUS` 이벤트 로그 5행 — 전수 판독 가능).

**known-red 4행 (R4 의 전제 실측)**: 등록부의 commands 는 4행 모두 해당 make recipe 와
정확히 일치하고, recipe 는 전부 **변수 무포함 python3 직호출**이다 — R4 반사 검사(§0-A-4)가
문자열 비교로 성립하며 HEAD 에서 PASS 할 것이다.

### 0-A-2. 다섯 결정 (증거 패킷 §5 의 질문에 대한 답)

| 질문 | 결정 | 근거 |
|---|---|---|
| 1. 발견 A 의 거처 | **GR-0 신설** | GR-1 기대 변경집합 밖(P6 위반이 되므로 GR-1 에 못 싣는다) · 계약이 다르다(등록부 신설 vs 배터리 자체 결함 수리 — 계약 1개=커밋 1개) · GR-1 P4 의 전제이므로 순서상 선행(번호 0) |
| 2. 발견 B 의 수리 범위 | 파서 수리+NC 는 **GR-1′ 범위 내**(계약문 이행의 수리다) · 신규 고아 27은 **`unwired` 범주**로 배속, **milestone 은 7 유지** · 처분은 **GR-7** | 27개는 상태 미측정이다(§G 의 "20개 전부 돌렸다" 에 미포함). 측정 없는 milestone 배속은 P5 를 측정·배속 혼합 단으로 만들고, known-red 후보가 몇인지 모른 채 P5 비용이 미정이 된다. CUDA 3개는 GR-1 실행 티어(GPU 불요) 사전등록과도 모순 |
| 3. 발견 C 의 거처 | **GR-1′ 변경집합에 픽스처 무수정 추가 + .gitignore 정밀 부정 1행** — tests/ 접촉 금지의 정밀화 | 금지의 원의는 게이트·픽스처 *수리* 의 단 분리(GR-3~6 의 몫)다. 미추적 자산의 무수정 추가는 수리가 아니라 **GR-1 자신의 배선 주장(preflight 행 실존)의 실체화** — 이 파일 없이는 fresh clone 에서 GR-1 의 계약이 거짓이다. 행 보류(대안)는 오늘 green 인 실게이트를 배터리에서 빼는 회수 캠페인의 역행, 별도 단(대안)은 1파일 추가에 과잉 |
| 4. R3·R4 | R3 = 등록부 문안 정직화(코드 0) · R4 = known-red 행 한정 채택 · 그 외 범주 반사는 미룸(대장 기재) | §0-A-4 |
| 5. GR-1 폐합 가부 | ★**폐합 불가 — 개정 후 재발주(GR-1′)** | ① P1 이 현행 파서로 27건 거짓 인증 — "배선 주장 교차검증 0 위반" 이라는 계약 문장 자체가 미이행 ② P4 는 GR-0 착지 전 원리적으로 불가능 ③ 배속표가 바뀐다(collective 41 → 14 + unwired 27). **착지 구현은 보존**하고(전면 재작성 아님) delta 만 재발주한다 |

### 0-A-3. ★공통 근원의 판정 — "배터리가 make 를 호출"은 이 캠페인에서 하지 않는다 (파급 실측)

이 저장소는 "무엇이 무엇을 부르는가/무엇으로 빌드하는가"를 세 곳에 중복 기재한다:
① Makefile 규칙 ② 배터리의 자체 `Build(...)` 명세 ③ (GR-1 이후) 등록부.
발견 A = ①②의 갈라짐, 발견 B = ①을 읽어 ③을 검증하는 규칙의 부정확.
근원 제거 후보 — 배터리가 Makefile 타깃을 호출하게 만들기 — 의 파급을 실측했다:

1. [실측] **산출 위치 충돌** — 배터리는 임시 디렉토리에 산출(`Build.output` = `build/` 하위
   경로), make recipe 는 저장소 루트에 `-o $@`. 전환하면 작업트리 오염 + 동시 배터리 런
   충돌, 또는 Makefile 규칙 10+개의 출력 매개변수화 재설계.
2. [실측] **빌드/실행 결합** — 게이트 recipe 다수가 컴파일+python driver 실행을 한 규칙에
   결합한다(a2_08 · a2_09 등). 배터리는 `build_all`(13개 빌드 병렬, 실패 시 런 전 중단)과
   `run_all`(러너 4개가 바이너리 **경로**·scratch·deck 인자를 수취 — Z 러너는 바이너리
   10개의 경로를 인자로 받는다)을 분리한다. make 타깃 호출로는 이 분리·매개변수화가
   성립하지 않는다.
3. [실측] **K·CP 의 링크 목록은 런타임 도출**(`cpu_link_sources()` — src/*.c 에서 main 정의
   TU 를 제외). 소스 주석이 "3rd recurrence of stale hardcoded lists" 를 기록한, 정적 목록
   노후가 세 번 재발해 도입한 수리다. make 로 옮기면 이 수리를 되돌리거나 Makefile 에 새
   도출 기계를 심어야 한다.
4. [실측] **픽스처 캐시 결합** — `cached_materialize`(`--cache-root`)가 빌드 스레드풀과
   동주한다(`build_all` 안에서 동시 제출). 전환은 이 결합의 재설계를 요구한다.

⟹ 근원 제거 = **빌드 인프라 재설계 계급**. 이 캠페인(잣대 수리 + "거짓말하지 않기 위한
최소 확장")의 계약 밖 — **미룬다**(하려면 별도 캠페인 사전등록). 미룸을 안전하게 만드는
것이 **GR-8** 이다: ①②의 갈라짐(발견 A 의 채널)을 매 배터리마다 기계 검출한다. 발견 A 는
커밋(`2dc2817`) 후 첫 배터리 완주 시도까지 눈멀어 있었다 — GR-8 상설화로 그 은폐 창이
0 이 되면, 명세 단일화의 지위는 "재발 방지" 에서 "편의" 로 강등된다. 그때 가서 비용을
다시 재는 것이 옳다.

### 0-A-4. 검수 권고 R3·R4 의 처분

**R3 (E11 배선이 recipe 절반)** — [실측] `selftest_emiss_e11_fluor_matrix` recipe 는 2명령:
① `python3 -m py_compile` 4종 ② seeded fixture 실행. preflight 는 ②만 배선했고,
PREFLIGHTS 행 형식(`(이름, 스크립트 경로, 인자)`)으로는 `-m py_compile` 을 실을 수 없다
[실측 — `run_preflights()` 의 호출 형식]. **처분 = 등록부 해당 행 wiring 문안의 정직화**:
"recipe 2명령 중 실행 게이트만 배선(py_compile 4종 미배선)" 을 사실대로 기재(코드 변경 0).
초록불의 인증 범위를 문안이 과장하지 않게 하는 것이 이 캠페인의 병명이다. 기계 커버리지
확장은 하지 않는다 — py_compile 은 문법 검사이고, ②의 실행이 핵심 모듈 import 를 이미
강제하므로 확장의 실익이 형식 개조 비용에 못 미친다.

**R4 (등록부 commands ↔ recipe 반사 무검증)** — **known-red 행에 한정 채택.**
known-red sweep 은 등록부 행의 commands 를 실행한다 [실측 `observe_known_red`]. Makefile
recipe 가 바뀌어도 sweep 은 옛 명령을 돌린다 — **서명 감시가 유령을 감시**하게 되는
구멍이며, 이는 GR-1 계약("서명 고정으로 **관측**")의 일부이므로 GR-1′ 범위다.
검사: known-red · kind=make-target 행의 commands 가 해당 make 규칙 recipe 의 명령열과
정확일치(4행 전부 변수 무포함 [실측 §0-A-1] — 문자열 비교 성립), 불일치 =
`known-red-recipe-drift:<이름>` FAIL. **그 외 범주의 반사는 미룬다**: preflight 는 기존
`preflight-command-drift` 가 이미 덮고, collective·unwired·milestone 의 commands 는
실행되지 않는 참고 정보라 트립와이어를 오도하지 않는다 — 단 등록부가 썩는 채널이므로
**확정 부채로 대장 기재**.

### 0-A-5. GR-1′ — 재발주 명세 (delta; GR-1 착지 구현은 보존)

**(a) 파서 수리.** `make_references()` 의 제외 집합을 정밀 규정한다:
① 대상 타깃이 LHS 에 포함된 rule 의 정의행+recipe 행 전부
② `clean` 이 LHS 에 포함된 rule 의 recipe 행 전부
③ `.PHONY` 논리행(기존) ④ 주석(기존).
**과도 제외 금지** — 제외는 위 4류에 국한한다. 방향의 비대칭을 기록한다: 제외를 더 넓히면
실배선 collective 가 `collective-reference-missing` 으로 **시끄럽게** 죽는다(fail-closed
방향, 자기 고지) — NC 가 필요한 쪽은 조용한 방향(과소 제외)이고 NC-R1e·f 가 그 둘을 닫는다.

**(b) collective 의 재규정.** collective 의 배선 주장은 **실행 호출**이다 — 어휘 언급(원장
문자열·수동 명령 문서·주석)은 배선이 아니다. 기계 검사는 어휘 토큰 실존에 머무른다
(한계 유지 — 셸 도달성 미증명, Codex NOTES 공개분). 따라서 **행의 wiring 은 실호출이
판독으로 확인된 파일만** 가리킨다. 재배속은 §0-A-1 의 명단 **그대로**: collective 14
(그 중 3행 wiring 재기입) · unwired 27. 명단과 다른 재분류가 필요해 보이면 고치지 말고
보고한다.

**(c) `unwired` 범주 신설** (스키마 v2 — §0-A-9):
- 뜻: "이 타깃을 부르는 곳이 없음이 **관측됨**". 인증하는 것은 그 사실뿐 — 게이트 내용의
  건강(빌드 가부·PASS/FAIL)은 무인증이다. known-red("죽어 있음이 관측됨")와 직교한다.
- 필수 필드: `observed`(관측일) · `disposition_rung`("GR-7") — 결손 시
  `unwired-row-incomplete:<이름>` FAIL. **처분 계획 없는 고아 등재는 원리적으로 불가**
  (known-red 의 수리 단 번호 필수와 동형 안전핀).
- 선택 필드: `lexical_mentions`(어휘 언급만 있는 파일의 허용 목록 — a2_08 행은
  `scripts/a2_08_finalize_artifacts.py` 를 기재) · `note`(배터리-그림자 등 사실 기재).
- **자기 만료 검사**: 메타 게이트가 매번 Makefile(수리된 파서 규칙) + `scripts/`·`tests/`
  의 `*.sh`·`*.py` 에서 토큰을 재탐색하고, `lexical_mentions` 밖에서 참조가 나타나면
  `unwired-now-referenced:<이름>` FAIL — 배선이 생기는 순간 행 재배속을 강제한다.
  한계(기재): 허용 목록 **파일 안**에 새 실호출이 생기면 못 본다(파일 단위 허용).

**(d) 픽스처 커밋.** `tests/fixtures/a210_cancellation_census.log` **무수정 추가** +
`.gitignore` 에 정밀 부정 1행(`!tests/fixtures/a210_cancellation_census.log` — 광역 `*.log`
규칙은 불변). provenance 는 커밋 전 운전석이 실측 시도(생성 명령/원 런 특정)하고, 못 찾으면
"미상" 으로 집행 기록에 기재한다 — 조용히 메우지 않는다. **tests/ 접촉 금지의 정밀화**:
"tests/ 아래 **기존 파일의 수정** 금지 · 게이트 스크립트 신설 금지 — 이 미추적 픽스처
1건의 무수정 추가만 허용"(GR-6 의 몫과 무충돌 — 다른 픽스처·다른 게이트).

**(e) Codex 추가 계약 조항** (§9 에 추가 적용): 파서 제외를 위 4류 밖으로 확대 금지 ·
unwired 필드 정확일치 · 재배속은 §0-A-1 명단 그대로(임의 재분류 금지 — 다르면 보고) ·
스키마 문자열은 v2 로 (구판 등록부는 `registry-unreadable` 로 fail-closed 되는 것이 옳다).

**기대 변경집합 (delta — GR-1 착지분에 추가; 여기 없는 변경은 위반)**:

| 파일 | 변경 |
|---|---|
| `scripts/check_gate_registry.py` | 파서 제외 4류 · unwired 검증(불완전 행 거부 + 자기 만료 검사) · known-red recipe 반사(`known-red-recipe-drift`) · NC-R1e/f/g/h 추가 · 스키마 문자열 v2 |
| `scripts/gate_registry.json` | schema v2 · 27행 collective→unwired(필드 포함) · 3행 wiring 재기입(§0-A-1) · E11 행 wiring 문안 정직화(§0-A-4) |
| `.gitignore` | 정밀 부정 1행 |
| `tests/fixtures/a210_cancellation_census.log` | 신규 추가(무수정) |
| 이 문서 | 집행 기록 |
| (변경 0) | `scripts/run_gate_battery.py` · `Makefile` — GR-1 착지분 그대로 |

**게이트 표 (개정판 — 판정문은 이 표와 대조한다)**:

| # | 조건 | 판정 자료 |
|---|---|---|
| **P1′** | 완전성 PASS — 재집계 전원 배속(스냅샷 64), 수리된 파서로 배선 교차검증 0 위반, collective 14 전원의 wiring 파일에 토큰 실존, unwired 27 전원 무참조 재확인 | 메타 게이트 출력 |
| **P2′** | known-red sweep 4행 pin 정확 일치 + recipe 반사 일치 — grammar-debug 실측(샌드박스 pin 과 다르면 실측이 이기고 차이는 집행 기록에 기재) | sweep 출력 |
| **P3′** | NC-R1a~h **8/8** 정확 사유 | NC 출력 |
| **P4′** | (**전제: GR-0 착지**) 배터리 1회 완주(grammar-debug) — D=19·K=7·Z=12·CP=4 불변 + 신규 preflight 행 전부 rc=0 | 배터리 로그 |
| **P5′** | milestone 집합 타깃 완주 — **7개 불변**(이 개정으로 늘지 않음을 명시) | 실행 로그 |
| **P6′** | 변경집합 = GR-1 착지분 + 위 delta 표와 정확 일치 | `git show --stat` + 명단 diff |
| **P7** | 픽스처 추적: `git ls-files` 1건 + 로컬 clone(grammar-debug, /gpfs scratch)에서 해당 preflight 1행 단독 rc=0. [추정] clone 비용 소형 — 실측이 크면 ls-files 확인으로 강등하고 사유 기재 | 명령 출력 |

**신설 NC**:

| # | 주입 | 기대 (정확 사유) |
|---|---|---|
| **NC-R1e** | 합성 Makefile: 타깃이 자기 recipe 에서만 자기 이름을 참조(`-o`·`./` 실행) — collective 로 등재 | `collective-reference-missing:<이름>` — **발견 B 채널 1호의 재주입** |
| **NC-R1f** | 합성 Makefile: `clean` 의 `rm -f` 목록에만 이름 실존 — collective 로 등재 | 동일 사유 — **채널 2호의 재주입** |
| **NC-R1g** | 합성 unwired 행의 타깃을 합성 Makefile 의 다른 rule 에서 실참조 | `unwired-now-referenced:<이름>` |
| **NC-R1h** | 합성 known-red 행의 commands 를 합성 Makefile recipe 와 불일치시킴 | `known-red-recipe-drift:<이름>` |

**기대치와 자문 ("다른 가설 아래서도 같은 값을 내는가")**:

| 기대 | 자문 | 증거력 |
|---|---|---|
| collective 14 · unwired 27 | **낸다** — 과도 제외 파서도 우연히 같은 수를 낼 수 있다. 수가 아니라 **명단**을 §0-A-1 과 diff 로 대조하고(P6′), 제외의 정확 국한은 NC-R1e/f 가 시연해야 증거 | 중간(명단+NC 결합 시) |
| unwired 27 무참조 재확인 PASS | **낸다** — 아무것도 안 찾는 탐색기도 PASS. NC-R1g 가 탐색기의 생존을 시연해야 의미 | 낮음(NC 결합 시 핵심) |
| known-red recipe 반사 HEAD PASS | **낸다** — 공허 검사도 PASS [실측: HEAD 4행이 이미 일치]. NC-R1h 와 결합해서만 의미 | 낮음(NC 결합 시 핵심) |
| clone 에서 픽스처 preflight rc=0 | **못 낸다** — 픽스처 부재 시 그 행이 죽는 것은 이미 관측됐다(Codex 샌드박스 rc=4, NOTES 실측). 죽음/생존의 짝이 성립 | 핵심 |

**NOTES.md 의 처분** (Codex 가 제기한 P6 충돌): `NOTES.md` 는 저장소 밖 워크트리 산출물로
**커밋하지 않는다**. 요지는 이 문서의 집행 기록에 흡수한다 — P6 위반이 아니다.
**64 재집계의 승인**: 신설 타깃 2개(`gate-registry-check`·`selftest-registry-milestone`)의
자기 등재는 §3 철회·분기("재집계 실측을 따른다")의 올바른 적용이다 — 완전성 검사가 자기
신설 타깃을 예외 처리하지 않은 것은 설계 의도에 부합한다.

### 0-A-6. GR-0 — 배터리 Z-a2-09 빌드 명세 재정합 (신설 단)

**계약**: 배터리의 Z-a2-09 빌드 명세가 Makefile 정본 링크 목록과 재정합되어,
SH-A209-IDSEAL 이 봉인한 계약의 회귀 감시(배터리 Z 행)가 다시 돈다.

**귀속**: 결함 유발 = `2dc2817`(운전석 커밋 — Makefile 타깃만 갱신, 배터리의 중복 빌드
명세 미갱신). 잠복 원인 = SH-A209-IDSEAL 사전등록의 P2 회귀 목록("변경파일 선별")이
**변경파일을 중복 명세로 소비하는 소비자**를 포함하지 않음 — 3층 인수 프로토콜의 선별
규칙에 대한 교훈으로 대장 기재. 발견 경로 = GR-1 P4(배터리 완주 요구) — 이 캠페인이 세운
잣대가 첫 완주 시도에서 실물 결함을 잡았다.

**기대 변경집합**: `scripts/run_gate_battery.py` 의 `Build("Z-a2-09", ...)` 소스 목록에
`"src/population_contract.c"` **1항 추가**. 이 문서(집행 기록). **다른 변경 0.**

**게이트**: P1 = grammar-debug 에서 해당 빌드 명령열 단독 실행 rc=0 ·
P2 = 변경집합 1항(`git diff`). 배터리 전체 완주는 GR-1′ P4′ 의 몫 — 같은 관측을 두 단에
중복 귀속하지 않는다.

**기대치와 자문**: "빌드 rc=0" 은 테스트 소스 약화·검사 제거 가설로도 나온다 → P2 diff
(추가 1항뿐, `tests/` 무접촉)가 그 가설을 배제한다. 이 1항으로 충분하다는 것은 [실측] —
Makefile 이 정확히 그 목록으로 링크에 성공한다(같은 소스 집합).

**순서**: GR-1′ 에 선행한다(커밋도 선행 — GR-1′ P4′ 의 전제).

**집행 기록 (운전석 실측, 2026-08-20)**

| 항목 | 값 |
|---|---|
| 코딩 | Codex — `Build("Z-a2-09")` 에 `"src/population_contract.c"` **1항 추가** |
| 코드 검수 | Fable — **인정, 지적 없음**. Q1~Q5 전부 [실측] 확인 |
| **P1** | **PASS** — 배터리가 실제로 쓰는 명령열을 `build_specs()` 에서 **도출해** 단독 실행: `BUILD_RC=0`. 로그 `/gpfs/kjhan/lumina/gates/gr0_20260820T090422Z/P1_z09_build.log` |
| **P2** | **PASS** — `git diff --numstat` = `1 1 scripts/run_gate_battery.py`, 치환 1행. `tests/`·`src/`·`Makefile` 무접촉 |
| 커밋 | 아래 |

★P1 은 손으로 재구성한 명령이 아니라 `sys.path` 에 `scripts` 를 넣고 `run_gate_battery`
를 정상 import 해 `build_specs(tmpdir,"gcc")` 가 내놓은 `Build.command` 를 그대로 돌린 것이다
— **배터리가 다음에 돌릴 바로 그 명령**이다. Codex 가 NOTES 에 적은 명령열과 토큰 단위 일치.

[실측 함정 기록] 처음에 `importlib.util.spec_from_file_location` 으로 합성 이름 로드를
시도했더니 `dataclasses._is_type` 이 `sys.modules` 에서 모듈을 못 찾아 깨졌다
(`AttributeError: 'NoneType' object has no attribute '__dict__'`). 배터리 내부를 프로그램으로
뜯을 때의 함정 — 정상 import 로 갈 것.

**검수가 남긴 부기**: PREREG 가 worktree 에서 미추적이라 §0-A-6 의 집행 기록 추가분은
P2 의 `git diff` 로 감사되지 않는다. 이 기록은 **본 저장소에서 운전석이 직접 쓴 것**이며
같은 커밋에 들어간다.

### 0-A-7. GR-7 — unwired 27 의 계측·처분 판정 / GR-8 — 중복 빌드 명세 정합 검사 (신설 단 2개)

**GR-7** (계측→판정 — known-red 4 에 대한 §7 의 「미지판」):

- **스텝 1 (계측, 운전석)**: 27개 전원의 1회 실측. CPU C 18 + PY 1 + 그림자 5 는
  grammar-debug 에서 make 타깃 완주(rc·첫 FAIL 줄 채집). CUDA 3 은 nvcc 실존 시 **빌드만**
  시도(실행은 GPU 티어 — 시도 여부와 결과를 그대로 기재). 그림자 5 는 추가로 make recipe
  의 driver 단계 대 배터리 러너 커버리지의 대조표. **grammar-debug 의 nvcc 유무는 모른다**
  — 이 계측이 첫 실측이다.
- **스텝 2 (판정, fresh Fable)**: 행별 처분 = ① 배선(범주·배선처 지정) ② known-red 전환
  (수리 단 번호 필수) ③ 은퇴 — 단 은퇴는 **지키던 계약의 소멸을 판정문이 확정할 때만**
  (검증불가 고아 금지 원칙: 삭제는 처분의 편의가 아니라 판정의 결론이어야 한다).
- **산출**: 판정문 `docs/VERDICT_UNWIRED_GATES_2026-08-XX.md` + 등록부 처분 필드 기입 =
  커밋 1. **처분의 집행(배선·전환·은퇴 커밋들)은 캠페인 밖 후속 단들.**
- **기대치와 자문**: "27개 중 몇이 죽어 있는가" 의 사전 기대를 **적지 않는다**(§11-8 과
  같은 규율 — 예상은 판정이 아니다). 유일한 사전등록 기대 = 그림자 5 의 배터리 대응
  빌드는 rc=0 (Z-a2-09 는 GR-0 이후). 자문: 이 기대는 "배터리가 green" 가설과 동치라
  독립 증거력이 없다 — 회귀 확인용.

**GR-8** (메타 게이트 확장 — 발견 A 채널의 상설 감시):

- **계약**: 배터리 Build 명세와 Makefile recipe 가 **같은 테스트 소스를 공유하는 쌍
  전수**에서, `.c` 입력 집합의 불일치가 이름 있는 사유로 매 배터리마다 검출된다.
- **기전** [실측 근거]: `check_gate_registry.py` 가 `run_gate_battery` 를 import 해
  `build_specs()` 실물을 얻는다 — 모듈 수준은 상수·함수 정의뿐이고 main 은 가드돼 있어
  import 부작용 0 [실측], `generate_composition_d_fixtures` 도 동일 [실측]. 각 Build 의
  `.c` 집합 ↔ 대응 make 게이트 타깃(recipe 가 같은 `tests/*.c` 를 포함)의 `.c` 집합을
  비교, 불일치 = `build-spec-drift:<Build 이름>` FAIL. 대응 없는 Build(D·K·CP·Z-validator·
  Z-tau·Z-population·Z-canonical — make 대응물 없음 [실측: `selftest_zinert*` 타깃 부재])는
  정의역 밖 — 출력에 `pairs=N unpaired=M` 으로 정의역을 명시(인증 범위 과장 금지).
  import 실패·형상 변화는 기존 `battery-contract-unreadable` 계열로 fail-closed.
- **NC**: ① 합성 쌍에서 소스 1항 제거 → `build-spec-drift` 정확 사유.
  ② ★**소급 음성대조**: GR-0 이전 형상(Z-a2-09 에 population_contract.c 부재)을 사본
  트리에 재현해 검사 실행 → `build-spec-drift:Z-a2-09` 검출 — **발견 A 그 자체의
  재주입**이다. HEAD PASS 와 짝지어 무조건-FAIL 가설을 배제한다.
- **기대치와 자문**: HEAD PASS 는 공허 검사 가설로도 나온다(낮음) — NC ①②가 핵심.
- **기대 변경집합**: `scripts/check_gate_registry.py`(검사+NC) · 이 문서. **등록부
  무변경** — 대응은 테스트 소스 파일을 키로 도출 가능하므로 새 기재를 창설하지 않는다
  (기재를 늘리면 그 기재가 네 번째로 썩는다).

### 0-A-8. 개정판 단 분해 · 순서 · 폐합 조건 (이 표가 정본)

| 단 | 계약 (1줄) | 소속 | 의존 | 커밋 |
|---|---|---|---|---|
| **GR-0** ★신설 | 배터리 Z-a2-09 빌드 명세가 Makefile 정본과 재정합 | SH | 없음 | 1 |
| **GR-1′** ★재발주 | (원 계약 유지) + 배선 교차검증이 실호출 판별로 공허하지 않고, 고아 27은 unwired 로 정직 배속되며 처분 단이 강제된다 | SH | GR-0(P4′) | 1 |
| GR-2 ~ GR-6 | 원문 §4~§8 그대로 | | 의존의 "GR-1" 은 "GR-1′" 로 읽는다 | |
| **GR-7** ★신설 | unwired 27 전원이 실측되고 행별 처분이 판정된다 | SH | GR-1′ | 1 (판정문+기입) |
| **GR-8** ★신설 | 중복 빌드 명세의 갈라짐이 매 배터리마다 이름 있는 사유로 검출된다 | SH | GR-1′ | 1 |

**순서**: GR-0 → GR-1′ → (GR-2 병행 가능 · GR-7 스텝 1 계측은 GR-3 이후 무변조 병행 가능)
→ GR-3 → GR-4 → GR-5 → GR-6 → GR-7(판정·기입) → GR-8. 트리-변조 상시 1개 규약 불변.
GR-7·GR-8 을 말미에 두는 근거: known-red 4 소거(진행 중 Stage-4 측정면의 보호)가 임계
경로이고, 둘 다 그 경로를 막지 않는다.

**캠페인 폐합 조건 (개정 — §0 안전핀의 확장)**:
① known-red 4행 전부 소거(원문 유지) +
② unwired 27행 **전원에 GR-7 판정의 처분 기입** +
③ GR-8 정합 검사 편입.
unwired 행의 **소거**(처분 집행)는 폐합 조건이 아니다 — 상태 미지 27개의 집행에 폐합을
인질 잡히면 캠페인이 무한 확장된다. 단 처분 미기입 행이 하나라도 남으면 폐합 불가
(은폐 창고 금지 — known-red 와 같은 규율).

### 0-A-8b. GR-1′ 집행 기록 (운전석 실측, 2026-08-20)

**증거**: `/gpfs/kjhan/lumina/gates/gr1p_20260820T093939Z/`

| 게이트 | 결과 |
|---|---|
| **P1′** 완전성 | **PASS** — `make_targets=64 registry_entries=86` |
| **P2′** known-red sweep | **PASS** — `entries=4`, 4행 전부 실행·서명 대조 |
| **P3′** 음성대조 | **PASS 8/8** — R1a~d(기존) + R1e·f(`collective-reference-missing`) · **R1g(`unwired-now-referenced`)** · R1h(`known-red-recipe-drift`) 전부 정확 사유 |
| **P4′** 배터리 완주 | **PASS** — `GATE_BATTERY_SUMMARY verdict=PASS rc=0`, preflight **20행 전부 rc=0**. ★GR-0 착지 덕에 원리적으로 가능해진 첫 완주다 |
| **P5** milestone 집합 | **PASS** rc=0 |
| **P6** 변경집합 | **PASS** — `run_gate_battery.py`·`Makefile` 이 착지본과 **sha256 동일**(Codex 무접촉), `.gitignore` 정밀 부정 **1행만** |

**픽스처**: sha256 `c4072ff0feaf086774fccd4aac9a9acd2f00bac0f25923d71559840160b4cd3d`,
3179 B — 발주 전 실측치와 **동일**(무수정 추가). `git check-ignore` 플레인 rc=1(비무시).
**provenance 미상** — 동일 sha256 0건, 실 census 런(`a210_cancellation_census_mgpu_k10_fixed`
08-15 · `_k12` 08-15 · `_tau_refresh_k24` 08-16) stderr 에서 첫 줄 문자 일치 **0건**.
5줄·`evaluated_cells=100` 이라 합성을 시사하나 **단정하지 않는다**.
⚠파생(대장 후보): 회귀 게이트가 **출처를 증명할 수 없는 픽스처**에 의존한다.

### 검수 지적의 처분

| # | 지적 | 처분 |
|---|---|---|
| **R1** | §0-A-5(d) 픽스처가 **worktree 에 없어** 미이행 | ★**운전석 프로비저닝 실수다.** 미추적 파일은 `git worktree add` 로 전파되지 않는데 운전석이 `cp ... 2>/dev/null` 로 오류를 삼켰다. Codex 는 추측 생성을 거부하고 NOTES 에 공개했다 — 옳은 처신. **본 저장소에서 운전석이 원본을 `git add`** 해 이행(sha256 재검 완료) |
| **R2** | E11 특례 코드(`E11_WIRING` pin `:24-27`, `;` 분리 `:405-408`)가 delta 표 미열거 | 코드 제거 아님 — [실측] 정직 문안은 기존 preflight 배선검사와 충돌해 **코드 무변경으로는 `preflight-wiring-missing` 으로 죽는다.** 해소가 fail-closed 이고 정직화를 기계 강제한다. **사유·좌표를 여기 명기해 사전등록 정합을 회복**한다 |
| **R3** | 자기 만료의 **source 채널**(scripts/tests 탐색)은 NC 무커버 | 대장 기재. 검수자가 독립 프로브로 채널 생존을 실측했으나 **상설 장치는 없다**. 후속 단에서 저비용 NC 1건 |
| **R4** | `lexical_mentions` 의 **파일 단위 한계** 기재 거처가 NOTES(미커밋)뿐 | ★여기로 흡수: **허용 목록 파일 *안에* 새 실호출이 생기면 자기 만료 검사가 못 본다.** |
| **R5** | 발주서의 `check-ignore -v` 기대("무출력") 오류 | [실측] `-v` 는 매칭된 부정 규칙을 출력(rc=0); 실질 판정은 **플레인 rc=1**. 차기 발주서 수정 |

**검수 한계(그대로 기재)**: 검수자는 P4′·P2′·P5 를 검증하지 않았다(로그인 노드 실행 금지) —
그 셋은 위 표의 운전석 실측이다. `collective` 14 의 wiring 파일 **내부 셸 도달성**은
계약이 명시적으로 둔 한계 그대로 미증명이다.

### 0-A-9. 부록 A 스키마 v2 (unwired 확장 — v1 원문은 부록 A 에 보존)

```json
"schema": "lumina-gate-registry-v2",
  ... "category" 에 "unwired" 추가 ...
"unwired": {
  "observed": "YYYY-MM-DD",
  "disposition_rung": "GR-7",
  "lexical_mentions": ["<어휘 언급만 있는 파일 경로>"],
  "note": "<사실 기재 — 예: battery-shadow Z-a2-09>"
}
```

- `unwired` 블록은 category=unwired 에서만 허용·필수(`observed`·`disposition_rung` 필수,
  `lexical_mentions`·`note` 선택). 결손 = `unwired-row-incomplete:<이름>` FAIL.
- 처분 기입(GR-7 후)의 필드 형상은 GR-7 판정문이 정본이다 — 여기서 선점하지 않는다.

### 0-A-10. 이 개정이 모르는 것 (추측으로 메우지 않는다)

1. **unwired 27 의 빌드·실행 상태 전부** — GR-7 스텝 1 이 첫 실측.
2. **grammar-debug 의 nvcc 유무** — 동상.
3. **픽스처 provenance** — 커밋 전 실측 시도, 미상이면 미상으로 기재(§0-A-5-d).
4. **배터리 Z 러너가 그림자 5 의 make recipe driver 단계 검사를 재현하는지** — GR-7 계측 항목.
5. **fresh clone 완결성의 나머지** — `tests/fixtures` 는 이 1건뿐 [실측]이나, 배터리가
   요구하는 덱(`data/...`)·기타 경로의 추적 여부는 전수 미조사. `.gitignore` 의 `*.log`
   광역 규칙이 미래 픽스처에 같은 함정을 놓는다 — 일반 처방 미정(열린 질문으로 §11 에 추가).
6. **P7 로컬 clone 의 비용** — [추정] 소형. 실측이 다르면 강등 조항(P7)대로.

---

## 1. 실측 사실 (이 사전등록이 새로 잰 것 — §F~§I 의 인용은 생략)

### 1-1. ★주입 리터럴 전수 census (지시 사항: "같은 병이 다른 주입에도 있는가")

[실측] HEAD(`40090b1` 작업트리)에서 세 게이트의 모든 주입 표적을 개별 확인했다.

| 게이트 | 주입 | 표적 리터럴 | HEAD 출현 | 판정 |
|---|---|---|---|---|
| a209-source | 1 | `if(blocked_line_cells){` | 1건 | 생존 |
| a209-source | 2 | `a209_publication_free(&c);return invalid_eta_cells?5:3;` | 1건 | 생존 |
| a209-source | 3 | `a209_sobolev_line_eta(` | 1건 | 생존 |
| a209-source | 4 | `a209_line_generation_bracket(&line_generation_begin,NULL)` | 1건 | 생존 |
| a209-source | 5 | `(tau==0.0)?1.0` (formula 측) | 1건 | 생존 |
| a209-source | 6 | `&line_generation_begin,&line_generation_end` | 1건 | 생존 |
| a209-source | **7** | `nlte_tau_line_uses_nlte(authority,shell,n_shells)` | **0건** | **사멸 (§I-3 그대로)** |
| a209-source | 8 | `population_line_level_number_density(` | 4건(전역 치환) | 생존 |
| tau-census | 1 | `tau_sobolev_require_refresh(opacity, "compute_tau_sobolev");` | 1건(`:3221`) | 생존 |
| tau-census | 2 | `tau_sobolev_mark_computed(opacity, "nlte_update_tau_sobolev");` | 1건(`:19599`, `_with_authority` 안 — 라벨 문자열은 옛 이름 그대로) | 생존 |
| tau-census | 3·4 | 파일 말미 rogue 함수 **추가** | 항상 적용 가능 | 생존 |
| event-measure | 1~4 | 함수 서두 삽입: `single_packet_loop`(`lumina_transport.c:519`)·`a208_publish_cpu_opacity`(`:8832`)·`transport_kernel`·`d_trace_virtual_packet` | 4개 함수 정의 전부 실존 | 생존 — **단 CPU-A208 주입 표적이 래퍼**다. 앵커를 impl 로 옮기면 주입 표적도 함께 옮겨야 한다(안 옮기면 자기 측정면 밖에 주입하는 공허한 NC 가 된다) |

⟹ **주입 7 의 병(표적 사멸)은 다른 주입에는 없다.** 단 event-measure 의 CPU-A208 주입은
"표적은 살아 있으나 엉뚱한 함수(래퍼)"라는 자매 병을 갖고 있다 — GR-5 수리 범위에 포함.

### 1-2. `selftest-a2-10-line-saturation` 의 원인 — 미분류였던 것을 특정했다

| 항목 | 실측 |
|---|---|
| 실패 지점 | 타깃의 6개 스크립트 중 4번째 `tests/a2_10_cmfgen_line_saturation_comparison_selftest.py:135` 의 positive 케이스 (1~3번째는 통과했다는 뜻) |
| 사유 발원 | `scripts/compare_a210_cmfgen_line_saturation.py:138-140` — `summary["target_ion_zero_based"]` KeyError → `ComparisonError("Lumina summary target ion is invalid")` |
| 픽스처 | 같은 selftest 의 `summary()`(`:59-81`)가 만드는 payload 에 그 키가 **없다** |
| 생산자 | `scripts/summarize_a210_line_saturation.py:348` 이 그 키를 **쓴다** — 생산 스키마에는 존재 |
| 계보 | 체커의 키 요구는 `a00b991`(08-18)에서 태어났고, 픽스처는 `f6c2eb6` 이후 미갱신 [실측 `git log -S`] |
| 자매 결함 | `scripts/check_a210_line_saturation_per_ion_coverage.py:134` 는 같은 키를 `.get(..., 3)` **fail-open 기본값**으로 읽는다 — 두 체커가 같은 키를 하나는 fail-closed, 하나는 fail-open 으로 다룬다 |

⟹ **분류: 픽스처 노후(체커 스키마 확장에 positive 픽스처 미동행).** 앵커 노후와 같은 계급의
병이 C 심볼이 아니라 JSON 스키마 공간에서 난 것이다. 체커·생산자는 서로 정합하므로
**고칠 쪽은 픽스처다** — 이 분류의 확인은 GR-6 안에서 fresh 컨텍스트 검수가 재확인한다.

### 1-3. `event-measure-check` — grep 으로 잰 만큼만 적는다

[실측] `a208_publish_cpu_opacity_impl`(`:8705`) 안에 접근자 호출(`:8775`)·status 검사(`:8778`)·
`a208_publication_free(&candidate); return 3;`(`:8789`) 전부 실존. CPU-T03/GPU 의 요구
토큰들(`bf_event_measure_get`·`event_measure_t03_blocks`·`d_bf_event_measure_get`·
`d_bf_event_measure_record_block(...)` 등)도 각 파일에 실존. 금지 토큰
(`bf_get_event_measure(`·`d_bf_get_chi(`) 0건. GPU owner 패턴(`gpu_opacity_kernels.cu:149`) 실존.

[추정] 따라서 전 실패 목록은 CPU-A208 래퍼(5줄 위임, `:8832-8839`)에서 나온 3건
(접근자 부재 + 정책 토큰 2건)뿐일 것이다. **그러나 게이트를 안 돌렸으므로 이것은 추정이다** —
census 로그는 첫 줄만 보존했다. GR-5 1단계가 전수를 실측한다. 또한 make 타깃의 두 번째 명령
`compare_event_measure_spectra.py --selftest`(합성 픽스처 — 싸다 [실측 스크립트 판독])는
첫 명령 실패로 **한 번도 도달한 적이 없다** — 상태 미지.

### 1-4. 기타 수리 대상 좌표 (전부 [실측])

- 래퍼 실물: `a209_publish_cpu_emissivity` `:9408-9414`(3줄 위임) ·
  `a208_publish_cpu_opacity` `:8832-8839`(5줄 위임) · `nlte_update_tau_sobolev` `:19602-19608`(4줄 위임).
- 커밋 리터럴 현재형: `a209_publication_commit_counted(&opacity->cpu_emissivity,&c,ctr)` (`:9402`).
- `if(!ctr)return 5;` — a209 impl `:9120` (a208 측은 `if (!ctr) return 5;` `:8736`).
- `_counted` 호출자 전수: `lumina_plasma.c` 1곳(`:9402`, impl 안) ·
  `emissivity_publication.c` 1곳(`:217`, 구형 래퍼 `a209_publication_commit` 의 순수 위임
  `return a209_publication_commit_counted(pub,c,&g_ctr);`) · tests 다수(잣대 밖).
- 벌크 memcpy writer 의 소유 함수: `nlte_population_candidate_commit_bundle`(`:21238`, memcpy `:21271`) ·
  `nlte_population_candidate_commit_seed_material`(`:21346`, memcpy `:21379`) — 모두 `src/lumina_plasma.c`.
- `nlte_update_tau_sobolev_with_authority`(`:19457`) 의 괄호: require `:19463` < 쓰기 `:19521`·`:19573` < mark `:19599`.
- CUDA 경유 검사 토큰 `nlte_update_tau_sobolev(nlte, atom, opacity` — `lumina_cuda.cu` 에 1건 생존.

### 1-5. ★`function_body` 는 3벌이 아니라 **4벌**이다 (§G-3-5 의 답의 일부)

[실측] `scripts/a2_07_population_census.py:96` 에 네 번째 구현이 있고,
`nlte_assemble_rate_matrix` 등 함수 이름에 앵커를 건다. 이 스크립트는 회귀 목록 밖 —
`run_a2_07_grammar_debug.sh`·`run_a2_07_lageunha.sh` 라는 **수동 레인**에서만 불린다.
현재 green 인지 **모른다**(미실행). 처분: 이 캠페인 범위 밖 — 등록부에 `manual-lane` 으로
등재하고(§3 GR-1), 앵커 건강은 열린 질문(§8)으로 남긴다.

---

## 2. 설계 결정 (증거 패킷 §5 의 다섯 질문)

### Q1 (단 분해·순서) — §0 에 확정. (e)-먼저 + known-red 등록부.

### Q2 (메타 게이트의 정의역) — **등록부 완전성만. 앵커-심볼 존재 검사는 별도 메타 게이트로 만들지 않는다.**

근거: 앵커가 죽으면 게이트 자신이 이미 시끄럽게 죽는다 — `function not found` RuntimeError
(rc≠0), 토큰 부재 FAIL, `injection-N-not-applied` FAIL. 셋 다 fail-closed 로 설계돼 있었고
실제로 그렇게 죽어 있었다. **유일하게 없던 것은 "누가 돌리는가"다.** 편입이 되면 앵커
사망은 리팩터 커밋 당일 배터리 red 로 나타난다. 앵커 목록을 별도 명세로 복제하면 그 명세가
또 하나의 썩는 등록부가 된다.

단, 앵커가 죽는 대신 **조용히 green 이 되는** 경로는 게이트 쪽을 고쳐 막는다.
§I-5 의 사각 6류에 대한 이 캠페인의 처분:

| # | 사각 | 이번 캠페인 | 어디서 |
|---|---|---|---|
| 1 | 벌크 승격 경로 | tau 측(`:21271`·`:21379`)만 — H-5-1 이 "같은 단" 으로 명령한 범위 | GR-2 판정 + GR-4 |
| 1' | 방출률 측 벌크 이식(`:21305`·`:21412` — `cpu_emissivity` 구조체·`line_source_S` memcpy) | **미룸** (인증 확장) | 별도 단, §H-5-3-②③ 과 묶음 |
| 2 | 래퍼 무검사화 | 덮음 — 래퍼 3개의 순수 위임 본문 **정확일치** 검사 신설 | GR-3(2개)·GR-5(1개) |
| 3 | 주석 속 토큰 오탐(조용한 green) | 덮음 — 추출기가 주석을 제거(문자열 리터럴은 보존 — reason 토큰이 문자열에 산다) | 부록 B (GR-3 부터) |
| 4 | 술어 함수 본문 무검사 | **미룸** (인증 확장) | 별도 단 |
| 5 | 파일 범위 2개 | **미룸** (인증 확장) | 별도 단 |
| 6 | `function_body` 정규식의 호출부 매치 가능성 | 덮음 — 정의-앵커 매칭(괄호 깊이 스캔+`{` 요구+유일성) | 부록 B |

미룸 3건의 공통 근거: 이 캠페인은 **복원 + "복원된 초록불이 거짓말하지 않기 위한 최소
확장"(H-2 의 명령)** 까지다. 그 밖은 인증 범위의 확장이라 각자 계약·판정이 필요하다
(§I-2: 인증 없이 초록 표면만 키우지 말 것 — 확장에도 같은 원칙이 적용된다).

### Q3 (`function_body` 4벌) — **통합한다: `scripts/gate_source_lib.py` 신설(GR-3), GR-4·GR-5 가 채택.**

- 근거: 4벌은 이미 발산했다(반환형·주입 helper 유무가 제각각 [실측]). 사각 3·6 의 수리를
  4곳에 복붙하면 다음 발산이 예정된다.
- 커밋 귀속: lib 는 GR-3 커밋에 속한다(그 단의 계약을 이행하는 기구). GR-4·GR-5 의 채택은
  각자의 커밋. `a2_07_population_census.py`(수동 레인)는 **건드리지 않는다** — 범위 밖.
- lib 자체의 시험: `tests/gate_source_lib_selftest.py` 신설(정의/호출부 판별·래퍼 중복
  정의 유일성 위반 검출·주석 토큰 무시 — 각 케이스 이름 있는 예외), preflight 편입.

### Q4 (음성 대조 설계) — 3층. **"게이트가 이제 PASS 한다"는 어느 층에도 없다.**

1. **자기-NC 의 이름 있는 사유 고정.** 현행 세 게이트의 NC 판정식은 `inspect(...)가 비어
   있지 않으면 검출` — **무관한 사유로 실패해도 검출로 셈한다.** §E 가 08-08 에 세 단을
   강등시킨 OR-결함과 같은 계급이다. 수리: 주입마다 **기대 사유 정확 문자열**을 고정하고
   (부록 C), 그 사유가 실패 목록에 없으면 `injection-N-wrong-reason` 으로 FAIL.
   `injection-N-not-applied` fail-closed 는 보존(증거 패킷 §4).
2. **외부 사본-트리 주입 1건/수리 단.** 자기-NC 는 게이트 코드가 자기 가설을 자기에게
   시험하는 순환 위험이 있다(N6-3 교훈). 각 수리 단에서 운전석이 scratch 사본 트리
   (저장소 불변)에 대표 실위반을 **손으로** 주입하고 게이트를 돌려 지정 사유 FAIL 을 받는다.
   HEAD=PASS 와 짝지어야 증거다(무조건-FAIL 게이트 가설을 배제하는 것은 이 짝이다).
3. **known-red 서명 드리프트 트립와이어**(GR-1) — 수리 착지 자체가 배터리에서
   `unexpected-pass` red 로 관측되고, 행 소거가 그 단의 회귀 편입 증거가 된다.

### Q5 (승격 경로 포함 여부) — **판정(GR-2)까지만 포함, 측정면 편입은 미룬다.**

- GR-2 는 §H-5-3 의 신규 판정 3건을 **한 패킷으로 발주**한다: ① 커밋 경로 memcpy writer 의
  등록부 지위(GR-4 를 막는 유일한 판정) ② 슬랩↔authority 짝의 런타임 결속 부재(스탬프 없음)
  ③ authority 거부 카운터 부재. ②③은 GR-4 를 막지 않는다 — 판정 결과는 대장·후속 단으로.
- 방출률 측 벌크 이식(§I-5·§I-7-6)의 측정면 편입은 이 캠페인 밖(위 Q2 표의 1').

---

## 3. GR-1 — 게이트 등록부와 회귀 편입 (구조 단)

**[개정 1]** GR-1 은 이대로 폐합 불가로 판정됐다(P1 이 현행 파서로 27건의 거짓 배선을
인증 — §0-A-2 결정 5). 착지 구현은 보존하고 **재발주 명세(GR-1′) = §0-A-5** 가 정본이다.
아래 원문은 초판 발주분의 기록으로 보존한다. 특히 "21개 고아의 배속" 문단의 수치는
§G census 스냅샷 기준이며, 개정 1 의 재측정으로 **진성 고아는 21+27=48/62** 로 정정됐다
(§0-A-1).

### 계약

> 게이트 성격의 make 타깃 62개 전원이 기계가 매번 검사하는 등록부에 정확히 하나의
> 범주로 배속되고, 죽은 4개는 서명 고정 known-red 로 **계속 실행·관측**되며,
> 등록부 완전성 검사 자체가 배터리 preflight 로 매번 돈다.

### 등록부 범주 (스키마는 부록 A)

| 범주 | 뜻 | 기계 검증 |
|---|---|---|
| `battery-core` | 배터리 D/K/Z/CP 본체 | EXPECTED_ROWS 존재 |
| `preflight` | `run_gate_battery.py` PREFLIGHTS 행 | 행 실존 교차검증 |
| `collective:<타깃>` | 다른 make 타깃/`.sh` 가 부른다 | 참조 실존 교차검증(연속행 인지 파서) |
| `milestone` | 빌드 요구 — 집합 타깃 `selftest-registry-milestone` 소속, 마디(3층 인수의 전량 층)에서 실행 | 집합 타깃 등재 교차검증 |
| `run-dependent` | 런/GPU 필요 — 요구 자원 명기 | 필드 존재 |
| `manual-audit` | 의도적 수동 감사 타깃(예: rc=3 semantics 의 덱 census) | 사유 필드 존재 |
| `manual-lane` | 수동 `.sh` 레인 전용 스크립트(예: a2_07 census) | 참조 실존 |
| `known-red` | 죽은 게이트 — 서명 고정, 매번 실행·대조 | pin 필드 전부 + 수리 단 번호 |

**21개 고아의 배속(사전등록)**: known-red 4(`event-measure-check`·`selftest-sh-radeq-source`·
`selftest-tau-writer-census`·`selftest-a2-10-line-saturation`) · run-dependent 1
(`selftest_mc_evt_access`, GPU) · manual-audit 1(`bf-edge-census`) · 나머지 15는 규칙으로
배속한다 — **순수 파이썬·무빌드·grammar-debug 단독 <10s 실측 → `preflight`, 아니면
`milestone`.** [추정] a2-10 픽스처 selftest 6종·`selftest-bf-edge-census` 는 preflight,
빌드 요구(`selftest_a2_17_jnu_seed`·`selftest_grid_roundtrip`·`selftest_sh_grid_loader`·
`selftest_emiss_e11_fluor_matrix`·`selftest_stage32_rung1` 등)는 milestone.
실측치가 추정과 다르면 규칙이 이긴다(규칙이 사전등록이고 셀 값은 아니다).

### 기대 변경집합 (여기 없는 변경은 위반)

| 파일 | 변경 |
|---|---|
| `scripts/gate_registry.json` | **신설** — 62개 전수 + PREFLIGHTS 스크립트 + manual-lane 스크립트, 부록 A 스키마 |
| `scripts/check_gate_registry.py` | **신설** — (i) 완전성: Makefile 을 연속행 인지로 파싱해 게이트 성격 패턴(`selftest*`·`*-check`·`*-census`·`*-gate`) 타깃이 전원 등록부에 있는가 + 각 범주의 배선 주장 교차검증; (ii) known-red sweep: 각 행의 명령열을 실행해 (실패 위치, rc, 첫 FAIL 줄, FAIL 줄 수) 를 pin 과 정확 대조; (iii) 자기-NC(§아래). 출력 태그 `[GATE-REGISTRY][COMPLETENESS\|KNOWN-RED\|NEGATIVE-CONTROL]` |
| `scripts/run_gate_battery.py` | PREFLIGHTS 에 `("GATE_REGISTRY", "scripts/check_gate_registry.py", ())` 1행 + 이번에 preflight 로 배속되는 green 게이트들의 행(스크립트당 1행, 이름 명시) 추가. 다른 변경 0 |
| `Makefile` | 집합 타깃 `selftest-registry-milestone`(milestone 배속분 나열) + `gate-registry-check`(메타 게이트 단독 호출용) 신설. 기존 규칙 무변경 |
| `docs/RUNG_GATE_REPAIR_LADDER_2026-08-20.md` | 이 문서 (집행 기록 추가) |

**접촉 금지**: `src/` 전부, 기존 게이트 스크립트 3개(GR-3~5 의 몫), tests(GR-6 의 몫),
덱·`/gpfs` 정본, env 노브(신설 0).

### 게이트 표

| # | 조건 | 판정 자료 |
|---|---|---|
| **P1** | 완전성 검사 PASS — 62개 전원 배속, 배선 주장 교차검증 0 위반, 미등재 게이트 성격 타깃 0 | 메타 게이트 출력 |
| **P2** | known-red sweep — 4행 전부 pin 정확 일치(pin 값은 집행 시 grammar-debug 재실행 실측으로 기입) | sweep 출력 |
| **P3** | NC-R1a~R1d 전부 이름 있는 사유로 시연 | 메타 게이트 NC 출력 |
| **P4** | 배터리 1회 완주(grammar-debug): D=19·K=7·Z=12·CP=4 행 불변 + 신규 preflight 행 전부 rc=0 | 배터리 로그 |
| **P5** | milestone 집합 타깃 1회 완주 — green 이던 게이트 전원 green 유지 | 실행 로그 |
| **P6** | 변경집합 = 위 표와 정확 일치 | `git show --stat` |

### 음성 대조 (전부 in-memory/합성 — 트리 불변)

| # | 주입 | 기대 (정확 사유) |
|---|---|---|
| **NC-R1a** | 합성 Makefile 에 게이트 성격 타깃 1개를 등록부 없이 추가 | `unregistered-gate-target:<이름>` FAIL |
| **NC-R1b** | 합성 known-red 행(소형 고정 명령)의 pin 서명을 한 글자 변조 | `known-red-signature-drift:<이름>` FAIL |
| **NC-R1c** | rc=0 인 명령을 known-red 로 가장 등재 | `known-red-unexpected-pass:<이름>` FAIL |
| **NC-R1d** | 합성 Makefile 의 `.PHONY` 백슬래시 **연속행**에만 이름을 숨긴 미등재 타깃 | NC-R1a 와 동일 검출 — **G-0 에서 고아 7개를 숨긴 바로 그 결함의 재주입** |
| (fail-closed) | 등록부 JSON 파손 / known-red 필수 필드 결손 | `registry-unreadable` / `known-red-row-incomplete` FAIL — 조용한 스킵 금지 |

### 기대치와 자문 ("다른 가설도 같은 값을 내는가")

| 기대 | 자문 | 증거력 |
|---|---|---|
| 배터리 preflight 에 GATE_REGISTRY rc=0 | **낸다** — 아무것도 안 검사하는 등록부도 rc=0 | 낮음 — NC 4건과 결합해서만 의미 |
| NC-R1a~d 4/4 정확 사유 | 못 낸다 — 검사 없는 등록부는 NC 를 못 깨운다 | 핵심 |
| D/K/Z/CP 행수 불변 | 낸다 — 배터리 본체 무접촉이면 자명. 그래서 이것은 증거가 아니라 **회귀 확인**(우리가 배터리 파일을 만졌으므로 필요) | 회귀 확인용 |
| known-red 4행 pin 일치 | **부분적으로 낸다** — pin 을 실측에서 베꼈으므로 당연 일치. 증거력은 일치가 아니라 **이후 매 배터리 실행이 드리프트를 감시한다는 배선** 자체 | 배선 확인용 |

### 철회·분기

| 관측 | 처분 |
|---|---|
| 62개 census 가 재집계에서 달라짐(타깃 증감) | 등록부는 재집계 실측을 따른다 — §G 의 62/21 은 스냅샷이지 계약이 아니다. 차이는 집행 기록에 기재 |
| 배터리 D/K/Z/CP 행수 변화 | **중단** — preflight 추가가 본체를 건드렸다는 뜻. 원인 특정 전 커밋 금지 |
| green 이던 15개 중 재실행 FAIL 발생 | 그 게이트를 known-red 로 배속(수리 단 번호 포함)하고 계속 — 단 새 FAIL 의 분류는 별도 계측 후 |

---

## 4. GR-2 — 판정 발주: 벌크 이식 writer 의 지위 (트리 무변조)

**발주물**: 운전석 증거 패킷(계측은 §1-4 좌표 + `candidate_bundle_commit_preflight`/
`candidate_material_commit_preflight` 의 실물 body) → **fresh Fable** 판정.
산출 = `docs/VERDICT_TAU_BULK_TRANSPLANT_2026-08-XX.md` 커밋 1개.

**판정 질문 (고정)**:
1. `:21271`·`:21379` 의 memcpy 이식은 "closed writer set" 계약 아래 **어떤 지위**인가 —
   (i) 괄호 의무(require→write→mark)를 지는 writer 인가, (ii) 괄호 대신 preflight+세대
   이식이라는 **다른 계약**을 지는 별개 류(transplant)인가, (iii) 위반인가.
2. (ii)라면 그 류의 **정적 인증 가능 표면**은 무엇인가 — 게이트가 무엇을 검사하면
   "이 memcpy 는 인증된 transplant" 라고 말할 자격이 생기는가(최소: 등록 함수 스팬 내 +
   스팬 내 preflight 호출 실존 + 세대 이식 필드 접촉 실존).
3. §H-5-3-②(슬랩↔authority 런타임 결속 부재)·③(거부 카운터 부재)의 처분 —
   대장 기재로 종결인가, 후속 단 개설인가.

**분기**: 판정 (i)/(ii) → GR-4 가 그 명세대로 등록부·검사를 짠다. 판정 (iii)(위반) →
**src 수리는 이 캠페인 밖**(잣대 수리 계약 위반이므로) — GR-4 는 두 memcpy 를
known-violation 행으로 붉게 남기고 부분폐합으로 보고, 별도 수리 단을 발의한다.

---

## 5. GR-3 — `selftest-sh-radeq-source` 수리 (§I-7 의 조직화)

### 계약

> 게이트가 HEAD 의 실제 계약 표면(impl·`_counted`·`_by` 술어)을 정확일치로 다시 재고,
> 8개 주입 전부가 **적용되고, 의도한 검사에, 이름 있는 사유로** 검출된다.

### 기대 변경집합

| 파일 | 변경 |
|---|---|
| `scripts/gate_source_lib.py` | **신설** — 부록 B 명세의 정의-앵커 추출기(+주석 제거) |
| `tests/gate_source_lib_selftest.py` | **신설** — 부록 B 의 판별 케이스들, 이름 있는 예외 검증 |
| `scripts/check_a209_source_failclosed.py` | (1) 앵커 이동: `a209_publish_cpu_emissivity`→`_impl` · `nlte_update_tau_sobolev`→`_with_authority` (§I-7-1). (2) 커밋 리터럴을 `a209_publication_commit_counted(&opacity->cpu_emissivity,&c,ctr)` **정확일치**로 고정 — 알터네이션 금지(§I-2). (3) **주입 7 재표적**: `a209_upper_population_for_tau` 스팬 **내부** 치환 `nlte_tau_line_uses_nlte_by(` → `a209_alt_authority_stub(` (스팬 내 미발견 시 `injection-7-not-applied` 보존) + 정적 검사 토큰도 `nlte_tau_line_uses_nlte_by` 로 갱신. (4) 신규 검사 3건(§I-7-4): 래퍼 2개(`:9408`·`:19602`)의 순수 위임 본문 **정확일치** · impl 의 `if(!ctr)return 5;` 토큰 · `_counted` 호출자 전수 = plasma 1곳(impl 스팬 내)+`emissivity_publication.c` 구형 래퍼 1곳. (5) NC 판정식을 부록 C 의 기대-사유 고정으로 교체(`injection-N-wrong-reason` 신설). (6) `function_body` 를 lib 채택으로 교체 |
| `scripts/run_gate_battery.py` | PREFLIGHTS 에 SH_RADEQ_SOURCE 행 + (GR-3 에서 lib selftest 행) 추가 |
| `scripts/gate_registry.json` | known-red 행 소거 → preflight 행 2건(게이트·lib selftest) |
| 이 문서 | 집행 기록 |

**접촉 금지**: `src/` 전부(**19건 중 실위반이 발견되어도** — 아래 분기), 다른 게이트 2개, 덱.

### 게이트 표

| # | 조건 | 판정 자료 |
|---|---|---|
| **P1** | HEAD 에서 `[SH-RADEQ-0][STATIC][PASS]` + `[NEGATIVE-CONTROL][PASS] injections=8 detected=8` — 사유 고정판으로 | 실행 로그(grammar-debug) |
| **P2** | 부록 C 의 8행 전부 — 주입별 **기대 사유 정확 문자열** 로그 확인 | NC 상세 출력 |
| **P3** | 외부 사본-트리 주입: `a209_upper_population_for_tau` 스팬에 `opacity->line_source_S` 읽기를 손으로 삽입 → `A2-09 production reads forbidden quotient source` 정확 사유 FAIL. HEAD=PASS 와 짝 | 사본 트리 diff + 게이트 출력 |
| **P4** | lib selftest PASS (부록 B 케이스 전부) | 실행 로그 |
| **P5** | 배터리 preflight 완주 — 신규 행 green, known-red 4→3 (서명 sweep 가 이 게이트의 pass 를 `unexpected-pass` 로 잡고, 행 소거 후 green — 트립와이어 작동의 실증) | 배터리 로그 2회(소거 전/후) |
| **P6** | 변경집합 준수 | `git show --stat` |

### 기대치와 자문

| 기대 | 자문 | 증거력 |
|---|---|---|
| STATIC 19→0 | **낸다** — 검사를 지운 게이트도 0 을 낸다. 그래서 이 기대는 P2·P3 없이 증거가 아니다 | 낮음 |
| 주입 8건 각각 지정 사유 | 못 낸다 — 검사를 지우면 사유가 안 나오고, OR-게이트면 무관 사유가 나온다(부록 C 가 구별) | 핵심 |
| 외부 주입 FAIL + HEAD PASS 짝 | 못 낸다 — 무조건-PASS 도 무조건-FAIL 도 짝을 못 만든다 | 핵심 |

### 철회·분기

| 관측 | 처분 |
|---|---|
| 앵커 교정 후 STATIC 에 **앵커 밖 실위반**이 나타남(§I-1 은 19건 전부 노후로 확정했지만, 검사 강화·신설 3건이 새 것을 드러낼 수 있다) | **보고·중단** — src 수리 금지(잣대 수리 계약). 실위반은 FINDING 문서로 기재하고 별도 단 발의. 이 단은 게이트가 그 위반을 **이름 있는 사유로 보는 상태**로 부분폐합 가능(known-red 재등재, 사유=실위반) |
| 주입 7 재표적이 스팬 내 표적을 못 찾음 | `injection-7-not-applied` 로 red 유지 — 조용한 통과 금지(설계 보존). 표적 재선정 후 재발주 |

**[개정 2]** 위 게이트 표의 P3 는 초판 기대 변경집합 (1)~(6)과 모순이었고(어느 항목도
금지-읽기 스캔을 upper 로 확대하지 않는다), 집행에서 정확히 그 이유로 실패했다.
이 단은 **GR-3′ 로 재발주**된다 — §5-A 가 정본(다섯 결정·delta 변경집합·개정판 게이트
표·부록 C-1a). 원문은 초판 발주분의 기록으로 보존한다.

---

## 5-A. 개정 2 (2026-08-20, 저자 Fable) — GR-3 P3 실패의 처분: 스캔 확대 + GR-3′ 재발주

근거 증거: 운전석 증거 패킷(`/tmp/claude-10396/gr3_p3d/EVIDENCE.md`) + **이 개정의 독립
재측정**(§5-A-1 — 전 항목을 HEAD 작업트리(GR-3 delta 적용분)에서 직접 다시 쟀고,
패킷을 두 곳 보완한다).

### 5-A-1. 독립 재측정 — 패킷 확인 + 보완 2건 + 신규 실측 4건

[실측 — 전부 이 개정 저자가 직접 확인]

**패킷 확인**: 게이트의 금지-읽기 검사(`check_a209_source_failclosed.py:79-80`)는
`publish`(= `a209_publish_cpu_emissivity_impl` 본문) 하나만 스캔한다. 같은 주입의
스팬별 대조 실험(upper 미검출 / impl 검출)의 논리 구조도 재확인 — 검사는 깨지지 않았고
범위가 좁다.

**보완 1 — 호출부는 2건이 아니라 4건이다**: `a209_upper_population_for_tau` 의 호출은
`:9008`(공개 접근자 `lumina_line_upper_population_for_tau`) · `:9037`(벌크 fill —
`lumina_main.c:387`·`lumina_cuda.cu:8006` 의 Q_E formal 생산자가 소비) 외에
**`:9243`(impl 의 생산 선-루프 — eta 공식에 n_upper 를 실공급하는 자리)** ·
**`:14915`(`a210_private_line_energy_build` `:14684` — A2-10 line-net 에너지)** 가 있다.
패킷 §3 의 "eta 공식에 먹일 생산자" 주장의 실제 근거는 `:9243` 이다. upper 에 밀반입이
생기면 A2-09 방출·Q_E formal·A2-10 line-net **세 소비자가 동시에** 오염된다.

**보완 2 — upper 는 게이트 탄생 이래 한 번도 커버된 적이 없다**: 구판(커밋 `fcaaad2`)의
금지-읽기 검사는 `function_body(text,"a209_publish_cpu_emissivity")` — 정규식 구조상
impl(이름 뒤 `_impl`)도 프로토타입(`;` 종결)도 매칭 불가라 **`:9408` 래퍼 3줄 본문**을
스캔했다(§I-1 의 "래퍼 스캔 산물" 그대로). 즉 P3 의 가정(upper 검출)은 구판·수리판
어느 판본에서도 성립한 적이 없다.

**신규 실측 1 — upper 에는 opacity 접근 경로 자체가 없다**: `a209_upper_population_for_tau`
의 시그니처(`:8953-8958`)에 `OpacityState` 가 없고, `lumina_plasma.c` 파일 전역에
OpacityState 객체·포인터 전역변수가 없다(전역은 `g_ew_tau_authority` 류뿐).
⟹ P3 의 인공 주입 텍스트는 **컴파일 불가**다. (이는 이 게이트 NC 전반의 성격이기도
하다 — 주입 3 의 호출 개명 등도 컴파일 불가. 정적 텍스트 게이트의 본질이며 결함이 아니다.)

**신규 실측 2 — bulk·writer 는 line_source_S 를 합법 접촉한다**:
`compute_tau_sobolev`(`:3219`)가 `:3246-3251` 에서 `opacity->line_source_S[at]=0.0` ·
`opacity->line_source_validity[at]=A208_EXACT_ZERO` 를 **쓴다**(영점화).
`nlte_update_tau_sobolev_with_authority`(`:19457`)는 `:19515-19593` 에서 line_source_S 의
**생산자**다. 금지-읽기 정규식은 토큰 매칭이라 **읽기/쓰기를 구별하지 못한다** ⟹
이 두 스팬으로의 확대는 HEAD 에서 즉시 false red 를 낸다.

**신규 실측 3 — formula 는 순수 스칼라**: `a209_sobolev_line_eta`
(`emissivity_publication.c:52`)는 스칼라 인자만 받고, 그 파일 전체에
`line_source_S`·`OpacityState` 출현 0건.

**신규 실측 4 — 확대 대상 스팬은 이미 추출되고 있다**: 게이트는 `upper`(`:90`)·
`formula`(`:104`)를 이미 `body(...)` 로 뽑아 다른 검사에 쓰고 있다 — 확대는 기존
정규식을 두 스팬에 더 적용하는 것뿐, 새 추출 기구가 필요 없다.

**부기 — 계약 문언과의 대조**: impl 내부 주석 `:9185` "No line_source_S read or division
is permitted here" [실측]. n_upper 는 직접형 `n_u*A_ul*h*nu*beta/(4*pi*dnu)` 의 입력이므로,
n_upper 를 S 역산으로 만들면 겉은 직접형이되 금지된 몫-소스가 뒷문으로 들어온다 —
[판정] 계약("never line_source_S")의 의미론적 표면은 n_upper 생산자를 포함한다.

### 5-A-2. 다섯 결정 (패킷 §5 의 질문에 대한 답)

| 질문 | 결정 | 근거 |
|---|---|---|
| 1. P3 표적 vs 게이트 범위 | **(c) 둘 다 — 단 "표적이 틀렸다"는 기각.** 표적(upper)은 대표 실위반 자리로서 옳다(§5-A-1 부기 — S 역산 밀반입이 금지 계약의 정확한 우회 경로). 실제 결함은 ① 사전등록의 자기모순 — P3 가 upper 검출을 요구하면서 기대 변경집합 (1)~(6) 어디에도 스캔 확대가 없다 [실측] ② 게이트 범위가 계약 문언·docstring("never line_source_S")보다 좁다. ①은 문언·변경집합 정정으로, ②는 확대로 각각 처분 | P3 는 제 역할을 했다 — §E 계급의 음성 대조가 실패해 구멍을 드러냈다. §H-2 "복원된 초록불이 거짓말하지 않기 위한 최소 확장" 조항의 적용 대상 |
| 2. 확대 범위 | **upper + formula 까지. bulk·writer 는 확대 금지.** | upper·formula 는 HEAD 무접촉 [실측] — false red 위험 0 + 방출 값사슬의 두 절점. bulk·writer 는 합법 접촉 [실측 §5-A-1] — 정규식이 읽기/쓰기 무구별이라 확대 즉시 false red 이고, 이를 green 으로 만들려면 정규식 약화·예외행이 필요해진다(fail-open 방향 — 금지). 확대 후에도 못 보는 것 = §5-A-5 에 전수 기재 |
| 3. 거처 | **GR-3 안 — [개정 2] + GR-3′ 재발주**(GR-1→GR-1′ 선례; 착지 구현 보존, delta 만) | upper 커버리지는 GR-3 자신의 사전등록 게이트(P3)가 **이미 건 주장**이다 — 건 주장의 이행은 그 단의 몫. §2-Q2 의 미룸 3건(1'·④·⑤)과 구별: 그것들은 어느 게이트도 안 걸었던 주장 + 새 기구 필요; 이것은 이미 추출 중인 스팬 [실측 §5-A-1-신규4]에 기존 정규식 적용 — "최소 확장" 의 정의역 안 |
| 4. 폐합 가부 | **지금 폐합 없음 — 부분폐합 커밋도 하지 않는다.** GR-3′ delta 착지 + 개정판 게이트 전 통과 시점에 **커밋 1개로 전폐합** | delta 가 미커밋이므로 "5/6 부분폐합 커밋 후 GR-3b" 는 실패한 사전등록 게이트를 가진 잣대를 원장에 올리는 것 — 불필요한 중간 상태. (a)-only(P3 를 impl 로 줄여 6/6 선언)는 기각: 사전등록 게이트를 구현에 맞춰 줄이는 것은 "초록불이 인증 범위를 과장" 의 거울상(잣대를 통과물에 맞춰 후퇴시키기) |
| 5. upper 의 현실 경로 | **오늘은 없다 [실측]** — 시그니처에 OpacityState 없음 · 파일 전역 opacity 없음 · 주입 텍스트 비컴파일. 위반의 현실화는 의도적 배관 리팩터(파라미터 추가 등)를 요구한다 | 그 리팩터가 정확히 정적 게이트가 지키라고 있는 미래다(§H-2·§G-2 의 병력: 리팩터 하나가 잣대를 조용히 무력화). 시급도 낮음 ≠ 가치 없음 — 비용 수 줄, false red 위험 0. **(b)의 시급도 = 낮음, 그러나 주장이 이미 걸려 있으므로 지금 이행** |

### 5-A-3. GR-3′ — 재발주 명세 (delta; GR-3 착지 구현은 보존)

**기대 변경집합 (delta — GR-3 착지분에 추가; 여기 없는 변경은 위반)**:

| 파일 | 변경 |
|---|---|
| `scripts/check_a209_source_failclosed.py` | (7) 금지-읽기 정규식(기존과 동일 패턴)을 `upper`·`formula` 스팬에 각각 적용 — 사유는 **스팬 지명**: upper → `A2-09 upper-population producer reads forbidden quotient source` · formula → `A2-09 line-eta formula reads forbidden quotient source` (impl 의 기존 사유 문자열 **불변**) (8) NC 주입 9·10 신설(부록 C-1a) — 스팬 내 실앵커 치환으로 in-memory 삽입, 미적용 시 `injection-N-not-applied` 보존, `EXPECTED_REASONS` 10행 (9) docstring 에 인증 스팬 명기: 금지-읽기 = impl·upper·formula 텍스트, 래퍼 2개 = 정확일치 pin |
| 이 문서 | 집행 기록 |
| (변경 0) | `scripts/gate_source_lib.py` · `tests/gate_source_lib_selftest.py` · `scripts/run_gate_battery.py` · `scripts/gate_registry.json` · `src/` 전부 · 다른 게이트 2개 — GR-3 착지분 그대로 |

**Codex 추가 계약 조항** (§9 에 추가 적용):
- **bulk(`compute_tau_sobolev`)·writer(`nlte_update_tau_sobolev_with_authority`) 스팬으로의
  확대 금지** — [실측 §5-A-1] 합법 접촉이 있어 false red. 이 두 스팬을 green 으로 만들기
  위한 정규식 약화·예외행 카브아웃 일절 금지.
- 부록 C-1 의 1~8행 사유 문자열·기존 정확일치 pin 전부 불변.
- 주입 9·10 의 삽입 앵커가 스팬 내에서 유일하지 않거나 부재하면 고치지 말고 보고.

**게이트 표 (개정판 — GR-3 판정문은 이 표와 대조한다)**:

| # | 조건 | 판정 자료 |
|---|---|---|
| **P1′** | HEAD 에서 `[SH-RADEQ-0][STATIC][PASS]` + `[NEGATIVE-CONTROL][PASS] injections=10 detected=10` | 실행 로그(grammar-debug) |
| **P2′** | 부록 C-1+C-1a 의 **10행** 전부 — 주입별 기대 사유 정확 문자열(1~8행 문자열 불변 확인 포함) | NC 상세 출력 |
| **P3′** | 외부 사본-트리 주입 짝 2건: ① `a209_upper_population_for_tau` 스팬에 `opacity->line_source_S` 읽기 삽입 → `A2-09 upper-population producer reads forbidden quotient source` 정확 사유 FAIL ② impl 스팬 동일 주입 → `A2-09 production reads forbidden quotient source` FAIL. 각각 HEAD=PASS 와 짝. (②의 08-20 운전석 실측은 구판 기준 — 확대판에서 재실행) | 사본 트리 diff + 게이트 출력 |
| **P4** | (원문 그대로) lib selftest PASS — lib 무변경이므로 기존 PASS 유효하나 배터리 재완주에 포함돼 재확인된다 | 실행 로그 |
| **P5′** | 배터리 재완주 1회 green(known-red 3 불변). ★초판 P5 의 트립와이어 시연(소거 전/후 2회, `unexpected-pass` 관측)은 08-20 실측으로 **이행 완료** — 게이트 내부 스캔 범위와 무관한 등록부 기구의 시연이므로 재요구하지 않는다 | 배터리 로그 + 초판 P5 실측 기록 |
| **P6′** | 변경집합 = GR-3 착지분 + 위 delta 표와 정확 일치. ⚠[실측] 현 작업트리에는 GR-3 밖 변경 3건(`scripts/stage_a210_line_saturation_diagnostic.sh` · `validation/a2_09/A2_09_SELFTEST.json` · `validation/a2_10/A2_10_SELFTEST.json` — Stage-4 캠페인·셀프테스트 산출물)이 동거한다 — **이 커밋에 넣지 않는다**(`git add -A` 금지 규약) | `git show --stat` + 명단 diff |

**기대치와 자문 ("다른 가설 아래서도 같은 값을 내는가")**:

| 기대 | 자문 | 증거력 |
|---|---|---|
| 확대 후 HEAD `[STATIC][PASS]` | **낸다** — 확대를 아예 안 넣어도 PASS 다(upper·formula 는 HEAD 무접촉). NC 9·10 + P3′ 이 확대의 실존을 시연해야 증거 | 낮음(단독) |
| NC 9·10 각각 지정 사유 검출 | 못 낸다 — 확대 미구현이면 `injection-9/10-not-applied` 또는 wrong-reason 으로 시끄럽게 죽는다 | 핵심 |
| P3′ 외부 주입 FAIL + HEAD PASS 짝 | 못 낸다 — 무조건-PASS 도 무조건-FAIL 도 짝을 못 만든다 | 핵심 |

### 5-A-4. 철회·분기 (초판 §5 표의 확장 — 제3의 경우의 소급 등재)

| 관측 | 처분 |
|---|---|
| (초판 ①②) | 그대로 유지 |
| 확대된 스팬에서 **실위반**(HEAD 의 line_source 접촉)이 나타남 | 초판 ① 과 동일 — 보고·중단, src 수리 금지, FINDING 기재. [실측] HEAD 무접촉이므로 나타나면 delta 적용 오류 우선 의심 |
| 주입 9·10 의 앵커가 스팬 내 부재/비유일 | `injection-N-not-applied` red 유지 — 조용한 통과 금지. 보고 후 앵커 재선정 |

### 5-A-5. 확대 뒤에도 이 게이트가 못 보는 것 (§I-2 거울상 — 넓힌 만큼 적는다)

1. **이행적 callee 본문** — `population_line_level_number_density` ·
   `build_lte_level_density_cache` · `nlte_tau_line_authority` 등의 본문은 무스캔.
   §2-Q2 사각 ④(술어 본문)와 동류 — **미룸 유지**(별도 단 승격 경로).
2. **upper 의 제4 호출자** `a210_private_line_energy_build`(`:14684`) — A2-10 측정면이라
   이 게이트 밖. opacity 가 스코프에 있으나 현재 line_source_S 무접촉 [실측].
3. **방출률 벌크 이식**(`:21273`/`:21381` line_source_S memcpy) — §2-Q2 표 1' 미룸 그대로.
4. **정규식의 읽기/쓰기 무구별** — 확대 스팬에 미래에 합법 쓰기가 생기면 false red 로
   **시끄럽게** 죽는다(fail-closed 방향 — 허용; 그때 사전등록으로 처분).
5. **CUDA 대응물** — 게이트는 CPU 파일 2개만 잰다(기존 한계, 이번 변경 없음).
6. **비컴파일 주입** — NC·외부 주입 전부 텍스트 수준(NC 1~8 도 동일) — 정적 텍스트
   게이트의 본질. 컴파일 가능한 실위반 재현은 요구하지 않는다.

### 5-A-6. 이 개정이 모르는 것

1. upper 가 미래 리팩터로 opacity 접근을 얻을 확률 — 오늘 경로 없음은 [실측], 미래는
   모른다(확대는 그 미지에 대한 저비용 보험이다).
2. GR-3 착지 구현의 내부 구조가 주입 9·10 의 앵커 선정과 충돌하는지 — Codex 구현에서
   판명(충돌 시 §5-A-4 분기).
3. 초판 P3 저자가 upper 를 지목한 것이 의도(대표 실위반)였는지 실수(스팬 혼동)였는지 —
   문서에 근거가 없어 **모른다**. 처분은 어느 쪽이든 동일하므로 판정에 영향 없음.

---

## 6. GR-4 — `selftest-tau-writer-census` 수리 (§H-5 의 조직화, GR-2 판정 조건부)

### 계약

> 등록부와 판별 정규식이 실제 writer 집합 — 원소 대입·벌크 memcpy·별칭 — 과 다시
> 일치하고, 등록부 밖의 어떤 형태의 tau 쓰기도 이름 있는 사유로 붙잡힌다.

### 기대 변경집합

| 파일 | 변경 |
|---|---|
| `scripts/check_tau_writer_generation.py` | (1) WRITERS 에서 `nlte_update_tau_sobolev` → `nlte_update_tau_sobolev_with_authority` **교체**(추가 아님 — 래퍼를 남기면 `registered writer has no tau writes` 지속, §H-5-1). (2) ASSIGN 확장: 원소 대입의 객체를 별칭 집합 `{opacity, public_opacity}` 로, **memcpy 계열 신설** — dest 표현식에 `tau_sobolev` 를 포함하는 모든 `memcpy(` 를 별도 패턴으로 수집, 신규 사유 `unregistered bulk tau writer (memcpy) at ...` (3) GR-2 판정 명세대로 transplant 류 등록(예상: 두 커밋 함수 스팬 + 스팬 내 preflight 호출 토큰 실존 검사 — **판정문이 정본, 이 예상이 아니라**). (4) 래퍼 `nlte_update_tau_sobolev`(`:19602`) 순수 위임 정확일치 검사. (5) NC 기대-사유 고정(부록 C) + NC 2건 신설(memcpy rogue·별칭 rogue). (6) `function_span` 을 lib 채택으로 교체 |
| `scripts/run_gate_battery.py` · `scripts/gate_registry.json` | preflight 행 추가 · known-red 행 소거 |
| 이 문서 | 집행 기록 |

**접촉 금지**: `src/`(두 memcpy 포함 — **판정이 무엇이든 src 는 불변**), 다른 게이트, 덱.

### 게이트 표 (요지 — 형식은 GR-3 과 동일)

P1 HEAD PASS(`writers=3 ... cuda_writers=0` + transplant 류 카운트) ·
P2 NC 6건 사유 고정(기존 4 + memcpy rogue + 별칭 rogue) ·
P3 외부 사본-트리 주입: 등록 함수 **밖**에 `memcpy(public_opacity->tau_sobolev, x, n)` 손 삽입
→ `unregistered bulk tau writer (memcpy)` 정확 사유 + HEAD PASS 짝 ·
P4 배터리 preflight, known-red 3→2 · P5 변경집합.

★P3 이 이 단의 존재 이유다 — **수리 전 게이트는 이 주입에 green 이었다**(H-2 실증).
같은 주입이 red 로 바뀌는 것이 "초록불의 인증 범위가 실제로 넓어졌다"의 실물 증거다.

### 기대치와 자문

| 기대 | 자문 | 증거력 |
|---|---|---|
| HEAD PASS | **낸다** — 정규식을 오히려 좁힌 가짜 수리도 PASS | 낮음 |
| 외부 memcpy 주입 red (수리 전 green 이던 주입) | 못 낸다 — 정규식이 실제로 넓어져야만 | 핵심 |
| `:19521`·`:19573` 이 `_with_authority` 카운트로 편입 | 부분적으로 낸다 — 스팬만 맞으면 됨. 괄호 순서 검사(require<첫쓰기, mark>끝쓰기)와 결합해야 의미 | 중간 |

### 철회·분기

GR-2 판정=(iii)위반 → 이 단은 두 memcpy 를 known-violation 으로 붉게 두는 **부분폐합**
(§4 분기). 확장 정규식이 예상 밖 쓰기 지점을 새로 찾음 → 각각 GR-2 와 같은 계급의
판정 발주 후 재개(무단 등록 금지 — 등록부는 판정을 요구하는 화이트리스트다, §G-3-1).

---

## 7. GR-5 — `event-measure-check` 수리 (계측 → 판정 → 수리)

유일하게 **판정을 안 받은** 죽은 게이트다(증거 패킷 §6). 그래서 이 단은 세 스텝이다.

**스텝 1 (계측, 운전석)**: grammar-debug 에서 두 명령을 각각 완주시켜 **전 실패 목록**
채집(§1-3 의 [추정] 3건을 실측으로 대체) + 미도달이던 `compare_event_measure_spectra.py
--selftest` 의 상태 실측.

**스텝 2 (판정, fresh Fable)**: 실패 각각을 앵커 노후/실위반으로 분류.
판정문 커밋 1개(`docs/VERDICT_EVENT_MEASURE_GATE_2026-08-XX.md`).
예상 [추정]: CPU-A208 3건 전부 앵커 노후(§1-3 의 grep 근거) — **예상은 판정이 아니다.**

**스텝 3 (수리, 판정대로)**: 기대 변경집합 —

| 파일 | 변경 |
|---|---|
| `scripts/check_event_measure_access.py` | (1) 앵커 `a208_publish_cpu_opacity` → `_impl`. (2) **CPU-A208 주입 표적도 impl 로** (§1-1 의 자매 병 — 안 옮기면 측정면 밖 주입). (3) 래퍼(`:8832`) 순수 위임 정확일치 검사 신설. (4) NC 기대-사유 고정(부록 C). (5) `function_body`·`inject_into_function` 을 lib 채택으로 |
| `scripts/run_gate_battery.py` · `scripts/gate_registry.json` | preflight 행(두 명령 각 1행) · known-red 행 소거 |
| 이 문서 | 집행 기록 |

게이트 표(요지): P1 make 타깃 완주(두 명령 다 — 두 번째 명령 사상 최초의 회귀 실행) ·
P2 NC 4건 사유 고정 · P3 외부 사본-트리 주입(impl 에 `bf->event_chi_bf[0]` 읽기 손 삽입 →
`CPU-A208: direct event-grid indexing bypass` + HEAD PASS 짝) · P4 배터리, known-red 2→1 ·
P5 변경집합.

분기: 스텝 2 가 실위반을 하나라도 확정 → 해당 항목은 GR-3 분기와 동일(보고·중단·별도 단,
src 불변). 두 번째 명령이 FAIL → 분류 후 같은 단에서 픽스처 수리(GR-6 계급) 또는 known-red
잔류(등록부에 사유 기재).

---

## 8. GR-6 — `selftest-a2-10-line-saturation` 픽스처 수리

### 계약

> 게이트 사슬의 positive 픽스처가 생산자 스키마 v1(`target_ion_zero_based` 포함)을
> 다시 싣고, 체커·생산자는 무접촉이다.

### 기대 변경집합

| 파일 | 변경 |
|---|---|
| `tests/a2_10_cmfgen_line_saturation_comparison_selftest.py` | `summary()` payload 에 `"target_ion_zero_based": 3`(픽스처 rows 의 `ion=3` 과 정합 [실측 `:44`]) — payload 를 파생하는 모든 변형(union·consistent·boundary)이 상속하는지 확인. + **NC 1건 신설**: 키 제거 payload → 체커가 정확 사유 `Lumina summary target ion is invalid` 로 거부함을 시연(역사적 결함 그 자체의 재주입 — 수리 전 코드에서 positive 가 죽던 그 자리) |
| `scripts/run_gate_battery.py` · `scripts/gate_registry.json` | preflight 행(6개 스크립트 각 1행) · known-red 행 소거 |
| 이 문서 | 집행 기록 |

**접촉 금지**: `scripts/compare_a210_cmfgen_line_saturation.py`(체커 무접촉 — 요구가 정당함은
생산자 `:348` 과의 정합으로 §1-2 가 보였다) · `scripts/summarize_a210_line_saturation.py` ·
`src/` · 덱.

게이트 표(요지): P1 타깃 6개 스크립트 완주 — 5·6번째(coverage·monitor, **사상 미도달**)의
상태가 처음으로 실측된다 · P2 신설 NC 시연 · P3 배터리, known-red 1→0 · P4 변경집합.

분기: 키 추가 후 **잔여 실패**가 나오면 각각을 분류 — 같은 계급(픽스처 스키마 드리프트)이면
같은 단에서 계속, 체커 결함으로 분류되면 **보고·중단**(체커 수리는 별도 계약).
`check_a210_line_saturation_per_ion_coverage.py:134` 의 fail-open 기본값은 이 단에서 고치지
않는다(픽스처 3파일 동반 갱신이 필요한 별개 계약) — **대장 기재**(§8 열린 질문 아님, 확정 부채).

### 기대치와 자문

| 기대 | 자문 | 증거력 |
|---|---|---|
| positive PASS | **낸다** — 체커의 키 검사를 지워도 PASS. 체커 무접촉 계약(P4 diff)과 결합해야 의미 | 낮음 |
| 키 제거 NC 가 정확 사유로 거부 | 못 낸다 — 체커가 실제로 fail-closed 이고 픽스처가 실제로 키를 실어야만 짝이 성립 | 핵심 |

---

## 9. Codex 가 지켜야 할 계약 (발주서에 이 절 그대로 첨부)

**[개정 1]** GR-1′ 발주에는 §0-A-5-(e) 의 추가 조항이 이 절과 함께 첨부된다.

1. **`src/` 접촉 0줄.** 잣대(scripts/tests/Makefile/등록부)만 고친다. 게이트를 통과시키기
   위해 소스를 고치고 싶어지면 — 그것이 바로 이 캠페인이 금지하는 행위다. 보고하라.
2. **클램프·floor·cap·새 env 노브 금지.** known-red 등록부는 서명 **정확일치** 고정이다 —
   느슨한 매칭(정규식 와일드카드·부분 문자열)으로 "웬만하면 통과"를 만들지 말 것.
3. **이미 있는 자산만**: `PREFLIGHTS`+`run_preflights()`(회귀 편입 기구 — 새 러너·새 배터리
   금지), 게이트 3본체의 구조와 `NEGATIVE-CONTROL` 틀, `injection-N-not-applied` fail-closed
   (보존 — 적용 못 한 주입의 조용한 통과 금지).
4. **정확일치 원칙(§I-2)**: 앵커·리터럴·래퍼 본문·기대 사유 전부 현재 형태에 정확일치로
   고정한다. 알터네이션(`구형 OR 신형`)으로 게이트를 넓히는 것 금지 — 인증 없이 초록
   표면만 커진다(§H-2).
5. **NC 판정식에 OR 금지**: 주입마다 부록 C 의 기대 사유가 실패 목록에 있는지 본다.
   "무언가 실패했으면 검출"은 §E 가 08-08 에 강등시킨 그 결함이다.
6. 덱·`/gpfs` 정본 불변. **계약 1개 = 커밋 1개. `git add -A` 금지** — 신설 파일은 명시적
   `git add`.
7. **변경집합은 각 단의 §가 전부다.** 필요해 보이는 추가 변경은 고치지 말고 보고하라.

---

## 10. 분장 장부 (실제 열은 집행 후 채운다 — 명목을 실제인 양 적는 것 금지)

각 단마다 이 표를 집행 기록에 복제해 채운다.

| 단계 | 규약상 담당 (개정13/14) | **실제** | 위반 |
|---|---|---|---|
| 사전등록 (이 문서) | Fable | Fable | — |
| 발주 | 운전석 | | |
| 코딩 | Codex (clean worktree) | | |
| 코드 검수 | Fable | | |
| 오프라인 게이트·배터리 실행 (grammar-debug) | 운전석 | | |
| 판정 (GR-2·GR-5 스텝2 포함) | Fable | | |
| 판정 감리 | Fable (**fresh 컨텍스트** — 판정과 같은 컨텍스트 금지) | | |
| 감리 반영·대장·커밋 | 운전석 | | |

하네스 제약과 규약이 충돌하면 말없이 해소하지 말고 실제 열에 그대로 적는다
(직전 단 교훈).

---

## 11. 열린 질문 / 이 사전등록이 모르는 것 (추측으로 메우지 않는다)

1. **event-measure 의 전 실패 목록** — [추정] CPU-A208 3건뿐. 실측은 GR-5 스텝 1.
2. **`compare_event_measure_spectra.py --selftest` 의 현재 상태** — 모른다(사상 미도달).
3. **line-saturation 타깃 5·6번째 스크립트(coverage·monitor)의 상태** — 모른다(미도달).
   GR-6 P1 이 처음 잰다.
4. **`a2_07_population_census.py` 의 앵커 건강** — 모른다. 4벌째 `function_body` 가 수동
   레인에 있다(§1-5). 등록부 `manual-lane` 등재까지만 이 캠페인의 몫.
5. **08-08~08-18 작업트리의 일시적 파손 여부** — 복원 불가(§I-6 상속). known-red 서명
   감시는 **앞으로의** 드리프트만 잡는다.
6. **`selftest_stage32_rung1` 출력의 `"beta_defect_negative_control": "... FAIL ..."` 줄**
   — rc=0 인데 FAIL 문자열이 로그 말미에 있다 [실측 orphan log]. NC 의 기대-실패 기록으로
   보이나 확인 안 했다. GR-1 sweep 의 서명 채집이 이런 줄을 실패로 오독하지 않도록
   **rc 를 1차 판별자로** 삼고, 이 줄의 정체는 GR-1 집행 기록에 실측 기재.
7. **패턴 밖 이름의 신설 게이트** — 메타 게이트의 완전성 패턴(`selftest*`·`*-check`·
   `*-census`·`*-gate`)을 벗어난 이름으로 게이트가 신설되면 못 잡는다. 기계로 닫을 방법을
   모른다 — 발주서·검수 규율(신설 게이트는 등록부 등재를 같은 커밋에)로 보완하고 한계로
   남긴다.
8. **GR-2 판정의 결과** — 예상을 적지 않는다. §H-2 가 "런타임 가드는 강하나 정적으로는
   무인증"까지 판독했고, 지위 결정은 판정자의 몫이다.

**[개정 1 추가]**

9. **unwired 27 의 상태 전부** — GR-7 스텝 1 이 첫 실측(§0-A-10-1).
10. **`.gitignore` `*.log` 광역 규칙의 일반 처방** — 미래 픽스처가 같은 함정(발견 C)에
    빠지는 것을 기계로 막을 방법을 모른다. 이번에는 정밀 부정 1행으로 해당 파일만 연다.
11. **어휘 언급 채널(오분류 3호)의 기계 폐쇄** — 문자열/문서 속 이름과 실행 호출의 판별은
    셸 의미론이라 기계로 못 닫는다. collective wiring 의 판독 확인 규정(§0-A-5-b)과
    unwired 의 `lexical_mentions` 허용 목록으로 우회하되, 허용 목록 파일 안의 새 실호출은
    못 본다는 한계를 안다.
12. **fresh clone 완결성의 나머지** — 덱·기타 경로의 추적 여부 전수 미조사(§0-A-10-5).

---

## 부록 A — `gate_registry.json` 스키마 (정본)

**[개정 1]** 스키마는 v2 로 확장됐다(`unwired` 범주·블록 신설 — §0-A-9 가 정본).
아래 v1 원문은 초판의 기록으로 보존한다. pin 실측 기입 규칙(값은 실측, 스키마가
사전등록)은 v2 에도 그대로 적용된다.

```json
{
  "schema": "lumina-gate-registry-v1",
  "entries": [
    {
      "name": "<make 타깃 또는 스크립트 경로>",
      "kind": "make-target | script",
      "commands": [["python3", "scripts/....py", "..."]],
      "category": "battery-core | preflight | collective | milestone |
                   run-dependent | manual-audit | manual-lane | known-red",
      "wiring": "<범주별 배선 주장: PREFLIGHTS 행 이름 / 참조 타깃·.sh 경로 /
                 집합 타깃 이름 / 요구 자원 / 사유>",
      "known_red": {
        "failing_command_index": 0,
        "rc": 0,
        "first_fail_line": "<정확 문자열>",
        "fail_line_count": 0,
        "registered": "YYYY-MM-DD",
        "repair_rung": "GR-N"
      }
    }
  ]
}
```

- `known_red` 는 category=known-red 에서만 허용·필수. 다섯 필드 하나라도 없으면 메타
  게이트가 `known-red-row-incomplete` FAIL.
- pin 값(초기 4행)은 **GR-1 집행 시 grammar-debug 재실행 실측**으로 기입한다.
  참고 예상 [orphan log 실측 첫 줄 + 이 문서 분석]:
  sh-radeq = rc 1·첫 줄 `[SH-RADEQ-0][STATIC][FAIL] missing fail-closed token
  'blocked_line_cells'`·19줄 / tau = 첫 줄 `...unregistered raw tau writer at
  lumina_plasma.c:19521`·[추정 3줄: 19521·19573·no-writes] / event = 첫 줄
  `[E-NE4][FAIL] CPU-A208: missing bf_event_measure_get`·[추정 3줄] / line-sat =
  실패 명령 index 3(0-기준)·`...target ion is invalid`.
  실측이 예상과 달라도 사전등록 위반이 아니다 — pin 필드 스키마가 사전등록이고 값은
  실측이다. 단 **차이는 집행 기록에 기재**한다(예상이 틀렸다는 것 자체가 정보다).

## 부록 B — `gate_source_lib.py` 추출기 명세 (정본; §I-5 사각 3·6 의 수리)

1. **정의-앵커 매칭**: 후보 = `\b<name>\s*\(`. 각 후보에서 여는 괄호부터 **괄호 깊이
   스캔**으로 인자 목록의 닫는 괄호를 찾고(현행의 게으른 `[^;]*?\)` 폐기 — 중첩 호출·
   개행에 취약), 그 뒤 공백 건너 `{` 가 와야 정의로 인정. `if(f(x)){` 류 호출부는 f 의
   닫는 괄호 뒤가 `{` 가 아니므로 원리적으로 배제된다.
2. **유일성**: 정의 매치가 0개면 `function not found: <name>` 예외(현행 보존 — fail-closed),
   **2개 이상이면 `ambiguous function anchor: <name>` 예외 신설** — 첫 매치를 조용히
   선택하는 것 금지.
3. **주석 제거**: 토큰 검색용 본문은 `/* */`·`//` 를 **줄 구조 보존**(공백 치환)으로 제거한
   판을 쓴다. **문자열 리터럴은 보존한다** — reason 토큰(`"[A2-09][BLOCKED]"` 등)이
   문자열에 산다. 이 비대칭은 한계로 기재: 문자열 속 가짜 토큰은 못 거른다.
4. **API**: `find_definition(text, name) -> (start, end)` · `body(text, name) -> str`(주석
   제거판) · `body_raw(text, name) -> str`(원문 — 래퍼 정확일치·행번호 보고용) ·
   `inject_at_head(text, name, stmt) -> str`.
5. selftest 케이스(전부 이름 있는 예외/판별): 정의 1+호출 2 픽스처에서 정의만 선택 ·
   중복 정의 → ambiguous · 부재 → not found · 주석 속 토큰이 body 에서 사라짐 ·
   문자열 속 토큰이 body 에 남음(한계의 문서화 겸).

## 부록 C — 주입 → 기대 사유 고정 표 (NC 판정식이 이 문자열을 요구한다)

### C-1. `check_a209_source_failclosed.py` (수리 후 8건)

| 주입 | 기대 사유 (실패 목록에 이 문자열이 있어야 검출) |
|---|---|
| 1 abort 분기 무력화 | `missing fail-closed token 'if(blocked_line_cells){'` |
| 2 free/return 제거 | `missing fail-closed token 'a209_publication_free(&c);return invalid_eta_cells?5:3;'` |
| 3 line-eta 호출 개명 | `missing fail-closed token 'a209_sobolev_line_eta'` |
| 4 begin 괄호 제거 | `raw tau consumption is not bracketed at both ends` |
| 5 tau=0 극한 오염 | `missing direct-formula token '(tau==0.0)?1.0'` |
| 6 end 괄호 자기복제 | `raw tau consumption is not bracketed at both ends` |
| 7 (재표적) upper 스팬 내 `_by` 호출 대체 | `A2-09 does not use shared NLTE tau authority` |
| 8 LTE 루틴 분열 | `bulk tau and A2-09 do not share LTE line population routine` |

(4·6 은 토큰 부재 사유도 함께 나올 수 있다 — 요구는 **지정 사유의 존재**이지 유일성이
아니다. 단 지정 사유 부재 시 `injection-N-wrong-reason`.)

### C-1a. [개정 2] 확장 — 주입 9·10 (금지-읽기 스캔의 upper·formula 확대, §5-A)

| 주입 | 기대 사유 |
|---|---|
| 9 upper 스팬 내 `opacity->line_source_S` 읽기 삽입(in-memory, 실앵커 치환) | `A2-09 upper-population producer reads forbidden quotient source` |
| 10 formula 스팬 내 동일 삽입 | `A2-09 line-eta formula reads forbidden quotient source` |

(1~8행 사유 문자열 불변. impl 의 기존 사유 `A2-09 production reads forbidden quotient
source` 도 불변 — 세 스팬의 사유가 서로 다른 것이 wrong-reason 판별의 전제다.)

### C-2. `check_tau_writer_generation.py` (수리 후 6건)

| 주입 | 기대 사유 |
|---|---|
| 1 require 제거(compute) | `compute_tau_sobolev: generation is not advanced before first write` |
| 2 mark 제거(_with_authority) | `nlte_update_tau_sobolev_with_authority: generation is not marked after last write` |
| 3 rogue 원소 대입 함수 추가 | `unregistered raw tau writer at lumina_plasma.c:` (행번호는 파일 말미 — 접두 일치) |
| 4 rogue CUDA writer 추가 | `duplicate/unregistered CUDA raw tau writer at lumina_cuda.cu:` (접두 일치) |
| 5 **신설** rogue memcpy 함수 추가 | `unregistered bulk tau writer (memcpy) at lumina_plasma.c:` (접두 일치) |
| 6 **신설** rogue 별칭 대입(`public_opacity->tau_sobolev[0]=`) 함수 추가 | `unregistered raw tau writer at lumina_plasma.c:` (접두 일치) |

(3~6 의 행번호 접두 일치는 완화가 아니다 — 주입이 파일 말미 추가라 행번호가 파일 길이의
함수이기 때문. 정적 검사의 pin 은 전부 정확일치.)

### C-3. `check_event_measure_access.py` (수리 후 4건 — 표적을 impl 로 이동)

| 주입 | 기대 사유 |
|---|---|
| CPU-T03 `bf->chi_bf[0]` 읽기 | `CPU-T03: direct event-grid indexing bypass` |
| CPU-A208 (**impl 에**) `bf->event_chi_bf[0]` 읽기 | `CPU-A208: direct event-grid indexing bypass` |
| GPU-T03 `d_chi_bf[0]` 읽기 | `GPU-T03: direct event-grid indexing bypass` |
| GPU-VPACKET `d_chi_bf[0]` 읽기 | `GPU-VPACKET: direct event-grid indexing bypass` |

---

## 집행 기록 (운전석 실측 — 단별로 §10 분장 장부 복제 포함; 사전 기입 금지)

(GR-1 부터 집행 시 채운다.)
