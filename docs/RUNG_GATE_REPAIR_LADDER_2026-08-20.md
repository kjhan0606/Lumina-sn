# 단 사전등록 — SH-GATEREPAIR 사다리: 회귀 등록부 신설 + 죽은 잣대 4개의 수리 (2026-08-20)

저자: Fable(사전등록, 분담 개정14) · 발단: `docs/GATE_RECOVERY_INVENTORY_2026-08-18.md` §F·§G·§H·§I
+ 운전석 증거 패킷(`/tmp/claude-10396/gaterepair/EVIDENCE.md`).
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

**순서**: GR-1 → (GR-2 발주, 트리 무변조라 GR-3 코딩과 병행 가능) → GR-3 → GR-4 → GR-5 → GR-6.
트리-변조 태스크는 상시 1개(규약). GR-3 을 수리 선두에 두는 근거: 수리 명세가 §I-7 로
이미 확정돼 있고, 이 게이트가 지키는 계약(선방출 직접형·tau 세대 괄호)이 **진행 중인
Stage-4 캠페인의 측정면**을 지킨다. GR-6 은 자명하고 독립이라 말미.

**이 캠페인에 넣지 않는 것(§5-5 승격 경로의 답)**: §2-Q5 에 근거와 함께.

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

---

## 부록 A — `gate_registry.json` 스키마 (정본)

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
