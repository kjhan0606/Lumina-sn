# 판정 — unwired 27행의 처분 (GR-7 스텝 2, 판정자 Fable)

발주: `docs/RUNG_GATE_REPAIR_LADDER_2026-08-20.md` §0-A-7 (GR-7 스텝 2). HEAD `b999fbd`.
증거: 운전석 계측 패킷 `/tmp/claude-10396/gr7/EVIDENCE.md`(§7 정정 포함, 로그
`/gpfs/kjhan/lumina/gates/gr7_20260820T143904Z/`) + 판정자 독립 실측(§2 — 전부 이 판정자가
HEAD 작업트리·로그에서 직접 판독; 실행 0회, 로그인 노드 규약 준수). 판정 착지 2026-08-21 이른 시각
(발주일 08-20; 파일명은 발주 명명 유지).

```
처분 총계: 배선 26 (preflight 1 · run-dependent 2 · milestone 23) + known-red 전환 1 + 은퇴 0

은퇴 0 은 회피가 아니라 판정이다 — 27행 전원의 피보호 계약이 HEAD 에서 생존함을
행별로 확인했고(§4 표), 계약 소멸을 확정한 판정문은 어느 행에도 존재하지 않는다.
은퇴의 사전등록 요건("지키던 계약의 소멸을 판정문이 확정할 때만")을 충족하는 행이 0개다.

★단 하나의 실결함: selftest_nlte_assemble — 링크 목록이 소스 진화를 못 따라간
GR-0 동형 결함 [실측 §2-4]. known-red 전환 + 수리 단 SH-UW-1 지정.

캠페인 폐합 조건 ②: 이 판정으로 27행 전원의 처분이 확정됐다. 운전석의 기입 커밋
(판정문 + 등록부 disposition 필드, §6 형상)이 착지하는 시점에 조건 ② 충족.
```

---

## 1. 계측 패킷의 정정 2건 (판정 전에 바로잡는다)

**1-1. [반영] 운전석 §7 자기 정정 — nvcc 3건은 한 부류가 아니다.** grammar-debug 는 CPU 전용
노드라 nvcc 부재가 정상이고, syntax 로그인 노드(nvcc 실존, CUDA 13.0)에서 빌드만 재측정한 결과
[실측 운전석, 로그 `step1_cuda_build_syntax.log`]:
`selftest_a2_12_gpu_lifecycle`·`selftest_a2_13_gpu_oracle` = **빌드 rc=0**(실행 미측정),
`selftest_nlte_assemble` = **rc=2 링크 에러**(실결함). 초판 §2 의 "셋 다 돌릴 수 없었다"는 철회됐고
이 판정은 3분화된 사실 위에 선다.

**1-2. [판정자 실측 정정] "is up to date" 는 3건이 아니라 6건이다.** 패킷 §3 은 그림자 "셋"만
`is up to date` 로 적었으나, step1_27.log 전수 판독 결과 [실측] **6건**이다:
`selftest_a2_08_signed_opacity` · `selftest_a2_09_emissivity` · `selftest_a2_10_radeq` ·
`selftest_cmfgen_adiabatic` · `selftest_physics_comparison` · `selftest_a2_16_seed`.
make 가 recipe 를 **한 줄도 실행하지 않았다** — 이 6건의 rc=0 이 인증하는 것은
"산출물 mtime 이 의존물보다 새롭다"뿐이다. 특히 a2_08/09/10 은 recipe 의 driver
(음성대조 주입 포함)가, cmfgen_adiabatic/physics_comparison 은 recipe 의 실행 단계가
**오늘 돌지 않았다**. 이 6건의 오늘 런타임 건강은 **모른다** (§8-1 재계측 요청).

## 2. 판정자 독립 실측 — 운전석이 재지 않은 것 (사전등록 §0-A-7 계측 항목의 잔여분)

### 2-1. 27행 전원의 「빌드만 / 주입·검출까지」 분류 [실측 — Makefile recipe 전수 판독]

| 부류 | 수 | 타깃 | recipe 가 인증 가능한 최대치 |
|---|---|---|---|
| **build+run** (recipe 에 실행 단계 실존) | 12 | `a2_08_signed_opacity`·`a2_09_emissivity`·`a2_10_radeq`(python driver 3) / `det_stage12`·`line_net_rate`·`cmfgen_adiabatic`·`atomic_internal_energy`·`physics_comparison`·`nlte_population_candidate`·`nlte_candidate_adiabatic`·`nlte_candidate_tau`·`a2_10_seed_commit`(`./바이너리` 직행 9) | 검사 실행까지 |
| **python 단독** | 1 | `physics_comparison_regrid` | 검사 실행까지 |
| **build-only** (실행 단계가 recipe 에 없음) | 11 | `a2_12_contract`·`a2_13_15_contract`·`wave32_{ew_rc,ew_io,within_sl_oom,boundary_q,counter_atomic}`·`emiss_ab_insitu`·`a2_03_producer_parity_fixture`·`a2_04_replay_commit`·`a2_16_seed` | **컴파일·링크까지만** |
| **CUDA build-only** | 3 | `nlte_assemble`·`a2_12_gpu_lifecycle`·`a2_13_gpu_oracle` | 컴파일·링크까지만 |

⟹ 오늘 실측으로 **검사 실행 + PASS 문구까지 확인된 것은 27 중 8건뿐**이다 [실측 로그]:
det_stage12 · line_net_rate · atomic_internal_energy · nlte_population_candidate ·
nlte_candidate_adiabatic · nlte_candidate_tau · a2_10_seed_commit · physics_comparison_regrid.
(build+run 12 중 5건은 §1-2 의 up-to-date 로 실행이 생략됐다.)

### 2-2. ★build-only 9건(CPU)의 바이너리는 **아무 데서도 실행되지 않는다** [실측]

`scripts/`·`tests/` 전수 grep — 이 9개 타깃/바이너리를 부르는 파일 0건. recipe 도 빌드에서
끝난다. ⟹ 이 9건의 검사 본문(main 의 주입·검출)은 **한 번도 관측된 실행 이력이 없다**
(적어도 배선 가능한 어떤 경로로도). rc=0 을 이들의 "게이트 건강"으로 읽으면 안 되는 이유의
실물이다. 단독 실행 가능성 [실측 main 시그니처]:

- 즉시 실행형(`main(void)`) 6: `wave32_ew_rc`(env 자체 설정) · `wave32_within_sl_oom` ·
  `wave32_boundary_q` · `wave32_counter_atomic` · `a2_03_producer_parity_fixture` · `a2_16_seed`
- 인자 필요 3: `wave32_ew_io`(argv[1]=산출 경로) · `emiss_ab_insitu`(argv[1]=OUTPUT_BASE) ·
  `a2_04_replay_commit`(INPUT OUTPUT 2인자 — 기존 driver `scripts/a2_04_l0_replay.py:288-297` 가
  정확히 그 호출 형식으로 replay.in 을 자가 생성해 부른다 [실측]; 단 그 driver 는
  `validation/chain_replay_parity59/` 의존을 가진다 — 추적 여부 미조사)

### 2-3. ★배터리-그림자 5 의 driver 단계 대 배터리 Z 러너 커버리지 대조표 [실측 — 사전등록이 GR-7 계측 항목으로 지정했으나 운전석이 재지 않은 것]

| 타깃 | make recipe | 배터리 Z 러너 (`run_zinert_selftest.py:128-132`) | 차이 |
|---|---|---|---|
| `a2_08_signed_opacity` | 빌드 + `run_a2_08_selftest.py`: baseline + **POISONS 8건 env 주입 음성대조**(marker in stderr + 기대 rc) + 산출물 6종(validation/a2_08) | 바이너리 **맨손 1회**(baseline rc 만) | N1–N8 음성대조·산출물 신선화 **미재현** |
| `a2_09_emissivity` | 빌드 + `run_a2_09_selftest.py`: baseline + POISONS 8건 + 정적 census + P4 appendix-A 대조 + 산출물 | 동상 | 동일 부류 미재현 |
| `a2_10_radeq` | 빌드 + `run_a2_10_selftest.py`: baseline + 마커 음성대조 8건 + radeq census + 산출물 | 동상 | 동일 부류 미재현 |
| `a2_12_contract` | **빌드만** | 자체 명세 빌드 + 맨손 실행 | **배터리 ⊋ make** |
| `a2_13_15_contract` | 빌드만 | 동상 | 배터리 ⊋ make |

⟹ **배터리 Z 러너는 그림자 3(a2_08/09/10)의 driver 검사를 재현하지 않는다.** 그리고
`run_a2_{08,09,10}_selftest.py` 를 부르는 곳은 저장소 전체에서 **Makefile recipe 뿐이다**
[실측 grep] ⟹ A2-08/09/10 발행 계약의 **주입 음성대조(각 8건)는 현재 unwired 타깃을 통해서만
실행 가능**하다. 이것이 그림자 3의 은퇴를 막고 배선을 요구하는 실측 근거다.

### 2-4. ★`selftest_nlte_assemble` 링크 결함의 국소화 [실측]

운전석 §7 의 미정의 심볼 전수(`nlte_population_candidate_begin` 외 7+)의 정의처를 판독:
**전부 `src/nlte_population_candidate.c`** (`:16`·`:59`·`:72` 외). 그리고 `Makefile:121-122`
(정의행+recipe)에는 `nlte_population_candidate.c` 토큰도 `$(NLTE_CANDIDATE_SRC)` 토큰도
**0건**이다(변수 정의 [실측] `Makefile:39` `NLTE_CANDIDATE_SRC = src/nlte_population_candidate.c`).
⟹ 소스(`src/lumina_plasma.c`)가 후보 API 를 얻었는데 이 타깃의 링크 목록만 안 따라간
**GR-0(Z-a2-09)·발견 A 와 같은 계급**의 빌드 명세 노후다. 1차 수리 후보 =
`$(NLTE_CANDIDATE_SRC)` 1항 추가 [실측 근거: 보고된 미정의 심볼 전부의 정의처가 그 한 파일].
추가 항목 필요 여부(예: 연쇄 미정의)는 수리 단이 실측한다 — 여기서 단정하지 않는다.

### 2-5. 처분 기입의 기계 정합 [실측 — `scripts/check_gate_registry.py`]

- unwired 블록 내부는 닫힌 집합(`UNWIRED_FIELDS`, `:47`)이라 새 키를 넣으면
  `unwired-row-incomplete` 로 죽는다.
- 반면 **entry 최상위**는 필수 키 존재 검사뿐(`:353-363` — `required.issubset`)이라
  추가 키를 거부하지 않는다.
⟹ 처분 기입은 **entry 최상위 `disposition` 필드**로 한다(§6 형상 — 현행 체커 무변경으로
기입 가능). 무검증 필드의 부패 채널은 확정 부채로 §7-3 에 기재.

## 3. 처분의 판정 규칙 (행별 표 앞에 명문화)

1. **known-red 는 "죽어 있음이 관측된" 행에만** 성립한다. 오늘 rc=0 인 24행은 어느 것도
   known-red 대상이 아니다 — known-red 는 실패 서명 pin 을 요구하는데 pin 할 실패가 없다.
2. **"돌릴 수 없었다/안 돌렸다"(CUDA 실행·up-to-date 6·build-only 9)는 known-red 가 아니다.**
   건강 미지를 적자로 등재하면 등록부가 거짓을 인증한다. 미지의 처분은 배선(+배선 전
   1회 실측 전제)이다.
3. **은퇴 요건**: 그 행이 지키던 계약의 소멸을 확정한 판정문 실존. §4 표의 "피보호 계약"
   열이 행별 생존 확인이고, 소멸 판정문은 0건이다.
4. **배선처의 비용 원칙**: preflight(매 배터리)는 python 단독·경량만
   (PREFLIGHTS 행 형식이 python 스크립트 한정 [실측 `run_preflights()`] — E11 의 R3 처분과
   같은 규율로 형식을 이 판정 때문에 개조하지 않는다). C 빌드+실행형은 **milestone**
   (`selftest-registry-milestone` 집합, 회수 마디마다 기계 실행 — GR-1′ P5 실측으로 실행이
   증명된 executor). 배터리 상시 Build 추가는 전량-티어 비용을 매 판에 물리는 방향이라
   기본 배제 — 3층 인수 프로토콜(parity 매 스텝 / 전량은 마디만)과 정합.
5. transplant 계약(GR-2 (ii))의 실행 증거를 담은 행은 **은퇴 금지** — GR-2 판정문 §7 의
   구체적 금지를 그대로 상속한다.

## 4. 27행 전원의 처분 표

범례: 상태 = 오늘 step-1 실측이 인증한 것. 처분 = ①배선(범주→배선처) ②known-red(수리 단) ③은퇴.
집행 단(SH-UW-1~4)은 §5. 전 행 [실측] 기반 — [추정] 은 개별 표기.

| # | 타깃 | 오늘 실측 상태 | 피보호 계약 (생존 근거) | 처분 | 근거 |
|---|---|---|---|---|---|
| 1 | `selftest_nlte_assemble` | **링크 실패**(syntax 재측정 rc=2) [실측 §1-1] | GPU bound-bound 조립 자가검사 — full-NLTE GPU 생산 경로 현역 | **② known-red 전환, 수리 단 SH-UW-1** | 유일한 관측된 실결함(§2-4). 계약 생존 + 게이트 사망 = known-red 의 정의 그 자체. 환경 탓 처분은 실결함 은폐(운전석 §7-4 동의) |
| 2 | `selftest_a2_12_contract` | 빌드 rc=0 (fresh) | GPU RF 계약의 CPU 검사 — **배터리 Z 가 같은 소스를 매판 빌드+실행** [실측] | ① 배선 → milestone (SH-UW-3) | 배터리 ⊋ make(§2-3)라 고유 표면은 "make-정본 빌드 가능성"뿐이나, GR-8 이 이 recipe 를 정본 쌍으로 쓰므로 은퇴 불가·비용은 마디당 컴파일 1회로 미미 |
| 3 | `selftest_a2_13_15_contract` | 빌드 rc=0 (fresh) | GPU physics 계약의 CPU 검사 — 배터리 Z 매판 실행 | ① 배선 → milestone (SH-UW-3) | #2 와 동일 논리 |
| 4 | `selftest_a2_12_gpu_lifecycle` | **빌드 rc=0**(syntax)·실행 미측정 | GPU RF mirror/lifecycle 실물 검사(CUDA) | ① 배선 → **run-dependent** (SH-UW-2) | `selftest_mc_evt_access` 선례(category=run-dependent, wiring="GPU and NVCC required") [실측 등록부]. 실행 건강은 GPU 티어 계측 항목(§8-3) |
| 5 | `selftest_a2_13_gpu_oracle` | 빌드 rc=0(syntax)·실행 미측정 | GPU 커널 oracle 대조(CUDA) | ① 배선 → run-dependent (SH-UW-2) | #4 동일 |
| 6 | `selftest_wave32_ew_rc` | 빌드 rc=0 · **본문 실행 이력 0** [실측 §2-2] | element-wide rc 배관·env 게이트 검사 — `lumina_element_wide.c` 생산 현역 | ① 배선 → milestone, **recipe 에 실행 단계 부가 필수** (SH-UW-4) | 실행 없는 배선은 rc=0 오독의 재생산. main(void) 즉시 실행형 |
| 7 | `selftest_wave32_ew_io` | 빌드 rc=0 · 본문 실행 이력 0 | EW 산출물 IO fail-closed(/dev/full 검사) | ① 배선 → milestone + 실행 단계(scratch 인자) (SH-UW-4) | argv[1] 필요 [실측] — 실행 단계는 mktemp 계열 scratch 로 (repo 오염 금지) |
| 8 | `selftest_emiss_ab_insitu` | 빌드 rc=0 · 본문 실행 이력 0 | E5 in-situ A/B/B2 조립 + seeded n_u 오염 검출 | ① 배선 → milestone + 실행 단계(OUTPUT_BASE 인자) (SH-UW-4) | #7 동일 형식 |
| 9 | `selftest_wave32_within_sl_oom` | 빌드 rc=0 · 본문 실행 이력 0 | `--wrap=malloc` OOM fail-closed | ① 배선 → milestone + 실행 단계 (SH-UW-4) | main(void) |
| 10 | `selftest_wave32_boundary_q` | 빌드 rc=0 · 본문 실행 이력 0 | 경계 q-투영·flux 감사 | ① 배선 → milestone + 실행 단계 (SH-UW-4) | main(void) |
| 11 | `selftest_wave32_counter_atomic` | 빌드 rc=0 · 본문 실행 이력 0 | OMP 카운터 원자성(-fopenmp) | ① 배선 → milestone + 실행 단계 (SH-UW-4) | main(void) |
| 12 | `selftest_a2_03_producer_parity_fixture` | 빌드 rc=0 · 본문 실행 이력 0 | A2-03 생산자(transport) parity — RF 계약 현역 | ① 배선 → milestone + 실행 단계 (SH-UW-4) | main(void). 배선된 `selftest_a2_03_radiation_field`(collective)와 표면이 다름(소비자 계약 vs 생산자 parity) — 중복 아님 |
| 13 | `selftest_a2_04_replay_commit` | 빌드 rc=0 · 본문 실행 이력 0 | A2-04 replay 커밋 경로 | ① 배선 → milestone + 실행 단계(driver `a2_04_l0_replay.py` 경유 후보) (SH-UW-4) | 2인자 필요 [실측]; driver 가 호출 형식 정확 일치 [실측 :297]. driver 의 validation/ 의존 추적 여부는 집행 단이 실측 |
| 14 | `selftest_a2_08_signed_opacity` | **up to date — recipe 미실행** [실측 §1-2] | A2-08 발행 계약 **주입 음성대조 8건 + 산출물 신선화 — 이 recipe 가 유일 실행 경로** [실측 §2-3] | ① 배선 → milestone (SH-UW-3; 전제=§8-1 강제 재실행 실측) | 배터리는 맨손 baseline 만 재현 — 은퇴하면 8건 음성대조가 실행 불능이 된다. 산출물(validation/) 재기록 부작용은 마디 워크플로와 정합(매 배터리 부적합 사유이기도 함) |
| 15 | `selftest_a2_09_emissivity` | up to date — recipe 미실행 | A2-09 음성대조 8건 + census + appendix-A — 유일 실행 경로 | ① 배선 → milestone (SH-UW-3; §8-1 전제) | #14 동일 |
| 16 | `selftest_a2_10_radeq` | up to date — recipe 미실행 | A2-10 radeq 음성대조 8건 + census — 유일 실행 경로 | ① 배선 → milestone (SH-UW-3; §8-1 전제) | #14 동일 |
| 17 | `selftest_det_stage12` | **실행 PASS**(NL1..NL5 positive=6) [실측] | DET stage1/2 radeq 발행 결정론 — K36 폐합 직후의 회귀 잣대 | ① 배선 → milestone (SH-UW-3) | 계약 최신·실행 검증 완료·recipe 자체가 build+run 완결형 |
| 18 | `selftest_line_net_rate` | 실행 PASS | line net-rate 수치 계약(K36 계열) | ① 배선 → milestone (SH-UW-3) | #17 동일 |
| 19 | `selftest_cmfgen_adiabatic` | up to date — recipe 미실행 | CMFGEN 단열항 모델 — A2-10 L6 `RADEQ_INCOMPLETE_ADIABATIC` 해소 방향의 잣대 | ① 배선 → milestone (SH-UW-3; §8-1 전제) | 계약이 현행 BLOCKED 해소의 전제 — 소멸은커녕 임계 |
| 20 | `selftest_atomic_internal_energy` | 실행 PASS | 원자 내부에너지 계약(K36 계열) | ① 배선 → milestone (SH-UW-3) | #17 동일 |
| 21 | `selftest_nlte_population_candidate` | 실행 PASS(shells=2 private_arrays=10) | 후보 생애주기·사적 배열 소유권 — **transplant 계약 인접 음성대조** | ① 배선 → milestone (SH-UW-3) | ★GR-2 §J-2 금지 상속: 은퇴 불가. 오늘 fresh 빌드+PASS 실측 |
| 22 | `selftest_nlte_candidate_adiabatic` | 실행 PASS | 후보 단열 준비의 바이트 보존 대조 | ① 배선 → milestone (SH-UW-3) | #21 동일 |
| 23 | `selftest_nlte_candidate_tau` | 실행 PASS(signed_tau=POSITIVE+NEGATIVE…) | ★**transplant bundle 커밋의 유일한 실행 음성대조**(fail-closed·바이트 보존·세대 착지) [GR-2 §7 실측] | ① 배선 → milestone (SH-UW-3, **우선 대상**) | GR-2 금지의 본체. 매 배터리 정적 pin(GR-4 T1–T7, TAU_WRITER_CENSUS preflight) + 마디 실행 NC 의 이중 커버리지. 한계 명기: tau-괄호·별칭 조건 표적 주입은 없음(계약 일부만 커버 — GR-4 T6/T7 의 몫) |
| 24 | `selftest_a2_10_seed_commit` | 실행 PASS(wrong_generation=BLOCKED) | ★transplant **seed 커밋의 유일한 실행 음성대조 3건**(공적 세대·provenance·Te 바이트) [GR-2 §7] | ① 배선 → milestone (SH-UW-3, 우선 대상) | #23 동일 |
| 25 | `selftest_physics_comparison` | up to date — recipe 미실행 | 물리 스냅샷 대조 하니스(C) — K36 판정 인프라 | ① 배선 → milestone (SH-UW-3; §8-1 전제) | 계약 현역 |
| 26 | `selftest_physics_comparison_regrid` | **실행 PASS**(python) | 스냅샷 regrid 대조 lane | ① 배선 → **preflight** (SH-UW-3) — PREFLIGHTS 행 `("PHYSICS_COMPARISON_REGRID", "tests/physics_comparison_regrid_selftest.py", ())` | 유일하게 preflight 행 형식에 그대로 맞는 행(python 단독·인자 0) [실측]. 기존 A2-10 계열 preflight 20여 행과 동종·동비용 계급 |
| 27 | `selftest_a2_16_seed` | up to date — **build-only recipe** | A2-16 seed capability 상태기계 + N16-1..5 — `seed_capability.c` 전역 링크 현역 | ① 배선 → milestone + 실행 단계 부가 (SH-UW-4) | 검사 본문 실행 이력 0 [실측 §2-2]. main(void) |

### 4-1. 은퇴 0건의 양방향 검토 (배선 몰이가 아님의 증명)

은퇴를 적극 검토한 행과 기각 사유:

- **#2·#3 (배터리 ⊋ make 인 그림자 2)**: 가장 강한 은퇴 후보였다. 기각 이유 둘 다 실측:
  (a) 은퇴 요건 미충족 — 계약은 생존하고 배터리가 지킨다는 사실은 "계약 소멸"이 아니다;
  (b) GR-8 의 build-spec-drift 검사가 정확히 이 make recipe 를 배터리 명세의 대조 정본으로
  쓴다(사전등록 §0-A-7 GR-8 기전) — 은퇴는 그 쌍을 unpaired 로 만들어 GR-0 계급 사고의
  상설 감시를 그 두 lane 에서 제거한다.
- **#1 (nlte_assemble)**: "죽어 있으니 은퇴" 는 처분의 편의다. 계약(GPU 조립 자가검사)은
  생산 GPU 경로와 함께 생존하고, 결함은 1항 링크 목록 노후로 국소화됐다(§2-4) — 수리 비용이
  은퇴 심사 비용보다 작다.
- **#12 (a2_03_producer_parity_fixture)**: 배선된 a2_03 계열과의 중복 여부를 검토 — 소스가
  다르고(transport 생산자 링크) 표면이 다르다 [실측 recipe]. 중복 아님.
- 나머지 행은 은퇴 논거 자체가 성립하지 않는다(피보호 계약 열 — 전부 HEAD 현역 소스의 계약).

역방향(배선 몰이) 검토: 배선 26 중 **매 배터리 비용을 새로 무는 행은 1건뿐**(#26 preflight,
python 경량). 나머지 25 는 마디(milestone)·등재(run-dependent)로 배터리 상시 비용 0 —
"배터리를 무겁게 하는" 방향의 몰이가 아니다. milestone 이 7→30 으로 커지는 비용은 마디당
1회성이며, 그 wall-time 은 미실측이므로 §8-4 로 계측을 강제한다.

## 5. 집행 단 (캠페인 밖 후속 — 사전등록 §0-A-7 "처분의 집행은 캠페인 밖")

트리-변조 상시 1개 규약 하에 순차. 명명은 SH- 접두(게이트 인프라, CMFGEN 대응물 없음).

| 단 | 계약 (1줄) | 내용 | 순위 |
|---|---|---|---|
| **SH-UW-1** | `selftest_nlte_assemble` 의 링크 명세가 소스 정본과 재정합되고, 그때까지 known-red 로 관측된다 | ① known-red 행 등재(§5-1 기계 한계 명기) ② `Makefile:121-122` 에 `$(NLTE_CANDIDATE_SRC)` 추가(부족 시 실측 순증 — 기대 변경집합은 링크 목록 한정, `tests/`·`src/` 무접촉) ③ 게이트 = syntax 빌드 rc=0 [규약: 로그인 노드 빌드 허용] ④ 행 재배속 run-dependent + known-red 소거. **수리가 같은 단에서 즉시 착지하면 known-red 경유 생략 가능**(행 직행 재배속 — 등재·소거가 같은 커밋이 되는 연극을 피한다) | **1 (실결함)** |
| **SH-UW-2** | CUDA 2행이 run-dependent 로 정직 등재된다 | `a2_12_gpu_lifecycle`·`a2_13_gpu_oracle` → category=run-dependent, wiring="GPU and NVCC required"(mc_evt_access 선례 문안). 실행 계측은 §8-3 (등재의 전제 아님 — 선례와 동일) | 4 (등재만) |
| **SH-UW-3** | 실행형 15행(그림자 5 + build+run 9 + regrid 1)이 실집행 배선된다 | regrid → PREFLIGHTS 1행 + 나머지 14 → `selftest-registry-milestone` 의존 편입(집합 하위 구조는 집행 재량, 단 milestone 진입점에서 도달 필수). **전제**: up-to-date 5건(#14·15·16·19·25)은 §8-1 강제 재실행 실측 선행 — red 발견 시 해당 행은 milestone 대신 known-red 등재(수리 단 신설)로 갈음. 우선 대상 = #23·#24(transplant 실행 증거) | 3 |
| **SH-UW-4** | build-only 9행(wave32 5·emiss_ab_insitu·a2_03_parity·a2_04_replay·a2_16)의 recipe 가 실행 단계를 얻고, 1회 건강 실측 후 milestone 배선된다 | recipe 에 실행 단계 부가(§4 표의 인자 요건; 산출은 mktemp scratch — 4번째 중복 빌드 명세를 만들지 않기 위해 별도 driver 스크립트 신설 대신 **recipe 내 실행**을 정본으로) + §8-2 실측 → green 이면 milestone 편입, red 면 known-red 등재(수리 단 신설) | **2 (미관측 검사 9건의 첫 실측 — 정보가치 최대)** |

배선 착지 시 등록부 자기 만료(`unwired-now-referenced`)가 발화해 행 재배속을 기계가
강제한다 [실측 체커] — 배선 커밋과 재배속 커밋이 어긋날 수 없다.

### 5-1. SH-UW-1 known-red 행의 기계 한계 (정직 문안 의무)

known-red sweep 은 grammar-debug(nvcc 부재 정상)에서 돈다 ⟹ pin 서명은 **nvcc 부재 서명**
(rc=2, `Error 127` 계열)이지 링크 결함이 아니며, **수리가 착지해도 sweep 은 계속 같은 서명으로
죽는다**(`known-red-unexpected-pass` 자기 만료가 이 행에서는 발화 불능). 행 note 에
"실결함=링크 에러(syntax 실측, §7 로그) / sweep 서명=환경 제약으로 결함 자체를 관측하지 못함 /
소거는 SH-UW-1 게이트(syntax 빌드 rc=0)가 직접 담당"을 명기하라. 이 한계를 안 적으면
등록부가 인증 범위를 과장한다 — 이 캠페인의 병명 그대로.

## 6. 처분 기입의 필드 형상 (정본 — 사전등록 §0-A-9 가 이 판정문에 위임)

entry **최상위**에 기입한다(§2-5: unwired 블록 내부는 체커가 닫혀 있고 최상위는 열려 있다 —
현행 체커 무변경으로 기입 가능):

```json
"disposition": {
  "verdict": "wire" | "known-red",
  "verdict_doc": "docs/VERDICT_UNWIRED_GATES_2026-08-20.md",
  "target_category": "preflight" | "milestone" | "run-dependent" | "known-red",
  "target_wiring": "<배선처 — 예: selftest-registry-milestone / PHYSICS_COMPARISON_REGRID / GPU and NVCC required / SH-UW-1>",
  "execution_rung": "SH-UW-1" | "SH-UW-2" | "SH-UW-3" | "SH-UW-4",
  "preconditions": "<선택 — 예: forced-rerun(§8-1) / run-step-required(§8-2)>"
}
```

`retire` 값은 형상에 두지 않는다 — 이 판정에 은퇴가 0건이고, 미래의 은퇴는 새 판정문을
요구하므로 그때 그 판정문이 형상을 확장한다. 기입 커밋 = 이 판정문 + 27행 disposition
필드 = **GR-7 의 커밋 1**(운전석).

## 7. 이 판정이 모른다고 적는 것

1. **up-to-date 6건의 오늘 런타임 건강** — recipe 미실행(§1-2). §8-1 이 첫 실측이 된다.
   (작업트리의 validation/a2_09·a2_10 산출물 수정 흔적은 최근 누군가 driver 를 돌렸음을
   시사하나 [실측 git status], 어느 형상에서였는지 모른다 — 건강 증거로 쓰지 않는다.)
2. **build-only 9건의 검사 본문 건강** — 실행 이력 0(§2-2).
3. **CUDA 2건의 실행 건강** — GPU 티어 미계측.
4. **milestone 30개 체제의 wall-time** — 미실측. 감내 불가로 실측되면 그것은 새 사실이고,
   처분 개정은 이 판정문의 개정 이력으로만 한다(집행 단의 재량 축소 금지).
5. **`nlte_assemble` 결함의 태생 커밋** — 미측정(§8-5).
6. **`a2_04_l0_replay.py` 의 `validation/chain_replay_parity59` 의존이 추적되는지** —
   fresh clone 완결성 미조사(사전등록 §0-A-10-5 와 같은 계급).
7. **루트의 `selftest_nlte_assemble.c` 가 `tests/` 밖에 있는 사유** — 관례 이탈로 보이나
   [추정] 판정 사안 아님, 기재만.

### 7-3. 확정 부채 (대장 기재 — 조용히)

- 최상위 `disposition` 필드는 **무검증**이다(§2-5). 썩는 채널 — R4 계열과 동형. 후속 저비용
  검증 1건(disposition 실존·형상 검사) 후보로 기재하되 이 판정의 요구사항은 아니다.
- 그림자 3(a2_08/09/10)의 milestone 실행은 validation/ 산출물을 재기록한다 — 커밋 규약
  (`git add -A` 금지·산출물 거처)과의 교차는 집행 단이 명문화할 것.

## 8. 운전석이 추가로 재야 할 것 (집행 전제 계측 — 판정은 위 표로 확정, 아래는 집행의 입력)

1. **up-to-date 6건 강제 재실행** [SH-UW-3·4 전제]: 산출물 제거 후(또는 `make -B`)
   `selftest_a2_08_signed_opacity`·`a2_09_emissivity`·`a2_10_radeq`·`cmfgen_adiabatic`·
   `physics_comparison`·`a2_16_seed`(빌드만) — rc + PASS 문구 + (a2_0x) N1–N8 카운트 채집.
   validation/ diff 는 관찰만 하고 처분은 집행 단 문서로.
2. **build-only 9건 바이너리 1회 실행** [SH-UW-4 전제]: 즉시 실행형 6 은 맨손,
   `wave32_ew_io`·`emiss_ab_insitu` 는 scratch 인자, `a2_04_replay_commit` 은
   `a2_04_l0_replay.py` 경유(의존 실측 포함). grammar-debug.
3. **CUDA 2건 GPU 티어 실행**: slurm job-per-run(h200→h100) 1회 — run-dependent 등재 후
   첫 건강 실측. 급하지 않다(등재가 실행을 전제하지 않음 — mc_evt_access 선례).
4. **milestone 배선 후 타깃별 wall-time**: `selftest-registry-milestone` 완주 시간 —
   §7-4 의 미지를 닫는다.
5. **`nlte_assemble` 태생**: `git log -S 'nlte_population_candidate_begin' -- src/lumina_plasma.c`
   와 `Makefile:121` 이력 대조 — known-red 행 `registered` 근거 + "며칠 눈멀었나" 확정.

## 9. 캠페인 폐합 조건 ② 의 판정

사전등록 §0-A-8: ② = "unwired 27행 전원에 GR-7 판정의 처분 기입".

- 처분 판정: **이 문서로 27/27 완료** (§4 표 — 배선 26 / known-red 1 / 은퇴 0).
- 기입: 운전석의 GR-7 커밋(판정문 + §6 형상 기입)이 착지하면 **조건 ② 충족**. 기입은
  현행 체커에서 기계 정합함을 실측했다(§2-5).
- 처분의 **집행**(SH-UW-1~4)은 사전등록 명문대로 폐합 조건이 아니다 — 단 집행 전까지
  이 27행의 게이트 표면은 §7 의 미지를 안은 채이며, 그 사실은 이 판정문이 가리고 있지 않다.

— 판정자 Fable, 2026-08-21 (발주 2026-08-20). 근거: `/tmp/claude-10396/gr7/EVIDENCE.md`(§7 포함) ·
`/gpfs/kjhan/lumina/gates/gr7_20260820T143904Z/step1_27.log`·`step1_cuda_build_syntax.log` ·
`Makefile` · `scripts/gate_registry.json` · `scripts/check_gate_registry.py` ·
`scripts/run_gate_battery.py` · `scripts/run_zinert_selftest.py` ·
`scripts/run_a2_{08,09,10}_selftest.py` · `scripts/a2_04_l0_replay.py` · `tests/*.c` main 시그니처 ·
`docs/VERDICT_TAU_BULK_TRANSPLANT_2026-08-20.md` §7 ·
`docs/GATE_RECOVERY_INVENTORY_2026-08-18.md` §G·§J-2 ·
`docs/RUNG_GATE_REPAIR_LADDER_2026-08-20.md` §0-A-5(c)·§0-A-7·§0-A-8·§0-A-9 · `docs/RUNG_NAMING.md`.
