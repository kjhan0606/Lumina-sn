# 미실행 게이트 전수 정리 — 2026-08-18

user 지시 "미실행 게이트 회수해" / "나머지도 정리해".
08-08 감사는 **세 단**(SH-GAMMA·DET-R6·SH-GRID)만 강등했으나, 전 단 문서를 훑은 결과
**MC-EVT(E단)에도 미실행 게이트가 있고 그 판정은 `BLOCKED`** 였다. 아래가 전수 목록이다.

## A. 오늘 회수 완료 — 5건

| 게이트 | 단 | 증거 |
|---|---|---|
| **B-6** 값 이동 장부 | SH-GRID | `validation/sh_grid_b6/` — T1 이동은 FUV 집중·최대 2.6e-4·생산 T_e 에서 4.5e-5 ⟹ 재빈닝 오차 상한 ≈3e-4. T2(sigma bake 전후)는 dlog·상단 **bit-identical**, 이동 ≤1.3e-15 |
| **NC2** 구세대 epoch 거부 | SH-GAMMA | `GAMMA_PUBLICATION_STALE_EPOCH` |
| **NC4** 이중발행 차단 | SH-GAMMA | rc=4 `GAMMA_DOUBLE_PUBLISH` + 배열 byte 보존 + 양성대조(다른 epoch=정당 재발행) |
| **N6-2** q-set 해시 한 글자 | DET-R6 | `QHASH_MISMATCH` — 근접 오탐도 거부 |
| **N6-3** 창 밖 선 VALID 위장 | DET-R6 | `NEGATIVE_OR_NONFINITE_JBAR cell=1 validity=1 value=-1`, `rejected_at=commit` |

증거: `validation/gate_recovery/`, `validation/sh_grid_b6/`.

## B. ★오늘 해소된 차단 — MC-EVT NE3 의 전제

08-08 에 Fable 이 MC-EVT 를 `BLOCKED` 로 판정한 사유 셋 중 하나가
"저주파 BF edge census 미완" 이었다. 그 census 는 **707개 활성 edge** 를 찾아
`action=REOPEN_SH_GRID` 를 냈고(최저 witness = Ca II level 61, 5.84852771e13 Hz),
그것이 SH-GRID 재개방 → sigma fresh bake 로 이어졌다.

**2026-08-18 재측정 결과 (오프라인, `validation/evt/BF_EDGE_CENSUS_2026-08-18.log`)**:

```
sigma_grid=[5.84127859e+13, 4.03625815e+16]  levels=24542
positive=24542  above=24542  below_or_at_all=0
[PASS] active_below_or_at_nu_min=0  oog_bf_exact_zero_contract=ELIGIBLE  rc=0
```

**707 → 0.** 자가검사(음성 대조)도 PASS. 새 하한 5.84127859e13 이 최저 활성 edge
5.84852771e13 보다 아래이므로 707개 전부 격자 내부가 됐다.
⟹ `docs/SH_GRID_REOPEN_CONTRACT_2026-08-08.md` §4 가 요구한 "재-census" 조건 충족,
격자 밖 BF 를 exact zero 로 분류할 **자격이 생겼다**(ELIGIBLE).
⟹ **다른 목적의 작업이 열흘 묵은 차단을 풀었다.**

단 재개방 계약은 "CMFGEN 동종 처리 대조" 도 요구한다
(`[5.84127859196e13,1.5e14)` 대역의 level 별 BF rate·emissivity 가 CMFGEN 과 닫힐 것).
`docs/CURRENT_PLAN.md` 의 "저주파 707 level×90 depth 재검증"(global rate 6.962e-5 ·
ion rate 1.573e-4 · weighted-L1 2.455e-4)이 이에 해당하는 것으로 보이나,
**두 문서가 서로를 인용하지 않아 계약 충족 선언은 보류**한다(연결 확인 필요).

## C. ★신규 발견 — 재개방 계약의 사전등록 범위 이탈

`SH_GRID_REOPEN_CONTRACT` 사전등록 vs 실제 구현:

| 항목 | 사전등록 | 실제 | 판정 |
|---|---|---|---|
| BF `dlog` | 0.00529831736655 보존 | 0.0052983173665480362 | **일치**(상대차 −3.7e-13 = 사전등록값의 소수 절단) |
| 하한 | 5.84127859196e13 | 5.8412785919616062e13 | **일치** |
| 상한 | **3.0e16 보존** | **4.0362581455823112e16** | **이탈** |
| bin 수 | **1178** | **1234** | **이탈**(+56 빈, 청색 끝) |

사유는 Si V 바닥 threshold 를 격자 안에 넣기 위한 것으로 보인다
(`docs/HANDOVER_2026-08-18.md` §3.1 "Si V ground threshold 가 새 grid 안에 들어오고
실제 CMFGEN row 가 등록된다"). 물리적으로 정당해 보이나 **재개방 계약의 기대 변경집합에
개정으로 등재되지 않았다.** 단 규약("이 목록 밖의 변경은 실패로 본다") 적용 대상이다.
⟹ 처분 = 조용한 대장 기재(본 문서) + 계약 개정 또는 이탈 승인은 user 판단.

## D. 아직 미실행 — 런 의존 5건

| 게이트 | 단 | 필요한 것 | 비고 |
|---|---|---|---|
| **Γ4** M1/M2 | SH-GAMMA | MC 팔이 A2-10 **항 조립**에 도달 + **셸별** 가열항 장부(감마 포함) | 08-08 기록은 "A2-10 성공 필요" 라 했으나, 항 장부는 NO_BRACKET 에서도 찍힌다 ⟹ **근을 찾을 필요는 없다**. 다만 현행 출력은 shell-0 진단뿐이라 **셸별 계측 추가 필요**(코딩) |
| **R6-4** MC 바이트-parity | DET-R6 | MC 팔 2회 런(결정론 발행 전/후) + **진짜 byte 비교기** | 08-08 실패 사유가 "필터된 로그 줄 비교" 였으므로 비교기부터 만들어야 한다 |
| **NE2** GPU 전송 음수 주입 | MC-EVT | GPU transport kernel 통합 주입 + 런 | Fable: "실제 GPU transport 통합 음수 주입 미실행" |
| **ME2** GPU ON/OFF 스펙트럼 | MC-EVT | GPU 런 2회(event ON/OFF) end-to-end | Fable: 배열 차이 0 은 **필요조건일 뿐**, 수송 분기·RNG 소비 순서가 달라질 수 있어 스펙트럼 대조로만 판정 가능 |
| **E4** 회귀 바이트-parity | MC-EVT | event ON 구성에서 기존 GPU 런과 byte 대조 | R6-4 와 같은 비교기 재사용 가능 |

**공통 선행물 1개**: 진짜 byte 비교기(필터·다중집합 비교 금지).
R6-4·E4 가 같이 쓰고, ME2 의 스펙트럼 대조에도 쓰인다. **이것부터 만드는 것이 효율적이다.**

**자원**: 현재 syn101 GPU 2쌍을 DET-SPROD 판정런(IV·III)이 점유 중.
GPU 2,3,4,5 는 비어 있으나 CPU 32코어가 이미 두 런에 할당돼 있다.
⟹ 판정런 착지 후 순차 발주가 맞다.

## E. 회수 원칙 (오늘 확인된 것)

N6-3 회수에서 검수가 잡은 결함이 교훈이다. Codex v1 의 판정식이 OR 라서
**commit 이 무관한 이유로 실패해도 PASS** 로 보고되는 구조였다 — 주입 경로를 한 번도
밟지 않고 통과한다. 08-08 에 세 단을 강등시킨 결함과 **같은 계급**이다.

⟹ **회수는 "게이트를 돌렸다" 가 아니라 "주입이 시도됐고, 의도한 가드가, 이름 있는
사유로 막았다" 를 셋 다 보여야 한다.** v2 가 `rejected_at` 단계 분류와 항상 출력되는
trace 를 넣은 덕에 이번 실행은 그 셋을 모두 보인다.

---

## F. 추가 발견 (2026-08-20) — `selftest-sh-radeq-source`: **아무도 안 돌리는데 19건 실패 중**

발견 경위: SH-A209-IDSEAL 코드 검수(Fable)가 R5 로 지적 → 운전석 실측 확인.

### 사실 [실측]

| 항목 | 값 |
|---|---|
| 게이트 | `scripts/check_a209_source_failclosed.py` (make 타깃 `selftest-sh-radeq-source`, `Makefile:168`) |
| 현재 상태 | **exit 1, `[SH-RADEQ-0][STATIC][FAIL]` 19건** |
| 회귀 목록 포함 | **없음** — `.PHONY`(`Makefile:333`)에만 이름이 있고, 어떤 집합 타깃에도·`run_gate_battery.py` 에도·어떤 `.sh` 에도 없다 |
| 최종 수정 | `a00b991`(08-18) |
| 고장 원인(검수 판독) | `function_body` 정규식이 impl/래퍼 분리 이후 3줄짜리 래퍼(`lumina_plasma.c:9408`)를 잡는다. 분리는 `f6c2eb6`(08-18) |
| SH-A209-IDSEAL 과의 관계 | **무관.** 그 단의 `lumina_plasma.c` diff 는 `@@ +9103,7` · `@@ +9129,19` 두 곳뿐으로 래퍼 미접촉 [실측 `git diff -U0`] |

### 왜 이것이 §E 원칙의 재발인가

이 게이트가 지키려던 계약은 사소하지 않다 — 스크립트 docstring: *"line emission uses the
direct `n_u*A_ul*h*nu*beta/(4*pi*dnu)` form, never `line_source_S`; the mutable raw tau slab
is generation-bracketed at both ends; writer/reader share one NLTE authority predicate…"*.
즉 **선방출 형식과 tau 세대 괄호를 지키는 정적 잣대**다. 그것이 08-18 이래 무효 상태로
방치됐고, 무효라는 사실조차 **오늘 코드 검수가 옆길에서 걸려 넘어져** 드러났다.

메모리 [[feedback_audit_the_yardstick_first]] 「회귀 목록에서 빠진 잣대는 조용히 죽는다」
(08-06, `a2_01_census_contract.py`)와 **같은 계급의 재발**이다. 그때의 교훈 ②
*"줄번호·정규식 앵커 기반 잣대는 회귀에 **영구 편입** 필수"* 가 이 게이트에는 적용되지
않았다.

### 처분 — 조용한 대장 기재 (이 단에서 고치지 않는다)

수리는 별개 계약이다. 다음 단이 답해야 할 것:
1. 정규식 앵커를 impl 함수(`a209_publish_cpu_emissivity_impl`)로 옮길 것인가, 아니면
   래퍼/impl 양쪽을 이어 붙여 볼 것인가?
2. 19건이 **전부** 앵커 문제인가, 아니면 그 안에 **진짜 계약 위반**이 섞여 있는가?
   ⚠이것을 확인하기 전에는 "앵커만 고치면 된다"고 단정하지 않는다 — 게이트가 죽은
   기간(08-18~) 동안 실제로 계약이 깨졌을 수 있다.
3. **회귀 목록 영구 편입** — 고친 뒤 집합 타깃/배터리에 넣지 않으면 또 죽는다.

### ★파생 감사 항목 (미실시)
`.PHONY` 에만 있고 어떤 집합 타깃에도 없는 게이트가 **이것 하나뿐인가?**
`Makefile:333` 행에는 `event-measure-check` · `selftest_mc_evt_access` 도 함께 있다.
전수 조사 필요 — **회귀 목록의 완전성 감사**(§E 교훈의 일반형).

---

## G. 회귀 목록 **완전성 감사** (2026-08-20, user 지시) — 고아 21개 · 죽은 게이트 4개 · 근원 1개

증거: `/gpfs/kjhan/lumina/gates/sh_a209_idseal_20260820T044201Z/orphan_gate_status.log`

### G-0. ★감사 도구 자신에게 결함이 있었다 (먼저 적는다)

1차 판정은 고아 **14개**였다. 그런데 `.PHONY` 선언이 백슬래시로 **8줄**(Makefile:331-338)에
걸쳐 있어, 연속행에 이름이 있으면 "참조됨"으로 잘못 분류됐다. 고치니 **21개**다 —
버그가 **7개를 숨겼고**, 그 중에 **이 감사를 촉발한 `selftest-sh-radeq-source` 자신**이 있었다.

⟹ 잣대의 완전성을 재는 잣대에도 같은 병이 있었다. [[feedback_audit_the_yardstick_first]] 에 기재.

### G-1. 사실

[실측] 저장소에 **집합 `selftest:` 타깃이 없다.** 회귀 목록은 `scripts/run_gate_battery.py`
하나뿐이다. 게이트 성격의 make 타깃 **62개 중 21개**가 배터리에도·다른 make 규칙에도·
어떤 `.sh` 에도 없다(`.PHONY` 등재는 실행이 아니다).

GPU 필요분 1개(`selftest_mc_evt_access`)를 뺀 **20개를 전부 돌렸다**:

| 결과 | 수 | 타깃 |
|---|---|---|
| PASS | 16 | bf-edge-census · selftest-bf-edge-census · a2-10-{cancellation-census, cmfgen-mapped-line, line-ion-owners, refinement-comparison, targeted-gate, targeted-reference} · a2_17_jnu_seed · cmf_error_envelope · cmf_exact_sliding · emiss_e11_fluor_matrix · gate_recovery · grid_roundtrip · sh_grid_loader · stage32_rung1 |
| **FAIL** | **4** | 아래 |
| 미실행 | 1 | selftest_mc_evt_access (GPU) |

### G-2. ★죽은 게이트 4개 — 셋이 **같은 근원**이다

| 게이트 | 증상 | 분류 | 근거 |
|---|---|---|---|
| `event-measure-check` | `[E-NE4][FAIL] CPU-A208: missing bf_event_measure_get` | **앵커 노후** | [실측] 접근자는 `a208_publish_cpu_opacity_impl` 안 `:8775` 에서 **실제로 호출된다**. 게이트는 5줄짜리 래퍼 `a208_publish_cpu_opacity`(`:8832`)의 본문을 본다 |
| `selftest-sh-radeq-source` | `[SH-RADEQ-0][STATIC][FAIL]` 19건 | **앵커 노후** | `a209_publish_cpu_emissivity` 도 impl/래퍼로 분리됨(래퍼 `:9408`) |
| `selftest-tau-writer-census` | `unregistered raw tau writer at lumina_plasma.c:19521` | **등록부 노후 + 미판정 위반** | [실측] 19521 은 `nlte_update_tau_sobolev_with_authority`(`:19457`) 안의 `opacity->tau_sobolev[at]=0.0`. 등록부 `WRITERS` 는 `compute_tau_sobolev`·`nlte_update_tau_sobolev`·`apply_overlap_corrections` 셋뿐 — `..._with_authority` 가 없다 |
| `selftest-a2-10-line-saturation` | `positive failed: … Lumina summary target ion is invalid` | **미분류** | 픽스처/자료 문제로 보이나 **확인하지 않았다** |

**공통 근원 [실측]**: 앞의 셋이 앵커로 삼는 함수들은 **`f6c2eb6`(2026-08-18)** 의 일괄 구현에서
**impl/래퍼로 분리되거나 이름이 바뀌었다**(`_impl` 신설, `..._with_authority` 신설).
정적 게이트는 **함수 이름에 앵커를 건다.** 그 커밋이 이름을 바꾸는 순간 셋이 동시에 무효가 됐고,
**셋 다 회귀 목록에 없어서 아무도 몰랐다.**

⚠**아이러니**: 이 게이트들은 정확히 그런 종류의 변경을 잡으라고 만든 것이다.
tau-writer-census 는 "허가받지 않은 raw tau writer 가 생기지 않았는가"를 묻는데,
그 커밋이 **바로 그 일을 했고**(새 이름의 writer 신설) 게이트는 같은 커밋에 의해 눈이 멀었다.

### G-3. 아직 답하지 않은 것 (단정하지 않는다)

1. `19521` 의 raw tau 쓰기가 **정당한가.** `element_inactive` 일 때 `tau=0` 은 물리적으로
   그럴듯하나, 등록부는 **판정을 요구하는 화이트리스트**다. 아무도 판정하지 않았다.
2. `sh-radeq-source` 의 19건이 **전부** 앵커 문제인가, 아니면 **진짜 계약 위반이 섞였는가.**
   게이트가 눈먼 08-18 이후 실제로 깨졌을 수 있다.
3. `a2-10-line-saturation` 의 실패 원인.
4. `selftest_mc_evt_access`(GPU 미실행)의 상태.
5. ★**앵커를 함수 이름에 거는 정적 게이트가 이 셋뿐인가?** 같은 리팩터에 눈먼 게이트가
   더 있을 수 있다 — `function_body(...)` 류 패턴 전수 조사 필요.

### G-4. 처분

- **수리는 별개 계약들**이다. 이 절은 census 이며 아무것도 고치지 않는다.
- ★**구조적 처방**: 고아를 하나씩 되살리는 것으로는 재발을 막지 못한다.
  필요한 것은 **집합 타깃(또는 배터리 편입)의 의무화**와,
  **"게이트가 앵커로 삼는 심볼이 존재하는가"를 검사하는 메타 게이트**다
  (`function not found` 를 FAIL 이 아니라 조용한 오탐으로 흘리는 구조가 이 사고의 절반이다).
- §E 원칙의 재확인: 회수는 "돌렸다"가 아니라 **"주입이 시도됐고, 의도한 가드가, 이름 있는
  사유로 막았다"** 를 셋 다 보여야 한다. 지금 이 넷은 **첫 항목조차 못 보인다.**

---

## H. 판정 — `:19521`·`:19573` tau writer (2026-08-20, 판정자 Fable, user 지시)

증거 패킷: 운전석 계측(`/tmp/claude-10396/tau19521/EVIDENCE.md`). 판정은 독립 Fable.

```
Q1 두 쓰기 자체            = 정당 (등록부 갱신) + 절차 위반 별도 기재
Q2 :20779 두 번째 호출     = 정당 (계약 문언과 양립) + 단서 2건
Q3 NULL 폴백 / EXACT_ZERO  = 정당 (fail-closed / 정직한 exact-zero) + 계측 부채 1건
Q4 눈먼 기간의 다른 위반   = 이 게이트 측정면 안에서는 배제 / 밖에서는 배제 불가
```

### H-1. 정당 판정의 근거 (운전석 재확인분)

- [실측] **세대 괄호 보존**: `tau_sobolev_require_refresh`(`:19463`) → 두 쓰기(`:19521`·`:19573`)
  → `tau_sobolev_mark_computed`(`:19599`). 양 끝 괄호가 추출 후에도 유지된다.
- [실측] 두 쓰기는 신설이 아니라 이사다 — `f6c2eb6^` 의 구 `:15444`·`:15469`.
- 판정은 authority 검사가 **강화**됐다고 본다: 구 코드는 술어를 쓰기 지점에 인라인 복사로
  갖고 있었고, 현 코드는 writer(`:19561`)와 reader(`a209_upper_population_for_tau`)가
  **문자 그대로 하나의 함수** `nlte_tau_line_shell_authorized_by`(`:8933`)를 호출한다.
  자매 게이트 docstring 의 *"writer/reader share one NLTE authority predicate"* 는
  전역 배열의 단일성이 아니라 **술어 함수의 단일성**을 뜻한다 ⟹ 매개변수화가 오히려 계약 이행.
- `element_inactive` 의 `tau=0` 은 **판별식 통과**: `lumina_zinert_Z_inactive_or_absent` 는
  존재비가 **전 셸 항등 0** 이거나 덱에 원소 행이 없을 때만 참이므로 **정확해가 이 가드를
  위반할 수 없다.** validity 도 `A208_VALID` 가 아닌 `A208_EXACT_ZERO` 로 정직하게 표기된다
  ⟹ zero laundering 아님.

### H-2. ★판정이 새로 찾아낸 것 — **게이트가 구조적으로 못 보는 벌크 writer**

[실측·운전석 재확인] `f6c2eb6` 이 신설:
```
src/lumina_plasma.c:21271  memcpy(public_opacity->tau_sobolev, candidate->tau_sobolev, …)
src/lumina_plasma.c:21379  memcpy(public_opacity->tau_sobolev, candidate->tau_sobolev, …)
```
`f6c2eb6^` 에는 **0건**.

게이트의 판별 정규식(`check_tau_writer_generation.py:26-28`)은
`\bopacity\s*->\s*tau_sobolev\s*\[[^\]]+\]\s*=` 다 — **`memcpy` 도 `public_opacity` 도 못 본다.**

⟹ ★**등록부만 고치면 게이트는 PASS 로 돌아가지만, 등록되지 않은 벌크 tau writer 두 개가
측정면 밖에 남는다.** 초록불이 실제로 인증하는 범위가 겉보기보다 좁아진다 —
지금 상태(빨간불)보다 **더 위험한 상태**다. 등록부 갱신과 정규식 확장은 **같은 단**에서
해야 한다.

이 경로는 세대를 advance/mark 하지 않고 후보의 세대값을 통째로 이식한다(`:21403-21408`).
SH-RADEQ-5 의 설계된 트랜잭션 커밋으로 읽히나, **"closed writer set" 등록부는 벌크-이식형
writer 를 상정한 적이 없다.** 별도 판정 필요.

### H-3. 절차 위반 (코드와 별개)

[실측] `docs/ORDER_SH_RADEQ_CMFGEN_TERMS_2026-08-08.md` §3 은 line input ownership 의
Fable 조건부 허용을 **"raw tau writer 3개 census 충족"** 위에 세웠다. `f6c2eb6` 은 그
조건 표면을 바꾸면서 해당 게이트를 돌리지 않았고(회귀 목록 부재) 등록부 판정도 받지 않았다.
**코드는 정당하나 변경 절차는 사다리 규율 위반이다.** 이 판정문이 그 미실시 판정의 소급 수행이다.

### H-4. 판정이 "모른다"고 적은 것

1. **자매 계약은 덮이지 않는다** — `selftest-sh-radeq-source` 19건의 앵커/실위반 분리는
   **판정되지 않았다.** 그 게이트의 눈먼 기간에 대해서는 아무것도 측정하지 않았다 ⟹ **배제 불가.**
2. 자매 게이트가 `a00b991` 시점에 한 번이라도 PASS 했는지 **판별 불가** — 열흘치 일괄
   커밋이라 내부 시계열이 소실됐다.
3. authority **거부 건수가 계측되지 않는다** — "authority 부재"와 "존재하나 거부"를
   구별하는 카운터·로그가 그 경로에 없다. 조용한 성능저하는 아니나(값이 정의된 nebular 로
   남고 validity 로 표시됨) **조용한 분기**이긴 하다. 계측 부채로 기재.
4. Q1 에 **순수 추출이 아닌 의미 변화 1건**이 섞였다 — clear-loop 이 매핑 검사 **앞**으로
   옮겨져, 비활성 원소의 **미매핑 선**도 이제 `tau=0/EXACT_ZERO` 를 받는다(구 코드는 안 받음).
   물리적으로는 정확하나(존재비 0 인 원소의 어떤 선도 tau≡0), **"단순 리네임"이라는 서사는
   부정확하다.** 운전석의 초기 서술이 그러했다 — 정정한다.

### H-5. 수리 명세 (별도 단 — 여기서 고치지 않는다)

1. 등록부에서 `nlte_update_tau_sobolev` → `nlte_update_tau_sobolev_with_authority` **교체**
   (추가 아님 — 래퍼를 남기면 `registered writer has no tau writes` 가 지속).
   ★**동시에** ASSIGN 정규식을 memcpy·별칭(`public_opacity`)까지 보도록 확장(H-2).
2. 회귀 목록 **영구 편입** + G-4 의 앵커-심볼 존재 메타 게이트.
3. 신규 판정 3건 발주: 커밋 경로 memcpy writer 의 지위 · 슬랩↔authority 짝의 런타임 결속
   부재(스탬프 없음) · authority 거부 카운터 부재.
4. 폐합 자격: 등록부 갱신 후 **음성대조 4건이 여전히 4/4 검출됨을 실행으로 시연**해야 한다
   (§E 원칙).

---

## I. 판정 — `sh-radeq-source` 19건의 귀속 (2026-08-20, 판정자 Fable, user 지시)

계측: 운전석(`/tmp/claude-10396/shradeq/EVIDENCE.md`). 판정·최종 감사: 독립 Fable.

```
판정: 인정(분류에 한정) — 19/19 게이트 노후 확정, 실위반 0건
      단 수리 범위를 "앵커·리터럴 갱신"으로 한정하는 것은 기각. 지적 R1·R2.
```

### I-1. 분류는 확정 — 실험 + 실물 대조 이중 확인

[실측·운전석] 소스 무변경, 게이트 사본의 앵커 2줄만 impl 로 교체 ⟹ **19 → 1**.
남은 1건도 리터럴 노후(커밋 호출이 `_counted` 변종으로 바뀜), 순서는 보존
(abort 9482 < promotion 13825 < commit 14272).

[실측·판정자] 토큰 존재를 넘어 **19개 검사의 의도를 impl 실물에 개별 대조**했다 —
fail-closed 토큰 13개 존재, tau 자격 술어 존재, 금지 quotient 읽기 무매치,
세대 괄호 begin(2101)<소비(5426)<end(8720) 실물 보존, abort 블록 안에 commit·
`EMISS_EXACT_ZERO` 세탁 없음, writer 에 authority 토큰 2개·`g_ew_tau_authority` 부재.
⟹ **계약의 19개 성질은 현 HEAD 에서 전부 실물로 성립한다.**

### I-2. 질문 2 답 — `_counted` 교체는 **의미 보존**(운전석 미확인분을 판정자가 확인)

[실측] `f6c2eb6^` 의 구 `a209_publication_commit` 본문과 `_counted` 본문 직접 diff:
**로직 동일**, 차이는 `g_ctr.X++` → `if(counter_sink)counter_sink->X++` 치환과 재포매팅뿐.
구명은 `_counted(pub,c,&g_ctr)` 위임 래퍼로 잔존하고, 공개 경로는 `a209_counters()`
=`&g_ctr` 를 넘기므로 **구 전역 카운터 행동과 비트 동일**. `ctr==NULL` 은 impl 의
`if(!ctr)return 5;` 가 봉쇄. `_counted` 호출자는 저장소 전체에서 impl 1곳.

★**게이트를 알터네이션(`구형 OR _counted`)으로 넓히지 말 것** — 인증 없이 초록 표면만
키운다(§H-2 교훈). 현재 형태에 **정확일치로 고정**하라.

### I-3. ★R1 — 음성대조 주입 7 은 **어떤 커밋된 트리에서도 적용된 적이 없다**

[실측·운전석 재확인] 게이트가 치환하려는 리터럴
`nlte_tau_line_uses_nlte(authority,shell,n_shells)` 의 출현:
**HEAD 0건 · `f6c2eb6^` 0건.** 실제 호출은 `nlte_tau_line_uses_nlte_by(...)` 로
파라미터화됐고 **여러 줄에 걸쳐** 있어 한 줄 정확일치 치환이 원리적으로 불가능하다.

[실측] 앵커·리터럴을 완전 교정한 판본을 돌리면 `[SH-RADEQ-0][STATIC][PASS]` 뒤
**`[NEGATIVE-CONTROL][FAIL] injection-7-not-applied`, exit 1** 이 남는다.

⟹ **운전석 실험은 static 1건 FAIL 에서 `return 1` 로 끊겨 음성대조 단계에 도달하지
못했고, 패킷은 이를 명시하지 않았다.** 수리를 "앵커 2 + 리터럴 1"로 한정하면 게이트는
여전히 빨간불이다. 주입 7 재표적이 **같은 단**에 들어가야 한다.

(게이트가 자기 주입 실패를 FAIL 로 내는 설계 자체는 옳다 — 적용 못 한 주입을 조용히
통과시켰다면 음성대조가 공허해졌을 것이다.)

### I-4. ★R2 — "눈먼 기간 08-18~08-20"은 **과소 서술**이었다

[실측] `f6c2eb6^` 에 `a209_sobolev_line_eta` **0건** — SH-RADEQ-0 구현 자체가 그 시점
커밋된 트리에 없었다. 게이트 스크립트의 최초 커밋은 **`a00b991`(08-18)**.

⟹ **이 게이트가 초록이었던 커밋된 트리는 하나도 없다.** 유일한 초록 기록은
`validation/a2_09/SH_RADEQ_FABLE_REVISE_CLOSURE_2026-08-08.md:21-22`
(injections=8 detected=8) 이며 그것은 **08-08 의 미커밋 작업트리**다.

무인증 구간을 **「08-08 작업트리 → 현재」**로 정정한다. 08-08~08-18 중간 상태는
git 에서 복원 불가(§H-4-2 와 같은 계급).

### I-5. 측정면 밖 우회 — 6류, 그 중 하나는 §H-2 의 방출률판

판정자가 열거한 원리적 사각 6류 중 실물로 확인된 것:

★**벌크 승격 경로** — [실측] `f6c2eb6` 신설 `nlte_population_candidate_commit_bundle`
(`:21238`)·`..._commit_seed_material`(`:21346`)이
`public_opacity->cpu_emissivity = candidate->opacity.cpu_emissivity`(`:21305`·`:21412`)
구조체 통째 이식 + `line_source_S`/`tau_validity` 슬랩 memcpy 를 수행한다.
**이 게이트의 어떤 검사도 이 경로를 보지 않는다.** tau 쪽 §H-2 와 **같은 계급**이다.

단 무가드는 아니다: `candidate_material_commit_preflight`(`:21077` 부근)가 이식 **전에**
세대 정합·closure≤1e-10·전 셀 status·CDF 단조·별칭 검사를 재수행한다.
**런타임 가드는 강하나 정적으로는 무인증** ⟹ 별도 단(§H-5-3 과 묶을 것).

기타: 래퍼 무검사화(앵커를 impl 로 옮기면 공개 래퍼 2개가 측정면 밖) · 엉뚱한 자리의
토큰(주석·로그 문자열도 셈) · 술어 함수 **본문** 무검사 · 파일 범위 2개 ·
`function_body` 정규식이 **호출부에도 매치 가능**(현재는 정의가 먼저라 미발화, 코드 이동
하나로 if-블록을 "함수 본문"으로 검사하게 됨 — G-4 메타 게이트에 포함할 것).

[실측] `f6c2eb6..HEAD` 창 내 src/ 추가 990줄 전수 grep: 신규 벌크 writer·신규 커밋
호출자·신규 `line_source_S` 소비자 **0건**. 위 승격 경로는 창 내가 아니라
**`f6c2eb6` 자체에서** 태어났다.

### I-6. 질문 4 — 배제 가능/불가의 경계

- **측정면 안(f6c2eb6→HEAD): 배제 가능** — 단 게이트가 아니라 **판정자의 직접 3점 diff** 로.
  앵커 대상 6개 함수 중 5개가 **비트 동일**, `a209_publish_cpu_emissivity_impl` 은 헝크
  **정확히 1개**(IDSEAL partition-stamp 블록), `_counted` 는 IDSEAL 거부 3건 추가뿐.
- **측정면 밖: 원리적 배제 불가.**
- **08-08~08-18 중간 작업트리의 일시적 파손: 모른다**(복원 불가).
- 요약: *"창 내 파손이 HEAD 에 잔존하는가"* 는 배제 가능, *"창 내에 일시적으로 존재했는가"* 는
  **판별 불가**.

### I-7. 수리 명세 (별도 단 — 여기서 고치지 않는다)

같은 단에서 함께, 폐합 자격은 §E 3요건(음성대조 **8/8 재시연** 포함):
1. 앵커 2건 impl 로 이동
2. 커밋 리터럴을 `_counted(...,ctr)` **정확일치 고정**(알터네이션 금지)
3. **주입 7 재표적**(R1) — 멀티라인 `_by` 호출 또는 술어 본문
4. 신규 검사 3건: 두 래퍼의 **순수 위임 본문** 정확일치 · `if(!ctr)return 5;` 토큰 ·
   `_counted` 호출자 전수=impl 1곳
5. **회귀 목록 영구 편입**(현재 `.PHONY` 뿐 — `Makefile:168`·`:333`) + G-4 메타 게이트

별도 단: 6. 승격 경로(`:21305`/`:21412`)를 측정면에 편입 — §H-5-3 과 묶을 것.

### I-8. 운전석 계측의 정확도
**수치 오류 0건.** 판정자가 §1·§2 의 모든 [실측] 수치를 독립 재현으로 확인했다.
결손 2건(R1 음성대조 단계 미도달 미명시 · R2 눈먼 기간 과소 서술)만 수정으로 부과됐다.

---

## J. GR-2 판정 착지 + 파생 실측 2건 (2026-08-20)

판정문 정본: `docs/VERDICT_TAU_BULK_TRANSPLANT_2026-08-20.md`.

**판정: (ii) transplant** — `:21271`·`:21379` 의 벌크 이식은 괄호 의무를 지는 writer 가
아니라 **「preflight 증명 + 세대 계보 이식」이라는 별개 계약**을 지는 류다.
결정적 근거: 이식되는 슬랩은 후보의 사적 `OpacityState` 위에서 **등록 writer 3종의 정상
괄호 안에서 생산**되고(`:20767`·`:20779`), 폐합이 생산 직후(`:20784`)와
preflight(`:21168-21170`)에서 이중 검증되며, 세대는 공적 계보의 연속이다.
(i)은 재괄호가 first-consumer 장부를 거짓으로 만드는 범주 오류, (iii)은 클램프 판별식
통과 + 원자성(첫 공적 바이트 이후 무실패 구간)으로 기각.

★**사전등록의 최소 인증선은 불충분**하다고 판정됐다 — "호출 실존 ≠ 가드됨"이고 필드 접촉
검사는 `=0` 사보타주를 통과시킨다. 대신 검사 7종(T1~T7) 명세가 GR-4 로 넘어간다.

### J-1. ★파생 실측 A — CUDA 에도 벌크 tau writer 가 있고, 초록불 문안이 그것을 가린다

[실측·운전석 재확인] `src/lumina_cuda.cu:10160`·`:10196`:
```c
memcpy(opacity.tau_sobolev, tau_save, nline * sizeof(double));
```
env-gated 진단(withParityP GATE1)의 byte-clean save/restore 다.

[실측] 게이트의 `CUDA_ASSIGN` 정규식은 `opacity(.|->)tau_sobolev[...] =` 만 본다 —
**memcpy 는 못 본다**. 그런데 PASS 문안(`check_tau_writer_generation.py:114`)은
`cuda_writers=0` 을 **하드코딩 리터럴**로 찍는다.

⟹ 그 문구는 *대입* 에 대해서는 참이지만 **모든 writer 에 대한 주장으로 읽힌다.**
§H-2 가 CPU 쪽에서 확인한 병("초록불이 인증 범위를 과장한다")의 **CUDA 판**이다.
⟹ **GR-4 의 기대 `cuda_writers=0` 은 그대로 쓸 수 없다.** GR-2b 판정이 선행해야 한다.

### J-2. ★파생 실측 B — transplant 계약의 유일한 음성대조가 `unwired` 다

[실측] preflight 의 음성 대조는 **소스에 실존**한다(seed 3건 + bundle 2건, 바이트 보존
전수 주장). 그런데 그것을 담은 타깃들이 [실측] 등록부에서 **`unwired`** 다.

★**감리 R4 정정 (2026-08-21)**: 초판은 "**두 타깃**" 이라 쓰고 **세 이름**을 나열하는
자기모순이었고, 그 세 이름은 GR-2 판정문의 지목과도 달랐다.
[실측] **정본은 GR-2 판정문 §7** 이며 거기 지목된 것은
`tests/a2_10_seed_commit_selftest.c:214-289`(seed preflight 음성대조 **3건**) ·
`tests/nlte_candidate_tau_selftest.c:404-533`(bundle 경로) — **둘**이다.
초판이 나열한 셋에는 **`selftest_a2_10_seed_commit` 이 빠져 있었다.**

⟹ 보호 대상은 **합집합**으로 읽는다: `selftest_nlte_population_candidate` ·
`selftest_nlte_candidate_adiabatic` · `selftest_nlte_candidate_tau` ·
**`selftest_a2_10_seed_commit`** — **4행**, 전부 `disposition_rung=GR-7`.
GR-7 판정은 이 **4행 전부를 배선으로 처분**해 어느 해석 아래서도 위반이 없다.

⟹ ★**GR-7 이 이들을 「은퇴」로 처분하면 방금 (ii)로 판정된 transplant 계약의 유일한
음성대조를 지우게 된다.** GR-7 판정에 이 제약을 전달한다 —
「은퇴는 지키던 계약의 소멸을 판정문이 확정할 때만」이라는 원칙이 여기서 구체적 금지가 된다.

[실측] 다만 그 음성대조에 **tau-괄호·별칭 조건의 표적 주입은 없다** — 있는 것은
바이트 보존 계열이다. 즉 transplant 계약의 **일부만** 덮는다.

### J-3. Q3 처분
§H-5-3 의 ②(슬랩↔authority 런타임 결속 부재)·③(authority 거부 카운터 부재)는
**대장 기재로 종결**하고 재개 트리거를 명문화했다 — 둘 다 `src` 0줄 계약과 A2-10 동결에
이중 저촉이라 지금 집행 불능이다. 부속 발견 1건 추가: **bundle 커밋에는 공적 세대 결속이
없고 seed 경로에는 있다 — 비대칭**[실측].

---

## K. GR-2b 판정 — CUDA 벌크 tau 복원은 (iv) 「진단 save/restore」 (2026-08-20)

판정문 정본: `docs/VERDICT_CUDA_TAU_RESTORE_2026-08-20.md`.

**판정: (iv) 진단 save/restore(관측자 상태 재생)** — 새 류를 정의했다.
(i) 기각: census 계약은 "production write" 인데 이 경로는 생산이 아니라 **등록 생산물의
바이트 재생**. (ii) 기각: preflight·세대 이식·소유권 이전 전부 부재. (iii) 기각: 클램프
판별식 통과(값 비변조)·기본 OFF·물리 무접촉.

### K-1. ★그러나 무조건이 아니다 — 판정이 찾아낸 화석 주석

[실측·운전석 재확인] `nlte_solve_all_gpu` 는 `src/lumina_cuda.cu:1938` 에서
**등록 writer `nlte_update_tau_sobolev` 를 실제로 호출한다.**
그런데 `:10316` 의 주석은 정반대로 말한다:
> *"The pure-CMFGEN loop's GPU NLTE solve does **NOT** call nlte_update_tau_sobolev…"*

**화석이다.** 그 결과 armed 런에서:
- **α (무신호 복원)**: 재솔브가 공적 tau/S_l 세대를 전진시킨 뒤, 복원 memcpy 가
  **세대에 아무 신호 없이 바이트만 되돌린다.**
- **β (메타 비복원)**: 저장·복원 목록에 **세대 스칼라·`tau_validity` 가 없다** ⟹
  armed 런은 「바이트 = 수렴 세대, 메타 = 진단 세대」로 잔존한다.

⟹ 붉은 대장행 2건 + 재개 트리거 3종으로 처분(판정문 §7).

### K-2. ★운전석 실측 — 정본 런은 전부 비무장이다 (오염 없음)

[실측] `/gpfs/kjhan/lumina` 전 코퍼스의 `RUN_FOOTER.txt` 중
`LUMINA_NLTE_FINAL_RESOLVE=1` **0건**. 현행 DET 정본 런 3종
(`l4_…pin_seed` · `p1_…physcmp_named_reason` · `idseal_…a209`) 전부 비무장.
무장 러너는 `scripts/launch_parity27~39*` 계열(구 캠페인)뿐이다.

⟹ **결손 α·β 가 정본 산출물을 오염시킨 적은 없다.** 판정문이 "K36 R6 실런의 러너를
특정 못함(모른다)" 으로 남긴 항목이 이 실측으로 닫힌다.

### K-3. `cuda_writers=0` 의 처분 — 조작이 아니라 **측정면 과장**

[실측] ASSIGN census 실패 분기(`check_tau_writer_generation.py:77-81`)가 리터럴 출력 앞을
막으므로 **거짓 주장은 아니다**. 그러나 문구가 *모든* writer 에 대한 주장으로 읽힌다.
GR-4 로 넘기는 명세 U1~U5: 전-src 벌크 census · 가드 앵커+블록 스팬 pin · 복원 문면 pin ·
**문안 정직화(리터럴 폐지 → 실측 카운트 + 류별 카운트 + 비인증 잔여 참조)** · NC 5건 사유 고정.

[실측] 오늘 기준 전-src 벌크 tau 경로 **정확히 4건** = transplant 2(CPU) + save-restore 2(CUDA).

### K-4. 동류 경로 — tau 밖에도 있다

| 경로 | 잣대 상태 |
|---|---|
| pops 복원 2건 | A2-07 census 가 `plasma.c` 만 읽어 **CUDA 사각** |
| `line_source_S` 복원 2건 | ★**S_l writer census 가 어디에도 없다** — 부재 자체가 발견 |
| `jbar` 주입 1 + 복원 1 | **무잣대**. `:10166` 은 복원이 아니라 raw 스냅샷의 *일시 게시* 로 구분 표기 |
| 블록 밖 친척 3건(`:10845`·`:1853`·`plasma:20136`) | 별류로 기재 |

### K-5. 태생 커밋 — 운전석의 정정이 틀렸다

[실측] 경로 한정 pickaxe(`git log -S "memcpy(opacity.tau_sobolev" -- src/lumina_cuda.cu`)는
**`a97d0e1` 단독**이다. GR-2 판정문의 귀속이 옳았고, 운전석이 본 "다른 커밋들"은
**경로 무제한 pickaxe 가 문서·대장을 매치**한 것이었다(예: `513ee92` = GR-2 판정문 인용).
운전석의 정정을 철회한다.
다만 `a97d0e1` 은 1,369파일 덩어리라 **저작 시점은 모른다**(정황상 07-25 무렵 [추정]).

---

## L. 스테이징된 검사기가 저장소 정본과 갈라졌다 (2026-08-20, 오늘 **세 번째** 같은 계급)

발견 경위: SH-A209-IDSEAL 판정런(job 321104)이 `model_rc=0` 으로 성공했는데
사후 게이트가 `DET_FLIGHT_FATAL A2-10 targeted log gate failed` 로 죽었다.

### 사실 [실측]

같은 로그에 두 판본을 돌린 결과:
```
스테이징 사본(런이 실제로 쓴 것) → A210_TARGETED_GATE_FAIL nonzero numerical repair field cap=64
저장소 정본                      → A210_TARGETED_GATE_PASS devices=2 refinements=36
                                    floor=0 cap=0 clamp=0 jitter=0 repair=0
```

| 항목 | 값 |
|---|---|
| 스테이징 사본 | `/gpfs/kjhan/lumina/det_stage12_fixed_te_a100x2_k36/idseal_20260820T044703Z_a209/input/check_a210_targeted_gate.py` sha256 `4b3c3fcc…` |
| 저장소 정본 | `scripts/check_a210_targeted_gate.py` sha256 `dffd4d0d…` |
| 차이 | 정본에만 있는 **예외 조항** — `EXACT-MULTIGPU-EPOCH` 출현 정본 2건 / 사본 1건 |
| 예외의 취지(정본 주석) | *"This record's unqualified `cap` is the exact solver's maximum iteration count. It is **execution metadata, not a cap applied to any physical value**."* |
| 실제 `cap=64` | `[cmf_fine][EXACT-MULTIGPU-EPOCH] … iterations=45 cap=64 residual=9.67e-09` = `LUMINA_CMF_FINE_ALI=64` |
| 예외가 정본에 들어온 커밋 | `a00b991`(08-18) |

⟹ **게이트 실패는 오탐이다.** 물리값 캡이 아니라 솔버 반복수이고, 정본 검사기는 그것을 안다.

### 귀속 — 운전석

L4 런의 `input/` 을 통째로 복제해 판정런을 스테이징하면서 **낡은 검사기까지 복제**했다.
봉인 입력 보존이 취지였으나 **검사기는 봉인 대상이 아니라 갱신 대상**이다.

[실측] `job.slurm` 은 deck·sigma·binary·topion 을 **sha256 으로 봉인 검증**하지만
**검사기는 `-x`(실행 가능) 만 본다** — 내용 검증이 없다.

### ★오늘 세 번째 같은 계급이다

| # | 갈라진 중복 | 단 |
|---|---|---|
| 1 | 배터리 빌드 명세 ↔ Makefile 타깃 | GR-0 |
| 2 | 게이트 앵커 ↔ 리팩터된 소스 함수명 | GR-3′ |
| 3 | **스테이징 검사기 ↔ 저장소 정본** | 이 절 |

셋 다 **"같은 것을 두 곳에 적어 두고 한쪽만 갱신"** 이다.
GR-8(중복 명세 정합 검사, `build-spec-drift`)이 1을 겨냥하는데, **3은 그 측정면 밖**이다 —
런 스테이징은 저장소가 아니라 `/gpfs` 에서 벌어진다.

### 처분

- **판정런의 실질 결과는 바뀌지 않는다** — `model_rc=0`, B1·B2·B3 적중, 정본 게이트 PASS.
- ⚠**그러나 `DET_FLIGHT_ACCEPT` 를 받지 못했다는 사실을 지우지 않는다.** 판정문에
  "낡은 사본으로 실패했고 정본으로 통과했다" 를 그대로 적는다.
- **수리 후보(별도 단)**: `job.slurm` 이 스테이징된 검사기·판정 스크립트도
  **sha256 으로 봉인 검증**하게 한다(deck·binary 와 동급). 또는 스테이징이 저장소
  정본을 매번 새로 복사하고 그 해시를 `RUN_FOOTER` 에 기재한다.
  ★어느 쪽이든 **"어느 판본이 판정했는가"가 산출물에 남아야** 한다 — 지금은 안 남는다.

### L-2. 같은 계급 **4번째** — 런 provenance 가 화석이다 (IDSEAL 판정이 발견)

[실측·운전석 재확인] 세 DET 런의 `input/git_head.txt` 가 전부 **`dd9f7c18`** 이고
**mtime 이 나노초까지 동일**하다(`2026-08-19 21:12:18.130`):

| 런 | git_head.txt | 실제 코드 |
|---|---|---|
| `l4_…pin_seed` | `dd9f7c18` | (당시 HEAD) |
| `p1_…physcmp_named_reason` | `dd9f7c18` | `ccfaab1`(P-1 계측) |
| `idseal_…a209` | `dd9f7c18` | **`2dc2817`**(SH-A209-IDSEAL) |

원인은 운전석의 `cp -a` — **원본 타임스탬프까지 그대로 복제**해 화석이 전파됐다.
`dd9f7c1`(08-19)은 DET-STAGE12 고정-T 레인 커밋으로 **이틀 전 것**이다.

⟹ **런 산출물이 "어느 코드가 이 결과를 냈는가" 를 틀리게 증언한다.**
바이너리 신원은 sha256 사슬과 런타임 출력으로 별도 입증되므로 이 단 판정에는 무영향이나,
**provenance 기록 자체가 거짓**이다.

§L(스테이징 검사기)과 **같은 뿌리**: 봉인 입력을 통째 복제하는 스테이징이
**갱신돼야 할 것까지 얼려 버린다.** 수리는 같은 단에서.

---

## M. `selftest_nlte_assemble` 이 링크 목록 누락으로 빌드 불가 — 오늘 **다섯 번째** 같은 계급

발견 경위: GR-7 스텝 1 계측 중 운전석이 `nvcc` 부재 3건을 **한 부류로 묶어** "돌릴 수
없었다" 로 적었고, **user 가 "grammar-debug 는 CPU 노드임" 을 지적**해 재측정하다 드러났다.

### M-1. 운전석 계측의 오류 (먼저 적는다)

`grammar-debug` 는 **CPU 전용 디버그 노드**라 `nvcc` 부재가 정상이고 **결함이 아니다.**
그런데 로그인 노드 `syntax` 에는 `nvcc` 가 있고(`/opt/ohpc/pub/cuda/13.0.2/bin/nvcc`,
CUDA 13.0) **규약상 빌드는 로그인 노드 허용**이다 — **잴 수 있는 것을 안 쟀다.**
"레인에서 못 돌렸다" 와 "돌릴 수 없다" 를 섞은 것이 오류였다.

### M-2. 재측정 — 셋이 갈린다 [실측, `syntax` 빌드만]

| 타깃 | `BUILD_RC` | 판독 |
|---|---|---|
| `selftest_a2_12_gpu_lifecycle` | **0** | 빌드 OK · 실행 미측정(GPU 티어) |
| `selftest_a2_13_gpu_oracle` | **0** | 동상 |
| **`selftest_nlte_assemble`** | **2** | ★**링크 에러 — 실결함** |

### M-3. 실결함의 정체 — 변수는 있는데 타깃만 안 쓴다

[실측] 미정의 심볼 8개 이상 — 전부 `src/nlte_population_candidate.c` 정의:
```
nlte_population_candidate_begin · _prepare_opacity_view · _prepare_thermodynamics
nlte_ion_solve_status_name · nlte_population_solve_stage_name
nlte_population_candidate_status_name · nlte_population_solve_diagnostic_reset …
```
`src/lumina_plasma.c` 가 부르는데 **정의가 안 링크된다.**

★[실측] **`Makefile:39` 에 `NLTE_CANDIDATE_SRC = src/nlte_population_candidate.c` 가 이미
있고 다른 타깃들은 그것을 쓴다.** `selftest_nlte_assemble`(`Makefile:121`)의 링크 목록에만
**없다.**
[실측] 그 심볼들은 **`f6c2eb6`(08-18)** 에서 들어왔다 — 오늘 내내 나온 그 커밋이다.

### M-4. ★오늘 다섯 번째 같은 계급

| # | 소스는 바뀌었는데 따라가지 않은 것 | 발견 단 |
|---|---|---|
| 1 | 배터리 빌드 명세(`Z-a2-09`) ↔ Makefile 타깃 | GR-0 |
| 2 | 게이트 앵커 ↔ 리팩터된 함수명 | GR-3′·GR-4·GR-5 |
| 3 | 스테이징 검사기 ↔ 저장소 정본 | §L |
| 4 | 런 provenance(`git_head.txt`) ↔ 실제 코드 | §L-2 |
| **5** | **selftest 링크 목록 ↔ 소스가 얻은 함수** | **이 절** |

전부 *"같은 것을 두 곳에 적어 두고 한쪽만 갱신"* 이다. **1 과 5 는 형태까지 동일**하다 —
링크 목록에 파일 하나가 빠졌고, 그것이 존재하는 변수로 이미 표현돼 있다.

### M-5. 처분

- **수리는 이 단이 아니다.** GR-7 은 처분 **판정**까지이고, 집행은 캠페인 밖 후속 단이다.
- 운전석 실측을 판정에 전달했고, **`known-red` 전환 후보**로 제시했다(수리 단 번호 필수).
  ⚠**환경 부재로 처분하면 실결함을 숨기는 것**임을 명시해 넘겼다.
- ★**GR-8(중복 명세 정합 검사)의 정의역 재검토 후보**: GR-8 은 배터리 빌드 명세 ↔ Makefile
  갈라짐을 겨냥하는데, **이 건은 Makefile 타깃 자신의 링크 목록 누락**이라 그 측정면에
  들어가는지 불명이다 — GR-8 사전등록 시 확인할 것.

### M-6. ★운전석 계측이 관대했다 — `make` 의 `rc=0` 을 게이트 건강의 잣대로 썼다

GR-7 판정이 운전석 패킷을 정정했고, 운전석이 재확인했다.

[실측] `step1_27.log` 에서 **`is up to date` 가 6건**이다:
`a2_08_signed_opacity` · `a2_09_emissivity` · `a2_10_radeq` · `cmfgen_adiabatic` ·
`physics_comparison` · `a2_16_seed`.
⟹ `make` 가 `rc=0` 을 냈지만 **recipe 가 한 줄도 안 돌았다.**

[실측] 실제로 PASS 문구를 낸 것은 **27 중 8건뿐**이다.

⟹ 운전석이 "24 rc=0" 을 성과처럼 보고한 것은 **관대한 판독**이었다.
**`rc=0` 은 "돌아서 통과했다" 가 아니라 "빌드가 최신이라 안 돌렸다" 일 수 있다.**
오늘 두 번 데인 순간값 함정과 같은 계열 — **요약 종료코드를 잣대로 썼다.**

[실측] 판정의 두 번째 정정도 확인했다: `run_a2_09_selftest.py` 는 `Makefile` 에 1건,
`run_gate_battery.py` 에 **0건**이다 ⟹ **배터리 Z 러너는 그림자의 driver 를 재현하지 않는다.**
POISONS 음성대조 8건이 그 unwired make recipe 로만 돈다 — **은퇴시켰으면 음성대조를
통째로 잃었다.**

### M-7. GR-7 판정 결과 (정본: `docs/VERDICT_UNWIRED_GATES_2026-08-20.md`)

**배선 26**(preflight 1 · run-dependent 2 · milestone 23) · **known-red 전환 1** · **은퇴 0**.

- known-red 1 = **`selftest_nlte_assemble`**(§M 의 실결함). 수리 단 **SH-UW-1** 지정.
- **은퇴 0 의 근거**: 27행 전원의 피보호 계약이 HEAD 현역이고, **계약 소멸을 확정한
  판정문이 0건**이다 — 사전등록의 은퇴 요건("지키던 계약의 소멸을 판정문이 확정할 때만")을
  충족하는 행이 없다. 가장 강한 은퇴 후보(배터리 ⊋ make 인 `a2_12_contract`·
  `a2_13_15_contract`)도 **GR-8 이 그 recipe 를 정본 쌍으로 쓰므로** 기각.
- GR-2 의 transplant 트리오 금지 **상속·준수**(오늘 셋 다 fresh 빌드 + PASS [실측]).
- CUDA 2건은 **run-dependent 배선** — "빌드 OK · 실행 미측정" 으로 기재, 환경 부재 처분 안 함.

**판정자 신규 실측 3건**(운전석 미측정 항목): ①`is up to date` 6건 ②배터리 Z 가 그림자
driver 미재현 ③**build-only 9건의 검사 본문은 실행 이력 0** — 배선 처분에
"recipe 실행 단계 부가 + 1회 건강 실측" 을 전제로 강제(SH-UW-4).

### M-8. ★GR-7 감리(검수 감리 겹 시범) — 판정 인정, 지적 6건

user 지시로 **판정 감리 겹을 처음 붙였다**(오늘까지 감리는 판정에만, 검수에는 없었다).
독립 fresh Fable. **판정: 인정** — 27행 처분에 행별 근거가 실존하고 표본 검증이 전부 일치.
관대한 처분 없음. 그러나 **판정자도 못 본 기계 충돌 1건**을 잡았다.

| # | 지적 | 처분 |
|---|---|---|
| **R1** | ★**`SH-UW-1` 이 체커의 `repair_rung` 형식을 위반한다** — `check_gate_registry.py:250-252` 가 `re.fullmatch(r"GR-[1-9][0-9]*", …)` 를 요구하고 `:378` 이 `wiring == repair_rung` 을 강제한다. [실측·운전석 재확인] `SH-UW-1` 은 **fullmatch 실패** ⟹ known-red 등재 즉시 `known-red-row-incomplete` fail-closed. **판정문 §2-5 는 disposition 채널의 기계 정합만 재고 known-red 채널은 재지 않았다 — 비대칭 검증** | ★**SH-UW-1 착수 전 필수 해소.** 셋 중 택일: 체커 rung 형식 확장(사전등록 필요) / GR-계열 번호 부여 / 판정문이 열어 둔 "즉시 수리 착지 시 known-red 경유 생략" |
| **R2** | 폐합 조건 ② 가 HEAD 에서 **미충족** — 사전등록은 "판정문 + 등록부 기입 = **커밋 1**" 인데 `83c3f31` 은 판정문+대장 2파일뿐이고 27행에 `disposition` **0건** | ★**해소** — 이 커밋에서 27행 전원 기입. 형상 이탈(판정문 선행 단독 커밋)을 여기 기재 |
| **R3** | `Z-a2-12` 쌍의 빌드 명세가 **이미 2항 어긋나 있다**(배터리는 `jnu_seed.c`·`seed_capability.c` 를 더 링크) ⟹ **GR-8 의 "HEAD PASS" 기대가 이미 거짓일 수 있다.** GR-0(부족)과 **반대 방향(과잉)** 의 drift | ★**GR-8 착수 전 정보 필수** — 사전등록 시 반영 |
| **R4** | 대장 §J-2 가 "두 타깃" 이라 쓰고 세 이름 나열, GR-2 §7 의 실제 지목(`a2_10_seed_commit` 포함 2개)과 불일치 | ★**정정 완료(2026-08-21 재시도)**(§J-2) — 보호 대상 = 합집합 **4행**. ⚠첫 커밋(`6225867`)에서 편집이 **앵커 불일치로 실패**했는데 커밋 메시지는 '정정 완료'라 적었다 — **허위 기재였고 이 커밋에서 바로잡는다** |
| R5 | 계측 패킷의 "wave32_* 4종" 은 **5종** | 기록 정확성, 무영향 |
| R6 | 배선 전제(1회 건강 실측)의 강제가 **절차뿐** — `unwired-now-referenced` 는 재배속만 강제하고 전제 실측은 기계가 침묵 | 집행 단 게이트 표에 "1회 실측 로그 경로" 를 판정 자료로 명기하면 닫힌다(체커 개조 불요) |

### M-9. ★운전석의 기입 자동화도 한 번 틀렸다 (기록)

첫 시도에서 판정문을 **200자 정규식 창**으로 훑어 처분을 뽑았더니
`milestone 25 · run-dependent 0` 이 나왔다 — 판정문의 실제 값은 `milestone 23 ·
run-dependent 2` 다. **판정문의 표를 열 단위로 파싱해 재작성**했고, 27/27 매칭 실패 0으로
확정했다. ⟹ **원장 기입을 대충 자동화하면 판정을 왜곡한다.** 오늘 세 번째 "잣대를 대충
읽은" 사례다(순간 스냅샷 · `make` 종료코드 · 이번).
