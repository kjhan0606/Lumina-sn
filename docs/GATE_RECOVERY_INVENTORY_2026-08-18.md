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
