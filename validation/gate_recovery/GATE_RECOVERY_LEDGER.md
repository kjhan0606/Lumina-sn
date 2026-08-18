# 미실행 게이트 회수 — 실행 기록 (2026-08-18)

user 지시 "미실행 게이트 회수해". 2026-08-08 에 사전등록됐으나 실행되지 않은 채
단이 "폐합" 으로 기재되어 감사가 강등시킨 게이트들을 회수한다.

산출: `selftest_gate_recovery` (sha256 앞자리 `633b9b9b84dfbe2b6b71`)
로그: `validation/gate_recovery/GATE_RECOVERY_2026-08-18.log` (`7e91c856597828ed5c45`)
실행처: grammar-debug 노드 (로그인 노드 연산 금지 규약 준수)
소스: `tests/gate_recovery_selftest.c` — 코딩=Codex, 검수=Fable, 빌드·실행=운전석 (개정13)

## 결과 — 4/4 PASS (양성 대조 4 동반)

| 게이트 | 주입 | 실측 증거 |
|---|---|---|
| **NC2** (SH-GAMMA) | 발행 후 다른 epoch 로 require | `GAMMA_PUBLICATION_STALE_EPOCH`. 양성: 같은 epoch 는 `OK` |
| **NC4** (SH-GAMMA) | 같은 epoch 로 재발행(INTERNAL_BATEMAN) | rc=4 · stderr `GAMMA_DOUBLE_PUBLISH` · **두 공개 배열 byte 보존**. 양성: `epoch+1` 은 정당한 재발행으로 `generation=2` 발행 성공 ⟹ 게이트가 과잉이 아님 |
| **N6-2** (DET-R6) | q-set 해시 **정확히 한 글자** 변경 | `[LINE_JBAR_VIEW][BLOCKED] reason=QHASH_MISMATCH`. 기존 selftest 는 전혀 다른 해시를 넣는 조잡한 형태였고, 이번에 근접 오탐까지 확인 |
| **N6-3** (DET-R6) | 창 밖 선을 `VALID` 로 위장(센티널 −1) | `[LINE_JBAR][BLOCKED] reason=NEGATIVE_OR_NONFINITE_JBAR cell=1 validity=1 value=-1` <br> trace: `commit_rc=-1 view_rc=-4 lookup_rc=-1 rejected_at=commit injection_commit_attempted=1` |

## ★N6-3 이 이번 회수의 요점

검수(Fable)가 Codex v1 에서 잡은 결함: 판정식이 OR 라서 **commit 이 무관한 이유로
실패해도 PASS** 로 보고됐다 — 주입 경로를 한 번도 밟지 않고 통과하는 구조이며,
08-08 에 세 단을 강등시킨 결함과 **같은 계급**이다(약한 증거로 PASS).

v2 가 `rejected_at` 단계 분류와 항상 출력되는 trace 를 넣은 덕에, 이번 실행은
**주입이 시도됐고**(`injection_commit_attempted=1`) **의도한 가드가**
(`reason=NEGATIVE_OR_NONFINITE_JBAR`) **commit 경계에서 막았다**는 것을 함께 보인다.
v1 이었다면 `rejected_at` 이 안 보여 이 구별이 불가능했다.

## 검수에서 잡은 다른 2건 (v2 에서 수리됨)

- **F1**: NC4 가 `GammaDeposition*` 를 `AtomicData*`·`PlasmaState*`·`Geometry*` 로 가장 —
  테스트의 메모리 안전성이 생산 코드의 **가드 순서에 의존**하게 된다(재배치 시 segfault).
  ⟹ 영-초기화 실구조체로 교체.
- **F3**: N6-3 이 owner 자신의 살아 있는 배열을 그 owner 로 들어가는 commit 의 source 로
  넘김(자기 별칭) ⟹ 독립 버퍼 복사.
- **F4**(운전석 처리): Codex 가 링크 의존성을 "모르겠다" 고 정직하게 남긴 부분.
  실측으로 확정 — 구현은 `src/lumina_plasma.c`, 링크 세트는 `selftest_seed_te_publish` 와
  동일, 플래그는 `-std=gnu11 -D_GNU_SOURCE`(`-std=c11` 로는 안 선다).

## 남는 관찰 (무해)

`injection_commit_attempted` 는 호출 직전에 무조건 1 로 세우므로 그 CHECK 자체는
형식적이다. 실질 증거는 **trace 한 줄**이며 운전석은 PASS 줄이 아니라 그 줄을 읽어야 한다.
