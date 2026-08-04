# Codex A6 — Wave-3.2 잔여 3건 최종 수리 보고서

## 1. 범위와 폐합 판정

입력은 `docs/CODEX_WAVE32_C5_REVIEW.md`의 rung 2–4 잔여 세 좌표로
한정했다.

| A6 rung | C5 인용 좌표 | 폐합 결과 |
|---|---|---|
| 1 `expected_outflow` 소유권 | C5 L12–19: file-static capture의 이중배열 증가가 비원자적이고 병렬 self-test에서 누락 | matrix placement와 독립 ledger를 동일 OpenMP critical 영역으로 묶고 production hook 병렬 fixture 추가 |
| 2 Python API 계약 태깅 | C5 L21–25: `check_artifact(..., non_contract_override=True)` 반환값에는 계약 상태가 없음 | `CheckResult.contract_status`를 API 반환 계약에 추가하고 CLI와 직접 소비자도 이 필드를 사용 |
| 3 NaN 우회 두 경로 | C5 L27–37: `n_elem=NaN`의 0 정상화와 유한 `tau` 합 overflow 우회 | 비교 전 `n_elem` finite 검사, 각 tau 합산 직후 finite 검사, 두 production helper seed fixture 추가 |

C5가 PASS로 인정한 Rung1 I/O 폐합(C5 L7–9)과 기존 보존행/matrix/flux
finite 감사는 수정하지 않았다.

## 2. 납품 패치

패치는 A5 누적 작업트리의 A6 시작 스냅샷을 기준으로 한 순차 증분이다.

1. `patches/w32a6_rung1.patch` — 122 lines,
   SHA-256 `53c218dcbe712a428bb259a087e90fd0b55be918509ce5fa1ceaf2e43d8956c2`
2. `patches/w32a6_rung2.patch` — 136 lines,
   SHA-256 `47fab63d718ebd57bba68409316b62b61a8591a74b8268dcd097cbdd5d681d85`
3. `patches/w32a6_rung3.patch` — 150 lines,
   SHA-256 `f212d77c38038f7259869d89a87b8bf934db6ad46d8f95f88227dcc95787e692`

별도 임시 트리에 1→2→3을 `patch -p1`로 적용하고 최종 관련 파일 전체를
작업트리 스냅샷과 비교했다.

```text
sequential_patch_apply=PASS
final_related_files_byte_identical=PASS
```

## 3. Rung 1 — `expected_outflow` 실행 가능한 소유권

### 기대 변경집합

- `src/lumina_element_wide.c`: production `nlte_ew_capture_transition()`의
  off-diagonal inflow, diagonal debit, `expected_outflow[channel][j]` 증가를
  `lumina_ew_capture_accumulate`라는 하나의 OpenMP critical 영역에 둔다.
- rate 계산, target/source mapping, 세 부동소수점 덧셈/뺄셈 산술식은
  변경하지 않는다.
- `tests/wave32_counter_atomic_selftest.c`: 실제 production transition hook을
  200,000회 병렬 호출해 이중배열 ledger와 matrix 양쪽을 검증한다.

구현 좌표는 `src/lumina_element_wide.c` L449–463이며, fixture가 사용하는
동일 production hook의 배열 설치/호출은 L466–496이다. 병렬 검증과 판정은
`tests/wave32_counter_atomic_selftest.c` L47–71이다.

### 자기검증과 음성 대조

1-thread 및 8-thread에서 동일했다.

```text
expected_outflow=200000 matrix_inflow=200000 matrix_debit=-200000 all_exact=1
invalid_rate_bad_rate=1 arrays_unchanged=1
```

EW-OFF/8-thread 대조에서는 production runtime meter 3종이 모두 0인 기존
계약을 유지하면서, fixture가 직접 압박하는 capture 배열은 여전히 정확했다.
NaN rate 음성 대조는 기존 rate guard에서 `bad_rate=1`이 되고 세 배열을 전혀
변경하지 않았다.

## 4. Rung 2 — Python API 계약 상태 반환

### 기대 변경집합

- `scripts/cmf_chieta_check.py`: 기존 payload 결과 여섯 필드와 명시적
  `contract_status`를 갖는 `CheckResult`를 반환한다. 값은 `CONTRACT` 또는
  `NON-CONTRACT`다.
- 비계약 기대값의 무단 호출은 기존처럼 `CheckError`; 명시 override로
  무결성 검사를 통과한 호출만 `NON-CONTRACT` 결과를 받는다.
- CLI도 `args.non_contract_override` 자체가 아니라 API 반환 상태로 출력과
  RC 2를 선택한다.
- 기존 직접 소비자 `cmf_chieta_roundtrip_selftest.py`를 named field 사용으로
  이행하고 `CONTRACT`를 확인한다.
- `tests/test_wave32_seeded_defects.py`에 직접 import한 API의 정상, 무단,
  명시 override 세 경로를 추가한다.

반환 계약은 `scripts/cmf_chieta_check.py` L31–38 및 L131–134, CLI 소비는
L150–166이다. API fixture는 `tests/test_wave32_seeded_defects.py` L81–113,
기존 round-trip 소비자 이행은 `scripts/cmf_chieta_roundtrip_selftest.py`
L78–82이다.

### 자기검증과 음성 대조

```text
api_contract_status=CONTRACT
api_unauthorized=REJECTED
api_override_status=NON-CONTRACT
iter7_bypass_rc=1 explicit_override_rc=2
```

정상 artifact의 write-read-write byte round-trip도 유지됐다.

```text
PASS LCMFCE01 write-read-write bitwise roundtrip
sha256=3981641ed3fa6f9bfac8425b248f546012501fac78094522816a22ab950c6d52 bytes=424
```

따라서 직접 import 소비자도 비계약 결과를 정상 계약 결과와 혼동할 수 없고,
CLI의 기존 실패 폐쇄 RC 의미론도 불변이다.

## 5. Rung 3 — NaN/overflow 우회 두 경로

### 기대 변경집합

- `src/lumina_element_wide.c`: `ew_boundary_population_fraction()`이
  `n_elem > 0.0` 비교보다 먼저 `isfinite(n_elem)`을 검사한다.
- `ew_boundary_tau_add()`가 개별 tau와 기존 누적값을 선검사하고,
  `tau_all += tau` 및 조건부 `tau_boundary += tau` 각각의 직후 결과를
  검사한다. 실패 시 두 누적값을 `INFINITY` 판정 sentinel로 바꾸고 -1을
  반환한다.
- production boundary 감사가 두 helper를 직접 사용한다.
- `tests/wave32_boundary_q_seed.c`: `n_elem=NAN`과
  `DBL_MAX + DBL_MAX`를 각각 같은 production helper에 seed한다. 유한 입력
  정상 대조도 함께 고정한다.

finite 선검사/합산 직후 검사는 `src/lumina_element_wide.c` L1778–1804,
production 연결은 L2237–2259이다. 두 seed와 정상 대조는
`tests/wave32_boundary_q_seed.c` L72–90, 판정은 L124–139이다.

### 자기검증과 음성 대조

```text
n_elem_finite_fraction=0.40000000000000002
n_elem_nan_fraction=inf gate_pass=0
tau_normal_rc=0/0 tau_all=3 tau_boundary=2
tau_first_rc=0 tau_overflow_rc=-1 tau_all=inf tau_boundary=inf
opacity_fraction=inf gate_pass=0
```

유한 입력의 기존 나눗셈 및 합산 결과는 각각 0.4와 `(3,2)`로 유지된다.
`n_elem=NAN`은 더 이상 조건식의 false branch에서 0으로 정상화되지 않고,
두 유한 `DBL_MAX`의 합 overflow도 opacity fraction 0으로 축소되지 않는다.
두 seed 모두 기존 `<=` gate에서 FAIL이다.

## 6. 통합 검증

실행한 CPU/offline 검증은 다음과 같다.

- `selftest_wave32_counter_atomic`: EW-ON 1-thread PASS, EW-ON 8-thread PASS,
  EW-OFF 8-thread PASS.
- `selftest_wave32_boundary_q`: 기존 정상/seed 전 항목과 신규 두 seed PASS.
- `tests/test_wave32_seeded_defects.py`: CLI 및 Python API 계약 경로 PASS.
- `scripts/cmf_chieta_roundtrip_selftest.py --no-build`: bitwise round-trip PASS.
- 세 Python 파일 `py_compile`: PASS.
- 세 patch 순차 적용 및 최종 byte 비교: PASS.

fixture 빌드 중 `src/lumina_cmfgen.c`의 기존 indentation/unused/OpenMP pragma
경고가 재출력됐으나 빌드와 모든 판정은 RC 0이었다. A6 변경 파일에서 새 빌드
오류는 없었다.

## 7. 규율 확인

- 물리 rate/population 및 유한 입력 산출식 변경 없음.
- 신규 clamp/floor/cap 및 사후 보정 0건.
- `INFINITY`는 기존 gate를 실패시키는 비물리 판정 sentinel이며 clamp가 아님.
- 신규 모델 실행 없음.
- GPU 빌드/실행 없음.
- 커밋 없음.
- 기존 dirty worktree의 비관련 파일은 수정하지 않음.
