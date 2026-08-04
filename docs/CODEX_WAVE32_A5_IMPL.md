# Codex A5 — Wave-3.2 협소 4건 마이크로 라운드 구현 보고서

## 1. 범위와 판정

입력은 `docs/CODEX_WAVE32_C4_REVIEW.md`의 FAIL 네 좌표로 한정했다.

| A5 rung | C4 인용 좌표 | 폐합 결과 |
|---|---|---|
| 1 I/O 오류 분류 | C4 L31: open 뒤 write/flush/close 실패가 `ew_dump_failed`와 RC에 반영되지 않음 | 모든 EW artifact completion을 공통 검사하고 실패 시 commit 전 `cleanup_fail`/RC -1 |
| 2 잔여 경쟁 | C4 L43-L49: 공유 `ew_cap.target_fail++`와 불완전 selftest | 공유 capture 정수 카운터 19종을 atomic primitive로 통일, 1/8-thread 전수 fixture |
| 3 iter 계약 우회 | C4 L70-L77: `--expected-iteration 7`/generation 7 우회와 pre-damp 허용 | 명시 `--non-contract-override` 없이는 거부; 있어도 `NON-CONTRACT`, RC 2 |
| 4 NaN fail-closed | C4 L83-L88: `b[0]=NaN`이 두 `>`를 통과해 residual 0 가능 | `b[0]` 및 동류 감사 입력/중간/출력 finite 선검사, 비유한은 `INFINITY` |

C4가 이미 PASS로 판정한 OOM 전파와 신규 clamp/D6 독립성 좌표(C4 L51-L58,
L90-L96)는 수정하지 않았다.

## 2. 납품 패치와 적용 순서

패치는 A4가 적용된 이 작업 시작 시점의 워킹트리를 기준으로 한 순차 증분이다.

1. `patches/w32a5_rung1_ew_io_fail_closed.patch` (153 lines)
2. `patches/w32a5_rung2_atomic_resweep.patch` (314 lines)
3. `patches/w32a5_rung3_iter_override_contract.patch` (115 lines)
4. `patches/w32a5_rung4_nan_fail_closed.patch` (172 lines)

별도 임시 트리에 1→4를 `patch -p1`로 적용하고 최종 관련 파일을 현재 워킹트리와
`cmp`했다.

```text
sequential_patch_apply=PASS
```

## 3. rung 1 — EW artifact write/close 오류

### 기대 변경집합

- `src/lumina_element_wide.c`: `fflush`, `ferror`, `fclose`를 모두 검사하는
  `ew_finish_file()` 추가.
- identity/raw/normalized/equilibrated/solution/diagnostics/provenance/manifest/
  boundary artifact 및 runtime manifest append의 모든 close를 공통 검사로 연결.
- artifact completion 실패는 `ew_dump_failed=1`이 되고, 기존 commit보다 앞선
  I/O gate에서 `cleanup_fail`로 이동해 RC -1을 반환.
- `tests/wave32_ew_io_selftest.c`, `Makefile`: 실제 buffered-write 실패 fixture 추가.

판정 상태의 RC 의미론은 수정하지 않았다. 완료된 과학 판정의 기존 `return 0`은
그대로이고, 새 분기는 오직 artifact I/O 실패에만 걸린다.

### 자기검증과 음성 대조

정상 임시 파일과 Linux `/dev/full`을 같은 production completion 함수에 통과시켰다.

```text
good_artifact_rc=0 dev_full_write_close_rc=-1
[EW][DUMP-FAIL] write/close /dev/full: No space left on device
```

정상 artifact는 RC 0, open에는 성공하지만 flush/close에서 ENOSPC가 발생하는 음성
대조는 RC -1이다. 소스의 직접 `fclose()` 재검색 결과도 공통 함수 내부 1건뿐이다.

## 4. rung 2 — 공유 capture counter atomic 전수 재스윕

### 기대 변경집합

- `src/lumina_element_wide.c`: shell-parallel assembler에서 도달 가능한 공유
  `ew_cap` 정수 증가를 `ew_atomic_inc_int()`로 통일.
- 대상은 `target_expected`, `target_mapped`, `target_fail`, `bad_rate`, 7개
  `channel_events`, `kramers_fallback`, `continuum_deleted`, `target_fallback`,
  `nstar_cap`, `nonfinite_guard`, BF estimator/pref-J/JEQB의 총 19종이다.
- floating matrix/ledger arithmetic과 물리 rate 계산은 변경하지 않음.
- `tests/wave32_counter_atomic_selftest.c`: runtime 3종뿐 아니라 capture 19종을
  200,000회씩 검증.

소스 재스윕:

```text
rg 'ew_cap\....++' src/lumina_element_wide.c
# no matches
```

### 자기검증과 음성 대조

```text
OMP_NUM_THREADS=1: capture_counters=19 target_fail=200000 all_exact=1
OMP_NUM_THREADS=8: capture_counters=19 target_fail=200000 all_exact=1
EW OFF, 8 threads: expected=0 save_restore=0 per_ion_pin=0 topstage_IV=0
                   capture_counters=19 target_fail=200000 all_exact=1
```

마지막 행의 capture 값은 selftest가 atomic primitive를 직접 압박한 값이고, EW OFF
음성 대조의 production runtime meter 3종은 기존 계약대로 모두 0이다.

## 5. rung 3 — iteration 계약 우회 차단

### 기대 변경집합

- `scripts/cmf_chieta_check.py`: 계약 상수를 iteration=10, generation=10,
  post-damp required로 고정.
- 기대 iteration/generation 변경 또는 pre-damp 허용은
  `--non-contract-override`가 없으면 검사 전에 FAIL/RC 1.
- 명시 override로 alternate metadata의 무결성 검사를 완료해도 출력은
  `PASS`가 아닌 `NON-CONTRACT`, RC는 2.
- Python API `check_artifact()`도 `non_contract_override=False`가 기본이므로 CLI
  외 호출에서 같은 무단 우회가 불가능.
- `tests/test_wave32_seeded_defects.py`: iter=7/generation=7 무단·명시 override
  두 경로를 고정 fixture로 추가.

### 자기검증과 음성 대조

정상 계약:

```text
PASS LCMFCE01 write-read-write bitwise roundtrip
sha256=3981641ed3fa6f9bfac8425b248f546012501fac78094522816a22ab950c6d52 bytes=424
```

동일한 iter=7/generation=7 payload의 무단 우회와 명시 override:

```text
FAIL: non-contract expectation requires --non-contract-override
unauthorized_rc=1
NON-CONTRACT: iteration=7 field_generation=7 post_damp=1 bytes=424
override_rc=2
```

따라서 alternate epoch가 정상 계약 `PASS`/RC 0으로 승격되는 경로는 없다.

## 6. rung 4 — NaN fail-closed 감사

### 기대 변경집합

- `src/lumina_element_wide.c`: conservation row가 `b[0]`, 각 coefficient,
  solution, 누적 LHS와 두 residual을 finite 선검사.
- 동류 감사인 channel column sum, independent matrix/debit ledger, IV↔V
  per-target flux도 입력·누적·정규화 결과가 비유한이면 `INFINITY` 반환.
- boundary fraction/opacity 집계도 비유한 입력을 skip하지 않고 `INFINITY`로
  전파. 기존 gate의 `residual <= tolerance` 비교는 이에 대해 false가 된다.
- `tests/wave32_boundary_q_seed.c`: 요구된 `b[0]=NaN` 및 matrix/flux NaN fixture.

### 자기검증과 음성 대조

```text
row_good=0 conservation_row_seeded=0.5
b0_nan_residual=inf b0_nan_gate_pass=0
matrix_nan=inf flux_nan=inf
```

정상 보존행은 residual 0을 유지한다. coefficient 손상은 0.5, `b[0]=NaN`은
`INFINITY`이며 `<=1e-12` gate를 통과하지 못한다. matrix와 flux 동류 감사의 NaN도
동일하게 fail-closed다.

## 7. 통합 검증

실행한 CPU/offline fixture:

- `selftest_wave32_ew_io`: PASS
- `selftest_wave32_ew_rc`: `bad_env=-1`, `forced_oom=-1`, `forced_io=-1`
- `selftest_wave32_counter_atomic`: 1-thread PASS, 8-thread PASS, EW-OFF PASS
- `selftest_wave32_boundary_q`: PASS
- `scripts/cmf_chieta_roundtrip_selftest.py`: PASS
- `tests/test_wave32_seeded_defects.py`: PASS
- 네 patch 순차 적용 및 최종 byte 비교: PASS
- Python `py_compile`: PASS

`test_wave32_seeded_defects.py`의 통합 결론:

```text
bad_eta_selector_rc=1
sidecar_writer_rc=1
metadata_consumer_rc={'wrong_iter': 1, 'wrong_generation': 1, 'pre_damp': 1}
iter7_bypass_rc=1 explicit_override_rc=2
PASS seeded defects: eta/iter/generation/post-damp consumer FAIL; D6 debit ledger FAIL
```

## 8. 규율 확인

- 물리 rate, population, matrix/ledger 부동소수점 산출식 변경 없음.
- 신규 clamp/floor/cap, 사후 보정, 모델 상수 없음.
- 신규 모델 실행 없음.
- GPU 빌드/실행 없음.
- 커밋 없음.
- 기존 dirty worktree의 비관련 파일은 수정하지 않음.

