# Codex A4 — Wave-3.2 마감 최종 라운드 구현 보고서

작성일: 2026-08-01 KST  
입력 검토: `docs/CODEX_WAVE32_C3_REVIEW.md`, `docs/CODEX_WAVE32_B3_TEST.md`  
범위: 물리 코어 변경 없이 견고성·실패 의미론·감사 독립성 마감  
최종 판정: **6/6 rung PASS**

## 0. 결론

요청된 여섯 rung을 소스에 반영하고 독립 unified patch 사다리로 납품했다.

- 판정 실패와 실행 실패를 분리했다. `EW_PASS`, `EW_VALID_P_ELEM_SCOPE_FAIL`,
  topology/numerical/boundary gate FAIL은 진단에 기록한 뒤 operational RC 0을
  반환한다. 잘못된 config, OOM, dump/runtime-manifest I/O 실패는 RC -1이며
  상위 solve와 실행 진입점까지 fail-closed로 전파된다.
- OpenMP가 공유하는 8개 EW 카운터의 update/read를 모두 atomic화했다.
- within-SL 및 solve 계층의 `void`/폐기 경로를 `int` 반환 경로로 바꾸고 CPU,
  CMFGEN, CUDA 호출자가 오류를 확인하게 했다.
- χ/η writer는 실제 iteration/generation/post-damp metadata만 기록한다. 기본
  iteration 10 계약은 새 소비자 `scripts/cmf_chieta_check.py`가 sidecar와 payload를
  함께 fail-closed 검증한다.
- M_V 감사에서 임의 `1e-14` 음수 문턱을 제거하고 rcond 기반 오차한계, 최종
  보존행/RHS 재독, 최종 Araw IV↔V 계수·flux 재계산으로 독립성을 확보했다.
- A3 runtime-counter 재현 deck과 실행 스크립트에 EW master/Z/shell/commit gate를
  명시했다.

공식 archived frozen-cell CPU 배터리는 `SUPER_LEVELS={0,1} × shell={0,8} ×
3 artifacts`의 **12/12 byte-identical**을 통과했다. s8 S의
`EW_VALID_P_ELEM_SCOPE_FAIL`은 진단에 남았지만 배터리를 중단시키지 않았고, Fe는
`EW_PASS`였다. 신규 모델 실행이나 GPU 실행은 하지 않았다. CUDA는 source compile만
수행했다.

## 1. rung별 기대 변경집합 사전등록

검증 전에 다음 파일 경계와 불변조건을 rung별 기대 변경집합으로 고정했다. 이후
patch 파일 목록과 최종 트리를 이 표에 대조했다.

| rung | 기대 소유 파일/변경 | 사전등록 불변조건 | 음성 대조 |
|---|---|---|---|
| 1 RC 의미론 | EW entry/status, CPU solve 호출자, header, RC fixture/Make target | 완료된 과학 판정은 RC 0; config/OOM/I/O만 RC -1; S scope fail이 배터리 중단 금지 | bad gate env, forced OOM, forced I/O 모두 -1 |
| 2 atomic 카운터 | `src/lumina_element_wide.c`, atomic selftest, Make target | 단일 스레드 값 불변; 다중 스레드 누락 없음; OFF 경로 0 | EW OFF 8-thread 0/0/0 |
| 3 OOM 전파 | solve API/header와 CPU·CMFGEN·CUDA 호출자, OOM fixture | 정상 legacy RC 0; 어느 allocation 실패도 상위 RC -1 | checked/legacy/solve 3개 OOM seed |
| 4 iter 소비자 계약 | CMF writer/API/call site, fixture, Python checker/tests/env | writer가 기대값을 추측하지 않음; 기본 소비자 계약 iteration=generation=10, post-damp | wrong iter, wrong generation, pre-damp, bad η, sidecar I/O |
| 5 M_V 독립 감사 | EW 감사/ledger와 boundary fixture | M_V·d_k 불변; raw min 상시 기록; 감사 입력은 최종 행렬/RHS | 보존행 손상, route 상실, q coupling 손상, 진짜 음수 |
| 6 재현 deck | A3 문서와 독립 shell script | runtime wrapper 전에 EW gate가 실제로 armed | counter-disable deck 1/0/0 |

기대 밖 파일을 patch에 넣지 않았다. 기존 dirty worktree의 A4 비소유 변경은 수정하거나
정리하지 않았고 커밋도 만들지 않았다.

## 2. 납품 patch 사다리

| rung | patch | SHA-256 | 크기 |
|---|---|---|---:|
| 1 | `patches/w32a4_rung1_rc_semantics.patch` | `42a7ad8dc3e84e819ebd4fd9e62637605585449303c5a241c6923b31f10aa876` | 14,508 B |
| 2 | `patches/w32a4_rung2_atomic_counters.patch` | `d2b17c69997dba71bbf865383db8a5da905b2ce9d1daeb990530bb2f8a1393f7` | 7,839 B |
| 3 | `patches/w32a4_rung3_oom_propagation.patch` | `e07b84865450e2d7a9ac94afa3544d097939fbd7f50ef8fd4f1394c849a0b37d` | 22,064 B |
| 4 | `patches/w32a4_rung4_iter_consumer_contract.patch` | `3826653c68178a28b46415d830e133585af04961533568eade1655cb19126f14` | 18,224 B |
| 5 | `patches/w32a4_rung5_mv_independent_audits.patch` | `f3afcb4397fb45862fef6d935400ee1acba9a44cbb60af2d432985cd6eb4da1f` | 26,878 B |
| 6 | `patches/w32a4_rung6_runtime_deck.patch` | `addab87abb81170f088129dfc2c1ab0986c2aab1fc641d8c33b424eba7b100b1` | 1,719 B |

사다리 자기검증은 A3 기준 임시 트리에서 수행했다.

1. rung 1→6 각각 `git apply --check` 후 적용: PASS.
2. 적용 완료 트리와 현재 patch 소유 파일 전체 `diff -qr`: 차이 0.
3. rung 6→1 각각 `git apply --check -R` 후 역적용: PASS.
4. 역적용 완료 트리와 최초 A3 임시 기준 트리 `diff -qr`: 차이 0.

즉 각 patch는 앞 rung이 적용된 상태를 정확히 전제로 하며, 사다리 전체가 순방향과
역방향 모두 폐합한다.

## 3. rung 1 — RC 의미론 분리

### 3.1 구현

`nlte_element_wide_run_status()`를 추가해 operational RC와 adoption verdict를 분리했다.
내부 `ew_run_impl()`은 완료된 판정의 pass 여부를 `verdict_pass_out`에 기록하고 RC 0을
반환한다. 따라서 다음은 모두 실행 성공이다.

- `EW_PASS`
- `EW_VALID_P_ELEM_SCOPE_FAIL`
- topology/numerical/boundary gate FAIL과 진단 기록 완료

반대로 아래는 RC -1이다.

- 명시적으로 요청했지만 유효하지 않은 EW gate config
- allocation/OOM
- dump open 실패 또는 최종 runtime manifest open/close 실패
- frozen harness의 유효하지 않은 commit gate

`nlte_solve_all()`은 config 상태와 EW operational RC를 검사하며, I/O 실패 시에도
후속 writeback을 진행하지 않는다. dump I/O가 실패하면 실제 commit 전에
`cleanup_fail`로 이동한다. CUDA 진입점도 잘못된 EW config를 확인해 -1을 반환한다.

### 3.2 자기검증

작은 독립 fixture 결과:

```text
bad_env_rc=-1 forced_oom_rc=-1 forced_io_rc=-1
```

공식 배터리의 s8 S 진단은 다음과 같았고 process는 계속 진행됐다.

```text
verdict,EW_VALID_P_ELEM_SCOPE_FAIL
topology_gate_pass,1
numerical_gate_pass,1
boundary_gate_pass,0
```

같은 셸의 Fe는 세 gate=1, `EW_PASS`였다. 이로써 B3-rung2의 “과학 판정을 process
실패로 취급해 배터리가 중단”되던 문제가 닫혔다.

## 4. rung 2 — 8개 카운터 atomic화

다음 공유 증가 지점 8개에 OpenMP `atomic update`를 적용했다.

- runtime: `save_restore_calls`, `per_ion_pin_calls`, `topstage_IV_calls`
- capture/manifest: `kramers_fallback`, `continuum_deleted`,
  `bf_estimator_bins`, `bf_pref_j_bins`, `bf_jeqb_bins`

snapshot/publish read도 대응하는 `atomic read`를 사용한다. 소스에는 8 update와 8 read,
총 16개의 명시적 atomic pragma가 있다.

독립 200,000회 selftest:

```text
OMP_NUM_THREADS=1: 200000 / 200000 / 200000
OMP_NUM_THREADS=8: 200000 / 200000 / 200000
EW OFF, 8 threads: 0 / 0 / 0
```

공식 단일-thread s0 manifest의 나머지 다섯 값은
`122 / 0 / 2257221 / 874461 / 0`으로 diagnostics와 manifest가 서로 일치했다.
카운터 값 산술을 바꾸지 않고 경쟁만 제거했다.

## 5. rung 3 — OOM 상위 전파

`nlte_precompute_within_sl_frac()`과 `nlte_solve_all()`을 `int` 반환 API로 바꾸었다.
기존 wrapper의 `(void)` 폐기를 제거하고 다음 실패를 상위에 전달한다.

- within-SL fraction allocation
- CE convergence state allocation
- EW status allocation 및 EW operational failure
- overlap save allocation
- CUDA solver의 대응 allocation과 CPU authority 호출

`src/lumina_main.c`, `src/lumina_cmfgen.c` 및 CUDA의 다섯 실행 호출자는 nonzero를
확인하고 `EXIT_FAILURE` 또는 -1로 종료한다.

OOM fixture 결과:

```text
checked_oom_rc=-1
legacy_oom_rc=-1
solve_oom_rc=-1
normal_legacy_rc=0
```

CPU clean build는 PASS했다. CUDA는 다음 source compile만 수행했고 RC 0이었다.

```text
nvcc -O2 -arch=sm_80 -std=c++14 -Xcompiler -fopenmp \
  -DLUMINA_HAS_CUDA_BF_GEMM -c src/lumina_cuda.cu \
  -o /tmp/w32a4_final_lumina_cuda.o
```

GPU 실행은 하지 않았다.

## 6. rung 4 — iter 계약의 소비자측 이전

### 6.1 writer

`cmfgen_dump_frozen_chieta()`는 호출 시점의 다음 값을 그대로 기록한다.

- `iteration`
- `field_generation`
- `post_damping`

writer 내부의 expected-iteration 환경변수 항등 검사는 제거했다. writer가 거부하는 것은
음수 iteration/generation이나 0/1이 아닌 post-damp처럼 metadata 자체가 표현 불가능한
경우뿐이다. CUDA call site는 damping 이후의 실제 `it, it, 1`을 전달한다.

### 6.2 consumer

`scripts/cmf_chieta_check.py`는 기본적으로 다음 Stage-3.1 계약을 검증한다.

```text
expected iteration = 10
expected field_generation = expected iteration (기본 10)
post_damping required
```

그 외에도 schema/endian/version, 크기, 내림차순 주파수, 양의 dnu, η 분해의 bitwise
항등성, sidecar metadata, payload SHA-256을 검증한다. 하나라도 불일치하면 RC 1이다.

### 6.3 자기검증

정상 offline roundtrip:

```text
bytes=424
sha256=3981641ed3fa6f9bfac8425b248f546012501fac78094522816a22ab950c6d52
write-read-write bitwise identical
```

음성 결과:

```text
bad_eta_selector_rc=1
sidecar_writer_rc=1, payload absent, quarantine present
wrong_iter_rc=1
wrong_generation_rc=1
pre_damp_rc=1
```

실제 CUDA iteration-10 capture는 금지 규율에 따라 실행하지 않았다. 이 rung의 납품은
writer/consumer 계약과 offline fixture이며, 향후 Stage-3.1 bench는 실제 sidecar를 같은
checker에 넘기면 된다.

## 7. rung 5 — M_V 감사 독립화

### 7.1 음수 판정

raw solution은 수정하지 않고 최소값을 항상 기록한다. 음수 판정 한계는

```text
bound = (1/rcond) × (N·eps / (1 - N·eps)) × ||x||∞
```

로 계산한다. rcond가 유효하지 않거나 bound가 비유한이면 통과 권한을 주지 않고 bound
0으로 fail-closed 판정한다. 기존 임의 `-1e-14` 기준은 제거했다.

fixture는 `-DBL_EPSILON`을 roundoff로 수용하면서 raw min을 그대로 기록했고,
`-1e-12` seed는 negative 1로 검출했다.

```text
raw_min=-2.2204460492503131e-16
error_bound=1.3322676295501888e-15
roundoff_negative=0
seeded_negative=1
```

### 7.2 보존행과 flux

- 보존행 감사는 최종 `Anorm`의 row 0 계수 전부, 최종 `b[0]`, 최종 `x`를 다시 읽어
  coefficient contract, RHS contract, equation residual의 최댓값을 사용한다.
- flux 감사는 최종 `Araw`의 각 IV↔M 계수를 per-target forward/reverse ledger와 대조한
  뒤 population-weighted flux도 다시 계산한다.
- reverse ledger를 target별 배열로 유지하므로 총 reverse rate가 우연히 보존되는
  q-redistribution도 잡는다.

요청된 세 음성 fixture 결과:

```text
conservation_row_seeded=0.5
boundary_route_seeded=1
q_coupling_seeded=0.1388888888888889
```

정상 대조는 row=0, flux=0이었다.

### 7.3 공식 frozen-cell 수치

| 항목 | s0 Fe | s8 Fe |
|---|---:|---:|
| verdict | EW_PASS | EW_PASS |
| M_V after | 2832339.0221105758 | 0.00088843230918138176 |
| M_V after / Fe | 0.017090515802328503 | 4.0083573320261023e-11 |
| raw solution min | 1.9630603858808599e-08 | 0.00054413295251736458 |
| negative bound | 1.37258826119308 | 105.66282383092533 |
| negative count | 0 | 0 |
| final-row residual | 3.5965825964823073e-16 | 1.0084478824633014e-15 |
| final-matrix flux residual | 0 | 0 |

s0 `M_V after / Fe=0.017090515802328503`은 B3의 정본과 정확히 같다. flux residual은
B3의 약 `2.138e-16`에서 0으로 보이지만 물리량 변화가 아니라 감사 정의를 최종 Araw
대 독립 ledger 직접 대조로 바꾼 결과다. pair ion fractions, pair level populations,
oracle artifact가 armed/unarmed에서 byte-identical이므로 d_k와 물리 산출에는 변화가
없다.

## 8. rung 6 — runtime counter 재현 deck

`docs/CODEX_WAVE32_A3_IMPL.md`의 양·음성 명령과 새
`scripts/wave32_runtime_counter_repro.sh`에 다음 공통 gate를 넣었다.

```text
LUMINA_SUPER_LEVELS=0
LUMINA_NLTE_ELEMENT_WIDE=1
LUMINA_NLTE_ELEMENT_WIDE_Z=26
LUMINA_NLTE_ELEMENT_WIDE_SHELL=8
LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0
```

B3가 gate를 보정했을 때 얻은 양성 `1/15/14`, 음성 `1/0/0`을 그대로 재현하는 deck이다.
`bash -n`과 gate 정적 대조는 PASS했다. 새 모델 실행 금지 때문에 runtime wrapper의
archived 모델 replay를 새로 수행하지 않았고, 경쟁 제거 자체는 rung 2의 1/8-thread
독립 selftest로 검증했다.

## 9. 공식 배터리와 전체 회귀

사용한 공식 archived frozen-cell CPU 명령:

```text
python3 scripts/wave32_r1_byte_invariant.py \
  --no-build --out /tmp/w32a4_final_battery
```

결과는 header를 제외한 12행 모두 `byte_equal=1`이었다.

| SUPER_LEVELS | shell | oracle | pair ion | pair level |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 1 | 1 |
| 0 | 8 | 1 | 1 | 1 |
| 1 | 0 | 1 | 1 | 1 |
| 1 | 8 | 1 | 1 | 1 |

추가 회귀:

- default CPU `make -B`: PASS
- RC config/OOM/I/O fixture: PASS
- atomic 1-thread/8-thread/OFF fixture: PASS
- within-SL OOM/정상 fixture: PASS
- boundary q/row/route/negative fixture: PASS
- independent matrix-debit seed: PASS (`seeded_residual=0.25`, gate FAIL)
- χ/η roundtrip 및 seeded consumer defects: PASS
- Python `py_compile`: PASS
- runtime deck `bash -n`: PASS
- CUDA source compile: PASS, 실행하지 않음
- `git diff --check`: PASS

빌드 중 표시된 기존 unused/misleading-indentation/비-OpenMP compile pragma 경고는 이번
rung에서 새로 만든 오류가 아니며 build RC는 0이었다.

## 10. 규율 대조

| 규율 | 결과 |
|---|---|
| 신규 clamp/floor/cap | **0**. patch의 관련 문자열은 기존 API/진단 context뿐이며 새 수치 보정 로직 없음 |
| 물리 산출 불변 | **PASS**. M_V 정본 동일, 12/12 artifact byte 동일, d_k 경로 변경 없음 |
| 신규 모델 실행 | **없음**. archived frozen-cell CPU replay만 사용 |
| GPU 실행 | **없음**. `nvcc -c` source compile만 수행 |
| 커밋 | **없음** |
| 신설 검사 음성 대조 | **모두 있음**: RC 3종, atomic OFF, OOM 3계층, χ/η 5종, M_V 감사 4종, debit seed, counter-disable |

## 11. A4 소유 파일

### 구현/헤더

`Makefile`, `src/lumina.h`, `src/lumina_element_wide.c`,
`src/lumina_plasma.c`, `src/lumina_main.c`, `src/lumina_cuda.cu`,
`src/lumina_cmfgen.c`, `src/lumina_cmfgen.h`

### 검사/fixture

`tests/wave32_ew_rc_wrap.c`, `tests/wave32_ew_rc_selftest.c`,
`tests/wave32_counter_atomic_selftest.c`, `tests/wave32_within_sl_oom.c`,
`tests/wave32_boundary_q_seed.c`, `tests/wave32_runtime_counter_wrap.c`,
`tests/wave32_cmf_chieta_negative.c`, `tests/test_wave32_seeded_defects.py`,
`scripts/cmf_chieta_writer_fixture.c`, `scripts/cmf_chieta_roundtrip_selftest.py`,
`scripts/cmf_chieta_check.py`, `selftest_nlte_assemble.c`, `test_nlte_te.c`

### 문서/실행 deck

`docs/CODEX_WAVE32_A3_IMPL.md`, `scripts/parity59_chieta.env`,
`scripts/wave32_runtime_counter_repro.sh`, 본 보고서와 여섯 patch.

`docs/CODEX_WAVE32_A4_IMPL.md`가 전체 보고서이며 별도 `-o` 요약 파일은 만들지 않았다.
