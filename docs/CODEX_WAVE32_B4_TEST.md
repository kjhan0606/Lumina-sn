# Codex B4 — Wave-3.2 A4 delta 검증

작성일: 2026-08-01 (Asia/Seoul)  
역할: 테스트 전용 (`src/` 직접 편집 없음; patch 역적용/재적용만 수행)  
대상: `patches/w32a4_rung{1..6}_*.patch`, `docs/CODEX_WAVE32_A4_IMPL.md`

## 0. 최종 판정

**전체 판정: 6/6 rung PASS.**

| rung | 판정 | 독립 실측 요약 |
|---:|---|---|
| 1 RC 의미론 | **[PASS]** | s8 S `SCOPE_FAIL`이 process RC 0으로 진행; bad env/OOM process RC 1; fixture 내부 RC `-1/-1/-1` |
| 2 atomic 카운터 | **[PASS]** | 1-thread와 8-thread 모두 `200000/200000/200000`; OFF는 `0/0/0` |
| 3 OOM 전파 | **[PASS]** | checked/legacy/solve OOM 모두 `-1`, 정상 legacy `0`; CPU·CUDA source compile 성공 |
| 4 iter 소비자 계약 | **[PASS]** | iter=10/generation=10/post-damp 양성 및 424-byte roundtrip; iter/generation/pre-damp/bad-eta 음성 RC 1 |
| 5 M_V 독립 감사 | **[PASS]** | M_V `0.017090515802328503`, Fe IV/anchor `0.993809035097`, D 개선 `57.836%`; row/route/q seed 전부 검출 |
| 6 runtime deck | **[PASS]** | A3-rung4 정정 deck을 실제 실행해 양성 `1/15/14`, 음성 `1/0/0` 재현 |

공식 frozen-cell matrix는 12/12 byte-identical이었다. COMMIT=1 s0 Fe는 실제
`commit_performed=1`이었고, 다른 실제 원소와 pair 레인은 byte 불변이었다. 신규 모델
실행과 GPU 실행은 하지 않았다. CUDA 검증은 `nvcc -c` source compile로 한정했다.

검증 산출 루트는 `/tmp/codex_wave32_b4.fqEqXU`이다. `/tmp` 보존은 보장되지 않는다.

## 1. 범위와 입력 원장

공통 archived 입력:

```bash
RUNROOT=/tmp/codex_wave32_b4.fqEqXU
FROZEN=/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59
MODEL=data/tardis_reference_toy06_19p48d_sivcaiv
```

- 신규 모델 실행: 0
- GPU 실행: 0
- archived parity59 frozen-cell CPU replay만 실행
- `src/` 직접 편집: 0
- patch 역적용/재적용 외 구현 변경: 0
- 커밋: 0
- 본 보고서 외 신규 저장소 파일: 0

A4 보고서에 등록된 patch SHA-256과 현재 파일은 6/6 일치했다.

| rung | SHA-256 |
|---:|---|
| 1 | `42a7ad8dc3e84e819ebd4fd9e62637605585449303c5a241c6923b31f10aa876` |
| 2 | `d2b17c69997dba71bbf865383db8a5da905b2ce9d1daeb990530bb2f8a1393f7` |
| 3 | `e07b84865450e2d7a9ac94afa3544d097939fbd7f50ef8fd4f1394c849a0b37d` |
| 4 | `3826653c68178a28b46415d830e133585af04961533568eade1655cb19126f14` |
| 5 | `f3afcb4397fb45862fef6d935400ee1acba9a44cbb60af2d432985cd6eb4da1f` |
| 6 | `addab87abb81170f088129dfc2c1ab0986c2aab1fc641d8c33b424eba7b100b1` |

## 2. 증분 사다리와 byte 복원 [PASS]

검증 전에 여섯 patch 헤더에서 소유 경로 24개를 추출해 mode, size, SHA-256을
기록했다. 6→1은 각 patch를 실제 역적용했고 1→6은 다시 실제 적용했다. 각 단계에서
다음을 모두 확인했다.

1. 적용 방향의 `git apply --check`
2. 적용 직후 실제 SHA 변경 경로와 해당 patch의 `---/+++` 기대 경로 집합 일치
3. 반대 방향의 `git apply --check`
4. 현재 단계 소스로 `make -B TARGET="$RUNROOT/<state>/build/lumina"`

| 동작 | 실제 변경 파일 수 | 기대 변경집합 | CPU build RC | 반대방향 check |
|---|---:|---|---:|---|
| reverse rung6 | 2 | 일치 | 0 | PASS |
| reverse rung5 | 2 | 일치 | 0 | PASS |
| reverse rung4 | 9 | 일치 | 0 | PASS |
| reverse rung3 | 9 | 일치 | 0 | PASS |
| reverse rung2 | 3 | 일치 | 0 | PASS |
| reverse rung1 | 6 | 일치 | 0 | PASS |
| apply rung1 | 6 | 일치 | 0 | PASS |
| apply rung2 | 3 | 일치 | 0 | PASS |
| apply rung3 | 9 | 일치 | 0 | PASS |
| apply rung4 | 9 | 일치 | 0 | PASS |
| apply rung5 | 2 | 일치 | 0 | PASS |
| apply rung6 | 2 | 일치 | 0 | PASS |

rung별 기대 경로는 다음과 같이 관측됐다.

- rung1: `Makefile`, `src/lumina.h`, `src/lumina_element_wide.c`,
  `src/lumina_plasma.c`, RC fixture 2개
- rung2: `Makefile`, `src/lumina_element_wide.c`, atomic fixture
- rung3: API/호출자 5개, CUDA/CMFGEN 포함, 기존 fixture/consumer 4개
- rung4: CMF writer/API/CUDA call site, fixture/env/test 5개, 신규 checker 1개
- rung5: `src/lumina_element_wide.c`, boundary seed fixture
- rung6: A3 보고서와 runtime deck script

최종 재적용 직후와 모든 배터리 종료 후 각각 최초 manifest에 대조했다. 두 번 모두
patch 소유 파일 **24/24가 mode/size/SHA-256 byte 일치**했다. 빌드 출력은 전부
`$RUNROOT`에 지정해 기존 저장소 바이너리를 덮어쓰지 않았다.

핵심 재현 명령:

```bash
# 실제 검증에서는 매 단계 전후 24-path SHA manifest도 비교했다.
for n in 6 5 4 3 2 1; do
  p=$(find patches -maxdepth 1 -name "w32a4_rung${n}_*.patch" -print)
  git apply -R --check "$p" && git apply -R "$p"
  git apply --check "$p"
  mkdir -p "$RUNROOT/reverse${n}"
  make -B TARGET="$RUNROOT/reverse${n}/lumina"
done

for n in 1 2 3 4 5 6; do
  p=$(find patches -maxdepth 1 -name "w32a4_rung${n}_*.patch" -print)
  git apply --check "$p" && git apply "$p"
  git apply -R --check "$p"
  mkdir -p "$RUNROOT/apply${n}"
  make -B TARGET="$RUNROOT/apply${n}/lumina"
done
```

## 3. 공식 12/12 byte matrix [PASS]

최종 소스로 CPU oracle을 `$RUNROOT/final/bin/bench_frozen_oracle`에 빌드한 뒤 실행했다.

```bash
python3 scripts/wave32_r1_byte_invariant.py \
  --no-build --bench "$RUNROOT/final/bin/bench_frozen_oracle" \
  --out "$RUNROOT/final/battery"
```

```text
PASS: 12 SUPER_LEVELS={0,1} x armed COMMIT=0/unarmed byte comparisons
```

| SUPER_LEVELS | shell | oracle | pair ion | pair level |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 1 | 1 |
| 0 | 8 | 1 | 1 | 1 |
| 1 | 0 | 1 | 1 | 1 |
| 1 | 8 | 1 | 1 | 1 |

대표 SHA-256은 s0 oracle
`b2c141f57638f349275143a244f68262d825abd465f5e0bbd7f2a1f7376d47b1`, s8 oracle
`f3c9b752ecd63ecd77ae38d9a61eb2a676b3d7a49c25e1a7eb22d6a56a825dde`였다.

s8 armed 두 번 모두 다음 두 판정을 기록한 뒤 process RC 0으로 끝났다. Python
driver가 `check=True`이므로 어느 process라도 nonzero였다면 matrix가 완주할 수 없다.

```text
Z=16 s=8 verdict=EW_VALID_P_ELEM_SCOPE_FAIL
Z=26 s=8 verdict=EW_PASS
```

따라서 과학 판정 FAIL과 operational failure를 분리한다는 rung1 주장이 실제 공식
배터리에서 확인됐다.

## 4. COMMIT=1 s0 Fe 격리 [PASS]

```bash
env -i PATH="$PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=0 \
  LUMINA_SUPER_LEVELS=0 LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=26 LUMINA_NLTE_ELEMENT_WIDE_SHELL=0 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 LUMINA_EW_FROZEN_COMMIT=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$RUNROOT/final/commit_s0_fe" \
  "$RUNROOT/final/bin/bench_frozen_oracle" "$FROZEN" "$MODEL" \
  "$RUNROOT/final/commit_s0_fe"
```

```text
RC=0
verdict=EW_PASS
commit_requested=1
commit_performed=1
commit_blocked_by=none
```

unarmed s0 oracle과 CSV row를 직접 비교했다.

| 분류 | 변경 행 |
|---|---:|
| Fe (`Z=26`) | 4 |
| 동일 셸 aggregate (`Z=0`) | 5 |
| 다른 실제 원소 | 0 |
| 전체 | 9/196 |

Fe 4행은 II/III/IV ion fraction과 Fe II `Gamma_photoion_total`이다. Z=0 5행은
1000/5000 Å free-free χ/η와 `cooling_ff_grid`이다. pair ion 3행과 pair level 4,198행은
각각 byte-identical이었다.

## 5. rung1 — RC 의미론 [PASS]

독립 함수 fixture:

```text
bad_env_rc=-1 forced_oom_rc=-1 forced_io_rc=-1
```

추가로 frozen executable process-level 음성을 실행했다.

```bash
# 공통 EW s0 Fe gate에 각각 추가
LUMINA_EW_FROZEN_COMMIT=bad  .../bench_frozen_oracle_rc ...
W32_FORCE_EW_OOM=1          .../bench_frozen_oracle_rc ...
```

```text
bad_env_process_rc=1
forced_oom_process_rc=1
[EW][CONFIG-FAIL] LUMINA_EW_FROZEN_COMMIT must be exactly 1
[EW][OOM] forced fixture allocation failure Z=26 s=0
```

완료된 S scope 판정은 RC 0, config/OOM은 RC nonzero이므로 요구된 분리가 성립한다.

## 6. rung2 — atomic 카운터 [PASS]

실제 production note/snapshot 함수와 OpenMP를 링크한 fixture를 실행했다.

```bash
COMMON='LUMINA_NLTE_ELEMENT_WIDE=1 LUMINA_NLTE_ELEMENT_WIDE_Z=26
LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0'

env -i PATH="$PATH" OMP_NUM_THREADS=1 $COMMON \
  "$RUNROOT/final/bin/selftest_wave32_counter_atomic"
env -i PATH="$PATH" OMP_NUM_THREADS=8 $COMMON \
  "$RUNROOT/final/bin/selftest_wave32_counter_atomic"
env -i PATH="$PATH" OMP_NUM_THREADS=8 W32_EXPECT_COUNTER_DISABLED=1 \
  "$RUNROOT/final/bin/selftest_wave32_counter_atomic"
```

```text
1 thread: expected=200000 save_restore=200000 per_ion_pin=200000 topstage_IV=200000
8 thread: expected=200000 save_restore=200000 per_ion_pin=200000 topstage_IV=200000
OFF/8:    expected=0      save_restore=0      per_ion_pin=0      topstage_IV=0
```

1/8-thread 산술이 동일하고 OFF 대조도 중립이다.

## 7. rung3 — OOM 전파 [PASS]

```bash
env -i PATH="$PATH" "$RUNROOT/final/bin/selftest_wave32_within_sl_oom"
```

```text
checked_oom_rc=-1
legacy_oom_rc=-1
solve_oom_rc=-1
normal_legacy_rc=0
```

즉 checked helper, legacy wrapper, 최상위 CPU solve가 같은 OOM을 상위로 전달한다.
12개 rung 상태 CPU build 외에 최종 CUDA translation unit도 source-only compile했다.

```bash
nvcc -O2 -arch=sm_80 -std=c++14 -Xcompiler -fopenmp \
  -DLUMINA_HAS_CUDA_BF_GEMM -c src/lumina_cuda.cu \
  -o "$RUNROOT/final/lumina_cuda.o"
```

결과는 RC 0이었다. CUDA binary/GPU 실행은 하지 않았다.

## 8. rung4 — iter 소비자 계약 [PASS]

정상 writer artifact를 `scripts/cmf_chieta_check.py`와 roundtrip reader로 검사했다.

```bash
"$RUNROOT/final/bin/selftest_cmf_chieta_dump" "$RUNROOT/final/chieta/normal.lcmfce"
python3 scripts/cmf_chieta_check.py "$RUNROOT/final/chieta/normal.lcmfce"
python3 scripts/cmf_chieta_roundtrip_selftest.py \
  --input "$RUNROOT/final/chieta/normal.lcmfce"
```

```text
checker RC=0: iteration=10 field_generation=10 post_damp=1 bytes=424
roundtrip RC=0
sha256=3981641ed3fa6f9bfac8425b248f546012501fac78094522816a22ab950c6d52
```

writer 자체의 표현 불가능 입력 7종도 모두 `rc=-1`, payload 없음이었다: 음수 χ,
불연속 radius, 비상승 frequency, nonfinite J, 음수 iteration/generation, 잘못된
post-damp 값.

소비자 음성:

| seed | writer RC | checker RC | 검출 이유 |
|---|---:|---:|---|
| bad η | 0 | 1 | eta decomposition bitwise mismatch |
| iteration 7 | 0 | 1 | expected 10 |
| generation 9 | 0 | 1 | expected 10 |
| pre-damp | 0 | 1 | post-damping required |

sidecar 경로 I/O 실패는 writer RC 1, payload 없음, `.quarantine` 존재였다. Python
checker/roundtrip/seeded test의 `py_compile`도 통과했다. writer는 metadata를 기록하고
소비자가 epoch 계약을 fail-closed 검사한다는 분리가 확인됐다.

## 9. rung5 — M_V 정본과 독립 감사 [PASS]

s0 Fe boundary artifact와 archived pair CSV에서 직접 다시 계산했다.

| 양 | 독립 실측 |
|---|---:|
| M_V after / Fe | `0.017090515802328503` |
| Fe IV / anchor(0.989) | `0.993809035097` |
| element II/III/IV | `2.7040118115e-13 / 3.2348486064e-5 / 0.982877135711` |
| D element | `0.8473612009` |
| D pair, exact pair fractions | `2.0096936222` |
| improvement, exact recompute | `57.83629945%` |
| final-matrix flux residual | `0` |
| final-row residual | `3.5965825964823073e-16` |

요청 정본 `0.0170905158 / 0.9938090 / D 57.836%`는 모두 불변이다. B3에 인쇄된
`57.83615352%`는 문서에 반올림해 적은 세 pair d (`4.8875/1.1385/0.00306`)를 다시
평균한 값이다. 현재 exact pair fraction으로 직접 재계산하면 `57.83629945%`이며,
등록된 3-decimal 정본 `57.836%`에서는 동일하다.

세 요구 seed와 추가 음수/행렬 seed를 실행했다.

```bash
"$RUNROOT/final/bin/selftest_wave32_boundary_q"
"$RUNROOT/final/bin/selftest_wave32_matrix_debit"
```

```text
row_good=0 conservation_row_seeded=0.5
flux_good=0 boundary_route_seeded=1 q_coupling_seeded=0.1388888888888889
roundoff_negative=0 raw_min=-2.2204460492503131e-16
error_bound=1.3322676295501888e-15 seeded_negative=1
baseline_residual=0 seeded_residual=0.25 gate_pass=0
```

보존행 손상, route 상실, 총 reverse가 같은 q 재분배가 모두 0이 아닌 감사 residual을
냈다. roundoff 음수는 수용하고 실제 `-1e-12` seed는 검출했다.

## 10. rung6 — A3-rung4 정정 runtime deck [PASS]

정적 `bash -n`뿐 아니라 납품 script를 실제 runtime wrapper에 연결해 실행했다.

```bash
bash scripts/wave32_runtime_counter_repro.sh \
  "$RUNROOT/final/bin/bench_runtime_counter" "$FROZEN" "$MODEL" \
  "$RUNROOT/final/runtime_deck"
```

```text
runtime_deck_rc=0
[W32-RUNTIME-COUNTERS] save_restore=1 per_ion_pin=15 topstage_IV=14
[W32-RUNTIME-COUNTERS] save_restore=1 per_ion_pin=0  topstage_IV=0
```

script의 EW master/Z/shell/commit gate가 wrapper 전에 실제로 armed됐다. 따라서 A3에서
문서 명령이 0줄을 내던 재현 deck 결함은 닫혔다.

## 11. 최종 규율 대조

| 규율 | 결과 |
|---|---|
| `src/` 수정 금지 | **PASS** — 직접 편집 0; 검증 목적 patch 역/재적용 후 24/24 byte 복원 |
| 신규 모델 실행 금지 | **PASS** — archived frozen-cell CPU replay만 실행 |
| GPU 실행 금지 | **PASS** — GPU run 0; CUDA는 source compile만 수행 |
| rung별 판정 | **PASS** — 6개 모두 `[PASS]`, 미해결 필수 항목 없음 |
| 기대 변경집합 | **PASS** — 12개 사다리 단계 모두 patch 헤더 집합과 일치 |
| 종료 byte 복원 | **PASS** — patch 소유 24/24 mode/size/SHA-256 일치 |
| 문서 형식 | **PASS** — 전체 보고서는 이 파일 하나이며 `-o` 요약 없음 |
| 최종 hygiene | **PASS** — `git diff --check`, runtime deck `bash -n` 모두 RC 0 |

최종 결론은 **A4 주장 6/6 PASS**다.
