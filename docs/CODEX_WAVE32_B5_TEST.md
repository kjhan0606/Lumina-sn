# Codex B5 — Wave-3.2 A5 스코프 검증 (최종 폐합)

## 0. 최종 판정

**A5 네 rung 모두 PASS이며, 필수 미폐합 항목은 없다.** 검증은 CPU/offline
fixture와 archived parity59 frozen-cell replay만 사용했다. 신규 모델 실행과 GPU
실행은 모두 0건이다.

| A5 rung | 판정 | 독립 재현의 핵심 결과 |
|---:|---|---|
| 1 EW I/O fail-closed | **PASS** | 정상 파일 RC 0, `/dev/full` write/flush/close RC -1 |
| 2 atomic 19종 재스윕 | **PASS** | 1/8-thread 각각 19종×200,000 exact; EW-OFF runtime 3종 0 |
| 3 iter override 계약 | **PASS** | 무단 iter=7 우회 RC 1; 명시 override `NON-CONTRACT`, RC 2 |
| 4 NaN fail-closed | **PASS** | 정상 row residual 0; `b[0]=NaN` → `inf` → gate pass 0 |

회귀 스팟도 모두 PASS다. 공식 byte matrix는 12/12, COMMIT=1 s0 Fe는
`commit_performed=1`이었으며 pair lane과 다른 실제 원소는 byte 불변이었다. M_V
정본은 `0.017090515802328503 / 0.993809035097 / D 57.836%`로 불변이다.

4→1 역적용과 1→4 재적용의 여덟 단계 모두 patch 적용 검사, 기대 변경집합,
반대 방향 검사, CPU 전체 빌드가 RC 0이었다. 검증용 patch 조작은 `/tmp` 복제본에서만
수행했다. 종료 시 복제본 7/7 및 실제 worktree 대상 7/7 파일이 최초 mode/size/SHA-256과
byte-identical했다.

## 1. 범위와 입력 원장

검증 기준:

```bash
ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
RUNROOT=/tmp/codex_wave32_b5.DSCI5I
FROZEN=/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59
MODEL=$ROOT/data/tardis_reference_toy06_19p48d_sivcaiv
```

대상 patch의 현재 SHA-256과 길이는 다음과 같다.

| rung | patch | lines | SHA-256 |
|---:|---|---:|---|
| 1 | `w32a5_rung1_ew_io_fail_closed.patch` | 153 | `4cfe6d5839e90bf858b1bf3ab85d6b4c7c8dfa18cf3331cb9efd13baffa6b06e` |
| 2 | `w32a5_rung2_atomic_resweep.patch` | 314 | `331a3aaaff4f774b46c1b0327360c4320ae11ac0ebe9059994273c445a3fe680` |
| 3 | `w32a5_rung3_iter_override_contract.patch` | 115 | `1ab339ed9093835782d3f152de9f8b5dbf18e5b19a1146ac05f3504fc2828e38` |
| 4 | `w32a5_rung4_nan_fail_closed.patch` | 172 | `f53d3a5e3f0b9beaea3e54c9c29c386920573fc08e676b2e5eb7eaae7fb774ad` |

최초 상태에서 네 patch 모두 정방향 check는 실패하고 역방향 check는 성공했다.
즉, 검증 입력 worktree는 A5 1→4가 모두 적용된 최종 상태였다.

```bash
for p in patches/w32a5_rung{1..4}_*.patch; do
  git apply --check "$p"
  git apply --reverse --check "$p"
done
```

실측은 다음 규율을 지켰다.

- `src/` 직접 편집 0건. patch 역/재적용도 실제 worktree가 아닌 `/tmp` 복제본에서 수행.
- 신규 모델 실행 0건. 공식 회귀는 archived parity59 frozen state의 CPU replay만 수행.
- GPU build/run 0건.
- 저장소 안 test binary overwrite 0건. 모든 검증 binary와 artifact는 `$RUNROOT`에 격리.
- 커밋 0건. 이 보고서 외 신규 저장소 산출물 0건.

## 2. rung 1 — EW artifact I/O fail-closed [PASS]

### 2.1 독립 양성/음성 재현

production `nlte_ew_test_dump_io()`를 링크한 CPU fixture를 별도 경로에 빌드했다.

```bash
gcc -O0 -w -std=gnu11 -D_GNU_SOURCE \
  -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
  -o "$RUNROOT/final/bin/selftest_wave32_ew_io" \
  tests/wave32_ew_io_selftest.c src/lumina_element_wide.c \
  src/lumina_plasma.c src/lumina_atomic.c -lm

"$RUNROOT/final/bin/selftest_wave32_ew_io" \
  "$RUNROOT/final/artifacts/good_ew_artifact.txt"
```

실측:

```text
process_rc=0
good_artifact_rc=0 dev_full_write_close_rc=-1
[EW][DUMP-FAIL] write/close /dev/full: No space left on device
```

정상 파일은 completion RC 0이고, open은 성공하지만 buffered write가 flush/close에서
ENOSPC가 되는 `/dev/full`은 RC -1이다. 음성이 RC 0으로 승격되는 경로는 없다.

### 2.2 정적 전수 대조

```bash
rg -n '\bfclose\(' src/lumina_element_wide.c
```

결과는 공통 `ew_finish_file()` 내부의 `fclose()` 한 건뿐이었다. 따라서 EW artifact
writer가 공통 completion 검사를 우회하는 직접 close는 남아 있지 않다.

## 3. rung 2 — capture counter 19종 atomic 재스윕 [PASS]

### 3.1 독립 1/8-thread 양성

```bash
gcc -O0 -w -std=gnu11 -D_GNU_SOURCE -DWAVE32_COUNTER_SELFTEST \
  -fopenmp -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
  -o "$RUNROOT/final/bin/selftest_wave32_counter_atomic" \
  tests/wave32_counter_atomic_selftest.c src/lumina_element_wide.c \
  src/lumina_plasma.c src/lumina_atomic.c -lm

COMMON='LUMINA_NLTE_ELEMENT_WIDE=1 LUMINA_NLTE_ELEMENT_WIDE_Z=26
LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0'

env -i PATH="$PATH" OMP_NUM_THREADS=1 $COMMON \
  "$RUNROOT/final/bin/selftest_wave32_counter_atomic"
env -i PATH="$PATH" OMP_NUM_THREADS=8 $COMMON \
  "$RUNROOT/final/bin/selftest_wave32_counter_atomic"
```

두 실행 모두 process RC 0이었다.

```text
1 thread: expected=200000 save_restore=200000 per_ion_pin=200000 topstage_IV=200000
          capture_counters=19 target_fail=200000 all_exact=1
8 thread: expected=200000 save_restore=200000 per_ion_pin=200000 topstage_IV=200000
          capture_counters=19 target_fail=200000 all_exact=1
```

19종 각각을 200,000회 압박했으며 1-thread와 8-thread에서 손실 increment가 없었다.

### 3.2 EW-OFF 음성

```bash
env -i PATH="$PATH" OMP_NUM_THREADS=8 W32_EXPECT_COUNTER_DISABLED=1 \
  "$RUNROOT/final/bin/selftest_wave32_counter_atomic"
```

```text
process_rc=0
expected=0 save_restore=0 per_ion_pin=0 topstage_IV=0
capture_counters=19 target_fail=200000 all_exact=1
```

OFF 대조의 production runtime meter 세 종은 모두 0이다. capture 19종의 200,000은
selftest primitive를 직접 압박한 별도 값이며 EW gate의 runtime 계수로 해석하지 않는다.

정적 재스윕도 통과했다.

```bash
rg 'ew_cap\.[A-Za-z0-9_]+\+\+' src/lumina_element_wide.c
# no matches
```

## 4. rung 3 — iter 무단 우회 차단 [PASS]

writer와 consumer를 저장소 binary와 분리해 빌드/실행했다.

```bash
gcc -O0 -w -std=c11 -D_GNU_SOURCE \
  -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
  -o "$RUNROOT/final/bin/selftest_cmf_chieta_dump" \
  scripts/cmf_chieta_writer_fixture.c src/lumina_cmfgen.c -lm

WRITER="$RUNROOT/final/bin/selftest_cmf_chieta_dump"
NORMAL="$RUNROOT/final/artifacts/normal.lcmfce"
ITER7="$RUNROOT/final/artifacts/iter7_generation7.lcmfce"

"$WRITER" "$NORMAL"
python3 scripts/cmf_chieta_check.py "$NORMAL"
python3 scripts/cmf_chieta_roundtrip_selftest.py --input "$NORMAL"

W32_FIXTURE_ITER=7 W32_FIXTURE_GENERATION=7 "$WRITER" "$ITER7"
python3 scripts/cmf_chieta_check.py "$ITER7" \
  --expected-iteration 7 --expected-field-generation 7
python3 scripts/cmf_chieta_check.py "$ITER7" \
  --expected-iteration 7 --expected-field-generation 7 \
  --non-contract-override
```

| 경로 | RC | 출력/판정 |
|---|---:|---|
| 정상 계약 checker | 0 | `PASS: iteration=10 field_generation=10 post_damp=1 bytes=424` |
| 정상 write-read-write | 0 | SHA `3981641...c6d52`, 424 bytes |
| iter=7/generation=7 기대값 무단 변경 | 1 | `FAIL: non-contract expectation requires --non-contract-override` |
| 같은 payload + 명시 override | 2 | `NON-CONTRACT: iteration=7 field_generation=7 post_damp=1 bytes=424` |

명시 override도 정상 계약의 `PASS`/RC 0으로 승격되지 않으므로 계약 우회가 닫혔다.

## 5. rung 4 — NaN → inf → gate 0 [PASS]

```bash
gcc -O0 -w -std=gnu11 -D_GNU_SOURCE \
  -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
  -o "$RUNROOT/final/bin/selftest_wave32_boundary_q" \
  tests/wave32_boundary_q_seed.c src/lumina_element_wide.c \
  src/lumina_plasma.c src/lumina_atomic.c -lm

"$RUNROOT/final/bin/selftest_wave32_boundary_q"
```

process RC는 0이었고 요구 좌표는 다음과 같다.

```text
row_good=0 conservation_row_seeded=0.5
b0_nan_residual=inf b0_nan_gate_pass=0
matrix_nan=inf flux_nan=inf
```

정상 보존행 residual은 0, 유한 coefficient 손상은 0.5이다. 요구 음성
`b[0]=NaN`은 residual `INFINITY`로 변환되고 `<=1e-12` gate 결과는 0이다. 동류
independent matrix와 boundary flux 감사의 NaN도 모두 `inf`였다.

추가 정상/음성 대조도 fixture RC 조건을 만족했다.

```text
flux_good=0 boundary_route_seeded=1 q_coupling_seeded=0.1388888888888889
roundoff_negative=0 raw_min=-2.2204460492503131e-16
error_bound=1.3322676295501888e-15 seeded_negative=1
```

## 6. 공식 12/12 byte matrix [PASS]

현재 최종 소스로 CPU oracle을 `$RUNROOT`에 빌드하고 공식 driver를 실행했다.

```bash
gcc -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE \
  -DLUMINA_FROZEN_ORACLE \
  -o "$RUNROOT/final/bin/bench_frozen_oracle" \
  bench_frozen_oracle.c src/lumina_plasma.c \
  src/lumina_element_wide.c src/lumina_atomic.c -lm

python3 scripts/wave32_r1_byte_invariant.py \
  --no-build --bench "$RUNROOT/final/bin/bench_frozen_oracle" \
  --out "$RUNROOT/final/byte_matrix2"
```

```text
RC=0
PASS: 12 SUPER_LEVELS={0,1} x armed COMMIT=0/unarmed byte comparisons
```

summary의 12 data row를 독립 집계했다.

| SUPER_LEVELS | shell | oracle/pair ion/pair level |
|---:|---:|---:|
| 0 | 0 | 3/3 byte_equal |
| 0 | 8 | 3/3 byte_equal |
| 1 | 0 | 3/3 byte_equal |
| 1 | 8 | 3/3 byte_equal |

대표 oracle SHA-256도 B4와 동일했다.

- s0: `b2c141f57638f349275143a244f68262d825abd465f5e0bbd7f2a1f7376d47b1`
- s8: `f3c9b752ecd63ecd77ae38d9a61eb2a676b3d7a49c25e1a7eb22d6a56a825dde`

## 7. COMMIT=1 s0 Fe 격리 [PASS]

```bash
DEST="$RUNROOT/final/commit_s0_fe"
env -i PATH="$PATH" \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=0 LUMINA_SUPER_LEVELS=0 \
  LUMINA_NLTE_ELEMENT_WIDE=1 LUMINA_NLTE_ELEMENT_WIDE_Z=26 \
  LUMINA_NLTE_ELEMENT_WIDE_SHELL=0 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 LUMINA_EW_FROZEN_COMMIT=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$DEST" \
  "$RUNROOT/final/bin/bench_frozen_oracle" "$FROZEN" "$MODEL" "$DEST"
```

```text
process_rc=0
verdict=EW_PASS
commit_requested=1
commit_performed=1
commit_blocked_by=none
```

unarmed s0 oracle의 196 data row와 직접 비교한 결과:

| 분류 | 변경 행 |
|---|---:|
| Fe (`Z=26`) | 4 |
| 같은 셸 aggregate (`Z=0`) | 5 |
| 다른 실제 원소 | 0 |
| 전체 | 9/196 |

Fe 4행은 II/III/IV `ion_fraction`과 Fe II `Gamma_photoion_total`이다. aggregate
5행은 1000/5000 Å `chi_ff`, `eta_ff` 및 `cooling_ff_grid`다.

`wave3_d8_pair_dump.py`로 생성한 pair lane은 둘 다 byte-identical이었다.

| artifact | `cmp` RC | SHA-256 |
|---|---:|---|
| pair ion fractions | 0 | `29e56721aa17a9e9d561ead771f391aa3930fc992108da82965e4ad5d17ad683` |
| pair level populations | 0 | `38154f7d4cd3fb40abed0601c44099eaf70ef3d22a1a2a7b1be9e516d9147fd6` |

## 8. M_V 정본 독립 재계산 [PASS]

COMMIT=1 s0 Fe의 boundary artifact와 새로 추출한 exact pair fractions를 읽어 문서
상수를 복사하지 않고 다시 계산했다.

| 양 | 독립 실측 |
|---|---:|
| M_V after / Fe | `0.017090515802328503` |
| Fe IV / anchor(0.989) | `0.993809035097338` |
| element II/III/IV | `2.7040118115262898e-13 / 3.2348486064135253e-5 / 0.98287713571126689` |
| pair II/III/IV | `7.6637817889940544e-7 / 0.0041958942145268845 / 0.98205090045939381` |
| D element | `0.847361200867` |
| D pair, exact fractions | `2.00969362217` |
| improvement | `57.83629945%` |
| final-matrix flux residual | `0` |
| final-row residual | `3.5965825964823073e-16` |

따라서 요구 정본 `0.017090515802328503 / 0.993809035097 / 57.836%`는 모두
불변이다. Fe IV 비율은 더 많은 자리로 `0.993809035097338`이며 등록 자리에서는
정확히 동일하다.

재계산식은 각 stage에 대해 `d_k=abs(log10(f_k/anchor_k))`,
`D=mean(d_k)`, `improvement=(D_pair-D_elem)/D_pair*100`이며 anchor는
`(9.93e-12, 0.000305, 0.989)`다.

## 9. 4-patch 증분 사다리 [PASS]

실제 worktree의 `Makefile`, `src`, `tests`, `scripts`, `patches`를 `$RUNROOT/ladder_clean`
으로 복제했다. 각 동작 직전/직후 union 7-path의 mode/size/SHA manifest를 만들고,
patch 헤더의 경로 집합과 실제 hash 변경 경로를 `cmp`했다.

각 단계의 공통 절차:

```bash
# reverse는 n=4 3 2 1, apply는 n=1 2 3 4
p=$(find patches -maxdepth 1 -name "w32a5_rung${n}_*.patch" -print)

git apply -R --check "$p" && git apply -R "$p"   # reverse 단계
git apply --check "$p"                           # 반대 방향 check

git apply --check "$p" && git apply "$p"         # apply 단계
git apply -R --check "$p"                        # 반대 방향 check

make -B CFLAGS='-O0 -w -std=c11' \
  TARGET="$RUNROOT/<state>/build/lumina"
```

| 동작 | 실제 변경 | 기대 변경 | 집합 cmp RC | 반대방향 check RC | CPU build RC |
|---|---:|---:|---:|---:|---:|
| reverse rung4 | 2 | 2 | 0 | 0 | 0 |
| reverse rung3 | 2 | 2 | 0 | 0 | 0 |
| reverse rung2 | 3 | 3 | 0 | 0 | 0 |
| reverse rung1 | 3 | 3 | 0 | 0 | 0 |
| apply rung1 | 3 | 3 | 0 | 0 | 0 |
| apply rung2 | 3 | 3 | 0 | 0 | 0 |
| apply rung3 | 2 | 2 | 0 | 0 | 0 |
| apply rung4 | 2 | 2 | 0 | 0 | 0 |

관측한 rung별 기대 변경집합:

- rung1: `Makefile`, `src/lumina_element_wide.c`, `tests/wave32_ew_io_selftest.c`
- rung2: `Makefile`, `src/lumina_element_wide.c`, `tests/wave32_counter_atomic_selftest.c`
- rung3: `scripts/cmf_chieta_check.py`, `tests/test_wave32_seeded_defects.py`
- rung4: `src/lumina_element_wide.c`, `tests/wave32_boundary_q_seed.c`

각 단계에서 기대 밖 hash 변경은 0건이었다. 모든 빌드는 CPU `lumina` target이며
GPU translation unit을 빌드하거나 실행하지 않았다.

## 10. 종료 byte 복원과 규율 대조

사다리 시작/종료 union manifest:

```bash
cmp "$RUNROOT/clean_final_start.txt" "$RUNROOT/clean_final_end.txt"
# RC 0: 7/7 mode, size, SHA-256 identical
```

실제 worktree도 검증 시작과 보고서 작성 직전의 target manifest를 대조했다.

```bash
cmp "$RUNROOT/worktree_initial_manifest.txt" \
    "$RUNROOT/worktree_pre_report_manifest.txt"
# RC 0: 7/7 mode, size, SHA-256 identical
```

| 규율 | 결과 |
|---|---|
| `src/` 수정 금지 | **PASS** — 직접 편집 0, 실제 worktree patch 조작 0 |
| rung별 판정 | **PASS** — 4/4 PASS |
| 양성/음성 독립 재현 | **PASS** — 네 rung 모두 요구 대조와 RC 확인 |
| 공식 byte matrix | **PASS** — 12/12 |
| COMMIT=1 s0 격리 | **PASS** — Fe 4 + aggregate 5, 다른 실제 원소 0, pair byte 동일 |
| M_V 정본 | **PASS** — `0.017090515802328503 / 0.993809035097 / 57.836%` |
| 증분 기대 변경집합 | **PASS** — 8/8 단계 exact set, build RC 0 |
| 신규 모델/GPU 실행 금지 | **PASS** — 신규 모델 0, GPU build/run 0 |
| 종료 byte 복원 | **PASS** — 복제본 7/7, 실제 worktree 7/7 |

최종 결론은 **A5 주장 4/4 PASS, A5 스코프 최종 폐합**이다.
