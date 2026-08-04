# Codex B3 — Wave-3.2 A3 증분 사다리 및 COMMIT=1 격리 검증

작성일: 2026-08-01 (Asia/Seoul)  
역할: 테스트 전용 (`src/` 직접 수정 없음; 검증 목적 patch 적용/역적용만 수행)  
대상: `docs/CODEX_WAVE32_A3_IMPL.md`, `patches/w32a3_rung{1..7}_*.patch`

## 0. 최종 판정

**전체 판정: FAIL**

7→1 물리 역적용과 1→7 재적용, 각 patch의 apply/reverse-check, 최종 소유 파일
byte 복원 자체는 성공했다. COMMIT=1 s0 Fe 양성도 `commit_performed=1`로 실행됐고,
타 원소·타 셸·pair 레인은 byte 불변이었다. M_V, q_t, D 개선, seeded 결함,
OOM, JEQB, runtime 카운터와 clean build도 재현됐다.

그러나 다음 두 필수 주장 때문에 전체 PASS를 줄 수 없다.

1. **rung2 standalone 정상 배터리 FAIL**: rung7이 아직 없는 rung2 상태에서 s0 Fe는
   `EW_VALID_P_ELEM_SCOPE_FAIL`이다. rung2의 새 실패 전파가 이를 RC=1로 정직하게
   전파하여 A3에 사전등록된 정상 RC=0 및 oracle byte 불변을 만족하지 못했다.
2. **최종 12/12 matrix FAIL**: 최종 상태에서도 공식 스크립트의 s8 armed Z 목록은
   `16,26`이다. Fe는 `EW_PASS`지만 S는 계속
   `EW_VALID_P_ELEM_SCOPE_FAIL`이고, rung2가 process RC=1로 전파한다. 따라서
   `SUPER_LEVELS=0,s8`에서 oracle 작성 전에 종료되어 12/12를 완주할 수 없다.

추가로 rung4 A3 보고서에 적힌 명령은 EW gate 변수가 없어 wrapper를 호출하지 않고
counter line을 0개 생성했다. EW gate를 명시한 보정 명령에서는 요구값 1/15/14와
1/0/0을 재현했다. 구현 경로는 유효하지만 기재된 재현 deck은 FAIL이다.

| rung | 판정 | 핵심 결과 |
|---:|---|---|
| 1 | **PASS** | pre-A3와 rung1 각각 12/12, 교차 비교도 12/12 byte-identical |
| 2 | **FAIL** | 정상 s0가 scope fail을 RC=1로 전파; bad env 음성 RC=1은 PASS |
| 3 | **PASS** | default/C2/JEQB 생산 GPU 우회 0/1/1, 정상 oracle byte 불변 |
| 4 | **FAIL (재현 deck)** | A3 기재 명령 counter 0줄; EW gate 보정 시 실제 경로 1/15/14, 음성 1/0/0 |
| 5 | **UNRESOLVED** | offline writer/η/debit는 PASS; 실제 CUDA OFF/iter=10 capture는 GPU 금지로 미실행 |
| 6 | **PASS** | OOM 내부 RC=-1, 정상 oracle byte 불변 |
| 7 | **PASS** | s0/s8 EW_PASS, M_V/q_t/flux/D 및 s8 무변화 재현 |

검증 산출 루트는 `/tmp/codex_wave32_b3.Ri2yt7`이다. `/tmp` 보존은 보장되지 않는다.

## 1. 범위, 입력, 원상복원

공통 입력:

```bash
RUNROOT=/tmp/codex_wave32_b3.Ri2yt7
FROZEN=/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59
MODEL=data/tardis_reference_toy06_19p48d_sivcaiv
```

- 신규 모델 실행: 0
- GPU 실행: 0
- archived parity59 frozen CPU replay만 실행
- `src/` 직접 편집: 0
- patch 검증 외 저장소 변경: 이 보고서 1개
- 커밋: 0

patch SHA-256은 A3 원장과 7/7 일치했다.

| rung | SHA-256 |
|---:|---|
| 1 | `8a32a405a64c52e375af71a8b74cb01ff310375af986261e6d123ad85946d4aa` |
| 2 | `f376230f37f6f40745a3de49e8b1ceaea170268c162610e2f0264e53b9cc8dc6` |
| 3 | `1023ce4ee9c503863193bfc809b418eac909fe2d078a7e4bf137179dcb026e44` |
| 4 | `63a1d77513a16f471e235ec2b15d950b903dedd949027981be0d7d02bd9b55b5` |
| 5 | `d4a285331520e3002cbe815b50f321e3b25ec5526d4929216cfa1ef8a291d7fd` |
| 6 | `6ecdbb124c744179db61d22ef62041a3ebece0d7936d72fcdb0eb880655468d6` |
| 7 | `bd9258939c22683907904cea6a073e136728b6a78bd4337a4728f74fbcb74d04` |

역적용 본검증:

```bash
for n in 7 6 5 4 3 2 1; do
  p=$(find patches -maxdepth 1 -name "w32a3_rung${n}_*.patch" -print)
  git apply -R --check "$p"
  git apply -R "$p"
  git apply --check "$p"
done
```

7개 모두 reverse-check, reverse-apply, forward-check-after-reverse가 PASS했다.
이후 1→7 각 단계에서도 `git apply --check`, 실제 apply,
`git apply -R --check`가 모두 PASS했다.

최종 rung7 상태에서 patch 소유 17개 파일이 최초 워킹트리와 17/17 byte 일치했다.
clean build가 덮어쓴 기존 바이너리 6개까지 최초 백업으로 복원한 뒤 총 23개가
23/23 byte 일치했다. 증거는 `final_owned_cmp.tsv`, `restoration_cmp.tsv`다.

## 2. 증분 사다리 상세

### 2.1 rung1 — 투영 빌더 [PASS]

pre-A3 기저와 rung1에서 각각 다음을 실행했다.

```bash
make -B
make -B bench_frozen_oracle
python3 scripts/wave32_r1_byte_invariant.py --no-build \
  --out "$RUNROOT/pre_a3/matrix_final"
python3 scripts/wave32_r1_byte_invariant.py --no-build \
  --out "$RUNROOT/rung1/matrix"
```

양쪽 모두 다음을 냈다.

```text
PASS: 12 SUPER_LEVELS={0,1} x armed COMMIT=0/unarmed byte comparisons
```

추가로 동일 `(SUPER_LEVELS,shell,artifact)`끼리 pre-A3와 rung1을 비교했다.
oracle, pair-ion, pair-level 총 12/12가 다시 byte-identical이었다
(`rung1/pre_cross_cmp.csv`). 즉 구조 통합 외 정상 산출 변화는 없었다.

### 2.2 rung2 — harness 실패 전파 [FAIL]

빌드는 RC=0이었다. A3의 정상 s0 명령을 그대로 실행하면:

```bash
env -i PATH="$PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=0 \
  LUMINA_SUPER_LEVELS=0 LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=26 LUMINA_NLTE_ELEMENT_WIDE_SHELL=0 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$RUNROOT/rung2/normal" \
  ./bench_frozen_oracle "$FROZEN" "$MODEL" "$RUNROOT/rung2/normal"
```

결과는 RC=1이었다.

```text
[EW] Z=26 s=0 N=303 verdict=EW_VALID_P_ELEM_SCOPE_FAIL ...
s0: element-wide observer failed
```

rung1에서는 같은 observer 음수가 폐기되어 RC=0과 oracle이 생성됐다. rung2에서는
oracle 작성 전에 종료됐다. patch가 `bench_frozen_oracle.c`만 바꾸므로 이 변화는
rung2 단독 귀속이다. 실패 전파 자체는 정직하지만 A3가 사전등록한 “정상 RC=0,
성공 파일 불변”과 양립하지 않는다.

오류 문자열 음성은 의도대로 PASS했다.

```bash
env -i PATH="$PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=0 \
  LUMINA_NLTE_ELEMENT_WIDE=1 LUMINA_NLTE_ELEMENT_WIDE_Z=26 \
  LUMINA_NLTE_ELEMENT_WIDE_SHELL=0 LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 \
  LUMINA_EW_FROZEN_COMMIT=bad \
  ./bench_frozen_oracle "$FROZEN" "$MODEL" "$RUNROOT/rung2/bad"
```

```text
RC=1
[EW][CONFIG-FAIL] LUMINA_EW_FROZEN_COMMIT must be exactly 1
s0: element-wide observer failed
```

### 2.3 rung3 — bf field/GPU 우회 [PASS]

CPU selftest를 현재 rung3 source와 링크해 세 독립 프로세스로 실행했다.

```bash
env -i PATH="$PATH" "$RUNROOT/rung3/wave32_bf_policy_selftest"
env -i PATH="$PATH" LUMINA_C2_MATRIX_BF=1 \
  "$RUNROOT/rung3/wave32_bf_policy_selftest"
env -i PATH="$PATH" LUMINA_NLTE_BF_JEQB=1 \
  "$RUNROOT/rung3/wave32_bf_policy_selftest"
```

```text
default source=0 use_gpu=1 gpu_bypass=0 production_bypass=0
C2      source=1 use_gpu=0 gpu_bypass=1 production_bypass=1
JEQB    source=2 use_gpu=0 gpu_bypass=1 production_bypass=1
```

unarmed s0 replay RC=0, rung1 oracle와 `cmp` RC=0이었다.

### 2.4 rung4 — runtime 카운터 [FAIL: A3 재현 명령]

A3 보고서에 적힌 양성 명령은 EW master/Z/shell gate가 없다.

```bash
env -i PATH="$PATH" W32_RUNTIME_COUNTER_ONLY=1 \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 \
  LUMINA_NLTE_ION_LOCK=1 LUMINA_TOPSTAGE_IV=1 \
  "$RUNTIME_WRAPPED_BENCH" "$FROZEN" "$MODEL" "$OUT"
```

RC=0이지만 `[W32-RUNTIME-COUNTERS]` 출력은 **0줄**이었다. wrapper는
`nlte_element_wide_matches()` 뒤에서만 호출되므로 이 명령으로 1/15/14를 재현할
수 없다.

EW gate를 명시한 보정 명령에서는 실제 owner path가 통과했다.

```bash
COMMON='LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 LUMINA_SUPER_LEVELS=0
LUMINA_NLTE_ELEMENT_WIDE=1 LUMINA_NLTE_ELEMENT_WIDE_Z=26
LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0'

env -i PATH="$PATH" W32_RUNTIME_COUNTER_ONLY=1 $COMMON \
  LUMINA_NLTE_ION_LOCK=1 LUMINA_TOPSTAGE_IV=1 \
  "$RUNTIME_WRAPPED_BENCH" "$FROZEN" "$MODEL" "$POS"
env -i PATH="$PATH" W32_RUNTIME_COUNTER_ONLY=1 \
  W32_RUNTIME_DISABLE_PIN_TOPSTAGE=1 $COMMON \
  "$RUNTIME_WRAPPED_BENCH" "$FROZEN" "$MODEL" "$NEG"
```

```text
positive save_restore=1 per_ion_pin=15 topstage_IV=14
negative save_restore=1 per_ion_pin=0  topstage_IV=0
```

patch 기능은 PASS지만, 검증 대상인 A3 문서의 재현 명령과 주장 조합은 FAIL로
판정했다.

### 2.5 rung5 — honest writer [UNRESOLVED; offline PASS]

```bash
make -B selftest_cmf_chieta_dump selftest_wave32_matrix_debit
./selftest_wave32_matrix_debit
python3 scripts/cmf_chieta_roundtrip_selftest.py --no-build
python3 tests/test_wave32_seeded_defects.py
```

```text
normal payload bytes=424
sha256=3981641ed3fa6f9bfac8425b248f546012501fac78094522816a22ab950c6d52
bad_eta_selector_rc=1
sidecar_writer_rc=1; payload absent; .quarantine present
baseline_residual=0 seeded_residual=0.25 gate_pass=0
```

offline 계약은 전부 PASS다. 실제 CUDA producer OFF 중립성과 parity59 producer
iter=10 capture는 신규 GPU 실행 금지 때문에 **UNRESOLVED-until-capture**다.

### 2.6 rung6 — within-SL OOM [PASS]

```bash
make -B selftest_wave32_within_sl_oom bench_frozen_oracle
./selftest_wave32_within_sl_oom
```

```text
within_sl_oom_rc=-1
[NLTE][OOM] within-super-level partition allocation failed
```

정상 unarmed s0 replay는 RC=0이고 rung3 oracle과 byte-identical이었다.

### 2.7 rung7 — Fe V 경계질량 [PASS]

```bash
make -B bench_frozen_oracle selftest_wave32_boundary_q
./selftest_wave32_boundary_q

# shell=0과 8에 대해 각각
env -i PATH="$PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL="$SHELL" \
  LUMINA_SUPER_LEVELS=0 LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=26 \
  LUMINA_NLTE_ELEMENT_WIDE_SHELL="$SHELL" \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$OUT" \
  ./bench_frozen_oracle "$FROZEN" "$MODEL" "$OUT"
```

q_t 음성 fixture:

```text
good=1 sum=1 negative=0 nonfinite=0 bad_sum=0
matrix_good=0 bad_debit=0.25 bad_target=0.25
```

s0/s8 모두 RC=0, rank 304/304, `EW_PASS`, 세 gate=1이었다.

## 3. COMMIT=1 양성 격리

### 3.1 s0 Fe 양성 [PASS]

```bash
env -i PATH="$PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=0 \
  LUMINA_SUPER_LEVELS=0 LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=26 LUMINA_NLTE_ELEMENT_WIDE_SHELL=0 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 LUMINA_EW_FROZEN_COMMIT=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$RUNROOT/commit/s0_commit" \
  ./bench_frozen_oracle "$FROZEN" "$MODEL" "$RUNROOT/commit/s0_commit"
```

```text
RC=0
verdict=EW_PASS
commit_requested=1
commit_performed=1
commit_blocked_by=none
```

unarmed 대조와 oracle 196개 data row를 원문 CSV row 단위로 비교했다.

| 분류 | 변경 행 |
|---|---:|
| Fe (`Z=26`) | 4 |
| 다른 실제 원소 (`Z`가 0,26 외) | 0 |
| 같은 s0 셸 aggregate (`Z=0`, free-free) | 5 |
| 전체 | 9/196 |

Fe 변경은 II/III/IV ion fraction 3행과 Fe II `Gamma_photoion_total` 1행이다.
Z=0 변경 5행은 `chi_ff/eta_ff` 1000/5000 Å와 `cooling_ff_grid`로, Fe writeback을
소비한 동일 파일럿 셸 집계다. 이를 타 원소 변화로 숨기지 않았으며 pilot downstream
산출로 분류했다. 다른 실제 원소 행은 전부 byte-identical이었다.

s0 oracle SHA-256:

```text
unarmed b2c141f57638f349275143a244f68262d825abd465f5e0bbd7f2a1f7376d47b1
commit  575983d1bc9a78fefd6469a3ef73f89e90f5b5074fff486ff0816ee6d3b297a6
```

### 3.2 pair 레인과 s8 음성 [PASS]

`wave3_d8_pair_dump.py`로 s0와 s8 pair 파일을 각각 생성해 직접 `cmp`했다.

| shell | artifact | byte | SHA-256 |
|---:|---|---|---|
| 0 | pair ion | 동일 | `29e56721aa17a9e9d561ead771f391aa3930fc992108da82965e4ad5d17ad683` |
| 0 | pair level | 동일 | `38154f7d4cd3fb40abed0601c44099eaf70ef3d22a1a2a7b1be9e516d9147fd6` |
| 8 | pair ion (Z26) | 동일 | `fe2673c665677c85caee16749f823d2daa062b5e176962289a4c3df430f5c7c1` |
| 8 | pair level (Z26) | 동일 | `2bb83f9c074a51dd4d7f5a46cf257d00bc289e647f31cab57457375c15ad2f19` |

s8 자체를 target으로 잡으면 Fe s8도 현재 `EW_PASS`이므로 commit 가능한 양성이다.
따라서 올바른 음성은 Fe/s0 selector를 유지한 채 `ONLY_SHELL=8`만 replay하는 것이다.
이 off-target s8 oracle은 unarmed와 byte-identical이었다.

```text
s8 unarmed/off-target bytes=28391
sha256=f3c9b752ecd63ecd77ae38d9a61eb2a676b3d7a49c25e1a7eb22d6a56a825dde
```

## 4. M_V 및 q_t 물리 검증

### 4.1 s0

boundary artifact의 직접 실측값:

| 양 | 값 |
|---|---:|
| Fe total | `165725777.6693131` |
| M_V before / Fe | `0.013749080149417918` |
| M_V after / Fe | `0.017090515802328503` |
| Fe II / III / IV | `2.7040118115e-13 / 3.2348486064e-5 / 0.982877135711` |
| Fe IV / anchor(0.989) | `0.993809035097` |
| q count / sum | `200 / 0.99999999999999967` |
| q min / max | `1.9630230708643974e-12 / 0.19590372281504895` |
| q checksum | `ce935d7987d1cebb` |
| forward / reverse | `68041.773126977947 / 68041.773126977889` |
| matrix-event flux residual | `2.1386737301870922e-16` |
| conservation residual | `3.5965825964820573e-16` |

조건부 anchor `(9.93e-12, 0.000305, 0.989)`에 대한 실측 d는:

```text
element = 1.5649406642 / 0.9744458792 / 0.00269705924
pair    = 4.8875       / 1.1385       / 0.00306
D_elem=0.8473612009, D_pair=2.0096866667
improvement=57.83615352%
```

전 성분 `d_elem < d_pair`, M_V≈0.01709, Fe IV/anchor≈0.9938, D≈57.84%를
모두 재현했다.

### 4.2 s8

```text
M_V after / Fe = 4.0083573320261023e-11
q sum = 0.99999999999999889
q min/max = 2.5233314490268101e-20 / 0.28730511531397634
q checksum = d8a58f4ed38136ec
matrix-event flux residual = 1.302857554127672e-15
pre/post II/III/IV max_abs_change = 2.4261370690226158e-11
```

A3의 `2.43e-11` 무변화 주장을 재현했다.

## 5. 최종 상태 배터리

### 5.1 공식 12/12 matrix [FAIL]

```bash
python3 scripts/wave32_r1_byte_invariant.py --no-build \
  --out "$RUNROOT/final/matrix"
```

`SUPER_LEVELS=0,s0` armed/unarmed은 완료됐으나 다음 s8 armed에서 RC=1이었다.

```text
[EW] Z=16 s=8 N=303 verdict=EW_VALID_P_ELEM_SCOPE_FAIL ...
[EW] Z=26 s=8 N=304 verdict=EW_PASS ...
s8: element-wide observer failed
```

S diagnostics:

```text
boundary_process_coverage=0
boundary_gate_pass=0
commit_performed=0
verdict=EW_VALID_P_ELEM_SCOPE_FAIL
```

Fe-only 성공을 12/12 PASS로 대체하지 않았다. 공식 스크립트가 Z=16,26을 요구하고,
rung2 계약이 어느 원소든 음수 반환을 process RC로 전파하므로 이 FAIL이 정본이다.

### 5.2 harness·카운터·JEQB [PASS]

최종 소스로 wrapper를 다시 링크한 결과:

```text
bad commit env RC=1
runtime positive = 1/15/14
runtime negative = 1/0/0
```

Fe s8 bf 카운터:

| 대조 | RC | Kramers | estimator | pref·J | JEQB | verdict |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0 | 122 | 1,936,628 | 1,195,054 | 0 | EW_PASS |
| no_kramers | 1 | 0 | 1,882,991 | 1,161,748 | 0 | EW_FAIL_SHADOW |
| no_pref | 0 | 122 | 3,131,682 | 0 | 0 | EW_PASS |
| JEQB | 0 | 122 | 0 | 3,131,682 | 3,131,682 | EW_PASS |

deletion seed는 `continuum_deletion=121`, `assembled_target_fail=7992`, RC=1로
fail-closed했다.

### 5.3 D3 음성 [PASS: fail-closed]

`grid_mismatch`:

```text
RC=1
continuum_coverage=4198/4198
kramers_continuum_count=4198
kramers_fallback_firing_count=4198
boundary route_count=0, valid_route_count=0
verdict=EW_FAIL_SHADOW
```

pre-rung7 D3와 달리 최종 rung7은 경계질량 producer에 CMFGEN σ 200/200을
요구한다. 따라서 모든 기존 continuum의 Kramers 전환은 계수되지만 경계 producer
coverage가 0이 되어 fail-closed하는 것이 최종 계약에 맞다.

`nstar_overflow`:

```text
RC=1
nstar_cap_firing_count=4188
nonfinite_guard_firing_count=0
verdict=EW_FAIL_SHADOW
```

값을 cap/floor로 통과시키지 않았다.

### 5.4 seeded/OOM/writer 재실행 [PASS/UNRESOLVED]

최종 상태에서 다시 얻은 값:

```text
matrix debit: baseline=0, seeded=0.25, gate=0
within-SL OOM: inner RC=-1
boundary q: good=1, negative=0, nonfinite=0, bad_sum=0
writer bad eta RC=1; sidecar failure RC=1 + quarantine
normal writer 424 bytes, SHA-256 398164...d52
```

CUDA 실제 producer 항목만 계속 UNRESOLVED다.

### 5.5 clean build [PASS]

```bash
make clean
make
```

```text
make clean RC=0
make RC=0
warning=60
compiler error=0
```

빌드가 덮어쓴 기존 바이너리는 최초 byte 백업으로 복원했고, 최종 23/23 byte
일치를 확인했다.

## 6. 결론

A3의 핵심 Fe 경계질량 물리와 COMMIT=1 격리는 실증됐다. 특히 성공 commit이 실제
Fe/s0 값을 바꾸면서 다른 실제 원소, off-target s8, pair 레인을 건드리지 않는다는
R1 계약 후반은 PASS다.

하지만 A3가 하나의 완전한 증분 사다리 및 최종 회귀로 PASS하려면 최소한 다음이
해결되어야 한다.

1. rung2 시점의 정상 자기검증 입력을 실제 `EW_PASS` 입력으로 정의하거나,
   scope-fail을 정상으로 부르는 A3 주장을 수정할 것.
2. 최종 12/12에서 s8 S의 boundary scope를 닫거나, 공식 matrix의 Z=16 포함 계약을
   변경하는 별도 승인 없이 Fe-only PASS로 축소하지 말 것.
3. rung4 재현 명령에 EW master/Z/shell gate를 명시할 것.
4. 허용된 GPU capture에서 R7 실제 producer OFF 중립성과 iter=10을 별도 폐합할 것.

현재 상태에서는 1·2가 필수 배터리를 직접 실패시키므로 전체 판정은 **FAIL**이다.
