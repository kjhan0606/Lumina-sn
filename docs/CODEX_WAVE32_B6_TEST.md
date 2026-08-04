# Codex B6 — Wave-3.2 A6 스코프 검증 (최종)

작성일: 2026-08-01 (Asia/Seoul)  
대상: `patches/w32a6_rung{1..3}.patch`, `docs/CODEX_WAVE32_A6_IMPL.md`

## 0. 최종 판정

**A6 세 rung와 요구 회귀 스팟은 모두 PASS다. 필수 미폐합 항목은 없다.**

| 좌표 | 판정 | 독립 실측 |
|---|---|---|
| rung 1 `expected_outflow` 병렬 소유권 | **PASS** | EW-ON 1/8-thread ledger/inflow/debit `200000/200000/-200000`; EW-OFF runtime meter `0/0/0` |
| rung 2 Python API 계약 3상태 | **PASS** | 정상 `CONTRACT`, 무단 기대값 `REJECTED`, 명시 override `NON-CONTRACT` |
| rung 3 NaN·overflow fail-closed | **PASS** | 두 음성 모두 `inf`, 기존 gate 0; 정상 대조 `0.4`, tau `(3,2)` |
| 회귀 스팟 | **PASS** | 12/12 byte 동일, M_V 정본 불변, D6 정상 0/seed 0.25 |
| 3-patch 사다리 | **PASS** | 3→1 역적용·1→3 재적용 6/6 변경집합 일치, 종료 6/6 byte 복원 |

신규 모델 실행과 GPU build/run은 0건이다. 회귀는 기존 archived parity59 frozen
state의 CPU replay만 사용했다. 실제 worktree의 `src/`에는 patch를 적용하거나 직접
편집하지 않았고, patch 사다리는 `/tmp` 복제본에서만 수행했다.

## 1. 입력 원장과 격리

```bash
ROOT=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
RUNROOT=/tmp/codex_wave32_b6.s7JgMV
FROZEN=/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59
MODEL=$ROOT/data/tardis_reference_toy06_19p48d_sivcaiv
```

검증 산출 루트인 `$RUNROOT`의 보존은 보장되지 않는다. A6 보고서에 등록된 patch
길이와 SHA-256을 독립 재확인했다.

| rung | lines | SHA-256 |
|---:|---:|---|
| 1 | 122 | `53c218dcbe712a428bb259a087e90fd0b55be918509ce5fa1ceaf2e43d8956c2` |
| 2 | 136 | `47fab63d718ebd57bba68409316b62b61a8591a74b8268dcd097cbdd5d681d85` |
| 3 | 150 | `f212d77c38038f7259869d89a87b8bf934db6ad46d8f95f88227dcc95787e692` |

초기 worktree에서 세 patch 모두 정방향 check RC 1, 역방향 check RC 0이었다.
즉 입력은 rung 1→3이 이미 적용된 A6 최종 상태였다.

```bash
cd "$ROOT"
sha256sum patches/w32a6_rung{1..3}.patch
wc -l patches/w32a6_rung{1..3}.patch
for p in patches/w32a6_rung{1..3}.patch; do
  git apply --check "$p"; echo "forward_rc=$?"
  git apply -R --check "$p"; echo "reverse_rc=$?"
done
```

검증 binary와 artifact는 모두 `$RUNROOT`에 격리했다. 실제 수행 시 `Makefile`,
`bench_frozen_oracle.c`, `src`, `scripts`, `tests`, `patches`를
`$RUNROOT/ladder`로 복제했다.

## 2. rung 1 — 병렬 `expected_outflow`와 EW-OFF 대조 [PASS]

### 2.1 독립 재현

```bash
cd "$RUNROOT/ladder"
make -B selftest_wave32_counter_atomic

for t in 1 8; do
  env -i PATH="$PATH" OMP_NUM_THREADS="$t" \
    LUMINA_NLTE_ELEMENT_WIDE=1 \
    LUMINA_NLTE_ELEMENT_WIDE_Z=26 \
    LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 \
    LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 \
    ./selftest_wave32_counter_atomic
done

env -i PATH="$PATH" OMP_NUM_THREADS=8 \
  W32_EXPECT_COUNTER_DISABLED=1 \
  ./selftest_wave32_counter_atomic
```

세 process 모두 RC 0이었다. EW-ON 1-thread와 8-thread 출력은 동일했다.

```text
expected=200000 save_restore=200000 per_ion_pin=200000 topstage_IV=200000
capture_counters=19 target_fail=200000 all_exact=1
expected_outflow=200000 matrix_inflow=200000 matrix_debit=-200000 all_exact=1
invalid_rate_bad_rate=1 arrays_unchanged=1
```

EW-OFF/8-thread 대조:

```text
expected=0 save_restore=0 per_ion_pin=0 topstage_IV=0
capture_counters=19 target_fail=200000 all_exact=1
expected_outflow=200000 matrix_inflow=200000 matrix_debit=-200000 all_exact=1
invalid_rate_bad_rate=1 arrays_unchanged=1
```

OFF에서 0이어야 하는 값은 EW gate를 통과하는 production runtime meter 세 종이다.
`capture_counters`와 `expected_outflow`는 fixture가 production capture primitive를 직접
압박한 값이므로 OFF 대조에서도 200,000인 것이 계약에 맞다. NaN rate는 기존 guard가
`bad_rate=1`로 세고 matrix와 ledger를 바꾸지 않았다.

### 2.2 critical 영역과 감사 독립성

`nlte_ew_capture_transition()`은 off-diagonal inflow, diagonal debit,
`expected_outflow[channel][j]` ledger를 하나의 명명된 OpenMP critical 영역에서
증분한다. D6의 `ew_channel_assembly_residual()`은 capture 종료 뒤 matrix inflow와
diagonal debit를 ledger 값에 각각 대조하며, ledger를 matrix에서 재구성하지 않는다.

critical은 세 write의 동시성 소유권만 직렬화하고 감사 입력의 자료 독립성을 없애지
않는다. §6의 ledger 고정/matrix-only corruption 음성이 이를 실행 경로에서 별도로
확인한다.

## 3. rung 2 — Python API 계약 3상태 [PASS]

fixture는 정상 artifact와 iter/generation 7 artifact를 각각 생성한 뒤 Python API를
직접 import해 세 상태를 검사한다.

```bash
cd "$RUNROOT/ladder"
python3 tests/test_wave32_seeded_defects.py
python3 scripts/cmf_chieta_roundtrip_selftest.py --no-build
python3 -m py_compile \
  scripts/cmf_chieta_check.py \
  scripts/cmf_chieta_roundtrip_selftest.py \
  tests/test_wave32_seeded_defects.py
```

```text
iter7_bypass_rc=1 explicit_override_rc=2
api_contract_status=CONTRACT api_unauthorized=REJECTED api_override_status=NON-CONTRACT
PASS LCMFCE01 write-read-write bitwise roundtrip
sha256=3981641ed3fa6f9bfac8425b248f546012501fac78094522816a22ab950c6d52 bytes=424
```

API 정상 호출은 `CheckResult.contract_status == "CONTRACT"`다. 기대 iteration과
generation을 7로 바꾸고 override를 주지 않은 호출은 `CheckError`이며, 같은 payload를
`non_contract_override=True`로 명시한 호출만 `NON-CONTRACT` 결과를 반환한다. CLI도
같은 상태 필드를 소비해 무단 RC 1과 명시 override RC 2를 유지한다. 비계약 결과가
정상 `PASS`/RC 0으로 승격되는 경로는 없다.

fixture build 중 `src/lumina_cmfgen.c`의 기존 indentation/unused/OpenMP pragma
경고가 다시 출력됐지만 build와 모든 판정은 RC 0이었다. A6 변경 Python 파일의
compile 오류는 없었다.

## 4. rung 3 — NaN·overflow FAIL 시연과 정상 대조 [PASS]

```bash
cd "$RUNROOT/ladder"
make -B selftest_wave32_boundary_q
./selftest_wave32_boundary_q
```

process RC는 0이며 두 음성과 정상 대조가 한 fixture에서 함께 고정됐다.

```text
n_elem_finite_fraction=0.40000000000000002
n_elem_nan_fraction=inf gate_pass=0
tau_first_rc=0 tau_overflow_rc=-1 tau_all=inf tau_boundary=inf
opacity_fraction=inf gate_pass=0
tau_normal_rc=0/0 tau_all=3 tau_boundary=2
```

- `n_elem=NaN`은 fraction 0으로 정상화되지 않고 `INFINITY`가 되어 기존
  `<=1e-8` gate를 실패한다.
- 두 유한 `DBL_MAX` 합의 overflow는 helper RC -1과 두 누적값 `INFINITY`가 되어
  기존 opacity `<=1e-4` gate를 실패한다.
- 정상 대조 `4/10=0.4`, tau `1+2=3`, boundary tau `2`는 그대로 유지된다.

두 sentinel은 비물리 입력을 gate에서 실패시키기 위한 값이며 clamp/floor/cap이나
유한 입력 산출식 변경이 아니다.

## 5. 회귀 스팟 — 12/12 byte matrix와 M_V [PASS]

### 5.1 archived frozen-cell byte matrix

```bash
gcc -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE \
  -DLUMINA_FROZEN_ORACLE \
  -o "$RUNROOT/final/bin/bench_frozen_oracle" \
  bench_frozen_oracle.c src/lumina_plasma.c \
  src/lumina_element_wide.c src/lumina_atomic.c -lm

python3 scripts/wave32_r1_byte_invariant.py \
  --no-build \
  --bench "$RUNROOT/final/bin/bench_frozen_oracle" \
  --frozen "$FROZEN" --model "$MODEL" \
  --out "$RUNROOT/final/byte_matrix"
```

실측은 공식 driver와 같은 네 환경 조합을 cell별로 독립 실행하고 각 artifact를
`cmp`한 것이다.

| SUPER_LEVELS | shell | oracle | pair ion | pair level |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1/1 | 1/1 | 1/1 |
| 0 | 8 | 1/1 | 1/1 | 1/1 |
| 1 | 0 | 1/1 | 1/1 | 1/1 |
| 1 | 8 | 1/1 | 1/1 | 1/1 |

총 **12/12 byte-identical**이다. 대표 oracle SHA-256도 B4/B5 정본과 같다.

- s0: `b2c141f57638f349275143a244f68262d825abd465f5e0bbd7f2a1f7376d47b1`
- s8: `f3c9b752ecd63ecd77ae38d9a61eb2a676b3d7a49c25e1a7eb22d6a56a825dde`

s0 pair SHA도 ion
`29e56721aa17a9e9d561ead771f391aa3930fc992108da82965e4ad5d17ad683`,
level `38154f7d4cd3fb40abed0601c44099eaf70ef3d22a1a2a7b1be9e516d9147fd6`로
기존 정본과 동일했다.

### 5.2 M_V 정본 독립 재계산

s0 Fe candidate boundary artifact와 같은 replay에서 추출한 exact pair fraction을
읽었다. 각 stage에 `d_k=abs(log10(f_k/anchor_k))`, `D=mean(d_k)`,
`improvement=(D_pair-D_elem)/D_pair*100`을 적용했다. anchor는
`(9.93e-12, 0.000305, 0.989)`다.

```bash
BASE="$RUNROOT/final/byte_matrix/armed_super0_s0"
python3 - "$BASE" <<'PY'
import csv, math, pathlib, sys
base = pathlib.Path(sys.argv[1])
b = next(csv.DictReader(
    (base / "lumina_ew_iter0011_z26_s000_boundary_mass.csv").open()))
rows = list(csv.DictReader((base / "pair_ion_fractions.csv").open()))
n_fe = float(b["n_Fe_total"])
elem = [float(b[k]) / n_fe for k in ("sum_II", "sum_III", "sum_IV")]
pair = [float(r["ion_fraction"]) for r in rows
        if int(r["Z"]) == 26 and int(r["stage"]) in (1, 2, 3)]
anchor = [9.93e-12, 0.000305, 0.989]
d_elem = sum(abs(math.log10(v/a)) for v, a in zip(elem, anchor)) / 3
d_pair = sum(abs(math.log10(v/a)) for v, a in zip(pair, anchor)) / 3
print("M_V/Fe", float(b["M_V_after"]) / n_fe)
print("FeIV/anchor", elem[2] / anchor[2])
print("improvement", (d_pair - d_elem) / d_pair * 100)
PY
```

| 양 | 실측 |
|---|---:|
| M_V after / Fe | `0.017090515802328503` |
| Fe IV / anchor | `0.993809035097338` |
| element II/III/IV | `2.7040118115262898e-13 / 3.2348486064135253e-5 / 0.98287713571126689` |
| pair II/III/IV | `7.6637817889940544e-7 / 0.0041958942145268845 / 0.98205090045939381` |
| D element | `0.847361200867` |
| D pair | `2.00969362217` |
| improvement | `57.83629945%` |
| matrix event flux residual | `0` |
| boundary row residual | `3.5965825964823073e-16` |

따라서 등록 정본 `0.017090515802328503 / 0.993809035097 / 57.836%`는 불변이다.

## 6. D6 독립 ledger 정상/음성 [PASS]

```bash
cd "$RUNROOT/ladder"
make -B selftest_wave32_matrix_debit
./selftest_wave32_matrix_debit
```

```text
baseline_residual=0 seeded_residual=0.25 gate_pass=0
```

정상 matrix와 event ledger는 residual 0이다. 음성은 ledger `{4,0}`를 그대로 둔 채
matrix의 source diagonal debit만 `-4`에서 `-3`으로 바꾼다. residual 0.25와 gate 0이
나오므로 D6가 matrix에서 ledger를 다시 계산하거나 critical 영역과 공통 결과를 단순
복사해 비교하는 감사가 아니다. rung 1 critical 추가 뒤에도 음성이 그대로 검출되어
감사 독립성이 훼손되지 않았다.

## 7. 3-patch 역/재적용 증분과 종료 복원 [PASS]

실제 worktree가 아닌 `$RUNROOT/ladder`에서만 수행했다. union 대상은 6개다.

```text
src/lumina_element_wide.c
tests/wave32_counter_atomic_selftest.c
scripts/cmf_chieta_check.py
scripts/cmf_chieta_roundtrip_selftest.py
tests/test_wave32_seeded_defects.py
tests/wave32_boundary_q_seed.c
```

각 단계 직전 union SHA-256을 보관하고, 단계 직후 실제 변경 경로를
`git apply --numstat`의 기대 경로와 대조했다. 역적용 직후에는 정방향 `--check`,
재적용 직후에는 역방향 `--check`도 수행했다.

| 동작 | 실제/기대 변경 | 집합 일치 | 반대방향 check |
|---|---:|---:|---:|
| reverse rung3 | 2/2 | PASS | PASS |
| reverse rung2 | 3/3 | PASS | PASS |
| reverse rung1 | 2/2 | PASS | PASS |
| apply rung1 | 2/2 | PASS | PASS |
| apply rung2 | 3/3 | PASS | PASS |
| apply rung3 | 2/2 | PASS | PASS |

- rung 1: `src/lumina_element_wide.c`, counter fixture
- rung 2: checker API, roundtrip consumer, Python seeded fixture
- rung 3: `src/lumina_element_wide.c`, boundary fixture

핵심 재현 명령:

```bash
mkdir -p "$RUNROOT/ladder"
cp -a Makefile bench_frozen_oracle.c src scripts tests patches \
  "$RUNROOT/ladder/"
cd "$RUNROOT/ladder"

# 실제 검증은 각 단계 전후 union SHA 변경집합도 numstat와 비교했다.
for n in 3 2 1; do
  p="patches/w32a6_rung${n}.patch"
  git apply -R --check "$p" && git apply -R "$p"
  git apply --check "$p"
done
for n in 1 2 3; do
  p="patches/w32a6_rung${n}.patch"
  git apply --check "$p" && git apply "$p"
  git apply -R --check "$p"
done
```

사다리 시작/종료의 mode, size, SHA-256 manifest는 **6/6 byte-identical**이었다.
최종 상태에서 세 patch의 역방향 check도 3/3 성공했다. 실제 worktree 대상 6개도
검증 시작/보고서 작성 직전 manifest가 **6/6 byte-identical**하여 `src/`를 포함한
대상 구현 파일에 검증 과정의 변화가 없었다.

## 8. 규율 대조

| 규율 | 결과 |
|---|---|
| `src/` 수정 금지 | **PASS** — 실제 worktree 직접 편집/patch 조작 0; `/tmp` 복제본만 조작 |
| 신규 런 금지 | **PASS** — 신규 모델 런 0; archived frozen CPU replay와 offline fixture만 실행 |
| GPU 금지 | **PASS** — GPU build/run 0 |
| 세 독립 재현 | **PASS** — 병렬/OFF, API 3상태, NaN·overflow/정상 대조 |
| 회귀 스팟 | **PASS** — 12/12 byte matrix, M_V 정본, D6 독립 ledger |
| patch 증분/복원 | **PASS** — 6/6 단계 집합 일치, 종료 복제본·worktree 각각 6/6 byte 확인 |
| 신규 clamp/floor/cap | **없음** |
| 커밋 | **없음** |

최종 결론은 **A6 주장 3/3 PASS, Wave-3.2 A6 스코프 최종 폐합**이다.
