# Codex B — Wave-3.2 수리 배치 검증 보고서

작성일: 2026-08-01  
역할: B 테스트 전용 (`src/` 수정 없음, 신규 모델/GPU 실행 없음)  
정본: `docs/WAVE32_REPAIR_BATCH_SPEC_2026-08-01.md`  
구현자 요약: `docs/CODEX_WAVE32_A_IMPL.md`

## 0. 최종 판정

**전체 판정: FAIL**

차단 사유는 세 가지다.

1. R1의 armed `COMMIT=0` 대 unarmed는 s0·s8의 6/6 산출물이 byte-identical로 PASS했지만, `COMMIT=1` frozen replay는 실제 commit 경로를 실행하지 않는다. frozen entry point가 `commit_requested=0`을 하드코딩하며(`src/lumina_element_wide.c:1505-1510`), 현 후보 verdict도 `EW_VALID_P_ELEM_SCOPE_FAIL`이라 production의 `if(pass && commit_requested)` 조건을 만족하지 않는다. 관측된 변화 집합은 0개였고, “파일럿만 변경 + off-target byte-identical” 계약을 실증하지 못했다.
2. R5 열합 게이트와 hot/cold 자체는 정상 및 seeded-defect 음성 대조를 통과했다. 그러나 `kramers_fallback_firing_count`, `continuum_deletion_firing_count`와 bf field-source 카운터가 manifest에 없으며, `save_restore_calls`, `per_ion_pin_calls`, `topstage_IV_calls`는 선언/출력 외 증가 배선이 없다. 공통 acceptance §6과 R5/D7을 만족하지 못한다.
3. 기본 `make clean && make`는 `M_PI` 미정의 7건으로 종료 코드 2를 반환했다. 성공한 frozen-oracle 타깃에도 경고 34줄이 있다.

| 항목 | 판정 | 요약 |
|---|---|---|
| R1 armed `COMMIT=0` vs unarmed | PASS | s0·s8, 6/6 byte-identical |
| R1 `COMMIT=1` pilot/off-target | **FAIL / UNRESOLVED** | 환경값은 1이나 harness가 commit 인자를 0으로 전달; 실제 변화 0 |
| R1 seeded difference | PASS | 1-byte 크기 차이를 R1 본 비교 코드가 RC=1로 검출 |
| R2/D8 pair baseline | PASS | R1의 pair ion/full-level 파일이 byte-identical |
| R3 JEQB falsifier | PASS | Fe estimator 0, JEQB/Planck 3,081,675 bins |
| R3 parity provenance | PASS | R5 bf 변화 8 key 외 16,473 key byte-identical; pair 해 byte-identical |
| R5 Kramers 내용 | PASS | Fe II 122준위, 삭제 0, Γ `+0.0158125051 dex` |
| R5 hot/cold | PASS | required=1, rebuilds=2, matrix byte-identical=1 |
| R5 열합 gate 음성 대조 | PASS | rad_bb ledger +1% 주입 → residual `9.90099e-3`, gate FAIL |
| R5 manifest/counter | **FAIL** | 필수 카운터 5개 manifest 누락; runtime `*_calls` 3개 미배선 |
| s0·s8 EW 기대 대조 | PASS(기대 대조) | R5에 따른 소폭 이동만 관측, 아래 원수치 기재 |
| 신규 clamp/floor/cap | PASS(범위 한정) | EW runtime legacy guard firing 0; Kramers는 pair 동등 fallback |
| clean build | **FAIL** | RC=2, error 7, warning 61 |

임시 산출물 루트는 `/tmp/wave32_b_test.dFOIMr`이다. `/tmp` 산출물은 영구 보존을 보장하지 않는다.

## 1. 작업트리와 규율

검증 시작 시점부터 저장소는 구현자 및 기존 사용자 변경으로 광범위하게 dirty였다. 기존 변경은 되돌리지 않았다. 이 B 작업에서 `src/`는 수정하지 않았다. seeded ledger 검사는 `/tmp/wave32_b_test.dFOIMr/lumina_element_wide_seed_ledger.c` 복사본만 기계적으로 한 줄 변경해 별도 binary로 빌드했다.

- 신규 모델 런: 0
- GPU 런: 0
- frozen parity59 CPU replay만 사용
- 소스 commit/revert: 0
- 보고서 외 저장소 파일 추가/수정: 0

## 2. R1 byte 불변식

### 2.1 정본 스크립트 완주

명령:

```bash
RUNROOT=/tmp/wave32_b_test.dFOIMr
make -B bench_frozen_oracle
python3 scripts/wave32_r1_byte_invariant.py \
  --no-build --out "$RUNROOT/r1_exact"
```

종료 코드 0, 출력:

```text
PASS: 6 armed COMMIT=0/unarmed byte comparisons
summary=/tmp/wave32_b_test.dFOIMr/r1_exact/byte_invariant_summary.csv
```

| shell | artifact | bytes | SHA-256 | byte equal |
|---:|---|---:|---|---:|
| 0 | `lumina_oracle_cell_s0.csv` | 27,488 | `7a79f4f345d5c8500b48530a7f859bbe05a57771f43e52f983639ef9a0839381` | 1 |
| 0 | `pair_ion_fractions.csv` | 290 | `29e56721aa17a9e9d561ead771f391aa3930fc992108da82965e4ad5d17ad683` | 1 |
| 0 | `pair_level_populations.csv` | 272,071 | `38154f7d4cd3fb40abed0601c44099eaf70ef3d22a1a2a7b1be9e516d9147fd6` | 1 |
| 8 | `lumina_oracle_cell_s8.csv` | 28,131 | `2ee175c6be4a0ab6ae07034722180c6312f0a6e55e1009fcdda23b72310693eb` | 1 |
| 8 | `pair_ion_fractions.csv` | 527 | `c2b977bd42f45b0d9763c9d70df0848b822059b3270212381dc723f30c6bd337` | 1 |
| 8 | `pair_level_populations.csv` | 316,974 | `13662c9881f209d64e4a4dd60807daf01209cab4b118cfad9334caa851c4472e` | 1 |

pair-wise numeric comparison도 s0 4,198 full-level행, s8 4,902행에서 max absolute/relative/dex difference가 전부 0이었다. 이 결과로 D8의 별도 pair-baseline 오염 분기는 관측되지 않았다.

### 2.2 `COMMIT=1` frozen replay

명령:

```bash
RUNROOT=/tmp/wave32_b_test.dFOIMr
mkdir -p "$RUNROOT/commit1_s8"
env -i PATH="$PATH" \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 \
  LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=16,26 \
  LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$RUNROOT/commit1_s8" \
  ./bench_frozen_oracle \
  /gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59 \
  data/tardis_reference_toy06_19p48d_sivcaiv \
  "$RUNROOT/commit1_s8"
```

프로세스 RC는 0이고 banner는 `commit=1`이었다. 그러나 결과는 다음과 같다.

- S와 Fe 모두 `EW_VALID_P_ELEM_SCOPE_FAIL`.
- `COMMIT=1` 대 `COMMIT=0` EW CSV: 16/16 byte-identical.
- `COMMIT=1` 대 unarmed 권위 산출물: oracle, pair ion fractions, pair level populations 3/3 byte-identical.
- pilot 변화도 0, off-target 변화도 0.

이는 성공적인 commit 격리가 아니다. frozen harness가 호출하는 `nlte_element_wide_run_labeled()`가 마지막 인자를 항상 0으로 전달한다. 또한 실제 production commit 문장도 `pass && commit_requested`일 때만 쓰는데, 현재 boundary gate가 실패하여 `pass=0`이다. 따라서 파일럿 변경 및 전 셸/전 원소 off-target 불변을 동시에 측정할 수 없다. R1 acceptance 1의 두 번째 절반은 **FAIL / UNRESOLVED**다.

### 2.3 R1 음성 대조

정상 생성된 unarmed s0 oracle의 복사본 크기를 27,488→27,489 bytes로 1 byte 늘린 뒤, `scripts/wave32_r1_byte_invariant.py` 모듈의 `main()`과 실제 byte 비교 코드는 그대로 사용하고 `run_cell`만 준비된 frozen 디렉터리를 반환하도록 대체했다.

```bash
RUNROOT=/tmp/wave32_b_test.dFOIMr
truncate -s 27489 \
  "$RUNROOT/r1_seed/unarmed_s0/lumina_oracle_cell_s0.csv"
RUNROOT="$RUNROOT" python3 -c '
import importlib.util, os
from pathlib import Path
from types import SimpleNamespace
root=Path(os.environ["RUNROOT"])
spec=importlib.util.spec_from_file_location("r1", "scripts/wave32_r1_byte_invariant.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.arguments=lambda: SimpleNamespace(
    no_build=True,
    frozen=Path("/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59"),
    model=Path("data/tardis_reference_toy06_19p48d_sivcaiv"),
    bench=Path("bench_frozen_oracle"), out=root/"r1_seed")
m.run_cell=lambda args,out,shell,armed: root/"r1_seed"/(
    f"armed_s{shell}" if armed else f"unarmed_s{shell}")
raise SystemExit(m.main())'
```

예상 실패 RC=1과 다음 메시지를 얻었다.

```text
FAIL s0 lumina_oracle_cell_s0.csv: byte diff
```

R1 비교기의 음성 대조는 PASS다.

## 3. R3 field-source 공유와 parity 회귀

### 3.1 JEQB falsifier

정상 armed 명령에 `LUMINA_NLTE_BF_JEQB=1`을 추가해 s8을 replay했다.

```bash
env -i PATH="$PATH" \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 \
  LUMINA_NLTE_BF_JEQB=1 \
  LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=16,26 \
  LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$RUNROOT/jeqb_s8" \
  ./bench_frozen_oracle "$FROZEN" "$MODEL" "$RUNROOT/jeqb_s8"
```

| Z | estimator bins | pref·J bins | JEQB/Planck bins |
|---:|---:|---:|---:|
| 16 | 0 | 410,128 | 410,128 |
| 26 | **0** | **3,081,675** | **3,081,675** |

A의 Fe 주장 `0 / 3,081,675`를 정확히 재현했다.

### 3.2 수리 전후 EW provenance

명령:

```bash
python3 scripts/wave32_r35_compare.py \
  --pre /tmp/w31_on_a.JuCpDY \
  --post "$RUNROOT/r1_full/armed_s8" \
  --expect-r5-only
```

결과 RC=0:

```text
provenance_union=16481
provenance_unchanged=16473
provenance_changed=8
provenance_added=4
provenance_removed=0
changed_by_channel=coll_bf:4,rad_bf:4
PASS: provenance differences are confined to R5 bf fallback channels
```

따라서 parity EW provenance는 의도된 R5 bf 행 외 16,473 key가 원 CSV row byte 기준으로 동일하다. pair 해는 §2의 6/6 byte 비교로 동일하다. R3는 PASS다.

## 4. R5 Kramers·계측·음성 대조

### 4.1 Kramers fallback 및 Γ

s8 diagnostics 실측:

| 원소 | continuum coverage | Kramers count | target coverage | deletion |
|---|---:|---:|---:|---:|
| S | 581/581 | 7 | 581/581 | 0 |
| Fe | 4,198/4,198 | **122** | 4,198/4,198 | **0** |

Fe II→Fe III radiative bf Γ:

| 양 | 값 |
|---|---:|
| pre | `223097.040529958 s^-1` |
| post | `231369.60987676261 s^-1` |
| delta | `8272.5693468046084 s^-1` |
| ratio | `1.0370805875647362` |
| delta dex | **`+0.015812505062937628`** |

A의 `+0.0158125 dex`, Fe II 122준위를 재현했다.

### 4.2 독립 열합과 hot/cold 정상 계측

| 원소 | max independent assembly residual | hot/cold required | rebuilds | byte-identical | max abs |
|---|---:|---:|---:|---:|---:|
| S | `4.0439393052921851e-16` | 1 | 2 | 1 | 0 |
| Fe | `1.7759724667253631e-15` | 1 | 2 | 1 | 0 |

두 원소 모두 topology/numerical gate는 1이다. boundary gate는 기존과 같이 0이며 전체 verdict는 `EW_VALID_P_ELEM_SCOPE_FAIL`이다.

### 4.3 열합 gate seeded defect

저장소 소스가 아닌 `/tmp` 복사본에서 다음 한 줄만 바꿨다.

```diff
- ew_cap.expected_outflow[channel][j] += rate;
+ ew_cap.expected_outflow[channel][j] += rate *
+     (channel == NLTE_EW_RAD_BB ? 1.01 : 1.0);
```

별도 binary:

```bash
cp src/lumina_element_wide.c "$RUNROOT/lumina_element_wide_seed_ledger.c"
perl -0pi -e 's/ew_cap\.expected_outflow\[channel\]\[j\] \+= rate;/ew_cap.expected_outflow[channel][j] += rate * (channel == NLTE_EW_RAD_BB ? 1.01 : 1.0);/' \
  "$RUNROOT/lumina_element_wide_seed_ledger.c"
gcc -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE \
  -DLUMINA_FROZEN_ORACLE -Isrc \
  -o "$RUNROOT/bench_seed_ledger" \
  bench_frozen_oracle.c src/lumina_plasma.c \
  "$RUNROOT/lumina_element_wide_seed_ledger.c" \
  src/lumina_atomic.c -lm
```

Fe s8 replay 결과:

```text
independent_assembly_residual_max,0.0099009900990100781
channel_independent_assembly_residual_rad_bb,0.0099009900990100781
topology_gate_pass,0
numerical_gate_pass,0
verdict,EW_FAIL_SHADOW
```

주입 결함으로 FAIL을 재현했으므로 D6 열합 gate의 음성 대조는 PASS다.

### 4.4 manifest/counter 대조와 차단 결함

diagnostics와 manifest에 공통인 다음 9개 key는 정상 산출에서 모두 일치했다.

```text
candidate_assembler_calls
candidate_pair_owner_calls
save_restore_calls
per_ion_pin_calls
topstage_IV_calls
hot_cold_required
hot_cold_rebuilds
hot_cold_matrix_byte_identical
hot_cold_matrix_max_abs
```

`candidate_assembler_calls`를 manifest에서 2→0, `hot_cold_rebuilds`를 2→0으로 각각 변조한 복사본은 독립 CSV checker가 모두 RC=1로 검출했다.

```text
candidate_assembler_calls: diagnostics=2 manifest=0 mismatch=True
hot_cold_rebuilds: diagnostics=2 manifest=0 mismatch=True
```

그러나 이 음성 대조만으로 R5 카운터 PASS를 줄 수 없다.

1. 다음 실카운터가 diagnostics에는 있으나 manifest에는 없다.

```text
kramers_fallback_firing_count
continuum_deletion_firing_count
bf_estimator_bin_consumptions
bf_pref_J_bin_consumptions
bf_JEQB_bin_consumptions
```

명세 R5/D3은 fallback/삭제 카운터를 manifest에 노출하라고 명시한다. baseline manifest coverage checker 자체가 RC=1이다.
2. `save_restore_calls`, `per_ion_pin_calls`, `topstage_IV_calls`는 `EWFireCounts` 선언과 CSV 출력 외 증가/대입 지점이 없다. frozen 산출의 0은 runtime 계측 결과가 아니라 zero-init의 결과다.
3. `candidate_pair_owner_calls`는 capture 시작 직후 `!nlte_ew_capture_active()`를 읽어 0이 된다. candidate observer 안에서 pair-owner가 우회됐다는 국소 계측에는 맞지만 production `nlte_solve_all`의 skip/save/restore/pin/topstage 동작을 계측하지 않는다.

따라서 D7 및 공통 acceptance §6의 “모든 신설 gate/counter” 요건은 **FAIL**이다.

## 5. s0·s8 EW 재측정과 B2 대조

`p_ref`는 B2와 동일하게 `docs/CODEX_ABS_STATE_5154.md`의 약 3자리 조건부 값을 사용했다. 따라서 아래 `d_k`도 해당 정밀도 한계를 갖는다. 해의 stage fraction은 solution CSV의 SL `ion_total`을 II–IV 합으로 나눴다.

### 5.1 s8 S

| stage | B2 pair `d_k` | B2 pre-repair elem `d_k` | post-repair elem `d_k` |
|---|---:|---:|---:|
| II | 0.1045 | 2.2934 | `2.29315926` |
| III | 0.00208 | 0.03130 | `0.0313020269` |
| IV | 2.1668 | 3.1270 | `3.12697795` |
| D | 0.75779 | 1.81723 | `1.81714641` |

stage fraction pre→post:

```text
II   0.000114487765323342 -> 0.000114557428700665  (Δ +6.96634e-8)
III  0.909060038713139    -> 0.909059975419189     (Δ -6.32940e-8)
IV   0.0908254735215375   -> 0.0908254671521104    (Δ -6.36943e-9)
```

### 5.2 s8 Fe

| stage | B2 pair `d_k` | B2 pre-repair elem `d_k` | post-repair elem `d_k` |
|---|---:|---:|---:|
| II | 0.5563 | 1.3395 | `1.31958361` |
| III | 0.02560 | 0.21505 | `0.215052545` |
| IV | 1.3398 | 2.1575 | `2.15752620` |
| D | 0.64057 | 1.23735 | `1.23072079` |

stage fraction pre→post:

```text
II   1.44159703493522e-6 -> 1.50913099663938e-6  (Δ +6.75340e-8)
III  0.607634806269439   -> 0.607634765233352     (Δ -4.10361e-8)
IV   0.392363752133526   -> 0.392363725635651     (Δ -2.64979e-8)
```

### 5.3 s0 Fe

| stage | B2 pair `d_k` | B2 pre-repair elem `d_k` | post-repair elem `d_k` |
|---|---:|---:|---:|
| II | 4.8875 | 1.5528 | `1.55741929` |
| III | 1.1385 | 0.9670 | `0.966959402` |
| IV | 0.00306 | 0.00479 | `0.00478941512` |
| D | 2.0097 | 0.8415 | `0.843056034` |

stage fraction pre→post:

```text
II   2.78075534793900e-13 -> 2.75124941329024e-13 (Δ -2.95059e-15)
III  3.29109513896121e-5  -> 3.29109513883479e-5  (Δ -1.26416e-15)
IV   0.999967089048332    -> 0.999967089048337     (Δ +4.32987e-15)
```

사전등록 기대와 비교하면 s8과 s0 모두 R5 fallback에 따른 소폭 이동만 관측됐다. R6는 미구현이므로 s0 Fe IV 창/경계 회복은 없고 기존 `SCOPE_FAIL`도 유지된다. 기대 밖의 큰 변화는 관측되지 않았다. 이 절은 변화량만 기록하며 물리 해석을 추가하지 않는다.

재계산 핵심 식:

```python
p_stage = ion_total_stage / sum(ion_total_II_to_IV)
d_stage = abs(log10(p_stage / p_ref_stage))
D = sum(d_stage) / 3
```

사용한 `p_ref`:

```text
s8 S  = (0.0225, 0.977, 6.78e-5)
s8 Fe = (3.15e-5, 0.997, 0.00273)
s0 Fe = (9.93e-12, 0.000305, 0.989)
```

## 6. 빌드 위생

### 6.1 기본 clean build

```bash
make clean
make
```

결과:

- `make clean`: RC=0
- `make`: **RC=2**
- error line: 7
- warning line: 61
- 실패 원인: `-std=c11`에서 `M_PI` 미정의
  - `src/lumina_plasma.c:1919`
  - `src/lumina_cmfgen.c:361,927,1056,1503,1890,2061`

주요 경고 분류:

| 경고 | 줄 수 |
|---|---:|
| ignored OpenMP atomic pragma | 17 |
| misleading `if` indentation | 14 |
| ignored OpenMP parallel pragma | 9 |
| misleading `for` indentation | 8 |
| 기타 unused/implicit declaration/control reaches end | 13 |

### 6.2 검증 타깃과 selftest

```bash
make -B bench_frozen_oracle
make -B selftest_ioniz_saha
./selftest_ioniz_saha
```

- `bench_frozen_oracle`: RC=0, warning 34줄
- `selftest_ioniz_saha` build: RC=0
- `selftest_ioniz_saha` run: RC=0, `=== done ===`

기본 clean build가 실패하므로 빌드 위생 최종 판정은 FAIL이다.

## 7. 공통 acceptance §1–6 판정

1. **R1 byte invariant 2종:** 첫 번째 PASS, 두 번째 `COMMIT=1` FAIL/UNRESOLVED → 전체 FAIL.
2. **parity 환경 회귀:** PASS. R5 bf 8 key 외 provenance 및 pair 해 동일.
3. **신규 floor/cap/clamp 0:** EW runtime legacy guard firing 0. Kramers는 pair lane과 동일한 내용 fallback이며 새 clamp가 아니다. Wave-3.2 소유 범위에서 PASS.
4. **s0·s8 재측정:** 완료. R5 소폭 변화만, R6 미구현 상태 유지.
5. **C 독립 리뷰:** B가 판정할 항목 아님. C 단계 대기.
6. **음성 대조:** R1 및 열합 gate는 PASS. manifest 미노출/미배선 카운터는 음성 대조 자격 자체가 없어 FAIL.

## 8. UNRESOLVED 및 후속 차단 목록

- `COMMIT=1`을 실제 production `nlte_solve_all` 경로에서 frozen state로 재생하는 harness가 없다. 현 labeled oracle은 commit을 강제로 0으로 만든다.
- 현 s8 후보는 boundary scope gate가 실패하므로, commit harness만 고쳐도 성공 commit은 일어나지 않는다.
- production skip/save/restore/pin/topstage 카운터는 실제 실행 지점에 배선되어 있지 않다.
- Kramers fallback/deletion 카운터가 manifest에 없다.
- R4 signed DIE/역 autoionization은 구현자 보고대로 여전히 UNRESOLVED이며 본 B 검증에서 해소되지 않았다.
- R6 Fe V window는 미구현 상태다.
- 기본 CPU clean build가 실패한다.

이 차단 항목이 해소되기 전에는 Wave-3.2 배치를 acceptance PASS 또는 production-ready로 부를 수 없다.
