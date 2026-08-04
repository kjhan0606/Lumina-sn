# Codex B2 — Wave-3.2 A2 계약별 격리 사다리 검증 보고서

작성일: 2026-08-01 (Asia/Seoul)  
역할: 테스트 전용 (`src/` 수정 없음)  
대상: `docs/CODEX_WAVE32_A2_IMPL.md`의 현재 작업트리 구현  
정본: `docs/WAVE32_REPAIR_BATCH_SPEC_2026-08-01.md`,
`docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md` §5.3

## 0. 최종 판정

**전체 판정: FAIL**

R1 소유권 수리, R3 장 선택, D3 좌표 정책, clean build 및 R7 writer 자체는
재현됐다. 그러나 필수 8개 카운터 중 production 소유 경로 양성을 입증하지 못한
runtime 카운터 3개가 남는다. `save_restore_calls`와 `topstage_IV_calls`는 현재
후보 조립 전후 snapshot 창으로는 구조적으로 잡히지 않으며,
`per_ion_pin_calls`도 실제 frozen 후보에서 gate를 켜도 0이었다. 따라서 hook 함수가
증가한다는 단위시험만으로 R5/D7 PASS를 주지 않았다.

COMMIT=1 성공 commit은 현 후보가 boundary gate에서 차단되므로 **UNRESOLVED**다.
R7 실제 CUDA-loop OFF 중립성과 parity59 iter=10 capture도 실행 금지 범위이므로
**UNRESOLVED-until-capture**다.

| rung | 판정 | 핵심 결과 |
|---|---|---|
| rung1 R1 소유권 경계 | **PASS** | `SUPER_LEVELS={0,1}` × s0/s8 × 3산출 = 12/12 byte-identical; 1-byte 결함 RC=1; production `NLTEConfig` 전후 hash 동일 |
| rung2 COMMIT=1 pass-through | **UNRESOLVED** | pass-through/차단 중립성은 PASS (`requested=1`, `performed=0`, `boundary_gate`, production 3/3 동일); 성공 commit 양성 격리는 미존재 |
| rung3 R3 단일 진실원 | **PASS** | provenance 16,481키 중 의도된 bf 8키만 변경; JEQB에서 estimator 0, Planck S 410,128 / Fe 3,081,675 |
| rung4 카운터 8개 | **FAIL** | bf 5개는 양성·음성 PASS; runtime 3개는 hook 단위만 1/0이며 실제 소유 경로 양성 미입증 |
| rung5 D3 좌표 정책 | **PASS** | grid mismatch→Kramers S 581/581, Fe 4198/4198; overflow→EW 거부 계수; GPU/collisional helper 및 manifest 의도차 확인 |
| rung6 clean build | **PASS** | `make clean` RC=0, `make` RC=0, warning 60, error 0 |
| rung7 R7 writer | **UNRESOLVED** | schema/왕복/결정성/잘못된 state는 PASS; 실제 CUDA OFF/iter=10 capture는 미실행 |

검증 산출 루트는 `/tmp/codex_wave32_b2.BCDvYF`다. `/tmp` 보존은 보장되지
않는다. 영구 증거는 이 보고서와 `tests/wave32_*.c` 테스트 소스다.

## 1. 범위와 규율

- 신규 모델 실행: **0**
- GPU 실행: **0**
- frozen parity59 CPU replay: 허용 범위에서만 실행
- clean CPU build와 R7 오프라인 fixture: 실행
- `src/` 수정: **0**
- B2가 추가한 저장소 파일: 이 보고서와 `tests/wave32_*.c` 7개
- 기존 dirty 작업트리의 구현자/사용자 변경: 복원·정리·커밋하지 않음

공통 입력:

```bash
RUNROOT=$(mktemp -d /tmp/codex_wave32_b2.XXXXXX)
FROZEN=/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59
MODEL=data/tardis_reference_toy06_19p48d_sivcaiv
```

판정 의미는 다음과 같다.

- **PASS**: 양성, 음성, 해당 rung의 교차 격리가 모두 관측됨.
- **FAIL**: 요구된 falsifier가 결함을 검출했거나 필수 양성/음성이 성립하지 않음.
- **UNRESOLVED**: 허용된 frozen/offline 입력으로 계약의 실제 실행점을 만들 수 없음.

## 2. rung1 — R1 소유권 경계 [PASS]

### 2.1 양성: 12/12 byte 매트릭스

재현 명령:

```bash
make -B bench_frozen_oracle > "$RUNROOT/rung1_build.log" 2>&1
python3 scripts/wave32_r1_byte_invariant.py --no-build \
  --out "$RUNROOT/rung1_matrix"
```

결과는 RC=0과 다음 문장이었다.

```text
PASS: 12 SUPER_LEVELS={0,1} x armed COMMIT=0/unarmed byte comparisons
```

| SUPER_LEVELS | shell | 산출 | bytes | SHA-256 | equal |
|---:|---:|---|---:|---|---:|
| 0 | 0 | oracle | 27,748 | `b2c141f57638f349275143a244f68262d825abd465f5e0bbd7f2a1f7376d47b1` | 1 |
| 0 | 0 | pair ion | 290 | `29e56721aa17a9e9d561ead771f391aa3930fc992108da82965e4ad5d17ad683` | 1 |
| 0 | 0 | pair level | 272,071 | `38154f7d4cd3fb40abed0601c44099eaf70ef3d22a1a2a7b1be9e516d9147fd6` | 1 |
| 0 | 8 | oracle | 28,391 | `f3c9b752ecd63ecd77ae38d9a61eb2a676b3d7a49c25e1a7eb22d6a56a825dde` | 1 |
| 0 | 8 | pair ion | 527 | `c2b977bd42f45b0d9763c9d70df0848b822059b3270212381dc723f30c6bd337` | 1 |
| 0 | 8 | pair level | 316,974 | `13662c9881f209d64e4a4dd60807daf01209cab4b118cfad9334caa851c4472e` | 1 |
| 1 | 0 | oracle | 27,748 | 위 s0 hash와 동일 | 1 |
| 1 | 0 | pair ion | 290 | 위 s0 hash와 동일 | 1 |
| 1 | 0 | pair level | 272,071 | 위 s0 hash와 동일 | 1 |
| 1 | 8 | oracle | 28,391 | 위 s8 hash와 동일 | 1 |
| 1 | 8 | pair ion | 527 | 위 s8 hash와 동일 | 1 |
| 1 | 8 | pair level | 316,974 | 위 s8 hash와 동일 | 1 |

따라서 D8의 `SUPER_LEVELS` 강제 오염도 이 관측 범위에서는 재현되지 않았다.

### 2.2 음성: seeded 1-byte 결함

정상 unarmed s0 oracle 복사본에 1 byte를 추가하고 같은 모듈의 `main()` 및
byte 비교 코드를 사용했다.

```bash
cp -a "$RUNROOT/rung1_matrix/." "$RUNROOT/rung1_seed/"
f="$RUNROOT/rung1_seed/unarmed_super0_s0/lumina_oracle_cell_s0.csv"
truncate -s $(( $(stat -c %s "$f") + 1 )) "$f"
# run_cell만 준비된 디렉터리를 반환하도록 monkey-patch한 뒤 m.main() 실행
```

결과:

```text
seed_before_bytes=27748 seed_after_bytes=27749 expected_rc=1 actual_rc=1
FAIL super=0 s0 lumina_oracle_cell_s0.csv: byte diff
```

### 2.3 production `NLTEConfig` 직접 관측

세 파일 비교 외에 `tests/wave32_nlte_state_wrap.c`를 link wrapper로 사용했다.
production 객체는 그대로 컴파일하고 bench 호출 심볼만 wrapper로 바꿨다. wrapper는
다음을 호출 전후 FNV-1a로 해시한다.

- `NLTEConfig` struct 전체와 고정 offset/ion 배열
- level/global/line map, `fl_to_super`, super anchor
- full-level populations와 `within_sl_frac`
- J/estimator/count/C2 배열과 shell tau

핵심 빌드/실행 명령:

```bash
gcc -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
  -Isrc -Dnlte_element_wide_run_labeled=wave32_wrapped_nlte_element_wide_run_labeled \
  -c bench_frozen_oracle.c -o "$RUNROOT/bench.o"
gcc -O2 -Wall -Wextra -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
  -Isrc -c tests/wave32_nlte_state_wrap.c -o "$RUNROOT/wrap.o"
# src/lumina_plasma.c, src/lumina_element_wide.c, src/lumina_atomic.c를
# 변경 없이 별도 object로 컴파일 후 위 두 object와 -lm 링크
env -i PATH="$PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 \
  LUMINA_SUPER_LEVELS=0 LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=16,26 LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 \
  "$RUNROOT/bench_nlte_state" "$FROZEN" "$MODEL" "$RUNROOT/rung1_state_s8"
```

결과:

```text
[W32-NLTE-STATE] Z=16 s=8 before=198c10d73d33e561 after=198c10d73d33e561 byte_unchanged=1 rc=-1
[W32-NLTE-STATE] Z=26 s=8 before=198c10d73d33e561 after=198c10d73d33e561 byte_unchanged=1 rc=-1
```

`rc=-1`은 현 후보의 정직한 boundary scope 실패다. 중요한 소유권 관측은 후보가
실패한 경우에도 production object가 바뀌지 않았다는 점이다. source에서도
COMMIT=0만 `EWPrivateView`를 만들고 모든 성공/실패 반환에서 해제한다
(`src/lumina_element_wide.c:1388-1417,1666-1679`).

## 3. rung2 — COMMIT=1 pass-through [UNRESOLVED]

### 3.1 pass-through와 차단 중립성 [PASS]

```bash
env -i PATH="$PATH" \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 LUMINA_SUPER_LEVELS=0 \
  LUMINA_NLTE_ELEMENT_WIDE=1 LUMINA_NLTE_ELEMENT_WIDE_Z=16,26 \
  LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 \
  LUMINA_EW_FROZEN_COMMIT=1 LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$RUNROOT/rung2_commit1_s8" \
  ./bench_frozen_oracle "$FROZEN" "$MODEL" "$RUNROOT/rung2_commit1_s8"
python3 scripts/wave3_d8_pair_dump.py --frozen "$FROZEN" \
  --armed-dir "$RUNROOT/rung2_commit1_s8" \
  --unarmed-dir "$RUNROOT/rung1_matrix/unarmed_super0_s8" \
  --shell 8 --z 16,26
```

S와 Fe가 동일했다.

```text
verdict,EW_VALID_P_ELEM_SCOPE_FAIL
commit_requested,1
commit_performed,0
commit_blocked_by,boundary_gate
```

COMMIT pass-through off/on 비교:

| 관측 | 동일 수 |
|---|---:|
| production oracle/pair ion/pair level | 3/3 byte-identical |
| EW identity/raw/normalized/equilibrated/solution/provenance (S+Fe) | 12/12 byte-identical |
| 달라진 것 | diagnostics/manifest의 commit telemetry만 |

### 3.2 성공 commit 양성 격리 [UNRESOLVED]

`commit_performed=1`이 되는 후보가 없다. 따라서 “파일럿 (원소, 셸)만 변경하고
off-target은 byte-identical”은 실행할 수 없었다. `performed=0` 실행을 성공 commit
PASS로 재분류하지 않는다.

## 4. rung3 — R3 bound-free 단일 진실원 [PASS]

### 4.1 parity provenance 격리

pre-A2 VCS commit은 없지만 기존 B 단계가 보존한 frozen provenance
`/tmp/w31_on_a.JuCpDY`가 남아 있어, patch diff가 아니라 keyed output falsifier로만
사용했다.

```bash
python3 scripts/wave32_r35_compare.py \
  --pre /tmp/w31_on_a.JuCpDY \
  --post "$RUNROOT/rung1_matrix/armed_super0_s8" \
  --expect-r5-only
```

```text
provenance_union=16481
provenance_unchanged=16473
provenance_changed=8
provenance_added=4
provenance_removed=0
changed_by_channel=coll_bf:4,rad_bf:4
fe2_to_fe3_rad_bf_gamma_delta_dex=0.015812505062937628
PASS: provenance differences are confined to R5 bf fallback channels
```

의도된 R5 Kramers 행 외 16,473키는 원 CSV row byte가 동일했고 제거는 0이었다.
또한 `LUMINA_C2_MATRIX_BF`와 `LUMINA_NLTE_BF_JEQB`를 읽는 production 위치는
`nlte_bf_field_source()` 한 곳뿐이었다 (`src/lumina_plasma.c:393-414`).

### 4.2 JEQB 전환 양성/음성

baseline과 별도 프로세스로 다음을 실행했다.

```bash
env -i PATH="$PATH" LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 \
  LUMINA_SUPER_LEVELS=0 LUMINA_NLTE_BF_JEQB=1 \
  LUMINA_NLTE_ELEMENT_WIDE=1 LUMINA_NLTE_ELEMENT_WIDE_Z=16,26 \
  LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$RUNROOT/rung3_jeqb_s8" \
  ./bench_frozen_oracle "$FROZEN" "$MODEL" "$RUNROOT/rung3_jeqb_s8"
```

| 원소 | mode | estimator | pref·J | JEQB/Planck |
|---|---|---:|---:|---:|
| S | baseline | 251,674 | 158,454 | 0 |
| S | JEQB | 0 | 410,128 | 410,128 |
| Fe | baseline | 1,935,679 | 1,145,996 | 0 |
| Fe | JEQB | 0 | 3,081,675 | 3,081,675 |

JEQB on/off에서 EW identity 2/2는 동일했고 field를 소유하는 matrix/provenance/
solution 10/10만 달라졌다. 이는 격리 위반이 아니라 이 rung의 의도된 관측이다.

## 5. rung4 — 카운터 8개 [FAIL]

### 5.1 카운터별 양성·음성 표

diagnostics와 manifest 값은 아래 모든 정상/seed 실행에서 서로 동일했다. 그러나
“값 출력 일치”와 “자기 production 경로 계측”은 별도로 판정했다.

| 카운터 | 양성: 자기 조건 | 양성값 (S / Fe) | 음성 조건 | 음성값 | 판정 |
|---|---|---:|---|---:|---|
| `kramers_fallback_firing_count` | 정상 missing σ row | 7 / 122 | 모든 row present 주입 | 0 / 0 | **PASS** |
| `continuum_deletion_firing_count` | invalid upper-anchor seeded defect | 7 / 121 | 정상 target | 0 / 0 | **PASS** |
| `bf_estimator_bin_consumptions` | parity estimator | 251,674 / 1,935,679 | JEQB | 0 / 0 | **PASS** |
| `bf_pref_J_bin_consumptions` | 정상 estimator hole | 158,454 / 1,145,996 | 모든 estimator bin 양수 | 0 / 0 | **PASS** |
| `bf_JEQB_bin_consumptions` | JEQB | 410,128 / 3,081,675 | baseline | 0 / 0 | **PASS** |
| `save_restore_calls` | production overlap save/restore 필요 | 실제 경로 양성 없음 | frozen candidate | 0 / 0 | **FAIL** |
| `per_ion_pin_calls` | `LUMINA_NLTE_ION_LOCK=1` | frozen candidate에서도 0 / 0 | gate OFF | 0 / 0 | **FAIL** |
| `topstage_IV_calls` | production topstage branch 필요 | 실제 경로 양성 없음 | frozen candidate | 0 / 0 | **FAIL** |

### 5.2 bf 카운터 falsifier

`tests/wave32_counter_input_wrap.c`는 한 프로세스에 정확히 한 입력 결함만 주입한다.

```bash
# no_kramers: cmfgen_has_sigma를 모두 present로 설정
env -i PATH="$PATH" W32_COUNTER_INPUT_MODE=no_kramers ... \
  "$RUNROOT/bench_counter_input" "$FROZEN" "$MODEL" "$RUNROOT/rung4_no_kramers_s8"
# no_pref: 0 estimator bin만 DBL_MIN 양수로 설정
env -i PATH="$PATH" W32_COUNTER_INPUT_MODE=no_pref ... \
  "$RUNROOT/bench_counter_input" "$FROZEN" "$MODEL" "$RUNROOT/rung4_no_pref_s8"
```

`no_kramers`는 Kramers 0/0으로 닫혔고 target map 불완전은 숨기지 않아
`EW_FAIL_SHADOW`였다. `no_pref`는 estimator가 S 410,128 / Fe 3,081,675가 되고
pref·J가 0/0이었다.

deletion 양성은 `tests/wave32_continuum_seed_wrap.c`로 private view 생성 직후
upper anchor stage를 손상시켰다. production `src/`를 복사·수정하지 않고,
EW object의 field-helper 심볼만 test wrapper로 링크했다.

```text
S:  assembled_target_fail=758,  kramers=7,   deletion=7,   EW_FAIL_SHADOW
Fe: assembled_target_fail=7992, kramers=122, deletion=121, EW_FAIL_SHADOW
```

즉 seeded 결함이 계수되고 gate가 실패했으므로 가짜 정상화는 없었다.

### 5.3 runtime 3개가 FAIL인 이유

`tests/wave32_counter_hook_selftest.c`로 primitive 자체는 확인됐다.

```text
unarmed save_restore=0 per_ion_pin=0 topstage_IV=0
armed   save_restore=1 per_ion_pin=1 topstage_IV=1
```

하지만 실제 계측 대상은 이 hook 직접 호출이 아니다.

1. 후보는 `nlte_assemble_rate_matrix()` 한 호출의 직전/직후만 snapshot한다
   (`src/lumina_element_wide.c:1464-1472`).
2. save/restore hook은 바깥 `nlte_solve_all()` pair loop 뒤에 있다
   (`src/lumina_plasma.c:17190-17195`). 후보 snapshot 창 밖이다.
3. topstage hook 조건은 명시적으로 `!ew_capture`다
   (`src/lumina_plasma.c:15894-15897`). 후보 capture 중에는 발화 불가능하다.
4. per-ion hook은 assembly 안에 있지만 frozen state에서 ion-lock을 켠 실제 실행도
   diagnostics/manifest 모두 0이었다.

전체 production solve를 frozen state에 재진입시키는
`tests/wave32_runtime_counter_wrap.c`도 시도했다. 이 fixture는 전체 solve에 필요한
상태를 보존하지 않아 RC=139로 종료됐고 runtime 양성 산출은 생성되지 않았다.
세 항목은 “hook이 배선돼 보인다”는 이유로 PASS 처리하지 않는다.

## 6. rung5 — D3 좌표 정책 [PASS]

### 6.1 grid 길이 불일치 → Kramers

```bash
env -i PATH="$PATH" W32_COUNTER_INPUT_MODE=grid_mismatch \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 LUMINA_SUPER_LEVELS=0 \
  LUMINA_NLTE_ELEMENT_WIDE=1 LUMINA_NLTE_ELEMENT_WIDE_Z=16,26 \
  LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$RUNROOT/rung5_grid_mismatch_s8" \
  "$RUNROOT/bench_counter_input" "$FROZEN" "$MODEL" \
  "$RUNROOT/rung5_grid_mismatch_s8"
```

| 원소 | continuum coverage | Kramers coverage | firing | deletion | verdict |
|---|---:|---:|---:|---:|---|
| S | 581/581 | 581 | 581 | 0 | `EW_VALID_P_ELEM_SCOPE_FAIL` |
| Fe | 4198/4198 | 4198 | 4198 | 0 | `EW_VALID_P_ELEM_SCOPE_FAIL` |

동일 실행의 production oracle은 정상 grid 실행과 byte-identical이었다. 따라서 이
입력 변경은 COMMIT=0 candidate 관측만 건드렸다.

### 6.2 pair 1e30 cap 비복제와 EW 거부

pair의 기존 cap은 `src/lumina_plasma.c:15712-15728`에 그대로 있다. EW는
`log_nstar >= log(DBL_MAX)`를 계수하고 `continue`하며 cap하지 않는다
(`src/lumina_element_wide.c:427-435`). `T_e=1 K`의 overflow 입력 결과:

```text
S:  nstar_cap_firing_count=581,  nonfinite_guard=0, EW_FAIL_SHADOW
Fe: nstar_cap_firing_count=4188, nonfinite_guard=0, EW_FAIL_SHADOW
```

재현:

```bash
env -i PATH="$PATH" W32_COUNTER_INPUT_MODE=nstar_overflow ... \
  "$RUNROOT/bench_counter_input" "$FROZEN" "$MODEL" \
  "$RUNROOT/rung5_nstar_overflow_s8"
```

### 6.3 나머지 좌표와 manifest

`tests/wave32_bf_policy_selftest.c` 결과:

```text
default source=0 use_gpu=1 gpu_bypass=0 selected=3 collisional=0
c2      source=1 use_gpu=0 gpu_bypass=1 selected=3 collisional=0
jeqb    source=2 use_gpu=0 gpu_bypass=1 selected=8.0019253929959525e-06 collisional=0
parity  source=1 use_gpu=0 gpu_bypass=1 selected=3 collisional=1
```

pair와 EW의 collisional 조건은 `nlte_bf_collisional_enabled()` 한 helper를 호출하고,
GPU 허용/우회는 field helper 반환값이 결정한다. EW 역재결합은
`within_sl_frac`의 `f_upper`를 rad/coll 양쪽에 곱한다
(`src/lumina_element_wide.c:453-460`).

manifest는 의도적 비동등을 정확히 명시했다.

```text
pair_inverse_saha_cap_register,legacy_pair_1e30_cap_unchanged
recombination_weighting_contract,EW_within_sl_frac;pair_D5_unweighted_defect_not_replicated
grid_mismatch_policy,pair_policy_adopted:Kramers_fallback_counted
```

## 7. rung6 — clean build [PASS]

```bash
make clean > "$RUNROOT/rung6_clean.log" 2>&1
make > "$RUNROOT/rung6_make.log" 2>&1
rg 'warning:' "$RUNROOT/rung6_make.log" | wc -l
rg 'error:' "$RUNROOT/rung6_make.log" | wc -l
```

| 항목 | 결과 |
|---|---:|
| `make clean` RC | 0 |
| `make` RC | 0 |
| warning | 60 |
| error | 0 |

경고 분류는 ignored OpenMP pragma 28, misleading indentation 22, unused 계열
9, `setenv` implicit declaration 1이다.

## 8. rung7 — R7 writer [UNRESOLVED]

### 8.1 오프라인 writer/왕복 [PASS]

```bash
make -B selftest_cmf_chieta_dump
python3 scripts/cmf_chieta_roundtrip_selftest.py --no-build
./selftest_cmf_chieta_dump "$RUNROOT/rung7/a.lcmfce"
./selftest_cmf_chieta_dump "$RUNROOT/rung7/b.lcmfce"
cmp "$RUNROOT/rung7/a.lcmfce" "$RUNROOT/rung7/b.lcmfce"
sha256sum "$RUNROOT/rung7/"{a,b}.lcmfce
```

```text
PASS LCMFCE01 write-read-write bitwise roundtrip
sha256=3981641ed3fa6f9bfac8425b248f546012501fac78094522816a22ab950c6d52 bytes=424
```

두 독립 write도 같은 424 bytes와 SHA-256을 냈다.

### 8.2 설계 정본 §5.3 필드 단위 대조

| 순서 | 설계 필드 | fixture 실측 |
|---:|---|---|
| 1 | magic/endian/version | `LCMFCE01`, `0x01020304`, 1 |
| 2 | nr/nnu/iter/generation | 2, 3, 10, 10 |
| 3 | flags/reserved/t_exp | 7, 0, 1,683,072 s |
| 4 | `r_edge[nr+1]` | `1e14,2e14,4e14` |
| 5 | descending `nu` | `4e14,2e14,1e14` |
| 6 | positive reversed `dnu` | `2e14,1e14,5e13` |
| 7 | `chi_total` | `3,2,1,6,5,4` |
| 8 | `chi_coherent` | `0.3,0.2,0.1,0.6,0.5,0.4` |
| 9 | `eta_fixed` | `27,16,7,72,55,40` |
| 10 | `eta_coherent` | `4.5,2.8,1.3,10.8,8.5,6.4` |
| 11 | `eta_total` | 각 cell에서 위 두 값의 IEEE-754 합과 bitwise 동일 |
| 12 | `J_producer` | `15,14,13,18,17,16` |
| 13 | sidecar SHA/audit | SHA 일치, bitwise=true, max_abs=0.0 |

고정 header 64 bytes와 45개 float64 360 bytes의 합이 424 bytes다. native struct
padding이나 trailing byte는 없다.

### 8.3 잘못된 state 음성 대조 [PASS]

`tests/wave32_cmf_chieta_negative.c`를 production writer와 링크했다.

```bash
gcc -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -ffunction-sections \
  -fdata-sections -Isrc -Wl,--gc-sections \
  -o "$RUNROOT/rung7_negative" tests/wave32_cmf_chieta_negative.c \
  src/lumina_cmfgen.c -lm
"$RUNROOT/rung7_negative" "$RUNROOT/rung7/should_not_exist.lcmfce"
```

| 결함 | writer RC | payload 존재 |
|---|---:|---:|
| negative χ | -1 | 0 |
| non-contiguous radius | -1 | 0 |
| non-ascending source ν | -1 | 0 |
| non-finite J | -1 | 0 |
| negative iteration | -1 | 0 |

writer는 파일을 열기 전에 state를 검증하므로 모두 fail-closed였다.

### 8.4 OFF 중립성과 실제 capture [UNRESOLVED]

source 상으로 writer gate는 optional J damping 직후에 있고, path가 비어 있거나
미설정이면 `if (dump_path && *dump_path)` 내부로 진입하지 않는다
(`src/lumina_cuda.cu:7539-7566`). malformed/out-of-range iter는 `EXIT_FAILURE`다.

그러나 실제 CUDA loop를 돌려 다음을 측정하는 것은 규율상 금지됐다.

- gate OFF 실행의 dump 0개 + 기존 수송 산출 byte-identical
- producer iter=10 post-damping 실제 parity59 payload
- consumer iter=11과 generation/lag 대조

따라서 source no-op 구조와 writer fixture는 PASS지만 실제 loop 중립성과 capture는
**UNRESOLVED-until-capture**다. 기존 C1/C2/line dump로 χ,η를 추정하지 않았다.

## 9. 교차 격리 종합표

기호: `=` byte-identical, `Δown` 해당 rung이 소유한 의도 변화, `T` telemetry만
변화, `U` 허용 입력으로 미실행, `—` 비교 대상 없음.

| gate on/off 쌍 | production oracle/pair | EW identity | EW matrix/provenance/solution | 타 counter/telemetry | R7 파일 | 판정 |
|---|---|---|---|---|---|---|
| R1 armed COMMIT=0 / unarmed (`SUPER=0,1`, s0/s8) | `=` 12/12 | armed에만 존재 | armed에만 존재 | armed observer만 존재 | 0/0 | **PASS** |
| R2 frozen COMMIT pass-through 1 / 0 | `=` 3/3 | `=` 2/2 | `=` 10/10 | `T` requested/blocked만 | 0/0 | **PASS**(차단 중립성) |
| R3 JEQB 1 / 0 | `Δown` pair-field oracle | `=` 2/2 | `Δown` 10/10 | `Δown` field counters | 0/0 | **PASS** |
| R4 runtime hook armed / unarmed | — | — | — | hook `1/1/1` / `0/0/0`; 실제 후보는 0 | 0/0 | **FAIL**(경로 계측) |
| R5 grid mismatch / matched | production oracle `=` | `Δown` checksum/coverage | `Δown` candidate bf | `Δown` Kramers만 | 0/0 | **PASS** |
| R6 clean build | 실행 산출 없음 | — | — | — | — | **PASS** |
| R7 dump gate OFF / ON | 실제 CUDA 비교 `U` | — | — | — | fixture ON만 생성 | **UNRESOLVED** |

인위적인 `nstar_overflow`, `no_pref`, invalid-anchor seed는 입력 자체가 다른
counter falsifier이므로 일반 gate 교차 격리 분모에 넣지 않았다. 해당 seed가
실패를 숨기지 않고 counter/verdict를 바꾸는지는 각 rung 표에 별도로 기록했다.

## 10. 결함 및 UNRESOLVED 원장

1. **FAIL-R5-D7-SNAPSHOT:** `save_restore_calls`는 후보 snapshot 창 밖에서 발화한다.
   현 diagnostics/manifest 값은 실제 production save/restore 계측이 될 수 없다.
2. **FAIL-R5-D7-TOPSTAGE:** topstage branch는 `!ew_capture`를 요구해 후보 snapshot
   중 발화 불가능하다.
3. **FAIL-R5-D7-PIN:** ion-lock을 켠 frozen 후보에서도 `per_ion_pin_calls=0`; 실제
   소유 경로 양성 증거가 없다.
4. **UNRESOLVED-R1-COMMIT-POSITIVE:** boundary gate를 통과하는 후보가 없어
   `commit_performed=1` 및 파일럿-only 변화가 미측정이다.
5. **UNRESOLVED-R7-CUDA-OFF:** 실제 CUDA loop OFF 중립성 미측정이다.
6. **UNRESOLVED-R7-CAPTURE:** parity59 producer iter=10 χ,η payload가 아직 없다.
7. R4 signed DIE/역 autoionization과 R6 Fe V 창은 A2 보고서대로 이번 테스트
   범위에서도 해소되지 않았다. 이 보고서가 이를 PASS로 승격하지 않는다.

## 11. 최종 acceptance

Wave-3.2 A2를 전체 PASS 또는 production-ready로 판정할 수 없다. 다음 재검의 최소
조건은 runtime 3개 카운터의 snapshot 범위를 실제 소유 경로와 일치시키고, 각 경로의
양성/음성을 보존 가능한 CPU fixture에서 재현하는 것이다. 이후 boundary gate를
통과하는 COMMIT=1 fixture와 실제 R7 iter=10 capture가 별도로 필요하다.

마지막 정적 위생 검사:

```bash
git diff --check
```

RC=0이었다. B2는 `src/`를 수정하지 않았다.
