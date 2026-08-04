# Codex A2 — Wave-3.2 보수 라운드 + R7 편입 구현 보고서

작성일: 2026-08-01 (Asia/Seoul)  
범위: `src/lumina_plasma.c`, `src/lumina_element_wide.c`, `src/lumina_cmfgen.c/.h`, `src/lumina_cuda.cu`, `Makefile`, `scripts/`  
규율: 신규 모델/GPU 실행 없음, frozen replay·CPU build·오프라인 writer fixture만 실행, 커밋하지 않음

## 0. 판정 요약

| 항목 | 구현/검증 판정 | 핵심 결과 |
|---|---|---|
| A2-1 R1 COMMIT=0 소스 불변식 | **PASS** | shadow 전용 33-slot 인덱스 view로 전역 레이아웃을 건드리지 않음. `SUPER_LEVELS={0,1}` × s0/s8의 권위 산출물 12/12 byte-identical |
| A2-2 R1 COMMIT=1 관측성 | **PASS(계측) / 양성 격리 UNRESOLVED** | frozen env를 실제 인자로 전달. 현 s8은 정직하게 `SCOPE_FAIL`, `requested=1`, `performed=0`, `blocked_by=boundary_gate` |
| A2-3 R3 단일 진실원 | **PASS(소스)** | C2/JEQB/parity field 선택과 GPU lookup 허용/우회를 한 helper로 통합 |
| A2-4 R5 카운터 | **PASS(배선/manifest)** | 필수 5개 manifest key 추가, runtime 3개를 실제 분기 hook과 전후 snapshot에 연결 |
| A2-5 D3 5좌표 | **PASS(정책 구현)** | grid·collisional·GPU 정렬; pair 1e30 cap과 D-5 무가중 결함은 EW에 복제하지 않고 manifest에 의도적 차이로 등재 |
| A2-6 clean build | **PASS** | `make clean && make` RC=0, error=0, warning=60(B의 61보다 1 감소), 신규 코드 유발 warning 없음 |
| A2-7 R7 frozen χ,η dump | **PASS(구현·오프라인 왕복) / 실모델 capture UNRESOLVED** | `LCMFCE01` v1, 424-byte fixture write→read→write bitwise PASS, SHA-256 일치. 규율상 실제 GPU/모델 capture 미실행 |

이번 라운드는 신규 clamp/floor/cap을 추가하지 않았다. pair의 기존 inverse-Saha `1e30` cap은 수정하지 않고 대장 좌표만 재확인했다.

## 1. 입력 근거와 수리 기준

두 필독 문서를 구현 전에 확인했다.

- B는 COMMIT=0 6/6 PASS가 있었지만 frozen entry가 commit 인자를 0으로 고정하고, 현 boundary verdict가 `SCOPE_FAIL`이라 실제 COMMIT=1 격리가 미검증이라고 판정했다(`docs/CODEX_WAVE32_B_TEST.md:14`, `77-105`).
- B는 manifest 필수 카운터 5개 누락과 runtime `*_calls` 3개 미배선을 지적했다(`docs/CODEX_WAVE32_B_TEST.md:15`, `259-294`).
- B clean build는 `M_PI` 7건 때문에 RC=2였고 warning 61건이었다(`docs/CODEX_WAVE32_B_TEST.md:16`, `375-414`).
- C는 unarmed `SUPER_LEVELS=0`에서 armed shadow가 전역 `super_mode`와 파생 배열을 바꿀 수 있으므로 기존 byte PASS가 환경 우연이라고 판정했다(`docs/CODEX_WAVE32_C_REVIEW.md:39-41`).
- C는 pair가 C2 cache와 GPU 허용 조건을 helper 밖에서 복제한다고 지적했다(`docs/CODEX_WAVE32_C_REVIEW.md:47-53`).
- C는 runtime 3개 카운터가 선언/출력뿐이고 실제 증가 지점이 없다고 확인했다(`docs/CODEX_WAVE32_C_REVIEW.md:79-89`).
- C의 D3 다섯 좌표는 grid 길이, inverse-Saha cap, 재결합 가중, collisional bf 조건, GPU 우회다(`docs/CODEX_WAVE32_C_REVIEW.md:91-103`).
- R7 정본은 호출 시점과 iter=10 계약(`docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md:343-348`), binary field 순서(`:349-365`), little-endian/descending ν/동시 역순/η 감사(`:368-370`)를 기준으로 삼았다.

## 2. A2-1 — R1 COMMIT=0 소스 불변식

### 2.1 수리

`nlte_element_wide_layout_enabled()`는 이제 EW가 단순히 armed됐다는 이유로 전역 레이아웃을 확장하지 않는다. production에서는 실제 commit knob가 켜진 경우만, frozen에서는 명시적인 `LUMINA_EW_FROZEN_COMMIT=1`인 경우만 commit-capable 전역 레이아웃을 요청한다(`src/lumina_element_wide.c:143-155`).

COMMIT=0 후보 조립에는 `EWPrivateView`를 도입했다(`src/lumina_element_wide.c:530-647`). 이 view는 원래 `NLTEConfig`의 radiation/estimator 입력을 얕게 참조하되 다음 항목은 모두 후보 소유 메모리로 만든다.

- 33개 EW ion slot과 level/super offset
- `nlte_to_global_level`, `global_to_nlte_level`, `nlte_line_map`
- `fl_to_super`, `super_anchor_global`, `within_sl_frac`
- 후보용 `nlte_level_populations`

`ew_run_impl()`은 `commit_requested==0`일 때만 이 private view로 조립하고, 성공·실패의 모든 반환 경로에서 해제한다(`src/lumina_element_wide.c:1388-1417`, `1666-1679`). 따라서 shadow 실행은 production `n_nlte_ions`, `super_mode`, offset/map, 파생 fraction, population buffer를 쓰지 않는다. 증상별 사후 복원 가드가 아니라 후보의 소유권/인덱싱 경계를 분리한 수리다.

### 2.2 확장된 byte 검증

`scripts/wave32_r1_byte_invariant.py`는 각 실행에 `LUMINA_SUPER_LEVELS`를 명시하고, `{0,1}` × shell `{0,8}`를 순회한다(`scripts/wave32_r1_byte_invariant.py:42-58`, `78-104`). 각 cell에서 아래 3개를 armed COMMIT=0과 unarmed 사이에 byte 비교한다.

1. `lumina_oracle_cell_s{shell}.csv`
2. `pair_ion_fractions.csv`
3. `pair_level_populations.csv`

재현 명령:

```bash
make -B bench_frozen_oracle
out=$(mktemp -d /tmp/w32r1_final.XXXXXX)
python3 scripts/wave32_r1_byte_invariant.py --no-build --out "$out"
```

결과:

```text
PASS: 12 SUPER_LEVELS={0,1} x armed COMMIT=0/unarmed byte comparisons
summary=/tmp/w32r1_final.YNLg3j/byte_invariant_summary.csv
```

두 super 설정에서 hash도 동일했다.

| shell | artifact | bytes | SHA-256 |
|---:|---|---:|---|
| 0 | oracle | 27,748 | `b2c141f57638f349275143a244f68262d825abd465f5e0bbd7f2a1f7376d47b1` |
| 0 | pair ion | 290 | `29e56721aa17a9e9d561ead771f391aa3930fc992108da82965e4ad5d17ad683` |
| 0 | pair level | 272,071 | `38154f7d4cd3fb40abed0601c44099eaf70ef3d22a1a2a7b1be9e516d9147fd6` |
| 8 | oracle | 28,391 | `f3c9b752ecd63ecd77ae38d9a61eb2a676b3d7a49c25e1a7eb22d6a56a825dde` |
| 8 | pair ion | 527 | `c2b977bd42f45b0d9763c9d70df0848b822059b3270212381dc723f30c6bd337` |
| 8 | pair level | 316,974 | `13662c9881f209d64e4a4dd60807daf01209cab4b118cfad9334caa851c4472e` |

## 3. A2-2 — R1 COMMIT=1 검증 가능화

`nlte_element_wide_run_labeled()`는 더 이상 `commit_requested=0`을 고정하지 않는다. frozen 전용 env `LUMINA_EW_FROZEN_COMMIT`을 읽고, 미설정은 0, 정확히 문자열 `1`만 1로 전달하며 그 밖의 값은 config failure로 닫는다(`src/lumina_element_wide.c:1690-1706`).

후보 판정에는 다음을 추가했다(`src/lumina_element_wide.c:1615-1637`).

- `commit_requested`
- `commit_performed = pass && commit_requested`
- `commit_blocked_by = not_requested | topology_gate | numerical_gate | boundary_gate | none`

실제 population 쓰기는 오직 `commit_performed`일 때 수행한다(`src/lumina_element_wide.c:1654-1665`). 세 값은 diagnostics와 manifest 양쪽에 기록한다(`src/lumina_element_wide.c:1647-1651`).

재현 명령:

```bash
run_dir=$(mktemp -d /tmp/w32_commit1.XXXXXX)
env -i PATH="$PATH" \
  LUMINA_FROZEN_ORACLE_ONLY_SHELL=8 \
  LUMINA_SUPER_LEVELS=0 \
  LUMINA_NLTE_ELEMENT_WIDE=1 \
  LUMINA_NLTE_ELEMENT_WIDE_Z=16,26 \
  LUMINA_NLTE_ELEMENT_WIDE_SHELL=8 \
  LUMINA_NLTE_ELEMENT_WIDE_COMMIT=0 \
  LUMINA_EW_FROZEN_COMMIT=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP=1 \
  LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR="$run_dir" \
  ./bench_frozen_oracle \
  /gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59 \
  data/tardis_reference_toy06_19p48d_sivcaiv "$run_dir"
```

결과(`/tmp/w32_commit1.NkeV3G`): Z=16과 Z=26 모두 동일하다.

```text
verdict,EW_VALID_P_ELEM_SCOPE_FAIL
commit_requested,1
commit_performed,0
commit_blocked_by,boundary_gate
```

이는 가짜 PASS가 아니다. 현 후보는 boundary scope gate를 통과하지 못하므로 쓰지 않는 것이 계약에 맞다. R6 M_V 창 이후 통과 구성이 생겼을 때 수행할 “pilot만 변화 + off-target byte-identical” 양성 격리 실증은 **UNRESOLVED**로 유지한다.

## 4. A2-3 — R3 bound-free field 단일 진실원

`nlte_bf_field_source()`를 field 값뿐 아니라 GPU lookup 사용 권한까지 결정하는 유일 helper로 확장했다(`src/lumina_plasma.c:385-415`). helper가 한 번만 다음을 판정한다.

- source 0: 기본 `pref*J`, legacy GPU lookup 허용
- source 1: `(parity || C2_MATRIX_BF) && estimator`, GPU lookup 금지 및 우회 기록
- source 2: JEQB `B_nu(T_e)`, GPU lookup 금지 및 우회 기록

pair 레인의 별도 C2 cache와 `!bf_jeqb && !c2mx` GPU 조건을 제거했다. pair는 helper가 반환한 `use_gpu_R_bf`만 소비하고, 다른 field를 선택한 상태에서 GPU table이 존재하면 `bf_gpu_field_bypass_levels`로 명시 계수한다(`src/lumina_plasma.c:15639-15669`). EW도 같은 helper를 사용한다(`src/lumina_element_wide.c:347-348`, `399-416`).

collisional bf on/off 정책도 `nlte_bf_collisional_enabled()` 한 곳에 두고(`src/lumina_plasma.c:417-420`), pair와 EW가 같은 helper를 호출한다(`src/lumina_plasma.c:15814-15829`, `src/lumina_element_wide.c:438-460`).

## 5. A2-4 — R5 카운터 완결

### 5.1 manifest 필수 5개

다음 key를 diagnostics 실카운터와 같은 `EWFireCounts` 값으로 manifest에 추가했다(`src/lumina_element_wide.c:1647-1651`).

- `kramers_fallback_firing_count`
- `continuum_deletion_firing_count`
- `bf_estimator_bin_consumptions`
- `bf_pref_J_bin_consumptions`
- `bf_JEQB_bin_consumptions`

frozen COMMIT=1 음성 대조의 Z=16/s8 실측값은 각각 `7, 0, 251674, 158454, 0`이었다. 따라서 manifest coverage는 단순 key 존재가 아니라 실제 조립 중 증가값을 노출한다.

### 5.2 runtime `*_calls` 3개

process runtime counter와 hook을 추가하고(`src/lumina_element_wide.c:220-251`), 실제 분기에 배선했다.

- `topstage_IV_calls`: top-stage IV branch 진입 직후 (`src/lumina_plasma.c:15894-15897`)
- `per_ion_pin_calls`: ion-lock pin branch 진입 직후 (`src/lumina_plasma.c:16476-16478`)
- `save_restore_calls`: saved lower-ion buffer 복원 branch 진입 직후 (`src/lumina_plasma.c:17190-17195`)

후보 assembler 호출 전후 snapshot의 delta를 후보별 `EWFireCounts`에 넣는다(`src/lumina_element_wide.c:1465-1472`). 이번 frozen s8에서는 세 값 모두 0이었다. 이 0은 리터럴/초기값을 그대로 출력한 것이 아니라 실제 hook delta이며, 조기 반환/비활성 분기가 호출을 막았다는 사실을 관측한 값이다.

## 6. A2-5 — D3 Kramers/bf 좌표별 정책

| 좌표 | 구현 결과 | 코드/manifest 근거 |
|---|---|---|
| 1. grid 길이 | pair 정책을 EW에 채택. `cmfgen_n_freq_bins != n_freq_bins`이면 table row를 쓰지 않고 Kramers fallback 및 계수 | `src/lumina_element_wide.c:319-325`, coverage `:693-710` |
| 2. inverse-Saha | EW의 표현범위 초과 거부+카운터 유지. pair의 기존 `1e30` cap은 복제/수정하지 않음 | EW `src/lumina_element_wide.c:425-433`; manifest `pair_inverse_saha_cap_register=legacy_pair_1e30_cap_unchanged` |
| 3. 재결합 가중 | EW의 `within_sl_frac` 가중 유지. pair D-5 무가중 결함을 복제하지 않음 | `src/lumina_element_wide.c:373-375`, `453-460`; manifest `recombination_weighting_contract=EW_within_sl_frac;pair_D5_unweighted_defect_not_replicated` |
| 4. collisional bf | 공유 helper를 통해 양 레인 동일 조건 | `src/lumina_plasma.c:417-420`, `15814`; `src/lumina_element_wide.c:438` |
| 5. GPU 우회 | field helper가 GPU lookup 가능/우회를 직접 결정하고 기록 | `src/lumina_plasma.c:393-414`, `15646-15658` |

위 2·3은 “pair와 무조건 같은 산술”이 아니라 차터가 정한 운전석 정책을 따른 의도적 비동등이다. 특히 pair 결함을 EW로 확산시키지 않았다.

## 7. A2-6 — clean build

`src/lumina_plasma.c` 1건과 `src/lumina_cmfgen.c` 6건의 비이식 `M_PI`를 프로젝트 관용구 `M_PI_VAL`로 교체했다. 현재 두 파일에 standalone `M_PI` 토큰은 없다.

M_PI 해소 뒤 CPU 기본 타깃에서 드러난 CUDA symbol link 공백도 기존 fallback 계약에 맞게 닫았다. CPU 빌드는 `cmf_solve_J_gpu()`와 `bf_gemm_compute_fine()`의 `-1` stub을 사용하고, CUDA 빌드는 Makefile의 `-DLUMINA_HAS_CUDA_BF_GEMM`로 stub을 제외한다(`src/lumina_cmfgen.c:22-61`, `Makefile:28`, `43-45`). 신규 수치 동작은 없으며 `-1`은 이미 존재하던 CPU fallback 신호다.

재현 명령 및 결과:

```bash
make clean && make > /tmp/w32_a2_clean_build.log 2>&1
rg -c 'warning:' /tmp/w32_a2_clean_build.log
rg -c 'error:' /tmp/w32_a2_clean_build.log
```

```text
RC=0
warning: 60
error: 0
```

B 기준 61건에서 60건으로 줄었다. 남은 60건은 기존 unknown OpenMP pragma 28, misleading-indentation 22, unused 계열 9, 기존 `setenv` implicit declaration 1이다. 이번 writer/helper/stub에서 발생한 신규 warning은 없다.

## 8. A2-7 (=R7) — frozen χ,η dump

### 8.1 writer와 호출 게이트

public writer 선언은 `src/lumina_cmfgen.h:123-124`, 구현은 `src/lumina_cmfgen.c:207-339`에 추가했다. writer는 입력 state, shell/grid 크기, 반경 연속성, ascending source ν, 양의 `dnu`, finite/nonnegative cell 필드와 η 산출 곱의 유한성을 파일을 열기 전에 검증한다(`src/lumina_cmfgen.c:211-257`).

CUDA loop에서는 `cmfgen_solve_J`와 optional J damping 직후에 게이트를 검사한다(`src/lumina_cuda.cu:7539-7566`).

- `LUMINA_CMF_FROZEN_CHIETA_DUMP`: 비어 있지 않은 경로일 때만 armed
- `LUMINA_CMF_FROZEN_CHIETA_ITER`: armed일 때 필수, strict integer, `[0, pc_iter)` 범위
- 미설정 dump gate: 분기 외 상태/출력/쓰기 없는 no-op
- 잘못된/missing iter: `EXIT_FAILURE`
- 정확히 선택된 `it`에서 writer 실패: `EXIT_FAILURE`

따라서 OFF는 기존 field와 결과를 건드리지 않으며, iter 불일치 상태를 임의로 가까운 epoch에 dump하지 않는다.

### 8.2 binary v1

writer는 native struct를 쓰지 않고 u32/u64/f64를 byte 단위 little-endian으로 직렬화한다(`src/lumina_cmfgen.c:185-205`, `266-304`). 순서는 정본과 같다.

1. `magic="LCMFCE01"`, endian marker, version
2. `nr`, `nnu`, `iteration`, `field_generation`
3. flags/reserved, `t_exp_s`
4. `r_edge`, descending `nu`, positive `dnu`
5. `chi_total`, `chi_coherent`
6. `eta_fixed=chi_tot*S_fixed`
7. `eta_coherent=chi_es*J`
8. `eta_total=eta_fixed+eta_coherent`
9. `J_producer`

현재 CMFGEN source grid가 ascending이므로 모든 frequency-dependent field를 shell-major 순서를 유지한 채 함께 역순으로 쓴다(`src/lumina_cmfgen.c:282-303`). flags는 post-damp/coherent-frozen/frequency-descending 세 bit를 모두 기록하며 generation은 선택 iter와 같다.

payload 전체 SHA-256을 writer가 계산하여 `<dump>.manifest.json`에 기록한다(`src/lumina_cmfgen.c:310-338`). sidecar에는 iter/generation, 세 flag, `eta_decomposition_bitwise=true`, max abs `0.0`도 포함한다.

### 8.3 오프라인 왕복 자기시험

`scripts/cmf_chieta_writer_fixture.c`는 모델/GPU 없이 2 shell × 3 bin state를 writer에 공급한다. `scripts/cmf_chieta_roundtrip_selftest.py`는 exact little-endian schema를 읽고 다음을 검사한다(`scripts/cmf_chieta_roundtrip_selftest.py:17-86`).

- magic/endian/version/reserved/flags/iter/generation
- 파일 길이와 trailing byte 없음
- strict descending ν, 모든 `dnu > 0`
- 각 cell의 `eta_fixed + eta_coherent`와 `eta_total` IEEE-754 byte 동일
- parse 후 재직렬화한 payload 전체 byte 동일
- sidecar SHA-256과 Python `hashlib.sha256` 동일

재현 명령:

```bash
make -B selftest_cmf_chieta_dump
python3 scripts/cmf_chieta_roundtrip_selftest.py --no-build
```

결과:

```text
PASS LCMFCE01 write-read-write bitwise roundtrip: /tmp/cmf_chieta_rt_lnk8aqyw/fixture.lcmfce
sha256=3981641ed3fa6f9bfac8425b248f546012501fac78094522816a22ab950c6d52 bytes=424
```

실제 parity59 χ,η capture는 신규 모델/GPU 런 금지 때문에 수행하지 않았다. 따라서 writer/schema/offline roundtrip은 PASS지만 production dump 파일의 존재와 Stage 3.1 수송 판별 입력 해소는 **UNRESOLVED**다.

## 9. 추가 정적 검증

```bash
python3 -m py_compile \
  scripts/wave32_r1_byte_invariant.py \
  scripts/cmf_chieta_roundtrip_selftest.py
git diff --check
rg '\bM_PI\b' src/lumina_plasma.c src/lumina_cmfgen.c
```

결과는 각각 RC=0, RC=0, 검색 결과 0건이다.

## 10. 변경 파일

- `src/lumina_plasma.c`: R3 shared field/GPU 및 collisional helper, pair 소비, runtime hook 배선, `M_PI_VAL`
- `src/lumina_element_wide.c`: R1 private view/commit telemetry, R5 counters, D3 정책/manifest
- `src/lumina_cmfgen.c`, `src/lumina_cmfgen.h`: R7 writer/schema/SHA-256, CPU fallback stubs, `M_PI_VAL`
- `src/lumina_cuda.cu`: post-damping R7 path/iter gate
- `Makefile`: CUDA feature define, R7 fixture target/clean
- `scripts/wave32_r1_byte_invariant.py`: SUPER_LEVELS 0/1 확장
- `scripts/cmf_chieta_writer_fixture.c`: 오프라인 writer fixture
- `scripts/cmf_chieta_roundtrip_selftest.py`: v1 reader/roundtrip/bitwise/SHA 검사

## 11. 정직하게 남는 UNRESOLVED

1. R1 COMMIT=1 양성 격리: 현 후보가 `boundary_gate`에서 막히므로 실제 commit 성공 사례가 없다. R6 M_V 창 뒤 B가 수행할 항목이다.
2. R4 signed DIE/역 autoionization: 이번 A2 소유 범위에서 구현하지 않았다. B의 기존 UNRESOLVED를 유지한다(`docs/CODEX_WAVE32_B_TEST.md:431`).
3. R6 M_V 경계 창: 이번 A2 범위 밖이며 구현되지 않았다. 따라서 기존 s0/s8 `SCOPE_FAIL`을 성공으로 재분류하지 않았다.
4. R7 실제 parity59 capture/수송 판별: 신규 GPU/모델 런 금지에 따라 실행하지 않았다. 오프라인 schema 시험으로 대체해 PASS를 가장하지 않았다.

## 12. 규율 준수

- 신규 clamp/floor/cap: **0**
- 기존 pair inverse-Saha `1e30` cap 복제/변경: **0**
- 신규 모델 실행: **0**
- GPU 실행: **0**
- frozen replay: R1 불변식 및 COMMIT 음성 대조만 실행
- 커밋: **하지 않음**
