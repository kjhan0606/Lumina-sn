## Phase 1.6 정정 확인

B15가 지적한 경미 2건은 Phase 1.6에서 수정됐다.

- `cmfgen_source_evidence.csv`의 newline 기준 물리 행은
  `cmfgen_sub.f:4421`, `cmfgen_sub.f:4423`, `mod_cmfgen.f:211`로 정정됐다.
  비교기도 Fortran form-feed를 행 구분자로 세지 않고 `\n`만 물리 행으로 센다.
- 구현 보고서의 snapshot 설명은 s0/s8의 16행만 `same_snapshot`, s43의
  8행은 상대차 `-1.8805437716e-3`의
  `different_snapshot_or_output_generation`임을 명시하도록 정정됐다.

아래 내용은 정정 전 B15 독립 시험 기록으로 보존한다. 두 메타데이터 결손에
대한 조건부 FAIL은 해소됐으며, 계산·결정론·OFF 시험 결과에는 변경이 없다.

## 종합 판정 (정정 전 기록)

**Gate B Phase 1.5: 조건부 FAIL — 산출물 메타데이터 정정 필요.**

계산·결정론·커버리지·OFF 오브젝트는 모두 통과했습니다. 하지만 `cmfgen_source_evidence.csv`의 `physical_line_1based` 3건이 실제 newline 기준 물리 행과 다릅니다. 또한 구현 보고서의 “모든 셀 동일 snapshot” 설명은 실제 s43 검사표와 모순됩니다.

소스 편집, GPU 실행, `git` 명령은 수행하지 않았습니다. 재실행 산출물은 `/tmp`에만 만들었습니다.

## 1. 명령과 exit

| 검증 | 실행 명령 | exit/결과 |
|---|---|---:|
| oracle 강제 재빌드 | `make -B bench_frozen_oracle` | 0 |
| 1차 실행 | `./bench_frozen_oracle logs/coevolve_consume_parity50 data/tardis_reference_toy06_19p48d_sivcaiv /tmp/gateb15_B1.emRYTr` | 0 |
| 2차 실행 | 동일 명령, 출력 `/tmp/gateb15_B2.kCZHjT` | 0 |
| 1차↔2차 CSV | `cmp ...s{0,8,43}.csv` | 셋 모두 0 |
| 1차↔제출 CSV | `cmp ... phase1_5/lumina_oracle_cell_s{0,8,43}.csv` | 셋 모두 0 |
| 비교기 구문 검사 | `PYTHONPYCACHEPREFIX=/tmp/gateb15_pycache python3 -m py_compile scripts/oracle_compare_cmfgen.py` | 0 |
| 비교기 실행 | `python3 scripts/oracle_compare_cmfgen.py --out-dir /tmp/gateb15_cmp.MJ3RVD` | 0 |
| 비교기 산출물↔제출본 | 7개 산출물 `cmp` | 모두 0 |
| OFF 기본 오브젝트 | `gcc ... -c src/lumina_plasma.c -o .../default.o` | 0 |
| OFF 명시 오브젝트 | `gcc ... -ULUMINA_FROZEN_ORACLE -c ... -o .../explicit_off.o` | 0 |
| OFF 오브젝트 비교 | `cmp default.o explicit_off.o` | 0 |
| OFF 심볼 감사 | `nm ... \| awk '/lumina_oracle\|g_oracle/'` | 두 오브젝트 모두 0개 |
| `n_e` identity 왕복 | CSV↔RVTJ 원문↔소스 선언 독립 검사 | 0 |
| 소스 physical-line 감사 | CSV 기재 행↔newline 기준 실제 행 | **1; 3건 불일치** |

재빌드에는 기존 C 소스 경고가 있었지만 오류 없이 완료됐습니다.

## 2. oracle 2회 byte-identical

| 셸 | 데이터 행 | 1차↔2차 | 제출본↔재실행 | SHA-256 |
|---:|---:|---:|---:|---|
| s0 | 182 | exit 0 | exit 0 | `526f490fee030ce9573ec1d00267c706991a8865539ebe9c432eefd5efd94d7f` |
| s8 | 182 | exit 0 | exit 0 | `8210eef19a569c452acb6b7cea1b7d29c9c3022efa73f8baccdfced90683d052` |
| s43 | 182 | exit 0 | exit 0 | `4f7819f51ffb04b02fa06768755179517ddd7aa5252a1f458d0afe8609f15e07` |

각 셀은 `state 28 / input 12 / bf 64 / ff 5 / bb 48 / collisional 16 / thermal 9`행이며, unavailable 행의 빈 사유는 0건입니다.

C2 적재/소비도 재현됐습니다.

| 셸 | loaded/expected | positive 소비 | fallback 소비 |
|---:|---:|---:|---:|
| s0 | 1000/1000 | 5,968,990 | 1,953,160 |
| s8 | 1000/1000 | 5,119,009 | 2,803,141 |
| s43 | 1000/1000 | 4,344,072 | 3,578,078 |

## 3. 확장 커버리지 총괄

엄격한 동일 수량 `compared` 기준:

- Phase 1: **33/484 = 6.82%**
- Phase 1.5: **79/546 = 14.47%**
- 증가: **+46행, +7.65%p, 2.39배**
- 별도 비동일 문맥 수치: `context_only_nonidentical` 9행
- 문맥값까지 포함한 숫자 쌍: **88/546**

| category | P1 compared/total | P1.5 compared | context | Lumina unavailable | CMFGEN unavailable | P1.5 total |
|---|---:|---:|---:|---:|---:|---:|
| bb | 18/124 | 8 | 0 | 120 | 16 | 144 |
| bf | 9/192 | 18 | 0 | 60 | 114 | 192 |
| collisional | 0/48 | 0 | 0 | 40 | 8 | 48 |
| ff | 0/15 | 0 | 3 | 0 | 12 | 15 |
| input | 0/0 | 0 | 0 | 0 | 36 | 36 |
| state | 6/84 | 44 | 0 | 18 | 22 | 84 |
| thermal | 0/21 | 9 | 6 | 3 | 9 | 27 |
| **합계** | **33/484** | **79** | **9** | **241** | **217** | **546** |

bb의 −10은 단순 후퇴가 아니라, Phase 1에서 C1 fallback을 대표 Jbar로 수치 대응했던 행을 Phase 1.5가 실제 raw-Jbar 기록이 있는 Si 행만 인정하도록 엄격화한 결과입니다.

### 수량별 확장 대조표 전문

| category | quantity | P1 compared/total | P1.5 compared | context | Lumina unavail | CMFGEN unavail | P1.5 total | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bb | `R_lu_radiative` | 0/24 | 0 | 0 | 20 | 4 | 24 | 0 |
| bb | `R_ul_spontaneous` | 0/24 | 0 | 0 | 20 | 4 | 24 | 0 |
| bb | `R_ul_stimulated` | 0/24 | 0 | 0 | 20 | 4 | 24 | 0 |
| bb | `jbar_input_raw` | 4/14 | 4 | 0 | 20 | 0 | 24 | 0 |
| bb | `jbar_representative` | 14/24 | 4 | 0 | 20 | 0 | 24 | −10 |
| bb | `sobolev_beta` | 0/14 | 0 | 0 | 20 | 4 | 24 | 0 |
| bf | `Gamma_photoion_total` | 9/24 | 9 | 0 | 15 | 0 | 24 | 0 |
| bf | `alpha_recomb_spont` | 0/24 | 0 | 0 | 15 | 9 | 24 | 0 |
| bf | `alpha_recomb_stim` | 0/24 | 0 | 0 | 15 | 9 | 24 | 0 |
| bf | `alpha_recomb_total` | 0/24 | 9 | 0 | 15 | 0 | 24 | +9 |
| bf | `chi_bf_at_1000A` | 0/24 | 0 | 0 | 0 | 24 | 24 | 0 |
| bf | `chi_bf_at_5000A` | 0/24 | 0 | 0 | 0 | 24 | 24 | 0 |
| bf | `eta_bf_at_1000A` | 0/24 | 0 | 0 | 0 | 24 | 24 | 0 |
| bf | `eta_bf_at_5000A` | 0/24 | 0 | 0 | 0 | 24 | 24 | 0 |
| collisional | `C_lu` | 0/24 | 0 | 0 | 20 | 4 | 24 | 0 |
| collisional | `C_ul` | 0/24 | 0 | 0 | 20 | 4 | 24 | 0 |
| ff | `chi_ff_at_1000A` | 0/3 | 0 | 0 | 0 | 3 | 3 | 0 |
| ff | `chi_ff_at_5000A` | 0/3 | 0 | 0 | 0 | 3 | 3 | 0 |
| ff | `cooling_ff_grid` | 0/3 | 0 | 3 | 0 | 0 | 3 | 0 |
| ff | `eta_ff_at_1000A` | 0/3 | 0 | 0 | 0 | 3 | 3 | 0 |
| ff | `eta_ff_at_5000A` | 0/3 | 0 | 0 | 0 | 3 | 3 | 0 |
| input | `bf_rate_estimator_bins_expected` | 0/0 | 0 | 0 | 0 | 3 | 3 | 0 |
| input | `bf_rate_estimator_bins_loaded` | 0/0 | 0 | 0 | 0 | 3 | 3 | 0 |
| input | `bf_rate_estimator_fallback_consumptions` | 0/0 | 0 | 0 | 0 | 3 | 3 | 0 |
| input | `bf_rate_estimator_positive_consumptions` | 0/0 | 0 | 0 | 0 | 3 | 3 | 0 |
| input | `raw_jbar_ion_recorded` | 0/0 | 0 | 0 | 0 | 24 | 24 | 0 |
| state | `T_e` | 3/3 | 3 | 0 | 0 | 0 | 3 | 0 |
| state | `T_rad` | 0/3 | 0 | 0 | 0 | 3 | 3 | 0 |
| state | `W` | 0/3 | 0 | 0 | 0 | 3 | 3 | 0 |
| state | `b_k_representative` | 0/24 | 14 | 0 | 10 | 0 | 24 | +14 |
| state | `ion_fraction` | 0/24 | 0 | 0 | 8 | 16 | 24 | 0 |
| state | `n_e` | 3/3 | 3 | 0 | 0 | 0 | 3 | 0 |
| state | `n_ion` | 0/24 | 24 | 0 | 0 | 0 | 24 | +24 |
| thermal | `cooling_adiabatic` | 0/3 | 0 | 0 | 0 | 3 | 3 | 0 |
| thermal | `cooling_bb_collisional` | 0/3 | 3 | 0 | 0 | 0 | 3 | +3 |
| thermal | `cooling_bf` | 0/3 | 0 | 3 | 0 | 0 | 3 | 0 |
| thermal | `cooling_bf_net` | 0/0 | 3 | 0 | 0 | 0 | 3 | +3 |
| thermal | `cooling_ff` | 0/0 | 0 | 3 | 0 | 0 | 3 | 0 |
| thermal | `heating_MA_LINE_DESTRUCT` | 0/3 | 0 | 0 | 3 | 0 | 3 | 0 |
| thermal | `heating_deposition` | 0/3 | 0 | 0 | 0 | 3 | 3 | 0 |
| thermal | `heating_photoion` | 0/3 | 0 | 0 | 0 | 3 | 3 | 0 |
| thermal | `thermal_net` | 0/3 | 3 | 0 | 0 | 0 | 3 | +3 |

재실행 결과는 제출 [CMFGEN 비교표](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_5/oracle_vs_cmfgen.csv) 및 나머지 6개 산출물과 전부 byte-identical입니다.

## 4. OFF 무접촉 오브젝트

| 빌드 | oracle 심볼 | SHA-256 |
|---|---:|---|
| 매크로 미정의 | 0 | `ed9973e420d3d0af9fdd73aca45d6f8750284fb43bd066a4a4b0995846d3dd55` |
| `-ULUMINA_FROZEN_ORACLE` | 0 | `ed9973e420d3d0af9fdd73aca45d6f8750284fb43bd066a4a4b0995846d3dd55` |

`cmp exit 0`. 즉 OFF에서 oracle 코드가 오브젝트에 남거나 기본 오브젝트를 변화시킨 증거는 없습니다.

## 5. `n_e` 단위 왕복

RVTJ 원문 행과 값은 정상입니다.

| 셸 | depth | RVTJ 행 | raw | post-conversion | 판정 |
|---:|---:|---:|---:|---:|---|
| s0 | 67 | 61 | `4.8528721000e+09` | `4.8528721000e+09` | identity |
| s8 | 54 | 59 | `7.3410434000e+08` | `7.3410434000e+08` | identity |
| s43 | 10 | 54 | `2.2602505000e+05` | `2.2602505000e+05` | identity |

직접 확인한 소스 사슬도 유효합니다.

- RVTJ header writer: `cmfgen_sub.f:4421`
- RVTJ value writer `WRITE ... ED`: `cmfgen_sub.f:4423`
- `ED(:)` 단위 선언 `Electron density (#/cm^3)`: `mod_cmfgen.f:211`

따라서 **값과 단위의 왕복 자체는 PASS**입니다.

다만 [cmfgen_source_evidence.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_5/cmfgen_source_evidence.csv)의 물리 행 번호는 틀립니다.

| evidence | CSV 기재 | 실제 newline 물리 행 | 판정 |
|---|---:|---:|---|
| `ne_writer_header` | 4490 | 4421 | 불일치 |
| `ne_writer_value` | 4492 | 4423 | 불일치 |
| `ne_unit_declaration` | 213 | 211 | 불일치 |

원인은 [비교기](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/oracle_compare_cmfgen.py:261)가 Fortran 파일에 있는 form-feed `\f`를 `splitlines()`로 별도 행처럼 세면서 결과를 `physical_line_1based`라고 기록하기 때문입니다. RVTJ에는 form-feed가 없어 3셀 왕복 행 번호는 정확합니다.

추가로 [snapshot consistency 표](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/gate_b_dual_oracle/phase1_5/cmfgen_snapshot_consistency.csv)는:

- s0/s8: 16행 `same_snapshot`, 최대 상대차 약 `5.91e-6`
- s43: 8행 모두 `different_snapshot_or_output_generation`, 상대차 `−1.8805437716e-3`

으로 기록합니다. 이는 “선택 셀/이온 전부 약 `5.75e-6` 동일 snapshot”이라고 쓴 [구현 보고서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/GATE_B_PHASE1_5_CODEX_A_REPORT.md:102)와 모순됩니다. 비교기 산출표 자체는 이 차이를 숨기지 않고 올바르게 표시합니다.
