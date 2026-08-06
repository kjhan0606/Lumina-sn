# A2-07 구현 보고서 — CPU population 온도·rate 소유권

상태: **IMPLEMENTED / LOCAL VALIDATION PASS / PHYSICAL LANES PENDING FABLE DRIVER**

명세: `docs/SPEC_A2_07_V1.md`  
기준 HEAD: `ece5aef8e192e2166b647ee00aae5fdd1f935a1c`  
구현: Codex, 검수·운전석: fable  
작성일: 2026-08-06

대형 36-case 배터리와 lageunha CHAIN/ORACLE_INPUT 물리 gate는 요청에 따라 Codex가
실행하지 않았다. 따라서 이 문서는 구현 폐합 후보 보고서이며, 두 물리 lane과 classic
paired-run 수치는 `BLOCKED_DRIVER_NOT_RUN`이다. 합성 self-check PASS를 물리 PASS로
재해석하지 않았다.

## 1. 구현 결론과 변경면

- `src/population_contract.h`, `src/population_contract.c`: 21개 population 상태,
  `T_e`/atomic SHA-256 stamp, energy-shift+compensated 단일 `Z(T_e)`, rank 검사,
  superlevel 합산, work-buffer transaction, 필수 카운터를 구현했다.
- `src/lumina.h:378-380,449,616-635`: `T_e_generation`, 최상단 원인 상태,
  partition/within-superlevel stamp, required/committed generation을 배치했다.
- `src/lumina_plasma.c:2272-2290`: 별도 `partition_functions_Te` 없이
  `atom->partition_functions` 하나만 게시한다. invalid/missing `T_e`와 stale hash는
  floor 없이 실패한다.
- `src/lumina_plasma.c:2459-2774`: ion stage와 electron-density 반복이 같은
  checked adjacent-stage helper를 사용한다. 생산 광이온화는 A2-05 canonical BF
  `Gamma`만 기여하며 legacy Saha는 observer shadow다. 기존 0.5 damping, 5%, 100회,
  보존 방정식은 바꾸지 않았고 미수렴은 게시하지 않는다.
- `src/lumina_plasma.c:15200-17155`: BF는 checked canonical view, BB는
  `B_lu*Jbar`, `B_ul*Jbar`, `A_ul` 분리율을 쓴다. raw J/legacy selector는
  `A2_06_DIAGNOSTIC_SHADOW` 안에서만 관측되며 첫 matrix write 전에 canonical 값으로
  교체된다. singular/isolated/non-finite solve는 rank/solve failure다.
- `src/lumina_plasma.c:6687-6765,17705-18121`: ion/level/`n_e`/partition은 모두
  work buffer에서 계산한 뒤 한 generation으로 commit한다. 실패 시 공개 포인터와 stamp를
  복구하고 generation을 올리지 않는다.
- `compute_radiative_equilibrium_te`는 자격 여부를 반환한다. ratio/no-root/floor/hybrid
  fallback은 `T_e_generation=0`이 되어 population에서 `POP_INVALID_TE`로 막힌다. 완전한
  명시적 fixed-`T_e` profile 또는 fallback 없는 해만 generation을 얻는다.
- legacy population fallback 설정과 `LUMINA_FROZENIN`은 생산 A2-07 solve에서
  `POP_FORBIDDEN_FALLBACK`로 거부한다. exact zero만 유효한 0이다.

변경 출력 allowlist는 CPU ion/level/superlevel population, `n_e`, partition과 population
상태/진단뿐이다. opacity·emissivity·spectrum 식은 바꾸지 않았다.

## 2. A2-01 원장 18행 종결표

동일 표를 `docs/A2_01_DISPOSITION_LEDGER.md` A2-07 ADDENDUM에도 추가했다.

| # | 고정 ID | 구현 후 위치 | 종결 |
|---:|---|---|---|
| 1 | A2-05 old9160 T_rad | `lumina_plasma.c:9279-9292` | `bf_rate_pop` → `LTE@T_e` |
| 2 | A2-05 old9162 W | `:9279-9292` | dilution/인자 제거 |
| 3 | A2-05 old11943 W | `:12082` | RADEQ W 전달 제거 |
| 4 | A2-05 old11943 T_rad | `:12082` | RADEQ `Te_lag` accessor |
| 5 | A2-05 old13672 W | `:13835-13836` | coupled W 전달 제거 |
| 6 | A2-05 old13672 T_rad | `:13835-13836` | coupled `LTE@T_e` |
| 7 | A2-06 old4879 T_rad | `:4938-4949` | macro/k-packet `LTE@T_e` |
| 8 | A2-06 old4880 W | `:4938-4949` | metastable dilution 제거 |
| 9 | A2-06 old12093 W | `:12230-12240` | RADEQ lower accessor |
| 10 | A2-06 old12100 W | `:12247-12257` | RADEQ upper accessor |
| 11 | A2-06 old13739 W | `:13903-13913` | coupled lower accessor |
| 12 | A2-06 old13743 W | `:13919-13929` | coupled upper accessor |
| 13 | BASE old2081 T_rad | `:2272-2284` | 단일 `Z(T_e)` 정본 |
| 14 | BASE old2082 W | `:2272-2284` | partition dilution/중복 저장소 제거 |
| 15 | BASE old7402 T_rad | `:7575-7585,7683-7693` | BF solved 또는 `LTE@T_e` |
| 16 | BASE old7403 W | 같은 위치 | lower/upper dilution 제거 |
| 17 | BASE old17832 T_rad | `:18100-18108` | dump는 `T_e`+population generation |
| 18 | BASE old17833 W | `:18085,18100-18108` | dump `T_rad/W` 제거 |

machine evidence: `validation/a2_07/A2_07_STATIC_CENSUS.json`. checker가 18개 ID의
집합 크기·유일성과 별도 partition 저장소 0건을 검사한다.

## 3. 원장 밖 17군 처분표

| 군 | 구현 후 처분 | 증거/후속 |
|---|---|---|
| partition 정본/fallback | 단일 `population_partition_build`; invalid 즉시 실패 | census PASS |
| 중복 `partition_functions_Te` | 선언·할당·작성·소비 제거 | 전역 hit 0 |
| Sobolev LTE 합성 | 공개 solved population 또는 `population_lte_level_fraction` | A2-08 식 인계 |
| macro/k-packet fallback | `LTE@T_e`, invalid fail | census transition root |
| recombination destination | 동일 partition/pop generation accessor | A2-09 target topology 인계 |
| BF opacity/rate population | solved 우선, reference는 `LTE@T_e` | opacity 식 A2-08 |
| `bf_rate_pop` | `T_e` 전용 signature | `:9279-9292` |
| all-level Gamma weight | solved NLTE 또는 동일 partition | checked BF helper |
| RADEQ untracked population | lower/upper 동일 accessor | A2-10 energy 식 인계 |
| coupled untracked population | RADEQ와 동일 accessor | 독립 fallback 0 |
| isolated-row anchor | rank-incomplete로 실패 | `population_dense_rank_check` |
| force/singular fallback | forbidden config/solve failure | Boltzmann publish 0 |
| within-superlevel 분배 | partition stamp 검증 후 work-buffer publish | `:17547-17625` |
| Boltzmann dump | `T_e`와 generation만 출력 | `:18085-18108` |
| ion rate 선택 복제 | ion/electron 경로가 같은 checked helper 공유 | Saha shadow 무기여 |
| level matrix BF/BB 조립 | checked BF/BB만 ACM에 기여 | diagnostic shadow allowlist |
| transactional publish/counters | 4배열 atomic commit, 실패 시 0 publish | selftest+`nlte_free` summary |

정적 checker의 production call-graph 금지 읽기는 0건이다. 좁은 allowlist는 (1) A2-10
온도/energy solve의 `T_rad/W`, (2) A2-13 GPU population, (3) matrix의 명시적 A2-06
observer shadow, (4) ion Saha observer shadow 네 항목이며 파일·함수·root·symbol·이유·후속
단계를 JSON에 기록했다.

## 4. validity, 카운터, rc와 게시 불변식

`PopulationStatus`는 명세의 `POP_OK`부터 `POP_FORBIDDEN_FALLBACK`까지 모두 구분한다.
A2-05/06의 STALE, UNSAMPLED, OOG, MISS, profile/qhash mismatch는 numerical zero로
소비되지 않는다. BF/BB generation이 한쪽만 다르면 solve가 시작되지 않는다.

정상 selftest의 실제 요약은 다음과 같다.

```text
[A2-07][POP-VIEW] pop_generation_required=4 pop_generation_committed=4 pop_shells_attempted=3 pop_shells_published=3 pop_bf_terms=2 pop_bb_terms=2 pop_exact_zero_terms=0 pop_blocked_stale=0 pop_blocked_unsampled=0 pop_blocked_oog=0 pop_blocked_miss=0 pop_blocked_profile=0 pop_blocked_qhash=0 pop_blocked_te=0 pop_blocked_partition=0 pop_rank_incomplete=0 pop_ne_not_converged=0 pop_solve_failed=0 pop_nonfinite=0 pop_generation_mismatch=0 pop_fallback_attempts=0 pop_partial_publish_attempts=0
```

`nlte_free`가 이 형식을 정확히 한 번 출력한다. C selftest는 invalid `T_e` 후 partition
bytes 불변, transaction 중간 NaN 후 ion/level/`n_e`/generation 불변, stale bit/hash와
generation, 모든 BF/BB invalid 상태, singular rank, superlevel membership를 검사한다.
게이트 rc는 명세대로 0/2/3/4/5를 사용한다.

## 5. L-2 gate·오라클·음성대조

계약 artifact는 `validation/a2_07/A2_07_ORACLE_CONTRACT.json`, 구현은
`scripts/a2_07_population_gate.py`다. s0-s8만 판정하고 CMFGEN 8개 필수 source,
file별 동일 generation, BF/BB/geometry/shell/crosswalk/binary/source/environment/`T_e`
hash를 확인한다. truth-active는 Lumina 결과 전에 고정되고 truth 분모 coverage와 CI
half-width를 별도 검사한다.

L-2ion, L-2level physical, internal partition은 운전석 실행 때 각각
`A2_07_<LANE>_L2ION_RESULT.json`, `A2_07_<LANE>_L2LEVEL_RESULT.json`,
`A2_07_<LANE>_PARTITION_CPU_RESULT.json`으로 분리된다. 동결 9이온×s0-s8의 81행과
population/BF/BB/total flow는 `A2_07_FROZEN_ION_CONTRIB.csv`로 기록된다.

현재 물리 판정은 다음과 같다.

| layer | CHAIN | ORACLE_INPUT | 이유 |
|---|---|---|---|
| L-2ion | BLOCKED_DRIVER_NOT_RUN | BLOCKED_DRIVER_NOT_RUN | lageunha 입력/실행 대기 |
| L-2level physical | BLOCKED_DRIVER_NOT_RUN | BLOCKED_DRIVER_NOT_RUN | 동일 |
| partition CPU | — | PASS_SELFTEST_ONLY | 실제 loaded-ion 전 범위 driver 대기 |

소규모 deterministic self-check에서는 baseline PASS 후 다음 네 child가 모두 실제 rc 4,
wrapper rc 0으로 검출되었다.

| ID | marker | poison | 결과 |
|---|---|---|---|
| N1 | `A2_07_NEG_STAGE_SWAP` | 인접 stage 교환 | child 4 / wrapper 0 |
| N2 | `A2_07_NEG_NEIGHBOR_NE` | 실제 인접 depth `n_e` 차용 | child 4 / wrapper 0 |
| N3 | `A2_07_NEG_TRAD_FOR_TE` | partition+level에 `T_rad` 대입 | child 4 / wrapper 0 |
| N4 | `A2_07_NEG_LEVEL_SHUFFLE` | crosswalk 고정 뒤 값 순열 | child 4 / wrapper 0 |

증거: `validation/a2_07/A2_07_GATE_SELF_CHECK.json`.

## 6. classic debt 17항목

Codex가 large paired-run을 하지 않았으므로 수치를 만들지 않았다. 아래 상태는 모두
`BLOCKED_DRIVER_NOT_RUN`이며 운전석의 `A2_07_CLASSIC_SWEEP.json`이 수치와 단위를 채운
뒤에만 제안 상태로 승격할 수 있다.

| ID | 현재 | 실행 후 제안 | 후속 |
|---|---|---|---|
| H01 | BLOCKED_DRIVER_NOT_RUN | PARTIAL_MEASURED | A2-10/A2-17 |
| H04 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-18 |
| H05 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-13 |
| H11 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-13 |
| H15 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-10 |
| S03 | BLOCKED_DRIVER_NOT_RUN | PARTIAL_MEASURED | A2-08 |
| S04 | BLOCKED_DRIVER_NOT_RUN | PARTIAL_MEASURED | A2-13 |
| S09 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-09/A2-13 |
| S14 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-13 |
| S15 | BLOCKED_DRIVER_NOT_RUN | PARTIAL_MEASURED 또는 BLOCKED_RANK_INCOMPLETE | A2-13 |
| S16 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-18 |
| G01 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-13 |
| G02 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-13 |
| G03 | BLOCKED_DRIVER_NOT_RUN | PARTIAL_MEASURED | A2-13 |
| G05 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-13 |
| G06 | BLOCKED_DRIVER_NOT_RUN | PARTIAL_MEASURED | A2-08/A2-13 |
| G10 | BLOCKED_DRIVER_NOT_RUN | MEASURED_OPEN | A2-18 |

현재 blocked artifact는 `validation/a2_07/A2_07_CLASSIC_SWEEP.json`, parser self-check는
`A2_07_CLASSIC_SWEEP_SELF_CHECK.json`, census 제안만
`CLASSIC_DEBT_CENSUS_A2_07_PROPOSED.diff`에 있다. `docs/CLASSIC_DEBT_CENSUS.md`는
수정하지 않았다.

## 7. 로컬 검증과 산출물

실행한 검증:

```bash
make lumina
make selftest_a2_03_radiation_field && ./selftest_a2_03_radiation_field
make selftest_a2_04_commit && ./selftest_a2_04_commit
make selftest_a2_05_bf_rate && ./selftest_a2_05_bf_rate
make selftest_a2_06_line_jbar && ./selftest_a2_06_line_jbar
make selftest_a2_06_dual_commit && ./selftest_a2_06_dual_commit
make selftest_a2_07_population && ./selftest_a2_07_population
python3 -m py_compile scripts/a2_07_population_census.py scripts/a2_07_population_gate.py scripts/a2_07_classic_sweep.py scripts/a2_07_regression_ledger.py
python3 scripts/a2_07_population_census.py --output validation/a2_07/A2_07_STATIC_CENSUS.json
python3 scripts/a2_07_population_gate.py --self-check --output validation/a2_07/A2_07_GATE_SELF_CHECK.json
python3 scripts/a2_07_classic_sweep.py --self-check --output validation/a2_07/A2_07_CLASSIC_SWEEP_SELF_CHECK.json
bash -n scripts/run_a2_07_grammar_debug.sh scripts/run_a2_07_lageunha.sh
```

`make lumina`와 여섯 selftest, gate/census/classic self-check, Python/Bash 문법 검사는
모두 PASS했다. 빌드에는 A2-07 이전부터 존재한 warning이 남지만 error는 없다.
grammar-debug 상세 로그와 hash는 `validation/a2_07/grammar_debug/`에 생성된다.

현재 주요 artifact:

| artifact | 역할 |
|---|---|
| `A2_07_STATIC_CENSUS.json` | 18 ID, 17군, call-graph/allowlist |
| `A2_07_GATE_SELF_CHECK.json` | deterministic baseline+N1–N4 |
| `A2_07_ORACLE_CONTRACT.json` | lane/manifest/threshold 계약 |
| `A2_07_CLASSIC_SWEEP.json` | large driver 미실행 BLOCKED 상태 |
| `A2_07_REGRESSION_LEDGER.jsonl` | 정확히 한 object; fable signoff 대기 |

최종 SHA-256:

| artifact | SHA-256 |
|---|---|
| `A2_07_STATIC_CENSUS.json` | `0e56a3a8c2f8a498c260711edacac85cb4174b0439268cb37b3084cad7c24617` |
| `A2_07_GATE_SELF_CHECK.json` | `7dc5bcb620bbcf3d715f90bbf50ac16b6d67c441aaa1947f2f799e2498b9bfbd` |
| `A2_07_ORACLE_CONTRACT.json` | `42354355a4af11dd0cfd43fe001b79982378479cc7ac52e05e6f412c26637118` |
| `A2_07_CLASSIC_SWEEP.json` | `55e06c5f4e3d057d4ce2009529f09ec6e3f4fe66ce9dd81ade51d4841a3f0a9a` |
| `A2_07_CLASSIC_SWEEP_SELF_CHECK.json` | `f343f9e3fc5fe8acf2b7973a431994320a1da1f8c9f311e0a59a0eb45a53c52b` |
| `CLASSIC_DEBT_CENSUS_A2_07_PROPOSED.diff` | `cf3322f3a1a1f915b138c311e7ef1eaeb4a94a9fe906c467d40a39d83b81bb08` |
| `A2_07_REGRESSION_LEDGER.jsonl` | `f5ea01a21731c55165bf3f26680b97694da4e797196d0766d3cd1a268061e92a` |

개정 8 grammar-debug snapshot의 source-tree implementation hash는
`20202349855f36ea3026e960d2e3e8f0388b93e8567bde2dad924c4387d2f48f`이다. 개정 9
수리 뒤 source hash와 회귀 대장 재발행은 전 배터리를 소유한 fable 운전석 몫이며,
이 로컬 수리에서는 과거 실행 증거를 덮어쓰지 않았다.

## 8. fable 운전석 실행 명령

grammar-debug:

```bash
ssh grammar "ssh grammar-debug 'bash /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_a2_07_grammar_debug.sh'"
```

lageunha에서는 동일 generation으로 사전 생성·동결한 두 manifest와 classic metric
파일을 지정한다.

```bash
ssh lageunha uptime
ssh lageunha 'A2_07_CHAIN_INPUT=/absolute/read-only/A2_07_CHAIN_INPUT.json A2_07_ORACLE_INPUT=/absolute/read-only/A2_07_ORACLE_INPUT.json A2_07_CLASSIC_METRICS=/absolute/read-only/A2_07_CLASSIC_METRICS.json bash /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_a2_07_lageunha.sh'
```

driver는 전 배터리, A2-05, A2-06 상태, 두 A2-07 lane, 네 child negative, classic sweep,
artifact/source/input hash와 단일 회귀 대장을 생성한다. upstream BLOCKED는 그대로
보존한다.

## 9. 남은 위험과 A2-08 인계

- 물리 CHAIN/ORACLE_INPUT, 81행 동결 이온 실측, truth `f_cov`/CI, loaded-ion 전체
  partition, classic paired metrics는 fable 운전석 실행 전까지 BLOCKED다.
- A2-06의 `BLOCKED_MISSING_RATE_EXPORT`가 해소되지 않으면 CHAIN/L-2level은 이를
  그대로 승계해야 한다. ORACLE_INPUT도 separated BF/BB export가 없으면
  `BLOCKED_MISSING_RATE_EXPORT`다.
- `compute_radiative_equilibrium_te`의 no-root/floor/hybrid 결과에는 generation을 주지
  않는다. 해법 자체는 A2-10 부채이며 A2-07에서 고치지 않았다.
- GPU population과 GPU의 `T_rad/W` fallback은 A2-13 범위다.
- CE/topology/rank/seed 독립성과 완전수렴은 A2-13/A2-18에 남는다.

A2-08은 `population_committed_generation`과 partition/within-SL stamp가 fresh인 공개
population만 읽어 opacity/Sobolev tau를 계산해야 한다. `T_rad/W` Boltzmann population을
다시 만들 수 없다. A2-06에서 재배치한 line-source/blanketed-heating 4행, G06 Si tau
skip, opacity generation의 exact binding과 stale failure를 함께 닫아야 한다. recombination
core/target·emissivity는 A2-09, `T_e` 해법 H01/H15는 A2-10으로 인계한다.

## 10. 의도된 τ 변화

A2-07 명세 §6의 population 소유권 전환에 따라 비-NLTE Sobolev 합성의 LTE 기준은
`LTE@T_rad,W`가 아니라 단일 정본 `LTE@T_e`다. 준위 fraction은 동일한
`population_lte_level_fraction`을 통해 `E_i-E_0`, `T_e`, dilution 없음으로 계산한다.
따라서 옛 `T_rad/W` τ와의 bit identity는 더 이상 합격 조건이 아니며 코드 회귀로
해석하지 않는다.

fable에서 관측한 `active_tau_bit_differences=1,234,800`은 생산 τ의 물리 오구현이
아니었다. 픽스처가 새 식을 독립 재현하면서 `E-E_0` 규약과 생산 helper의 연산 순서를
완전히 복제하지 않아 대수적으로 같은 값 다수가 1 ULP 차이로 bitwise 판정에 걸렸다.
개정 9 기준은 고정 deck의 `active_lines=2,211,572`,
`active_tau_bit_differences=0`, `active_tau_fnv64=1cfbc8dba0b0f23f`로 사전등록하며,
픽스처와 Z-INERT runner가 세 값을 함께 강제한다. 이 처분은 생산 물리식 수정이 아니라
새 population 물리에 대한 잣대 개정이다.
