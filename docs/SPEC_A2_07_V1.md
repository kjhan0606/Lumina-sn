# A2-07 구현 명세 V1 — 물질 population 온도·rate 소유권 이관

상태: **구현 전 규범 명세**

저작: Codex

검수·운전석: fable

구현: Codex

대상 단계: A-2 캠페인의 **A2-07 하나**

후속 구현 보고서: `docs/CODEX_IMPL_A2_07.md`

이 문서의 `MUST`, `MUST NOT`, `SHALL`, `FAIL`, `BLOCKED`는 시험 가능한 계약어다. 구현 편의를 이유로 완화할 수 없다.

## 0. 결론과 범위

A2-07의 단일 계약은 다음과 같다.

> CPU 물질 population을 계산할 때 partition·LTE/Boltzmann excitation은 오직 유효한 `T_e`를 사용하고, ion/level solver의 복사율은 A2-05의 generation-bound BF `Gamma` view와 A2-06의 generation-bound BB `LineJbarCache`만 사용한다. 입력·generation·solve가 유효하지 않으면 population을 게시하지 않고 명시적으로 실패한다.

이 단계가 바꾸는 것은 **주어진 `T_e`에서의 population 온도 소유권과 복사율 공급원·validity 전파**다. 다음은 범위 밖이다.

- 복사평형으로 `T_e` 자체를 푸는 일: A2-10.
- ionization/level SE의 항, 계수, topology, CE/DR 물리식, 시간적분법 또는 수렴법을 새로 설계하는 일: A2-13/A2-18 등 후속 단계.
- opacity·Sobolev tau·emissivity 식을 바꾸는 일: A2-08/A2-09.
- transport를 바꾸는 일: A2-11.
- GPU population 이관: A2-13. A2-07의 판정 대상은 CPU다.

다만 범위 밖 부채가 A2-07 경로에서 발화하면 숨기지 않는다. 계측하고, 유효한 population을 만들 수 없으면 `BLOCKED_*` 또는 `FAIL_*`로 끝낸다. 옛 population, Saha, 희석 Planck, 0, floor, NaN 치환으로 계속 실행하지 않는다.

## 1. 확인한 규범 기초와 기준점

구현자는 다음 파일과 폐합 커밋을 이 명세의 상위 계약으로 사용한다.

| 대상 | 저작 시 확인한 기준 |
|---|---|
| 온도·소유권·게이트 규범 | `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:57-112,155-190,264-372,428-459,662,689-710,720-773` |
| A2-01 원장 | `docs/A2_01_DISPOSITION_LEDGER.md:1-243`; 본문 157행, A2-05/A2-06 addendum 포함 |
| 오라클 안전대·동결 이온 | `docs/OUTSIDE_LOOP_POOL.md:1018-1029,1091-1119,1205-1227` |
| A2-05 설계·검수·폐합 | `docs/SPEC_A2_05_V2.md`, `docs/CODEX_REVIEW_A2_05.md`, `validation/a2_05/A2_05_CLOSURE.md`; commit `d8b987032377f4e64677df1215ffaaed781cb982` |
| A2-06 설계·구현·폐합 | `docs/SPEC_A2_06_V1.md`~`docs/SPEC_A2_06_V5.md`, `docs/CODEX_IMPL_A2_06.md`, `validation/a2_06/A2_06_CLOSURE.md`; commit `ece5aef8e192e2166b647ee00aae5fdd1f935a1c` |
| 앞 단계 커밋 | A2-01 `cf93109044abc0ae4f212b7e2b55bc7e4843017b`; A2-02 `bca476a7b675790cf393871b860b785a9ebd7860`; A2-03 `606f23a79c812e3edb532a07930a8f98f5817c7d`; A2-04 `bafd2bbdfbcdb7e84b6f573b37785e30185865f0` |

요청문에 적힌 `docs/CODEX_IMPL_A2_05.md`는 저작 시 작업트리에 존재하지 않았다. 이를 만들어 인용하지 않으며, 존재하는 A2-05 검수·폐합 문서와 커밋을 기준으로 삼는다.

상위 단계의 현재 상태를 하류가 덮어쓰면 안 된다.

- A2-05 CHAIN은 기전 검사는 통과했지만 truth-side `f_cov` 부족으로 물리 판정 자격이 없었던 셸이 있다. A2-07은 이를 자체 population PASS로 세탁하지 않는다.
- A2-06 L-1bb의 현재 상태는 `BLOCKED_MISSING_RATE_EXPORT`다. A2-07 CHAIN이 그 입력에 의존해 물리 판정을 할 수 없으면 `BLOCKED_UPSTREAM_L1BB`를 기록한다.
- ORACLE_INPUT은 upstream rate만 주입한다. CMFGEN population을 solver 입력으로 주입하여 정답을 복사해서는 안 된다.

## 2. 구현 산출물과 변경 허용면

구현 단계에서 허용되는 변경은 CPU population의 온도·rate 소비, 그 validity/계측, A2-07 게이트·fixture·보고서·원장 addendum다. 최소 산출물은 다음과 같다.

1. CPU 코드와 헤더의 population view/상태/카운터 이관.
2. `scripts/a2_07_population_gate.py`와 필요한 CPU selftest/fixture.
3. `validation/a2_07/` 아래 재현 가능한 manifest, CHAIN/ORACLE_INPUT 결과, negative-control 결과, classic sweep, 동결 이온 기여표, 회귀 대장.
4. `docs/A2_01_DISPOSITION_LEDGER.md`의 A2-07 addendum: 아래 18행의 구현 후 위치와 1:1 처분, 목록 밖 발견분.
5. `docs/CLASSIC_DEBT_CENSUS.md`는 **직접 편집하지 않는다**. 제안 diff만 `validation/a2_07/CLASSIC_DEBT_CENSUS_A2_07_PROPOSED.diff`와 구현 보고서에 남긴다.
6. `docs/CODEX_IMPL_A2_07.md` 한 편.

물리 출력 변경 allowlist는 단계 시작 전에 다음으로 제한한다.

- CPU ion fraction, CPU level/superlevel population, `n_e`, `Z(T_e)` 및 population validity/진단.
- opacity·emissivity·spectrum은 A2-07의 합격 observable이 아니다. population 변경의 하류 파급은 수치 diff로 기록하되 A2-08/A2-09 PASS로 해석하지 않는다.

덱과 `/gpfs` 입력은 불변이다. 로그인 노드에서는 읽기·grep·git·문서 작업과 허용된 빌드 외 연산을 하지 않는다. `/usr/bin/time`, commit, push를 사용하지 않는다.

## 3. 온도 계약

### 3.1 유일한 excitation/partition 온도

다음 계산의 입력 온도는 모두 같은 shell의 `plasma->T_e[s]`다.

- `Z(T_e)=sum_i g_i exp[-(E_i-E_0)/(k T_e)]`.
- LTE/Boltzmann level fraction과 superlevel 내부 분배.
- rate 계산에 필요한 detailed-balance/Boltzmann population factor.
- Boltzmann 진단 기준값.

이 경로의 함수 인자와 호출 그래프에서 `T_rad`, `W`, `T_e_T_rad_ratio`를 제거한다. `W=1`을 넘기는 호환 코드는 금지한다. `W`는 이 경로에 존재하지 않는 변수여야 한다. `T_rad`로부터 색온도·유효온도·fit 온도를 새로 파생하여 대신 쓰는 것도 금지한다.

`T_e` 배열이 없거나, shell 인덱스가 범위를 벗어나거나, 값이 non-finite 또는 `<=0`이면 `POP_INVALID_TE`다. 다음은 모두 금지한다.

- `T_e_T_rad_ratio*T_rad`, `T_rad`, 이웃 shell `T_e`, 1 K 같은 상수로 대체.
- `Z`를 `1`, `1e-300` 또는 이전 값으로 대체.
- level fraction을 1이나 0으로 대체.
- 오류가 난 shell만 옛 population으로 유지하면서 새 generation이라고 표시.

### 3.2 단일 partition 구현

생산 partition은 하나의 CPU helper와 하나의 저장소만 가져야 한다. helper는 `(atomic model, ion id, T_e)`만으로 값을 만들고 `T_rad/W/PlasmaState`를 받지 않는다. 수치 규약은 다음과 같다.

1. 각 ion에서 `E_0=min(E_i)`를 먼저 구해 지수를 `E_i-E_0`로 이동한다.
2. 유한한 `E_i`, 양의 정수 `g_i`, 유효한 runtime membership만 합한다.
3. compensated summation을 사용한다.
4. 합이 0/non-finite이면 floor하지 않고 `POP_INVALID_PARTITION`으로 실패한다.
5. `atom->partition_functions`와 `atom->partition_functions_Te` 같은 병렬 정본을 유지하지 않는다. 하나를 정본으로 승격하고 다른 하나는 제거한다. ABI 때문에 잠시 남으면 alias이며 별도 할당·별도 계산·별도 generation을 가질 수 없다.

현재 `src/lumina_plasma.c:2181-2232`의 parity 조건부 `T_e`와 `T_e` 부재 시 `T_rad,W` 유지 경로는 무조건 `T_e` 경로로 바뀌어야 한다. `src/lumina_plasma.c:4510-4529`의 별도 `partition_functions_Te` 작성도 단일 정본으로 합쳐야 한다.

### 3.3 파생 캐시 generation 결박

partition과 superlevel 내부 fraction은 다음 stamp를 가진 불변 view로 게시한다.

```text
required_population_generation
computed_population_generation
te_generation
te_manifest_sha256
atomic_model_sha256
n_shells
n_ions_or_levels
status
```

`te_manifest_sha256`는 shell 순서·단위·IEEE-754 값까지 포함한다. 소비 시 요구 generation, `T_e` hash, atomic-model hash, shape 중 하나라도 다르면 `POP_STALE_DERIVED_TEMPERATURE`로 즉시 실패한다. 캐시를 부분 갱신하거나 이전 generation의 일부를 재사용하지 않는다.

`T_e`는 A2-07에서 복사장 파생량이 아니다. A2-10이 공급한 입력으로 취급하며, 생성 주체와 generation을 manifest에 기록한다. `T_rad/W`에서 복원한 `T_e`에는 generation 자격을 부여하지 않는다.

## 4. rate 소비와 population 게시 계약

### 4.1 입력 view

CPU population solve 하나는 같은 solve token 아래 다음을 받는다.

- A2-05 BF: `RadiationField`의 checked canonical view와 `BfRateResult` 계열 `Gamma` 조회.
- A2-06 BB: checked line view와 `LineJbarCache`; `R_lu=B_lu*Jbar`, `R_ul^stim=B_ul*Jbar`, `R_ul^sp=A_ul`.
- 3절의 `T_e`/partition view.
- 기존 밀도·원자모형·충돌·재결합·CE/DR 입력. 이들의 물리식은 이 단계에서 변경하지 않는다.

BF와 BB view는 각자의 committed generation과 population solve가 요구한 generation이 정확히 일치해야 한다. 둘의 generation 의미가 다른 경우 manifest에 `(radfield_generation,line_cache_generation)` 쌍을 고정하며, 재시도 중 어느 한쪽만 바꾸지 않는다.

A2-05의 `VALID`와 `EXACT_ZERO`만 물리 입력이다. `EXACT_ZERO`는 유효한 수치 0이다. A2-05/06의 `STALE`, `UNSAMPLED`, `OOG`, `MISS`, profile/query-hash 불일치, disabled/uncommitted 상태는 수치 0이 아니며 solve를 차단한다.

### 4.2 ion solver

현재 ion stage 보존식·인접 stage 연결·정규화·전하중성 반복은 그대로 둔다. 단, 생산 경로의 광이온화 항은 A2-05 canonical `Gamma`만 받는다.

- `bf_rate_estimator`, `W B_nu(T_rad)`, nebular/dilute Saha를 새 rate의 대체 공급원으로 쓰지 않는다.
- rate view가 준비되지 않았다는 이유로 기존 Saha split로 되돌아가지 않는다.
- `num=den=0`일 때 이전 split 또는 Saha ratio를 반환하지 않는다. 실제로 모든 물리 항이 `EXACT_ZERO`인지와 연결 행의 rank를 검사하고, 결정 불능이면 `POP_RANK_INCOMPLETE`다.
- `n_e<=0`, non-finite ratio, overflow를 `1e10`, `0`, `1e30`, `1e-300`으로 고친 뒤 게시하지 않는다. 원인 상태를 반환한다.
- H04의 0.5/5%/100회 반복법 자체는 고치지 않지만, 종료 이유와 전하중성 잔차를 계측한다. 미수렴 결과는 게시하지 않는다.

`compute_ion_populations_shell`과 `compute_electron_density`에 복제된 rate 선택은 하나의 checked ion-rate helper를 공유해야 한다. 두 경로가 서로 다른 fallback 또는 generation을 고를 수 없다.

### 4.3 level solver

NLTE 행렬의 기존 항과 보존행은 유지하되 모든 활성 복사 항을 checked helper에서 받는다.

- BF 행렬 항은 A2-05 canonical `Gamma`.
- BB 상향·stimulated 하향은 A2-06 `Jbar`; spontaneous는 원자자료의 `A_ul`.
- population이 필요한 부수 경로도 공개된 solved population 또는 `LTE@T_e` reference만 받는다. 자체 `T_rad/W` Boltzmann population을 재합성하지 않는다.
- checked helper가 false/invalid를 반환했는데 호출자가 0을 사용해 행렬을 계속 조립하면 계약 위반이다.
- singular/non-finite solve, 강제 LTE 환경변수, 고립행은 `Boltzmann@T_rad` 결과를 물리 population으로 게시할 수 없다. `POP_SOLVE_FAILED`, `POP_RANK_INCOMPLETE` 또는 명시적 diagnostic-only 결과로 끝낸다.

`LUMINA_NLTE_FORCE_LTE_LEVELS`, legacy rate-source 선택, population fallback을 켜는 설정은 생산 A2-07 lane에서 설정 자체가 `FAIL_FORBIDDEN_FALLBACK_CONFIG`다. 별도 negative-control child process에서만 허용한다.

### 4.4 transactional publish

solve는 work buffer에 수행한다. 다음 조건이 모두 참일 때만 ion population, level population, `n_e`, partition view를 한 population generation으로 게시한다.

1. 모든 필수 `T_e`, BF, BB, atomic lookup이 유효하다.
2. partition과 superlevel fraction의 stamp가 일치한다.
3. ion closure와 전하중성 solve가 수렴했다.
4. level solve가 finite·rank-valid이고 보존잔차가 허용 범위다.
5. 결과가 finite, non-negative이며 명시적 physical zero와 missing이 구별된다.

실패하면 공개 배열과 `computed_population_generation`은 직전 committed 상태 그대로이며, 새 generation처럼 보이는 부분 갱신은 0건이어야 한다. 호출 스택 최상단은 nonzero rc와 원인 enum을 받는다.

## 5. A2-01 원장 18행 1:1 처분

행 식별자는 이관 전 원장 위치로 고정하고, 현행 위치는 저작 HEAD `ece5aef`에서 재측정했다. 한 원장 행을 합쳐 없애지 말고 구현 보고서에서 각각 종결한다.

| # | 출처 | 원장 행 | 현행 소비 위치 | A2-07 종결 조건 |
|---:|---|---|---|---|
| 1 | A2-05 재배치 | old `:9160 T_rad` | `src/lumina_plasma.c:9288` | `bf_rate_pop` 지수를 canonical `T_e` population accessor로 대체 |
| 2 | A2-05 재배치 | old `:9162 W` | `src/lumina_plasma.c:9290` | dilution 제거; `W` 인자 자체 제거 |
| 3 | A2-05 재배치 | old `:11943 W` | `src/lumina_plasma.c:12074` | 호출자가 `W`를 population 공급자에 전달하지 않음 |
| 4 | A2-05 재배치 | old `:11943 T_rad` | `src/lumina_plasma.c:12074` | 호출자가 `T_rad`를 전달하지 않음 |
| 5 | A2-05 재배치 | old `:13672 W` | `src/lumina_plasma.c:13805` | coupled 호출도 동일 accessor 사용 |
| 6 | A2-05 재배치 | old `:13672 T_rad` | `src/lumina_plasma.c:13805` | coupled 호출도 동일 accessor 사용 |
| 7 | A2-06 재배치 | old `:4879 T_rad` | `src/lumina_plasma.c:5009-5010` | fallback exponent를 `T_e` 정본 accessor로 대체 |
| 8 | A2-06 재배치 | old `:4880 W` | `src/lumina_plasma.c:5011` | metastable dilution 분기 제거 |
| 9 | A2-06 재배치 | old `:12093 W` | `src/lumina_plasma.c:12224` | lower population fallback의 dilution 제거 |
| 10 | A2-06 재배치 | old `:12100 W` | `src/lumina_plasma.c:12231` | upper population fallback의 dilution 제거 |
| 11 | A2-06 재배치 | old `:13739 W` | `src/lumina_plasma.c:13872` | coupled lower population의 dilution 제거 |
| 12 | A2-06 재배치 | old `:13743 W` | `src/lumina_plasma.c:13876` | coupled upper population의 dilution 제거 |
| 13 | 기존 partition | old `:2081 T_rad` | `src/lumina_plasma.c:2204` | partition을 무조건 `T_e`로 계산 |
| 14 | 기존 partition | old `:2082 W` | `src/lumina_plasma.c:2205,2230` | partition의 `W` 항 제거 |
| 15 | 기존 rate Boltzmann | old `:7402 T_rad` | `src/lumina_plasma.c:7530,7534,7564` | BF 부수 population을 solved 또는 `LTE@T_e`로만 공급 |
| 16 | 기존 rate Boltzmann | old `:7403 W` | `src/lumina_plasma.c:7531,7570` 및 upper `:7678` | lower/upper dilution 제거 |
| 17 | 기존 진단 | old `:17832 T_rad` | `src/lumina_plasma.c:18025-18027` | Boltzmann reference와 출력 온도는 유효한 `T_e`; fallback 없음 |
| 18 | 기존 진단 | old `:17833 W` | `src/lumina_plasma.c:18028` | Boltzmann diagnostic에서 `W` 제거 |

처분 결과는 `docs/A2_01_DISPOSITION_LEDGER.md` addendum에 구현 후 실제 줄번호와 함께 기록한다. 18개 원장 ID 집합의 누락·중복·미처분은 census checker 실패다.

## 6. 원장 밖 소비자 전수검사

18행만 고치고 끝내지 않는다. 다음 현행 소비 군을 정적 call graph와 grep/AST checker에 사전등록한다.

| 군 | 현행 위치 | 요구 처분 |
|---|---|---|
| partition 정본·fallback | `src/lumina_plasma.c:2181-2232` | 3절의 단일 `Z(T_e)` 정본, missing `T_e` 즉시 실패 |
| 중복 `partition_functions_Te` | `src/lumina.h:445`, `src/lumina_atomic.c:1962,2609`, `src/lumina_plasma.c:4510-4529,5002` | 별도 소유·계산 제거 또는 동일 저장소 alias |
| Sobolev용 LTE level 합성 | `src/lumina_plasma.c:2942-2976` | opacity 식은 유지하고 population 입력만 공개 population/`LTE@T_e` accessor로 전환 |
| macro/k-packet level fallback | `src/lumina_plasma.c:4477-4479,4998-5013` | `T_e` accessor; missing이면 fail, `T_rad/W` 분기 제거 |
| recombination destination population | `src/lumina_plasma.c:5126-5137,5659-5669` | 같은 `Z(T_e)`와 population generation 사용 |
| BF opacity/rate population | `src/lumina_plasma.c:7529-7580,7675-7681` | solved population 우선; 필요 reference는 `LTE@T_e`; opacity 식은 A2-08로 유보 |
| `bf_rate_pop` | `src/lumina_plasma.c:9280-9291,12074,13805` | `W/T_rad` 없는 checked accessor로 교체 |
| all-level Gamma weight | `src/lumina_plasma.c:10456-10483` | 이미 쓰는 `T_e`를 단일 partition view와 결박 |
| RADEQ untracked population | `src/lumina_plasma.c:12203-12233` | lower/upper 모두 동일 population generation; `W/T_rad` 제거 |
| coupled untracked population | `src/lumina_plasma.c:13856-13877` | RADEQ와 같은 accessor; 독립 fallback 금지 |
| isolated-row anchor | `src/lumina_plasma.c:16835-16883` | Boltzmann anchor 게시 금지; rank-incomplete 명시 실패 |
| force/singular solve fallback | `src/lumina_plasma.c:17208-17216,17361-17419` | `Boltzmann@T_rad` 게시 금지; 실패 전파 |
| within-superlevel 분배 | `src/lumina_plasma.c:17625-17669` | `T_e`/Z invalid 시 1 K·fraction 1 대체 금지, generation stamp 적용 |
| Boltzmann dump | `src/lumina_plasma.c:18018-18032` | `T_e`만 사용하고 진단 generation 기록 |
| ion rate 선택 복제 | `src/lumina_plasma.c:2415-2684,2721-2849` | checked BF helper 필수화; field-not-built Saha fallback 제거 |
| level matrix rate 조립 | `src/lumina_plasma.c:15145,15558-15573,16232-16244,16553-16564` | BF/BB invalid 상태를 solve 실패로 전파; numerical zero로 소비 금지 |

정적 checker의 합격 조건은 전역 문자열 0건이 아니라 **CPU 생산 population call graph 안의 금지 소비 0건**이다. A2-08/09/10/11/13에 배정된 opacity, emissivity, RADEQ energy, transport, GPU 및 명시적 output-only 진단은 경로·이유·후속 단계를 allowlist에 적는다. allowlist는 파일:행, 함수, 도달 root, 읽는 심볼, 비물리 이유를 가져야 하며 범위 패턴이나 파일 전체 허용은 금지한다.

checker는 최소한 다음 root에서 도달성을 확인한다.

- `compute_plasma_state`, `compute_ion_populations*`, `compute_electron_density`.
- `nlte_solve_all` 및 level matrix 조립/solve 경로.
- BF/BB rate가 population을 읽는 CPU 경로.
- Sobolev/opacity가 level population을 다시 합성하는 CPU 경로.

금지 패턴은 `plasma->T_rad`, `plasma->W`, `T_e_T_rad_ratio`, `bf_rate_estimator`, raw `jbar_line/j_blue`, legacy line-source selector, `Boltzmann@T_rad`, missing-to-zero/floor다. 별칭·구조체 복사·helper 인자도 추적한다.

## 7. validity, fallback, 카운터, 종료코드

### 7.1 상태

최소 상태 enum은 다음 원인을 구별한다.

```text
POP_OK
POP_EXACT_ZERO
POP_INVALID_TE
POP_INVALID_PARTITION
POP_STALE_DERIVED_TEMPERATURE
POP_BF_STALE / POP_BF_UNSAMPLED / POP_BF_OOG / POP_BF_MISS
POP_BB_STALE / POP_BB_UNSAMPLED / POP_BB_OOG / POP_BB_MISS
POP_PROFILE_MISMATCH / POP_QUERY_HASH_MISMATCH
POP_ATOMIC_MISSING
POP_RANK_INCOMPLETE
POP_NE_NOT_CONVERGED
POP_SOLVE_FAILED
POP_NONFINITE
POP_FORBIDDEN_FALLBACK
```

원 upstream enum을 보존할 수 있으면 새 이름으로 뭉개지 말고 함께 기록한다. 첫 오류와 누적 오류 수를 모두 남긴다.

### 7.2 필수 카운터

다음을 population context에 두고 atomic/thread-local reduction 후 `nlte_free`에서 정확히 한 줄의 `[A2-07][POP-VIEW]` 요약을 낸다.

```text
pop_generation_required, pop_generation_committed
pop_shells_attempted, pop_shells_published
pop_bf_terms, pop_bb_terms, pop_exact_zero_terms
pop_blocked_stale, pop_blocked_unsampled, pop_blocked_oog, pop_blocked_miss
pop_blocked_profile, pop_blocked_qhash, pop_blocked_te, pop_blocked_partition
pop_rank_incomplete, pop_ne_not_converged, pop_solve_failed, pop_nonfinite
pop_generation_mismatch, pop_fallback_attempts, pop_partial_publish_attempts
```

A2-05/06의 기존 카운터도 함께 보고하고 합계 불변식을 검사한다. 정상 PASS lane은 필수 입력에 대해 blocked·fallback·partial publish가 0이고, attempted shell과 published shell이 같으며 committed generation이 required generation과 같다. 실패 주입 lane은 해당 원인 카운터가 `>0`이고 게시가 0이어야 한다.

### 7.3 rc

| rc | 의미 |
|---:|---|
| 0 | 모든 필수 gate PASS, 또는 negative wrapper가 기대한 child FAIL을 확인 |
| 2 | 사용법·I/O·manifest·parser·hash 오류 |
| 3 | 판정 입력 또는 upstream 자격 부족으로 `BLOCKED_*` |
| 4 | 계산은 가능했으나 물리 metric 또는 내부 불변식 FAIL |
| 5 | forbidden fallback, stale generation, partial publish 등 계약 위반 |

`BLOCKED`를 PASS로 출력하지 않는다. 각 artifact는 `status`, `reason_code`, `child_rc`, `wrapper_rc`를 모두 가진다.

## 8. L-2 공통 오라클·lane 계약

### 8.1 불변 manifest

CHAIN과 ORACLE_INPUT은 동일한 덱, 원자모형, geometry, `T_e`, 밀도·조성, 초기 population/seed 정책을 사용한다. 차이는 즉시 upstream 복사율 입력뿐이다. manifest는 최소 다음 SHA-256과 metadata를 가진다.

- CMFGEN `POPCAL`, `POPCOB`, `POPIRON`, `POPNICK`, `POPSIL`, `POPSUL`, `RVTJ`, 필요한 `*OUT`와 superlevel membership 자료.
- A2-05 BF rate input/export, A2-06 line/Jbar input/export.
- geometry와 shell mapping, atomic model과 level crosswalk.
- Lumina 실행 파일, source tree, 환경변수, RNG seed/stream, `T_e` 배열.
- CMFGEN run token/iteration 및 파일별 generation 일치 증명.

서로 다른 CMFGEN run/generation 파일을 섞으면 rc 2다. 공개 진리와 `jnu4`의 역할이 다르면 source별로 명시하고 같은 물리 generation임을 증명하지 못한 조합은 물리 PASS에 쓰지 않는다.

### 8.2 lane

| lane | population solver 입력 | 목적 | 자격 처리 |
|---|---|---|---|
| CHAIN | 현재 실행에서 commit된 A2-05 BF view + A2-06 line view | 전체 배선·validity 전파 | upstream L-1 자격을 그대로 승계. 부족하면 `BLOCKED_UPSTREAM_*` |
| ORACLE_INPUT | 동일 인터페이스에 deterministic truth-side BF `Gamma`와 BB `Jbar`를 commit | population solver 자체 분리 | population은 주입하지 않음. rate export/coverage 없으면 `BLOCKED_MISSING_RATE_EXPORT` |

ORACLE_INPUT도 checked view와 동일 generation/profile/query-hash 검사를 통과해야 한다. 테스트 전용 raw 배열 직접 대입은 금지한다.

### 8.3 truth-side universe와 `f_cov`

crosswalk와 truth-active 집합은 Lumina의 성공/실패 상태를 읽기 전에 고정한다.

1. truth population 또는 truth flow 기여를 내림차순으로 누적하여 99.9%에 도달하는 최소 집합을 `truth_active`로 정한다. 경계 동률은 모두 포함한다.
2. `f_cov = sum(truth contribution of usable matched states) / sum(truth contribution of truth_active states)`다. 분모는 항상 truth다.
3. stale/unsampled/OOG/MISS/atomic-unmatched 상태는 분자에서 빠질 수 있지만 분모에서는 빠지지 않는다.
4. 상태 필터 후 남은 집합으로 분모를 다시 만들거나, Lumina population이 큰 것만 고르는 순환 필터를 금지한다.
5. coverage 미달은 metric을 좋아 보이게 하는 제외가 아니라 `BLOCKED_COVERAGE`다.

CHAIN의 Monte Carlo 불확실성은 population solver까지 전파한 독립 replicate로 평가한다. rate-bin CI만으로 nonlinear population CI를 대신하지 않는다. 각 판정 metric의 양측 95% CI half-width가 해당 허용폭의 1/3 이하일 때만 자격이 있다. ORACLE_INPUT이 완전 결정적이면 CI=0으로 기록한다.

수치상 자격선은 ion TV `<=0.0333`, `n_e` median `<=0.0333`, `n_e` P95 `<=0.0667`, level 합계오차 `<=0.0333`, level log P95 `<=0.10 dex`의 CI half-width다. coverage는 95% 단측 하한이 `>=0.95`여야 한다. dominant-stage는 각 replicate의 truth 정답집합 일치율과 stage별 선택 빈도를 함께 기록하고, 95% 구간이 판정을 뒤집을 수 있으면 `BLOCKED_MC_UNCERTAINTY`다.

## 9. L-2ion 게이트

### 9.1 입력·좌표·안전대

CMFGEN truth는 `POPCAL`, `POPCOB`, `POPIRON`, `POPNICK`, `POPSIL`, `POPSUL`, `RVTJ`에서 같은 generation으로 읽는다. POP level population을 원소·spectroscopic ion stage별로 합치고 원소 총밀도와 교차검산한다.

물리 판정 셸은 **정확히 s0-s8**이다. `s9`는 10,706 km/s 경계가 내부를 통과하므로 포함하지 않는다. `s10+`도 포함하지 않는다. `docs/OUTSIDE_LOOP_POOL.md:1205-1227`의 측량을 manifest에 복사하고, 실제 geometry 경계를 다시 hash/확인한다. 오염 셸을 hold, 외삽, 이웃 복사, 평균하여 안전대에 섞지 않는다.

RVTJ depth와 Lumina shell은 velocity cell boundary로 매칭한다. exact common cell이 아니면 양의 intensive quantity에 한해 사전등록된 log interpolation을 사용하고, 좌표 밖 외삽은 금지한다. 매칭 규칙은 Lumina 결과를 읽기 전에 고정한다.

### 9.2 동결 9이온 영향의 필수 실측

`jnu4`는 다음 9이온을 동결한 run이다.

```text
Si VI, S VI, Ca VI, Fe VI, Fe VII,
Ni VI, Ni VII, Co VI, Co VII
```

spectroscopic numeral은 charge로 명시 변환한다. 즉 VI는 `q=5`, VII는 `q=6`이며 원자모형의 내부 index를 그대로 stage로 간주하지 않는다.

각 이온과 s0-s8에 대해 다음을 `validation/a2_07/A2_07_FROZEN_ION_CONTRIB.csv`에 기록한다.

```text
element, ion, charge, shell, velocity,
n_ion_truth, n_element_truth, population_share,
bf_outflow, bb_incident_flow, total_radiative_flow, rate_flow_share,
population_dominant, rate_dominant, exclusion_reason,
source_file, source_generation, crosswalk_status
```

정의는 다음과 같다.

- `population_share = n_frozen_ion / sum_stages(n_stage)`.
- `bf_outflow = sum_l n_l Gamma_l`.
- `bb_incident_flow = sum_transitions [n_l R_lu + n_u(R_ul^stim + A_ul)]`; 한 transition을 한 번만 센다.
- `rate_flow_share`의 분모는 같은 원소·shell의 모든 stage에 대한 `bf_outflow+bb_incident_flow`다.
- frozen ion이 원소에서 population argmax이거나 `population_share>=0.5`, 또는 `rate_flow_share>=0.5`이면 그 shell에서 지배다.

rate 기여는 POP만으로 추정하지 않는다. 같은 generation의 truth rate export/crosswalk가 없으면 `BLOCKED_FROZEN_RATE_CONTRIBUTION`이다. 9개 중 하나라도 누락되거나 charge 표기가 틀리면 rc 2다.

한 원소가 s0-s8 중 한 셸에서라도 동결 stage 지배이면 그 원소 전체를 L-2ion **물리 오차 집계에서 제외**한다. 제외는 truth-side freeze 정보만으로 정하며 Lumina 오차를 본 뒤 바꾸지 않는다. 제외 원소도 표·coverage·closure에는 남기고 `EXCLUDED_FROZEN_DOMINANT`로 표시한다. 모든 비교 원소가 제외되면 PASS가 아니라 `BLOCKED_NO_ELIGIBLE_ELEMENT`다.

### 9.3 metric과 합격선

각 eligible 원소·shell에서 `f_q=n_q/n_element`를 계산한다.

- ion fraction TV: `0.5*sum_q |f_q^L-f_q^C| <= 0.10`.
- dominant stage: `argmax_q f_q^L == argmax_q f_q^C`. truth 동률 허용폭은 입력 정밀도로 사전등록하고 모든 동률 stage를 정답 집합으로 둔다.
- `n_e` symmetric error: `2|n_e^L-n_e^C|/(|n_e^L|+|n_e^C|)`의 s0-s8 중앙값 `<=0.10`, P95 `<=0.20`.
- closure: 각 원소·shell에서 `|sum_q n_q-n_element|/n_element <=1e-10`.

TV와 closure는 모든 eligible 원소·s0-s8에서 통과해야 한다. aggregate 하나로 나쁜 원소를 숨기지 않는다. `n_e`는 원소 제외와 무관하게 s0-s8 전체에서 판정한다. coverage는 truth 기준 `>=0.95`를 추가 자격으로 적용한다.

## 10. L-2level 게이트

### 10.1 물리 비교와 partition 검사의 분리

두 검사는 별도 artifact·status를 갖고 둘 다 통과해야 L-2level PASS다.

1. **NLTE physical population**: Lumina solved population과 CMFGEN POP population 비교.
2. **internal partition**: Lumina `Z(T_e)`와 독립 CPU 수치 oracle 비교.

CMFGEN NLTE population을 Boltzmann 분배의 정답으로 쓰지 않는다. internal partition 검사는 CMFGEN population을 전혀 읽지 않는다.

### 10.2 level/superlevel crosswalk

crosswalk key는 다음 모두를 포함한다.

```text
Z, spectroscopic ion, normalized label/configuration,
excitation energy, g, parent/core id, superlevel membership id
```

- energy 허용차는 `1e-6 eV`다. label·`g`·parent·membership가 모두 같은 후보 안에서만 energy 최근접을 쓴다.
- 후보가 둘 이상 같은 허용차로 남으면 `AMBIGUOUS`, 없으면 `UNMATCHED`다. 어느 경우도 0 population으로 치환하지 않는다.
- index 또는 파일 순서만으로 연결하지 않는다.
- authoritative membership는 같은 generation의 `*OUT`/level-superlevel 연결 자료와 Lumina atomic membership에서 읽고 hash한다.
- cardinality 1인 identity level만 개별 level끼리 비교한다.
- 비자명 superlevel은 Lumina의 모든 member population을 먼저 합한 뒤 대응 CMFGEN superlevel 총합과 비교한다. CMFGEN superlevel 총합을 Lumina 개별 level 하나와 직접 비교하지 않는다.
- membership가 없거나 한 member라도 중복 귀속되면 그 unit은 unmatched이고 coverage 분모에는 남는다.

### 10.3 물리 metric

단위는 위 규칙으로 만든 level 또는 superlevel 비교 unit이다. 각 ion·s0-s8에서 truth ion population을 분모로 한다.

- matched population coverage `sum_matched n_u^C / n_ion^C >=0.95`.
- 합계오차 `sum_matched |n_u^L-n_u^C| / n_ion^C <=0.10`.
- 양쪽이 양수인 matched unit의 `|log10[(n_u^L/n_ion^L)/(n_u^C/n_ion^C)]|` P95 `<=0.30 dex`.
- truth positive인데 Lumina가 0/non-finite이면 log 표본에서 버리지 않고 hard FAIL로 별도 집계한다. 양쪽 exact zero만 log 집계 밖의 valid zero다.

동결 지배 원소 제외 규칙은 9.2절과 동일하게 적용하며, 제외 unit을 coverage 분모에서 몰래 지우지 않고 `EXCLUDED_FROZEN_DOMINANT` subtotal로 별도 보고한다. 판정 가능한 eligible ion이 없으면 BLOCKED다.

### 10.4 내부 `Z(T_e)` CPU gate

독립 oracle은 long double 지수, energy shift, compensated summation으로 원자모형에서 직접 계산한다. 생산 double 결과와 모든 loaded ion 및 다음 온도 집합에서 비교한다.

- 실제 s0-s8 `T_e` 전부.
- 지원 온도 범위의 하단·중앙·상단 고정 fixture.
- 큰 energy span, degeneracy, underflow 경계를 포함한 synthetic fixture.

상대오차는 `|Z_prod-Z_ref|/|Z_ref| <=1e-10`이며 모든 case가 통과해야 한다. `Z_ref=0`, invalid `T_e`, invalid level 자료는 수치 비교 대상이 아니라 명시적 입력 실패다. GPU `5e-5` 기준은 후속 단계이며 A2-07 PASS에 사용하지 않는다.

## 11. 음성 대조 4종

각 poison은 별도 child process에서 한 가지만 바꾸고, 실제 발화 marker와 기대 metric FAIL을 확인한다. static `getenv()` cache 때문에 같은 process에서 poison이 안 바뀌는 일을 금지한다.

| ID | poison | 고정 marker | 기대되는 거부 | child/wrapper |
|---|---|---|---|---|
| N1 | 사전선택한 eligible 원소의 인접 stage `q,q+1` 연결/label 교환 | `A2_07_NEG_STAGE_SWAP` | TV `>0.10` 또는 dominant-stage mismatch | child rc 4, wrapper rc 0 |
| N2 | s0-s8의 `n_e(s)`를 사전등록한 인접 depth 값으로 차용 | `A2_07_NEG_NEIGHBOR_NE` | `n_e` median `>0.10` 또는 P95 `>0.20` | child rc 4, wrapper rc 0 |
| N3 | partition·LTE reference에 `T_e` 대신 같은 shell의 옛 `T_rad` 대입 | `A2_07_NEG_TRAD_FOR_TE` | `Z` 상대오차 `>1e-10` 및 적어도 하나의 level metric FAIL | child rc 4, wrapper rc 0 |
| N4 | crosswalk를 고정한 뒤 matched unit의 population 값만 비항등 순열 | `A2_07_NEG_LEVEL_SHUFFLE` | 합계오차 `>0.10` 또는 log P95 `>0.30 dex` | child rc 4, wrapper rc 0 |

각 poison은 baseline 결과를 읽기 전에 witness를 정한다. 자연 데이터의 인접 차가 약해 문턱을 넘지 못하면 poison 강도를 임의로 키워 PASS하지 않고 `BLOCKED_WEAK_NEGATIVE_CONTROL`로 끝낸다. N2는 여전히 인접 depth 차용이어야 하며 원거리/circular shuffle로 바꾸지 않는다. N3는 실제 셸이 둔감할 때 `T_e != T_rad`인 고정 synthetic partition fixture가 반드시 실패해야 하지만, 그 사실만으로 physical level negative를 대신하지 않는다.

marker 미발화, 예상과 다른 rc, baseline FAIL, poisoned PASS는 모두 전체 gate FAIL이다.

## 12. classic debt sweep 17항목

이 sweep은 **발견·계측·영향 기록**이며 새 물리 수리 허가가 아니다. 모든 항목은 `validation/a2_07/A2_07_CLASSIC_SWEEP.json`과 구현 보고서에 `fired`, `hit_count`, `affected_shells/species`, `impact_metric`, `impact_value`, `population_path_live`, `disposition`, `evidence`를 남긴다. 측정 불능은 `UNMEASURED`가 아니라 구체적인 `BLOCKED_*`다.

영향은 가능하면 기존 toggle의 paired run으로 측정한다. toggle이 없는 hard-code는 상태를 바꾸지 않는 shadow 계측으로 term/population/잔차 기여를 기록한다. 계측을 위해 물리값을 고치는 새 fallback을 넣지 않는다.

| ID | 저작 시 현행 위치 | 발화·영향 실측 계약 | A2-07 처분 및 census 제안 |
|---|---|---|---|
| H01 | `src/lumina_main.c:93`; `src/lumina_atomic.c:823-829`; `src/lumina_plasma.c:3115` | `.9*T_rad` seed/default가 적용된 shell 수, generation, 그 값이 partition/population에 도달한 횟수; 공급 `T_e` 대 명시적 입력 lane의 ion TV/level log 차 | population의 fallback 소비는 제거. `T_e` 해법은 A2-10/A2-17 잔존. `PARTIAL_MEASURED` 제안 |
| H04 | `src/lumina_plasma.c:2716-2720,2746,2841-2846` | shell별 iteration, cap hit, 최종 `abs(ne_calc-ne)/ne`, damping 전후 변화와 L-2ion `n_e` 오차 | 반복법 수리 금지; 미수렴 게시 금지. `MEASURED_OPEN` |
| H05 | `src/lumina_plasma.c:17701-17710,17778-17940`; GPU `src/lumina_cuda.cu:1027-1043` | CE 외부 반복 횟수·cap hit·최종 max residual, CE 전후 ion TV/level sum; CPU/GPU 발화 구분 | CPU 계측, 식/한도 불변; GPU A2-13. `MEASURED_OPEN` |
| H11 | `src/lumina.h:542-543,635,657,673-674` | deck 요구/할당/사용/잘림 수와 headroom: ion, pair, CE, DR 각각; 잘림 객체의 truth population/rate flow 비율 | 동적 topology 수리 금지; 잘림이 active면 BLOCKED. `MEASURED_OPEN` |
| H15 | `src/lumina_plasma.c:18054-18056,18079-18082` | 0.025/0.05/35 eV 상수 발화 shell, nonthermal ion rate/전체 ionization rate 비율과 on/off shadow population 영향 | gamma/Spencer-Fano 수리 금지, A2-10 인계. `MEASURED_OPEN` |
| S03 | `src/lumina_plasma.c:2174-2232,2942-2976` | `T_rad/W` partition·level 합성 hit와 영향 population/partition 차; Sobolev 소비까지 lineage | partition/population은 A2-07에서 이관, tau 식은 A2-08. `PARTIAL_MEASURED` |
| S04 | `src/lumina_plasma.c:2515-2684,2721-2849` | Saha/fallback 대 canonical-rate branch shell 수, 각 rate source의 ion TV·`n_e` 영향 | 공급원만 canonical rate로 교체; ion 방정식 재설계 금지. `PARTIAL_MEASURED` |
| S09 | `src/lumina_plasma.c:3203-3217,3333-3342,3440-3620,6183-6241,7190-7250,7399-7445,7669` | ground-core route와 data-driven target route 횟수, 버려진/없는 excited-core 채널 수, 관련 recombination flow 및 population 비율 | core topology 수리 금지, A2-09/A2-13 인계. active missing channel은 coverage 차감/BLOCKED. `MEASURED_OPEN` |
| S14 | `src/lumina_atomic.c:1293-1313` | ion별 full level 수, cutoff 위 member 수, superlevel population 및 rate-flow 비율 | membership를 L-2 crosswalk에 사용; cutoff 수리 금지. `MEASURED_OPEN` |
| S15 | `src/lumina_plasma.c:16435-16522,16835-16883` | top reservoir/isolated anchor hit, 해당 ion population·rate-flow 비율, rank 상태 | old anchor 게시 금지로 결함 노출; topology 수리 금지. `PARTIAL_MEASURED` 또는 `BLOCKED_RANK_INCOMPLETE` |
| S16 | `src/lumina_plasma.c:16806-16815` | backward-Euler term이 적용된 ion/shell, steady/time-dependent term 비율과 기존 on/off paired population 차 | 시간격자 수리 금지. `MEASURED_OPEN` |
| G01 | `src/lumina_atomic.c:1859`; `src/lumina_plasma.c:1803-1809` | neutral-ladder default 및 `1e10 eV` sentinel 횟수, affected ion의 truth population/rate-flow 비율 | 원자자료 합성 수리 금지; active missing이면 BLOCKED. `MEASURED_OPEN` |
| G02 | `src/lumina_plasma.c:2137-2154` | `zeta=1` default ion/shell 수, 관련 recombination/ion population 비율 | zeta 수리 금지; canonical-rate path에서 live 여부 명시. `MEASURED_OPEN` |
| G03 | CPU `src/lumina_plasma.c:17208-17216,17361-17419`; GPU `src/lumina_cuda.cu:1365-1480` | 강제/singular/nonfinite 횟수, fallback이 덮을 population 질량, solver residual | CPU fallback population 게시 금지; solver 자체 수리는 후속. GPU 잔존 명시. `PARTIAL_MEASURED` |
| G05 | `src/lumina_atomic.c:2500-2570` | synthetic O IV 생성 횟수, O population/`Gamma`/rate-flow 기여 | 실제 원자모형 수리 금지; gate active이면 별도 subtotal 또는 BLOCKED. `MEASURED_OPEN` |
| G06 | CPU `src/lumina_plasma.c:17435-17463`; GPU `src/lumina_cuda.cu:1852-1862`; wrappers `scripts/slurm_prod_dr7ion_ce17_ionlock.sh:46`, `scripts/slurm_skipsi_physical_champion.sh:115`, `scripts/slurm_ddc15_FI_prod.sh:180` | skip mask 발화, Si population은 풀렸는지, tau 미갱신 line 수와 population/opacity 기여 | A2-07 solver는 Si도 동일 owner. tau skip 수리는 A2-08, GPU A2-13. `PARTIAL_MEASURED` |
| G10 | `src/lumina_main.c:172-175`; GPU `src/lumina_cuda.cu:7118-7121`; wrappers `scripts/slurm_prod_dr7ion_ce17_ionlock.sh:44`, `scripts/slurm_skipsi_physical_champion.sh:114`, `scripts/slurm_ddc15_FI_prod.sh:179` | start=0 대 start=5 동일 입력 paired run의 terminal ion TV, level log P95, `n_e` Esym 및 iteration history | seed 정책 수리 금지. 차이가 metric 허용선을 넘으면 물리 gate BLOCKED/FAIL. `MEASURED_OPEN` |

`docs/CLASSIC_DEBT_CENSUS.md` 제안 diff의 처분 칸에는 위 상태와 evidence artifact 경로를 적는다. `UNMEASURED`를 단순히 `MEASURED`로 바꾸려면 수치와 단위가 있어야 한다. `fired=false`도 실행 조건과 branch counter로 증명한다.

## 13. 구현 순서

순서를 바꾸어 하류 결과로 상류 결함을 숨기지 않는다.

1. 저작 HEAD를 기록하고 18행 및 6절 소비 군을 static manifest로 동결한다.
2. population status/result와 transactional work buffer, generation stamp, 카운터를 먼저 구현한다.
3. partition을 단일 `Z(T_e)` 정본으로 합치고 모든 Boltzmann accessor의 `T_rad/W` 인자를 제거한다.
4. ion solver의 복사율 공급을 checked BF view로 필수화한다. 식·topology·수렴 상수는 바꾸지 않는다.
5. level matrix와 모든 population 부수 소비를 checked BF/BB view와 공개 population accessor로 이관한다.
6. invalid/fallback/singular/rank-incomplete를 최상단 rc까지 전파하고 partial publish를 차단한다.
7. static census checker와 unit/negative fixture를 통과시킨다.
8. classic 17항목을 계측한다. 수리하지 않고 disposition diff를 만든다.
9. ORACLE_INPUT 후 CHAIN 순으로 L-2ion/L-2level을 실행한다. upstream blocked는 그대로 기록한다.
10. 전 회귀판을 재실행하고 정확히 한 §11 대장 행과 구현 보고서를 작성한다.

구현 중 새 population 소비자를 발견하면 멈추지 말고 manifest와 원장 addendum의 `DISCOVERED_OUTSIDE_CENSUS`에 추가한다. 발견 수 0도 grep/AST 명령과 결과를 기록해야 한다.

## 14. selftest와 회귀 전판

### 14.1 A2-07 필수 selftest

- analytic 2/3-level `Z(T_e)`와 long-double oracle, 상대오차 `<=1e-10`.
- missing/nonfinite/nonpositive `T_e`가 `POP_INVALID_TE`, 게시 0.
- `T_e` hash 또는 generation 1 bit 변경이 stale failure, 게시 0.
- BF `EXACT_ZERO`는 valid zero, BF/BB `UNSAMPLED/OOG/MISS/STALE`는 solve 차단.
- BF와 BB generation 중 하나만 갱신하면 mismatch failure.
- 중간 shell 실패 시 전체 새 generation partial publish 0.
- singular/isolated row가 Boltzmann fallback을 게시하지 않음.
- superlevel aggregate와 membership ambiguity/unmatched 처리.
- 4종 negative marker·rc.
- `nlte_free` counter 합계와 정상 fallback 0.

### 14.2 회귀 목록

현재 저장소에 존재하는 선행 gate를 실제 target/script 이름으로 실행하고 구현 보고서에 명령·rc·artifact를 기록한다.

1. `make lumina`.
2. `python3 scripts/a2_01_census_contract.py`.
3. A2-02/A2-02C replay와 기존 negative controls.
4. A2-03 radiation-field selftests.
5. A2-04 commit/replay selftests 및 `scripts/a2_04_l0_replay.py`.
6. `python3 scripts/run_gate_battery.py`의 전 36 case.
7. A2-05 BF selftest와 `scripts/a2_05_l1bf_gate.py`; 기존 CHAIN eligibility/status를 정직하게 보존.
8. A2-06 line/dual-commit selftests와 `scripts/a2_06_l1bb_gate.py`; 현재 blocked 원인을 PASS로 완화하지 않음.
9. A2-07 static census, population selftest, L-2ion/L-2level 두 lane과 4 negative.

script의 실제 CLI는 구현 시 `--help`와 소스를 열어 확정한다. 존재하지 않는 옵션을 명세 예시대로 추측해 실행하지 않는다.

## 15. 운전석 명령 — 2노드

구현자는 아래 두 driver script를 저장소에 만들고, 운전석은 동일 source tree에서 실행한다. driver 안에는 `set -euo pipefail`, source hash 확인, 환경변수 dump, rc 보존, artifact hash를 넣는다.

### grammar-debug: 빌드·정적·결정론 selftest

```bash
ssh grammar "ssh grammar-debug 'bash /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_a2_07_grammar_debug.sh'"
```

이 driver는 `make lumina`, census/static checker, A2-03~A2-07 CPU selftest, ORACLE_INPUT deterministic fixture를 수행한다. 로그인 노드에서 테스트 binary를 실행하지 않는다.

### lageunha: 전 배터리·CHAIN/ORACLE_INPUT 물리 gate

먼저 상태를 확인한 뒤 실행한다.

```bash
ssh lageunha uptime
ssh lageunha 'bash /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_a2_07_lageunha.sh'
```

이 driver는 전 36-case 배터리, A2-05 L-1bf, A2-06 wiring/L-1bb 상태 확인, A2-07 두 lane, 4 negative, classic sweep을 수행한다. CPU thread 수는 driver가 명시하며 oversubscription을 금지한다. `/gpfs`의 덱·오라클은 읽기 전용이고 결과는 `validation/a2_07/`에 쓴다. 어느 driver에서도 `/usr/bin/time`을 사용하지 않는다.

운전석은 stdout의 PASS 문자열만 서명하지 않는다. 각 JSON의 source/input hash, rc, status, coverage, metric, counter, negative marker를 확인한다.

## 16. §11 단계 회귀 대장

`validation/a2_07/A2_07_REGRESSION_LEDGER.jsonl`에는 A2-07에 대해 정확히 한 JSON object를 남긴다. 필드는 `docs/ORDER_L0_JNU_OWNER_BY_CODEX.md:689-710`을 그대로 포함한다.

```text
stage_id=A2-07
contract=SPEC_A2_07_V1
source_tree_hash
input_manifest_hash
oracle_id
node
command
exit_status
new_layer_status={L2ION:{CHAIN,ORACLE_INPUT},L2LEVEL:{CHAIN,ORACLE_INPUT,PARTITION_CPU}}
all_previous_layer_statuses
negative_control_status={stage_swap,neighbor_ne,trad_for_te,level_shuffle}
coverage={truth_f_cov,ion,level,frozen_disclosure}
metric_values={ion_tv,dominant_stage,ne_esym,closure,level_sum,level_log_p95,Z_relerr}
changed_output_allowlist
guard_hits
fallback_hits
rng_seed
mc_confidence
artifact_paths
driver_signoff
```

`new_layer_status`는 `PASS`, `FAIL_*`, `BLOCKED_*`를 lane별로 보존한다. ORACLE_INPUT PASS와 CHAIN BLOCKED를 하나의 A2-07 PASS로 합치지 않는다. 최종 stage 상태는 가장 약한 필수 상태다. 전 단계 status 중 하나가 악화되면 원인 조사 전 서명하지 않는다.

## 17. 구현 보고서 필수 목차

`docs/CODEX_IMPL_A2_07.md`는 최소 다음을 포함한다.

1. source tree/commit, input manifest, 실행 노드와 exact command.
2. 18행 1:1 이관 대응표와 `docs/A2_01_DISPOSITION_LEDGER.md` diff.
3. 목록 밖 소비자 발견표, 구현 후 production call-graph 금지 읽기 0 증명, 잔류 allowlist.
4. 온도·rate view·generation·transactional publish 구현과 파일:행.
5. validity/fallback/counter 표와 `[A2-07][POP-VIEW]` 실제 출력.
6. classic sweep 17항목의 발화·영향 수치표와 `CLASSIC_DEBT_CENSUS_A2_07_PROPOSED.diff`.
7. L-2ion CHAIN/ORACLE_INPUT 판정표, s0-s8 shell manifest, truth-side `f_cov`, CI.
8. 동결 9이온 population·BF·BB·total rate 기여 실측표와 지배 원소 제외 전/후 subtotal.
9. L-2level physical crosswalk/coverage/metric 표와 독립 `Z(T_e)` CPU 표.
10. 음성 대조 4종의 poison, witness, marker, 기대/실제 metric, child/wrapper rc.
11. 회귀 전판과 정확히 한 §11 대장 행.
12. 미해결 BLOCKED/부채와 A2-08 인계.

모든 표는 machine-readable artifact 경로와 SHA-256을 병기한다. 파일명·줄번호는 구현 완료 직전 다시 `rg -n`으로 측정한다.

## 18. A2-08 및 후속 인계

A2-07은 generation-bound CPU population accessor를 남긴다. A2-08은 이를 읽어 opacity/Sobolev tau를 계산하며 `T_rad/W` Boltzmann population을 다시 합성할 수 없다.

A2-08 인계 항목은 다음과 같다.

- `src/lumina_plasma.c:2942-2976`의 tau/opacity 계산식 및 population 이외 scalar 소유권.
- A2-06 addendum이 A2-08로 재배치한 line-source/blanketed-heating 4행.
- G06의 Si tau skip과 모든 원소 동일 opacity 소유권.
- A2-07 population generation과 opacity generation의 exact binding 및 stale failure.

A2-09에는 recombination core/target와 emissivity, A2-10에는 `T_e` 해법·H01/H15, A2-13에는 GPU·dynamic topology·CE/solver 부채, A2-18에는 완전수렴/seed 독립성을 넘긴다. 인계는 A2-07 PASS로 오인하지 않도록 각 미해결 debt/status를 그대로 쓴다.

## 19. 합격 판정 체크리스트

A2-07은 다음이 모두 참일 때만 구현 폐합 후보가 된다.

- [ ] 원장 18행이 각각 구현 후 위치와 종결 처분을 가진다.
- [ ] CPU 생산 population call graph에서 partition/LTE/Boltzmann의 `T_rad/W` 읽기가 0이다.
- [ ] `T_e` 부재·invalid가 fallback이 아닌 명시적 실패이며 partial publish가 0이다.
- [ ] 단일 `Z(T_e)` 정본과 generation/hash 결박이 있다.
- [ ] ion solver의 필수 BF rate와 level solver의 필수 BF/BB rate가 checked A2-05/06 view만 소비한다.
- [ ] invalid rate가 numerical zero가 아니며 `EXACT_ZERO`만 valid zero다.
- [ ] `bf_rate_estimator`, dilute Planck, raw Jbar가 CPU 생산 population을 공급하지 않는다.
- [ ] L-2ion과 L-2level을 CHAIN/ORACLE_INPUT으로 각각 기록하고 upstream BLOCKED를 보존한다.
- [ ] 모든 POP 물리 판정은 s0-s8만 사용한다.
- [ ] 동결 9이온의 population·rate 기여가 모두 수치로 기록되고 지배 원소가 물리 집계에서 제외된다.
- [ ] `f_cov` 분모는 truth이고 상태 필터로 재정의되지 않는다.
- [ ] superlevel membership aggregation 뒤 비교하며 개별 level 직접 비교가 없다.
- [ ] CMFGEN NLTE population과 internal Boltzmann partition 검사를 분리한다.
- [ ] CPU `Z(T_e)` 상대오차가 전 case `<=1e-10`이다.
- [ ] 4 negative가 각각 고유 marker와 기대 metric FAIL, child rc 4, wrapper rc 0을 보인다.
- [ ] classic 17항목 모두 발화·영향 수치 또는 구체적 BLOCKED를 갖고 census에는 제안 diff만 있다.
- [ ] `nlte_free` 카운터와 fallback 0/게시 불변식이 맞는다.
- [ ] 전 36-case 배터리, L-0, L-1bf, L-1bb wiring/status, census checker, `make lumina` 회귀를 통과 또는 정직한 upstream BLOCKED로 기록한다.
- [ ] §11 대장 행이 정확히 하나이고 fable 운전석 서명이 있다.

## 20. 저작 시 실측한 사실

이 절은 명세 작성 시점의 측량이며 구현 후 줄번호로 대체하는 표가 아니다.

1. 저작 HEAD는 `ece5aef8e192e2166b647ee00aae5fdd1f935a1c`이며 A2-06 폐합 커밋이다. 직전 A2-05는 `d8b987032377f4e64677df1215ffaaed781cb982`다.
2. `docs/A2_01_DISPOSITION_LEDGER.md`는 243줄이고 본문에 `행 수: 157`, `미분류: 0`을 선언한다. A2-07 이관 대상은 A2-05 재배치 6행 + A2-06 재배치 6행 + 기존 partition/rate/diagnostic 6행 = **18행**이다.
3. `rg -n 'bf_rate_pop\(' src/lumina_plasma.c` 결과는 **3곳**이다: 정의 `:9280`, 호출 `:12074`, `:13805`.
4. `rg -n 'partition_functions_Te' src/lumina_plasma.c` 결과는 **2곳**이다: 작성 `:4529`, 소비 `:5002`. 헤더 선언은 `src/lumina.h:445`, 할당/해제는 `src/lumina_atomic.c:1962,2609`다.
5. `rg -n 'partition_functions\[' src/lumina_plasma.c`의 텍스트 hit는 **19곳**이다. 따라서 원장 18행만으로 partition 소비 전수가 닫히지 않는다.
6. `rg -n 'Boltzmann@T_rad' src/lumina_plasma.c`의 텍스트 hit는 **6곳**이다: `:8564,:8780,:16837,:16997,:17208,:17376`. 주석이라도 실제 fallback 경로의 역인덱스로 등록했다.
7. 현재 partition fallback은 `src/lumina_plasma.c:2204-2210`, level weight는 `:2220-2232`; `T_e`가 없으면 `T_rad,W` baseline을 유지한다고 코드가 명시한다.
8. 현재 BF population helper는 `src/lumina_plasma.c:9280-9291`에서 `T_rad` 지수와 metastable 외 `W` weight를 사용한다.
9. 현재 diagnostic은 `src/lumina_plasma.c:18025-18028`에서 `T_e` 부재 시 `T_e_T_rad_ratio*T_rad`를 사용하고 `T_rad,W`를 함께 읽는다.
10. current fixed `n_e` 반복은 `src/lumina_plasma.c:2746,2841-2846`에서 최대 100회, damping 0.5, 5% 종료 조건이다.
11. classic debt 17개 원행은 `docs/CLASSIC_DEBT_CENSUS.md:38,41-42,48,52,64-65,70,75-77,85-87,89-90,94`에 존재하며 모두 저작 시 `UNMEASURED`다.
12. `jnu4`의 동결 9이온은 `docs/OUTSIDE_LOOP_POOL.md:1018-1029`, 공개 진리 대비 `n_e=0.0481`인 14,000 km/s 표본은 `:1091-1119`, s0-s8 안전대와 s9 경계는 `:1205-1227`에서 확인했다.
13. `docs/CODEX_IMPL_A2_05.md`는 존재하지 않았다. 존재하지 않는 파일을 근거로 만들지 않았고 `validation/a2_05/A2_05_CLOSURE.md`와 `docs/CODEX_REVIEW_A2_05.md`를 확인했다.
14. 저작 시작 시 작업트리에는 사용자의 기존 변경·미추적 산출물이 다수 있었다. 이 명세 작성에서는 그 파일들과 `src/`를 수정하지 않는다.

이 측량을 구현 종료 때 같은 명령으로 다시 실행하고, hit가 이동·증가·감소한 이유를 `docs/CODEX_IMPL_A2_07.md`에 기록해야 한다.
