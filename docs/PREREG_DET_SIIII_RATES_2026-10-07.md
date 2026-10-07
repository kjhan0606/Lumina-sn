# 사전등록 — DET-SIIII-RATES: Si III 상위 준위의 율 예산과 E′−R′ 밀도 차이 귀속

2026-10-07 · 기안자 astra · read-only  
운전자 파일화 대상: `docs/PREREG_DET_SIIII_RATES_2026-10-07.md` — **신설 예정**  
상태: **사전감사 전 기안. 구현·기계 게이트·실행·결과 판정 미집행.**

## §0. 목표와 판단 범위

### Q-GOAL — 최종목적에 직접 기여하는가?

선행 DET-SIIII-POP은 Si III 1892.03 Å의 상위 준위 2와 2542.58 Å의 상위 준위 5에서 E′·R′의 세대 7 복원 절대밀도가 크게 다름을 확인했다. 그러나 그 밀도를 공급·배출하는 율 경로는 아직 확인하지 않았다.

이 단의 목표는 **두 상위 준위의 세대 7 상태에서 생산 정의에 따른 과정별 유입·유출 예산을 구성하고, E′−R′ 밀도 차이 중 자료가 지지하는 귀속 범위를 판정하는 것**이다. 이를 위해 다음을 한 묶음으로 수행한다.

1. 소비 캡처에서 필요한 세대 7 준위 인구를 확장 복원한다.
2. 생산 율 방정식에 실제 존재하는 복사·충돌·이온화·재결합·비열적 경로를 확인하고, 계산 가능한 계수와 체적당 율을 산출한다.
3. 오프라인 산술 폐합, 세대 7 상태의 평형 잔차, 생산 해 재현을 구분하여 검사한다.
4. 밀도 차이를 율 계수 변화·공급 준위 인구 변화·비평형 잔차로 구분하고, 결손 때문에 식별되지 않는 부분을 명시한다.

산출은 NRB 반복법, 다른 율 경로, 또는 기록 결손 중 무엇을 후속 조사해야 하는지 결정하는 데 소비한다. 이 단을 생략하면 큰 준위밀도 비를 특정 과정의 인과적 기여율로 오인할 수 있다.

**기안자 자기답: PASS.** 사전감사자의 판단은 별도로 기록한다.

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

대상은 두 준위와 셸 17–30이다. 다른 준위·이온은 이들의 유입·유출 또는 생산 해 폐합에 필요한 연결만 읽는다.

- 생산 코드 수정·재빌드, GPU 생산 런, 새 수송 계산, 반복 연장, CMFGEN 계산은 하지 않는다.
- 결손 J̄를 얻기 위한 새 복사장 replay는 하지 않는다.
- 전 원소 율 지도, 반복별 새 dump, 광도 분해 재실행은 하지 않는다.
- 새 과학 검사는 자료 연결과 율·폐합·분기를 검사하는 두 게이트로 묶는다.
- 자료가 부족한 전체 행렬을 먼저 구현하지 않는다. 입력 연결표에서 재구성 가능한 범위를 정한 뒤 해당 범위만 구현한다.

**기안자 자기답: APPROPRIATE.**

### 승인 근거

`/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/evidence/USER_DECISION_2026-10-07.txt`

사용자 2026-10-07 「1」: 두 Si III 상위 준위의 율 경로를 비교하고 E′·R′의 NRB 설정 차이와 연결한다.

### 사전 확인한 중요한 제한

**세대 7 인구와 세대 7 복사장의 조합을 이미 수렴한 통계평형해로 가정하지 않는다.**

T2의 `cmfgen_run`은 최종 R6 복사장 발행과 관측 캡처를 먼저 수행하고, 그 뒤 R7 물질 갱신을 수행한다. 따라서 캡처의 인구 세대 7은 그때 발행된 복사장 세대 7에 대한 후속 물질 해와 구별된다. 최종 저장 인구는 세대 8이다. FIXED_TE도 전자밀도·이온 인구·준위 인구의 동일성을 보장하지 않는다.

또한 모든 소비 쌍의 τ·S가 존재한다는 사실은 다음을 보장하지 않는다.

- 모든 준위가 역산 가능함.
- 모든 율 전이의 J̄가 저장됨.
- 모든 생산 방정식의 경계 인구·정규화·전자 상태가 같은 시점으로 저장됨.

따라서 **완전한 율 예산과 생산 해 폐합의 달성을 미리 보증하지 않는다.** 미달하면 원래 목표의 미달로 기록하며, 부분 율 표 작성으로 목표를 바꾸지 않는다.

### 주장 상한

허용하는 주장은 다음이다.

- 지정된 세대·상태에서 생산 코드가 사용하는 과정별 율.
- 같은 상태에서 NRB 적용 형태를 바꿨을 때의 대수적 계수 차이.
- 완전한 예산에 대한 밀도 차이의 조건부 대수적 귀속.
- 기록된 입력과 출력 사이의 생산 방정식 재현 여부.

다음은 판정하지 않는다.

- 어느 처리가 물리적으로 옳은지.
- 여섯 반복 동안 축적된 밀도 차이의 독립적인 인과 기여율.
- NRB의 수렴성·생산 승격·수리 승인.
- 공개 CMFGEN 준위 인구나 율과의 일치.
- 스펙트럼 잔차 감소 또는 특정 과정의 인과적 배제.

NRB는 선 자기 소스를 새 인구에 연결하는 **반복법 변경**이라는 선행 정의를 승계한다. 고정된 전체 J̄를 사용하는 행렬의 단순 항등 재작성으로 취급하지 않는다.

## §1. 기준선·입력·보존

### 1.1 경로

```text
P = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
C = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
T2 = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source
E = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1
R = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1
B = /gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007
D = /gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007

새 도구·러너 = D/tools
새 산출 = D/results/<실행 식별자>
집행 증거 = D/evidence/<집행 식별자>
```

### 1.2 승계하는 기록

- DET-SIIII-POP 폐합 `aeca08c`, 계약·판정:
  - `docs/PREREG_DET_SIIII_POP_2026-10-07.md`
  - `docs/VERDICT_DET_SIIII_POP_2026-10-07.md`
- 선행 정본 산출: B의 `results/a2/`.
- 선행 종료감사: B의 `astra_closeout.md`, `astra_closeout2.md`.
- 표적 선정: `docs/VERDICT_DET_OBS_LINEDECOMP_2026-10-07.md`.
- NRB 정의와 응답:
  - `docs/PREREG_DET_NRB_E_2026-10-02.md`
  - `docs/VERDICT_DET_NRB_E_2026-10-02.md`.

선행 인구 역산과 캡처·행 대응 검증을 승계하되, 이번에 확장하는 준위·율 연결은 새로 확인한다. 과거 감사의 작성자·판정은 보존한다.

### 1.3 입력과 권위

아래 런 상대 경로는 E·R 각각에 적용한다.

| 입력 | 용도·권위 |
|---|---|
| `work/det_obs_capture/lines.bin` | 소비 쌍의 τ·S_used·mapped, 세대 7 인구 역산 |
| 같은 디렉터리의 `manifest.json`, `meta.txt` | 세대·배열 형식·소비 권위 |
| 같은 디렉터리의 `Te.f64`, `ne.f64`, `J.f64`, `nu.f64`, `dnu.f64` | 세대 7 물질·coarse 복사장; 해당 생산 소비 경로가 확인될 때만 율에 사용 |
| 같은 디렉터리의 연속체 계수·기하 배열 | 필요한 생산 식의 입력 연결 확인; 새 수송 계산에는 사용하지 않음 |
| `work/w5_lines.csv` | 저장된 선·셸의 J̄·J_c·β·τ_eff·오차·operator·세대 |
| `input/model/line_list.csv`, `levels.csv` | 전이·준위·Einstein 계수·통계중량·에너지·입력 묶음 |
| 같은 model 디렉터리의 원자·충돌·광이온화 자료 | 실제 활성 율 분기에 필요한 자료만 사용 |
| `input/deck.sha256`, `input/resolved_lumina.exports`, `input/run_build_identity.json` | 봉인·설정·생산 판본 |
| `work/nlte_levels_final.csv`, `work/cmfgen_final_publication.json`, `work/lumina_plasma_state.csv` | 세대 8 보조 비교 또는 적격한 후속 해 재현 |
| `work/lumina_ion_channels.csv` | 이온 사다리의 채널 계수; 준위별 율로 대체하지 않음 |
| `work/physics_comparison/physics_DET_iter*.ion.csv` | 시점이 확인된 이온 경계·정규화 보조 자료 |

동일한 숫자의 `gen`, `call`, `iter`를 같은 시점으로 간주하지 않는다. 각 writer의 호출 위치와 입력·출력 의미를 연결한다.

### 1.4 신원과 보존

- 생산 식의 권위는 현재 작업 트리가 아니라 **T2 생산 판본**이다.
- 소비한 소스·소형 입력·선택 추출물·도구·산출의 경로와 해시를 manifest에 기록한다.
- 대형 캡처의 기존 봉인 신원은 승계하고, 새 판독에서 선택한 레코드의 신원을 기록한다. 캡처 전체 복제·문자열 변환은 하지 않는다.
- E·R의 설정 차이는 경로 이식을 제외하고 NRB와 진단 캡처 키라는 운전자 확인을 입력 근거로 연결한다. 다른 물리 설정 차이가 확인되면 해당 비교를 중단하고 원인을 기록한다.
- 수정 재판독은 새 실행 식별자를 사용한다. 기존 결과·원본·실패 로그를 덮어쓰지 않는다.
- D는 사용자 결정문 때문에 이미 존재할 수 있다. 새 결과 하위 디렉터리는 기존 경로와 충돌하면 거부한다.

## §2. 표적과 생산 방정식의 범위

### 2.1 표적

| 전이 | 선 목록 0-based 데이터 행 | 이온 | 하위→상위 준위 | 선행 단의 주요 셸 | 본 단의 율 예산 주요 셸 |
|---|---:|---|---|---|---|
| Si III 1892.03 Å | 908565 | Z=14, ion=2 | 0→2 | 22–30 | 24·30 |
| Si III 2542.58 Å | 1097363 | Z=14, ion=2 | 4→5 | 17–22 | 19 |

개정 1은 본 단의 율·밀도 결과 판독 전에 자료 가용성만을 근거로 율 예산 주요 셸을 다시 등록한다. 근거는 `/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/evidence/w5_coverage_2026-10-07.txt`다. E′·R′ 양쪽의 `w5_lines.csv`에서 rad 7·material 7 기록은 셸 10·15·19·24·30에만 있고, 각 셸에는 Si III 준위 2에 연결된 선 31개와 준위 5에 연결된 선 17개의 J̄가 모두 있다. 율 예산 주요 셸은 선행 단의 주요 셸과 이 공통 기록 셸의 교집합으로 정한다. 율·밀도 값이나 원하는 귀속 분기를 기준으로 선택하지 않았다.

두 표적 모두 셸 17–30의 자료 가용성과 결손을 기록하며, 본문에서 별도 수식어 없이 쓰는 “주요 셸”은 위 표의 “본 단의 율 예산 주요 셸”을 뜻한다. 선행 단의 주요 셸과 그 인구 결과는 변경하지 않는다. 선행 주요 셸 중 1892.03 Å의 22·23·25–29와 2542.58 Å의 17·18·20–22는 J̄ 기록 결손으로 전체 율 예산·밀도 차이 귀속을 `NOT_IDENTIFIABLE`로 남긴다. 이 셸들을 coverage와 결손 보고에서 삭제하지 않는다.

셸 17–30 중 J̄가 없는 나머지 셸도 같은 결손 처리를 적용한다. J̄가 있으나 해당 표적의 율 예산 주요 셸이 아닌 셸은 보조 결과로만 다루며 주 결론의 대체 셸로 쓰지 않는다. 셸 10·15는 기존 판독 범위 밖이므로 추가 분석하지 않는다. J̄의 완전한 기록만으로 상대 준위 인구·J_c·생산 분기·경계 시점까지 완전하다고 간주하지 않으며, 주요 셸에도 기존 적격성·폐합·귀속 기준을 그대로 적용한다. 이번 사전 개정 이후 결과를 보고 주요 셸을 다시 고르거나 인접 셸로 보간하지 않는다.

선 ID는 CSV 헤더를 제외한 데이터 행이다. 별도 `line_id` 열로 결합하지 않는다.

표적 방출선만으로 율 예산을 만들지 않는다. **각 상위 준위에 들어오거나 나가는 모든 생산 경로**를 목록화한다. 표적 준위가 다른 전이의 하위 준위인 경우도 포함한다.

### 2.2 실제 solve 좌표

T2의 다음 경로를 따른다.

| 절대 소스 경로 | 함수·기호 | 역할 |
|---|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c` | `load_atomic_data` | 원자자료 로드와 `LUMINA_SUPER_CUTOFF` 적용 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c` | `nlte_build_projection`, `nlte_precompute_within_sl_frac_checked` | full-level과 solve 좌표 대응 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/population_contract.c` | `population_within_superlevel_build` | super-level 내부 Boltzmann 분율 |

확인한 E′ exports에는 `LUMINA_SUPER_LEVELS="1"`, `LUMINA_SUPER_CUTOFF="100"`이 있다. 로더는 CSV의 `super_level`을 그대로 사용하는 대신 준위 번호 100 미만을 개별 준위로 두고, 그 이상을 같은 이온의 super-level 100으로 묶는다.

따라서 이 설정이 양쪽 런에 동일하게 적용되었음을 확인한 뒤:

- 준위 2·5는 개별 solve 준위로 취급한다.
- 고준위 공급·배출항에는 실제 투영 분율을 적용한다.
- CSV에 적힌 저준위 묶음을 그대로 재사용하지 않는다.
- 고준위 전체를 서로 독립적인 평형 방정식으로 풀지 않는다.

생산 행렬에서 같은 super-level 내부 전이가 제외되는 조건도 그대로 따른다.

### 2.3 직접 항과 간접 경로

T2의 기본 Si NLTE 쌍은 Si II·III이며, `nlte_assemble_rate_matrix_impl`의 기본 bound-free 경로는 하위 이온 준위와 상위 이온 바닥 준위를 연결한다. DR과 기본 비열적 이온화도 지정된 바닥 준위 경로를 사용한다.

양쪽 exports와 기본값을 확인하여 해당 구성이 맞으면, **Si III 여기 준위 2·5에 직접 걸리는 광이온화·재결합·비열적 항은 구조적으로 없을 수 있다.** 이때는 다음을 구별한다.

- `STRUCTURAL_ZERO`: 활성 생산 방정식에 그 직접 경로가 없음.
- `EXACT_ZERO`: 경로는 있으나 입력·생산 정의에 의해 율이 정확히 0.
- `MISSING`: 경로는 있으나 계산에 필요한 자료가 없음.

직접 항의 구조적 0은 이온화·재결합·비열적 과정의 **간접 영향이 없다는 뜻이 아니다.** 이들은 Si III 바닥 인구, 다른 준위, 전자밀도, 쌍 정규화에 영향을 줄 수 있다.

생산 소스에 있는 Ca–Si 전하교환도 실제 연결 여부를 확인한다. 활성 경로를 예산에서 누락하지 않되, 표적과 무관한 반응 전수조사는 하지 않는다.

## §3. 세대 7 인구 복원과 자료 결손

### 3.1 복원식

선행 계약의 생산식과 수치 오차 처리를 승계한다.

\[
a=\frac{g_l}{g_u},\qquad
A_\nu=\frac{2h\nu^3}{c^2},\qquad
K=C_{\rm Sob}f_{lu}\lambda t_{\rm exp},
\]

\[
d=\frac{\tau}{K},\qquad
n_l=d\left(1+\frac{S}{A_\nu}\right),\qquad
n_u=\frac{g_u}{g_l}d\frac{S}{A_\nu}.
\]

권위는 T2의 `nlte_update_tau_sobolev_with_authority`, `a208_signed_sobolev_counted`, `a208_line_source_counted`, `det_obs_capture_pair`다.

복원 대상은 표적 율의 상대 준위와, 조건부 생산 해 재현에 필요한 Si 준위로 제한한다. 동일 캡처에서 Si III 소비·매핑 선을 한 번 선택 판독하여 재사용한다.

### 3.2 적격성

복원마다 다음을 확인한다.

- 전이·준위·셸 대응과 같은 세대의 τ·S 권위.
- 생산 source fallback·변형의 적용 여부.
- 역산 분모와 유한성.
- 생산 상수·연산 순서·저장 정밀도를 반영한 구간.

한 준위가 여러 선에서 복원되면 공통 인구와 양립하는지 확인한다. 양립하지 않는 값을 평균하여 숨기지 않는다. 부적격 선을 제외하려면 사유가 생산 권위나 역산 조건으로 정해져야 하며, 원하는 율 결론을 기준으로 고르지 않는다.

실제 super-level 내부 분율과 그 세대의 온도 권위가 확인되면, 복원된 구성 준위에서 같은 super-level의 인구를 유도할 수 있다. 이를 `G7_PROJECTION_DERIVED`로 구분한다. 서로 다른 super-level 사이를 LTE 비로 채우지 않는다.

### 3.3 결손 인구

역산 또는 확인된 생산 투영으로 얻지 못한 준위는 `MISSING_G7_POPULATION`이다.

- 0, 바닥 인구의 임의 비율, 세대 8 인구로 채우지 않는다.
- 해당 인구가 필요한 유입항은 미정의로 둔다.
- 율 계수가 알려졌더라도 체적당 율은 결손일 수 있다.
- 결손 상대 준위 때문에 총유입이 미정의이면 부분합을 총유입으로 부르지 않는다.

### 3.4 세대 8 사용 조건

세대 8은 다음 용도만 허용한다.

1. 같은 준위의 세대 차이 보고.
2. §5의 시점 연결이 입증된 후속 생산 해 재현.
3. 세대 7 결손을 대신하지 않는 별도 보조 계산. 이 경우 사용 인구·복사장·전자 상태 세대를 모두 표기하고 `G8_AUXILIARY` 또는 `MIXED_GENERATION_AUXILIARY`로 표시한다.

복원 가능한 일부 준위에서 세대 차이가 작다는 이유로 결손 준위의 차이도 작다고 가정하지 않는다. 세대 8 대입 결과는 세대 7 주 판정과 지배 과정 분기에 사용하지 않는다.

## §4. 생산 정의에 따른 율 예산

### 4.1 기본 정의

상태 \(X\in\{E',R'\}\), 표적 준위 \(u\), 과정 \(p\)에 대해 전이 계수 \(P^X_{ij,p}\)의 단위는 \({\rm s^{-1}}\), 준위밀도의 단위는 \({\rm cm^{-3}}\)다.

\[
I^X_{u,p}
 =\sum_{j\ne u} n^X_jP^X_{ju,p}+s^X_{u,p},
\qquad
D^X_{u,p}
 =\sum_{j\ne u}P^X_{uj,p},
\]

\[
O^X_{u,p}=n^X_uD^X_{u,p},
\qquad
F^X_{u,p}=I^X_{u,p}-O^X_{u,p}.
\]

外部 인구를固定 입력으로 쓰는 생산 source는 \(s_{u,p}\)에 해당한다. 구현과 문서에서는 이를 **외부 인구를 고정 입력으로 쓰는 source**라고 표기한다.

\[
I_u=\sum_p I_{u,p},\quad
D_u=\sum_p D_{u,p},\quad
O_u=n_uD_u,\quad
\mathcal R_u=I_u-O_u.
\]

체적당 율 \(I,O,F,\mathcal R\)의 단위는 \({\rm cm^{-3}\,s^{-1}}\)이다. 양의 \(F\)는 표적 인구를 늘리는 방향이다.

단순히 \({\rm s^{-1}}\) 계수의 크기를 비교하여 유입 지배를 판정하지 않는다.

### 4.2 Bound-bound 복사

권위:

- T2 `lumina_plasma.c`: `nlte_assemble_rate_matrix_impl`, `nlte_bb_rate_graph_contains`, `nlte_bb_jbar_canonical`, `bb_nrb_arm`.
- T2 `line_net_rate.c`: `line_net_bracket_rates`, `line_net_cmfgen_exponx`, `line_net_sobolev_radiation`.

생산의 `A2_06_DIAGNOSTIC_SHADOW` 계산은 뒤에서 덮어쓰므로 율 권위로 사용하지 않는다.

전이 \(l\to u\)에서 NRB 비적용 형태는

\[
P^\mathrm{off}_{lu}=B_{lu}\bar J,\qquad
P^\mathrm{off}_{ul}=A_{ul}+B_{ul}\bar J.
\]

NRB 적용 가능한 선은 생산자가 발행한 \(ct=\beta J_c\)와 \(\tau_{\rm eff}\)를 사용하여

\[
P^\mathrm{on}_{lu}=B_{lu}ct,\qquad
P^\mathrm{on}_{ul}=\beta A_{ul}+B_{ul}ct.
\]

β는 `line_net_cmfgen_exponx`의 작은 τ 다항식·큰 τ 분기를 그대로 따른다. 새로운 escape-probability 근사식으로 바꾸지 않는다.

INIT, domain 밖, checked view 상태, skip-Z, `SRCE_CHK`, provenance 조건도 생산 분기대로 처리한다. 특히 생산 domain 밖 전이는 펌프의 제외와 자발방출의 존재를 각각 확인한다. 선 전체를 임의로 0으로 만들지 않는다.

J̄·J_c 사용 우선순위:

1. 같은 상태·선·셸·operator·세대의 실제 저장값.
2. 필요한 생산 입력이 모두 저장된 경우의 대수적 재구성.
3. 그 밖은 `MISSING_LINE_FIELD`.

`w5_lines.csv`의 call 7은 rad 7·material 7과 함께 확인한다. 같은 키의 모순되는 중복행은 거부한다. 기록이 없는 선에 coarse J를 보간하여 J̄ 또는 J_c로 대입하지 않는다. τ·S만으로 외부 펌프를 결정했다고 주장하지 않는다.

### 4.3 충돌 여기·탈여기

권위는 T2 `nlte_assemble_rate_matrix_impl`의 활성 충돌 분기다. `LUMINA_NLTE_COLL_FIX`, `LUMINA_OMEGA_CMFGEN`, parity, 실제 collision table 적용 여부를 확인한다.

`NLTE_COLL_FIX` 분기가 실제로 활성일 때:

\[
C_{lu}
 =n_e\frac{8.629\times10^{-6}}{g_l\sqrt{T_e}}
   \Upsilon\,e^{-\Delta E/kT_e},
\qquad
C_{ul}
 =n_e\frac{8.629\times10^{-6}}{g_u\sqrt{T_e}}\Upsilon,
\]

\[
\Upsilon=
\max\!\left[
14.5 f_{lu}
\left(\frac{13.6057}{\Delta E_{\rm eV}}\right)^2\,0.2,\,
{\tt AXELROD\_OMEGA}
\right].
\]

상수와 활성화 조건은 T2 값 그대로 사용한다. 충돌 자료 파일이 존재한다는 이유만으로 생산이 사용하지 않은 표를 새로 적용하지 않는다.

전용 충돌표·metastable 경로가 활성이라면 생산의 중복 억제와 보간 규칙까지 따른다. 동일 준위쌍에 여러 생산 항이 있으면 생산 조립 방식대로 합산하고, 임의의 중복 제거를 하지 않는다.

### 4.4 광이온화·재결합·충돌 이온화·비열적

| 과정 | 생산 권위 | 처리 |
|---|---|---|
| 준위 광이온화 | `nlte_assemble_rate_matrix_impl`, `nlte_bf_gamma_canonical` | 실제 threshold·단면·canonical field 적분 |
| 복사 재결합 | 같은 assembler의 `I_rec`, `n_star_ratio`, spin 분기 | 자발·유도 Milne 항과 실제 상위 이온 연결 |
| 충돌 이온화·삼체 재결합 | 같은 assembler의 `nlte_bf_collisional_enabled` 분기 | 활성 여부와 실제 source·destination 보존 |
| DR | 같은 assembler의 `dr_lookup`, `dr_alpha_eval` 사용부 | 생산의 바닥 준위 routing 보존 |
| 비열적 이온화 | 같은 assembler의 `gamma_dep->nonthermal_ioniz_rate` 사용부 | 생산의 원자당 계수와 기본 ground-to-ground routing 보존 |
| 전하교환 | 같은 assembler의 `CE_REACTIONS`, `nlte_get_ion_density` 사용부 | 필요한 상대 이온 인구의 권위·시점 확인 |

생산의 복사 재결합은 일반적으로 단순한 `photoionization × LTE ratio`가 아니다. 실제 Milne 적분의 자발항을 보존한다. coarse J를 쓰는 적분과 canonical view를 쓰는 적분도 임의로 통일하지 않는다.

`lumina_ion_channels.csv`는 `parity_rate_se_ratio_checked`의 이온 사다리 계수다. 그 `gamma_phot`, `gamma_nt`, `ne_alpha_rr`, `ne_alpha_dr` 등을 준위 2·5에 인구 비례로 배분하지 않는다. 이온 단위 기록은 간접 경로의 보조 증거로만 연결한다.

직접 경로가 구조적으로 없으면 그 사실을 결과표에 기록한다. 간접 경로의 인과적 기여를 계산할 자료가 없으면 `INDIRECT_PATH_NOT_IDENTIFIED`로 남긴다.

## §5. 폐합의 정의

### 5.1 오프라인 산술 폐합

다음은 도구의 필수 검사다.

- 과정별 유입·유출 합과 해당 행렬 행의 계산이 같은 값이어야 한다.
- 과정 분해·계수 분해·§6 밀도 차이 분해의 합이 등록 항등식과 맞아야 한다.
- 보존행을 물리 유입·유출항으로 세지 않는다.
- super-level 투영에서 빠지거나 두 번 더해진 전이가 없어야 한다.

이 검사는 **오프라인 도구의 내부 일관성**이며 생산 해 재현이나 통계평형 달성의 증거가 아니다.

### 5.2 세대 7 상태의 평형 잔차

완전한 표적 행이 확보된 경우

\[
\epsilon_u=
\frac{|\mathcal R_u|}{I_u+O_u}
\]

를 보고한다. 분모가 정확히 0인 고립 상태는 별도 표기한다.

세대 7에서 \(\mathcal R_u\ne0\)인 결과를 자동으로 판독기 오류로 처리하지 않는다. R6 캡처와 후속 물질 갱신의 시차에서 생긴 **고정 상태의 비평형 잔차**일 수 있다.

반대로 τ·S 역산의 순·역 대입 성공이나 일부 전이의 상쇄만으로 \(\mathcal R_u=0\)이라고 판정하지 않는다.

### 5.3 생산 방정식과 같은 해의 재현

생산 해 재현은 다음 조건을 모두 만족할 때만 시도한다.

1. 비교 대상 생산 solve의 입력 복사장, \(T_e,n_e\), 경계 인구, 정규화와 투영이 식별된다.
2. 필요한 모든 활성 계수가 저장값 또는 정확한 생산식으로 재구성된다.
3. 비교할 저장 인구가 그 solve의 출력임이 호출 순서·세대 기록으로 확인된다.
4. 오프라인 해를 비교 대상의 표적 인구로 맞추거나 역으로 보정하지 않는다.

최소 범위는 표적에 필요한 Si solve 계다. 생산이 사용하는 이온쌍·정규화·경계조건을 유지한다. 이를 위해 전 원소 전자밀도 반복을 새로 구현하지 않는다. 필요한 경계 상태가 없으면 `PRODUCTION_SOLUTION_NOT_TESTABLE`이다.

세대 7 view를 고정하여 새로 푼 해는 `FROZEN_G7_SOLUTION`이다. 이것은 세대 7 저장 인구와 같아야 하는 해가 아니다.

세대 8 출력과 비교하려면, 최종 R7이 소비한 rad 7 외에 **그 solve에서 사용한 전자밀도·경계 인구·정규화**까지 일치해야 한다. `Te`가 같거나 `population_generation=8`이라는 사실만으로 적격 판정을 하지 않는다.

### 5.4 수치 수용 기준

새로운 임의의 상대오차 백분율을 두지 않는다.

- binary64 입력에는 생산 연산과 재구성의 반올림 오차를 전파한다.
- 텍스트 값에는 실제 writer 형식의 반 단위와 명시된 오차 상한을 반영한다.
- W5의 `Jbar_err_upper`, `J_c_err_upper`는 함께 전파한다.
- 상쇄가 있는 순율에는 절대오차를 사용한다.
- 결손은 수치 오차 상한에 숨기지 않는다.

생산 solver 기준은 T2 `population_contract.h`의 다음 상수를 사용한다.

```text
POP_GENERATOR_RESIDUAL_LIMIT = 1.0e-12
POP_DENSE_BACKWARD_ERROR_LIMIT = 1.0e-12
```

각 기준은 `population_generator_stationary_gth`와 `population_dense_solve_equilibrated`의 **해당 잔차 정의**에만 적용한다. 이를 곧바로 인구의 상대 전방오차 허용치로 사용하지 않는다.

생산 해 재현은 다음을 함께 요구한다.

- 완전한 입력·경계·출력 시점 연결.
- 해당 생산 잔차 기준 통과.
- 독립적으로 저장된 인구와의 차이가 입력 저장오차·계수 오차·검증된 해 오차 범위 안임.
- 정규화와 표적 외의 solve 성분에서도 모순이 없음.

단지 넓은 구간이 겹친다는 이유로 같은 해라고 단정하지 않는다. 해의 민감도 때문에 차이를 가릴 수 없으면 `SOLUTION_AGREEMENT_UNRESOLVED`다. 완전한 동일 입력에서 설명되지 않는 차이가 남으면 `SOLUTION_MISMATCH`다.

## §6. 밀도 차이 귀속과 과학 분기

### 6.1 부호와 정확한 대칭 분해

모든 차이는

\[
\Delta X=X^{E'}-X^{R'},\qquad
\bar X=\frac{X^{E'}+X^{R'}}2
\]

로 정의한다.

완전한 예산과 양의 \(\bar D_u\)가 있으면

\[
\Delta n_u
=
\sum_p A_{u,p}+A_{u,\rm residual},
\]

\[
A_{u,p}
=
\frac{\Delta I_{u,p}-\bar n_u\Delta D_{u,p}}
     {\bar D_u},
\qquad
A_{u,\rm residual}
=
-\frac{\Delta\mathcal R_u}{\bar D_u}.
\]

각 항의 단위는 \({\rm cm^{-3}}\)다. 이는

\[
\Delta(n_uD_u)=\bar D_u\Delta n_u+\bar n_u\Delta D_u
\]

에서 나오는 정확한 항등식이다.

잔차 항을 생략하여 평형을 강제하지 않는다. 이 분해의 폐합 자체를 생산 해 폐합으로 부르지 않는다.

### 6.2 율 계수 변화와 공급 인구 변화

유입 차이는

\[
\Delta(n_jP_{ju,p})
=
\bar n_j\Delta P_{ju,p}
+\bar P_{ju,p}\Delta n_j
\]

로 나눈다.

이에 따라 \(A_{u,p}\)를 다음으로 분리한다.

- **율 계수 변화 항:** \(\bar n_j\Delta P_{ju,p}\)와 표적의 유출 계수 변화.
- **공급 준위 인구 변화 항:** \(\bar P_{ju,p}\Delta n_j\).
- 필요한 경우 생산의 외부 source 변화 항.

공급 인구 변화가 큰데 이를 복사 또는 충돌 **계수 자체의 변화**로 부르지 않는다. 이 구분은 이미 다른 곳에서 생긴 인구 차이를 특정 국소 과정의 원인으로 재명명하는 것을 막는다.

### 6.3 NRB 형태 변화와 복사장 변화

같은 상태에서 NRB on·off 양쪽 형태가 적격하게 계산되는 선에 대해

\[
\Delta P_{\rm form}
=
\frac{
[P_{\rm off}(E)-P_{\rm on}(E)]
+[P_{\rm off}(R)-P_{\rm on}(R)]
}{2},
\]

\[
\Delta P_{\rm field}
=
\frac{
[P_{\rm off}(E)-P_{\rm off}(R)]
+[P_{\rm on}(E)-P_{\rm on}(R)]
}{2}
\]

를 사용한다. 그러면 실제 적용이 E=off, R=on인 선에서

\[
P_{\rm off}(E)-P_{\rm on}(R)
=
\Delta P_{\rm form}+\Delta P_{\rm field}.
\]

여기서 `field` 항은 J̄·J_c·β 등 상태에 따른 저장 입력 변화다. 새 수송이나 상태 재수렴을 하지 않는 대수적 분해다.

양쪽에서 NRB 비적용인 선은 실제 생산 분기대로 처리한다. 필요한 J_c·provenance가 없으면 `NRB_FORM_NOT_IDENTIFIABLE`이다.

같은 인구에서 자기 포획항이 상쇄되어 on·off 순율 차이가 작을 수 있다. 이는 반복법이 다음 인구를 갱신하는 효과가 작다는 증거가 아니다. 생산 Einstein 계수·Sobolev 상수·source 정의의 일치를 확인하지 않고 이상적인 상쇄를 강제하지 않는다.

### 6.4 지배 판정

전체 밀도 차이 귀속의 비교 항은 다음처럼 고정한다.

- NRB 형태에 의한 복사 계수 항.
- 나머지 복사 입력 변화 항.
- 충돌 계수 변화 항.
- 실제 직접 이온화·재결합·비열적·전하교환 계수 또는 source 항.
- 공급 준위 인구 변화 항.
- 비평형 잔차 항.

어떤 항 \(A_k\)가

\[
|A_k|_{\rm lo}>
\sum_{j\ne k}|A_j|_{\rm hi}
\]

를 만족하고 방향도 수치적으로 분리될 때만 `DOMINANT`라 한다. 결손 항의 상한이 없으면 이 판정을 하지 않는다. 순차이가 작을 때 임의로 기여율을 만들지 않는다.

| 코드 | 분기 | 의미 |
|---:|---|---|
| 0 | `NRB_FORM_DOMINANT` | 등록한 대수적 밀도 차이 분해에서 NRB 형태 항이 지배 |
| 1 | `OTHER_RATE_DOMINANT` | 다른 율 계수 또는 source 항이 지배; 과정명을 함께 기록 |
| 2 | `PARTNER_POPULATION_DOMINANT` | 공급 준위 인구 차이가 지배; 그 인구 차이의 원인은 미식별 |
| 3 | `RESIDUAL_DOMINANT` | 세대 7 비평형 잔차 차이가 지배; 평형 원인 귀속 불가 |
| 4 | `MIXED` | 완전한 유효 예산에서 단일 지배항이 없음을 구별할 수 있음 |
| 5 | `UNRESOLVED` | 완전한 유효 예산이나 수치 구간 때문에 방향·지배 경계가 미분리 |
| 6 | `NOT_IDENTIFIABLE` | 필요한 인구·율·시점·권위가 결손 |

서로 반대 방향의 분리된 항이 있으면 `CANCELLING`을 추가한다.

부분 율만 있으면 해당 선·과정의 차이를 보고할 수 있지만 전체 지배 분기는 6이다. 표적 방출선 하나의 복사 순율 차이를 준위 전체의 지배 과정으로 승격하지 않는다.

각 주요 셸을 개별 판정한다. 모든 주요 셸에서 같은 분기가 성립할 때만 그 셸 집합 전체에 같은 결론을 쓴다. 그렇지 않으면 셸별 결과와 결손을 보고한다. 사후 다수결·셸 가중치·새 평균 지표를 도입하지 않는다.

## §7. 변경집합과 필수 산출

### 변경집합 — DET-SIIII-RATES (표)

아래 도구·러너만 신설한다. 신규 분석 함수는 `det_siiii_rates_` 접두를 사용한다.

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `scripts/det_siiii_rates_readout.py` | `det_siiii_rates_*` | 입력 연결·선택 판독·율 재구성·폐합·귀속·직접시험·산출 |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/tools/run_siiii_rates.slurm` | 오프라인 실행 | §9의 실행 조건 충족 후 환경·자원·명령·rc 기록 |

### 변경집합 끝

생산 소스·바이너리·기존 판독기·원자자료·캡처는 변경하지 않는다. 기존 역산 코드를 import하면 그 의존 파일 신원을 기록하고 원본을 수정하지 않는다.

| 필수 산출 | 내용 |
|---|---|
| `coverage.csv` | 필요한 전이·준위·시점, 실제 자료 출처, 복원·결손·구조적 0, 총예산 가능 여부 |
| `populations.csv` | 직접 복원·투영 유도 인구와 구간, 세대 8 보조 비교의 구분 |
| `rates.csv` | 전이별 과정·계수·인구·유입·유출·순율·구간·생산 분기·자료 권위 |
| `budget.csv` | 표적·셸·상태별 과정 합, 잔차, 완전성, 밀도 차이 분해와 분기 |
| `closure.json` | 산술 폐합·세대 7 잔차·생산 해 재현을 구분한 결과와 사유 |
| `checks.json` | 실제 분석 경로를 거친 정상·음성대조 결과 |
| `result.json` | 주요 셸별 결론, 목표 달성 범위, 결손과 주장 상한 |
| `manifest.json` | 계약·입력·도구·환경·명령·작업·산출 신원 |
| 판정문·종료감사 | 목표 달성·미달·비용·분장·이탈·후속 결정 |

미정의 수치는 `null` 또는 빈 수치칸과 명시적인 사유로 기록한다. `STRUCTURAL_ZERO`, 계산된 0, 결손을 같은 값으로 표현하지 않는다.

## §8. 최소 게이트·직접시험·음성대조

| 게이트 | 막는 실제 실패 | 통과 조건 |
|---|---|---|
| G-LINK | 선·준위·셸·세대 오결합, 잘못된 super-level 대응, 이온 원장의 준위율 대체, 결손의 0 처리 | §1–§4의 입력 권위·생산 분기·coverage가 실제 분석에 적용됨 |
| G-RATE | 잘못된 NRB·충돌 계수, 유입·유출 부호·중복 오류, 거짓 평형 폐합, 잔차 생략, 과도한 지배 판정 | §4–§6의 독립 작은 정답·항등식·결손·분기가 실제 분석 출력과 일치 |

생산 레이아웃을 갖춘 작은 가상 런을 만들고 **실제 `analyze` 경로**를 호출한다. 시험 전용 검사식에 참·거짓을 직접 넣는 방식만으로 통과시키지 않는다.

필수 사례는 다음으로 묶는다.

1. **정상 정답**
   - 독립적으로 지정한 인구와 전이 계수에서 유입·유출·잔차의 정답을 만든다.
   - NRB off·on과 생산 충돌식, 고준위 투영을 검사한다.
   - 알려진 평형해와 비평형 상태를 모두 포함한다.
   - E=R에서 차이 0, E/R 교환에서 부호 반전을 확인한다.

2. **연결·결손 음성대조**
   - 선 행 또는 준위 연결 오류, rad/material 세대 오류를 주입한다.
   - 필요한 J̄ 한 항과 상대 준위 인구를 제거한다.
   - 분석기가 이를 오류 또는 `NOT_IDENTIFIABLE`로 처리하고 총예산·지배 PASS를 내지 않는지 확인한다.
   - 세대 8 대입과 이온 단위 계수의 준위율 대체를 거부한다.

3. **율·폐합 음성대조**
   - 실제 분석에 쓰이는 시험용 계수 경로에서 β 또는 유도방출 항을 누락시켜 독립 정답 불일치를 검출한다.
   - 과정 누락·부호 반전·투영 중복 중 해당 구현에 실재 가능한 결함을 주입한다.
   - 비평형 상태를 평형이라고 선언하거나 밀도 차이 분해에서 잔차 항을 빼면 검출되어야 한다.

4. **판정 경계**
   - 지배 부등식의 등호·구간 겹침을 `DOMINANT`로 분류하지 않는다.
   - 결손 상한이 없는 부분합을 지배로 분류하지 않는다.
   - 구조적 0과 자료 결손의 출력이 구별되는지 확인한다.

각 결함은 주입 위치·기대 사유·실제 사유·분석 rc를 남긴다. 필수 사례가 누락된 시험 요약은 거부한다. 시험 개수를 늘리기 위한 별도 배터리는 만들지 않는다.

기존 기계 프리플라이트는 다음 도구를 사용한다.

- `scripts/check_prereg_preflight.py`
- `scripts/check_gate_referents.py`
- `scripts/check_doc_canonical.py`
- `scripts/check_dispatch_preflight.py`

각 실행의 stdout·stderr·rc와 검사 문서의 정확한 해시를 보존한다. PASS 요약만 남기지 않는다.

## §9. 실행 환경·비용·재현

### 9.1 환경과 집행 조건

요청된 자원 범위는 **a10 Slurm의 CPU 전용 오프라인 작업, GPU 예약·생산 실행 없음**이다. 로그인 노드에서는 파일 조회와 소스 판독만 한다. 직접시험·기계 게이트·수치 분석은 계산 노드에서 수행한다.

**실행 환경 확인:** syntax의 `/etc/slurm/job_submit.lua`에서 GPU 요청 검사 대상은 a40·h100·a100·a100_pcie·h200이며, a10은 제외되어 GPU를 요청하지 않는 CPU 작업을 받는다. 필터 주석에도 a10의 CPU 전용 작업 사용 근거가 명시되어 있다. 사용자 2026-10-06 지시 「selftest 는 여기에서 slurm 으로 작업을 제출」에 따라 이 단의 실행 경로는 a10 Slurm CPU 전용 작업으로 등록한다.

CPU 전용이라는 이유로 러너 제출을 `ENVIRONMENT_BLOCKED`로 두지 않는다. 운전자는 §9.2의 CPU·메모리·시간 상한을 명시하여 집행하고 GPU 예약량은 0으로 유지한다. 로그인 노드에서는 계산하지 않는다.

### 9.2 비용 상한

다음은 성능 예상값이 아니라 이 단의 등록 지출 상한이다.

| 항목 | 상한 |
|---|---:|
| 동시 작업 | 1개 |
| 작업당 CPU | 4개 |
| 작업당 메모리 | 16 GiB |
| 시험·기계 게이트·분석·수정 재판독 누적 할당 경과 | 60분 |
| 누적 CPU 예약량 | 4 core·h |
| GPU 예약·생산 실행 | 0 GPU·h · 0회 |
| 새 도구·시험·산출·로그 합계 | 256 MiB |

실패 작업도 비용에 포함한다. 자동 재제출은 하지 않는다. 특정된 판독기 결함은 같은 범위·상한 안에서 수정할 수 있다.

대형 캡처는 선택 판독하고 결과를 재사용한다. 반복 스캔·전체 복제·전체 CSV 변환을 하지 않는다. 입력 결손을 해소하려는 새 수송·생산 실행은 이 상한 안에서도 승인되지 않는다.

### 9.3 계획 인터페이스

아래 명령은 신설 도구의 계획 CLI이며 §9.1이 충족된 계산 노드에서만 실행한다.

```bash
python3 -B "$P/scripts/det_siiii_rates_readout.py" selftest \
  --production-source "$T2" \
  --out "$D/results/$RUN_ID/selftest"

python3 -B "$P/scripts/det_siiii_rates_readout.py" analyze \
  --e-run "$E" \
  --r-run "$R" \
  --production-source "$T2" \
  --prior-pop "$B/results/a2/analysis" \
  --selftest "$D/results/$RUN_ID/selftest" \
  --out "$D/results/$RUN_ID/analysis"
```

기계 프리플라이트의 계획 명령은 다음과 같다. `ORDER`는 운전자가 작성한 실제 구현 발주서다.

```bash
python3 -B "$P/scripts/check_prereg_preflight.py" \
  "$P/docs/PREREG_DET_SIIII_RATES_2026-10-07.md" --root "$P"

python3 -B "$P/scripts/check_gate_referents.py" \
  "$P/docs/PREREG_DET_SIIII_RATES_2026-10-07.md" --root "$P"

python3 -B "$P/scripts/check_doc_canonical.py" \
  "$P/docs/PREREG_DET_SIIII_RATES_2026-10-07.md"

python3 -B "$P/scripts/check_dispatch_preflight.py" \
  "$ORDER" "$P/docs/PREREG_DET_SIIII_RATES_2026-10-07.md" --root "$P"
```

러너는 `module purge` 후 실제 사용할 모듈을 고정하고, Python·의존성·스레드 수를 기록한다. 파이프나 최종 성공 문구가 앞 명령의 실패를 숨기지 않도록 각 rc를 보존·전파한다.

재현 기록에는 변수의 실제 값, 작업 디렉터리, 전체 명령, 문서·도구 신원, 작업 ID, 종료 상태, 자원·경과, 로그 경로가 있어야 한다.

CLI의 표현 차이는 집행 기록에 남길 수 있다. 물리식·세대·범위·수용 기준 변경은 astra 계약 개정으로 처리한다.

## §10. 완료 기준과 처분

### 10.1 필수 완료 내용

운전자 판정과 astra 종료감사는 다음을 함께 확인한다.

1. 실제 생산 topology와 표적의 직접·간접 경로.
2. 주요 셸의 인구·율 coverage 및 결손의 정확한 목록.
3. 계산 가능한 과정별 계수와 유입·유출·순율.
4. 산술 폐합, 세대 7 비평형 잔차, 생산 해 재현의 구분.
5. 적격한 경우의 밀도 차이 귀속과 지배 분기.
6. 실제 분석 경로의 정상·음성대조.
7. 기계 게이트 원로그·재현 명령·비용·보존·분장·이탈.
8. 원래 질문 중 해소한 부분과 남은 부분.

### 10.2 처분

- **`PASS`**  
  두 표적의 모든 주요 셸에서 필요한 세대 7 율 예산이 완전하고, 생산 방정식·출력 연결에 대한 적격한 폐합 검증이 완료되며, 등록된 귀속 분기를 낼 수 있다. 유효한 `MIXED` 또는 `UNRESOLVED`도 정당한 답일 수 있다. 이 PASS의 공간적 범위는 §2.1에서 개정 등록한 율 예산 주요 셸에 한정된다. 선행 주요 셸 중 J̄ 결손 셸의 귀속은 여전히 미해소이며, 판정문 첫 문단에 이 범위를 명시한다. 해당 결손만으로 개정된 주요 셸의 PASS를 배제하지 않지만, 이를 선행 주요 셸 전체의 율 귀속 달성으로 표시하지 않는다. 이 PASS도 물리적 정오·인과 확정·수리 승인을 뜻하지 않는다.

- **`PASS_LIMITED`**  
  계산 가능한 필수 산출과 결손 처리가 완료됐지만, J̄·인구·경계 시점 등의 결손 또는 생산 해 재현 불가 때문에 원래 율 귀속 목표가 일부 미해소다. 첫 판정 문단에 **“전체 율 예산/생산 해 폐합/밀도 차이 귀속 중 무엇이 미달인지”**를 적는다. 부분 기록에서 얻은 NRB 차이를 전체 준위의 원인으로 승격하지 않는다.

- **`REWORK`**  
  연결·생산 분기·율·오차·폐합·분기 구현이 계약과 다르거나, 필수 음성대조가 실제 분석 경로를 검증하지 못한다.

- **`BLOCKED`**  
  실행 환경·자원·핵심 입력·생산 권위 때문에 가능한 것으로 등록한 최소 산출도 완료하지 못한다.

설명되지 않는 `SOLUTION_MISMATCH`는 생산 오류로 즉시 판정하지 않는다. 오프라인 구현 오류와 입력·시점 불일치를 먼저 구분한다. 생산 수리가 필요하다는 결론이 나와도 이 단에서 수리하지 않는다.

유효한 수치 미분리와 자료 결손을 같은 분기로 합치지 않는다. 결손 목록을 정확히 만들었다는 이유만으로 완전한 목표 달성을 선언하지 않는다.

## §11. 집행 기록·분장 장부

이번 직접 발주의 분장을 적용한다.

| 업무 | 담당 | 이해관계·규칙 |
|---|---|---|
| 기안·개정·종료감사 | astra | 계약 기안자가 자기 계약의 이행을 감사한다는 한계 명시 |
| Q-GOAL·Q-LEAN 사전감사 | Fable | 호출 실패 시 gpt-6.1-sol; 실패 기록과 실제 모델 ID 보존 |
| 구현·직접시험·기계 게이트·실행·판독·판정 | 운전자 | 구현자·판정자 겸임 명시 |
| 범위·비용 확대·후속 물리 결정 | 사용자 | 현재 승인 밖 결정 |

미집행 항목을 예상 결과로 채우지 않는다.

| 집행 항목 | 담당 | 상태·증거 |
|---|---|---|
| 사용자 결정 | 사용자 | §0 결정문 |
| 본문 기안 | astra | `gpt-6-astra`, read-only, 본문 |
| 정본 파일화·기준선 연결 | 운전자 | 미집행 |
| 기계 프리플라이트 | 운전자 | 미집행; 명령·문서 해시·stdout·stderr·rc 보존 |
| 사전감사 | Fable 또는 기록된 대행 | 미집행; Q-GOAL·Q-LEAN을 이 순서로 기록 |
| 실행 환경 규칙 확인 | 운전자 | `/etc/slurm/job_submit.lua`의 a10 GPU 요청 검사 제외 확인; §9.1의 CPU 전용 Slurm 경로 등록, 제출 미집행 |
| 입력 연결·도구 구현·신원 고정 | 운전자 | 미집행 |
| G-LINK·G-RATE·음성대조 | 운전자 | 미집행 |
| 본 분석·산출 신원 | 운전자 | 미집행 |
| 비용·보존·재현 | 운전자 | 미집행 |
| 목표 달성·미달 판정 | 운전자 | 미집행 |
| 종료감사 | astra | 미집행 |
| 이탈·미해결·후속 결정 | 실제 담당 | 미집행 |

게이트나 감사의 과거 실행 로그가 없으면 `LOG_NOT_PRESERVED`라고 기록한다. 나중 재실행을 과거 실행의 원로그로 표시하지 않는다.

## §12. 기계 선언

아래 선언은 §6.4의 **표적·셸별 귀속 코드**를 검증한다. 전체 과학 계산이나 생산 해 폐합의 대체물이 아니다.

`v`는 단일 코드여야 한다. 0–6 외의 값, 결손, 복수 코드는 `X`다. 각 코드에는 별도로 coverage·폐합 상태·지배 과정명·세대·사유를 기록한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SIIII-RATES (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_siiii_rates_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/tools/run_siiii_rates\\.slurm)",
    "symbol": "det_siiii_rates_",
    "roots": [
      "scripts/det_siiii_rates_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/tools/run_siiii_rates.slurm"
    ],
    "expected_extra": [
      "scripts/det_siiii_rates_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/tools/run_siiii_rates.slurm"
    ]
  },
  "branches": {
    "regimes": [
      [0.0, 0.0],
      [1.0, 1.0],
      [2.0, 2.0],
      [3.0, 3.0],
      [4.0, 4.0],
      [5.0, 5.0],
      [6.0, 6.0],
      [9.0, 9.0]
    ],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2,3,4,5,6) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",
      "c2": "sum(1 for x in v if x == 2)",
      "c3": "sum(1 for x in v if x == 3)",
      "c4": "sum(1 for x in v if x == 4)",
      "c5": "sum(1 for x in v if x == 5)",
      "c6": "sum(1 for x in v if x == 6)"
    },
    "rules": [
      {"name": "X", "predicate": "not valid"},
      {"name": "NRB_FORM_DOMINANT", "predicate": "valid and c0 == 1"},
      {"name": "OTHER_RATE_DOMINANT", "predicate": "valid and c1 == 1"},
      {"name": "PARTNER_POPULATION_DOMINANT", "predicate": "valid and c2 == 1"},
      {"name": "RESIDUAL_DOMINANT", "predicate": "valid and c3 == 1"},
      {"name": "MIXED", "predicate": "valid and c4 == 1"},
      {"name": "UNRESOLVED", "predicate": "valid and c5 == 1"},
      {"name": "NOT_IDENTIFIABLE", "predicate": "valid and c6 == 1"}
    ],
    "adversarial_fixtures": [
      {"name": "nrb", "mix": [[0,1]]},
      {"name": "other-rate", "mix": [[1,1]]},
      {"name": "partner-population", "mix": [[2,1]]},
      {"name": "residual", "mix": [[3,1]]},
      {"name": "mixed", "mix": [[4,1]]},
      {"name": "unresolved", "mix": [[5,1]]},
      {"name": "missing-physics", "mix": [[6,1]]},
      {"name": "invalid-code", "mix": [[9,1]]},
      {"name": "missing-code", "mix": []},
      {"name": "multiple-codes", "mix": [[0,1],[1,1]]}
    ]
  },
  "references": [
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/CLAUDE.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_DET_SIIII_POP_2026-10-07.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_SIIII_POP_2026-10-07.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_OBS_LINEDECOMP_2026-10-07.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_DET_NRB_E_2026-10-02.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_NRB_E_2026-10-02.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/astra_closeout.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_pop_20261007/astra_closeout2.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_rates_20261007/evidence/USER_DECISION_2026-10-07.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c",
      "flags_existing": [
        "nlte_assemble_rate_matrix_impl",
        "nlte_solve_ion_shell",
        "nlte_bb_rate_graph_contains",
        "nlte_bb_jbar_canonical",
        "bb_nrb_arm",
        "nlte_bf_gamma_canonical",
        "parity_rate_se_ratio_checked",
        "nlte_build_projection",
        "nlte_precompute_within_sl_frac_checked",
        "nlte_update_tau_sobolev_with_authority",
        "CE_REACTIONS"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c",
      "flags_existing": [
        "line_net_bracket_rates",
        "line_net_cmfgen_exponx",
        "line_net_sobolev_radiation"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c",
      "flags_existing": [
        "load_atomic_data",
        "LUMINA_SUPER_CUTOFF"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/population_contract.c",
      "flags_existing": [
        "population_within_superlevel_build",
        "population_generator_stationary_gth",
        "population_dense_solve_equilibrated"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/population_contract.h",
      "flags_existing": [
        "POP_GENERATOR_RESIDUAL_LIMIT",
        "POP_DENSE_BACKWARD_ERROR_LIMIT"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/opacity_publication.c",
      "flags_existing": [
        "a208_signed_sobolev_counted",
        "a208_line_source_counted"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c",
      "flags_existing": [
        "cmfgen_run",
        "det_obs_capture_pair",
        "det_obs_final_pass_hook",
        "lumina_r7_publish_and_solve_te"
      ]
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/check_prereg_preflight.py"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/check_gate_referents.py"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/check_doc_canonical.py"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/check_dispatch_preflight.py"
    }
  ]
}
```

## 기안 메모

- 공유 문맥 전문, 프로젝트 정체성, 선행 계약·판정·종료감사, T2의 관련 생산 함수, E′ exports와 입력 헤더·캡처 manifest를 읽었다. 파일 수정·수치 분석·시험·제출은 하지 않았다.
- 생산 소스는 세대 7 캡처 뒤에 물질 갱신을 수행한다. 같은 세대 번호만으로 평형해·전자 상태의 동일성을 가정하지 않았다.
- 표적의 직접 bound-free·비열적 항과 간접 이온화 영향을 구분했다. 최종 활성 분기 확인은 양쪽 런의 설정·기본값 연결에 남아 있다.
- 개정 1은 본 단의 율·밀도 결과 판독 전에 W5 자료 가용성 증거로 율 예산 주요 셸을 1892.03 Å의 24·30, 2542.58 Å의 19로 등록했다. 두 런의 기록 셸에는 표적 준위에 연결된 모든 선의 J̄가 있지만, 상대 준위 인구·J_c·경계 상태 등의 적격성은 별도 확인 대상이다. 선행 주요 셸의 나머지는 J̄ 결손과 `NOT_IDENTIFIABLE`을 보존하며 세대 8이나 coarse J로 메우지 않는다.
- 비용은 지출 상한이다. a10의 CPU 전용 작업 허용을 제출 필터에서 확인하여 §9.1의 환경 충돌·`ENVIRONMENT_BLOCKED` 기재를 정정했다. 기존 비용 상한은 유지한다.