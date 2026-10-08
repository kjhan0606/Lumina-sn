# 사전등록 — DET-COLL-OMEGA-OFFLINE: 저장 상태에서의 Si III·S III 충돌율 비교와 다음 물리 단의 우선순위

2026-10-08 · astra 기안 · 읽기 전용  
운전자 파일화 대상(신설): `docs/PREREG_DET_COLL_OMEGA_OFFLINE_2026-10-08.md`  
상태: **기안. 파일화·기계 게이트·구현·직접시험·CPU 판독·판정·종료감사는 미집행.**

이번 단은 **저장 상태에 두 충돌 처방을 대입하는 오프라인 율 비교**다. 생산 소스 변경, 생산 바이너리 실행, GPU 예약·실행은 없다. 산출의 소비처는 **NRB 정상 상태를 확인하는 별도 단 이후, 충돌 처방 변경을 다음 물리 단으로 우선할지에 관한 결정**이다.

## §0. Q-GOAL·Q-LEAN 자기답과 과학 질문

### Q-GOAL — 최종목적에 직접 기여하는가?

최종목적은 Monte-Carlo 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현하는 것이다. 직전 단은 Si III·S III의 충돌표가 덱에 존재하지만 E′·R′·X의 율 행렬에서 소비되지 않았다는 사실을 확인했다. 남은 질문은 **그 소비 경로를 바꾸면 동일한 저장 물질 상태에서 충돌율 자체가 얼마나 달라지는가**다.

이번 단은 다음을 측정한다.

1. 고정 전이 집합에서 CMFGEN에서 가져온 표가 덮는 범위와 덮지 않는 범위.
2. 같은 저장 \(n_e,T_e\)에서 COLL_FIX와 봉인 생산 OMEGA 분기가 만드는 상·하향 충돌율의 차이.
3. 표·근사식·중복 억제·선 없는 표 전이가 각각 그 차이에 참여하는 방식.
4. 그 결과가 충돌 처방 변경의 다음 단 우선순위를 높이는지, 보류하게 하는지, 추가 설계 없이는 순서를 정할 수 없게 하는지.

스펙트럼 어긋남을 직접 줄이는 단은 아니다. 이번에 줄이는 것은 **충돌 자료 소비 차이의 율 규모에 관한 불확실성**이다. 최종목적에 더 가까운 NRB 판정런은 사용자 결정에 따라 별도 계약으로 진행한다. 이번 오프라인 비교가 그 단을 대체하거나 지연시키지 않는다.

이 비교를 생략하면 실제 율 차이를 모른 채 충돌표 활성화를 다음 변경으로 선택하거나, 표 미소비라는 사실만으로 효과를 크게 추정하게 된다.

**기안자 자기답: PASS.**

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

| 측정·확인 | 소비하는 결정 |
|---|---|
| 고정 표본의 COLL_FIX/OMEGA 율 비 분포 | 다음 물리 단으로 충돌 처방 변경을 우선할지 |
| 표 수록·미수록과 생산 분기별 분해 | 다음 후보가 표 자료의 교체인지, 근사식까지 포함하는 처방 변경인지 |
| 표적 준위에 직접 연결되는 전이의 율 차이 | 표적 전이만 작다는 이유로 연결 경로의 변경을 놓치지 않을지 |
| 상세평형과 봉인 C 계산 대조 | 비교 산출을 우선순위 결정에 사용할 자격이 있는지 |
| 미수록·미적용·결손 목록 | 다음 계약이 어디까지 주장하고 어떤 전이를 남겨야 하는지 |

주 비교는 R′의 물질 세대 7로 제한한다. E′·X는 직전 단의 충돌 소비 사실을 승계하는 근거로만 사용한다. R′ 세대 3에서는 기존 반례인 line 471957·sh10만 대조한다.

다음은 추가하지 않는다.

- E′·R′·X의 전체 세대 궤적 재분석.
- W5 원천·준위 인구의 재복원.
- 방사율과의 경쟁, 순유량, 다준위 수지 또는 SE 해.
- 공개 CMFGEN 스펙트럼 재비교.
- 충돌표 변경 뒤의 인구·이온화·스펙트럼 예측.
- 다른 원소의 율 census.
- 다른 온도·밀도·상수값 탐색.
- 별도 생산 진단 기능 또는 작은 검사별 감사.

새 판독기 하나에 입력 연결, 율 계산, 직접시험, 분포와 다음 결정표를 묶는다. CPU 러너 하나가 이를 집행한다. 내부 게이트는 입력 연결과 판독 검증으로 통합한다.

**기안자 자기답: APPROPRIATE.**

### 승인 근거·분장

승인 근거는 다음 파일의 「병행할 만한 수 분짜리 오프라인 비교」 항목이다.

`/gpfs/kjhan/lumina/cmfgen_map/coll_omega_20261008/evidence/USER_DECISION_2026-10-08.txt`

같은 결정문의 NRB 판정런은 별도 계약의 범위다.

- 기안·개정·종료감사: astra.
- 경량 사전감사: 사용자 결정 개정25-ab에 따라 생략.
- 구현·직접시험·기계 게이트·실행·판독·판정: 운전자.
- 코드 감사: 해제 상태 유지.
- 범위·비용 확대, 생산 변경, 후속 런: 사용자 결정.

운전자는 구현자와 판정자를 겸한다. astra는 계약 저자로서 종료감사를 맡는다. 경량 사전감사 생략을 독립 감사 통과로 표시하지 않는다.

### 사전 노출·주장 상한

기안자는 직전 계약·판정·result.json, 기존 COLL_FIX 판독 함수, 봉인 충돌 소비 소스, 덱 manifest와 충돌 CSV의 일부 내용에 노출됐다. 표본과 우선순위 기준은 이 노출 뒤 고정한다. **이번 율 비의 수치 분석은 수행하지 않았다.**

이번 결과로 다음을 주장하지 않는다.

- 충돌표 활성화의 스펙트럼·이온화 효과나 개선 방향.
- 물리적 정답 또는 CMFGEN 전체 충돌 처리와의 일치.
- R′ 세대 7이 정상 상태라는 주장.
- 저장 상태가 변경 뒤에도 유지된다는 주장.
- 생산 승격, 기본값 전환 또는 후속 런 승인.

## §1. 경로·기준선·승계 사실

### 1.1 경로

```text
P = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn

C = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
T = C/source/src
R = C/Rprime/run1
M = R/input/model

D = /gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008
S = /gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008
A = S/results/r4/analysis/analysis

O = /gpfs/kjhan/lumina/cmfgen_map/coll_omega_20261008

고정 표본 권위 = D/r3_lock_census.csv
직전 표본 대조 = A/sample.json
직전 충돌 소비 판정 = A/result.json

새 판독기 = P/scripts/det_coll_omega_offline_readout.py
새 CPU 러너 = O/tools/run_det_coll_omega_offline.slurm
새 결과 = O/results/<실행 식별자>
집행 증거 = O/evidence/<집행 식별자>

CMFGEN 소스 참조 = /gpfs/kjhan/cmfgen_src/cur_cmf
```

O에는 사용자 결정문이 이미 있다. 결과와 시험 산출은 미사용 실행 식별자 아래에만 기록한다. 입력 런·직전 결과·기존 도구의 출력 경로를 쓰기 대상으로 삼지 않는다.

### 1.2 기준선·신원

기안 시 확인한 프로젝트 루트는 P이고 HEAD는 다음과 같다.

```text
7a0b6dc296e41cd2cf8f740f686c8fa2a6ac3a7e
```

형식 기준은 `docs/PREREG_DET_SEEDMEM_SCOPE_2026-10-08.md`다.

실행 전 운전자가 계산 장소에서 다음 연결을 확인하고 실제 소비 파일의 SHA-256을 기록한다.

- R의 STAGE_IDENTITY.txt와 input/run_build_identity.json.
- input/resolved_lumina.exports와 봉인 소스의 기본값·분기.
- input/deck.sha256와 실제 소비 원자 파일.
- input/fixed_te_profile.txt와 physics manifest의 온도 레인.
- §2의 표본 권위 및 저장 물질 파일.
- 재사용 Python 함수, C 참조 계산에 소비하는 소스·헤더.
- 본 계약, 새 판독기, 러너와 직접시험 산출.

생산 바이너리는 신원 기록의 대상이며 실행 대상이 아니다. 기안자가 해시를 재계산한 것으로 표시하지 않는다.

### 1.3 승계 사실

직전 판정문 §5와 result.json의 collision_authority를 다음 범위로 승계한다.

- E′·R′·X 모두 COLL_FIX per-line 분기를 사용했다.
- 세 런의 `LUMINA_NLTE_COLL_FLOOR`는 0이었다.
- 일반 충돌표 로더는 OMEGA_CMFGEN·ARTIS_PARITY 게이트 때문에 호출되지 않았다.
- 해당 율 행렬은 Si III·S III의 col_data를 소비하지 않았다.

여기서 **COLL_FLOOR=0은 COLL_FIX 식 내부의 \(\Upsilon\ge1\) 처방이 없다는 뜻이 아니다.** 별도 열화율 하한과 COLL_FIX 충돌강도 식을 구별한다.

line 471957·sh10의 call 3에 관한 기존 배제는 **고정 복사 배경·덱 계수·상세평형을 유지한 2준위 정상 모형**에 한정된다. 이번 율 비교로 그 배제를 해소하거나 전체 결합계로 확대하지 않는다.

### 1.4 직접 확인한 생산 관계

아래 위치는 T 판본 기준의 대략적인 행 번호다.

| 근거 | 확인한 관계 |
|---|---|
| lumina_cuda.cu 약 7132행 | OMEGA 단독 활성화도 manifest 로더를 호출하며 실패하면 종료 |
| lumina_cuda.cu 약 7146행 | 표 로드 뒤 omega_cmfgen_arm 호출 |
| lumina_atomic.c 약 2715행 | IGC1 이진 표의 헤더·준위 수·색인 검사 |
| lumina_atomic.c 약 2815행 | status가 OK인 manifest 행의 로드와 Fe III 중복 제외 |
| lumina_plasma.c 약 759행 | artis_col_rates의 충돌강도 입력 경로와 상·하향 율 |
| lumina_plasma.c 약 828행 | ma_ru_upsilon의 선형 온도 보간 |
| lumina_plasma.c 약 895행 | omcm_ex_e1x의 실제 근사 계산 |
| lumina_plasma.c 약 958행 | 미수록 전이의 van Regemorter·OMEGA_SET 분기 |
| lumina_plasma.c 약 969행 | 양의 표 보간값을 tier 1로 선택 |
| lumina_plasma.c 약 996행 | 원소·이온·준위쌍을 표 레코드에 연결 |
| lumina_plasma.c 약 19260행 | OMEGA → ARTIS_PARITY → COLL_FIX 선택 순서 |
| lumina_plasma.c 약 19690행 | 일반 col_data 전이를 별도 순회하여 율을 추가 |
| lumina_plasma.c 약 12390행 | 다른 소비처인 RADEQ 계수는 10400 K에서 준비 |
| det_siiii_rates_readout.py 약 191행 | 기존 COLL_FIX Python 구현 |
| omega_gen_v3.f 약 116행 | CMFGEN 원본의 표 내부 log-log 보간 |
| omega_gen_v3.f 약 162행 | CMFGEN 원본의 미수록 전이 근사와 SAME_N 처리 |

소스 관계를 확인한 것과 저장 런에서 해당 분기가 실행됐다는 것은 별개다. OMEGA 활성화 쪽은 **봉인 분기를 저장 상태에 대입한 비교값**으로 표시한다.

## §2. 식별 가능성·표본·세대의 사전 고정

### 2.1 주 표본

D의 r3_lock_census.csv에서 다음 조건을 만족하는 기존 키를 그대로 사용한다.

\[
Z\in\{14,16\},\qquad ion=2,\qquad
shell\in\{10,15,19,24,30\},\qquad
beta2_E<10^{-3}.
\]

- Si III: 86개 선·셸 키.
- S III: 424개 선·셸 키.
- 합계: **510개 키**.
- line 600693의 다섯 셸과 line 471957·sh10을 포함한다.

새 율·표 수록 여부·율 비·과거 모형 잔차를 선택 조건으로 사용하지 않는다. 키를 정렬하여 계산 전에 sample.json에 기록하고 직전 sample.json과 대조한다. 불일치를 키 재선택으로 맞추지 않는다.

주 표본은 직전 조사에서 선택된 표본이며 전체 Si III·S III 전이의 무편향 표본이 아니다.

### 2.2 연결 전이

각 셸에서 주 표본의 하·상준위를 모은다. 보조 집합은 그 준위에 직접 닿는 **같은 이온의 한 단계 연결 전이**다.

다음을 합쳐 준위쌍으로 중복 제거한다.

1. line_list.csv의 연결 선 전이.
2. 해당 이온 이진 충돌표의 연결 준위쌍. 방사 선이 없는 표 전이도 포함한다.

연결된 상대 준위에서 다시 확장하지 않는다. 다른 셸·이온을 추가하지 않는다. 한 준위쌍에 여러 line_id가 있으면 모든 선을 보존하되, 준위쌍 합계와 선별 성분을 구분한다.

직전 판정의 선 집합 계수인 표본 선 161개·사용 대상 선 3193개는 **방사 선 집합의 대조값**이다. 이번에 추가로 드러나는 선 없는 표 전이를 그 수에 억지로 맞추지 않는다.

보조 집합은 주 표본의 성공·실패 분모에 합치지 않는다.

### 2.3 저장 상태

| 용도 | 런·셸·세대 | 실제 파일 |
|---|---|---|
| 주 비교와 연결 전이 | R′, 셸 10·15·19·24·30, 물질 세대 7 | physics_DET_iter0004.manifest.json 및 그 shell_file |
| 기존 반례 연결 대조 | R′, line 471957·sh10, 물질 세대 3 | physics_DET_iter0000.manifest.json 및 그 shell_file |

위 파일은 R의 work/physics_comparison에 있다.

- \(n_e\): shell.csv의 `n_e_cm3`.
- \(T_e\): 같은 행의 `T_e_K`.
- 셸 연결: 행 순서가 아닌 `shell_id`.
- iter0004: population_generation=7, radiation_generation=6, te_generation=6.
- iter0000: population_generation=3, radiation_generation=2, te_generation=2.
- 두 덤프의 transaction_status=COMMITTED와 FIXED_T 레인을 확인한다.

온도는 해당 물질 덤프가 기록한 값을 사용하고, manifest의 별도 te_generation도 보존한다. 파일명만으로 세대를 추정하지 않는다.

세대 7은 **고정 비교 배경**이다. 이를 정상 상태 또는 실제 OMEGA 실행 상태로 부르지 않는다. 저장 \(n_e\)가 과거 내부 반복의 모든 율 평가 시점과 같다는 주장도 하지 않는다.

세대 3 대조는 다음 순서 판정의 분모에 넣지 않는다.

### 2.4 식별 범위

식별 대상은 저장 상태에 조건부인 충돌강도, 상·하향 원자율, 생산 분기의 채널 선택과 중복 억제다.

다음은 이번 입력·계산으로 식별하지 않는다.

- 변경 뒤 \(n_e,T_e,J,\tau\), 준위 인구 또는 이온분율.
- 방사 과정과 충돌 과정의 상대적 중요도.
- 전체 SE 행렬의 해와 super-level 인구 응답.
- 스펙트럼 변화와 공개 CMFGEN 일치도.
- NRB 정상 상태에서 다시 평가했을 때의 율 분포.

## §3. 입력 권위·복원·소비 경로

### 3.1 원자 자료 권위

수치 계산의 권위는 M의 다음 파일이다.

- levels.csv.
- line_list.csv.
- coldata_cmfgen_manifest.csv.
- ige_col_14_2_cmfgen.bin.
- ige_col_16_2_cmfgen.bin.

이진 표는 manifest의 실제 out_bin을 따라 읽는다. 파일 이름만으로 원소를 정하지 않고 헤더의 Z·ion0와 덱을 대조한다.

참고 CSV는 다음과 같이 구별한다.

| 파일 | 대상·용도 |
|---|---|
| siii_terms_col_cmfgen.csv | Si III, Z=14·ion0=2의 항 분할 자료와 출처 |
| siii_col_cmfgen.csv | S III, Z=16·ion0=2의 자료와 출처 |
| siii_terms_col_cmfgen_provenance.txt | Si III 항에서 세부 준위쌍으로 옮긴 경위 |

Si III의 manifest는 147준위·178개 매핑 전이·9개 온도를, S III는 덱 256준위·227개 매핑 전이·8개 온도를 기록한다. 이는 기안 시 읽은 metadata이며 실행 시 이진 헤더·레코드와 대조한다.

S III 참고 CSV의 원본 380준위 표기를 덱 이진 표의 256준위와 혼동하지 않는다. 원본 항 자료를 다시 분할하거나 참고 CSV로 이진 표를 대체하지 않는다.

### 3.2 로더 적격성

생산 OMEGA 활성화는 전체 manifest 로드의 성공을 전제로 한다. 따라서 status가 OK인 행의 실제 파일과 봉인 로더 검사에 필요한 준위 수·레코드를 확인한다. Fe III의 기존 표와 겹치는 행은 봉인 로더의 제외 규칙을 따른다.

이는 다른 원소의 율 조사로 확대하지 않는다. **생산이 로더 단계에서 종료할 자료를 정상 활성화로 표시하지 않기 위한 입력 검사**다.

표의 비유한 값, 잘못된 색인, 모순 중복 또는 서로 다른 매핑을 조용히 삭제·평균·교체하지 않는다. 실패 사유와 해당 범위를 기록한다.

### 3.3 고정 분기 설정

저장 기준선은 R의 유효 설정과 소스 기본값으로 확인한다.

- COLL_FIX 활성.
- OMEGA_CMFGEN·ARTIS_PARITY 비활성.
- 별도 COLL_FLOOR=0.
- CPU NLTE 조립 경로.
- 표적 이온의 추가 metastable 충돌 경로 비활성.
- super-level 설정과 실제 매핑 유지.

비교 쪽은 **다른 조건을 고정한 채 OMEGA 선택만 활성화한 봉인 분기 평가**다.

- ARTIS_PARITY는 활성화하지 않는다.
- OMEGA_SET은 봉인 기본값 0.1로 고정한다.
- 온도는 §2의 저장값을 사용한다.
- 부모 셸의 환경변수를 암묵적으로 상속하지 않는다.
- OMEGA 관련 정적 초기화 때문에 설정별 C 참조 실행은 별도 프로세스로 분리한다.

### 3.4 세 가지 장부

각 전이에 다음을 따로 기록한다.

| 장부 | 의미 |
|---|---|
| `table_coverage` | 같은 Z·ion0·준위쌍이 로드된 이진 표에 존재하는가 |
| `omega_tier` | 해당 저장 온도에서 표, van Regemorter, OMEGA_SET 중 무엇을 선택하는가 |
| `production_contribution` | 선별 순회·표 순회 중 어디서 얼마를 더하며 무엇을 억제·건너뛰는가 |

표에 행이 있다는 사실만으로 tier 1 또는 실제 행렬 기여를 선언하지 않는다. 생산 함수는 보간값이 양수인 경우 tier 1을 선택한다. 선별 순회와 표 순회의 적용 조건도 별도로 확인한다.

방사장 주파수 영역 밖이라는 이유만으로 충돌 전이를 제거하지 않는다. 봉인 소스의 방사 분기와 충돌 계산 도달 조건을 구별한다.

## §4. COLL_FIX·OMEGA 율과 생산 분기

### 4.1 공통 정의와 COLL_FIX

\[
\Delta E=|E_u-E_l|\,{\tt EV\_TO\_ERG},\qquad
x=\frac{\Delta E}{k_BT_e},\qquad
K=8.629\times10^{-6}.
\]

유효 충돌강도 \(\Upsilon\)에 대해 원자율은

\[
C_{lu}=\frac{n_eK\Upsilon}{g_l\sqrt{T_e}}e^{-x},
\qquad
C_{ul}=\frac{n_eK\Upsilon}{g_u\sqrt{T_e}}
\]

이며 단위는 \({\rm s^{-1}}\)이다. \(n_e\)를 제외한 율계수와 혼동하지 않는다.

봉인 COLL_FIX 식은

\[
\Upsilon_{\rm FIX}
=
\max\!\left[
14.5\,f_{lu}
\left(\frac{13.6057}{\Delta E_{\rm eV}}\right)^2
\,0.2,\;1.0
\right].
\]

기존 `det_siiii_rates_coll`을 수정 없이 재사용할 수 있다. 재사용 시 입력 적격성 검사는 호출 전에 수행한다. 잘못된 입력에서 함수가 반환한 0을 정상적인 물리적 영으로 받아들이지 않는다.

### 4.2 생산 OMEGA 선택

**표 수록·양의 보간값**

봉인 Lumina 분기는 이진 표의 \(\Upsilon(T)\)를 온도에 대해 선형 보간한다. 선택된 구간과 두 끝값을 기록한다. 온도 범위 밖에서는 봉인 함수의 끝점 사용 여부를 별도 표지로 남긴다.

**표로 선택되지 않은 선 전이, \(f_{lu}>10^{-5}\)**

Si III·S III에 적용되는 식은

\[
F_L=\frac{\Delta E}{h\,10^{15}},
\]

\[
\Upsilon_{\rm vR}
=
47.972\,
\bar g(x)\,
f_{lu}g_l/F_L,
\]

\[
\bar g(x)=\max\!\left[0.2,\;0.276\,{\cal E}_{\rm src}(x)\right].
\]

\({\cal E}_{\rm src}\)는 봉인 `omcm_ex_e1x`의 다항식·유리식이다. 수학 라이브러리의 다른 \(e^xE_1(x)\) 구현으로 바꾸지 않는다.

**그 밖의 유효 미수록 선 전이**

\[
\Upsilon_{\rm SET}=0.1.
\]

경계 \(f_{lu}=10^{-5}\)는 OMEGA_SET 쪽이다.

준위 연결이나 양수 조건이 잘못되어도 생산 함수가 OMEGA_SET을 반환할 수 있다. 이 경우를 정상적인 약한 선 근사와 구별하여 `INVALID_INPUT_FALLBACK`으로 기록하고 과학 비교의 적격값으로 사용하지 않는다.

### 4.3 실제 추가와 중복 억제

생산 OMEGA 분기의 충돌 기여는 다음과 같다.

| 경우 | 선별 순회 | 일반 표 순회 |
|---|---|---|
| 표에서 양의 \(\Upsilon\)를 선택한 선 | 충돌항 0 | 적격한 표 레코드를 한 번 추가 |
| 표로 선택되지 않은 선 | vR 또는 SET 율 추가 | 적격한 별도 표 기여가 있으면 원문 조건대로 기록 |
| 방사 선이 없는 표 준위쌍 | 선별 항 없음 | 적격하면 표 율 추가 |
| 같은 solve 단위 내부의 표 전이 | 선 매핑 상태 별도 기록 | 생산의 동일 solve 색인 조건으로 제외 |
| 잘못된 매핑·비양의 에너지차 등 | 원문 도달·반환 상태 기록 | 건너뛰는 사유 기록 |

주 비교는 **준위쌍에 실제로 공급되는 충돌 성분을 한 번씩 합한 값**이다. 표 전이에 대해 선별 0만 읽고 OMEGA 율이 0이라고 하지 않는다. 반대로 각 line_id에 같은 표 율을 반복해서 더하지 않는다.

같은 준위쌍의 선 집합을 \(L_p\)라 하면, 적격한 서로 다른 solve 단위 사이에서

\[
C_{{\rm FIX},p}=\sum_{\ell\in L_p}C_{{\rm FIX},\ell},
\]

\[
C_{\Omega,p}
=
\sum_{\ell\in L_p}C_{{\rm fallback},\ell}
+
\sum_{t\in p}C_{{\rm table},t}.
\]

상·하향 각각 계산한다. 합산 순서는 원래 선 행·표 레코드 순서를 보존한다.

율은 **within-SL 분율을 곱하기 전의 준위쌍 원자율**로 표시한다. solve 단위와 생산의 적용·제외 상태는 함께 기록한다. 이를 완성된 super-level 행렬 원소나 전체 행렬 재생으로 부르지 않는다.

같은 solve 단위 내부의 연결 전이는 미시적 율을 보존하되 순서 결정용 유효 연결 분포에서 분리한다. 이는 율 크기에 따른 사후 제외가 아니라 고정 매핑에 따른 분리다.

### 4.4 원본 CMFGEN과의 차이

이번 분모는 **봉인 Lumina OMEGA 분기**다. CMFGEN 전체 실행의 율이라고 명명하지 않는다.

확인된 차이는 다음과 같다.

- Lumina 표 보간은 선형 온도, CMFGEN 원본은 표 내부 log-log 보간.
- Lumina의 해당 근사에는 원본 SAME_N에 따른 \(\bar g\) 선택이 없다.
- Lumina의 미수록 근사는 선 목록을 통해 추가된다. 모든 가능한 준위쌍을 자동 생성하는 것으로 취급하지 않는다.
- OMEGA 전역 스위치는 Si III·S III 이외의 표도 로드하며 RADEQ 준비 경로에도 영향을 준다. RADEQ의 10400 K 평가를 이번 NLTE 저장 온도 평가와 혼합하지 않는다.

이 차이를 없애는 수리 또는 별도 CMFGEN 율 재구현은 이번 범위에 없다.

### 4.5 clamp/floor 금지의 적용

새 판독기는 음수·결손·비유한 값, 율 비 분모, 온도 또는 오차를 clamp/floor로 수리하지 않는다. 표 값에 새로운 하한을 씌우지 않는다.

§4.1·§4.2의 기존 하한과 끝점 처리는 **비교 대상인 봉인 처방의 구성요소**다. 이를 제거하여 다른 처방으로 바꾸지도, 새 수리 기능으로 추가하지도 않는다. 기존 처방의 존재·선택 여부를 기록하며 물리적 정당성을 승인하지 않는다.

## §5. 율 비·상세평형·다음 결정

### 5.1 율 비와 분포

주 비교량은 준위쌍 합계의

\[
R_{lu}=\frac{C_{{\rm FIX},lu}}{C_{\Omega,lu}},
\qquad
R_{ul}=\frac{C_{{\rm FIX},ul}}{C_{\Omega,ul}}
\]

이다. 양의 값에 대해

\[
d=\log_{10}R_{ul}
\]

을 사용한다.

두 처방이 같은 에너지·통계중량과 상세평형을 사용하면 상·하향 비는 같아야 한다. 두 비를 모두 계산하여 이 관계를 검사하되, 서로 독립적인 물리 결과로 세지 않는다.

분포는 다음 집합을 분리한다.

- 주 표본 Si III 86키, S III 424키.
- 각 집합의 표 선택·vR·SET 및 혼합 성분.
- 주 표본을 제외한 연결 준위쌍·셸 집합.
- 선 없는 표 전이.
- 생산 적용 제외·결손·수치 미분리.

각 분포는 고정 분모, 양의 비 개수, 영의 상태, 최소·10백분위·중앙값·90백분위·최대, 두 배 초과 차이의 개수를 출력한다. 분위수는 정렬된 \(d\)의 nearest-rank 방식으로 고정하고 필요하면 \(10^d\)로 돌려 표시한다.

주 표본에 같은 준위쌍이 여러 line_id로 나타나면 기존 510키 집계와 고유 준위쌍 집계를 함께 구별한다. 표본 빈도를 독립적인 전이 개수로 해석하지 않는다.

### 5.2 상세평형

양의 적격 율에 대해

\[
\epsilon_{\rm DB}
=
\frac{C_{lu}/C_{ul}}
{(g_u/g_l)e^{-x}}-1
\]

을 COLL_FIX, 표 율, fallback 율과 준위쌍 합계에서 검사한다.

실행 구현의 비교선은

\[
|\epsilon_{\rm DB}|\le10^{-12}
\]

로 고정한다. 이는 구현 일관성 기준이며 물리 모델의 정확도 기준이 아니다.

- 실제 미적용 채널의 0은 상세평형 검사 대상에서 분리한다.
- underflow 등으로 비가 정의되지 않으면 4로 남긴다.
- 결손을 0으로 채우지 않는다.
- 상세평형 통과만으로 충돌강도 정규화·표 선택·중복 억제가 맞다고 하지 않는다.

### 5.3 사전 고정한 우선순위 기준

다음 명제를 주 표본의 각 키에 적용한다.

> H-SMALL: 저장 상태에서 COLL_FIX와 생산 OMEGA의 준위쌍 충돌율 차이가 두 배 이내다.

\[
\frac12\le R_{ul}\le2.
\]

유효한 한쪽 율만 구조적으로 0이면 두 배 이내 명제의 반례다. 양쪽 모두 생산상 미적용이면 변화 없음으로 별도 계수하고 양의 율 분포에는 넣지 않는다.

이온별 고정 분모 \(N_Z\)에 대해 확정 반례 수를 \(B_Z\), 미분리·결손 수를 \(U_Z\)라 두고

\[
f_Z^{\rm lo}=B_Z/N_Z,\qquad
f_Z^{\rm hi}=(B_Z+U_Z)/N_Z
\]

를 계산한다.

**우선순위 문턱은 20%**다. 완전 적격 자료에서는 Si III 18키 이상 또는 S III 85키 이상의 확정 반례가 이에 해당한다. 두 배와 20%는 **후속 작업 순서를 정하는 공학적 기준**이며 스펙트럼 유의성의 문턱이 아니다.

| 결과 | 다음 결정표 |
|---|---|
| 어느 이온이든 \(f_Z^{\rm lo}\ge0.20\) | `NEXT_CANDIDATE`: NRB 정상 상태 확인 뒤 충돌 처방 변경을 다음 물리 단의 우선 후보로 제안 |
| 주 표본은 두 이온 모두 \(f_Z^{\rm hi}<0.20\), 연결 집합에서 같은 문턱을 확정 초과 | `CONNECTED_ONLY`: 연결 경로의 변경은 크지만 표적 전이의 율 근거는 약함. 연결 전이 범위를 명시한 후속 설계를 먼저 제안 |
| 주 표본과 연결 집합 모두 문턱 미만임이 확정 | `NOT_PRIORITIZED`: 이번 율 증거만으로 충돌 단을 바로 다음에 우선하지 않음 |
| 결손·경계 미분리 때문에 위 분기가 정해지지 않음 | `UNDECIDED`: 순서 결정의 미확정 사유와 필요한 최소 증거를 명시 |

연결 집합은 §2.2에서 고정한 고유 준위쌍·셸을 이온별로 사용한다. 같은 solve 단위 내부 전이는 별도 계수하고 유효 연결 분모에서 분리한다. 분모와 분리 사유는 율 계산 전에 확정한다.

### 5.4 표·근사·미수록 목록의 소비

next_contract.json은 위 결정과 함께 다음을 반드시 제시한다.

- 확정 반례 중 표 선택 전이와 미수록 근사 전이의 개수·비율.
- 표로 추가되는 선 없는 연결 전이.
- 같은 준위쌍의 중복 선 때문에 생기는 기준선 합계와 표 단일 추가의 차이.
- 미수록 전이별 vR·SET 선택 및 생산 미적용 사유.
- Lumina와 CMFGEN 원본 사이에 남는 §4.4의 차이.

차이가 미수록 근사식에서 주로 나오면 다음 후보를 「충돌표만 활성화」라고 쓰지 않는다. **표·미수록 근사·추가 연결·중복 억제를 포함하는 충돌 처방 변경**으로 범위를 표시한다.

상세평형 또는 C 참조 계산 대조에 실패하면 큰 율 비를 우선순위 근거로 소비하지 않는다.

## §6. 기대치·수치 기준·과학 코드

### 6.1 등록 기대

| 항목 | 등록 기대와 한계 |
|---|---|
| 기존 분기 | 저장 R′의 COLL_FIX 소비 사실이 재확인될 것 |
| OMEGA 표 전이 | 선별 항 억제와 별도 표 추가를 함께 복원해야 함 |
| 미수록 선 | 이온에 표가 있다는 이유만으로 0이 되지 않고 vR·SET 분기를 따를 것 |
| 상세평형 | 적격한 각 충돌 채널과 같은 준위쌍 합계에서 성립할 것 |
| 율 차이 규모 | 사전 수치 기대 없음. 크거나 작다는 결론을 모두 허용 |
| 기존 반례 | 세대 3의 새 충돌율만으로 기존 2준위 배제를 해소했다고 주장하지 않음 |

### 6.2 수치 처리

비교는 **저장 문자열을 binary64로 읽은 입력에 조건부**다. 저장 온도·밀도의 실제 물리적 불확실성까지 포괄하는 오차 상한이라고 부르지 않는다.

- 에너지차는 levels.csv에서 구한다. \(h\nu\)로 임의 치환하지 않는다.
- 이진 표의 값·온도격자와 실제 상수·연산 순서를 보존한다.
- Python과 봉인 C 참조의 양의 율 상대차는 \(10^{-12}\) 이하여야 한다.
- 정확한 영과 분기·색인·적용 상태는 정확히 일치해야 한다.
- 두 배 경계의 \(\log_{10}\) 거리 \(10^{-10}\) 이내는 4로 처리한다. 이 폭은 사후 확대하지 않는다.
- 비의 0 또는 무한대는 명시적 상태로 저장한다. 작은 양수를 넣어 로그를 만들지 않는다.
- JSON의 수치 필드에는 NaN·Infinity를 쓰지 않는다. 값이 정의되지 않으면 null과 상태를 함께 둔다.
- 결손·미적용·부동소수점 underflow를 구별한다.

독립 고정밀 직접시험은 작은 고정 입력에만 사용하며 원식의 정규화와 분기 경계를 확인한다. 실자료 전체의 고정밀 재계산은 추가하지 않는다.

### 6.3 과학 코드

결과 스키마는 `DET_COLL_OMEGA_OFFLINE_RESULT_V1`이다.

| 코드 | 명칭 | 의미 |
|---:|---|---|
| 0 | `SURVIVES` | 적격값이 등록 명제를 만족 |
| 1 | `REFUTED` | 적격값에서 등록 명제 위반이 분리 |
| 4 | `UNRESOLVED` | 경계·영·수치 표현 때문에 미분리 |
| 5 | `NOT_IDENTIFIABLE` | 필요한 입력·권위·정의가 부족 |
| X | 적격성 실패 | 연결·구현·선택·결과 구조가 잘못됨 |

전칭 명제는 확정 반례 1, 필요한 결손 5, 미분리 4, 전건 만족 0의 순서로 집계한다. X가 있는 범위에는 적격한 최종 집합 판정을 부여하지 않는다.

H-SMALL의 반례는 조사 실패가 아니다. 상세평형·봉인 구현 재현 실패는 판독 적격성 문제이므로 수리 대상이다. 순서 결정 문자열과 집행 처분을 과학 코드에 섞지 않는다.

## §7. 최소 변경집합·출력

### 변경집합 — DET-COLL-OMEGA-OFFLINE (표)

새 구현 심볼의 접두사는 `det_coll_omega_offline_`로 한다.

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `scripts/det_coll_omega_offline_readout.py` | 판독기·내장 직접시험 — 신설 | 고정 집합·입력 연결·표 판독·생산 분기 계산·C 참조 연결·직접시험·분포·결정표·신원 기록 |
| `/gpfs/kjhan/lumina/cmfgen_map/coll_omega_20261008/tools/run_det_coll_omega_offline.slurm` | CPU 러너 — 신설 | 환경 초기화·CPU 참조 빌드·selftest·analyze·게이트·시간·자원·rc·결과 보존 |

### 변경집합 끝

기존 판독기·게이트·생산 소스·원자 자료는 수정하지 않는다. 직접시험은 새 판독기의 selftest에 포함한다. 별도 상설 분석기나 시험 스크립트를 추가하지 않는다.

필요한 C 어댑터·발췌 소스·실행 파일·가상 입력은 새 판독기가 실행 결과 경로에 생성하는 산출물이다. 이를 등록 밖 상설 도구로 만들지 않는다.

| 필수 산출 | 내용·소비처 |
|---|---|
| sample.json | 510키, 연결 준위쌍·선·셸, 세대, 고정 분모와 매핑 |
| rates.csv | 입력, 표 수록, tier, 선별·표별 성분, 상·하향 합계, 율 비, 상세평형, 적용·수치 상태 |
| uncovered.csv | 미수록 전이와 미적용 전이의 구분, fallback·사유·연결 대상 |
| result.json | 분포, 검증 결과, H-SMALL 코드, 적격성·결손 |
| next_contract.json | §5의 순서 결정, 표·근사별 근거, 후속 범위와 주장 상한 |
| selftest.json | 정상·음성 대조, 실제 실패 사유·rc, Python/C 대조 |
| manifest.json | 계약·도구·입력·C 발췌·출력 신원, 명령·환경·시간·비용 |

rates.csv는 행 종류를 구분하여 선 성분·표 성분·준위쌍 합계를 한 파일에 보존한다. 합계 행만 분포에 사용한다. uncovered.csv는 같은 판독 결과의 결정용 목록이며 별도 계산 경로를 갖지 않는다.

계약·발주·판정·감사문·OWNER.txt·비용 장부는 집행 문서다.

## §8. 게이트·직접시험

### 8.1 내부 게이트

| 게이트 | 방지할 실패 | 통과 조건 |
|---|---|---|
| G-LINK | 원소·표본·준위·셸·세대·소스 혼동 | §1–§3의 입력과 고정 집합 연결 확인 |
| G-READOUT | 식·보간·분기·중복 계상·영 처리 오류 | §4–§6의 산출 검증과 §8.2 직접시험 충족 |

표 미수록 자체는 게이트 실패가 아니다. 미수록을 표 수록으로 바꾸거나 0으로 은폐하는 것이 실패다.

### 8.2 Python 재구현과 봉인 C 검증

Python이 OMEGA 경로를 재구현하면 **동일 Python 함수의 재호출을 참조 계산으로 사용하지 않는다.**

참조는 봉인 C의 다음 부분을 수정 없이 소비한다.

- 표 로더·준위쌍 매핑.
- ma_ru_upsilon.
- omcm_ex_e1x·omcm_gbar·omcm_fallback·omega_cmfgen_line.
- artis_col_rates의 양의 충돌강도 경로.
- COLL_FIX 계산부.
- 선별 tier 억제와 일반 표 추가의 적용 조건·계산부.

CPU 참조는 작은 어댑터로 필요한 배열·스칼라만 제공한다. 원문 발췌를 사용하면 각 발췌의 원본 파일·시작/끝 표지·내용 해시와 원문 일치를 기록한다. 어댑터가 검증할 식·분기를 다시 작성해서 대신하지 않는다.

실제 선택 입력의 전이별 결과를 일괄 대조한다. 생산 프로그램 초기화·수송·물질 반복·SE solve는 호출하지 않는다.

| 시험 묶음 | 정상 대조 | 주입 결함과 요구 결과 |
|---|---|---|
| 입력 연결 | 정확한 510키·Z·ion0·셸·세대 | Si/S 파일 교환, 준위 색인 1 이동, 반례 삭제, 세대 7을 8로 교환 → G-LINK FAIL |
| 이진 표·로더 | 정상 헤더·격자·레코드와 manifest | 잘린 레코드, 준위 수 불일치, 누락 파일, 모순 중복 → 원인 있는 실패 |
| 표 보간 | 격자점·구간 내부·끝점 처리 | 선형 대신 log-log, 온도 구간 교환, 저장 온도 대신 10400 K → C 대조 FAIL |
| 미수록 분기 | \(f\) 경계 양쪽과 경계값, \(x=1\) 양쪽 | 잘못된 \(f\) 경계, \(g_l\) 누락, 에너지차 거듭제곱 오류 → FAIL |
| 추가·중복 억제 | 표 전이 1회, 미수록 선 fallback, 선 없는 표 전이 추가 | 표를 선마다 중복 추가, tier 1의 선별 0만 최종값으로 사용, 이온 전체 fallback 억제 → FAIL |
| solve 적용 | 서로 다른 단위와 같은 단위 내부 분리 | 같은 단위 내부 표 전이를 유효 연결로 집계, 유효 연결 누락 → FAIL |
| 율·상세평형 | 알려진 상·하향 율과 독립 고정밀 대조 | 통계중량 교환·지수 부호 오류 → 상세평형 FAIL. 두 율에 같은 오계수를 곱함 → 상세평형이 살아도 C 대조 FAIL |
| 영·결손·결정표 | 구조적 영, 결손 5, 경계 4, 정확한 고정 분모 | 결손 0 채움, 분모에 작은 값 추가, 실패 행 삭제, 20% 경계 오분류 → FAIL |
| 보존 | 새 실행 경로 사용 | 기존 결과 경로 재사용 → 쓰기 전 거부 |

주입 결함은 가상 입력 또는 시험 전용 경로에만 적용한다. 봉인 소스·실제 덱·저장 런을 변조하지 않는다.

각 음성 대조는 실제 판독 경로를 통과한 뒤 등록 검사가 실패해야 한다. 주입 플래그를 읽었다는 사실만으로 성공 처리하지 않는다. 최종 코드 신원에 대해 직접시험을 수행하고, 감사 수리로 코드가 바뀌면 해당 시험과 판독을 다시 연결한다.

### 8.3 기계 프리플라이트

기존 도구를 수정 없이 사용한다.

- `scripts/check_prereg_preflight.py`
- `scripts/check_gate_referents.py`
- `scripts/check_doc_canonical.py`
- `scripts/check_dispatch_preflight.py`
- `scripts/check_driver_hygiene.py`
- `scripts/check_gate_roster.py`

계약·발주·러너 신원과 실제 argv·stdout·stderr·rc를 보존한다. 같은 문서 신원에 대한 dispatch 내부 검사를 불필요하게 반복하지 않는다.

경량 사전감사는 개정25-ab에 따라 생략한다. 기계 프리플라이트가 과학식과 판독의 정확성을 보장한다고 표시하지 않는다.

## §9. 환경·비용·재현 명령

### 9.1 환경

직접시험·게이트·CPU 빌드·판독·집계는 **a10 CPU Slurm**에서 수행한다. 로그인 노드에서는 파일 조회·소스 판독만 한다.

- 작업명: `LUMINA_` 접두.
- 제외 노드: `--exclude=syn102`.
- GPU 예약: 0.
- O에 OWNER.txt 유지.
- CPU·메모리·시간을 명시.
- module 초기화 후 Python·컴파일러·의존성 판본 기록.
- 호스트·job ID·스레드 설정·실행 명령·종료 상태 기록.
- 시간 측정은 Bash `time` 또는 확인된 동등 수단 사용.
- 일반 파일 읽기·쓰기와 스트리밍 해시로 구현.
- 재귀적인 전체 트리 탐색·일괄 복사 없이 소비 파일 목록으로 처리.
- 상태 조회는 사용자가 기다리는 동안에만 60초보다 자주 하지 않음.

### 9.2 비용 상한

직전 DET-SEEDMEM-SCOPE의 최종 실행은 Slurm 90초였고, 시험 9.4초·판독 79.2초였다. 수리·개발·재현을 포함한 누적 실행은 828.1초, CPU 예약량은 0.900 core·h였다.

이번에는 물질 배경과 출력 범위를 줄이되 표·분기 C 대조와 감사 수리 여유를 둔다.

| 항목 | 상한 |
|---|---:|
| 동시 계산 | 1작업 |
| 작업당 CPU | 2 |
| 메모리 | 8 GiB |
| 전체 빌드·직접시험·판독 1회 | 300초 |
| 전체 실행 배분 | 초회 300초＋수리·재현 3회×300초 |
| 독립 게이트·개발·보조 확인 | 누적 300초 |
| 합산 실행 경과 | **1500초** |
| CPU 예약량 | **\(5/6\) core·h 이하** |
| 새 도구·시험·결과·증거·로그 합계 | **192 MiB 이하** |
| GPU 예약·생산 실행 | **0 GPU·h, 0회** |

실패·개발·직접시험·수리·재현·기계 게이트를 모두 합산한다. 같은 작업의 러너 시간과 Slurm 경과는 중복 합산하지 않고 큰 값을 비용에 사용한다. 대기·문서 작성·감사 시간은 계산 경과와 구별한다.

선 목록은 한 번 순회하여 필요한 원소·전이만 보존한다. 선마다 덱을 다시 읽지 않는다. 입력 전체를 복사하지 않는다.

자동 반복 제출은 하지 않는다. 남은 예산으로 완료할 수 없으면 실제 지출·미완 항목·추가 필요량을 보고한다. 비용 부족을 과학 반증으로 바꾸지 않는다.

### 9.3 계획 CLI

아래는 인터페이스 등록이며 기안자가 실행한 명령이 아니다.

```bash
P=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
C=/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
D=/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008
S=/gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008
O=/gpfs/kjhan/lumina/cmfgen_map/coll_omega_20261008

python3 -B "$P/scripts/det_coll_omega_offline_readout.py" selftest \
  --production-source "$C/source/src" \
  --out "$O/results/$RUN_ID/selftest"

python3 -B "$P/scripts/det_coll_omega_offline_readout.py" analyze \
  --prereg "$P/docs/PREREG_DET_COLL_OMEGA_OFFLINE_2026-10-08.md" \
  --census "$D/r3_lock_census.csv" \
  --prior-sample "$S/results/r4/analysis/analysis/sample.json" \
  --prior-result "$S/results/r4/analysis/analysis/result.json" \
  --r-run "$C/Rprime/run1" \
  --production-source "$C/source/src" \
  --selftest "$O/results/$RUN_ID/selftest" \
  --out "$O/results/$RUN_ID/analysis"
```

RUN_ID는 미사용 실행 식별자다. 표본·셸·세대·상수·판정 문턱을 바꾸는 분석 옵션은 만들지 않는다.

```bash
python3 -B "$P/scripts/check_prereg_preflight.py" \
  "$P/docs/PREREG_DET_COLL_OMEGA_OFFLINE_2026-10-08.md" --root "$P"

python3 -B "$P/scripts/check_gate_referents.py" \
  "$P/docs/PREREG_DET_COLL_OMEGA_OFFLINE_2026-10-08.md" --root "$P"

python3 -B "$P/scripts/check_doc_canonical.py" \
  "$P/docs/PREREG_DET_COLL_OMEGA_OFFLINE_2026-10-08.md"

python3 -B "$P/scripts/check_dispatch_preflight.py" "$ORDER" --root "$P"
```

ORDER는 이 계약을 소비하는 발주서다. 위 Python 명령도 CPU 할당 안에서 집행한다.

```bash
sbatch \
  --job-name=LUMINA_DET_COLL_OMEGA_OFFLINE \
  --partition=a10 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=8G \
  --gres=gpu:0 \
  --time=00:05:00 \
  --exclude=syn102 \
  "$O/tools/run_det_coll_omega_offline.slurm" "$RUN_ID"
```

재현은 같은 입력·코드 신원으로 새 결과 경로에서 수행한다. 과학 산출을 비교하고 시간·호스트·실행 식별자 같은 집행 metadata의 차이를 분리한다. 재현도 누적 비용에 포함한다.

## §10. 처분·완료 조건·후속 결정

### 10.1 필수 완료 내용

판정문 첫 부분에 다음을 명시한다.

- 생산 변경·GPU 실행은 0.
- 비교는 저장 R′ 상태에 조건부인 충돌율 수준.
- 다음 단 우선순위와 그 표본·기준.
- 표에서 생긴 차이와 미수록 근사에서 생긴 차이.
- 스펙트럼·이온화 효과, 물리적 정답, 생산 승격은 미판정.

필수 내용은 다음과 같다.

1. 510키와 연결 집합의 고정 분모·적격성·결손.
2. 표 수록·미수록·실제 생산 기여의 구분.
3. COLL_FIX/OMEGA 분포와 상세평형.
4. 표 전이의 중복 억제 및 선 없는 표 전이의 처리.
5. line 600693과 line 471957의 명시적 결과. 세대 3 대조의 제한.
6. §5의 다음 결정표와 후보 변경의 정확한 범위.
7. Python/C 대조·음성 대조·입출력 신원.
8. 실제 비용·원본 보존·이탈.
9. 운전자 판정, astra 종료감사, 실행 주장 재현 상태.

큰 차이가 나와야 완료되는 계약은 아니다. 작은 차이·미확정 결과도 등록 기준과 한계를 정확히 소비한 결정표가 있으면 유효한 조사 결과다.

### 10.2 집행 처분

| 처분 | 기준 |
|---|---|
| `PASS_LIMITED` | 등록 비교·직접시험·한계·다음 결정표를 적격하게 완료 |
| `REWORK` | 입력·식·분기·중복·시험·집계·귀속·비용 기록의 계약 위반 |
| `INCOMPLETE` | 필수 입력·참조 계산 또는 등록 자원으로 최소 산출을 완료하지 못함 |

전체 처분의 상한은 `PASS_LIMITED`다. 율 차이의 크기를 집행 성공으로 바꾸지 않는다. 정당한 미수록·식별 한계를 기록한 것과 판독기 구현이 미완인 것을 구별한다.

### 10.3 후속 결정의 경계

이번 단은 next_contract.json을 사용자와 운전자에게 넘기고 끝난다.

`NEXT_CANDIDATE`도 다음 런을 승인하지 않는다. 후속 계약은 다음을 별도로 확정해야 한다.

- NRB 기준 상태와 물질 상태의 권위.
- 표만 바꾸는지, 미수록 근사·추가 연결까지 바꾸는지.
- 전역 OMEGA 스위치의 다른 이온·소비처 영향.
- 유지할 물리와 변경할 물리 하나.
- 실제 이온화·스펙트럼 측정량과 수용 기준.
- 생산 변경집합·GPU 비용·실행 횟수.

등록된 CPU 수리 여유는 생산 변경이나 추가 물리 런으로 전용하지 않는다.

## §11. 집행 기록·분장

| 업무 | 담당 | 규칙 |
|---|---|---|
| 기안·개정 | astra | 모델 ID·사전 노출 기록 |
| 경량 사전감사 | 생략 | 개정25-ab: GPU 0·생산 변경 0, 독립 감사 통과 아님 |
| 구현·직접시험·기계 게이트·실행·판독·판정 | 운전자 | 구현·판정 겸임 명시 |
| 코드 감사 | 해제 | 별도 정례 감사 추가 없음 |
| 종료감사 | astra | 계약 저자라는 이해관계 명시 |
| 실행 주장 재현 | 운전자 | 입력·코드·명령·출력 신원과 재현 결과 기록 |
| 범위·비용 확대·생산 변경·후속 런 | 사용자 | 이번 계약에서 자동 발생하지 않음 |

종료감사 수리는 등록 변경집합과 누적 비용 안에서 수행한다. 현행 폐합 규약의 지적 처리 순서를 따르되 미달 목표나 남은 폐합 차단을 통과로 바꾸지 않는다.

| 기안 시 항목 | 상태 |
|---|---|
| 공유 문맥 전체 판독 | 완료 |
| 프로젝트 루트·HEAD 확인 | 완료 |
| 사용자 결정·직전 계약·판정·result.json 판독 | 완료 |
| 관련 Python 함수·봉인 충돌 소스·CMFGEN 관계 판독 | 완료 |
| 덱 manifest·참고 CSV·저장 상태 metadata 판독 | 완료 |
| 표본 추출·해시 재계산·율 계산 | 미실행 |
| 파일화·기계 게이트 | 운전자 집행 전 |
| 구현·직접시험·CPU 제출 | 미실행 |
| 판정·종료감사·재현 | 미실행 |

기안자는 파일 수정, 계산 실행, 시험, 게이트 또는 작업 제출을 수행하지 않았다. 향후 결과를 예상 PASS로 기입하지 않는다.

## §12. 기계 선언

선언은 변경집합, 실제 파일 참조와 §6.3의 코드 분할을 검사한다. 참조 대상은 모두 파일이다. 과학식·채널 선택·표 매핑·세대·우선순위 판정의 정확성은 직접시험과 종료감사에서 확인한다.

`v`는 과학 명제 하나의 코드다. 유효 코드는 0·1·4·5이며 결손·복수 코드·다른 값은 X다. 코드 5와 결과 코드 자체의 부재를 구별한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-COLL-OMEGA-OFFLINE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_coll_omega_offline_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/coll_omega_20261008/tools/run_det_coll_omega_offline\\.slurm)",
    "symbol": "det_coll_omega_offline_",
    "roots": [
      "scripts/det_coll_omega_offline_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/coll_omega_20261008/tools/run_det_coll_omega_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_coll_omega_offline_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/coll_omega_20261008/tools/run_det_coll_omega_offline.slurm"
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
      [9.0, 9.0]
    ],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,4,5) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",
      "c4": "sum(1 for x in v if x == 4)",
      "c5": "sum(1 for x in v if x == 5)"
    },
    "rules": [
      {"name": "X", "predicate": "not valid"},
      {"name": "SURVIVES", "predicate": "valid and c0 == 1"},
      {"name": "REFUTED", "predicate": "valid and c1 == 1"},
      {"name": "UNRESOLVED", "predicate": "valid and c4 == 1"},
      {"name": "NOT_IDENTIFIABLE", "predicate": "valid and c5 == 1"}
    ],
    "adversarial_fixtures": [
      {"name": "survives", "mix": [[0,1]]},
      {"name": "refuted", "mix": [[1,1]]},
      {"name": "unresolved", "mix": [[4,1]]},
      {"name": "not-identifiable", "mix": [[5,1]]},
      {"name": "invalid-two", "mix": [[2,1]]},
      {"name": "invalid-three", "mix": [[3,1]]},
      {"name": "invalid-nine", "mix": [[9,1]]},
      {"name": "missing-code", "mix": []},
      {"name": "multiple-codes", "mix": [[0,1],[1,1]]}
    ]
  },
  "references": [
    {
      "path": "CLAUDE.md"
    },
    {
      "path": "docs/PREREG_DET_SEEDMEM_SCOPE_2026-10-08.md"
    },
    {
      "path": "docs/VERDICT_DET_SEEDMEM_SCOPE_2026-10-08.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/coll_omega_20261008/evidence/USER_DECISION_2026-10-08.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008/r3_lock_census.csv",
      "flags_existing": ["line_id", "shell", "beta2_E"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008/results/r4/analysis/analysis/sample.json"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008/results/r4/analysis/analysis/result.json",
      "flags_existing": ["collision_authority"]
    },
    {
      "path": "scripts/det_siiii_rates_readout.py",
      "flags_existing": ["det_siiii_rates_coll"]
    },
    {
      "path": "scripts/det_seedmem_scope_readout.py",
      "flags_existing": ["det_seedmem_scope_coll_authority"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c",
      "flags_existing": [
        "ma_ru_upsilon",
        "omcm_ex_e1x",
        "omcm_gbar",
        "omcm_fallback",
        "omega_cmfgen_line",
        "omega_cmfgen_arm",
        "artis_col_rates",
        "nlte_assemble_rate_matrix_impl"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c",
      "flags_existing": [
        "load_ion_coldata",
        "load_ion_coldata_manifest"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cuda.cu",
      "flags_existing": [
        "load_ion_coldata_manifest",
        "omega_cmfgen_arm"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina.h",
      "flags_existing": [
        "H_PLANCK",
        "K_BOLTZMANN",
        "EV_TO_ERG"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1/STAGE_IDENTITY.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1/input/run_build_identity.json"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1/input/resolved_lumina.exports",
      "flags_existing": [
        "LUMINA_NLTE_COLL_FIX",
        "LUMINA_NLTE_COLL_FLOOR",
        "LUMINA_NLTE_ASSEMBLE_GPU",
        "LUMINA_SUPER_CUTOFF"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1/input/model/coldata_cmfgen_manifest.csv",
      "flags_existing": [
        "ige_col_14_2_cmfgen.bin",
        "ige_col_16_2_cmfgen.bin"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1/input/model/siii_terms_col_cmfgen.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1/input/model/siii_col_cmfgen.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1/work/physics_comparison/physics_DET_iter0004.manifest.json",
      "flags_existing": [
        "population_generation",
        "te_generation",
        "shell_file"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1/work/physics_comparison/physics_DET_iter0000.manifest.json"
    },
    {
      "path": "/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/omega_gen_v3.f",
      "flags_existing": [
        "OMEGA_SET",
        "SAME_N",
        "EX_E1X_FUN"
      ]
    },
    {
      "path": "/gpfs/kjhan/cmfgen_src/cur_cmf/subs/ex_e1x_fun.f",
      "flags_existing": ["EX_E1X_FUN"]
    },
    {
      "path": "scripts/check_prereg_preflight.py"
    },
    {
      "path": "scripts/check_gate_referents.py"
    },
    {
      "path": "scripts/check_doc_canonical.py"
    },
    {
      "path": "scripts/check_dispatch_preflight.py"
    },
    {
      "path": "scripts/check_driver_hygiene.py"
    },
    {
      "path": "scripts/check_gate_roster.py"
    }
  ]
}
```
