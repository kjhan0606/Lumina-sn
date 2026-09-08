# 사전등록 기안 — CMFGEN 어긋남 지도: 쌍둥이 T_e 레인
**단 식별자: DET-CMFGEN-TWIN-MAP · 2026-09-08**

저자 = astra. 상태 = **기안 / INPUT_PENDING / AUDIT_PENDING**. 실행 승인이 아니다.

이번 기안에서는 지정 문서 판독, 소스 조회, 기존 파일의 메타데이터 조회와 `python3 -B`를 이용한 소형 표 판독만 수행했다. 파일 생성·수정·빌드·GPU·Slurm·네트워크·프리플라이트 실행은 하지 않았다.

## 1. 계약 — 어느 칸을 어느 크기로 채우는가

이 단의 산출은 **공개 CMFGEN 원자료 속도점에 직접 대응한 Lumina 이산 상태의 어긋남 지도 1건**이다.

- **양:** 원소별 단계 분율 `f_q`, 전자밀도 `n_e`. Saha 단계 밀도와 NLTE 준위 합은 서로 다른 권위로 구분한다.
- **위치:** O-0 §2의 봉인 base 격자에서, 각 셸 내부에 있는 CMFGEN 원자료 행 중 속도가 가장 작은 행 하나. 표본 선정에는 T·분율·전자밀도를 사용하지 않는다.
- **크기:** 각 칸에 분율의 부호 있는 차이와 전자밀도의 비·dex 차이를 기록한다. 양수가 아닌 값에 로그를 적용하거나 작은 수를 더하지 않는다.
- **필수 표본:** 위 규칙으로 선정한 셸별 표본에서 Saha 권위의 원소·단계별 분율과 공개 `n_e`를 기록한다. NLTE 권위의 분율은 실제 계산·공개된 준위 집합에 한하여 별도 기록한다.
- **주장:** 이번 단은 어긋남의 위치와 크기를 처음 채우는 측정 계약이다. 물리 수리로 CMFGEN 오차가 감소한다고 약속하지 않는다. T 조건 변경 뒤 `n_e`의 원자료 대비 차이가 감소하는지는 §6의 반증 가능한 가설로 판정한다.

O-0의 자료·모델·시각·격자·레인 배선 수치는 **`docs/O0_CMFGEN_DIVERGENCE_MAP_SURVEY_2026-09-08.md` §1–5를 인용**한다. 그 문서의 §6에서 미확인으로 남긴 작동 여부는 승계하여 미확인으로 둔다.

### Q-GOAL — 계약 검사보다 먼저 답할 고정질문

> 이 단의 구현 방향이 프로젝트 최종목적인 「Monte-Carlo 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현한다」에 부합하는가?
>
> ① 이 단이 CMFGEN 어긋남 지도의 어느 칸을 줄이는가—양·위치·크기. 못 대면 그렇게 답하라.  
> ② 산출이 물리 측정인가, 계측·게이트·문서 정비인가.  
> ③ 더 가까운 대안 표적이 지금 열려 있는가. 있으면 지목·비교하라.  
> ④ 이 단을 건너뛰면 최종목적 경로가 무엇을 잃는가.
>
> 특히 **미측정 칸을 채우는 것과 오차를 줄이는 것을 구분**하여 판단하라. 부합하지 않으면 «폐합 차단»으로 분류하라. 본 기안에는 Q-GOAL 판정을 미리 부여하지 않는다.

## 2. 입력·봉인·재사용 범위

### 2.1 고정 입력

`BASE`는 다음 절대경로다.

```text
/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260906T234024Z_dc1_ne_closure_on
```

참조 자료는 다음 로컬 공개 자료만 사용한다.

```text
data/standart_data1/toy06/phys_toy06_cmfgen.txt
data/standart_data1/toy06/ionfrac_ca_toy06_cmfgen.txt
data/standart_data1/toy06/ionfrac_co_toy06_cmfgen.txt
data/standart_data1/toy06/ionfrac_fe_toy06_cmfgen.txt
data/standart_data1/toy06/ionfrac_ni_toy06_cmfgen.txt
data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt
data/standart_data1/toy06/ionfrac_s_toy06_cmfgen.txt
```

모델·시각 선택은 O-0 §1–2를 따른다. 자체 CMFGEN 런은 `docs/VERIFICATION_REGISTERS.md`의 오라클 부적격 기록에 따라 사용하지 않는다.

`BASE/input/current_source_identity.json`과 `BASE/RUN_FOOTER.txt`에 기록된 바이너리 지문은 다음과 같다.

```text
ca721b07cb4fef632b1a0a0b2d3c7b11cdaac0e95f74e6cc56b131a7c5449059
```

이는 **[저자 실행: 기록 판독]**이며 바이너리 전체 해시의 독립 재계산은 아니다. 운전자는 사용 직전 실물 해시를 재계산한다. 현재 작업 트리에는 미커밋 변경이 있으므로 현재 HEAD나 작업 트리 바이너리를 base의 대용으로 사용하지 않는다.

### 2.2 보존 규칙

새 증거 루트 `EVID`와 새 실행 루트 `RUN`의 절대경로는 운전자가 실행 전에 봉인한다. 둘 다 `BASE`와 그 하위에 둘 수 없다.

- base 내부에 출력·임시파일·캐시·해시 결과를 쓰지 않는다.
- 실행 입력은 독립 복사한다. base를 가리키는 쓰기 가능한 하드링크·심볼릭링크를 허용하지 않는다.
- 기존 실행 루트를 재사용하거나 덮어쓰지 않는다.
- 실행 전후 base 봉인을 확인하고, 해시 결과는 `EVID`에만 기록한다.
- rollback은 새 변경집합과 새 레인의 비채택이다. 기존 증거를 삭제하거나 원상복원 명목으로 다시 쓰지 않는다.

### 2.3 직전 단 자산

`docs/PREREG_NE_THRESH_2026-09-08.md`의 집행 기록과 주차 처분을 따른다.

- O-0·기존 입력 identity·봉인 보호 방식은 출처를 유지하여 참조할 수 있다.
- 미커밋 NE-THRESH 계측·탐침·replay·R-ACQ 모드는 이번 바이너리에 편입하지 않는다.
- sol selftest 보고를 astra 재검수 완료로 승격하지 않는다.
- 직전 단의 GPU 예산·허용오차·오프라인 oracle 자격을 승계하지 않는다.

## 3. 보간 없는 표본 정의와 비교 한계

### 3.1 원자료 행 직접 선택

셸 `s`에 대해 다음 집합을 만든다.

```text
A_s = {j : v_inner[s] ≤ CMFGEN.vel_mid[j] < v_outer[s]}
j_s = A_s에서 vel_mid가 가장 작은 원자료 행
```

단위 변환만 허용한다. `A_s`가 비면 `INPUT_PENDING`이며 가까운 다른 셸의 값으로 메우지 않는다.

`phys`와 모든 `ionfrac` 파일에서 시각·속도 키가 일치하는 같은 행을 사용한다. 행 번호만으로 조인하지 않는다. 중복 속도·시각 불일치·원소 파일 결손은 실패다.

**[저자 실행: 소형 원자료 판독]** 현재 입력에는 이 규칙으로 모든 base 셸에 표본이 존재했다. 선정된 블록 내 0기반 행 인덱스는 다음과 같다.

```text
21,23,25,27,28,30,31,33,36,39,
43,46,50,52,54,56,58,60,61,63,
64,65,66,67,68,69,70,71,72,73,
74,75,76,77,78,79,80,81,82,83,
84,85,87,88,89,90,91,92,93,94
```

운전자는 이 목록을 독립 재현하고 원문 행·파일 해시와 함께 `anchors.csv`에 봉인한다.

### 3.2 T 입력의 의미

각 셸의 입력 T는 **선정된 원자료 행의 T를 그대로 복사**한다.

이는 CMFGEN T 곡선을 보간하거나 셸 평균 T를 추정하는 절차가 아니다. 선정점에서 비교할 새 이산 고정-T 입력을 정의한다. Lumina 셸 값을 그 셸 내부 표본점에서 평가한다는 이산화 규약도 함께 기록한다.

다음은 금지한다.

- 다른 CMFGEN 속도점으로 결과를 확장하기
- 미선정 행에 같은 T가 맞는다고 주장하기
- 셸 평균·연속 프로파일의 일치를 주장하기
- 빈 셸을 최근접값·평균·외삽으로 채우기
- 지도 그림에서 평활화로 미측정 칸을 채우기

따라서 산출 명칭은 **“셸별 원자료 표본점 지도”**이며 전 속도 연속지도가 아니다.

### 3.3 조건 불일치 장부

| 축 | 계약상 처리 | 무효화되는 주장·칸 |
|---|---|---|
| 시각·모델 | O-0 §1–2의 identity를 실물 재확인 | 불일치하면 해당 비교 전부 무효 |
| 격자 | 원자료 점 대 Lumina 셸 이산값 | 연속 곡선·셸 평균·격자 수렴 오차 주장 불가 |
| T | 선정점의 입력 T와 실제 공개 T를 확인 | 불일치한 표본의 쌍둥이-T 비교 무효 |
| 밀도·조성 | base를 유지하고 차이를 별도 기록 | 잔차를 원자율·배선 오류 하나에 귀속할 수 없음 |
| 복사장 | Lumina 자체 장 사용 | T+J 쌍둥이·율 단독 검증으로 부를 수 없음 |
| 반복수 | base의 봉인 반복수를 유지 | 고정점·반복 수렴·정상해 정확도 주장 불가 |
| 정상상태 가정 | 상위 설계 §4의 시간의존 항 caveat 승계 | 잔차를 모두 Lumina 구현 오차로 해석할 수 없음 |
| population 권위 | Saha와 NLTE를 분리 | 둘을 합쳐 단일 NLTE 분율이라고 부를 수 없음 |
| 수송 레인 | base 레인을 유지 | MC 스펙트럼 재현을 직접 입증하지 않음 |

상위 설계 `docs/DESIGN_DC_DD_CMFGEN_CONTRAST_2026-09-05.md` §4의 외곽 구분을 보고용 층화로 유지한다. 해당 구분은 시간의존 효과의 검출 문턱이나 외곽 자료 자동 폐기 조건이 아니다.

## 4. 측정량과 출력 계약

### 4.1 단계 분율

Saha 권위:

\[
f^{S}_{Zq,s}=\frac{n^{S}_{Zq,s}}{N_{Z,s}}.
\]

NLTE 권위:

\[
f^{N}_{Zq,s}=\frac{\sum_{\ell\in(Z,q)}n_{\ell,s}}{N_{Z,s}}.
\]

`N_Z`는 해당 실행의 원소 총수밀도다. 모델의 밀도·조성·원자질량 규약에서 독립 산출하고 생산 단계 총량과 대조한다. 원자질량을 임의의 정수 질량수로 대신하지 않는다.

- 참조가 제공하는 단계만으로 분모를 재정규화하지 않는다.
- NLTE 단계가 없으면 `NOT_SOLVED`; 있어야 할 덤프가 없으면 `INPUT_PENDING`.
- NLTE 결손을 Saha·LTE 분배·S-역산·관측 준위 외삽으로 메우지 않는다.
- 실제 계산한 NLTE 단계 목록은 코드·덱·실효 구성에서 실행 전에 봉인한다.
- 정확한 0은 보존한다. 음수·비유한 값은 교정하지 않고 실패 증거로 보존한다.

비교량:

\[
\Delta f^a_{Zq,s}=f^a_{Zq,s}-f^C_{Zq,j_s},
\qquad a\in\{S,N\}.
\]

`Δf`와 `100Δf` percentage point를 기본 크기로 기록한다. 양쪽 모두 양수인 칸에만 `log10(f^a/f^C)`를 추가한다.

### 4.2 전자밀도

최종 공개 상태의 `n_e`를 사용한다.

\[
r_{e,s}=n^L_{e,s}/n^C_{e,j_s},\qquad
d_{e,s}=\log_{10}r_{e,s}.
\]

원값·비·부호 있는 dex·절댓값·표본 위치를 모두 기록한다. 전하 합으로 계산한 수를 실제 공개 `n_e`의 대용으로 쓰지 않는다.

### 4.3 기록자의 세대 정합

O-0 §5는 배선 실존의 근거이며 다음을 증명하지 않는다.

- 반복 번호가 없는 최종 파일의 정확한 공개 세대
- NLTE 덤프 호출 카운터와 수송 반복 번호의 일치
- private trial과 최종 commit의 일치
- 동일 세대의 T·`n_e`·Saha·NLTE 동시 관측

따라서 오프라인에서 **정확한 base 바이너리의 소스 계보**를 대상으로 호출 순서·공개 시점·포인터 복원·출력 도달성을 확인한다. 후보 계산의 덤프를 최종 공개 상태라고 읽지 않는다.

기존 출력으로 이 연결을 입증할 수 없으면 `INPUT_PENDING`이다. GPU 획득런이나 추가 반복으로 조사하지 않는다. 새 생산 계측이 필요하면 변경 범위를 계약 개정으로 먼저 등록한다.

### 4.4 필수 산출물

`EVID` 아래 다음 상대경로를 고정한다.

| 경로 | 내용 |
|---|---|
| `inputs/identity.json` | 원본·binary·source·실효 exports·도구·계약 identity |
| `inputs/anchors.csv` | 셸 경계·원자료 키·원문 행·선정 T |
| `inputs/fixed_te_profile.txt` | 선정 원자료 T의 직접 복사 |
| `offline/base_map.csv` | base에 먼저 적용한 지도와 무효 사유 |
| `offline/writer_contract.json` | 기록자·호출·공개 세대·필드·권위 대응 |
| `offline/expectations.json` | §6 규칙·base 원값·표현 정밀도·해시 |
| `offline/resource_card.json` | 비용 근거·저장 산식·실측·미확인 항목 |
| `offline/selftest.json` | 양성 및 결함 주입별 결과 |
| `run/output_manifest.json` | 실제 산출 경로·세대·파일 identity |
| `result/divergence_map.csv` | 원값·차이·권위·유효성·条件差 |
| `result/summary.json` | 극값 위치·분포·가설 판정·결손 목록 |
| `result/gates.json` | 등록 게이트 전건의 실제 결과 |

지도에서 결손·미계산·T 불일치·세대 불명은 서로 다른 상태로 표시한다.

## 5. 잣대를 base에 먼저 적용한 결과

**[저자 실행]** §3의 표본 선택과 §4.2의 식을 `BASE/work/lumina_plasma_state.csv`에 적용했다.

| 항목 | 현재 결과 | 해석 |
|---|---:|---|
| 셸별 원자료 표본 존재 | 전 셸 존재 | 좌표 선택 가능 |
| base 공개 T와 선정 원자료 T의 일치 | 일치 표본 없음 | base의 쌍둥이-T 자격 불충족 |
| `median(abs(d_e))` | `0.038859615680874218 dex` | T 조건이 다른 원자료 대비 차이 |
| `max(abs(d_e))` | `0.21875037534438782 dex`, shell 1 | 같은 제한을 갖는 극값 |
| Saha·NLTE 전체 분율 지도 | `INPUT_PENDING` | base work에 필요한 전체 덤프 부재 |

이 값들은 **이번 표본 규칙으로 새로 판독한 수**다. O-0의 기존 수치를 재주장한 것이 아니다.

base의 전자밀도 차이를 “정확도 오차”로 승격하지 않는다. 기존 분율 부재를 직전 설계의 근사 분율로 채우지 않는다.

운전자는 같은 식을 독립 재현하고 원자료의 출력 자릿수에 따른 표현 불확실성 구간을 함께 산출한다. 이 기안의 소수 출력은 그 구간보다 정밀한 물리 측정을 뜻하지 않는다.

## 6. 사전 기대치와 작동 확인

### 6.1 런에서만 판정할 물리 가설

**E-NE: 선정 표본 전체에서 쌍둥이-T 레인의 전자밀도 차이 중앙값이 base보다 작다.**

\[
M_{\rm twin}=\operatorname{median}_s|d^{\rm twin}_{e,s}|,
\qquad M_{\rm base}=\operatorname{median}_s|d^{\rm base}_{e,s}|.
\]

판정은 원자료와 출력 형식의 자릿수에서 유도한 표현 구간을 사용한다.

- `upper(M_twin) < lower(M_base)`이면 **감소 관측**.
- `lower(M_twin) > upper(M_base)`이면 **증가 관측**.
- 나머지는 **표현 정밀도 내 미분리**.

자의적인 epsilon·floor·맞춤 계수는 사용하지 않는다. 구간 계산법은 합성 수와 독립 산술로 오프라인 검증한 뒤 실행 전에 봉인한다.

**코드·구성에서 강제되는가? 아니오.** T 입력은 강제하지만 CMFGEN `n_e`를 입력하거나 맞추지 않는다. 복사장·단계 분배·밀도 차이 때문에 증가할 수도 있다.

가설의 근거는 O-0 §4와 상위 설계 §2의 T 교란 문제다. 그 근거는 감소의 정리나 감소 배수 예측이 아니다. 결과가 증가여도 지도 측정 자체를 실패로 바꾸지 않는다.

**base 선적용:** base를 자기 자신과 비교하면 구간이 겹치므로 ‘감소’ 조건을 충족하지 않는다. 이 자기비교는 작동 확인이며 물리 기대치가 아니다.

### 6.2 분율의 기대치

실측 base 분율과 독립 예측이 없으므로 **분율 개선 방향·배수·허용오차는 등록하지 않는다**. 이번 런에서 얻을 각 `Δf`가 측정 대상이다.

“Fe IV가 지배할 것”, “분율이 일정 배수 개선될 것”, “Saha와 NLTE가 같을 것”을 기대치로 만들지 않는다. 지도를 얻은 뒤 수리 단의 표적을 정하는 것은 후속 결정이다.

### 6.3 작동 확인으로만 등록하는 항목

| 항목 | 강제 여부 | 분류 |
|---|---|---|
| 선정 원자료 T를 입력 파일에 복사 | 생성기에서 강제 | 작동 확인 |
| 공개 T가 등록 T와 일치 | 고정-T 구현의 계약 | 작동 확인 |
| 지정 표본·단계의 행 피복 | 출력·판독 계약 | 작동 확인 |
| 분율 산식·단계 합·원소 총량 연결 | 정의·보존 관계 | 작동 확인 |
| 반복수·바이너리·실효 exports 일치 | 구성에서 강제 | 작동 확인 |
| CMFGEN 대비 분율·`n_e` 차이 | 강제되지 않음 | 물리 측정 |
| E-NE 감소 여부 | 강제되지 않음 | 반증 가능한 물리 가설 |

## 7. offline-first 집행 순서

### O-1. 기전 오프라인 특정

T를 읽는 위치, 후보 물질 계산, 최종 공개, Saha·NLTE·`n_e` 기록자까지 연결한다. 목표는 **T 불일치라는 비교 교란을 제거할 수 있는지**의 확인이다. 이 단계에서 잔여 어긋남의 물리 원인을 확정하지 않는다.

추가로 확인할 사항:

- 진단용 균일 T 키와 고정 프로파일의 상호작용
- 초기 상태와 후속 공개 상태의 T 적용 시점
- strict env가 덤프 키를 수용하는지
- 기존 stager의 균일 T·진단 비교 전제가 새 레인을 거부하거나 잘못 인증하는지
- `n_e`와 population의 동일 공개 세대 여부

### O-2. 수리안 오프라인 검증

이 단의 수리안은 **입력 T 대응과 판독의 비교 자격을 바로잡는 것**이다. Saha 반복기·율·NLTE 방정식은 변경하지 않는다.

sol은 원자료 직접 선택, 프로파일 생성, 세대 대응, 지도 산술, 실행 입력 delta, 결함 주입 selftest를 구현한다. 기존 고정-T loader의 실제 경로를 사용한 CPU 시험으로 입력 소비를 확인한다.

기존 출력·기존 바이너리로 계약을 만족시킬 수 없으면 여기서 중단한다. 자료획득 GPU 런을 추가하지 않는다.

### O-3. 기대치·자원·입력 봉인

다음을 모두 확정한 뒤 운전자가 계약 1건의 커밋을 만든다.

- 표본·원자료·T 프로파일·실효 exports
- source·binary identity
- 기록자의 공개 세대 계약
- §6 표현 구간과 가설 판정식
- base 선적용 및 음성대조 결과
- 자원 자격과 신규 출력량
- 사전감사·코드 검수 결과

계약 개정은 `OLD:`/`NEW:` 쌍으로 남긴다. 결과를 본 뒤 표본·분모·문턱·허용오차를 바꾸지 않는다.

### R-1. 판정런 1회

선행조건 충족 후 새 루트에서 한 번만 실행한다.

물리 입력 변경은 등록 T 프로파일이다. 추가 변경은 기존 덤프 활성화와 경로 재배치에 한정하며 전체 실효 exports delta를 봉인한다. 원자 덱·밀도·조성·반복수·수송 설정·NE-closure 상태를 임의로 바꾸지 않는다.

시간 초과·중도 실패·필수 덤프 결손은 그 상태로 보존한다. 자동 재런·관찰런·OFF 인증 GPU 런·반복수 연장은 없다.

## 8. 게이트와 음성대조

아래 게이트는 모두 **미실행**이다. 각 게이트는 정상 fixture의 수용과 해당 결함 주입의 FAIL을 실제로 시연한 뒤에만 사용 자격을 얻는다. 단순히 예상 rc를 출력하는 selftest는 인정하지 않는다.

### 게이트 명부 (표)

| ID | 조건 | 음성대조 |
|---|---|---|
| G-ID | 원자료·base·source·binary·실효 exports·새 경로 identity 일치 | 원자료 byte 변조, stale exports 해시, base 내부 목적지 또는 링크 주입 |
| G-COORD | §3의 직접 표본 선택·시각·속도 키 조인·피복 일치 | 행 하나 삭제, 속도 단위 오독, 다른 시각 블록, 셸 밖 표본 주입 |
| G-TE | 원자료 T 복사와 실제 loader 소비·공개 T 연결 | 셸 누락·중복, 두 셸 T 교환, 균일 T 덮어쓰기 |
| G-EPOCH | T·`n_e`·Saha·NLTE의 기록자·공개 세대 대응 입증 | private trial 또는 이전 세대 덤프를 최종 상태로 교체 |
| G-POP | 단계 라벨·준위 집합·원소 분모·출력 피복·유한성 정합 | II/III 라벨 교환, 준위 중복·누락, 잘못된 원소 분모, NaN 주입 |
| G-METRIC | §4 산식과 표현 구간, §6 분기 재현 | 비의 분자·분모 교환, 한 칸 수치 변조, 0을 작은 양수로 치환 |
| G-RUN | 등록 1회·반복수·입력 delta·필수 출력·종단 상태·비용 상한 준수 | 반복수 변경, 덤프 결손, timeout 또는 추가 실행 기록 주입 |
| G-PRESERVE | 실행 전후 base 봉인 불변, 출력은 새 루트에만 존재 | 사설 base fixture의 파일 교체·부모 경로 우회 주입 |

### 게이트 명부 끝

G-TE는 T의 물리적 정답성을 검사하지 않는다. G-POP도 코드가 보존식을 강제한다는 이유로 CMFGEN 일치를 인증하지 않는다.

기존 고정-T 범위 가드는 이 단에서 확대하거나 강화하지 않는다. 정확해가 위반 가능한 새로운 clamp·floor·cap·jitter·유사 가드를 만들지 않는다.

사전등록·발주·문서 정본·운전자 위생·게이트 참조·명부 검사는 기존 기계를 사용하고 내장 selftest 결과를 함께 기록한다. 이 기안에는 실행하지 않은 검사 PASS를 적지 않는다.

## 9. 자원과 비용 자격

### 9.1 확인한 근거

- 기존 판정문 `docs/VERDICT_DC1_NECLOSE_R2_2026-09-08.md`는 base 작업의 경과시간을 기록한다. 이는 **문서 인용**이며 이번 세션의 Slurm 재조회가 아니다.
- `BASE/submit_superlevel_b.sh`에서 기존 GPU·CPU·메모리 할당을 판독했다. 그 walltime 요청을 실사용량으로 읽지 않는다.
- **[저자 실행: 파일 메타데이터 합계]** base 입력의 일반파일 총량은 `5,393,486,835 byte`, work 일반파일 총량은 `33,212,368 byte`였다. 두 하위 트리에서 파일 심볼릭링크는 발견되지 않았다.
- **[저자 실행]** 이번 공개 phys·ionfrac 자료의 합계는 `3,552,686 byte`였다.
- 새 T에서의 실행시간·peak 메모리와 전체 NLTE 덤프량은 **미측정**이다.

### 9.2 제안 지출 상한

| 자원 | 제안 상한 | 근거와 한계 |
|---|---:|---|
| GPU | 판정런 1회, GPU 2개 × 최대 4시간 = **8 GPU-hour** | 기존 동일 반복 규모의 약 3시간 실적을 참고한 지출 한도. 새 T의 완주시간 예측은 아님 |
| 런 CPU | 32 CPU × 최대 4시간 = **128 할당 CPU-hour** | base의 CPU 할당 규모. 실제 TotalCPU와 구분 |
| 오프라인 CPU | 단일 CPU 누적 **1 CPU-hour** | 소형 표·기록자 시험·판독 검증에만 사용. 대형 율 oracle·전체 수송 replay는 범위 밖 |
| 저장 | 신규 총량 **16 GiB 상한** | 기존 입력·work 크기는 확인. 추가 덤프 상한은 실행 전 산출 필요 |
| 런 메모리 | 기존 할당 규모를 출발점으로 검토 | 새 레인 peak 보장 아님. 자격 자료 필요 |

GPU는 현행 규약의 H200 우선·H100 차순을 따른다. 기존 A100 실적을 동일 성능 보장으로 전용하지 않는다.

**자원 상태는 INPUT_PENDING이다.** 실행 전에 운전자는 다음을 봉인한다.

1. 기존 적합 작업의 GPU 종류·개수·경과시간·CPU 할당·메모리 기록.
2. 실제 덱의 준위 수, 기록 호출 상한, 행당 byte 상한으로 계산한 덤프량.
3. 소형 오프라인 시험의 시간·메모리 실측.
4. 새 루트의 가용 저장공간과 위 지출 한도 안에서의 실행 수용 여부.

근거가 부족하면 제출하지 않는다. 상한이 있다는 이유로 완료 가능성이 입증됐다고 쓰지 않는다. 초과 시 자동 증액·재런하지 않는다.

## 10. 변경집합과 재현 명령

생산 C/CUDA 변경·새 바이너리 빌드는 현재 변경집합에 없다. 그것이 필요하다는 발견은 범위 확대 사유이며 실행 전 계약 개정 대상이다.

### 변경집합 — CMFGEN_TWIN_MAP (표)

| 경로 | 담당 | 내용 |
|---|---|---|
| `scripts/cmfgen_twin_map.py` | sol | 원자료·base 판독, 직접 표본, 분율·전자밀도 지도, 구간 산술, 내장 selftest |
| `scripts/stage_cmfgen_twin_map.sh` | sol | 독립 staging, 전체 delta 검증, 출력·비용·보존 검사, 내장 selftest |

### 변경집합 끝

두 파일은 신설 예정이다. 기존 stager와 주차된 NE-THRESH 변경을 수정하지 않는다.

다음은 **구현할 CLI 계약이며 현재 실행 가능한 도구라고 주장하지 않는다.** `EVID`·`RUN`·사전등록 파일의 절대경로가 봉인되면 운전자가 치환한 명령 전문을 보존한다.

```bash
python3 -B scripts/cmfgen_twin_map.py --selftest
bash scripts/stage_cmfgen_twin_map.sh --selftest

python3 -B scripts/cmfgen_twin_map.py prepare \
  --base "$BASE" \
  --reference-dir data/standart_data1/toy06 \
  --evidence "$EVID"

python3 -B scripts/cmfgen_twin_map.py analyze \
  --contract "$EVID/offline/expectations.json" \
  --run "$RUN" \
  --evidence "$EVID"
```

`prepare`는 기존 결손을 `INPUT_PENDING`으로 출력하고 계산런을 시작하지 않는다. `analyze`는 사전 기대치 파일을 갱신하지 않는다. 각 명령의 stdout·stderr·rc를 별도로 보존한다.

운전자는 기존 문서 정본 검사와 다음 사전등록 검사를 파일화 후 수행한다.

```bash
python3 -B scripts/check_prereg_preflight.py "$PREREG" --root .
python3 -B scripts/check_gate_referents.py "$PREREG" --root .
```

## 11. 판정과 폐합

### 분기 (표)

| 분기 | 조건 | 의미 |
|---|---|---|
| TM-X | 입력·감사·자원·실행·필수 증거·게이트 중 미충족 존재 | 미집행 또는 비교 자격 미충족. 원인을 명명 |
| TM-M | 필수 지도와 게이트가 유효하고 E-NE는 감소 관측이 아님 | 지도 측정 완료. 증가와 미분리를 별도 기록 |
| TM-D | 필수 지도와 게이트가 유효하고 E-NE 감소 관측 | 등록 표본의 `n_e` 원자료 대비 차이 감소 |

### 분기 끝

`TM-D`도 물리 수리 성공·CMFGEN 정확도 인증·MC 스펙트럼 재현을 뜻하지 않는다. NLTE 비계산 칸과 조건 불일치 칸은 결과에 남긴다.

종료감사 후 운전자는 실측 주장과 분기를 동일 입력·바이너리·증거·도구 해시로 독립 재현하여 `reproduced / not-reproduced / blocked`를 기록한다. 근거 없는 원인 확정이나 후속 수리 착수는 하지 않는다.

## 12. INPUT_PENDING과 획득 방법

| 항목 | 현재 결손 | 획득 방법 |
|---|---|---|
| 정확한 실행 계보 | 실물 binary 해시·대응 source·effective exports 검증 | 운전자 읽기 전용 검증과 새 증거 루트 봉인 |
| 기록자 자격 | 최종 공개 상태와 덤프 세대 연결 미시연 | sol 오프라인 경로 시험, astra 코드 검수 |
| NLTE 모집단·분모 | 실제 계산 단계·전 준위·원소 총량 계약 미봉인 | 정확한 덱·구성·source의 오프라인 census |
| 표현 구간 | 출력 자릿수 전파의 검증 구현 없음 | 독립 합성 산술·결함 주입 후 봉인 |
| 실행 경로 | 새 루트·runner·전체 exports delta 미확정 | 운전자 파일화·staging 준비 |
| 자원 자격 | 새 덤프량·오프라인 비용·하드웨어 수용 근거 | §9의 기존 기록·구조 산식·CPU 시험 |
| 사전감사·코드 검수 | 미집행 | 개정23 담당에게 발주 |

base 전체 분율 덤프의 부재는 **base 대비 분율 개선을 판정할 수 없는 한계**로 보존한다. 이를 얻기 위한 base 재런은 이번 단에 등록하지 않는다.

## 13. 집행 기록과 분장 장부

현행 분장은 사용자 직접 지시와 `CLAUDE.md` 개정23을 따른다.

| 단계 | 규약상 담당 | 실제 | 위반·제한 |
|---|---|---|---|
| 계획 기안 | astra | 본 기안 작성 | 실행·감사 verdict 없음 |
| 계약 사전감사 | astra | 미집행 | 자기 계획 감사임을 표기해야 함 |
| 코딩·직접 selftest | sol | 미집행 | 현재 변경집합 구현 없음 |
| 코드 검수 | astra | 미집행 | 첫 질문: 봉인 실물에 적용해도 되는가 |
| 파일화·프리플라이트·staging·실행 | 운전자 | 미집행 | 이번 응답은 파일화하지 않음 |
| 판정·종료감사 | astra | 미집행 | 실행 전 PASS 기재 없음 |
| 실측 주장 독립 재현·대장 | 운전자 | 미집행 | 폐합 전 필수 |

발주 프리플라이트 rc, 미실행 주장 확인 결과, 발주 중단 사유와 주인, 실제 담당을 이 절에 기록한다. 다른 판정문에 장부를 복제하지 않는다.

## 14. prereg-preflight 선언

아래 선언은 구조·참조 검사용이다. 통과하더라도 `INPUT_PENDING`·자원 자격·감사 조건을 해제하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — CMFGEN_TWIN_MAP (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts)/[A-Za-z0-9_/.]+\\.(?:py|sh)",
    "symbol": "CMFGEN_TWIN_MAP",
    "roots": ["scripts"],
    "expected_extra": [
      "scripts/cmfgen_twin_map.py",
      "scripts/stage_cmfgen_twin_map.sh"
    ]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    "metrics": {
      "count": "len(v)",
      "invalid": "sum(1 for x in v if x < 0.5)",
      "not_decreased": "sum(1 for x in v if x > 1.5)"
    },
    "rules": [
      {
        "name": "TM-X",
        "predicate": "count == 0 or invalid > 0"
      },
      {
        "name": "TM-M",
        "predicate": "count > 0 and invalid == 0 and not_decreased > 0"
      },
      {
        "name": "TM-D",
        "predicate": "count > 0 and invalid == 0 and not_decreased == 0"
      }
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "input-pending", "mix": [[0.0, 1]]},
      {"name": "valid-decrease", "mix": [[1.0, 9]]},
      {"name": "valid-not-decreased", "mix": [[1.0, 8], [2.0, 1]]},
      {"name": "invalid-takes-precedence", "mix": [[0.0, 1], [2.0, 1]]}
    ],
    "residual": "TM-X"
  },
  "references": [
    {"path": "CLAUDE.md"},
    {"path": "docs/O0_CMFGEN_DIVERGENCE_MAP_SURVEY_2026-09-08.md"},
    {"path": "docs/DESIGN_DC_DD_CMFGEN_CONTRAST_2026-09-05.md"},
    {"path": "docs/VERIFICATION_REGISTERS.md"},
    {"path": "docs/PREREG_NE_THRESH_2026-09-08.md"},
    {"path": "docs/VERDICT_DC1_NECLOSE_R2_2026-09-08.md"},
    {
      "path": "src/lumina_plasma.c",
      "flags_existing": [
        "LUMINA_FIXED_TE_PROFILE",
        "LUMINA_NLTE_LEVEL_DUMP"
      ]
    },
    {
      "path": "src/lumina_cuda.cu",
      "flags_existing": [
        "LUMINA_ION_POP_DUMP",
        "LUMINA_ION_POP_DUMP_ITER",
        "lumina_plasma_state.csv"
      ]
    },
    {"path": "scripts/check_prereg_preflight.py"},
    {"path": "scripts/check_gate_referents.py"}
  ]
}
```

분기 입력은 등록 게이트 전건과 E-NE 판정 한 건으로 구성한다. 게이트 미충족은 `0`, 충족은 `1`; E-NE는 감소 `1`, 증가·미분리 `2`, 판정 불능 `0`이다. 누락된 게이트를 벡터에서 빼고 통과시키지 않는다.


## 집행 기록 1 — 기안 파일화 · 운전자 독립 재현 (2026-09-08, 운전자 기재)

**경위**: Q-GOAL 사전감사가 DET-NE-THRESH 를 `부분부합·BLOCKING 1` 로 막고 대안을 지목했고,
**user 가 2026-09-08 「1번으로 갑니다」로 이 표적을 선택**했다. 계획 기안 = astra(`gpt-6-astra`,
read-only, 106,855 토큰). 운전자는 기안 본문을 **축자 복사**해 파일화했다(보고행 3줄 `PLAN_SCOPE`·
`INPUT_PENDING`·`GOAL_CELL` 은 계약 본문이 아니므로 아래에 옮겨 적는다 — 개정 3 정정의 재발 방지).

```
PLAN_SCOPE: 1 run / ≤8 GPU-hour / ≤129 CPU-hour — 자원 자격·선행조건 충족 후의 제안 상한
INPUT_PENDING: 실행 identity, 기록자·공개 세대 연결, NLTE 모집단·분모, 표현 구간 검증, 실행 경로·delta, 자원 자격, 사전감사·코드 검수
GOAL_CELL: base 각 셸의 CMFGEN 원자료 표본점에서 권위별 f_q 차이와 공개 n_e 비·dex를 기록하는 쌍둥이-T 표본 지도
```

### 운전자 게이트 (전건 실행)

| 게이트 | rc | 세부 |
|---|---|---|
| `check_prereg_preflight.py` | **0** | `changeset files=2` · `branch-partition trials=200000 rules=3 fixtures=5 residual=TM-X` · `references-resolved refs=10` |
| `check_gate_referents.py` | **0** | `changeset_exempt=16` |
| `check_doc_canonical.py` | **0** | 인용 문서 전건 정본 단일 |

### 운전자 독립 재현 — §3.1·§5 의 `[저자 실행]` 주장 (개정15·16 의무)

도구 = `/gpfs/kjhan/lumina/cmfgen_map/verify_plan_claims.py` (운전자 저작, 기안과 독립 구현).
입력 = 공개 `phys_toy06_cmfgen.txt` 19.480 d 블록(100행) · `BASE/input/model/geometry.csv`(50셸) ·
`BASE/work/lumina_plasma_state.csv`.

| 주장 | 기안 | 운전자 재현 | 판정 |
|---|---|---|---|
| 전 셸에 표본 존재 | 전 셸 존재 | 결손 0 | **reproduced** |
| 표본 행 인덱스 50건 | `21,23,25,…,93,94` | 문자열 전건 일치 | **reproduced** |
| `median\|d_e\|` | `0.038859615680874218` | `0.03885961568087422` (동일 double) | **reproduced** |
| `max\|d_e\|` · 극값 셸 | `0.21875037534438782` · shell 1 | 동일 · shell 1 | **reproduced** |

⚠**재현은 계약의 정당성이 아니라 산술의 일치만 뜻한다.** 이 수는 **base(전 셸 10,020 K 고정)** 대비
값이므로 T 교란이 살아 있는 상태의 크기이며, 쌍둥이-T 레인의 결과가 아니다. 「Lumina 의 n_e 가
CMFGEN 대비 9.4 % 틀리다」로 읽지 마라 — 조건 불일치 장부(§3.3)의 T 행이 이 칸을 제한한다.

★**이 캠페인 최초의 CMFGEN 기준 수치**라는 점만 기재한다. 정확도 판정이 아니다.

### 사전감사 발주 (진행)

발주서 `/gpfs/kjhan/lumina/cmfgen_map/preaudit/ORDER.md` — **Q-GOAL 을 첫 문항**으로 싣고
「미측정 칸을 채우는 것과 오차를 줄이는 것을 구분하라」를 명시했다. **자기 계획 자기 감사**
(기안·감사 모두 astra)임을 발주서에 표기하게 했다 — 독립 감사 PASS 로 읽지 않는다.


## 집행 기록 2 — 사전감사 1회차 · 재현 범위 확정 (2026-09-08, 운전자 기재)

| 항목 | 결과 |
|---|---|
| 사전감사 (astra, **자기 계획 자기 감사**) | **`Q_GOAL: 부분부합` · `PRE_AUDIT: 통과` · `BLOCKING: 0`** |
| 산출 | `/gpfs/kjhan/lumina/cmfgen_map/preaudit/ASTRA_PREAUDIT.md` |
| 감사 대상 계약 SHA-256 | `0c0d17b7cb9665e745ac5f67d9d799e1df8dc5e39116b87328fc798d356c34e3` — **현행본과 일치**(운전자 재계산) |

**Q-GOAL = 부분부합, 계급 «기재로 족함» ⟹ 정지 사유 아님.** 근거: ① **줄인다고 입증된 CMFGEN 오차 칸은
아직 없다** — 미측정 칸을 채우는 것을 오차 감소로 인정하지 않는다. 다만 계약이 이를 수리 성공으로
주장하지 않으므로 설계 차단으로 삼지 않는다. ② 물리 상태의 **대조 측정**이지 MC 스펙트럼 정확도
검증은 아니다. ③ D-C-2·D-D 가 더 직접적이나 어느 쪽이 CMFGEN 오차를 더 줄일지는 미확정 —
**대안의 존재를 이 진단의 필연적 열등성으로 읽지 않는다**. ④ **유용한 진단이지 필수 관문은 아니다.**
⟹ **제한된 진단 표적으로 정당화**되며 완전 부합이나 오차 감소 실적으로 승격하지 않는다.

**«기재로 족함» 6건 (설계 변경 없이 여기 기록)**: ⓐ O-0 §4 의 「T 불일치가 지배한다」·「7자릿수 차이를
만들었다」는 **인과 확정으로 입증되지 않았다 — 가설의 동기로만 쓴다** ⓑ G-METRIC 은 `|log10(r)|` 가
비를 뒤집어도 같으므로 **부호 있는 원값·비·dex** 를 검사해야 한다(§4 가 이미 요구) ⓒ base 자기비교는
감소 분기의 **음성 확인일 뿐 감소 검출 능력의 양성 검증이 아니다** ⓓ 조건 불일치 장부는 완전 인증
불가 — 원자자료·준위 절단·내경계 복사 조건·γ/비열·진단 복사장 설정도 구분해 남길 필요 ⓔ 봉인 보존은
**실행 검증 미시연** ⓕ 음성대조 방향은 적절하나 **전부 미시연**.

### 재현 범위 확정 (감사의 직접 질문에 대한 답)

증거 = `/gpfs/kjhan/lumina/cmfgen_map/evid/reproduce.log` (도구·계약·입력·산출 sha256, 명령, rc, 경로).

| | 항목 |
|---|---|
| **운전자 독립 재현 완료** | §3.1 표본 행 인덱스 50건 · §5 `median\|d_e\|` · `max\|d_e\|` · 극값 셸 — 운전자 저작 독립 스크립트(`tool_sha256=8a50b0d0…`), rc 0, 결정적 출력 `repro_stdout.log`(`sha256=cbd4d1a2…`) |
| **운전자 실행 완료(게이트)** | `check_prereg_preflight` rc 0 · `check_gate_referents` rc 0 · `check_doc_canonical` rc 0 — 감사는 이번에 **재실행하지 않았다** |
| **미재현 — 미확인 유지** | 전 `ionfrac` 키 조인 · T 불일치 전건 · 표현 구간 · 실제 loader 소비와 공개 세대 연결 · 전체 모집단·분모 · 실물 바이너리 동일성 · 자원 자격 · 보존 게이트 |

⚠**감사의 재현은 운전자 스크립트의 재실행이었다 — 새 독립 구현이 아니다.** 두 겹이 아니라 한 겹이다.


## 집행 기록 3 — sol 구현 · 코드 검수 1회차 «차단 7» · 운전자 이탈 1 (2026-09-08)

### sol 구현 (개정23) 착지

`scripts/cmfgen_twin_map.py`(862행) · `scripts/stage_cmfgen_twin_map.sh`(279행). **운전자 재현**:
`CMFGEN_TWIN_MAP_SELFTEST_PASS positive=8 negative=28` rc 0 · `CMFGEN_TWIN_MAP_STAGER_SELFTEST_PASS
positive=5 negative=8` rc 0. ⚠sol 보고 말미가 내부 사고로 오염돼 명령줄이 판독 불가였다 —
**보고를 근거로 쓰지 않고 소스·실행으로 확인**했다.

### 운전자 오프라인 `prepare` 실행 결과

| 시도 | 결과 |
|---|---|
| 1 | `EVIDENCE_NOT_NEW_SAFE_ABSOLUTE` — 운전자가 증거 디렉토리를 미리 만든 탓. **게이트가 옳게 거부** |
| 2 | 참조 자료 전건 `INPUT_PENDING` — `data/` 는 **git 미추적**이라 worktree 에 없다. **판독기가 자료를 지어내지 않았다** |
| 3 (절대경로) | `status=INPUT_PENDING shells=50/50 pending=1`, 잔여 = `final_publication_generation_link` |

산출: 쌍둥이 T_e 프로파일 50행(shell 0 = **1.8900E+04 K** … shell 49 = **2.4385E+04 K**; base 는 전 셸
10,020 K) · `base_map.csv` 1,700행 = `PUBLIC_NE/CONDITION_MISMATCH/T_NOT_TWIN` **50** +
`SAHA_STAGE/INPUT_PENDING/BASE_STAGE_DUMP_MISSING` **1,650**(계약이 등록한 한계) ·
전자밀도 행에 원값·비·부호 dex·**표현 구간**(`dex_lo`/`dex_hi`)·조건 불일치 8축.
계보 확인: 판독기가 잡은 소스 저장소·바이너리가 봉인 base 의 `current_source_identity.json`
(`git_head=364ee72`, `binary_sha256=ca721b07…`)과 **일치** — 올바른 해석이었다.

### 코드 검수 (astra, 구현자와 다른 계열) — **`CODE_REVIEW: 차단 7` · `SAFE_FOR_RUN: NO`**

| | «폐합 차단» |
|---|---|
| **B1** | `n_L=0, n_C>0` 의 정의되는 비 0 을 기록하지 않고 실패. 분모 0 미처리 — **실제 BASE 원소 총량 300칸 중 126칸이 정확한 0**, `0/0` 이 `decimal.InvalidOperation` 발생. 실패 셀·좌표·원값 미보존 |
| **B2** | ★**selftest 자기참조** — 정상 fixture 기대값을 검사 대상 함수로 생성. 검수가 `scalar_metric(a,b)`→`원함수(b,a)` 로 **메모리 변조했는데 selftest 가 그대로 `positive=8 negative=28` PASS**. 생산 `prepare/analyze` 는 시험된 판정기를 쓰지도 않는다 |
| **B3** | 분석 말미가 게이트 전건을 **일괄 `PASS`** 로 만든다 ⟹ 실패런도 `TM-M/TM-D` 로 승격 가능 |
| **B4** | stager 목적지 검사가 **정규화 없는 문자열 접두사** — BASE 경로에 `/./` 를 끼운 **내부 목적지를 rc 0 으로 수용**. EVID 의 BASE 내부 여부 미검사 |
| **B5** | 실효 exports 에 **`LUMINA_NLTE_SKIP_Z="14"`** 가 있는데 `active_ions.csv` 전체를 계산 집합으로 가정. 기록자는 superlevel 이 아니라 **개별 준위**를 순회 ⟹ 정상 덤프를 결함으로 오판 |
| **B6** | 결손 하나가 판독 가능한 셀까지 버리고 **빈 지도**를 만든다. T 불일치가 상태로 남지 않고 예외 종료 |
| **B7** | 복사하는 `input/job.slurm` 이 **GPU 1개·48시간 요청** — 계약의 4시간·8 GPU-hour 상한을 집행할 runner 자격 없음. 자원 검사도 신규 총량이 아닌 덤프 추정치 단독 |

«기재로 족함» 2: ⓐ 계약 기안자도 astra 이므로 **계약 자체에 대한 독립성은 없다** ⓑ 사전감사가 인용한
계약 SHA(`0c0d17b7…`)와 현행(`0a7ee6fb…`)이 다르다 — **사전감사 PASS 를 현행본 동일 해시 검증으로
전용할 수 없다**(차이는 집행 기록 2 추가분이며 계약 본문 §1~§14 변경 아님).

### ★운전자 이탈 (지우지 않고 기재)

**개정16 ② 위반**: 「코드 검수의 첫 답(봉인 접촉 가부)이 오기 전 운전자는 봉인 접촉 게이트를 돌리지
않는다」인데, 운전자가 **Q0 답 이전에 `prepare` 를 봉인 base 에 돌렸다**. 검수의 Q0 는 `NO` 였고 B4 는
**봉인 내부 쓰기가 가능한 코드**임을 실측으로 보였다 — 운전자가 지정한 증거 루트가 BASE 밖이라
실현되지 않았을 뿐, 규약이 막으려던 바로 그 위험이었다.

**봉인 실측 (사후 확인)**: BASE 내부 09-08 변경 파일 **0건** · 바이너리 sha256 기록치와 실측 일치
(`ca721b07…`, `BINARY_SEAL_INTACT`) · 증거 루트 2곳 모두 `inside_base=False`. **봉인 무손상.**
그러나 이것은 **결과가 무사했다는 것이지 절차가 옳았다는 뜻이 아니다.**

### 처분

수정은 **sol** 에 발주(개정23). 발주서에 ① selftest 자기참조 제거(기대값을 검사 대상 함수로 만들지
말 것·**구현 함수 변조 시 FAIL 시연**을 반영의 증명으로) ② 경로 판정을 정규화·실경로로, 검수가
통과시킨 형태를 음성대조에 포함 ③ 게이트 일괄 통과 금지 ④ 셀 단위 상태 보존 ⑤ 실효 구성에서
모집단 읽기 ⑥ runner 의 자원 상한 집행을 명시했다. 발주 프리플라이트 rc 0.

