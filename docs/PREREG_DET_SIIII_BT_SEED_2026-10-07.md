# 사전등록 — DET-SIIII-BT-SEED: 세대 0 물질 온도 변경·오프라인 검증·판정런 1회

2026-10-07 · astra 기안 · read-only  
운전자 파일화 대상: `docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md` — 신설  
상태: **생산 변경·직접시험·빌드·기계 게이트·이번 계약의 경량 사전감사·판정런 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답과 과학 질문

### Q-GOAL — 최종목적에 직접 기여하는가?

최종목적은 Monte-Carlo 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현하는 것이다. 직전 단은 Si III 1206.50 Å의 초기 씨앗과 후속 유지 기전을 특정하고, 초기 물질 온도를 봉인된 FIXED_T 프로파일로 바꾸는 후보 B-T의 조건부 기대치를 등록했다. 그러나 그 생산 개입이 실제 초기 물질에 도달하는지와, 첫 수송 뒤 발행되는 J̄가 등록 씨앗으로 이동하는지는 미검증이었다.

이번 목표는 **B-T의 세대 0 물질 온도 개입을 구현하고 오프라인에서 검증한 뒤, E′를 기준으로 한 번의 판정런에서 씨앗 전달과 등록된 후속 기대의 결과를 측정하는 것**이다. 직접 겨냥하는 공개 비교의 어긋남은 E′N의 NUV/OPT 및 2500–3000 Å 과잉이다. 초기 씨앗 변경이 이 어긋남을 줄인다고 선취하지 않는다.

과학 질문은 다음 순서로 고정한다.

1. **1차 질문:** B-T 생산 개입 뒤 발행 call-1 J̄가 Si III 1206.50 Å의 다섯 등록 셸에서 각각 \(\mathcal B_\ell(T_{{\rm pin},s})\)에 상대편차 \(10^{-3}\) 이내로 도달하는가?
2. **기전 질문:** 물질 세대 7의 \(n_4/n_0\)는 직전 기대치의 조건부 값과 얼마나 다른가? 새 런에서도 축약 모형이 적용되는가? 고정 E′ 배경을 사용한 예측의 생산 외삽은 성립하는가?
3. **2542 Å 질문:** 실제 \(n_0,n_4,n_5,S,\tau,Q\) 응답은 등록 P1·P2 경계 수치와 어떻게 다른가?
4. **공개 비교의 2차 질문:** 등록한 NUV·UV2500 잔차 반감, 네 대역 보존, 여덟 대역 RMS 개선을 만족하는가?

생략하면 온도 배열 변경을 실제 씨앗 전달로 오인하거나, 씨앗 전달 성공을 스펙트럼 개선으로 확대하게 된다. 이번 단은 이 둘을 실제 생산 경로에서 구별한다.

**기안자 자기답: PASS.**

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

후보는 B-T 하나이며 GPU 판정런은 한 번이다. 기존 E′와 R′를 다시 실행하지 않는다. 생산 변경은 초기 온도 소유자와 그 bootstrap 발행 신원에 한정한다. 기존 프로파일 로더·검증기·물질 계산·수송·W5·최종 캡처·스펙트럼 작성기를 재사용한다.

오프라인 검증은 생산 함수를 연결한 작은 CPU 시험, OFF 대조, 결함 주입, 생산 빌드와 실행 신원 확인으로 묶는다. 전체 INIT 수송을 별도의 GPU 예비 런으로 추가하지 않는다. 판정런의 call 1이 실제 씨앗 전달 시험이다.

판독은 다음에 한정한다.

- 1206.50 Å 다섯 셸의 call 1–7과 물질 세대 7.
- 2542.58 Å의 기존 등록 셸.
- 기존 공개 비교 정의에 따른 스펙트럼.
- 이미 고정된 122 선·셸의 call-1 이동 요약. 새로운 기전 census나 황 이온화율 분석은 만들지 않는다.

소비자는 운전자와 사용자다. 오프라인 결과는 제출 가능 여부를, call 1은 씨앗 전달 여부를, 세대 7·2542 Å·공개 비교는 B-T의 다음 처분을 결정한다.

**기안자 자기답: APPROPRIATE.**

### 승인 근거·분장

승인 근거는 다음 파일이다.

`/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/evidence/USER_DECISION_2026-10-07.txt`

사용자의 「승인함.」과 그 파일의 운전자 해석을 이번 발주의 명시적 범위와 함께 적용한다. 승인 범위는 **B-T 생산 변경·그 오프라인 검증·판정런 1회**다. 등록된 내부 작업 사이에 실행 승인을 반복 요청하지 않는다.

- 기안·계약 개정·종료감사: astra.
- 경량 사전감사: Fable fresh, Q-GOAL·Q-LEAN 전용 1회.
- 생산 변경 구현·직접시험·게이트·빌드·제출·판독·판정: 운전자(Opus 5.5).
- 코드 감사: 해제.
- 범위·비용 확대, 추가 판정런, 후속 과학 결정: 사용자.

Fable 사전감사는 기계 프리플라이트 뒤, 구현·빌드 전에 받는다. 직전 단의 감사나 기안자 자기답을 이번 감사로 대신하지 않는다. 호출 실패를 통과로 기록하거나 담당자를 조용히 바꾸지 않는다.

운전자는 구현자와 판정자를 겸한다. astra는 계약 저자가 종료감사를 수행한다. 이 이해관계를 기록한다.

### 사전 노출·주장 상한

기안자는 직전 계약·판정·기대치·사전감사, 봉인 소스와 E′ 실행 기록에 노출됐다. 기존 결과의 재인용은 새 예측 성공이 아니다.

이번 단은 다음을 선취하지 않는다.

- B-T 또는 기존 E′·R′ 중 어느 상태가 물리적으로 옳다는 판단.
- B-T의 기본값 전환이나 생산 승격.
- 초기값 독립성, 씨앗 기억 제거, 반복 수렴.
- 전체 생산 해의 독립 재현.
- 공개 CMFGEN 준위 인구와의 일치.
- 공개 비교 범위 밖 파장의 일치.
- 122건의 선 개수로부터 황 광이온화·에너지 기여율을 추론하는 것.

## §1. 경로·기준선·승계 사실

### 1.1 경로

```text
P  = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
C  = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
T2 = C/source
E  = C/Eprime/run1
R  = C/Rprime/run1

D  = /gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007
A0 = D/results/t5/analysis
N  = /gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002

B  = /gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007
T3 = B/source
B3 = B/cuda_build1
X  = B/BT/run1

새 CPU 러너 = B/tools/run_siiii_btseed_offline.slurm
새 GPU 러너 = B/tools/run_siiii_btseed_gpu.slurm
새 판독 결과 = B/results/<실행 식별자>
새 집행 증거 = B/evidence/<집행 식별자>

공개 스펙트럼 = P/data/standart_data1/toy06/spectra_toy06_cmfgen.txt
```

B는 승인문과 기안 산출 때문에 이미 존재할 수 있다. 새 source·build·run·결과 대상은 기존 산출을 덮어쓰지 않는다. 충돌한 경로는 용도를 확인하고 새 식별자로 보존한다.

### 1.2 기준선과 기대치의 권위

- 기안 시 저장소 HEAD: `bc7fe2d9edee0d53d33dd905d6b7d022aea04561`.
- 직전 계약: `docs/PREREG_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md`.
- 직전 판정: `docs/VERDICT_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md`.
- 직전 정본 산출: A0.
- 생산 기준판: T2와 E의 run_build_identity.json.
- 판정런의 직접 비교 기준: E.
- R은 기존 NRB 실측을 설명하는 참조이며 새 실행 대상이 아니다.

기대치 정본은 다음 파일이다.

`/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/results/t5/analysis/expectations.json`

```text
sha256 = 5a26aba7213ee1fecce8f4ff75c21bd2bba1d45692ec394ed56cee85d9e5262a
```

운전자는 실행 전에 실제 파일의 신원을 확인한다. 본문의 수치 표는 읽기 편의를 위한 전사이며, 판독기는 이 파일의 구간·조건·대상 키를 직접 소비한다. 기대치 파일을 새 실측값으로 갱신하지 않는다.

E의 기록된 실행 신원은 다음과 같다.

```text
gpu_profile = h200x2
outer_iterations = 6
binary_sha256 = 8ef03fb37b3e0de986f9c9ba15f0b7bbf8f98800d62ab18fb666ec518914f5d8
run_build_identity_sha256 = 4dd5d30e08c25cfe3aa29746643c77b2d847c821c27c00e64dc2bcdec4a43dc4
effective_exports_sha256 = 90a13354d8d063541568c8c76d329510702fb657e999f84008e337e76172d460
```

기록값과 실제 입력·빌드 장부의 연결은 실행 전 확인 대상이다. 기안 중 해시 재계산이나 수치 검증을 수행한 것으로 기록하지 않는다.

### 1.3 기안자가 직접 확인한 생산 경로

아래 행 번호는 T2 판본을 가리킨다.

| 근거 | 직접 확인한 내용 |
|---|---|
| lumina_atomic.c 약 1057행 | 세대 0의 opacity.t_electrons 전 셸을 config.T_inner로 채움 |
| lumina_cuda.cu 약 7156행 | 그 배열을 plasma.T_e로 복사하고 T_e_generation을 0으로 설정 |
| lumina_cuda.cu 약 7548행 | 첫 수송 전에 seed 온도 발행, bootstrap 창, solver 소유 τ 준비 순서로 호출 |
| lumina_plasma.c 약 8205행 | lumina_publish_seed_te가 온도 배열의 정본 해시를 만들고 세대 1로 발행 |
| lumina_plasma.c 약 8295행 | 발행 온도로 분배함수·전자밀도·이온 인구를 계산하고 물질 거래를 구성 |
| lumina_plasma.c 약 10245행 | 미커밋 NLTE 상태의 초기 상위 인구는 LTE 물질 경로를 사용 |
| lumina_plasma.c 약 17316행 | a210_fixed_te_profile_load가 셸 ID·중복·결손·비유한·비양수 값을 검사하고 정본 온도 해시를 만듦 |
| lumina_plasma.c 약 17498행 | 후속 a210_production_solve가 LUMINA_FIXED_TE_PROFILE의 존재로 FIXED_T 레인을 선택 |
| radeq_publication.h 약 162행 | 프로파일 크기와 기존 지원 온도 영역을 검증 |
| lumina_cmfgen.c 약 7630행 | INIT 선 물질이 생산 상위 인구와 τ를 소비하여 흡수·방출을 구성 |
| lumina_cmfgen.c 약 9074행 | 국소 T_e의 Planck 값을 전달해의 초기 추정값으로 설정 |
| lumina_cmfgen.c 약 9263행 | 물질 배열과 별도의 T_inner·inner_bb_scale을 정확 전달해에 전달 |
| lumina_cmfgen.c 약 11770행 | INIT J̄ 생성·복사 발행 뒤 seed material predictor로 진행 |

현재 저장소의 lumina_atomic.c는 T2의 해당 파일과 직접 대조했으며 차이가 없었다. 이를 다른 생산 파일이나 전체 현재 트리가 T2와 같다는 주장으로 확대하지 않는다.

E의 같은 stderr 안에서 INIT call 1 표지는 약 118행, 첫 FIXED_T 레인 표지는 약 242행이다. stdout과 stderr의 행 번호를 서로 비교하여 시간 순서를 주장하지 않는다.

### 1.4 승계 사실과 보존

직전 단은 B-T의 조건부 씨앗·준위비 계산을 완료했지만 실제 씨앗 전달을 검증하지 않았다. 등록 NRB-off 배경에서는 씨앗 기억이 남는다. 셸 10의 등록 씨앗 이동은 상승이며, 다른 네 주 셸은 하강이다.

2542 Å의 P1·P2는 조건부 경계 수치 기록이다. 실제 응답의 상한·하한이 아니다. 공개 스펙트럼의 반감 기대는 설계 목표이며 오프라인 수송 예보다.

T2·E·R·A0·기존 판독기·원자 자료·공개 자료는 보존한다. T3는 T2에서 재구성한 별도 판본이며, 현재 저장소의 다른 생산 변경을 함께 가져오지 않는다.

## §2. 생산 개입의 정의와 고정 범위

### 2.1 후보·표적

후보는 **B-T 하나**다.

| 용도 | 전이·데이터 행 | 셸 |
|---|---|---|
| 1차 씨앗 전달·후속 기전 | Si III 1206.50 Å, 0→4, 행 600693 | 10·15·19·24·30 |
| 2542 Å 실제 응답 | Si III 2542.58 Å, 4→5, 행 1097363 | 10·15·17·18·19·20·21·22·24·30 |
| 보조 씨앗 이동 | 직전 정본의 122 선·셸 키 | 키 그대로 유지 |

행 번호는 헤더를 제외한 0-based 데이터 행이다. W5의 line_id가 이 행 권위와 연결되는지 원자 정보로 확인한다.

세대 0 온도 변경은 주 셸만이 아니라 **프로파일의 모든 셸**에 적용한다. 이번 입력은 50셸이다. 일반 로더는 실제 기하의 셸 수를 소비하며 50을 생산 코드에 하드코딩하지 않는다.

### 2.2 분기와 온도 소유자

새 생산 노브는 **`LUMINA_BT_SEED` 하나**다.

| 설정 | 동작 |
|---|---|
| 미설정 또는 정확히 `"0"` | 기존 T_inner 씨앗 경로 |
| 정확히 `"1"` | 아래 B-T 경로 |
| 빈 문자열·그 밖의 값 | 명시적 설정 오류로 종료 |

ON은 비어 있지 않은 `LUMINA_FIXED_TE_PROFILE`을 요구한다. 또한 이번 개입의 적용 영역을 결정론적 생산 경로로 제한하여 `LUMINA_PURE_CMFGEN=1`, `LUMINA_DET_TRANSACTIONAL=1`을 요구한다. 이 조건을 만족하지 않으면 기존 온도로 되돌아가지 않고 종료한다.

온도 소유자는 lumina_atomic.c의 **세대 0 opacity.t_electrons 생성 지점**이다.

1. 기존 경계 온도 해석을 완료한 뒤 해당 지점에서 B-T 분기를 판별한다.
2. ON이면 기존 a210_fixed_te_profile_load와 a210_fixed_te_profile_validate를 사용한다.
3. 전 셸이 검증된 후에만 배열을 소유자에게 넘긴다.
4. `opacity.t_electrons[s] = T_pin[s]`이 되며, 이후 기존 CUDA 초기화의 복사 경로가 plasma.T_e에 전달한다.
5. OFF에서는 기존 할당·T_inner 대입 경로를 유지한다.

프로파일 경로는 기존 FIXED_T 노브 하나로 공유한다. 초기화 전용 프로파일, 온도 배율, 셸 선택 노브를 추가하지 않는다.

### 2.3 프로파일 거부와 원자적 적용

ON에서 다음은 발행 전에 거부한다.

- 프로파일 미지정·빈 경로·열기 또는 읽기 실패.
- 행 형식 오류, 중복 셸, 누락 셸, 범위 밖 셸 ID.
- 비유한·비양수 온도.
- 기존 FIXED_T 지원 영역 밖 온도.
- 기하와 프로파일의 셸 수 불일치.
- 프로파일과 실제 발행 대상 온도 배열의 불일치.

기존 지원 영역은 3500–140000 K다. 이 검사는 지원 영역 판별이며 값을 경계로 바꾸는 연산이 아니다. 온도·인구·τ·방출에 clamp·floor·cap·jitter·fallback을 추가하지 않는다.

실패한 부분 프로파일을 채워 쓰거나 T_inner로 보충하지 않는다. 실패 시 세대 발행과 물질 커밋이 진행되지 않아야 한다.

### 2.4 Bootstrap 발행과 FIXED_T의 관계

세대 0 배열 변경만으로 발행 신원이 정합하다고 간주하지 않는다. lumina_publish_seed_te의 ON 경로에서 다음을 수행한다.

- 같은 프로파일을 기존 로더로 읽어 검증한다.
- 실제 plasma.T_e와 셸별 binary64 값 및 정본 온도 해시가 일치하는지 확인한다.
- 일치 확인 뒤 기존의 세대 0→1 발행을 수행한다.
- ON의 발행 메타데이터는 `te_lane=FIXED_T`, `re_root_required=0`, 실제 `pinned_shells`, 정본 `te_profile_sha256`을 기록한다.
- 기존 `te_manifest_sha256`은 실제 발행 배열에서 계산하며 프로파일 해시와 일치해야 한다.
- 출처가 FIXED_T 프로파일인 **bootstrap LTE/Saha seed 발행**임을 완전한 해시와 함께 기록한다.

이 발행은 열평형 근 찾기나 후속 FIXED_T 물질 거래의 실행 결과가 아니다. 새 열평형 잔차·가짜 ledger·새 세대 증가를 만들지 않는다. OFF의 기존 발행 메타데이터와 로그는 보존한다.

후속 a210_production_solve의 FIXED_T 로더·풀이·잔차 진단·발행 순서는 바꾸지 않는다. 초기 발행과 후속 고정 온도 발행들이 같은 봉인 프로파일을 소비했는지 런 기록에서 확인한다. 파일 바이트 해시와 정본 온도 배열 해시는 서로 다른 신원으로 기록한다.

### 2.5 물질 일관성과 불변 범위

기존 순서를 유지한다.

```text
세대 0 온도 소유자
→ plasma.T_e 복사
→ seed T_e 세대 1 발행
→ bootstrap LTE/Saha
→ 분배함수·전자밀도·이온 인구
→ solver 소유 τ
→ LTE 상위 인구·선 방출
→ INIT 수송
→ J̄·복사 발행
→ seed material predictor
→ 기존 후속 FIXED_T 반복
```

바뀐 온도는 모든 관련 물질 계산의 입력이다. 기존 T_inner에서 만든 인구·분배함수·τ를 남겨 둔 채 온도나 fs.J만 교체하지 않는다.

다음은 변경하지 않는다.

- config.T_inner, 내경계 광도·색·inner_bb_scale·기하.
- 원자 자료·선 행·준위·충돌·광이온화 계수.
- LTE/Saha·NLTE·전자밀도 풀이와 수용 기준.
- τ·흡수·방출 계산식과 음의 opacity 처리 정책.
- INIT warm start의 식, 전달 연산자·격자·허용오차.
- NRB 설정, 반복 수, W5 표본, 최종 캡처와 관측자계 작성기.
- 기존 후속 온도 프로파일과 FIXED_T 풀이.
- 난수·입력 덱·침적·경계 조건.

온도 변경에 따라 인구·전자밀도·연속체·τ·방출·수송 결과가 달라지는 것은 개입의 결과다. 그 값을 기존값으로 강제 보존하지 않는다.

## §3. 판정런 전 오프라인 검증

### 3.1 요건 ②의 완료 범위

이번 사전 검증은 **구현한 생산 개입이 등록한 온도 소유자·물질 경로·발행 신원에 실제로 작동함**을 보여야 한다.

준비 상태를 다음처럼 구별한다.

| 항목 | 제출 전 필요한 상태 |
|---|---|
| 기전의 오프라인 특정 | 직전 정본과 §1의 생산 경로 연결 확인 |
| intervention_implementation_verified | 아래 실제 생산 함수 시험 통과 |
| offline_material_consistency_verified | 온도·물질·τ·방출·발행 연결 시험 통과 |
| production_build_verified | T2＋등록 변경집합의 빌드·신원 확인 |
| expectations_registered | 직전 기대치 신원과 이번 수용 기준 고정 |
| seed_delivery_verified | **미검증 유지** — 판정런 call 1에서 시험 |
| judgment_run_authorized | 사용자 승인과 위 선행 조건 충족 시 참 |

오프라인 물질 시험의 통과를 실제 INIT 수송의 씨앗 전달 성공으로 기록하지 않는다. 직전 사전감사의 두 경로 중 이번은 **생산 개입을 오프라인에서 확인한 뒤, 판정런 자체를 씨앗 전달 시험으로 사용하는 경로**다.

### 3.2 실제 생산 함수에 연결한 CPU 시험

새 C 시험은 T3의 실제 초기화 함수와 기존 생산 함수를 연결한다. 검증용 별도 Python 온도 대입으로 생산 개입을 대신하지 않는다.

작은 합성 원자 모형과 생산 형식 프로파일을 사용하며 다음을 확인한다.

1. **소유자와 분기**
   - 실제 초기화 함수가 OFF에서 T_inner, ON에서 프로파일 값을 반환한다.
   - 봉인 프로파일의 50셸 전부를 읽어 값과 셸 대응을 확인한다.
   - 서로 다른 온도의 작은 프로파일로 셸 교환을 검출한다.
   - 실제 생산 호출 지점이 시험한 함수에 연결됐는지 변경 소스로 확인한다.

2. **발행**
   - 실제 lumina_publish_seed_te를 호출한다.
   - ON에서 배열·정본 해시·프로파일 신원·레인·세대가 일치한다.
   - 중복 발행과 프로파일 불일치는 커밋 전에 거부한다.
   - OFF 발행은 기준판과 같다.

3. **물질·τ·방출**
   - 실제 bootstrap 창과 물질 계산 경로를 사용한다.
   - 분배함수·전자밀도·이온 인구·준위 인구가 같은 발행 온도를 소비한다.
   - 실제 solver 소유 τ, 실제 LTE 상위 인구 경로, INIT가 사용하는 line_net_sobolev_material을 연결한다.
   - 원자 모형은 준위 0·4·5에 대응하는 전이와 닫힌 전리 사다리를 포함하여 Saha·Boltzmann·τ·방출을 독립 참조와 대조할 수 있게 한다.
   - 전하·원소 보존과 기존 전자밀도 수용 기준을 확인한다.

온도 복사와 동일 배열의 신원은 정확 일치를 요구한다. 작은 모형의 분배함수·준위비·τ·방출은 독립 고정밀 계산에 대해 상대편차 \(10^{-10}\) 이내를 요구한다. 전자밀도 반복 잔차는 기존 생산 기준 \(10^{-8}\)을 유지한다. 정확한 영은 별도로 검사하고 임의의 작은 분모를 넣지 않는다.

합성 모형의 원자계수는 일관된 독립 정답용이다. 실제 Si III 원자계수의 직렬화 차이를 이 합성 모형에 맞춰 고치지 않는다. 실제 표적의 \(\mathcal B\) 비교는 직전 등록값을 사용한다.

### 3.3 OFF 대조와 결함 주입

T2와 T3의 OFF 경로를 같은 시험 입력·컴파일 조건에서 비교한다. 미설정과 `"0"` 모두 시험한다. 다음 물리 산출은 바이트가 같아야 한다.

- 초기 온도 배열.
- seed 발행의 기존 필드.
- 분배함수·전자밀도·이온 및 준위 인구.
- τ·상위 인구·국소 선 방출.

새 노브 등록으로 인한 헤더·바이너리 신원 차이는 별도로 기록한다. OFF 대조 때문에 전체 E′ GPU 런을 추가하지 않는다.

음성 대조는 다음 결함을 실제 시험·판독 경로에 주입한다.

| 결함 | 검출해야 할 실패 |
|---|---|
| ON인데 T_inner 배열을 그대로 반환 | 온도 소유자 불일치 |
| 온도만 바꾸고 기존 인구·분배함수·τ를 남김 | 물질 온도·세대·수치 불일치 |
| 발행 직전 한 셸의 온도 또는 프로파일 변경 | 프로파일·발행 배열 불일치 |
| 중복·누락 셸, 비유한 값, 잘못된 노브 값 | 입력 거부, 발행 없음 |
| fs.J 변경만을 물질 개입으로 표시 | 생산 소유자 검증 실패 |
| 새 노브를 전집에서 누락 | STRICT 등록 검사 실패 |

결함 주입은 시험 사본이나 시험 어댑터에서만 한다. 생산 결함 노브는 추가하지 않는다. 시험은 실제 오류 사유와 비영 종료를 확인해야 하며, 기대 상태를 직접 출력하는 방식은 인정하지 않는다.

### 3.4 짧은 실행의 가능 범위

작은 CPU 물질 시험은 전체 주파수 격자·GPU·INIT 전달해를 만들지 않으므로 짧은 실행으로 구성할 수 있다. 봉인 프로파일 판독과 작은 원자 모형 시험을 포함한 한 회의 상한은 §9의 300초다. 가능성과 비용은 구현 뒤 실측한다.

기존 `LUMINA_VALIDATION_EXIT_AFTER_R6`는 T2에서 단일 반복과 MGPU A/B 모드를 요구한다. E′의 여섯 반복 설정에 종료 노브만 붙인 동등한 예비 런이 아니다. 따라서 이번 오프라인 검증에는 사용하지 않는다.

CPU 시험이 실제 생산 함수를 연결하지 못하거나 등록 상한 안에서 필요한 증거를 만들지 못하면 요건 ②는 미완이다. 이를 문면 확인만으로 통과시키지 않으며 판정런을 제출하지 않는다.

### 3.5 빌드와 제출 전 반증 조건

T3는 T2 기준판에 §7의 변경만 적용한다. E의 B2 빌드 절차와 같은 컴파일러·CUDA·최적화·아키텍처·링크 소스를 사용한다.

다음이면 제출 준비 실패다.

- 실제 빌드 소스가 T2＋등록 변경집합과 다름.
- 시험한 초기화·발행 구현과 생산 빌드의 구현이 다름.
- OFF 물리 산출이 달라짐.
- ON이 프로파일을 일부만 적용하거나 잘못된 입력을 보충함.
- 실제 온도와 발행·분배함수·인구·τ 신원이 불일치함.
- 음성 대조가 결함을 검출하지 못함.
- 기대치·실행 설정·등록 비용이 확정되지 않음.

## §4. 판정런 1회의 설계

### 4.1 E′ 대비 변경

새 런 X는 E를 기준으로 한다.

| 항목 | X의 설정 |
|---|---|
| 생산 바이너리 | T2＋§7 변경집합으로 새로 빌드한 B3 |
| 물리 개입 | `LUMINA_BT_SEED=1` |
| 후속 온도 | E의 fixed_te_profile과 같은 바이트 |
| NRB | E와 같은 OFF |
| 바깥 반복 | 6 |
| 생산 호출 인자 | E와 같은 `model 1 2 spectrum nlte` 형태 |
| 결정론적 반복 수 | `LUMINA_PURE_CMFGEN_ITER=6` 유지 |
| 수송·격자·오차 기준 | E와 동일 |
| W5·관측자계 작성기·캡처 | E와 동일하게 활성화 |
| 경로 | X의 입력·work·출력 경로로 이식 |
| GPU | h200 두 장 우선, 실행 전 h100 두 장으로 대체 가능 |

명령행의 `2`를 결정론적 바깥 반복 수로 오독하지 않는다. 실제 여섯 반복은 E의 유효 exports와 실행 경로를 승계한다.

허용 exports 차이는 새 노브와 런 경로 이식뿐이다. GPU 프로파일 변경은 자원 설정으로 별도 기록한다. NRB·프로파일·반복 수·허용오차·W5 선택을 함께 바꾸지 않는다.

### 4.2 소스·입력·런 신원

운전자는 다음 연결을 제출 전에 봉인한다.

```text
T2 기준판과 빌드 manifest
→ 등록 patch
→ T3 실제 소스
→ B3 빌드 명령·컴파일러·바이너리
→ X 입력·exports·러너
→ Slurm 작업
→ W5·캡처·스펙트럼
→ 판독 결과
```

기존 186경로 빌드 목록을 기준 목록으로 보존하되, 새 시험·생성물과 실제 컴파일 의존 목록을 별도로 명시한다. 새 파일을 포함한 목록을 억지로 186개로 맞추지 않는다.

T3의 env 전집은 기존 생성기를 실행하여 생성한다. 생성기가 소비한 파일 목록을 기록하고, 기존 이름의 임의 삭제·추가 없이 새 생산 노브가 등록됐는지 확인한다.

X에는 OWNER.txt, STAGE_IDENTITY.txt, SUBMIT_COMMAND.txt, 바이너리·덱·원자 자료·프로파일 신원, 실행 명령과 유효 exports를 둔다. T3와 B3를 E의 기존 신원으로 표시하지 않는다.

### 4.3 필요한 런 산출과 소비 세대

필수 산출은 다음이다.

- stdout·stderr·모델 종료 코드·Slurm 종료 기록·실행 footer.
- W5 call 1–7.
- 기존 DET 관측자계 스펙트럼과 작성기 상태.
- 최종 det_obs_capture의 manifest·meta·lines.bin·Te·ne 및 기존 payload.
- 기존 최종 발행 manifest.
- 입력·바이너리·소스·노브·프로파일 연결 기록.

call 1은 다음 신원을 요구한다.

```text
operator = INIT_SHARED_GAUSSIAN
rad_generation = 1
material_generation = 0
```

여기서 W5의 material_generation 0은 미커밋 NLTE 표지다. seed T_e 세대 1이나 bootstrap 물질 거래가 없다는 뜻이 아니다.

W5 행을 썼다는 사실만으로 복사 발행 완료를 선언하지 않는다. 같은 런의 R6 신원과 INIT 이후 소비 기록을 연결한다. 불완전 실행에서는 실제 commit 이후에만 나타나는 기존 소비 증거가 있어야 call-1 값을 발행값으로 인정한다.

최종 주 판독은 기존 작성기가 소비한 **물질 7·복사 7·iter 5** 캡처다. 물질 7의 준위비는 radiation 6 소비와 연결한다. 후속 최종 덤프의 세대 8 인구로 대체하지 않는다. τ 세대 번호는 물질 세대 번호와 별도로 검사한다.

### 4.4 실행 횟수와 실패 후 규칙

- 승인된 생산 모델 진입은 **최대 한 번**이다.
- 자동 재큐·자동 재제출·자동 반복 연장을 금지한다.
- h200에서 아직 모델을 시작하지 않은 대기 작업을 취소한 뒤 h100으로 옮기는 것은 같은 한 번의 실행 선택이다. 동시 활성 제출은 하나만 둔다.
- 모델 시작 전 제출·스테이징 오류를 수리할 수 있으나, 시작하지 않았다는 기록을 남긴다.
- 모델이 시작된 뒤의 시간 초과·노드 실패·생산 오류·과학 기대 반증은 재실행 권한을 만들지 않는다.
- 저장 자료의 판독기 결함은 원시 자료를 보존한 채 CPU 재판독으로 수리한다.
- 생산 코드 수정이나 추가 GPU 런이 필요하면 원인·기존 증거·필요 범위를 보고하여 새 결정을 받는다.

씨앗 전달 실패를 보고 같은 런의 반복 수를 늘리거나 다른 후보를 실행하지 않는다.

## §5. 기대치·판독·반증

### 5.1 1차 씨앗 전달

등록 씨앗은 생산 Gaussian 프로파일 평균이다.

\[
\mathcal B_\ell(T)
=
\frac{\sum_{i\in Q_\ell}e^{-x_i^2}\Delta\nu_i B_{\nu_i}(T)}
{\sum_{i\in Q_\ell}e^{-x_i^2}\Delta\nu_i},
\qquad |x_i|\le4.
\]

국소 B-T 씨앗에는 기하 희석이나 내경계 배율을 곱하지 않는다.

\[
D_{{\rm seed},s}
=
\left|
\frac{\bar J_{1,s}^{X}}
{\mathcal B_\ell(T_{{\rm pin},s})}-1
\right|
\le10^{-3}.
\]

**다섯 셸 모두** 만족해야 1차 기대가 생존한다.

| 셸 | 등록 \(\mathcal B_\ell(T_{\rm pin})\) 중앙값 | 기존 씨앗 대비 등록 log10 이동 |
|---:|---:|---:|
| 10 | 2.014908105200524e-6 | +0.11844878384728963 |
| 15 | 7.883913161767992e-7 | −0.2890646278679971 |
| 19 | 3.4987647289771947e-7 | −0.641891719894521 |
| 24 | 5.548876602472329e-7 | −0.4416013914091447 |
| 30 | 1.2731039304049982e-6 | −0.08094258670997274 |

판독기는 기대치 파일의 전체 구간과 W5의 Jbar_err_upper를 사용한다. 셸 10의 상승을 실패로 처리하지 않는다. 기존 E′ 값에 그대로 남은 경우는 새 등록 씨앗에 대한 편차로 판단한다.

온도·발행 연결이 적격한데 \(D_{\rm seed}\)가 기준 밖이면 **씨앗 전달 기대의 반증**이다. 생산 입력이 잘못 적용됐거나 발행 신원이 틀렸다면 적격성 실패이며 과학 반증으로 바꾸지 않는다.

### 5.2 물질 세대 7의 기전 기대

직전 등록 중앙값은 다음과 같다.

| 셸 | 등록 \(r_{7,s}^{E{\rm bg}}=(n_4/n_0)_{7,s}\) |
|---:|---:|
| 10 | 2.671971870665429e-5 |
| 15 | 1.04549217830839e-5 |
| 19 | 4.639748621639949e-6 |
| 24 | 7.358410219645104e-6 |
| 30 | 1.6882600042749766e-5 |

등록 비교는

\[
D_{{\rm reg},s}
=
\left|
\frac{(n_4/n_0)_{7,s}^{X}}
{r_{7,s}^{E{\rm bg}}}-1
\right|
\le10^{-3}
\]

이다. 실제 세대 7의 비는 같은 세대 캡처에서 생산 관계를 역산하여 얻는다. 적용 권위·mapped 여부·source fallback·부호 조건을 먼저 확인한다.

다음 결과를 분리하여 모두 출력한다.

1. **등록값 일치:** 위 고정 기대치와의 비교. 새 런 배경으로 기대치를 다시 계산하여 대체하지 않는다.
2. **새 런의 축약 모형 적용성:** 실제 \(\bar J_6^X\)와 세대 7 물질을 사용한 고립 전이 비 및 충돌 포함 비의 비교.
3. **고정 배경 조건의 변화:** E′와 X의 call 2–7 β·\(J_c\), 실제 \(T_e,n_e\) 차이.
4. **후속 사상 검사:** 기존 생산 \(\kappa,\Delta B\)를 유지하여 실제 X 이력의 한 단계 응답을 검사한다.

방사 전용 비는

\[
r_{\rm off}(J)=
\frac{B_{lu}J}{A_{ul}+B_{ul}J}
\]

이며, 충돌 포함 비는

\[
r_{\rm off+C}(J)=
\frac{B_{lu}J+C_{lu}}
{A_{ul}+B_{ul}J+C_{ul}}
\]

다. 적용성 비교의 상대 기준은 직전과 같은 \(10^{-3}\)이다.

등록값과의 불일치는 고정 E′ 배경 예측의 생산 외삽 결과로 남긴다. 새 런에서 축약 모형까지 실패했는지는 별도로 판정한다. 배경이 달라졌다는 이유로 등록값 비교를 삭제하지 않으며, 등록값 일치만으로 축약 모형의 전제가 입증됐다고 쓰지 않는다.

B-T는 기존 상태에서의 이동을 시험한다. 씨앗 기억 제거는 성공 조건이 아니다.

### 5.3 2542 Å 실제 응답과 등록 경계 수치

물질 세대 7에서 E′와 X의 \(n_0,n_4,n_5,\tau_{45},S_{45}\)를 같은 권위로 연결한다.

\[
\alpha_i=\frac{n_i^X}{n_i^E},\qquad
q_{45}=\frac{g_4n_5}{g_5n_4}.
\]

양의 흡수가 분리된 경우에

\[
Q_{45}=S_{45}(1-e^{-\tau_{45}})
\]

와 \(Q_{45}^X/Q_{45}^E\)를 계산한다. 이는 국소 공명 주입량이며 광도나 탈출 스펙트럼이 아니다.

등록 P1·P2 중앙값은 다음과 같다.

| 셸 | P1 | P2 |
|---:|---:|---:|
| 10 | 1.0 | 0.7606721039051447 |
| 15 | 0.9999999998138888 | 1.9513661070283475 |
| 17 | 0.9996672064628609 | 3.3076394242480482 |
| 18 | 0.9957093835072364 | 3.7628682244263065 |
| 19 | 0.9732424575052895 | 4.298666073780585 |
| 20 | 0.9222289588647341 | 4.482332103272149 |
| 21 | 0.9220059292771745 | 3.4709213443916296 |
| 22 | 0.8876602476533414 | 3.020851240299066 |
| 24 | 0.809457675056473 | 2.2396013332815996 |
| 30 | 0.8886658517077437 | 1.0707332701375298 |

P1은 \(n_0\) 고정·\(\alpha_5=\alpha_4\), P2는 \(n_0\) 고정·\(\alpha_5=1\)의 대조다. 셸 17·18·20·21·22의 등록값은 W5 반복 이력 예측이 아니라 씨앗 비만 사용한 경계 기록임을 보존한다.

실측은 두 값 사이·아래·위·수치상 미분리로 위치를 기록할 수 있지만, 사이에 있어야 한다는 수용 조건을 만들지 않는다. \(\alpha_0\)를 실제로 측정하며

\[
\alpha_4=
\frac{n_0^X}{n_0^E}
\frac{(n_4/n_0)^X}{(n_4/n_0)^E}
\]

를 유지한다. 준위비의 배율을 절대 인구 배율로 치환하지 않는다.

부호 변화·source 대체·특이점 때문에 역산이 지원되지 않으면 원래 τ·S와 사유를 남긴다. 다른 세대 인구로 채우거나 양의 흡수 식을 연장하지 않는다.

### 5.4 공개 비교의 2차 기대

공개 CMFGEN의 **19.48 d** 열과 기존 적분 정의를 유지한다.

- NUV: 2500–4000 Å.
- OPT: 4000–6000 Å.
- UV2500: 2500–3000 Å.
- 여덟 대역: 2500–3000, 3000–3500, 3500–4000, 4000–5000, 5000–6000, 6000–7000, 7000–8000, 8000–10000 Å.

경계는 선형 보간, 적분은 사다리꼴 규칙이다. 중복 파장과 외삽을 허용하지 않는다.

\[
C_b^X=\frac{L_b^X}{L_{\rm OPT}^X},\qquad
d_b^X=\log_{10}\frac{C_b^X}{C_b^{\rm public}},
\qquad
R_8^X=\sqrt{\frac18\sum_b(d_b^X)^2}.
\]

등록 기준 중앙값은 다음과 같다. 실제 판정은 기대치 파일의 구간을 사용한다.

| 기대 | 기준 |
|---|---:|
| \(|d_{\rm NUV}^X|\) | ≤ 0.11551623544341516 |
| \(|d_{\rm UV2500}^X|\) | ≤ 0.3065384957773474 |
| 3000–3500·3500–4000·4000–5000·5000–6000 Å 각각의 \(|d_b^X|\) | ≤ 0.05 dex |
| \(R_8^X\) | < 0.2818388065975096 |

모든 조건이 생존해야 공개 비교의 2차 기대가 생존한다. 절대 대역 광도와 OPT 변화도 기록하여 정규화 분모의 효과를 드러낸다.

E 캡처 런과 E′N의 스펙트럼 연결은 직전 판정의 동일성 기록을 승계하고 신원을 확인한다. 그 연결이 성립하지 않으면 E에 대한 개입 비교와 E′N에서 등록한 공개 비교 기준을 구별한다. 임계값을 E의 새 측정값으로 재설정하지 않는다.

### 5.5 122건과 기대치 개정

직전 extension의 고정 122키에서 call-1 J̄, E′ 대비 이동, 등록 B-T 씨앗과의 편차를 요약한다. 이온별·셸별 결과와 결손을 보존하며 성공 사례만 다시 고르지 않는다.

이 보조 결과는 다섯 주 셸의 1차 판정을 대신하지 않는다. 새로운 황 기전·전선 위치·광이온화율의 검증으로 확대하지 않는다.

기대치를 바꿀 필요가 있으면 astra가 **OLD/NEW·변경 이유·영향받는 질문·이미 노출된 증거**를 적은 사전 개정을 작성한다. 해당 결과에 노출된 뒤 바꾼 기대는 이 런의 사전 예측으로 세지 않는다. 기대치 파일을 조용히 덮어쓰거나 사후 허용폭을 넓히지 않는다.

## §6. 수치 기준·과학 코드·실행 분기

### 6.1 수치 처리와 적격성

- W5 오차 상한, 텍스트 반올림, 생산 상수·연산 순서, 캡처 역산 오차를 반영한다.
- 기대치의 \([lo,mid,hi]\)를 점값으로 축소하지 않는다.
- 새로운 exp·log 계산은 독립 고정밀 참조로 확인한다.
- 영 분모·부호 미분리·비유한 결과를 정상값으로 채우지 않는다.
- 모형 오차를 부동소수점 오차 구간에 섞지 않는다.
- 불일치하는 세대·셸·행·바이너리·프로파일을 결합하지 않는다.

입력 연결이나 구현이 계약과 다르면 해당 판정의 적격성은 `X`다. 과학 가설의 반증 코드로 대신하지 않는다.

### 6.2 과학 코드

이번 결과 스키마는 `DET_SIIII_BT_SEED_RESULT_V1`이다. 다음 코드는 **각 과학 명제별**로 사용한다. 직전 계약의 숫자 코드 의미를 그대로 승계하지 않는다.

| 코드 | 명칭 | 조건 |
|---:|---|---|
| 0 | `SURVIVES` | 등록 조건을 유효 구간 전체가 만족 |
| 1 | `REFUTED` | 등록 조건 위반이 유효 구간으로 분리 |
| 4 | `UNRESOLVED` | 유효 구간이 판정 경계와 겹침 |
| 5 | `NOT_IDENTIFIABLE` | 필요한 입력·권위·정의가 부족 |
| X | 적격성 실패 | 잘못된 연결·구현·코드값·결과 구조 |

미실행 상태는 `FUTURE_TEST`로 기록하고 과학 코드를 부여하지 않는다. 2542 Å의 P1·P2 위치와 122건 이동 요약은 수치 기록이며 독립적인 성공 코드로 만들지 않는다.

결과에는 적어도 다음을 별도 보존한다.

- 다섯 셸 각각과 전체의 seed_delivery.
- 셸별 registered_material7_match.
- 셸별 reduced_model_applicability.
- 공개 비교 조건별 상태와 전체 상태.

전칭 조건은 확정 반례 하나로 코드 1이다. 반례가 없고 결손이 있으면 5, 결손도 없고 경계 미분리가 있으면 4, 전부 만족하면 0이다. 결손 때문에 이미 확인한 반례를 지우지 않는다.

### 6.3 실행 분기

| 조건 | 집행 |
|---|---|
| 오프라인 개입·물질·음성 대조 실패 | 제출하지 않고 등록 변경 범위 안에서 수리 |
| 빌드·입력·기대치 연결 실패 | 제출하지 않고 신원 연결 수리 |
| 선행 조건 충족 | 승인된 X 한 번 실행 |
| X가 적격하게 완료 | 1차·기전·2542 Å·공개 비교를 각각 판독 |
| X에서 1차 기대 반증 | 반증 보존, 2차 결과도 판독, 자동 재실행 없음 |
| X에서 1차 생존·2차 반증 | 씨앗 전달 성공과 공개 비교 실패를 함께 보고 |
| X가 불완전 종료 | 확보된 적격 증거의 범위만 판독하고 미완 항목 명시 |
| 판독기 결함 | 원시 자료를 보존하고 CPU 재판독 |
| 추가 생산 실행 필요 | 사용자 후속 결정으로 연결 |

## §7. 최소 변경집합·출력

### 변경집합 — DET-SIIII-BT-SEED (표)

새 구현 심볼의 접두사는 `det_siiii_btseed_`로 한다. T3의 기존 파일은 T2 사본에 대한 수정이며 봉인 T2 자체를 수정하지 않는다.

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/src/lumina_atomic.c` | 세대 0 초기화, det_siiii_btseed_* | 엄격한 노브 분기, 기존 프로파일 로더·검증 호출, 전 셸 온도 소유권, 출처 기록 |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/src/lumina_plasma.c` | lumina_publish_seed_te의 ON 경로 | 실제 배열·프로파일 일치 확인, bootstrap 발행 메타데이터·완전한 신원 기록 |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/src/env_universe.h` | 생성물 | 기존 생성기에 의한 LUMINA_BT_SEED 등록 |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/validation/instrumentation_debt/ENV_UNIVERSE.json` | 생성물 | 같은 생성 실행의 전집·소비 근거 기록 |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/tests/det_siiii_btseed_selftest.c` | CPU 시험 — 신설 | 실제 초기화·발행·물질·τ·방출 함수 시험, OFF 대조·시험용 결함 주입 |
| `scripts/det_siiii_btseed_readout.py` | 판독기 — 신설 | 입력 연결·시험 검증·스테이징·등록 기대 판독·과학 코드·비용 장부 |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/tools/run_siiii_btseed_offline.slurm` | CPU 러너 — 신설 | 생성·CPU 시험·B2 절차를 승계한 빌드·게이트·판독의 명령·시간·종료 기록 |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/tools/run_siiii_btseed_gpu.slurm` | GPU 러너 — 신설 | E 러너를 기반으로 X 한 번 실행, 신원·자원·시도 수·완료 확인 |

### 변경집합 끝

기존 derive_env_universe.py, 생산 헤더의 자료구조, 수송 소스, 기존 판독기와 기계 게이트는 변경하지 않는다. 이 집합으로 구현할 수 없으면 필요한 변경을 사전 개정으로 넘긴다.

X의 job.slurm은 등록 GPU 러너의 실행용 사본이다. 빌드 manifest·patch·입력 사본·로그·계약·판정·감사문은 집행 산출이다. 새 생산 기능이나 별도 판독 구현을 이 이름으로 표 밖에 추가하지 않는다.

기존 판독기의 필요한 순수 함수만 재사용할 수 있다. 실제 import 경로·함수·파일 신원을 기록하며, 기존 analyze 전체를 호출하여 이번 범위나 새 소스 신원 검사를 우회하지 않는다.

| 필수 판독 산출 | 내용 |
|---|---|
| offline.json | 실제 생산 함수 시험, OFF 대조, 주입 결함·기대/실제 사유·rc, 빌드 준비 상태 |
| seed_1206.csv | 다섯 셸 call 1–7, 등록 씨앗, 편차·오차 구간·이동 방향·세대 7 비교 |
| response_2542.csv | E′·X의 같은 세대 인구·α·τ·S·Q, 등록 P1·P2와 비교 |
| spectrum.csv | 절대 대역 광도·OPT 정규화·로그 잔차·R8·2차 기대 |
| extension.csv | 고정 122키의 call-1 이동·등록 씨앗 편차·결손 |
| result.json | 명제별 코드, 적격성, 선행 조건, 미완·반증·주장 한계 |
| manifest.json | 계약·기대치·입력·소스·빌드·실행·의존·출력 신원, argv·환경·비용 |

기대치 사본이 필요하면 원본 바이트와 해시를 보존한 읽기용 사본으로 둔다. 수정된 새 expectations.json을 결과물로 만들지 않는다.

## §8. 게이트·직접시험

### 8.1 내부 게이트

| 게이트 | 방지할 실패 | 통과 조건 |
|---|---|---|
| G-LINK | 기준판·프로파일·행·셸·세대·실행 신원 혼동 | §1·§4의 입력과 실제 출력 연결, 허용 변경집합 확인 |
| G-INTERVENTION | 온도만 바뀌거나 발행·물질이 낡은 상태 | §2·§3의 실제 생산 함수 시험·OFF 대조·음성 대조 통과 |
| G-READOUT | 기대치 교체·오차 누락·세대 대체·반증 은폐 | §5·§6의 실제 판독 경로 시험 통과 |

과학 기대가 반증된 것은 게이트 실패가 아니다. 게이트가 반증을 정확히 기록하지 못한 경우가 실패다.

### 8.2 판독기 직접시험

작은 생산 형식 가상 입력으로 실제 analyze를 호출하고 저장 산출을 검사한다.

- 다섯 셸 전부 기준 안, 한 셸만 기준 밖, 경계 중첩, 필수 행 결손.
- Jbar_err_upper를 제거하면 잘못 생존하는 경계 사례.
- call 1을 call 2로, 물질 7을 세대 8로, 셸·선 행을 서로 바꾼 입력.
- 같은 이름의 다른 프로파일·바이너리·기대치 파일.
- 1차 생존과 공개 비교 반증이 함께 존재하는 사례.
- 등록 세대 7 값과 새 배경 재계산값이 달라, 기대치 대체 결함을 검출하는 사례.
- \(n_0^X/n_0^E\ne1\)인 2542 Å 사례와 그 인자를 생략하는 결함.
- P1·P2 바깥의 실제 응답을 자동 실패로 처리하는 결함.
- OPT만 달라지는 색 변화, 일부 대역 개선과 R8 악화의 공존.
- 반례와 결손이 함께 있어도 전체 전칭 반증을 보존하는 사례.

새 시험은 이번 추가 연결과 분기에 집중한다. 기존 판독기의 전체 시험군을 정례 반복하지 않는다.

### 8.3 기계 프리플라이트

기존 도구를 수정 없이 사용한다.

- `scripts/check_prereg_preflight.py`
- `scripts/check_gate_referents.py`
- `scripts/check_doc_canonical.py`
- `scripts/check_dispatch_preflight.py`
- `scripts/check_driver_hygiene.py`
- `scripts/check_gate_roster.py`

계약·발주·러너의 실제 신원과 stdout·stderr·rc를 보존한다. 같은 문서 신원에 대한 dispatch 내부 prereg 검사는 중복 실행하지 않는다.

기계 검사는 과학식·생산 개입 검증·독립 사전감사를 대신하지 않는다.

## §9. 환경·비용·재현 명령

### 9.1 실행 환경

CPU 시험·빌드·게이트·판독은 a10 CPU Slurm에서 수행하며 GPU 예약은 0이다. 수동 실행 장소는 lageunha다. 로그인 노드는 파일 조회·소스 판독에만 사용한다.

GPU 생산 런은 Slurm job-per-run이다.

- h200 두 장 우선, 모델 시작 전 h100 두 장으로 대체 가능.
- a40·a100은 이번 판정런에 사용하지 않는다.
- 상주 러너·interactive 할당 금지.
- 작업명은 `LUMINA_` 접두.
- 기본 제외는 `--exclude=syn102`.
- 명시적 CPU·메모리·시간·GPU 요청과 `--no-requeue`.
- module purge·load와 버전 기록.
- 런 루트에 OWNER.txt.
- 계산·로그·작업 산출은 B 아래에 둔다.

src 편집은 동시에 하나만 수행한다. 원본 보존은 명시적 파일 목록·별도 결과 경로·내용 해시·덮어쓰기 거부로 구현한다.

### 9.2 비용 상한

E의 실측은 H200 두 장으로 10:02:57, 약 20.10 GPU·h다. 이번 상한은 성능 예측이 아니라 중단 기준이다.

| 항목 | 상한 |
|---|---:|
| 생산 모델 진입 | 1회 |
| GPU 수 | 2 |
| GPU 작업 경과 | 18시간 |
| GPU 예약량 | **36 GPU·h** |
| GPU 작업 CPU·메모리 | 16 CPU·64 GiB |
| CPU 전용 작업 동시성 | 1작업 |
| CPU 전용 작업 CPU·메모리 | 4 CPU·32 GiB |
| CPU 전용 누적 실행 경과 | **7200초** |
| CPU 전용 예약량 | **8 core·h 이하** |
| 새 입력 사본·빌드·런·판독·실패 기록 합계 | 64 GiB |

CPU 전용 경과 예산은 다음과 같이 배분한다.

| 용도 | 예산 |
|---|---:|
| 생산 빌드 초회와 제출 전 수리 빌드 1회 | 1800초 × 2 |
| CPU 개입·물질·음성 대조 시험 초회와 수리 3회 | 300초 × 4 |
| 판독 초회와 감사 수리 재판독 3회 | 400초 × 4 |
| 생성·스테이징·신원 확인·기계 게이트·보조 실행 | 누적 800초 |
| 합계 | 7200초 |

GPU 예산에 별도 GPU 예비 시험이나 재판정런을 넣지 않는다. GPU 작업의 호스트 CPU 예약은 최대 288 core·h이며 CPU 전용 8 core·h와 구별한다.

실패·수리·독립 보조 확인도 비용에 포함한다. Slurm 경과와 같은 작업의 러너 경과를 이중 합산하지 않고 큰 쪽을 집계한다. 대기·문서 작성·사람 감사 시간은 계산 시간과 분리한다.

lageunha에서는 Bash `time`을 사용한다. 시간을 기록하지 않은 수치 실행은 하지 않는다. 감사 수리 재판독 예산을 제출 전에 다른 계산으로 소진하지 않는다.

상한에 닿으면 완료한 증거와 미완 항목을 보고한다. 시간·용량 부족을 과학 가설의 반증으로 처리하지 않는다.

### 9.3 계획 CLI

다음은 구현할 인터페이스다. 실제 명령·작업 장소·출력 경로·rc를 저장한다.

```bash
python3 -B "$P/scripts/det_siiii_btseed_readout.py" selftest \
  --production-source "$T3" \
  --baseline-source "$T2" \
  --profile "$E/input/fixed_te_profile.txt" \
  --expectations "$A0/expectations.json" \
  --out "$B/results/$RUN_ID/selftest"
```

CPU 러너는 위 시험, env 생성, B2와 같은 nvcc 절차의 생산 빌드 및 제출 전 연결 검사를 수행한다. 실제 빌드 명령 전체를 B3에 보존하며 새 출력 디렉토리를 사용한다.

```bash
python3 -B "$P/scripts/det_siiii_btseed_readout.py" stage \
  --baseline-run "$E" \
  --production-source "$T3" \
  --build "$B3" \
  --offline "$B/results/$OFFLINE_ID" \
  --expectations "$A0/expectations.json" \
  --gpu-profile h200x2 \
  --run "$X"
```

stage는 제출하지 않는다. 등록된 조건을 검증하고 X의 구체적인 입력·러너·신원을 만든다. h100 대체 시 실행 전 gpu-profile과 자원 기록을 일치시킨다.

```bash
sbatch --exclude=syn102 \
  --output="$X/slurm-%j.out" \
  --error="$X/slurm-%j.err" \
  "$X/job.slurm" "$X"
```

```bash
python3 -B "$P/scripts/det_siiii_btseed_readout.py" analyze \
  --run "$X" \
  --baseline-run "$E" \
  --prior-analysis "$A0" \
  --production-source "$T3" \
  --public-spectrum "$P/data/standart_data1/toy06/spectra_toy06_cmfgen.txt" \
  --selftest "$B/results/$OFFLINE_ID/selftest" \
  --out "$B/results/$READOUT_ID/analysis"
```

후보·셸·반복·임계값을 결과에 맞춰 고르는 CLI는 만들지 않는다.

```bash
python3 -B "$P/scripts/check_prereg_preflight.py" \
  "$P/docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md" --root "$P"

python3 -B "$P/scripts/check_gate_referents.py" \
  "$P/docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md" --root "$P"

python3 -B "$P/scripts/check_doc_canonical.py" \
  "$P/docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md"

python3 -B "$P/scripts/check_dispatch_preflight.py" "$ORDER" --root "$P"
```

## §10. 처분·완료 조건·후속 결정

### 10.1 필수 완료 내용

판정문은 다음을 처음에 명시한다.

- 실제 생산 변경과 실행 횟수.
- 오프라인 요건 ②의 검증 범위.
- 다섯 셸의 1차 씨앗 전달 결과.
- 등록 세대 7 기대와 축약 모형 적용성의 구별.
- 공개 비교의 2차 결과와 실제 잔차 변화.
- 미완·결손·반증·기존 생산 해 재현의 한계.
- 생산 승격이나 물리적 정답을 판정하지 않았다는 점.

전체 완료에는 적격한 오프라인 증거, 한 번의 등록 실행, 필수 판독, 비용·보존 장부, 운전자 판정, astra 종료감사가 필요하다.

과학 기대의 성공은 완료 조건이 아니다. 기대가 반증돼도 등록 시험을 적격하게 수행하고 결과를 보존했다면 시험 목적을 달성할 수 있다.

### 10.2 집행 처분과 과학 결과

| 처분 | 기준 |
|---|---|
| `PASS` | 이번 전체 처분으로 사용하지 않음 |
| `PASS_LIMITED` | 등록된 개입 검증·판정런·필수 판독이 적격하게 완료되고 주장 상한을 지킴 |
| `REWORK` | 변경집합·입력 연결·시험·판독·분기·주장의 계약 위반 |
| `INCOMPLETE` | 필수 실행 또는 산출이 완료되지 않아 이번 목표 전체에 답하지 못함 |

전체 처분의 상한은 `PASS_LIMITED`다. `PASS_LIMITED`는 씨앗 전달·공개 비교의 생존을 뜻하지 않는다. 과학 결과 코드는 §6에 별도로 기록한다.

부분 실행에서 call 1만 적격하게 확보했다면 그 1차 결과는 보존하되, 세대 7·공개 비교까지 완료한 것으로 닫지 않는다.

### 10.3 결과별 다음 경로

| 결과 | 후속 판단 |
|---|---|
| 오프라인 개입 검증 실패 | 생산 제출 없이 구현·설계 원인 구분 |
| 온도·발행은 적격하나 씨앗 전달 반증 | INIT 수송의 원천·경계·혼합 문제를 다음 오프라인 질문으로 제시 |
| 씨앗 전달 생존, 등록 세대 7 외삽 반증 | 실제 배경 변화와 축약 모형 실패를 구별 |
| 씨앗 전달 생존, 공개 비교 반증 | 초기 씨앗 변경만으로 등록 스펙트럼 목표를 충족하지 못했다고 보고 |
| 1차·2차 모두 생존 | 이 한 번의 고정 온도·고정 반복 실험에서의 결과로 제한 |
| 필수 산출 결손 | 확보한 결과·미완 질문·추가 실행 필요성을 보고 |

어느 경로도 기본값 전환, 생산 승격, 추가 GPU 런, 반복 연장을 자동 승인하지 않는다.

## §11. 집행 기록·분장

| 업무 | 담당 | 규칙 |
|---|---|---|
| 기안·개정 | astra | 실제 모델 ID·사전 노출·OLD/NEW 기록 |
| 경량 사전감사 | Fable fresh | Q-GOAL·Q-LEAN 1회, 이번 계약 대상으로 시행 |
| 구현·직접시험·게이트·빌드·제출·판독·판정 | 운전자(Opus 5.5) | 구현자·판정자 겸임 명시 |
| 코드 감사 | 해제 | 정례 코드 감사를 추가하지 않음 |
| 종료감사 | astra | 계약 저자라는 이해관계 명시 |
| 실행 가능한 주장 재현 | 운전자 | 원시 자료 보존, CPU 재판독 비용에 포함 |
| 범위·비용 확대·추가 실행 | 사용자 | 이번 1회 승인과 구별 |

종료감사 지적은 개정18 ③에 따라 1·2회차 전건 반영, 3회차부터 «폐합 차단»만 반영한다. 나머지는 관측으로 기록한다. 이 규칙이 필수 과학 산출의 결손이나 비용 초과를 없애지는 않는다.

| 기안 시 항목 | 상태 |
|---|---|
| 공유 문맥·작업 장소·프로젝트 신원 | 확인 |
| 사용자 결정·직전 계약·판정·기대치·사전감사 | 직접 판독 |
| T2 초기화·발행·물질·FIXED_T·수송 경로 | 관련 소스 직접 판독 |
| 현재 lumina_atomic.c와 T2 대조 | 차이 없음 확인 |
| E 실행 설정·빌드 절차·W5·캡처 신원·로그 순서 | 관련 기록 직접 판독 |
| 생산 변경·파일화·수치 계산 | 미집행 |
| 이번 계약 기계 게이트·Fable 사전감사 | 미집행 |
| 직접시험·빌드·제출·판정·종료감사 | 미집행 |

기안자는 파일 수정·수치 분석·시험·게이트·빌드·제출을 수행하지 않았다. 미래 항목을 예상 PASS로 채우지 않는다.

## §12. 기계 선언

선언은 변경집합·참조 실체와 §6.2의 코드 분할을 검사한다. 온도 소유권·물질 일관성·수치 구간·과학 판정의 정확성은 직접시험과 종료감사 대상이다.

`v`는 판정할 과학 명제 하나의 코드다. 유효 코드는 0·1·4·5다. 결손·복수 코드·그 밖의 값은 `X`다. 미실행 기대·집행 처분·P1/P2 위치를 이 코드에 섞지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SIIII-BT-SEED (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/(?:source/src/(?:lumina_atomic\\.c|lumina_plasma\\.c|env_universe\\.h)|source/validation/instrumentation_debt/ENV_UNIVERSE\\.json|source/tests/det_siiii_btseed_selftest\\.c|tools/run_siiii_btseed_(?:offline|gpu)\\.slurm)|scripts/det_siiii_btseed_readout\\.py)",
    "symbol": "det_siiii_btseed_",
    "roots": [
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/src/lumina_atomic.c",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/src/lumina_plasma.c",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/src/env_universe.h",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/validation/instrumentation_debt/ENV_UNIVERSE.json",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/tests/det_siiii_btseed_selftest.c",
      "scripts/det_siiii_btseed_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/tools/run_siiii_btseed_offline.slurm",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/tools/run_siiii_btseed_gpu.slurm"
    ],
    "expected_extra": [
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/src/lumina_atomic.c",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/src/lumina_plasma.c",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/src/env_universe.h",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/validation/instrumentation_debt/ENV_UNIVERSE.json",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/source/tests/det_siiii_btseed_selftest.c",
      "scripts/det_siiii_btseed_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/tools/run_siiii_btseed_offline.slurm",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/tools/run_siiii_btseed_gpu.slurm"
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
      {"name": "prior-code-three-rejected", "mix": [[3,1]]},
      {"name": "invalid-nine", "mix": [[9,1]]},
      {"name": "missing-code", "mix": []},
      {"name": "multiple-codes", "mix": [[0,1],[1,1]]}
    ]
  },
  "references": [
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/CLAUDE.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/evidence/USER_DECISION_2026-10-07.txt"
    },
    {
      "path": "docs/PREREG_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md"
    },
    {
      "path": "docs/VERDICT_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md"
    },
    {
      "path": "docs/PREREG_DET_OBS_CAPTURE_2026-10-06.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/fable_preaudit.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/results/t5/analysis/expectations.json",
      "flags_existing": [
        "B-T",
        "primary_expectation",
        "mechanism_expectation",
        "2542_boundary_records",
        "secondary_spectral_expectation"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/results/t5/analysis/manifest.json"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/results/t5/analysis/extension.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c",
      "flags_existing": [
        "config_prec_parse_switch",
        "Generation-zero material-temperature seed",
        "opacity->t_electrons"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cuda.cu",
      "flags_existing": [
        "lumina_publish_seed_te",
        "lumina_prepare_solver_owned_tau",
        "opacity.t_electrons"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c",
      "flags_existing": [
        "lumina_publish_seed_te",
        "a210_fixed_te_profile_load",
        "a210_production_solve",
        "lumina_line_upper_population_for_tau"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/radeq_publication.h",
      "flags_existing": [
        "a210_fixed_te_profile_validate",
        "A210_PRODUCTION_TE_MIN_K",
        "A210_PRODUCTION_TE_MAX_K"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c",
      "flags_existing": [
        "cmf_fine_line_material",
        "cmf_fine_exact_owner_solve",
        "INIT_SHARED_GAUSSIAN",
        "EXIT_AFTER_R6_REQUIRES_SINGLE_ITER_MGPU_AB"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/impl/build_b2.sh"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1/STAGE_IDENTITY.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1/job.slurm"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1/SUBMIT_COMMAND.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1/input/resolved_lumina.exports",
      "flags_existing": [
        "LUMINA_FIXED_TE_PROFILE",
        "LUMINA_PURE_CMFGEN_ITER",
        "LUMINA_DET_OBS_CAPTURE"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1/input/run_build_identity.json"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1/input/fixed_te_profile.txt"
    },
    {
      "path": "scripts/derive_env_universe.py",
      "flags_existing": [
        "emit_header",
        "ENV_UNIVERSE.json",
        "config_prec_parse_switch"
      ]
    },
    {
      "path": "scripts/det_siiii_repair_readout.py"
    },
    {
      "path": "scripts/det_siiii_1206mech_readout.py",
      "flags_existing": [
        "det_1206mech_profile_mean",
        "det_1206mech_r_off",
        "det_1206mech_F"
      ]
    },
    {
      "path": "scripts/det_siiii_pop_readout.py",
      "flags_existing": ["recover", "capture_records"]
    },
    {
      "path": "scripts/det_siiii_rates_readout.py",
      "flags_existing": ["det_siiii_rates_coll"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/figs/cmp_pub_paired.py",
      "flags_existing": ["band_integral"]
    },
    {
      "path": "data/standart_data1/toy06/spectra_toy06_cmfgen.txt"
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
