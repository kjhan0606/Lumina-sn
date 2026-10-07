# 사전등록 — DET-SIIII-REPAIR-OFFLINE: 초기 씨앗·NRB 수리 후보의 오프라인 검증과 판정런 기대치

2026-10-07 · 기안 · read-only  
운전자 파일화 대상: `docs/PREREG_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md` — 신설  
상태: **Fable fresh 사전감사와 후속 사용자 결정을 반영하는 결과 판독 전 개정 1. 판독기 미구현·실행 0회이며, 개정 본문의 기계 게이트·직접시험·종료감사는 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답과 과학 질문

### Q-GOAL — 최종목적에 직접 기여하는가?

최종목적은 Monte-Carlo 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현하는 것이다. 직전 단은 Si III 1206.50 Å에서 E′의 초기 복사장 유지와 느린 자기 되먹임을 특정했지만, 그 고정을 푸는 NRB 상태 R′는 공개 CMFGEN에 대한 NUV/OPT 근접도를 오히려 악화시켰다. 승인 시점의 실질 후보는 NRB(R′)였으나, 초기 원천을 특정한 뒤 B-T(초기 물질에 FIXED_T 일관 적용)로 옮겨 갔다. 운전자는 이 경위와 B-T 씨앗 전달의 오프라인 미검증을 사용자에게 보고했고, 사용자는 「축소 수용」과 「다음 판정런의 1차 기대로」를 결정했다. **B-T는 씨앗 전달 미검증으로 런 발주 요건 ②가 이 단 뒤에도 열린다. 사용자 결정에 따라 다음 판정런의 1차 기대는 발행 call-1 J̄가 지정 씨앗 𝓑(T_pin)에 상대 편차 1e-3 이내로 접근하는 것이며, 스펙트럼 로그 잔차 반감 기대는 2차다.**

이번 목표는 **A의 기존 실측을 재인용하고 B-T의 조건부 씨앗 이동·민감도와 남는 의존성을 구별하며, 2542 Å의 경계 수치와 다음 판정런의 기대치를 등록하여 후속 사용자 결정의 입력을 제공하는 것**이다. B-T의 씨앗 전달 검증이나 판정런 발주 요건 ②의 완결을 이번 산출로 약속하지 않는다.

질문은 다음과 같다.

1. A(NRB)의 기존 실측 이탈과 B-T(INIT 정정)의 조건부 이동은 어떻게 다른가? B-T의 구성상 남는 씨앗 기억과 기존 \(T_{\rm inner}\) 상태에서의 이동을 어떻게 구별하는가?
2. B의 전제인 INIT 형성에서 초기 물질 온도, 선 원천, 전달 초기 추정값, 내경계의 역할을 어디까지 저장자료와 소스로 특정할 수 있는가?
3. 후보별 \(n_4/n_0\) 변화 중 무엇을 오프라인 대수로 예측할 수 있는가? \(n_5\), 2542 Å의 \(S_{\rm used}\)·τ, 최종 UV2500에는 어떤 미지수가 남는가?
4. 선행 122 선·셸, 특히 S III 67건의 초기 원천 양립성과 B-T 씨앗 이동·기억 민감도는 어떠한가?
5. 공개 CMFGEN 기준에서 다음 한 번의 판정런이 무엇을 만족하거나 반증해야 하는가?

직접 줄이려는 어긋남은 E′N의 2500–3000 Å 과잉 약 +0.61 dex와 NUV/OPT 과잉 약 +0.23 dex다. 동시에 이미 공개값에 가까운 3000–6000 Å 네 대역을 보존해야 한다.

이번 산출은 조건부 인구비·국소 주입 응답과 반증 범위라는 물리 분석이다. **이번 단의 스펙트럼 잔차 감소는 0**이다. 스펙트럼 개선을 직접 확인하는 대안은 판정런이지만, 후보의 개입 위치와 기대치가 미정인 상태에서 실행하면 offline-first 요건을 충족하지 못한다.

생략하면 초기값을 옮긴 것을 고정 해제로 오인하거나, 준위 4 감소를 UV2500 감소로 바로 치환하거나, FUV 감소를 공개 CMFGEN 일치로 오인하게 된다.

**기안자 자기답: PASS.**

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

다음 범위로 묶는다.

- 1206.50 Å: 기존 다섯 셸의 이력·캡처와 수리 후보의 조건부 대수를 연결한다.
- INIT: 초기 온도와 물질·복사 생성 순서를 소스로 추적하고, 인구비에서 선 원천을 계산한다. 전체 Saha 계산이나 INIT 전달 replay는 하지 않는다.
- 2542.58 Å: 기존 캡처처에서 두 상태의 인구·τ·S를 읽고, P1·P2의 경계 수치를 기록한다.
- 확장: **선행에 이미 특정된 122 선·셸 전부**에 초기 원천 비교와 B-T 씨앗 이동·기억 민감도만 적용한다. 그 안의 성공 사례를 다시 골라 모집단을 축소하지 않는다.
- 공개 비교: 기존 cmp_pub_paired.py와 같은 정의를 판독기 안의 작은 함수로 사용하여 반올림 전 기준값을 보존한다. 새 스펙트럼이나 광도 분해를 만들지 않는다.
- 새 판독기와 러너만 만든다. 내부 검사는 입력 연결과 예측 대수·분기를 확인하는 두 게이트로 묶는다.

122건을 모두 유지하는 이유는 S III 67건을 포함한 초기 원천 비교와 B-T 씨앗 이동·기억 민감도의 범위를 얻기 위해서다. 필요한 원자계수·call 1·등록 배경만 판독하며, 확장을 위한 캡처 역산과 선행 기전 시험의 재현은 하지 않는다. 다른 이온의 전 준위 율 예산·충돌망·광이온화율 계산은 추가하지 않는다.

소비자는 운전자와 사용자다. 후보표는 다음 실행의 필요성을, INIT 결과는 개입 위치를, 2542 Å 결과는 스펙트럼 기대의 한계를, 황 결과는 확장 주장의 범위를 결정한다.

**기안자 자기답: APPROPRIATE.**

### 승인 근거·분장

승인 근거는 다음 결정문이다.

`/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/evidence/USER_DECISION_2026-10-07.txt`

최초 승인은 DET-SIIII-1206-MECH §10.3 코드 0 경로의 수리안 오프라인 검증 설계와 122건 부분집합의 기전 시험 확장을 대상으로 했다. 같은 결정문의 마지막 항에 따라 이번 범위는 후보 A·B-T와 B-W 대조, 초기 원천 비교·B-T 씨앗 이동·기억 민감도로 축소하며, 다음 판정런의 1차 기대는 B-T 씨앗 전달로 둔다. 이번 발주의 구체적 요구가 개정 범위를 정한다.

사용자 결정 개정25-aa의 사전감사 생략은 운전자 모델이 Fable인 동안에만 적용한다. 이번 운전자는 **Opus 5.5**이므로 **Fable fresh 경량 사전감사(Q-GOAL·Q-LEAN) 1회**를 둔다. 기안·기계 게이트 후, 구현 발주 전에 수행한다. Fable 호출 실패 시 **gpt-6.1-sol**이 대행하며, 실패 증거와 실제 응답 모델 ID를 보존한다. 기안자 자기답이나 운전자의 기계 게이트를 독립 사전감사로 기록하지 않는다.

- 기안·개정·종료감사: astra.
- 경량 사전감사: Fable fresh. 호출 실패 시 gpt-6.1-sol 대행.
- 구현·직접시험·게이트·실행·판독·판정: 운전자(Opus 5.5).
- 코드 감사: 해제.
- 남는 독립 검사는 **Fable 사전감사와 astra 종료감사**다. 사전감사 대행 시 실제 담당 모델을 기록한다. astra가 계약 저자라는 이해관계는 남는다.

### 사전 노출·주장 상한

본 계약은 선행 실측을 읽은 뒤 등록한다. 기존 E′·R′의 재현은 새 자료에 대한 예측 성공이 아니다. 새 후보 계산의 식·조건·허용폭·선택 규칙은 이번 분석 전에 고정한다.

판정하지 않는 것은 다음과 같다.

- 어느 후보가 물리적으로 옳은 수리인지.
- NRB 또는 INIT 변경의 생산 승격.
- 새 생산 해의 수렴성과 재현.
- 실행하지 않은 판정런의 성공.
- 선 개수 비율을 이용한 황 광이온화·에너지 기여율.
- 공개 스펙트럼이 없는 λ<2500 Å의 CMFGEN 일치.

## §1. 경로·기준선·승계 사실

### 1.1 경로

```text
P  = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
C0 = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
T2 = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source
E  = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1
R  = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1
M  = /gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007
N  = /gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002
L  = /gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007
D  = /gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007

선행 기전 정본 = M/results/t7/analysis
공개 비교의 E′N = N/Eprime/run1/work/lumina_spectrum_det_obs.csv
공개 비교의 R′N = N/Rprime/run2/work/lumina_spectrum_det_obs.csv
공개 스펙트럼 = P/data/standart_data1/toy06/spectra_toy06_cmfgen.txt

새 러너 = D/tools/run_siiii_repair.slurm
새 산출 = D/results/<실행 식별자>
새 집행 증거 = D/evidence/<집행 식별자>
```

D가 승인문 때문에 이미 존재하는 것은 허용한다. 새 결과 경로는 기존 결과를 덮어쓰지 않는다.

### 1.2 기준선·신원

- 저장소 확인 기준 HEAD: `0b73f4c3eb84367122d4b21c363740d3351a8aaa`.
- 직전 계약: `docs/PREREG_DET_SIIII_1206_MECH_2026-10-07.md`.
- 직전 판정: `docs/VERDICT_DET_SIIII_1206_MECH_2026-10-07.md`.
- 직전 종료감사 정본: M의 astra_closeout3.md.
- 직전 실행 신원: M/results/t7/identity.sha256 및 analysis/manifest.json.
- 생산식 권위: T2와 E·R의 run_build_identity.json. 현재 작업 트리의 생산 소스로 대체하지 않는다.

기존 판독기의 필요한 순수 보조 함수만 import할 수 있다. 실제 import 경로·함수·파일 신원을 새 manifest에 기록한다.

선행 신원 장부의 값은 다음과 같다.

```text
det_siiii_pop_readout.py
63cbea718eec73a84b21e4cf907e89934b5cbf2011026834cd50cc8e686976be

det_siiii_rates_readout.py
2fb17ef88356cbb5841463c2f2de6c5674d140e332be0474bb8c8b5330229dfd
```

1206-MECH 판독기는 t7 실행 뒤 시험명 정정 이력이 있으므로 실행 당시 신원과 현재 import 신원을 별도로 기록한다. 이름이 같다는 이유로 t7 실행 파일과 동일하다고 선언하지 않는다. 기존 analyze 전체를 호출하여 새 표적·셸·세대 선택을 우회하지 않는다.

### 1.3 승계 사실

1. E′의 1206.50 Å는 다섯 셸에서 희석 없는 내경계 흑체에 가까운 call 1 값을 call 1–7 동안 유지했다.
2. 변화율 설명에는 생산 \(\kappa\)가 필요하다. β만의 이상화는 반증됐다.
3. E′의 세대 7 \(n_4/n_0\)는 \(T_{\rm exc}\simeq10019.96\) K로 표현된다. R′는 같은 씨앗에서 이탈했으며 셸 10의 이탈은 상승이었다.
4. 122건은 **씨앗 유사성과 유지가 함께 관측된 선·셸 집합**이다. 다른 전이의 고립 두 준위 기전이 이미 검증된 집합은 아니다.
5. 2542.58 Å는 UV2500의 순 mapped E′−R′ 차이에서 약 0.827을 차지했다. 공개 스펙트럼 잔차의 82.7%를 설명한다는 뜻이 아니다.
6. 세대 7 생산 해의 재현 불가와 공개 CMFGEN 준위 인구 부재를 승계한다.
7. 과거 판정·감사 저자와 처분은 보존한다.

### 1.4 입력 권위

| 입력 | 사용 |
|---|---|
| E·R의 W5 call 1–7 | 실제 초기값·이력, 후속 β·\(J_c\)·\(L_l\), operator와 세대 |
| E·R의 캡처 lines.bin | 같은 세대의 τ·\(S_{\rm used}\), 상·하 인구 역산 |
| 캡처 manifest·meta·Te·ne | 소비 권위, 세대, 국소 조건, fallback 여부 |
| 봉인 line_list·levels | 데이터 행, 끝점, \(A,B,f,\nu,\lambda,g,E\) |
| geometry·config·fixed_te_profile | 기하 희석 비교값, 내경계 온도, 등록된 고정 온도 |
| resolved_lumina.exports·run_build_identity | 실제 분기·생산 판본 |
| E·R의 stdout·stderr | bootstrap·INIT·seed predictor·FIXED_T 순서 |
| M의 정본 산출 | 122건 고정 키와 선행 수치·상태 |
| N의 짝 스펙트럼·공개 스펙트럼 | 공개 비교 기준과 기대치의 수치화 |

세대 8 인구는 이번 필수 계산에 사용하지 않는다. 세대 7의 빈 칸을 채우는 대체 자료로도 사용하지 않는다.

공개 비교의 N 런과 캡처 E·R 런은 경로가 다르다. 캡처 런의 저장 스펙트럼과 N의 대응 스펙트럼을 대조하고, 입력·생산 판본 연결을 기록한다. 수치가 다르면 차이를 공개하고 두 기준선을 구별한다. N에서 측정한 값을 캡처 런에서 새로 측정한 값으로 표기하지 않는다.

### 1.5 기안자가 직접 확인한 INIT 경로

| 근거 | 확인 내용 |
|---|---|
| [lumina_atomic.c](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c)의 약 1057행, 초기 물질 온도 | 모든 셸의 t_electrons를 config의 T_inner로 설정 |
| [lumina_cuda.cu](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cuda.cu)의 약 7156행, plasma 초기화 | 위 배열을 plasma.T_e에 복사 |
| [lumina_plasma.c](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c)의 약 8205행, 초기 온도 발행 | bootstrap 온도를 세대 1로 발행 |
| [lumina_plasma.c](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c)의 약 10245행, 초기 선 인구 | 미커밋 NLTE 상태에서는 LTE 물질 인구를 사용 |
| [lumina_cmfgen.c](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c)의 약 7630행, 미세 선 물질 | 인구·τ에서 선 흡수·방출을 구성 |
| [lumina_cmfgen.c](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c)의 약 9048행, warm start | \(B_\nu(T_e)\)로 J의 초기 추정값 설정 |
| [lumina_cmfgen.c](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c)의 약 9263행, 정확 전달해 호출 | T_inner·inner_bb_scale과 물질 배열을 전달 |
| [lumina_cmfgen.c](/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c)의 약 11770행, INIT 발행 순서 | INIT 복사 발행 후 seed material predictor 수행 |

따라서 **call 1에서 warm start의 국소 \(T_e\)와 \(T_{\rm inner}\)가 다르다는 전제는 성립하지 않는다.** T2의 bootstrap 온도는 모든 셸에서 \(T_{\rm inner}\)다. 선행에 제시된 8913–10255 K는 후속 고정 온도이며 INIT 물질 온도로 소급할 수 없다.

이 소스 연결만으로 실제 INIT 전달해의 국소 선 방출·경계·산란 기여율까지 유일하게 특정한 것은 아니다.

## §2. 표적·수리 후보·고정 범위

### 2.1 표적

| 용도 | 전이·행 | 셸 |
|---|---|---|
| 주 기전·후보 예측 | Si III 1206.50 Å, 0→4, 행 600693 | 10·15·19·24·30 |
| UV 연결의 주 비교 | Si III 2542.58 Å, 4→5, 행 1097363 | 19 |
| UV 연결의 공간 확인 | 같은 2542.58 Å | 다섯 W5 셸 및 선행 주요 셸 17–22 |
| 확장 | 선행 122 선·셸 | 선행 키 그대로 |

행 번호는 헤더를 제외한 0-based 데이터 행이다. 별도 line_id 열로 바꾸지 않는다.

2542 Å의 공간 확인에 필요한 같은 셸의 1206 Å 캡처도 읽을 수 있다. 그러나 W5에 없는 셸의 반복 이력을 보간하거나 복제하지 않는다. 정량 이력 예측의 주 범위는 다섯 W5 셸이다.

### 2.2 고정 확장 집합

M의 정본 census에서 다음 조건을 함께 만족한 키를 고정한다.

- E′ call 2 기준 \(\tau_{\rm eff}>100\).
- λ<2000 Å.
- E′의 씨앗 유사성과 call 1–7 유지가 모두 `SURVIVES`.

승계 키 수는 122건이며 이온별로 S III 67·Si III 32·Ca III 9·S II 8·Si II 6이다. 이는 **선 개수가 아니라 선·셸 쌍의 수**다.

이번 결과에서 반증·결손이 발생해도 그 키를 분모에서 제거하지 않는다. 122건의 구성 불일치는 입력 연결 문제로 처리하며 새 선별로 수를 맞추지 않는다.

### 2.3 후보의 정확한 의미

| 후보·대조 | 오프라인 정의 | 생산 개입에 관한 한계 |
|---|---|---|
| A — 후보 | 기존 R′의 NRB armed 계수와 실측 재인용 | 이미 측정된 유한 응답을 승계하며 새 효과로 세지 않음 |
| B-T — 후보 | 초기 물질에 FIXED_T를 일관 적용하는 제안과, NRB off의 유효 INIT 선 씨앗을 \(\mathcal B_\ell(T_{{\rm pin},s})\)로 둔 조건부 모형 | T_pin은 봉인 fixed_te_profile. 실제 씨앗 전달은 미검증 |
| B-W — B-T 감도·대조 열 | NRB off의 유효 INIT 선 씨앗을 \(W_s\mathcal B_\ell(T_{\rm inner})\)로 둔 조건부 모형 | 기하 희석을 두꺼운 선의 정답으로 가정하지 않으며 후보로 세지 않음 |

고정 배경에서 C ≡ A는 구성상 확정이므로 C-W·C-T 열과 별도 판정을 두지 않는다.

여기서 B는 먼저 **물질 갱신이 소비하는 유효 선 씨앗에 대한 반사실 계산**으로 정의한다.

B-T에 대응하는 생산 수리 제안은 고정 온도 프로파일을 bootstrap 물질 생성 전부터 일관되게 사용하도록 초기화 순서를 정정하는 것이다. 이는 온도 배열 한 칸이나 `fs.J`만 바꾸는 제안과 다르다. 인구·분배함수·τ·방출·발행 신원의 일관성이 필요하다. 이 단에서는 그 생산 변경을 구현하지 않는다.

B-W에는 현재 지정 자료만으로 일관된 물질·전달 개입이 정해져 있지 않다. B-W는 B-T의 씨앗 민감도 대조 열이며 후보가 아니다. 두꺼운 선에 \(W B\)를 강제하는 것을 물리 수리로 승격하지 않는다.

### 2.4 고정 해제의 두 뜻

다음을 별도로 판정한다.

1. **기존 \(T_{\rm inner}\) 상태 이탈:** 기존 씨앗 또는 기존 여기비와 달라짐.
2. **씨앗 기억 제거:** 초기 씨앗을 달리해도 후속 상태가 초기값 차이를 충분히 잊음.

B가 다른 온도나 희석값에 고정되는 것은 첫째의 성공일 수 있지만 둘째의 성공은 아니다. A의 국소 계수에서 직접 자기 펌프가 사라지는 것 또한 전체 다준위 해의 초기값 독립성을 증명하지 않는다.

## §3. INIT 전제의 오프라인 특정

### 3.1 이번 단에서 특정할 범위

INIT 문제를 별도 선행 단으로 분리하지 않는다. 이번 단에서 다음을 검증한다.

- 초기 물질 온도의 실제 공급자와 값.
- 그 온도를 소비한 LTE 인구·τ·선 방출의 연결.
- warm start와 전달 경계의 구별.
- INIT 복사 발행, seed material predictor, 고정 온도 적용의 순서.
- 위 경로에서 예상되는 초기 선 원천과 저장 call 1의 양립성.

**INIT 전달해의 원천별 기여율이나 유일한 원인은 계산하지 않는다.** 초기 전달계수·장 배열이 없으면 그 결손을 명시한다. call 2 또는 세대 7의 τ·연속체로 INIT 배열을 대신하지 않는다.

### 3.2 초기 원천의 정량 예측

초기 LTE 전이비를

\[
r_{\rm LTE}(T)=\frac{g_u}{g_l}
\exp\!\left[-\frac{E_u-E_l}{kT}\right]
\]

로 둔다. 생산 상수와 저장 원자계수를 보존하여

\[
a=\frac{g_l}{g_u},\qquad
k_\tau=C_{\rm Sob}f_{lu}\lambda\frac{\nu}{c},
\qquad
K_\nu=\frac{h\nu A_{ul}}{4\pi k_\tau}
\]

를 정의한다. 양의 흡수에서 초기 선의 방출/흡수 비는

\[
S_{\ell,\rm boot}(T)
=K_\nu\frac{r_{\rm LTE}(T)}
{1-a r_{\rm LTE}(T)}
\]

다.

같은 이온 총수와 분배함수는 이 비에서 소거된다. 따라서 전체 초기 Saha 해를 다시 계산할 필요가 없다.

다음을 출력한다.

- \(T_{\rm boot}\), \(T_{\rm inner}\), \(T_{\rm pin}\).
- \(r_{\rm LTE}(T_{\rm boot})\).
- \(S_{\ell,\rm boot}(T_{\rm boot})\).
- 생산 흑체 프로파일 평균 \(\mathcal B_\ell(T_{\rm boot})\).
- 실제 call 1 J̄.
- 위 값들의 비·차와 수치 구간.

직렬화된 \(\nu,\lambda,E,A,B,f\)의 관계를 강제로 보정하지 않는다. 원천이 Planck 함수와 정확히 같다고 대입하지 않는다.

### 3.3 등록 가설과 반증

| 가설 | 생존 조건 | 반증 조건 |
|---|---|---|
| I-T 초기 온도 공급 | 봉인 판본·실제 설정·실행 순서가 모든 대상 셸의 \(T_{\rm boot}=T_{\rm inner}\)를 지지 | call 1 이전 다른 온도 공급 또는 덮어쓰기 확인 |
| I-P 초기 선 인구 | INIT의 τ와 상위 인구가 같은 bootstrap LTE 물질에서 옴 | 서로 다른 온도·인구 권위, 선별 다른 공급자 확인 |
| I-S 원천 양립성 | 1206 Å에서 \(S_{\ell,\rm boot}/\mathcal B_\ell-1\)의 절댓값이 \(10^{-3}\) 이하 | 기준 밖으로 분리 |
| I-J 저장 초기값 양립성 | call 1과 해당 초기 원천·흑체 비교의 상대 편차가 각각 \(10^{-3}\) 이하 | 기준 밖으로 분리 |

I-T·I-P는 소스·설정·실행 증거로 판정한다. 문자열 존재만으로 생존 처리하지 않는다.

I-S·I-J가 생존해도 “국소 선 방출만으로 INIT 장이 만들어졌다”는 결론은 허용하지 않는다. 초기 물질·warm start·경계가 같은 온도를 공유하여 구별되지 않는 부분은 `ORIGIN_NOT_SEPARATED`로 남긴다.

확장 122건에는 같은 원천 비교를 적용하되, 실패를 1206 Å의 성공으로 덮지 않는다.

### 3.4 B에 대한 귀결

- `fs.J` 초기 추정값만 바꾸면 정확 전달해의 경계·물질·방정식은 그대로다. **warm start 변경만으로 최종 call 1 씨앗이 바뀐다고 예측하지 않는다.**
- B-T의 후속 시험은 물질 초기화의 온도 일관성을 바꾸는 개입과, 실제 발행된 call 1이 예상 씨앗으로 이동했는지를 함께 확인해야 한다.
- 기하 희석 \(W B\)는 투명한 경계장 비교값이다. 두꺼운 선의 국소 원천을 반드시 그 값으로 만들어야 한다는 근거가 아니다.
- 초기 원천의 온도 경로를 특정한 결과와, B가 실제 전달 출력에 원하는 씨앗을 만드는지를 검증한 결과는 구별한다.

B의 생산 개입과 발행 씨앗 사이의 연결이 미검증이면 **조건부 씨앗 계산만 완료된 것**이다. 이를 런 발주 요건 ②의 무조건 충족으로 표기하지 않는다.

## §4. 후보별 1206 Å 예측과 122건 확장

### 4.1 씨앗의 정의

생산 Gaussian 프로파일·격자·가중치를 사용한다.

\[
\mathcal B_\ell(T)
=b\frac{\sum_{i\in Q_\ell}w_i B_{\nu_i}(T)}
{\sum_{i\in Q_\ell}w_i},
\quad
w_i=e^{-x_i^2}\Delta\nu_i,\quad |x_i|\le4.
\]

내경계 비교에는 실제 inner_bb_scale을 사용한다. 국소 열적 B-T에는 국소 Planck 원천의 정의를 사용하며 내경계 배율을 임의로 곱하지 않는다.

\[
r_s=\frac{r_{{\rm inner},s}+r_{{\rm outer},s}}2,\qquad
R_{\rm in}=r_{{\rm inner},0},
\]

\[
W_s=\frac12\left[1-\sqrt{1-(R_{\rm in}/r_s)^2}\right].
\]

계산에는 geometry의 저장값을 사용한다. 발주의 반올림 W 값은 입력으로 쓰지 않는다.

\[
J_{1}^{B-W}=W_s\mathcal B_\ell(T_{\rm inner}),\qquad
J_{1}^{B-T}=\mathcal B_\ell(T_{{\rm pin},s}).
\]

### 4.2 방사 고립 전이의 비와 off 사상

\[
r_{\rm off}(J)=\frac{B_{lu}J}{A_{ul}+B_{ul}J},
\qquad
r_{\rm A}(J_c)=\frac{B_{lu}J_c}{A_{ul}+B_{ul}J_c}.
\]

\[
\kappa=\frac{h\nu B_{lu}/4\pi}{k_\tau},
\qquad
\Delta B=B_{ul}-aB_{lu}.
\]

\[
F_k(J)=\beta_{k+1}J_{c,k+1}
+(1-\beta_{k+1})
\frac{\kappa A_{ul}J}{A_{ul}+\Delta B J},
\]

\[
q_k(J)=
(1-\beta_{k+1})
\frac{\kappa A_{ul}^{\,2}}
{(A_{ul}+\Delta B J)^2}.
\]

기준선 재현은 실제 \(J_k\)를 넣는 조건부 한 단계 시험이다. B의 계산은 등록된 대체 \(J_1\)에서 시작하여 이 사상을 call 7까지 적용한다.

이때 β·\(J_c\)는 **E′의 저장 이력으로 고정한 배경**이다. B가 만들 실제 τ·연속체의 예보가 아니다. 동일하게 R′ 배경에서의 민감도도 계산하여 두 배경의 차이를 공개한다. 두 결과의 범위를 실제 생산 해의 보장 구간으로 부르지 않는다.

\(\kappa=1\), \(\Delta B=0\)으로 강제하거나 관측 \(L_{k+1}\)을 예측식에 넣지 않는다.

### 4.3 충돌과 다준위 한계

1206 Å에는 세대 7의 실제 \(T_e,n_e\)와 생산 충돌 계수를 사용한 보조 비교를 둔다.

\[
r_{\rm off+C}(J)=
\frac{B_{lu}J+C_{lu}}
{A_{ul}+B_{ul}J+C_{ul}},
\]

\[
r_{\rm A+C}(J_c,\beta)=
\frac{\beta B_{lu}J_c+C_{lu}}
{\beta(A_{ul}+B_{ul}J_c)+C_{ul}}.
\]

이는 같은 물질 배경에서 고립 전이의 방사·충돌 경쟁을 보는 계산이다.

- armed에서 방사 항만의 비는 β가 소거되지만, **충돌이 있으면 전체 비에서 β를 소거할 수 없다.**
- 세대 7 충돌 계수를 call 1–7 전체의 실제 계수로 대입하지 않는다.
- 방사 전용 예측과 방사·충돌 예측의 상대 차이가 \(10^{-3}\)를 넘으면 해당 조건에서 방사 전용 정밀 예측의 적용성을 반증한다.
- 다준위 교환과 이온 총수 변화는 남는다. 고립 전이비를 생산 평형해로 표시하지 않는다.

확장 122건에는 이온별 충돌망을 새로 이식하지 않는다. 그 결과는 §4.6의 초기 원천 비교와 B-T 씨앗 이동·기억 민감도로 제한한다.

### 4.4 등록할 수치와 예상 방향

다섯 셸마다 다음을 저장한다.

- 후보 A의 실제 \(r_7^{R'}\) 재인용과 방사·방사충돌 조건부 비.
- 후보 B-T의 \(J_1\), \(r_{\rm off}(J_1)\), 조건부 call 7 예측.
- B-T의 감도·대조 열 B-W에 같은 양을 기록.
- A 재인용값과 B-T·B-W 조건부 값의 E′ 대비 비, 로그 변화, 여기온도.
- 구성상 남는 씨앗 기억과 배경 민감도.

다음 값은 선행 결과와 반올림 W를 이용한 **규모 안내**이며, 새 분석의 정답이나 허용폭이 아니다.

| 셸 | E′ \(n_4/n_0\) | B-W 대조의 방사 씨앗비 규모 | B-T 후보의 방사 씨앗비 규모 | A 후보의 관측 \(r_7^{R'}\) 재인용 |
|---:|---:|---:|---:|---:|
| 10 | \(2.0342\times10^{-5}\) | \(5.9\times10^{-7}\) | \(2.67\times10^{-5}\) | \(2.1370\times10^{-5}\) |
| 15 | \(2.0342\times10^{-5}\) | \(3.5\times10^{-7}\) | \(1.05\times10^{-5}\) | \(2.3632\times10^{-6}\) |
| 19 | \(2.0342\times10^{-5}\) | \(2.4\times10^{-7}\) | \(4.64\times10^{-6}\) | \(2.2806\times10^{-7}\) |
| 24 | \(2.0342\times10^{-5}\) | \(1.6\times10^{-7}\) | \(7.36\times10^{-6}\) | \(6.5422\times10^{-8}\) |
| 30 | \(2.0341\times10^{-5}\) | \(1.2\times10^{-7}\) | \(1.69\times10^{-5}\) | \(2.6106\times10^{-8}\) |

B-T는 모든 셸에서 낮추는 후보가 아니다. 사전감사의 선중심 Wien 어림에서 1206 Å의 셸 10 B-T 씨앗 이동은 약 +0.12 dex이고, 122건 중 셸 10의 중앙값은 약 +0.14 dex로 역방향이다. 122건 중 셸 19의 중앙값은 약 −1.0 dex다. 이 값들을 기대치의 방향·규모 안내로 등록하되, 프로파일 평균의 계산값이나 허용폭으로 대체하지 않는다. B-W와 B-T 사이를 연속 보간하여 공개 스펙트럼에 맞추는 매개변수 탐색은 하지 않는다.

### 4.5 기억 민감도와 구성상 확정 사항

off의 등록 배경 아래에서

\[
\mathcal M_7
=\left|\frac{\partial J_7}{\partial J_1}\right|
=\left|\prod_{k=1}^{6}q_k\right|
\]

을 계산한다. 또한 B-W와 B-T의 차이에 대해

\[
\mathcal M_7^{WT}
=\left|\frac{J_7^{B-W}-J_7^{B-T}}
{J_1^{B-W}-J_1^{B-T}}\right|
\]

를 보고한다. 분모가 0과 분리되지 않으면 후자는 미분리이며, 미분 결과로 채우지 않는다.

**기억 제거의 비교 기준은 \(\mathcal M_7\le10^{-3}\)이다.** 이는 물리 오차가 아니라 초기 차이를 천분의 일 이하로 줄인다는 조작적 기준이다. 등록된 1206 Å off 배경에서 B·D의 기억 유지(\(q^6\simeq0.99996\))는 구성상 확정이며 새 예측·결과 코드·판정 분기로 삼지 않는다.

- B-T: 기존 \(T_{\rm inner}\) 상태에서의 조건부 이동을 계산하고 B-W를 감도·대조 열로 둔다. 구성상 남는 기억과 이동을 구별하며, 초기값 치환을 기억 제거로 표시하지 않는다.
- A: 고정한 외부 배경의 고립 전이비에는 직접적인 기존 \(J_1\) 인수가 없다. 기존 R′의 실측 이탈과 이 국소 구조를 별도로 보고한다.

- D는 후보에서 제외하고 result.json의 「국소 q·필요 반복수」 한 칸에 구성상 산술만 기록한다. \(0<q<1\)에서 \(q\simeq0.999993\)과 \(m=\lceil\log(10^{-3})/\log q\rceil\)를 기록하며, 그 밖에서는 필요 횟수를 정의하지 않는다. 이를 전체 계의 수렴 불능 증명이라 부르지 않으며 실제 반복 연장은 하지 않는다.

### 4.6 122건의 초기 원천 비교와 B-T 씨앗 민감도

각 고정 키에 다음을 적용한다.

1. INIT 원자계수·초기 원천 비교.
2. B-T 지정 씨앗의 기존 씨앗 대비 이동·조건부 비와 기억 민감도.

이 확장에서는 L/J̄·F_k·q_k·ρ의 기전 시험, 캡처 인구 역산, R′ armed 자격의 재현을 수행하지 않는다. §4.2의 사상·미분은 B-T의 조건부 씨앗 이동·기억 민감도 계산에만 사용하며 기전 재현 판정을 만들지 않는다.

S III 67건에 대해 초기 원천 양립성의 성립 부분집합·확정 반례·미분리·결손과 B-T 씨앗 이동·기억 민감도를 보고한다. 초기 원천 비교의 반례를 보존하되, 이 축소 분석으로 “67건 모두 같은 기전”을 판정하지 않는다.

이 결과만으로 황 이온화 전선의 위치, S II 광이온화율, 연속체장의 변화는 예측할 수 없다.

## §5. 2542 Å 응답과 공개 CMFGEN 비교

### 5.1 세대 7의 기준 상태

같은 캡처와 생산 역산으로 1206 Å의 \(n_0,n_4\), 2542 Å의 \(n_4,n_5\)를 구한다. 같은 셸의 \(n_4\) 복원 구간이 정합하는지 확인한다.

불일치를 평균으로 없애지 않는다. 세대, 원자 대응, 소비 자격을 확인한 뒤에도 정합하지 않는 셸의 연쇄 예측은 식별 불가로 둔다.

A에 대해서는 기존 R′/E′의 \(n_4,n_5,S,\tau\)와 국소 주입의 실측 차이를 보고한다. 전체 UV2500 차이나 공개 잔차에 대한 비율로 치환하지 않는다.

### 5.2 1206 Å에서 2542 Å로의 조건부 연쇄

\[
q_{45}=\frac{g_4n_5}{g_5n_4},\qquad
A_{\nu,45}=\frac{2h\nu_{\rm cap}^{3}}{c^2},
\]

\[
S_{45}=A_{\nu,45}\frac{q_{45}}{1-q_{45}},\qquad
\tau_{45}=K_{45}\left(n_4-\frac{g_4}{g_5}n_5\right).
\]

후보의 밀도 배율을 \(\alpha_4,\alpha_5\)로 두면,

\[
\frac{\tau'_{45}}{\tau_{45}}
=\frac{\alpha_4-\alpha_5q_{45}}{1-q_{45}},
\qquad
\frac{S'_{45}}{S_{45}}
=\frac{\alpha_5(1-q_{45})}
{\alpha_4-\alpha_5q_{45}}.
\]

\(n_4/n_0\)의 예측으로부터 절대 밀도 배율을 얻으려면,

\[
\alpha_4
=\frac{n_0'}{n_0}
\frac{(n_4/n_0)'}{n_4/n_0}
\]

가 필요하다. **\(n_0'=n_0\)는 조건부 계산의 가정이며, 관측 사실이 아니다.**

B-T에 대해서는 다음 경계 수치 기록용 대조를 고정한다.

| 대조 | 가정 | 목적 |
|---|---|---|
| P1 같은 비율 응답 | \(n_0\) 고정, \(\alpha_5=\alpha_4\) | \(n_5/n_4\)가 유지될 때의 τ 응답 |
| P2 상위 고정 | \(n_0\) 고정, \(\alpha_5=1\) | \(n_4\) 감소만으로 S·주입이 증가하는 반례 |
| A 실측 | 같은 세대의 E′·R′에서 두 배율을 취득 | 실제 두 준위의 동반 변화와 비교 |

P1·P2는 경계 수치 기록으로 한정하며, 셸 19에서는 두 대조의 국소 주입비가 1을 가운데 둘 것으로 예견된다. 이를 새 방향 예측의 성공으로 세지 않는다. 두 대조는 실제 응답의 상·하한이 아니며, 다른 준위·이온화의 응답은 그 바깥에 있을 수도 있다.

### 5.3 오프라인에서 계산할 국소 주입

양의 τ에 대해

\[
Q_{45}=S_{45}(1-e^{-\tau_{45}})
\]

와 그 후보비를 계산한다. 이는 하나의 공명에서의 국소 주입이며, 광도도 스펙트럼도 아니다.

P1에서는

\[
S'_{45}=S_{45},\qquad \tau'_{45}=\alpha_4\tau_{45},
\]

\[
\frac{Q'_{45}}{Q_{45}}
=\frac{1-e^{-\alpha_4\tau_{45}}}
{1-e^{-\tau_{45}}}.
\]

따라서 \(\alpha_4<1\)이면 국소 주입은 감소하지만, 두 상태에서 모두 포화된 채라면 감소는 작다. 인구비 감소를 같은 배율의 광도 감소로 치환하지 않는다.

P2에서는 양의 흡수를 유지한 채 \(n_4\)만 줄이면 S는 증가하고 τ는 감소한다. \(S\tau\)는 일정하며,

\[
Q=(S\tau)\frac{1-e^{-\tau}}{\tau}
\]

이므로 국소 주입이 증가하는 경우가 있다. **“준위 4가 줄면 UV2500도 반드시 줄어든다”는 명제의 반례**로 실제 분석 경로에서 확인한다.

\(\alpha_4-\alpha_5q_{45}\le0\)이거나 경계와 겹치면 양의 τ 식을 연장하지 않는다. 부호·특이점을 기록하고 S나 τ를 잘라내지 않는다.

### 5.4 오프라인 예측의 경계

| 양 | 이번에 가능한 것 | 다음에 남기는 것 |
|---|---|---|
| 1206 Å \(n_4/n_0\) | A 실측 재인용, B-T 조건부 예측과 B-W 감도·대조 | 실제 초기장·다준위 응답 |
| 2542 Å \(S,\tau,Q\) | B-T의 P1·P2 경계 수치 기록과 실측 A | 후보가 만드는 \(n_0,n_5\) |
| UV2500 스펙트럼 | 기존 E′·R′와 공개값의 비교 | 전달·중첩·재분배·다른 원천의 변화 |
| NUV/OPT | 기존 색비와 새로운 수용 기준 | 분자·분모의 동시 변화 |
| 황 전선 | 122건의 초기 원천 비교와 B-T 씨앗 이동·기억 민감도 | 국소 기전의 추가 검증·연속체·이온화율·전선 이동 |

### 5.5 공개 비교의 정의

공개 CMFGEN은 19.48 d의 열을 사용한다. 가까운 시각을 편의에 따라 다시 고르지 않는다.

- NUV: 2500–4000 Å.
- OPT: 4000–6000 Å.
- UV2500: 2500–3000 Å.
- 여덟 대역: 2500–3000, 3000–3500, 3500–4000, 4000–5000, 5000–6000, 6000–7000, 7000–8000, 8000–10000 Å.

파장 중복을 거부하고, 경계는 인접점의 선형 보간, 적분은 사다리꼴 규칙을 사용한다. 대역 밖으로 외삽하지 않는다.

\[
C_b^X=\frac{L_b^X}{L_{\rm OPT}^X},\qquad
d_b^X=\log_{10}\frac{C_b^X}{C_b^{\rm public}},
\qquad
R_8^X=\sqrt{\frac18\sum_b(d_b^X)^2}.
\]

알려진 반올림 값은 E′N 0.282, R′N 0.357 dex이며, NUV/OPT는 공개 0.775, E′N 1.319, R′N 0.415다. spectrum_reference는 기존 `/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/figs/cmp_pub_paired.py`의 대역 적분·OPT 정규화·로그 잔차·RMS 정의와 일치하는 판독기 안의 작은 함수로 산출한다. 공개 시각은 위에서 고정한 19.48 d를 사용하며, 원 스펙트럼에서 얻은 반올림 전 값을 보존한다. 기존 스크립트의 파일 신원과 실제 사용한 적분 함수·의존 파일 신원을 manifest에 기록한다.

OPT 정규화만으로 개선을 가장하지 않도록 대역의 절대 광도와 OPT 변화도 같은 표에 남긴다. 새로운 거리·진폭·색 맞춤은 추가하지 않는다.

### 5.6 다음 한 번의 판정런에 남길 기대치

다음 후보를 \(X\)로 둔다. 다음은 **반증 가능한 설계 기대이며, 오프라인 대수에서 보장된 스펙트럼 예보가 아니다.** 사용자 결정에 따라 B-T 씨앗 전달이 1차 기대이고 스펙트럼 로그 잔차 반감은 2차 기대다. 이번 expectations.json은 후속 사용자 결정의 입력이며 판정런 승인 자체가 아니다.

**기전 기대**

- B-T의 1차 기대: 발행 call-1 J̄가 지정 씨앗 \(\mathcal B_\ell(T_{{\rm pin},s})\)에 상대 편차 \(10^{-3}\) 이내로 접근한다. 씨앗 전달의 충족 여부를 스펙트럼 기대와 별도로 판정한다.
- B-T: 씨앗 추종과 기존 \(T_{\rm inner}\) 상태에서의 이동을 조사한다. §4.4의 셸 10 역방향(1206 Å 약 +0.12 dex, 122건 중 셸 10 중앙값 약 +0.14 dex)을 기대치에 포함하며, 씨앗 기억 제거를 성공 조건으로 선취하지 않는다.
- A: 기존 R′의 INIT 후 armed 작동과 유한 반복 응답을 재인용한다. 고정 배경에서의 직접 씨앗 의존성 제거와 실제 유한 반복 응답을 구별한다.
- 축약 모형을 적용할 수 있는 셸에서는 등록 예측에 대한 준위비 편차를 \(10^{-3}\) 기준으로 시험한다. 실패는 모형의 외삽 반증으로 남긴다.
- 2542 Å는 이번에 기록한 P1·P2 경계 수치와 다음 실측을 비교하며, 어느 쪽에도 대응하지 않는 경우도 정당한 결과로 둔다.

**공개 비교의 2차 설계 기대**

\[
|d_{\rm NUV}^X|\le\frac12|d_{\rm NUV}^{E'N}|,
\]

\[
|d_{\rm UV2500}^X|\le\frac12|d_{\rm UV2500}^{E'N}|,
\]

\[
|d_b^X|\le0.05\ {\rm dex}
\quad
(b=3000\!-\!3500,\ 3500\!-\!4000,\ 4000\!-\!5000,\ 5000\!-\!6000),
\]

\[
R_8^X<R_8^{E'N}.
\]

즉 NUV와 UV2500의 로그 잔차를 적어도 절반으로 줄이고, 이미 가까운 네 대역을 보존하며, 여덟 대역 RMS도 개선한다는 기대다. 절반 감소는 이번에 선택하는 과학적 목표이며, 수치 오차나 공개 자료의 관측 불확실성이 아니다.

E′ 기준에서는 NUV 잔차를 약 ±0.115 dex, UV2500 잔차를 약 ±0.305 dex 이내로 만드는 것에 해당한다. A의 기존 R′는 이 공개 비교 기대를 만족하지 않는다. B-T의 씨앗 전달이 성립해도 이 2차 기대의 충족을 보장하지 않는다.

후보별로 다음을 반드시 명시한다.

- 1206 Å의 수치 예측과 조건.
- 2542 Å의 수치 민감도와 미정인 인구 응답.
- 스펙트럼에 대해서는 위 설계 기대.
- 도출하지 못한 스펙트럼 예보를 `NOT_PREDICTED_OFFLINE`으로 두는 이유.

설계 기대의 등록을 정량적인 전달 예보를 얻었다는 주장으로 바꾸지 않는다. 기대가 반증된 뒤에 절반 감소 조건을 완화하지 않는다.

## §6. 수치 기준·과학 결과·후보의 적용 범위

### 6.1 수치 처리

- binary64와 텍스트 출력의 반올림, W5의 오차 상한, 생산 연산 순서를 반영한다.
- exp·log·프로파일 평균은 독립 고정밀 참조로 검사한다.
- 상쇄되는 변화량에는 절대 오차를 유지한다.
- 공유 입력의 상관을 무시하여 좁은 구간을 만들지 않는다.
- 후보의 물리적 모형 오차를 부동소수점 구간에 섞지 않는다.
- 결손을 다른 세대·다른 셸·인접 선으로 보충하지 않는다.

상태는 다음과 같이 둔다.

| 상태 | 의미 |
|---|---|
| `SURVIVES` | 등록 조건 전체를 유효 구간이 만족 |
| `REFUTED` | 조건 위반이 구간 또는 확정된 소스·실행 증거로 분리 |
| `UNRESOLVED` | 유효한 값이 판정 경계를 가로지름 |
| `NOT_IDENTIFIABLE` | 필요한 입력·권위·정의가 부족 |

미실행 미래 예측에는 `FUTURE_TEST`를 붙이고 `SURVIVES`로 두지 않는다. 명시적으로 대상 밖인 가설에는 적용 여부의 이유를 기록하고 결손과 구별한다.

### 6.2 결과 코드

후보 A·B-T의 셸별 구성 상태를 저장하고, 현재 평가할 수 있는 오프라인 가설에 다음 우선순위로 코드를 붙인다. A의 기존 실측은 재인용으로 표시한다. B-W는 감도·대조 열이며 독립 후보 코드를 갖지 않는다. 122건에서는 §4.6의 초기 원천 양립성만 상태로 판정하고, 씨앗 이동·기억 민감도는 조건부 수치로 기록한다.

| 우선순위·코드 | 명칭 | 조건 |
|---|---|---|
| 1 | `INIT_PREMISE_REFUTED` | 후보가 요구하는 INIT 전제에 확정 반증 |

| 3 | `REDUCED_MODEL_REFUTED` | 적용을 주장한 준위비·한 단계 응답 모형에 확정 반증 |
| 0 | `OFFLINE_HYPOTHESES_SURVIVE` | 적용 대상 오프라인 가설이 모두 생존 |
| 5 | `NOT_IDENTIFIABLE` | 위 확정 분기가 없고 필수 정보가 결손 |
| 4 | `UNRESOLVED` | 남은 유효 정보가 경계에서 미분리 |

입력의 잘못된 결합이나 산술 구현의 결함은 과학 코드로 두지 않고 입력 적격성 `X`와 구체적인 이유를 남긴다.

B·D의 기억 유지와 §2.3의 C 관련 항등성은 구성상 확정 사항으로만 기록하며 결과 코드·판정 분기를 만들지 않는다. 종전 코드 2는 사용하지 않고 나머지 코드 번호는 유지한다. 이 기록은 B-T의 기존 상태에서의 이동 예측을 취소하지 않는다.

코드 0은 스펙트럼 개선·물리적 올바름·런 승인을 뜻하지 않는다. 미래 예측을 코드 0의 성립 조건에 섞지 않는다.

### 6.3 집합 결과

전체의 다수결 코드는 만들지 않는다.

- 1206 Å: 주 셸 19와 다른 네 셸을 함께 제시.
- 2542 Å: 주 셸 19와 17–22의 공간 확인을 함께 제시.
- 확장: 122건 전체, 이온별, 셸별, 선행 β 구간별.
- 황: S III 67건과 S II 8건을 나누어 보고.

평가 범위 안의 전칭 명제에 확정 반례가 하나라도 있으면 반증한다. 122건에서 이 명제의 범위는 초기 원천 양립성이며 동일 기전의 전칭 검증이 아니다. 다른 결손을 이유로 그 반례를 지우지 않는다. 부분집합은 키와 성립 수의 하한·상한을 남긴다.

### 6.4 오프라인 검증과 런 준비의 구별

후보 A·B-T에 대해서만 다음을 별도 칸에 기록한다. B-W·C·D에는 후보 준비 상태를 만들지 않는다.

1. `algebra_verified`: 조건부 식과 구현의 검증.
2. `baseline_applicability`: 기존 이력·캡처에 의한 모형 적용성.
3. `intervention_defined`: 생산의 어느 입력·소유자·시점을 바꾸는가.
4. `seed_delivery_verified`: 그 개입이 지정된 유효 씨앗을 만든다는 근거.
5. `expectations_registered`: 방향·양·반증 조건·대상 시점의 고정.
6. `future_run_authorized`: 이번에는 거짓.

조건부 대수만 완료된 B-T를 “생산 수리의 오프라인 검증 완료”로 요약하지 않는다. B-T의 `seed_delivery_verified`는 이번 단에서 미검증이며 런 발주 요건 ②가 남는다. `expectations_registered`는 다음 사용자 결정의 입력이 갖춰졌다는 뜻이고, `future_run_authorized`는 이번에 거짓이다.

## §7. 최소 변경집합·출력

### 변경집합 — DET-SIIII-REPAIR-OFFLINE (표)

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `scripts/det_siiii_repair_readout.py` | `det_siiii_repair_*` — 신설 | 고정 입력 연결, INIT 전제표, A 실측 재인용·B-T 조건부 대수·B-W 감도 대조, 122건 초기 원천 비교·B-T 씨앗 이동·기억 민감도, 2542 Å 경계 수치, 기존 cmp_pub_paired.py 정의와 같은 작은 공개 비교 함수, 구간·분기·직접시험 |
| `/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/tools/run_siiii_repair.slurm` | 오프라인 러너 — 신설 | 환경·신원·시간·용량 기록, 시험·게이트·분석, 로그·rc 보존 |

### 변경집합 끝

새 함수의 접두사는 `det_siiii_repair_`로 한다. CLI 진입점도 해당 함수에 연결한다.

생산 소스·바이너리·원자 자료·봉인 입력·기존 판독기·기계 게이트·선행 결과는 변경하지 않는다. 계약·판정·감사의 문서화는 운전자의 기록 업무로 둔다.

| 필수 출력 | 내용 |
|---|---|
| init_origin.json | 초기 온도·물질·복사의 경로, 소스 신원, 실행 증거, 각 INIT 가설과 미특정 부분 |
| candidate_1206.csv | 다섯 셸의 후보 A 실측 재인용·B-T 조건부 값과 B-W 감도·대조 열, 씨앗·준위비·이력 종점·기억·충돌 민감도 |
| extension.csv | 122건 고정 키의 초기 원천 비교·반례·결손과 B-T 씨앗 이동·기억 민감도 |
| response_2542.csv | 실측 E′/R′와 B-T의 P1·P2 인구·S·τ·국소 주입 경계 수치 |
| spectrum_reference.csv | 기존 cmp_pub_paired.py 정의에 맞춘 공개·E′N·R′N의 반올림 전 절대 대역 광도·색·잔차·다음 목표 |
| expectations.json | 다음 사용자 결정의 입력: 후보 A·B-T의 조건부 값·기준선·반증 임계값·준비 상태, B-T 씨앗 전달 1차 기대·셸 10 역방향·스펙트럼 반감 2차 기대 |
| result.json | 과학 코드·구성 상태·고정 집합 집계·미해결 질문, 구성상 확정 사항과 D의 「국소 q·필요 반복수」 한 칸 |
| selftest.json | 실제 analyze 경로, 독립 참조, 결함 주입 결과, 시험명과 내용 |
| manifest.json | 계약·입력·의존·소스·argv·출력 신원, 환경·비용·이탈 |

출력은 새 실행 식별자 아래에 만들고 실패 기록도 보존한다. 원시 W5·캡처 전체, 전체 주파수 배열, 시험용 실제 입력 복제본은 저장하지 않는다.

## §8. 게이트·직접시험

| 게이트 | 방지할 실패 | 통과 조건 |
|---|---|---|
| G-LINK | 상태·행·셸·시점·세대 혼동, 고정 집합 변경, 권위 결손의 보충 | §1–§3의 연결과 결손 처리가 실제 출력까지 유지됨 |
| G-REPAIR | 씨앗 치환과 기억 제거의 혼동, β·κ 생략, 인구비에서 광도로의 비약, 잘못된 분기 | §4–§6의 대수·독립 참조·반례가 실제 analyze 출력에서 확인됨 |

과학 예측의 반증은 게이트 실패가 아니다.

### 8.1 직접시험의 원칙

작은 생산 형식의 가상 입력으로 **실제 analyze를 호출하고 저장된 출력을 검사**한다. 보조 함수만의 시험이나 기대 코드의 직접 출력은 인정하지 않는다.

- 유리 연산 부분은 binary64 입력을 정확한 유리수로 취급하여 독립 계산한다.
- exp·log는 독립 고정밀 상·하한과 비교한다.
- 출력 구간이 독립 참조를 포함하며, 분리 가능한 반례를 과대한 구간으로 미분리 처리하지 않는지 확인한다.
- 시험명·입력 결함·기대 이유·실제 이유·rc를 같은 기록으로 연결한다.

### 8.2 필수 실패 검출

1. **연결·시점**
   - INIT의 Sobolev 전용 칸이 없는 정상 입력.
   - INIT의 NLTE 물질 세대 0과 bootstrap 원자 인구 세대 1의 구별.
   - call·셸·행·주파수·세대 교환, 모순된 중복, 권위 결손, source fallback.
   - 122건의 결과 의존 재선별, 결손과 확정 반례의 동시 존재.

2. **INIT**
   - bootstrap 온도와 후속 고정 온도가 다른 예.
   - warm start만의 변경을 물질·경계 변경으로 오인하는 결함.
   - 원자계수가 완전한 Planck 관계를 만족하지 않는 예.
   - 소스 경로만으로 전달 기여율을 확정 처리하는 결함.

3. **후보 대수**
   - 비단위 κ, 0이 아닌 ΔB, β만으로 단순화하는 것의 반례.
   - 서로 다른 씨앗의 차이가 그대로 남는 off 모형과 고정 배경의 armed 모형.
   - 충돌을 남기면서 β를 소거하는 결함.

   - \(q\ge1\), 예측 변화 0, 기억 차이의 분모 0을 정상값으로 채우지 않는 처리.

4. **2542 Å**
   - 알려진 인구로 만든 τ·S의 역산.

   - 상위 인구 고정·하위 감소에서 국소 주입이 증가하는 반례.
   - \(n_4/n_0\) 배율을 무조건 절대 인구 배율로 치환하는 결함.
   - 흡수 부호 변화·특이점을 무시하는 결함.

5. **공개 비교·분기**
   - OPT만 변해도 색이 변하는 예.
   - 일부 대역 개선과 전체 RMS 악화의 공존.
   - 공개 범위 밖·시각 불일치·경계 미분리.
   - 과학 코드 0·1·3·4·5의 분기, 구성상 확정 사항을 코드 2로 부호화하지 않고 미실행 예측을 생존 처리하지 않는 처리.

선행 판독기의 모든 시험을 다시 실행하지 않는다. import 기능은 이번 실제 분석 경로의 시험으로 필요한 범위를 확인한다.

### 8.3 기계 프리플라이트

기존의 다음 도구를 그대로 사용한다.

- `scripts/check_prereg_preflight.py`
- `scripts/check_gate_referents.py`
- `scripts/check_doc_canonical.py`
- `scripts/check_dispatch_preflight.py`

문서 신원·stdout·stderr·rc를 보존한다. 같은 문서 신원에 대해 dispatch가 호출한 prereg 검사는 중복 실행하지 않는다.

기계 게이트는 과학식·독립 감사를 대신하지 않는다.

## §9. 환경·비용·재현 명령

### 9.1 환경

등록 환경은 a10 CPU Slurm, GPU 예약 0이다. 수동 실행 장소는 lageunha로 한다. 로그인 노드에서는 파일 조회·소스 판독만 수행하고 직접시험·게이트·수치 분석은 계산 장소에서 실행한다.

러너는 CPU·메모리·시간·GPU 예약 0을 명시하고 해당 클러스터의 현행 제외 규칙을 따른다. module 초기화, Python·의존성 버전, 스레드 수, 호스트, job ID를 기록한다.

lageunha에서는 사전에 부하를 확인하고 같은 자원 상한을 사용한다. 시간 측정에는 Bash `time`을 사용하며 `/usr/bin/time`을 전제하지 않는다.

### 9.2 비용 상한

직전의 91초·24 MiB와 감사 수리 재실행을 고려하여 **초회와 적어도 세 번의 완전한 수리 재판독**을 확보한다.

| 항목 | 상한 |
|---|---:|
| 동시 계산 | 1작업 |
| CPU | 4 |
| 메모리 | 16 GiB |
| 한 번의 전체 시험·분석 | 120초 |
| 전체 실행 예산 | 초회 120초＋수리 재실행 3회×120초 |
| 독립 게이트·보조 확인 예산 | 누적 120초 |
| Slurm·수동을 합친 누적 실행 경과 | **600초** |
| CPU 예약량 | **0.667 core·h 이하** |
| 새 도구·시험·출력·로그 합계 | **192 MiB 이하** |
| GPU 예약·생산 실행 | **0 GPU·h, 0회** |

용량은 전체 실행당 32 MiB를 네 번분, 도구·게이트·실패 기록 등에 64 MiB를 예상한 배분으로 둔다. 이는 실측값이 아니라 예산이다.

- 실패·수리·재현·수동 수치 실측을 모두 합산한다.
- 같은 Slurm 작업의 러너 시간과 Slurm 경과를 이중 합산하지 않고, 둘 다 기록하여 비용 집계에는 큰 쪽을 사용한다.
- 대기 시간, 문서 작성, 사람의 감사 시간은 계산 경과와 나눈다.
- 시간을 측정하지 않는 수치 실행은 하지 않는다.
- 초회 실측 후에도 세 번분의 수리 여력을 확보한다.
- 재실행은 원인 수정과 잔여 예산을 확인한 뒤 수행한다. 무조건 자동 재제출하지 않는다.

W5·line_list·캡처는 상태별로 필요한 키를 모아 한 차례 순회 판독하며, 선마다 다시 훑지 않는다. 출력 용량은 러너가 만든 파일 목록에서 집계한다.

상한 안에 마칠 수 없으면 실제 비용·미완 항목·필요한 추가 작업을 보고한다. 비용 초과를 과학 가설의 반증으로 삼지 않는다.

### 9.3 계획 CLI

```bash
python3 -B "$P/scripts/det_siiii_repair_readout.py" selftest \
  --production-source "$T2" \
  --out "$D/results/$RUN_ID/selftest"

python3 -B "$P/scripts/det_siiii_repair_readout.py" analyze \
  --e-run "$E" \
  --r-run "$R" \
  --prior-mechanism "$M/results/t7/analysis" \
  --writer-root "$N" \
  --public-spectrum "$P/data/standart_data1/toy06/spectra_toy06_cmfgen.txt" \
  --production-source "$T2" \
  --selftest "$D/results/$RUN_ID/selftest" \
  --out "$D/results/$RUN_ID/analysis"
```

후보·셸·집합·임계값은 계약으로 고정한다. 결과에 맞추는 선택 옵션은 만들지 않는다.

```bash
python3 -B "$P/scripts/check_prereg_preflight.py" \
  "$P/docs/PREREG_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md" --root "$P"

python3 -B "$P/scripts/check_gate_referents.py" \
  "$P/docs/PREREG_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md" --root "$P"

python3 -B "$P/scripts/check_doc_canonical.py" \
  "$P/docs/PREREG_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md"

python3 -B "$P/scripts/check_dispatch_preflight.py" "$ORDER" --root "$P"
```

실제 argv·작업 장소·출력 경로·rc를 보존한다. 재현도 새 결과 경로를 사용하며 같은 누적 상한에 포함한다.

## §10. 처분·완료 조건·후속 결정

### 10.1 필수 완료 내용

판정문 처음에 다음을 명시한다.

- 이번 스펙트럼 잔차 감소는 0.
- 생산 해 재현의 미달을 승계.
- 기존 결과·소스에 노출된 계약이라는 점.
- 후보 A·B-T의 기존 상태 이탈·모형 적용성과 B-W 감도·대조; 기억 유지·C 관련 항등성·D 산술은 구성상 확정 기록.
- INIT에서 특정한 범위와 미특정 부분.
- 후속 판정런은 미실행·미승인.

완료 본문에는 다음이 필요하다.

- 다섯 셸의 1206 Å A 실측 재인용·B-T 조건부 예측·B-W 감도 대조와 명시한 조건.
- 2542 Å의 실측 A와 P1·P2 경계 수치, 주입 방향이 유일하지 않다는 한계.
- 기존 스크립트 정의로 보존한 공개 비교의 반올림 전 값과 다음 사용자 결정에 넣을 기대치.
- 122건 전체와 S III 67건의 초기 원천 양립성에 대한 성립·반증·미분리·결손과 B-T 씨앗 이동·기억 민감도.
- 런 발주 세 요건의 A·B-T별 충족 상태, 특히 B-T 씨앗 전달 미검증으로 남는 요건 ②.
- 시험·게이트·입력 신원·원 명령·비용·보존·이탈.
- astra 종료감사와 운전자에 의한 실행 가능한 주장의 재현 상태.

결손 목록만 만들고 후보 검증 전체가 완료됐다고 쓰지 않는다. 목표의 미달 부분을 그대로 남긴다.

### 10.2 집행 처분과 과학 결과

| 처분 | 기준 |
|---|---|
| `PASS` | 이 단의 전체 처분으로는 사용하지 않음 |
| `PASS_LIMITED` | 등록 분석·시험·반증·기대치 출력이 적격하게 완료되고 조건부 범위와 미달을 명시 |
| `REWORK` | 선택·연결·식·구간·분기·시험의 계약 위반, 결손 은폐, 과도한 주장 |
| `BLOCKED` | 필수 입력·권위·등록 자원의 실제 문제로 최소 분석을 완료할 수 없음 |

**전체 처분의 상한은 `PASS_LIMITED`다.**

과학 예측이 반증되어도 도구가 올바르게 반증을 나타냈다면 `PASS_LIMITED`로 닫을 수 있다. 한편 처분이 `PASS_LIMITED`여도 후보가 생산 수리로 검증됐다는 뜻은 아니다.

### 10.3 결과별 다음 경로

| 결과 | 다음 판단 |
|---|---|
| A의 기억 해제는 지지되지만 공개 비교는 악화 | 둘 다 유지하며 A 단독을 공개 일치의 수리로 삼지 않음 |
| B-T의 조건부 씨앗 이동 | 구성상 남는 기억과 구별하고, B-W는 감도·대조로만 소비 |
| B-T의 INIT 전제 또는 씨앗 전달이 미확인 | 요건 ②의 미완을 명시하고, 다음 판정런의 1차 씨앗 전달 기대에 연결 |
| 2542 Å의 P1·P2 경계 수치 | 준위 5의 실제 응답을 미해결로 남기고 UV 감소를 단정하지 않음 |
| S III 초기 원천 비교의 반례 | 해당 원천 양립성 주장만 제한하고 동일 기전 전체의 판정으로 확대하지 않음 |

새 판정런은 이번에 발생하지 않는다. 이번 expectations.json을 다음 사용자 결정의 입력으로 사용한다. 별도 기대치 재등록을 중복 요구하지 않으며, 다음 결정은 이 파일의 조건부 값·1차 및 2차 기대·반증 조건을 인용하여 한 후보·한 번의 범위와 실행 권한을 정한다.

1. 기전의 오프라인 특정이 성립한 범위.
2. 해당 후보의 오프라인 검증과 남은 조건.
3. 구현할 변경집합과 이번 expectations.json에 등록된 초기장·인구·2542 Å·공개 비교의 기대치 및 반증 조건을 연결.
4. 비교 기준선, 고정 반복 구간, 소비 세대, 비용, 필요한 출력.
5. 선택한 후보명과 이번 expectations.json의 파일 신원을 실행 전에 확정. 기대치 변경이 필요하면 변경 이유를 명시한 사전 개정으로 처리하며, 결과를 본 뒤 재고정하지 않음.

여러 후보를 실행하여 결과가 좋은 것을 고르는 계획, 실패 후 자동 반복 연장, 사후 임계값 변경은 이 계약에서 발생하지 않는다.

## §11. 집행 기록·분장

| 업무 | 담당 | 한계·규칙 |
|---|---|---|
| 기안·개정 | astra | 실측 모델 ID를 기록 |
| 경량 사전감사 | Fable fresh | 개정25-aa에 따라 운전자가 Opus 5.5인 이번 계약에는 Q-GOAL·Q-LEAN 1회 시행. 호출 실패 시 gpt-6.1-sol 대행, 실패 증거·실제 모델 ID 보존. 미실시를 PASS로 바꾸지 않음 |
| 구현·직접시험·게이트·실행·판독·판정 | 운전자(Opus 5.5) | 구현자와 판정자의 겸임을 명시 |
| 코드 감사 | 해제 | 정례 감사를 추가하지 않음 |
| 종료감사 | astra | 구현·판정으로부터 독립. 계약 저자라는 이해관계를 명시. 개정18 ③에 따라 1·2회차는 전건 반영하고, 3회차부터는 «폐합 차단»만 반영하며 나머지는 관측으로 남긴 뒤 폐합. 누적 실행 비용 상한 600초 유지 |
| 범위·비용 확대·생산 변경·판정런 | 사용자 | 이번 기안에서 자동 발생하지 않음 |

| 집행 항목 | 개정 1 시 상태 |
|---|---|
| 공유 문맥·작업 장소·프로젝트 신원 | 확인 완료 |
| 사용자 결정·선행 계약·판정 | 직접 판독 |
| T2 초기 온도·인구·선 원천·INIT 순서 | 관련 소스와 기존 로그를 직접 판독 |
| 선행 다섯 셸의 준위비 | 정본 excitation을 직접 판독 |
| Fable fresh 경량 사전감사(Q-GOAL·Q-LEAN) | claude-fable-5-1 사전감사 직접 판독: Q-GOAL 부합(조건부), Q-LEAN 과잉, «폐합 차단» 1·«기재로 족함» 8. 정본은 D/fable_preaudit.md. 개정 후 독립 재판정을 받은 것으로 바꾸지 않음 |
| 사전감사 후 사용자 결정 | 결정문 마지막 항의 「축소 수용」·「다음 판정런의 1차 기대로」를 직접 판독하여 개정 1에 반영 |
| 새 후보 수치 계산·122건 초기 원천 비교·B-T 씨앗 민감도 | 판독기 미구현·실행 0회 |
| 파일화·기계 게이트·구현·직접시험 | 초안 파일화 완료(HEAD 82246f1). 개정 1 파일화·기계 게이트·구현·직접시험은 미집행 |
| 운전자 판정·종료감사 | 미실행 |
| 생산 변경·판정런 | 이번 대상 밖 |

기안자는 파일 변경, 수치 분석, 시험, 게이트, 작업 제출을 수행하지 않았다. 미래 항목을 예상 PASS로 채우지 않는다.

기안 메모:

- 원문의 고립 전이·대수·원천 관련 용어 오기는 문맥에 맞는 한국어로 바로잡았으며, 수식·기준값은 변경하지 않았다.
- 위 표의 기존 소스·선행 산출 판독 기록은 직전 기안의 집행 이력으로 보존한다. 개정 1에서는 공유 문맥·작업 장소·프로젝트 신원(HEAD 82246f1)·현행 규약·대상 전문·사전감사·사용자 결정 마지막 항·기존 공개 비교 스크립트와 적분 정의·기계 게이트 소스를 판독했다. 사전감사 원문은 `/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/fable_preaudit.md`다.

## §12. 기계 선언

다음은 변경집합·참조 실체와 §6.2의 코드 부호화를 검사한다. 과학식, INIT 귀속, 후보 선택, 구간 계산의 올바름은 직접시험과 종료감사로 확인한다.

`v`는 선언 검사용 결과 코드 하나다. 유효 코드는 0·1·3·4·5이며, 그 밖의 값·결손·복수 값은 `X`로 둔다. 2는 구성상 확정 사항의 잘못된 코드화를 거부하는 음성 대조이고 9도 음성 대조용이다. 기억 유지·C 관련 항등성·D 산술, 집행 처분, 미실행 미래 기대를 결과 코드에 섞지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SIIII-REPAIR-OFFLINE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_siiii_repair_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/tools/run_siiii_repair\\.slurm)",
    "symbol": "det_siiii_repair_",
    "roots": [
      "scripts/det_siiii_repair_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/tools/run_siiii_repair.slurm"
    ],
    "expected_extra": [
      "scripts/det_siiii_repair_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/tools/run_siiii_repair.slurm"
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
      "valid": "len(v) == 1 and all(x in (0,1,3,4,5) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",

      "c3": "sum(1 for x in v if x == 3)",
      "c4": "sum(1 for x in v if x == 4)",
      "c5": "sum(1 for x in v if x == 5)"
    },
    "rules": [
      {"name": "X", "predicate": "not valid"},
      {"name": "OFFLINE_HYPOTHESES_SURVIVE", "predicate": "valid and c0 == 1"},
      {"name": "INIT_PREMISE_REFUTED", "predicate": "valid and c1 == 1"},

      {"name": "REDUCED_MODEL_REFUTED", "predicate": "valid and c3 == 1"},
      {"name": "UNRESOLVED", "predicate": "valid and c4 == 1"},
      {"name": "NOT_IDENTIFIABLE", "predicate": "valid and c5 == 1"}
    ],
    "adversarial_fixtures": [
      {"name": "offline-hypotheses-survive", "mix": [[0,1]]},
      {"name": "init-premise-refuted", "mix": [[1,1]]},
      {"name": "construction-fact-as-code-rejected", "mix": [[2,1]]},
      {"name": "reduced-model-refuted", "mix": [[3,1]]},
      {"name": "unresolved", "mix": [[4,1]]},
      {"name": "not-identifiable", "mix": [[5,1]]},
      {"name": "invalid-code", "mix": [[9,1]]},
      {"name": "missing-code", "mix": []},
      {"name": "multiple-codes", "mix": [[0,1],[3,1]]}
    ]
  },
  "references": [
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/CLAUDE.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/evidence/USER_DECISION_2026-10-07.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_repair_20261007/fable_preaudit.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_DET_SIIII_1206_MECH_2026-10-07.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_SIIII_1206_MECH_2026-10-07.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/astra_closeout3.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/results/t7/identity.sha256"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/results/t7/analysis/census.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/results/t7/analysis/history.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/results/t7/analysis/excitation.csv"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_1206mech_20261007/results/t7/analysis/result.json"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_OBS_WRITER_2026-10-02.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_OBS_LINEDECOMP_2026-10-07.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/VERDICT_DET_SIIII_POP_2026-10-07.md"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/det_siiii_1206mech_readout.py"
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/det_siiii_pop_readout.py",
      "flags_existing": ["recover", "capture_records", "prod_tau_S"]
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/det_siiii_rates_readout.py",
      "flags_existing": ["det_siiii_rates_coll", "det_siiii_rates_rad"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c",
      "flags_existing": ["t_electrons"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cuda.cu",
      "flags_existing": ["opacity.t_electrons"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c",
      "flags_existing": [
        "lumina_publish_seed_te",
        "lumina_init_seed_material_predictor",
        "lumina_line_upper_population_fill_for_tau",
        "bb_nrb_arm"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/population_contract.c",
      "flags_existing": ["population_line_level_number_density"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c",
      "flags_existing": [
        "cmf_fine_line_material",
        "cmf_fine_exact_owner_solve",
        "cmfgen_fine_jbar",
        "INIT_SHARED_GAUSSIAN"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c",
      "flags_existing": [
        "line_net_sobolev_material",
        "line_net_einstein_opacity_ratio",
        "line_net_bracket_rates"
      ]
    },
    {
      "path": "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06/spectra_toy06_cmfgen.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/figs/cmp_pub_paired.py",
      "flags_existing": ["band_integral"]
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
