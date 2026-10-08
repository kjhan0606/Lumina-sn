# 사전등록 — DET-SEEDMEM-SCOPE: 1206 밖 두꺼운 셀의 NRB 모형 적용성과 저장 자료의 기전 식별 범위

2026-10-08 · astra 기안 · 읽기 전용  
운전자 파일화 대상(신설): `docs/PREREG_DET_SEEDMEM_SCOPE_2026-10-08.md`  
상태: **기안. 기계 게이트·경량 사전감사·구현·직접시험·수치 판독·판정·종료감사는 미집행.**

이번 단은 **기존 저장 자료만 사용하는 오프라인 조사**다. 생산 소스 변경, 생산 바이너리 실행, GPU 예약·실행은 없다.

## §0. Q-GOAL·Q-LEAN 자기답과 과학 질문

### Q-GOAL — 최종목적에 직접 기여하는가?

최종목적은 Monte-Carlo 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현하는 것이다. diag6는 Si III 1206.50 Å의 지연 자기항 구조를 특정했지만, 다른 선의 추종·차이 유지까지 같은 직접 기전으로 귀속하지 않았다. 특히 Si III line 471957·sh10은 off 상태에서 추종하면서도 R′의 2준위 NRB 원천 예측과 어긋나는 반례다.

이번 목표는 **1206 밖의 고정 표본에 대해, 저장 자료가 허용하는 모형 적용성·국소 의존성·미식별 범위를 판정하고 다음 NRB 판정런의 P1·P2 설계에 사용할 명시적인 변경표를 만드는 것**이다. 스펙트럼을 개선하거나 새 NRB 상태를 생성하는 단이 아니다.

과학 질문은 다음 순서로 고정한다.

1. **식별 가능성:** 각 선·셸·세대에서 어떤 원천·율·준위 인구를 저장 자료와 생산 관계로 복원할 수 있는가? 무엇은 필요한 입력이 없어 식별할 수 없는가?
2. **2준위 모형:** 고정된 다른 Si III·S III 두꺼운 셀에서, 같은 런의 저장 배경을 소비한 2준위 NRB 모형이 R′의 원천을 상대편차 \(10^{-3}\) 이내로 재구성하는가?
3. **불일치의 범위:** 다준위 연결, super-level 제약, 충돌 자료의 실제 소비, 국소 선 연산자, 연속체 배경 중 무엇을 저장 증거로 확인·배제할 수 있는가? 설명되지 않은 잔차는 무엇인가?
4. **직접·배경 의존성:** 저장 입력을 생산 순수 함수에 교차 대입하여 국소 자기 방출항에 대한 율의 의존성을 분리할 수 있는가? 그 결과를 전체 결합계의 인구 응답에까지 확대할 수 있는가?
5. **다음 결정:** 위 결과가 P1의 대상·허용오차·반증 해석과 P2의 측도·대상·주장 범위를 어떻게 바꾸는가?

설계 감사 지적 6에 대한 답은 다음과 같다.

- **P1 대상:** D2 또는 L2라는 이유만으로 1206 밖을 같은 원천 합치 기준에 넣을 수 있는지 판단한다. 국소 율의 자기항 제거 시험과 2준위 원천 합치 시험을 분리한다.
- **P1 허용오차:** \(10^{-3}\)을 보장된 생산 오차 상한으로 쓰지 않는다. 고정 표본에서 적용성이 반증되면 대상을 제한하거나 모형을 바꾸며, 관측 최대 잔차로 허용오차를 늘리지 않는다.
- **P2 기준:** 선택된 이온합·전자밀도의 작은 변화가 복사장 또는 복원 가능한 준위 인구의 작은 변화와 함께 나타나는지 검사한다. 다음 계약에서 필요한 측도와 저장 세대를 명시한다.
- **귀속:** 국소 율의 명시적 자기항 제거, 조건부 축약 모형의 응답, 저장 상태의 다준위 수지, 전체 결합계의 씨앗 기억 제거를 서로 다른 주장으로 분리한다.

따라서 이번 산출의 소비처는 다음 NRB 계약의 **대상·기준·필요 출력·귀속 문구**다. 생략하면 이미 확인된 반례를 포함하는 집합에 1206의 기준을 일반화하거나, 국소 율의 성질을 결합계의 기억 제거로 읽을 위험이 남는다.

**기안자 자기답: PASS.**

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

다음 측정만 둔다. 각 행의 소비처가 없는 추가 측정은 하지 않는다.

| 이번 측정·확인 | 다음 계약에서 소비하는 결정 |
|---|---|
| 선·셸·call·물질 세대·복사 세대·율 적용 권위 | P1에서 실제 판정할 수 있는 대상과 필요한 저장 출력 |
| 고정 표본의 R′ 2준위 원천 잔차와 E′·X 대조 | P1 원천 합치 대상, \(10^{-3}\)의 적용 범위, 반증 해석 |
| 충돌 활성 분기·자료 소비 및 선 연결·super-level 구조 | 불일치가 모형 생략에서 생겼는지, 별도 구현 문제인지의 구분 |
| 세대 7의 표적 전이 수지와 연결된 다른 전이 수지 | 다준위 설명을 지지할 수 있는 범위와 남는 미귀속 잔차 |
| 생산 순수 율 함수의 고정 배경 교차 대입 | P1의 국소 자기항 제거 시험과 원천 합치 시험의 분리 |
| W5 범위의 이온합·전자밀도·복사장·복원 가능한 준위 인구 변화 | P2에 필요한 측도, 두 연속 변화의 정의, 정상성 주장 상한 |

새 전 이온 census, 스펙트럼 적분, 공개 CMFGEN 재비교, 고정점 탐색, 반복수 예측, 다른 상수값 시험은 추가하지 않는다. 기존 L2·D2는 표본의 과거 관측 표지로만 읽으며 새 추종 분류기를 만들지 않는다.

분석은 새 판독기 하나에 묶는다. 생산 순수 함수의 CPU 연결, 판독 직접시험, 입력 적격성 검사를 같은 실행 묶음으로 수행한다. 별도 코드 감사나 작은 검사별 감사를 추가하지 않는다.

**기안자 자기답: APPROPRIATE.**

### 승인 근거·분장

승인 근거는 다음 파일이다.

`/gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008/evidence/USER_DECISION_2026-10-08.txt`

사용자가 선택한 것은 「오프라인 조사 추가」다. 이번 범위는 GPU 없는 저장 자료 조사이며, 생산 변경 또는 후속 판정런의 승인을 포함하지 않는다.

- 기안·개정·종료감사: astra.
- 경량 사전감사: Fable fresh, Q-GOAL·Q-LEAN 전용 1회.
- 판독기 구현·직접시험·게이트·실행·판독·판정: 운전자(Opus 5.5).
- 코드 감사: 해제.
- 범위·비용 확대, 생산 변경, 후속 판정런: 사용자 결정.

기안자의 자기답은 독립 사전감사가 아니다. 운전자는 구현자와 판정자를 겸하며, astra는 계약 저자로서 종료감사를 수행한다. 이 이해관계를 기록한다.

### 사전 노출·주장 상한

기안자는 diag6 진단·수치 감사·설계 감사, 선행 계약·판정, 관련 판독기와 봉인 생산 소스에 노출됐다. 표본은 이 노출 뒤 고정한다. 이미 알려진 1206 합치와 line 471957 반례의 재현은 새로운 예측 성공으로 세지 않는다.

이번 단의 생산 상태 변경과 스펙트럼 잔차 감소는 **0**이다. 다음을 선취하지 않는다.

- NRB의 생산 승격이나 기본값 전환.
- E′·R′·X 중 어느 상태의 물리적 정답.
- 공개 CMFGEN 스펙트럼·준위 인구와의 일치.
- 전체 생산 해의 재현.
- 전체 결합계의 직접·간접 인과 기여율.
- 반복 수렴, 두 씨앗의 수렴해 동일성, 씨앗 기억 제거.
- 후속 NRB·\(T_{\rm pin}\) 런의 성공이나 실행 권한.

## §1. 경로·기준선·승계 사실

### 1.1 경로

```text
P = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn

C = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
T = C/source
E = C/Eprime/run1
R = C/Rprime/run1

B = /gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007
X = B/BT/run1

D = /gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008
O = /gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008

고정 표본 권위 = D/r3_lock_census.csv
새 CPU 러너 = O/tools/run_det_seedmem_scope.slurm
새 결과 = O/results/<실행 식별자>
새 집행 증거 = O/evidence/<집행 식별자>

CMFGEN 소스 참조 = /gpfs/kjhan/cmfgen_src/cur_cmf
```

O에는 사용자 결정문이 이미 있다. 새 결과 경로는 기존 경로를 덮어쓰지 않는다. 입력 런·선행 진단·기존 판독기의 출력 경로로 쓰지 않는다.

### 1.2 기준선·신원

기안 시 확인한 프로젝트 루트는 P이며, 저장소 HEAD는 다음과 같다.

```text
d0e024f25a45f1034be77aa3ac174d313c4e05b5
```

형식 기준은 `docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md`, 오프라인 설계 참조는 `docs/PREREG_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md`다.

런별 권위는 각 런의 다음 저장물로 연결한다.

- input/run_build_identity.json.
- input/resolved_lumina.exports.
- input/deck.sha256와 실제 소비 덱.
- input/fixed_te_profile.txt.
- STAGE_IDENTITY.txt, RUN_FOOTER.txt, stdout.log, stderr.log.
- W5와 캡처·physics_comparison의 manifest.

E′·R′는 T 계열이고 X는 B-T 변경판이다. X를 T와 같은 바이너리로 표시하지 않는다. 이번에 재사용하는 생산 함수·상수·관련 소비 경로가 각 실행 판본에서 같은지 해당 파일과 빌드 기록으로 확인한다.

운전자는 계산 장소에서 실제 소비 파일의 SHA-256과 실행 의존성을 기록한다. 기안 중 해시를 재계산한 것으로 기록하지 않는다.

### 1.3 승계 사실

- diag6 4회차는 폐합됐다. Fable 수치 감사는 기존 1206의 25쌍 합치와 line 471957 반례를 독립 재계산했다.
- 기존 P1 검증 범위는 line 600693·W5 다섯 셸·call 3–7이다.
- 기존 1206 최대 상대오차는 약 \(3.216\times10^{-4}\), sh24·call 3이다.
- line 471957은 Si III 993.52 Å, 준위 1→12다. sh10의 기존 NRB 모형 잔차는 call 3 약 −7.0%, call 4–7 약 −23.0%, +5.3%, −14.6%, −5.2%다.
- L2·D2는 관측상 추종 표지다. 직접 기전이나 간접 기전의 분류가 아니다.
- 기존 P2의 \(0.0043\) dex는 선택된 인구 측도의 작은 변화 기준이다. 수렴 기준으로 승격하지 않는다.
- 원천 합치에 사용하는 물질 세대 7은 복사 세대 6을 소비한 상태다. 최종 캡처의 복사 세대 7과 혼동하지 않는다.

위 수치는 기존 감사 결과의 재인용이다. 이번 새 판독 결과와 별도 열에 둔다.

### 1.4 직접 확인한 생산 관계

아래 행 번호는 T 판본의 대략적인 위치다.

| 근거 | 확인한 관계 |
|---|---|
| line_net_rate.c 약 219행 | 상위 인구·A·주파수로 방출을 구성하고 τ로 적분 불투명도를 구성 |
| line_net_rate.c 약 267행 | β와 \((1-\beta)/\tau\)의 안정적인 동반 계산 |
| line_net_rate.c 약 299행 | \(\bar J=\beta J_c+L_l\)의 장부 구성 |
| line_net_rate.c 약 443행 | NRB 율은 continuum_term·τ와 원자계수를 소비하며 장부 일치를 검사 |
| lumina_cmfgen.c 약 9800행 | 같은 생산 결과의 \(J_c,\beta,L_l,\bar J,\tau_{\rm eff}\)를 W5에 기록 |
| lumina_plasma.c 약 19190행 | off 율과 NRB 순수 함수 호출의 실제 선택 |
| lumina_plasma.c 약 19260행 | 충돌식 분기와 per-line 충돌 계수 |
| lumina_plasma.c 약 19690행 | 일반 col_data 행렬 추가의 활성 조건 |
| lumina_plasma.c 약 21420행 | mapped NLTE 인구에서 τ와 저장 선 원천을 생성 |

소스의 관계와 저장 실행에서 해당 관계가 적용됐다는 권위 확인은 구별한다.

## §2. 식별 가능성·표본·세대의 사전 고정

### 2.1 표본 집합

주 표본은 D의 r3_lock_census.csv에서 다음 조건을 모두 만족하는 **기존 행의 키**로 고정한다.

\[
Z\in\{14,16\},\qquad ion=2,\qquad
shell\in\{10,15,19,24,30\},\qquad
beta2\_E<10^{-3}.
\]

- 기존 저장 집계 기준 Si III 86셀, S III 424셀, 합계 **510 선·셸 쌍**이다.
- line 600693의 다섯 셸은 재현 대조다.
- 1206 밖의 주 조사 집합은 **505셀**이다.
- line 471957·sh10은 필수 반례다.
- L2·D2·잔차·차이 유지율을 추가 선택 조건으로 쓰지 않는다.
- 파장 제한이나 새로운 이온을 추가하지 않는다.

이 집합은 원래 diag6 census의 씨앗 차이·자료 유효성 선택을 승계한다. 따라서 전체 두꺼운 Si III·S III 모집단의 무편향 표본이라고 주장하지 않는다.

기존 CSV의 고정 키를 정렬해 sample.json에 먼저 기록한다. 위 개수·필수 키·덱 신원의 불일치는 적격성 실패다. 키를 다시 골라 개수를 맞추지 않는다.

결손·적용 불가·반례가 생겨도 510키를 분모에서 제거하지 않는다.

### 2.2 연결 전이와 준위의 범위

다준위 조사에 필요한 보조 집합은 주 표본 전이의 하·상준위에 연결된 **같은 Si III·S III 이온의 전이 전부**다. 덱과 생산 활성 분기로 결정한다.

- 셸은 해당 표적 셸이며 W5 다섯 셸 밖으로 확대하지 않는다.
- 보조 전이는 주 표본의 성공 건수에 넣지 않는다.
- 연결 전이를 잔차가 큰 순서로 선택하지 않는다.
- 같은 준위·전이가 여러 표적에 연결되면 한 번 복원·계산하고 재사용한다.
- 누락된 연결 전이를 0으로 두지 않는다.

준위 변화 측도의 대상은 주 표본의 하·상준위 키를 중복 제거한 고정 집합이다.

### 2.3 call·세대

| 용도 | 고정 범위 | 연결 |
|---|---|---|
| 표본 신원·과거 씨앗 표지 | W5 call 1–7 | call 1은 INIT이며 β·\(J_c\)·\(L_l\) 모형 입력이 아님 |
| 원천·율 모형 비교 | E′·R′·X call 3–7 | 예측 입력은 call \(k-1\), 물질 배경은 인구 세대 \(k\) |
| 국소 율 교차 대입 | E′·X call 2–6 | 대응 물질 응답은 세대 3–7 |
| 실제 R′ 율 재구성 | R′ call 2–6 | R′ 원천 모형의 입력 |
| 다준위 수지 | 세대 7 | 인구는 세대 7, 율의 복사 입력은 call 6 |
| 이온합·전자밀도 변화 | 인구 세대 3–8 | 인접 변화 3→4부터 7→8 |
| 복사장·복원 준위 변화 | call·물질 세대 3–7 | 인접 변화 3→4부터 6→7 |
| 독립적인 인구 복원 대조 | 세대 7 캡처 | W5 call 7의 물질 신원과 연결 |

physics_DET_iter0000부터 iter0005까지의 인구 세대는 3부터 8이다. 파일명 산술만 믿지 않고 manifest의 population_generation과 radiation_generation을 검사한다.

call 2에 필요한 세대 2 전자밀도를 세대 3으로 대신하지 않는다. call 8, 세대 8 W5, 세대 6 최종 캡처를 만들어내지 않는다.

### 2.4 식별 가능성 선행 판정

수치 가설을 검사하기 전에 다음 표의 상태를 셀·세대별로 출력한다.

| 질문·양 | 등록 상태와 조건 |
|---|---|
| W5 원천 장부·β·\(J_c\)·\(L_l\) | 해당 call의 정상 Sobolev 행과 신원이 있으면 식별 가능 |
| 2준위 조건부 원천 | call 3–7의 이전 복사 입력과 같은 물질 세대 \(n_e,T_e\), 원자계수·충돌 분기가 확보되면 식별 가능 |
| 생산 NRB 율의 국소 자기항 의존성 | 정상 장부와 순수 함수 적용 조건이 있으면 call 2–6에서 식별 가능 |
| W5에 대응하는 mapped 준위 인구 | §3.2의 역산 조건과 생산 인구 권위가 확인된 셀·세대에서 대수적 복원 가능 |
| 세대 7 연결 전이 수지 | 모든 필요한 인구·율·활성 분기가 같은 세대로 확보되는 범위에서 식별 가능 |
| 세대 2 충돌 포함 물질 응답 | 필요한 \(n_e\) 저장 권위가 없으므로 이번 등록에서는 **식별 불가** |
| INIT call 1의 NRB 원천 분해 | INIT에는 해당 Sobolev 장부가 없으므로 **식별 불가** |
| 전체 결합 사상 \(G(J,z)\)의 실제 교차 응답 | 이번 입력은 모든 소비 상태·내부 반복·제약을 재생하는 완전한 체크포인트가 아니므로 **식별 불가** |
| 각 경로의 씨앗 기억 인과 기여율 | 세 런의 동반 변화와 국소 함수만으로는 **식별 불가** |
| 두 씨앗의 NRB 수렴해 동일성 | NRB·부분 \(T_{\rm pin}\) 궤적과 두 수렴해가 없으므로 **식별 불가** |
| 공개 스펙트럼 변화의 원인·방향 | 이번 측정 범위에서 **식별 불가** |

마지막 네 질문은 이번 단의 미완 계산 목록으로 남겨 반복하지 않는다. 등록 단계에서 코드 5로 닫고, 어떤 추가 증거가 필요할지만 기록한다.

국소 함수 또는 축약 모형의 교차 대입을 완전한 생산 \(G\)의 평가라고 명명하지 않는다.

## §3. 입력 권위·복원·소비 경로

### 3.1 입력 연결

다음 순서로 연결한다.

```text
런·빌드 신원
→ 유효 설정·덱·온도 레인
→ 선 행·준위·셸
→ W5 call·복사 세대·물질 세대
→ 같은 세대 물질 표 또는 캡처
→ 생산 함수 입력
→ 과학 판정
```

선 식별자는 헤더를 제외한 0-based 데이터 행 권위와 line_id 열을 함께 확인한다. \(Z\), ion, 하·상준위, 주파수도 대조한다.

- 같은 키의 모순 중복, 세대 교환, 잘못된 덱 연결은 X다.
- 실제 저장 결손은 해당 명제의 코드 5다.
- SRCE_CHK 또는 다른 적용 분기는 원래 상태를 보존한다. 정상 NRB 식을 강제로 적용하지 않는다.
- 생산 함수의 장부 일치 검사는 세대 검사의 대체가 아니다.
- 최종 캡처의 mapped·원천 대체·transaction·세대 조건을 확인한다.

### 3.2 같은 세대 W5에서의 인구 복원

준위별 파일이 없다는 이유만으로 모든 준위 인구를 식별 불가로 처리하지 않는다. 정상 mapped 선의 생산 장부는 일부 인구를 대수적으로 보존한다.

생산 함수가 반환하는 동반값을

\[
q(\tau)=\frac{1-\beta}{\tau}
\]

라 하되, 작은 τ에서는 생산 함수의 실제 다항식·연산 순서를 사용한다.

정상 양의 흡수, SRCE_CHK 미적용, \(A_{ul}>0\), \(f_{lu}>0\), 생산 인구 권위가 확인된 경우,

\[
\eta_\ell=
\frac{L_l}{(ct_{\rm exp}/\nu_\ell)q(\tau_\ell)},
\]

\[
n_u=\frac{4\pi\eta_\ell}{h\nu_\ell A_{ul}},
\qquad
n_l=
\frac{\tau_\ell}
{C_{\rm Sob}f_{lu}\lambda_\ell t_{\rm exp}}
+\frac{g_l}{g_u}n_u.
\]

이는 **그 W5 물질 세대의 생산 인구 복원값**이다. 별도 인구 파일의 직접 실측이라고 쓰지 않는다.

필수 확인은 다음과 같다.

1. τ와 상위 인구가 같은 생산 권위·물질 세대에서 왔다는 연결.
2. 역산값을 수정하지 않은 생산 material·radiation 함수에 넣은 재구성.
3. 세대 7에서는 별도 캡처의 τ·S 역산과 같은 준위 인구의 양립성.
4. 여러 선에서 복원한 같은 준위 인구 구간의 양립성.

세대 7 대조에만 성공했다고 세대 3–6의 신원 검사를 생략하지 않는다. 역산 특이점·양수성 미분리·적용 권위 결손은 사유와 함께 4 또는 5로 남긴다. 다른 세대 인구를 대입하지 않는다.

### 3.3 충돌 자료 소비

다음 세 사실을 분리한다.

- 덱에 충돌 자료가 존재한다.
- 로더가 자료를 읽었다.
- 이번 율 행렬이 그 자료를 실제 소비했다.

각 런의 유효 설정, 기본값을 정하는 함수, 로더 기록, 행렬 조립의 활성 조건을 연결한다. Si III·S III에 대해 per-line 근사, 일반 col_data 추가, 중복 억제, 비활성 분기를 기록한다.

기존 `det_siiii_rates_coll`은 해당 봉인 충돌 분기가 실제 적용되는 경우에만 import하여 사용한다. 이를 모든 이온·분기에 통용되는 생산 충돌 함수로 간주하지 않는다.

기존 충돌식의 원자 모형 분기는 원문대로 보존한다. 새 clamp/floor, 음수 제거, 작은 분모 대체, 결손의 0 채움은 넣지 않는다. 충돌표를 임의로 켜거나 다른 표로 바꾸는 감도 시험도 하지 않는다.

### 3.4 다준위·선 중첩·연속체의 식별 수준

| 후보 설명 | 이번에 확인할 내용 | 허용되지 않는 귀속 |
|---|---|---|
| 다준위 결합 | 연결 전이·실제 solve 단위·세대 7의 복원 가능한 순유량 | 연결선이 있다는 이유만으로 불일치 원인 확정 |
| super-level | 표적 준위가 독립 미지수인지, 같은 묶음인지, 분배 제약이 무엇인지 | 개별 준위의 수지가 곧 독립 SE 행이라고 가정 |
| 충돌 자료 소비 | 실제 활성식·소비 자료·중복 억제 | 파일 존재를 소비 증거로 사용 |
| 선 중첩 | 국소 연산자의 명시적 결합과 배경장을 만드는 경로 | NONOVERLAP이라는 이름만으로 모든 간접 선 영향을 배제 |
| 연속체 배경 | 저장 \(J_c\)가 국소 율에 들어가는 영향 | \(J_c\) 차이를 특정 연속체 물리 과정의 인과 기여로 명명 |
| 구현·권위 | 장부·세대·인구 소유자·원자계수 일치 | 입력 연결 실패를 물리적 반증으로 처리 |

CMFGEN의 전체 전달·반복계를 재구현하지 않는다. 이번 질문에 필요한 관계가 Lumina 봉인 소스와 실행 기록으로 정해지면 추가 CMFGEN 소스 조사를 늘리지 않는다.

## §4. 2준위 모형·반례·다준위 수지

### 4.1 원천 정의

\[
a_\ell=\frac{g_l}{g_u},\qquad
r_\ell=
\frac{h\nu_\ell B_{lu}/(4\pi)}
{C_{\rm Sob}f_{lu}\lambda_\ell\nu_\ell/c}.
\]

측정 원천은

\[
S_{\tau,t,k}=\frac{L_{l,t,k}}{1-\beta_{t,k}},
\qquad
\widetilde S_{E,t,k}=\frac{S_{\tau,t,k}}{r_\ell}.
\]

\(1-\beta\)가 0과 분리되지 않으면 이 원천 표현은 판정하지 않는다. §3.2의 인구 역산 가능성과 원천 표현의 가능성을 별개로 기록한다.

이번의 \(\widetilde S_E\)는 생산 불투명도 관계에 맞춘 비교량이다. 덱의 유한 정밀도 계수에 대해 Einstein 관계를 강제로 정확하게 만들지 않는다.

인구비 \(x=n_u/n_l\)로부터 비교할 원천은

\[
\mathcal S_\ell(x)=
\frac{A_{ul}}{B_{lu}}
\frac{x}{1-a_\ell x}.
\]

\(A\), \(B\), ν, λ의 직렬화 차이와 \(B_{ul}-a_\ell B_{lu}\)를 보존한다. 익숙한 Planck 형태로 치환해서 생기는 차이를 수리값으로 사용하지 않는다.

### 4.2 조건부 2준위 비교

call \(k=3,\ldots,7\)에 대해 같은 런의 call \(k-1\) 복사 입력과 인구 세대 \(k\)의 \(n_e,T_e\)를 사용한다.

\[
x_{\rm off}=
\frac{B_{lu}\bar J_{k-1}+C_{lu}(k)}
{A_{ul}+B_{ul}\bar J_{k-1}+C_{ul}(k)},
\]

\[
x_{\rm NRB}=
\frac{B_{lu}\beta_{k-1}J_{c,k-1}+C_{lu}(k)}
{A_{ul}\beta_{k-1}
+B_{ul}\beta_{k-1}J_{c,k-1}+C_{ul}(k)}.
\]

실제 NRB 계수의 중앙값은 생산 `line_net_bracket_rates` 호출로 얻는다. 적용 불가 반환을 정상 NRB 결과로 바꾸지 않는다.

- E′·X에는 off 모형을 비교한다.
- R′에는 NRB 모형을 비교한다.
- R′의 방사 전용 값은 충돌 포함 값과의 차이를 해석하는 대조로만 둔다.
- 출력 원천에는 call \(k\)의 β, 예측 율에는 call \(k-1\)의 β를 사용한다.

주 판정량은

\[
d_{t,\ell,s,k}
=
\frac{\widetilde S_{E,t,\ell,s,k}}
{\mathcal S_\ell(x_{t,\ell,s,k})}-1,
\qquad |d|\le10^{-3}.
\]

이는 **저장 배경 조건부 모형 적용성**이다. \(n_e,T_e,\beta,J_c\)까지 예측한 시험이 아니다.

모형이 실패한 셀도 원천·율·입력·부호·오차 구간을 그대로 출력한다. 그 셀을 제외한 평균으로 전칭 합치를 주장하지 않는다.

### 4.3 line 471957·sh10의 필수 처리

call 3–7을 모두 검사한다. 기존 call 3 반례와 양·음 잔차가 공존했던 이력을 함께 보존한다.

call 3에서는 추가로 다음을 확인한다.

- \(\widetilde S_{E,\mathrm{meas}}\)와 직전 \(J_c\)의 순서.
- 실제 소비 충돌식의 상세평형 관계.
- 상세평형 2준위 모형이 주는 원천 범위와 측정값의 양립성.
- 직렬화 계수 차이를 보존한 결과와 기존 진단의 근사식 차이.

기존 약 −7%를 새 값의 합격 목표로 삼지 않는다. 다른 결과가 나오면 입력·식·계수 차이를 먼저 설명하고, 기존 반례가 해소됐다고 자동 선언하지 않는다.

### 4.4 세대 7의 다준위 수지

주 표본의 각 전이에 대해 세대 7 인구와 call 6 복사 입력으로 다음을 계산한다.

\[
F_{\ell}=n_l P_{lu}^{\ell}-n_u P_{ul}^{\ell}.
\]

상준위로 들어가는 방향을 양으로 둔다. \(P\)에는 확인된 생산 방사·충돌 계수를 넣는다.

각 하·상준위에 대해 표적 전이 이외의 연결 전이 순유량도 합산한다. 같은 준위쌍의 다른 선과 별도 충돌 채널을 누락하거나 중복 계상하지 않는다.

판독은 다음 수준으로 제한한다.

1. **표적 전이만의 균형:** \(F_\ell\)이 0과 분리되는가?
2. **다른 연결 전이의 보상:** 반대 부호의 합이 있는가?
3. **닫힌 수지 여부:** 필요한 모든 활성 채널과 solve 제약이 확인되는가?
4. **남는 잔차:** 누락·제약·저장 배경의 한계 때문에 무엇을 설명하지 못하는가?

독립 solve 준위이며 활성 항 전체가 확보되고 표적 순유량이 0과 분리될 때만,

\[
\frac{|F_\ell+F_{\rm other}|}{|F_\ell|}\le10^{-3}
\]

을 「저장 배경에서 다른 전이가 표적 불균형을 보상한다」는 등록 기준으로 사용한다. 구간 전체로 판정한다.

활성 항이 일부만 확보되면 부분 수지로 표시하며, 남는 잔차를 미측정 연속체 항으로 이름 붙이지 않는다. super-level 내부 준위에 독립 행의 폐합을 요구하지 않는다.

이 결과는 저장 상태의 율 수지에 관한 것이다. 외부 반복에서 기억이 어떤 경로로 전달됐는지의 인과 분해는 아니다.

## §5. 고정 배경 교차 대입·P2·후속 설계

### 5.1 생산 순수 함수의 교차 대입

수정하지 않은 봉인 line_net_rate.c를 작은 CPU 공유 라이브러리로 빌드해 판독기에서 호출한다. 생산 바이너리를 빌드하거나 실행하지 않는다.

call \(j=2,\ldots,6\)에서

\[
c_t=\beta_tJ_{c,t},\qquad L_t=L_{l,t},
\]

\[
z_t=(c_t,\tau_t,n_{e,t,j+1},T_{e,t,j+1},
\text{원자계수·적용 권위})
\]

로 둔다.

E′·X의 같은 선·셸·call에 대해

\[
H(L_E,z_E),\quad H(L_X,z_E),\quad
H(L_E,z_X),\quad H(L_X,z_X)
\]

를 모두 평가한다. 교차 상태의 \(\bar J\)는 반드시 해당 조합의 \(c+L\)로 재구성한다. \(\bar J\)만 바꿔 장부를 깨는 시험을 자기항 시험으로 쓰지 않는다.

H의 출력은 다음 두 층이다.

- 생산 함수가 반환한 방사 율 계수.
- 그 계수와 등록 충돌식으로 계산한 2준위 조건부 원천.

등록 기대는 다음과 같다.

- 유효 NRB 적용 영역에서 \(c,\tau\)를 고정하면 \(L\) 교환으로 생산 방사 율 계수는 변하지 않는다.
- off 방사 율은 \(c+L\)을 소비하므로 해당 변화가 남는다.
- 2준위 NRB 원천의 같은 불변성은 축약 모형의 구성상 성질이다. 실제 인구의 기억 제거가 아니다.

NRB 율의 불변성은 같은 실행에서 같은 나머지 입력을 사용하는 함수 출력의 정확한 동일성으로 검사한다. \(10^{-3}\)을 이 함수 불변성의 허용오차로 사용하지 않는다.

### 5.2 조건부 의존성과 미귀속 잔차

상호작용을 한쪽 순서에만 배정하지 않도록 다음 대칭 분해를 사용한다.

\[
D_L=\frac12\{
H(L_X,z_E)-H(L_E,z_E)
+H(L_X,z_X)-H(L_E,z_X)\},
\]

\[
D_z=\frac12\{
H(L_E,z_X)-H(L_E,z_E)
+H(L_X,z_X)-H(L_X,z_E)\}.
\]

\[
D_L+D_z=H(L_X,z_X)-H(L_E,z_E).
\]

E′·X의 실제 원천 차이와 비교할 때는 모형 잔차 차이도 남긴다.

\[
\Delta\widetilde S_{E,\mathrm{meas}}
=D_L+D_z+\Delta e_{\rm model}.
\]

이 합은 대수적 장부다. 합이 맞는 것을 모형 검증이나 인과 분리 성공으로 세지 않는다.

- \(D_L\): 지정한 τ·배경을 고정한 **국소 방출항 의존성**.
- \(D_z\): 지정한 나머지 입력의 **조건부 배경 의존성**.
- \(\Delta e_{\rm model}\): 축약 모형으로 설명하지 못한 부분.

τ 자체가 인구에 의존하므로 \(D_z\) 전체를 다른 물리 경로의 간접 기억으로 명명하지 않는다. 실제 두 궤적의 모든 직접·간접 경로를 분리했다는 주장은 코드 5로 유지한다.

### 5.3 P2의 저장 자료 점검

대상은 W5 다섯 셸이다.

- 이온합: Si II·Si III·S II·S III의 NLTE 합.
- 전자밀도: 같은 셸의 \(n_e\).
- 복사장: 고정 510키의 \(\bar J\).
- 준위 인구: §2.2의 고정 하·상준위 집합 중 같은 세대로 적격하게 복원되는 값.

양의 양 \(q\)에 대해

\[
m_q(g)=\left|\log_{10}\frac{q(g)}{q(g-1)}\right|
\]

를 계산한다. 비교선은 기존과 같은 \(0.0043\) dex다.

- 이온합·전자밀도는 \(g=4,\ldots,8\).
- 복사장·복원 준위는 \(g=4,\ldots,7\).
- 공통 범위의 두 연속 변화는 \(g=4,5\), \(5,6\), \(6,7\)로 고정한다.
- \(g=7,8\)의 인구 측도만으로 복사·준위의 두 연속 정상성을 선언하지 않는다.

이번 고정 Si·S 이온합에는 존재비 문턱으로 셀을 사후 제거하지 않는다. 양 끝이 모두 증명된 정확한 0이면 구조적 영으로 기록한다. 한쪽만 0이고 다른 쪽이 양수로 분리되면 작은 변화 조건은 반증이다. 로그 계산을 위해 양수 floor를 넣지 않는다.

복원 결손은 해당 준위의 5이며, 유효한 다른 준위의 반례를 지우지 않는다.

### 5.4 다음 P1·P2로 넘기는 결정 규칙

다음은 이번 결과의 소비 규칙이며 후속 런의 실행 승인이 아니다.

| 이번 결과 | 다음 계약에 넘길 변경 |
|---|---|
| 1206의 기존 25쌍 적용성 유지 | P1 원천 합치의 기준 범위와 \(10^{-3}\) 비교선 유지 |
| 1206 밖 고정 이온 집합에서 확정 반례 | 해당 집합에 원천 합치 전칭 조건을 일반화하지 않음 |
| 1206 밖 Si III 또는 S III의 고정 집합 전체가 call 3–7에서 생존 | 그 **집합 전체**를 다음 계약의 조건부 확장 후보로 제시. 통과 셀만 골라 새 집합을 만들지 않음 |
| 생산 NRB 율의 국소 자기항 불변성 확인 | P1을 「율의 자기항 제거」와 「원천의 축약 모형 합치」로 분리 |
| 불변성은 성립하지만 원천 모형은 반증 | NRB 구현 실패와 축약 모형 실패를 구별하도록 P1 반증 문구 변경 |
| 다준위 수지의 적격한 보상 확인 | 해당 셀·세대의 모형 생략 설명에만 사용 |
| 수지 결손 또는 잔차 유지 | 원인을 미귀속으로 남기고 필요한 최소 저장량을 명시 |
| 이온합·\(n_e\) 조건을 만족하지만 복사·준위 조건은 반증 | 다음 P2에 해당 복사·준위 측도를 별도로 등록해야 함 |
| 모든 등록 측도가 작음 | 「W5 등록 측도의 두 연속 작은 변화」만 허용 |
| 이온합·\(n_e\)부터 기준 미달 | P2 정상성 미달. 다른 측도의 작은 변화로 대체하지 않음 |

허용오차를 잔차의 최대값·분위수·사후 적합으로 정하지 않는다. \(10^{-3}\)과 \(0.0043\) dex의 변경이 필요하다는 제안은 별도 근거와 사전 개정 대상이다.

call 8 이후의 P1, W5 밖의 P2, 전체 복사장 수렴, 두 씨앗 수렴해 비교는 이번 증거의 검증 범위 밖이다.

## §6. 기대치·수치 기준·과학 코드

### 6.1 등록 기대와 반증

| 명제 | 기대 | 반증 또는 제한 |
|---|---|---|
| H-MODEL | 기존 1206 합치는 유지되지만 1206 밖 전칭 합치는 성립하지 않을 수 있음 | 각 셀·call의 \(10^{-3}\) 위반을 그대로 보존 |
| H-COUNTEREXAMPLE | line 471957·sh10은 NRB 축약 모형의 적용 제한을 재현할 것으로 기대 | 새 결과가 다르면 기존·새 식과 입력 차이를 설명하고 기존 결론을 자동 삭제하지 않음 |
| H-LOCAL | 유효 NRB 함수는 고정 \(c,\tau\)에서 \(L\) 교환에 불변 | 같은 적격 입력에서 출력 변화가 확인되면 반증 |
| H-MULTILEVEL | 일부 원천 불일치는 연결 전이 수지와 양립할 수 있음 | 완전한 수지가 없거나 등록 폐합 조건을 충족하지 못하면 원인 확정 불가 |
| H-P2 | 이온합·전자밀도의 작은 변화와 복사·준위의 작은 변화는 별도 명제 | 어느 축의 확정 반례도 다른 축으로 상쇄하지 않음 |

H-MULTILEVEL은 모든 불일치가 다준위 결합으로 설명된다는 전칭 가설이 아니다. 표적별 수지 명제로 판정한다.

### 6.2 수치 처리

- 텍스트 직렬화, 부동소수점 연산, 역산, W5의 오차 상한을 전파한다.
- Jbar_err_upper는 생산 주석상 전체 산술 오차를 포괄하지 않는다. 이를 유일한 오차로 쓰지 않는다.
- exp·log와 역산의 작은 직접시험은 독립 고정밀 참조로 확인한다.
- ν와 \(c/\lambda\), Einstein 계수, \(r_\ell\)을 임의로 같게 만들지 않는다.
- \(1-\beta\), 인구 차이, 순유량, 상대오차 분모의 0 분리를 검사한다.
- 모델의 생략 오차를 수치 오차 구간에 넣지 않는다.
- 원천·율·인구의 음수·비유한·특이점을 clamp/floor로 수리하지 않는다.
- 중앙값·구간·부호·정의 가능 여부를 함께 출력한다.

### 6.3 과학 코드

결과 스키마는 `DET_SEEDMEM_SCOPE_RESULT_V1`이다.

| 코드 | 명칭 | 의미 |
|---:|---|---|
| 0 | `SURVIVES` | 적격한 구간 전체가 등록 명제를 만족 |
| 1 | `REFUTED` | 적격한 구간으로 등록 명제 위반이 분리 |
| 4 | `UNRESOLVED` | 입력은 있으나 구간이 경계·영·부호와 겹쳐 미분리 |
| 5 | `NOT_IDENTIFIABLE` | 필요한 저장 입력·권위·정의가 부족하거나 등록 단계에서 식별 불가로 닫음 |
| X | 적격성 실패 | 잘못된 연결·구현·선택·결과 구조 |

전칭 명제는 다음 순서로 집계한다.

1. 확정 반례가 있으면 1.
2. 반례가 없고 필요한 결손이 있으면 5.
3. 결손도 없고 경계 미분리가 있으면 4.
4. 모두 만족하면 0.

X가 있는 범위에는 적격한 최종 집합 판정을 부여하지 않는다. 이미 확보된 적격 반례는 별도로 보존한다.

구성상 항등식, 과거 수치 재인용, 집행 처분, 미실행 미래 기대는 과학 코드에 섞지 않는다. 아직 실행하지 않은 항목은 `FUTURE_TEST`다.

## §7. 최소 변경집합·출력

### 변경집합 — DET-SEEDMEM-SCOPE (표)

새 구현 심볼의 접두사는 `det_seedmem_scope_`로 한다.

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `scripts/det_seedmem_scope_readout.py` | 새 판독기·내장 직접시험 — 신설 | 고정 표본·식별 가능성·생산 순수 함수 연결·복원·모형 잔차·교차 대입·세대 7 수지·P2·과학 코드·입출력 신원 |
| `/gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008/tools/run_det_seedmem_scope.slurm` | CPU 러너 — 신설 | 환경 초기화·순수 함수 CPU 라이브러리 빌드·selftest·analyze·시간·자원·rc·출력 보존 |

### 변경집합 끝

생산 소스, 기존 판독기, 기존 게이트, 원자 자료, 기존 런·진단 산출은 수정하지 않는다.

직접시험은 새 판독기의 selftest 명령에 포함한다. 별도 분석기·보조 진단 스크립트·새 생산 기능을 표 밖에 추가하지 않는다.

기존 판독기의 필요한 순수 함수는 import할 수 있다. 실제 함수·경로·파일 신원을 기록한다. 기존 analyze 전체를 호출해 고정 셸·세대·대상 제한을 우회하지 않는다.

순수 함수 CPU 라이브러리, 가상 입력, 실행 로그는 등록 도구가 만드는 결과물이다. 라이브러리는 봉인 line_net_rate.c와 line_net_rate.h를 수정 없이 소비한다.

| 필수 산출 | 내용 |
|---|---|
| sample.json | 고정 510키, 대조·반례·이온 집합, 보조 연결 전이·준위, 선택 권위 |
| identifiability.csv | 질문·키·세대별 필요 입력·가용성·권위·5의 사유 |
| model.csv | 런·키·call별 측정 원천·예측·잔차 구간·코드 |
| pathways.csv | 소비 분기·세대 7 인구·표적 및 다른 전이 수지·미귀속 잔차 |
| cross_substitution.csv | 고정 배경의 조합 입력·생산 율·축약 원천·조건부 분해 |
| stationarity.csv | 고정 P2 측도·세대쌍·구간·반례·결손 |
| next_contract.json | P1 대상·허용오차·반증 문구, P2 측도·세대·필요 출력, 주장 상한 |
| selftest.json | 정상 대조·주입 결함·기대/실제 실패·rc·시험 신원 |
| result.json | 명제별·집합별 코드, 적격성, 식별 불가, 미완·처분 근거 |
| manifest.json | 계약·도구·입력·생산 함수·출력 신원, argv·환경·비용 |

계약·발주·판정·감사문·OWNER.txt·비용 장부는 집행 문서다.

## §8. 게이트·직접시험

### 8.1 내부 게이트

| 게이트 | 방지할 실패 | 통과 조건 |
|---|---|---|
| G-LINK | 표본·런·선·셸·세대·권위 혼동 | §1–§3의 연결과 고정 집합을 충족하고 신원 근거를 보존 |
| G-READOUT | 잘못된 식·결손 은폐·과도한 귀속 | §4–§6의 실제 판독 경로와 §8.2의 정상·음성 대조 통과 |

과학 가설의 반증은 게이트 실패가 아니다. 반증을 정확히 기록하지 못하는 것이 게이트 실패다.

식별 불가로 사전 등록한 질문을 5로 출력했다는 이유로 게이트를 실패시키지 않는다.

### 8.2 직접시험

작은 생산 형식 가상 입력으로 실제 analyze 경로를 호출하고 저장 산출을 검사한다. 정상 사례가 통과하며, 아래 주입 결함은 실제 검사가 FAIL을 내야 한다.

| 시험 묶음 | 정상 대조 | 주입 결함과 요구 결과 |
|---|---|---|
| 표본 고정 | 전체 키·필수 반례 유지 | 반례 삭제·실패 셀 제외·다른 셸 추가 → G-LINK FAIL |
| 세대 연결 | \(k-1\) 복사와 \(k\) 물질 | 이전 β를 현재 β로 교환, 물질 7을 8로 교환 → FAIL |
| 결손 처리 | 결손은 5, 기존 반례는 1 유지 | 세대 2 \(n_e\)를 세대 3으로 채움, 결손을 0으로 채움 → FAIL |
| W5 역산 | 알려진 인구→생산 장부→역산 구간 포함 | \(r_\ell\) 생략·잘못된 A·준위 교환 → FAIL |
| 독립 권위 대조 | W5 복원과 세대 7 캡처 양립 | mapped 권위·원천 대체·캡처 세대 변조 → FAIL |
| 생산 NRB 연결 | 정상 함수의 장부·율 재현 | β·유도방출 생략 또는 장부 불일치 무시 → FAIL |
| 자기항 교차 대입 | \(c,\tau\) 고정, \(L\)과 \(\bar J=c+L\)의 일관된 교환 | \(L\) 교환 때 \(c\)도 바꾸고 고정 배경으로 표시 → FAIL |
| 다준위 수지 | 알려진 연결 전이의 부호·합·결손 보존 | 연결 전이 누락·중복·부분합을 완전 수지로 표시 → FAIL |
| 충돌 소비 | 비활성 표는 소비하지 않음 | 자료 파일 존재만으로 col_data 소비 처리 → FAIL |
| P2 | 두 연속 변화와 정확한 영·결손 처리 | 한 변화만으로 통과, 문턱 이탈·영 교차 셀 삭제 → FAIL |
| 수치·코드 | 경계 중첩 4, 반례 1, 결손 5 | 오차 구간 제거·허용오차 사후 확대·코드 혼합 → FAIL |
| 보존 | 새 결과 경로에만 기록 | 기존 결과 경로 재사용 → 쓰기 전에 거부 |

주입은 가상 입력 또는 시험 전용 분기에 한정한다. 봉인 소스·실제 런·기존 CSV를 변조하지 않는다.

시험은 「결함 주입 플래그가 켜졌음」만 확인하지 않는다. 결함이 실제 판독 경로에 도달했고, 등록 검사에서 실패했다는 산출·사유·rc를 남긴다.

### 8.3 기계 프리플라이트

기존 도구를 수정 없이 사용한다.

- `scripts/check_prereg_preflight.py`
- `scripts/check_gate_referents.py`
- `scripts/check_doc_canonical.py`
- `scripts/check_dispatch_preflight.py`
- `scripts/check_driver_hygiene.py`
- `scripts/check_gate_roster.py`

같은 문서 신원에 대한 dispatch 내부 prereg 검사는 중복 실행하지 않는다. 계약·발주·러너의 신원과 실제 argv·stdout·stderr·rc를 보존한다.

기계 프리플라이트 뒤, 구현 발주 전에 Fable fresh의 Q-GOAL·Q-LEAN 사전감사를 1회 받는다. 기계 검사는 과학식과 독립 사전감사를 대신하지 않는다.

## §9. 환경·비용·재현 명령

### 9.1 환경

CPU 시험·게이트·판독은 a10 CPU Slurm에서 수행한다. 수동 수치 실행 장소는 lageunha다. 로그인 노드에서는 파일 조회·소스 판독만 한다.

- 작업명은 `LUMINA_` 접두.
- 기본 제외 노드는 `--exclude=syn102`.
- CPU·메모리·시간·GPU 예약 0을 명시.
- 런 루트 O에 OWNER.txt 유지.
- module 초기화, Python·C 컴파일러·의존성·스레드·호스트·job ID 기록.
- 일반 파일 읽기·쓰기와 스트리밍 해시만으로 구현.
- 파일시스템 고유 기능, 공유 경로의 일괄 복사·재귀 탐색·특수 보호 장치에 의존하지 않음.
- lageunha에서는 부하를 먼저 확인하고 같은 자원 상한을 적용.
- 시간 측정은 Bash `time` 또는 확인된 동등 수단 사용.

순수 함수 CPU 빌드와 시험은 새 결과 경로 안에서만 한다. 생산 프로그램의 초기화·수송·물질 반복을 호출하지 않는다.

### 9.2 비용 상한

직전 REPAIR-OFFLINE의 누적 Slurm 223초와 BT-SEED 오프라인 시험 약 190초를 참고하되, 이번에는 고정 표본 확대·입력 신원·역산·감사 수리 여유를 포함한다.

| 항목 | 상한 |
|---|---:|
| 동시 계산 | 1작업 |
| 작업당 CPU | 4 |
| 메모리 | 16 GiB |
| 한 번의 전체 빌드·직접시험·판독 | 600초 |
| 전체 실행 배분 | 초회 600초＋수리·재현 3회×600초 |
| 독립 게이트·보조 확인 배분 | 누적 600초 |
| Slurm·수동 합산 실행 경과 | **3,000초** |
| CPU 예약량 | **\(10/3\) core·h 이하** |
| 새 도구·가상 입력·출력·로그 합계 | **256 MiB 이하** |
| GPU 예약·생산 실행 | **0 GPU·h, 0회** |

- 실패·직접시험·수리·재현·수동 계산·기계 게이트를 모두 합산한다.
- 같은 작업의 러너 시간과 Slurm 경과는 이중 합산하지 않는다. 둘 다 기록하고 비용에는 큰 값을 사용한다.
- 대기·문서 작성·사람의 감사 시간은 계산 경과와 구별한다.
- 초회 뒤에도 등록된 수리·재현 여력을 남긴다.
- 자동 반복 제출은 하지 않는다.
- 잔여 예산으로 완료할 수 없으면 실제 비용·미완 작업·필요량을 보고한다. 비용 부족을 과학 반증으로 바꾸지 않는다.

W5·덱은 필요한 키를 모아 파일별로 순회하며 선마다 다시 읽지 않는다. 캡처는 필요한 기록만 읽는다. 사용 입력과 생성 출력의 명시적 목록으로 신원·용량을 집계한다.

### 9.3 계획 CLI

아래 명령은 인터페이스 등록이며 기안자가 실행한 기록이 아니다.

```bash
P=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
O=/gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008
D=/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008
C=/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
X=/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/BT/run1

python3 -B "$P/scripts/det_seedmem_scope_readout.py" selftest \
  --production-source "$C/source/src" \
  --out "$O/results/$RUN_ID/selftest"

python3 -B "$P/scripts/det_seedmem_scope_readout.py" analyze \
  --prereg "$P/docs/PREREG_DET_SEEDMEM_SCOPE_2026-10-08.md" \
  --census "$D/r3_lock_census.csv" \
  --e-run "$C/Eprime/run1" \
  --r-run "$C/Rprime/run1" \
  --x-run "$X" \
  --production-source "$C/source/src" \
  --selftest "$O/results/$RUN_ID/selftest" \
  --out "$O/results/$RUN_ID/analysis"
```

RUN_ID는 러너가 정한 미사용 실행 식별자다. 별도 결과 선택·선 선택·잔차 임계값·call 절단 옵션은 만들지 않는다.

```bash
python3 -B "$P/scripts/check_prereg_preflight.py" \
  "$P/docs/PREREG_DET_SEEDMEM_SCOPE_2026-10-08.md" --root "$P"

python3 -B "$P/scripts/check_gate_referents.py" \
  "$P/docs/PREREG_DET_SEEDMEM_SCOPE_2026-10-08.md" --root "$P"

python3 -B "$P/scripts/check_doc_canonical.py" \
  "$P/docs/PREREG_DET_SEEDMEM_SCOPE_2026-10-08.md"

python3 -B "$P/scripts/check_dispatch_preflight.py" "$ORDER" --root "$P"
```

ORDER는 이번 계약을 소비하는 운전자 발주서다. 실제 게이트 묶음은 §8.3을 충족하고 원 명령을 기록한다.

러너는 selftest와 analyze를 모두 지원하며 다음 자원으로 제출한다.

```bash
sbatch \
  --job-name=LUMINA_DET_SEEDMEM_SCOPE \
  --partition=a10 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --gres=gpu:0 \
  --time=00:10:00 \
  --exclude=syn102 \
  "$O/tools/run_det_seedmem_scope.slurm" "$RUN_ID"
```

재현은 같은 봉인 입력·도구 신원으로 새 결과 경로를 사용하며 누적 비용에 포함한다.

## §10. 처분·완료 조건·후속 결정

### 10.1 필수 완료 내용

판정문 첫 부분에 다음을 명시한다.

- 생산 변경·GPU 실행·스펙트럼 잔차 감소는 0.
- 기존 결과에 노출된 계약이며 알려진 사실의 재현은 새 예측 성공이 아님.
- 1206 밖 모형 적용 범위와 확정 반례.
- 국소 함수에서 식별한 의존성과 전체 결합계에서 식별하지 못한 기여.
- 다음 P1·P2의 변경 제안과 그대로 남는 기준.
- 생산 승격·물리적 정답·공개 일치·후속 런 승인은 없음.

필수 본문은 다음을 포함한다.

1. 510키 전체의 적격성·결손·코드와 고정 분모.
2. 1206 대조 및 line 471957·sh10의 call 3–7 결과.
3. 다른 Si III·S III의 모형 적용성.
4. 충돌 소비·다준위·중첩·연속체에 대해 확인한 사실과 미귀속.
5. 세대 7 수지와 교차 대입의 입력·세대·적용 범위.
6. P2 측도별 변화와 두 연속 변화의 정확한 세대.
7. next_contract.json의 결정표.
8. 시험·게이트·신원·비용·보존·이탈.
9. astra 종료감사 및 실행 가능한 주장에 대한 운전자 재현 상태.

원인을 모두 확정해야만 완료되는 계약은 아니다. 다만 식별 불가의 근거와 후속 설계에 대한 귀결 없이 결손 목록만 내는 것은 완료가 아니다.

### 10.2 집행 처분과 과학 결과

| 처분 | 기준 |
|---|---|
| `PASS_LIMITED` | 등록 조사·직접시험·반증·식별 한계·후속 결정표를 적격하게 완료 |
| `REWORK` | 선택·세대·식·소비 권위·시험·귀속·비용 기록의 계약 위반 |
| `INCOMPLETE` | 필수 입력 또는 등록 자원으로 최소 산출을 완료하지 못함 |

**전체 처분의 상한은 `PASS_LIMITED`다.**

과학 코드 1이나 사전 등록한 5가 있어도 조사 자체를 적격하게 완료할 수 있다. 반대로 코드 0이 많아도 입력 연결·음성 대조·후속 결정표가 미완이면 완료 처분을 내리지 않는다.

### 10.3 다음 경로

이번 단은 next_contract.json을 사용자와 운전자의 다음 결정 입력으로 넘기고 종료한다.

후속 NRB 판정런을 선택한다면 별도 계약에서 다음을 확정한다.

- P1의 국소 율 명제와 원천 모형 명제.
- 대상·call·허용오차·위반 해석.
- P2의 대상·측도·두 연속 변화·최대 반복 도달 시 처리.
- 이번에 식별 불가였던 질문을 실제로 시험하려면 필요한 최소 저장량.
- 비교할 씨앗·물리 변경 하나·GPU 비용·실행 횟수.

상수 정합 수리, 충돌 자료 변경, 반복 연장, 추가 GPU 런은 이번 결과에서 자동 발생하지 않는다.

## §11. 집행 기록·분장

| 업무 | 담당 | 규칙 |
|---|---|---|
| 기안·개정 | astra | 모델 ID와 사전 노출 기록 |
| 경량 사전감사 | Fable fresh | Q-GOAL·Q-LEAN 1회. 미실시를 통과로 기록하지 않음 |
| 구현·직접시험·게이트·실행·판독·판정 | 운전자(Opus 5.5) | 구현·판정 겸임 명시 |
| 코드 감사 | 해제 | 정례 감사 추가 없음 |
| 종료감사 | astra | 계약 저자라는 이해관계 명시 |
| 실행 주장 재현 | 운전자 | 정확한 입력·도구·명령·출력 신원과 재현 상태 기록 |
| 범위·비용 확대·생산 변경·GPU 런 | 사용자 | 이번 계약으로 자동 승인되지 않음 |

종료감사 수리는 등록 변경집합과 누적 비용 안에서 수행한다. 현행 폐합 규약에 따라 1·2회차는 전건 반영하고, 3회차부터는 «폐합 차단»을 반영하며 나머지는 관측으로 기록한다. 그 규칙이 미달 목표나 남은 «폐합 차단»을 통과로 바꾸지는 않는다.

| 기안 시 집행 항목 | 상태 |
|---|---|
| 공유 문맥 전체 판독 | 완료 |
| 작업 장소·프로젝트 루트·HEAD 확인 | 완료 |
| 사용자 결정·diag6 폐합·수치 감사·설계 감사 판독 | 완료 |
| 선행 계약·관련 판정·판독기·생산 관계 판독 | 완료 |
| 새 표본 추출·해시 계산·수치 분석 | 미실행 |
| 파일화·기계 게이트 | 운전자 집행 전 |
| Fable fresh 경량 사전감사 | 미실행 |
| 구현·직접시험·CPU 실행 | 미실행 |
| 운전자 판정·astra 종료감사 | 미실행 |
| 생산 변경·GPU 런 | 이번 대상 밖 |

기안자는 파일 변경, 계산, 시험, 게이트, 작업 제출을 수행하지 않았다. 향후 항목을 예상 PASS로 채우지 않는다.

## §12. 기계 선언

선언은 변경집합·참조 실체와 §6.3의 코드 분할을 검사한다. 식별 가능성·세대 정렬·물리식·구간·귀속의 정확성은 직접시험과 종료감사 대상이다.

`v`는 과학 명제 하나의 코드다. 유효 코드는 0·1·4·5이며 결손·복수 코드·그 밖의 값은 X다. 코드 5와 결과 코드 자체의 결손을 구별한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SEEDMEM-SCOPE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_seedmem_scope_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008/tools/run_det_seedmem_scope\\.slurm)",
    "symbol": "det_seedmem_scope_",
    "roots": [
      "scripts/det_seedmem_scope_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008/tools/run_det_seedmem_scope.slurm"
    ],
    "expected_extra": [
      "scripts/det_seedmem_scope_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008/tools/run_det_seedmem_scope.slurm"
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
      "path": "/gpfs/kjhan/lumina/cmfgen_map/seedmem_scope_20261008/evidence/USER_DECISION_2026-10-08.txt"
    },
    {
      "path": "docs/PREREG_DET_SIIII_BT_SEED_2026-10-07.md"
    },
    {
      "path": "docs/PREREG_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md"
    },
    {
      "path": "docs/VERDICT_DET_SIIII_BT_SEED_2026-10-08.md"
    },
    {
      "path": "docs/VERDICT_DET_SIIII_REPAIR_OFFLINE_2026-10-07.md"
    },
    {
      "path": "docs/VERDICT_DET_SIIII_1206_MECH_2026-10-07.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008/LOOP_CLOSED.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008/DIAGNOSIS.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008/r4_fable_numaudit.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008/DRIVER_DESIGN_AUDIT_r4.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008/r3_lock_census.csv",
      "flags_existing": ["line_id", "shell", "beta2_E", "L2", "D2"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008/r3_lock_census.log"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag6_seedmem_20261008/r3_stationarity.py"
    },
    {
      "path": "scripts/det_siiii_btseed_readout.py"
    },
    {
      "path": "scripts/det_siiii_repair_readout.py",
      "flags_existing": ["det_siiii_repair_rC", "det_siiii_repair_rAC"]
    },
    {
      "path": "scripts/det_siiii_1206mech_readout.py",
      "flags_existing": ["det_1206mech_kappa", "det_1206mech_check_row"]
    },
    {
      "path": "scripts/det_siiii_rates_readout.py",
      "flags_existing": ["det_siiii_rates_coll", "det_siiii_rates_rad"]
    },
    {
      "path": "scripts/det_siiii_pop_readout.py",
      "flags_existing": ["recover", "capture_records", "prod_tau_S"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c",
      "flags_existing": [
        "line_net_sobolev_material",
        "line_net_cmfgen_exponx",
        "line_net_sobolev_radiation",
        "line_net_bracket_rates"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.h",
      "flags_existing": [
        "LineNetBracketRates",
        "LINE_NET_BRACKET_NOT_APPLICABLE"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c",
      "flags_existing": [
        "nlte_assemble_rate_matrix_impl",
        "omega_cmfgen_enabled",
        "nlte_coll_fix_enabled",
        "line_net_bracket_rates"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c",
      "flags_existing": [
        "w5_cont_lines_write",
        "line_net_sobolev_material",
        "line_net_sobolev_radiation"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_atomic.c"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina.h"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Eprime/run1"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/Rprime/run1"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/siiii_btseed_20261007/BT/run1"
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
