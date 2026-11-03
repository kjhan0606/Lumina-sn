# Lumina-sn 과거 문제와 해결 방법 색인

최초 정리: 2026-10-30. 같은 문제를 처음부터 다시 분석하거나, 진단용 구판을
보고 과거 수리가 사라졌다고 오인하지 않기 위한 프로젝트 메모다.
**과거에 무엇을 어떻게 해결했는지**와 **그 증거를 지금 어디까지 재사용할 수
있는지**를 구분한다. 현재 계산 상태·승인 한도는 최신 인수인계서와 실제 실행
기록을 따른다. 이 색인 자체가 새 계산 승인이나 최종 물리 합격 근거는 아니다.

## 새 문제를 조사할 때

1. 아래에서 증상·과정이 가까운 항목을 찾고, 연결된 정본의 해당 절만 먼저 읽는다.
2. 과거와 현재의 producer·소스/바이너리, 입력 물리·원자모형·장/물질 세대 중
   이번 문제에 관계있는 차이만 확인한다. 같은 증상이 같은 원인이라는 뜻은 아니다.
3. 기존 수리 재사용, 적용 대상이 다른 판본, 실제 회귀, 새로운 원인, 아직 미확인
   중 무엇인지 작업 기록에 짧게 구분한다. 과거 자료를 재사용할 수 없으면 그
   차이를 남기되, 과거에 해결한 사실 자체를 없애거나 처음 하는 일로 기록하지 않는다.
4. 의미 있는 원인·수리·비채택 결론이 바뀔 때 해당 항목을 갱신한다. 작은 확인마다
   새 문서·감사·계측기·승인 절차를 만들지 않는다. 원본 이력과 작성자는 보존한다.

초기 색인은 최근 CMFGEN/Si/S 작업에서 반복해 찾았던 문제를 우선 담았다.
미등재는 미실시를 뜻하지 않는다. 초기 TARDIS/MC 개발 이력은
[HISTORY](HISTORY.md), 날짜별 진행은 `HANDOVER_YYYY-MM-DD.md`에서 추가로 찾는다.
옛 모델에서의 조성·경험적 조정이나 당시 PASS를 현 CMFGEN 물리 계약에 옮기지 않는다.

## P01 음수 J와 외부 입사 경계

- 과거 처리: native의 음수 H/J·N/J를0으로 자르는 대신 실제 광선에서
  `H=hJ-Hinc`, `N=nJ-Ninc`를 만들고, 공간 기울기·첫 내부 flux·이전 주파수
  상수까지 연결하는 private affine 수리를 구현했다.
- 입증 범위: 10월19일 두 주파수 국소 수리는 통과했지만, 전체 연결은66069에서
  다른 MOM 불일치를 남겼다. 경계 부호만 복원하거나 이전 H를 임의 조정하는
  방법은 전체 수리로 채택하지 않았다.
- 재사용: 10월30일 78회 포획은 부모 MOM 산술의 진단 계열이며 affine 판본이
  아니다. 그 포획의 경계 절단만으로 과거 커밋 소실·새 회귀를 주장하지 않는다.
  구현은 `scripts/prepare_cmfgen_affine_repair.py`에서 찾는다.
- 정본: [Si/S 묶음](BUNDLE_CMFGEN_SI_S_BF_CHANNELS_2026-10-19.md)의
  `실제 광선과 입사 경계 수리`, `428716 종료`;
  [판본 구분과 재현](CMFGEN_FORMAL_MOM_LINK_2026-10-30.md).

## P02 formal source 왕복 진동과 양의 연속 보간

- 과거 처리: 포물선 source 기여가 음수일 때만 선형으로 바꾸는 불연속 때문에
  실제64658 packet의2-cycle이 생겼다. `formal_primary_v11`에서 연속적인
  quadratic Bezier control point와 양의 적분으로 고쳤다. J/intensity floor가 아니다.
- 입증 범위: 원래 jump가 약2.88e-6에서1e-14 미만으로 줄었고, 기존 실패의
  파생 source 재생은3회에 종료했다. 별도428782는197291주파수를 연결했지만
  인구/T 갱신 전 진단이며 released-T 앵커나 실제90격자 보존 PASS는 아니다.
- 재사용: 이미 있는 `scripts/prepare_cmfgen_formal_primary.py`와 private
  판본을 먼저 확인한다. native `USE_LAM_ES` 설정 전환과 같은 구현으로 취급하지
  않는다. 물리 플래그를 바꾸는 단순 우회와 기존 private 수리를 구분한다.
- 물질 연결의 범위: 완료 `toy06_19p48d_field_feedback_429062`의 물질71은
  물질70에서 만든 장을 고정한 `USE_FIXED_J=T`, `DO_LAM_IT=T`, `FIX_T=T`
  경로다. 실제 stage/runtime도 `full_coupled_Jacobian=False`,
  `released_T=False`, `joint_convergence=False`로 기록한다.
  native `cmfgen_sub.f`의 두 `DO_VAR_CONT` 호출은 모두
  `.NOT. LAMBDA_ITERATION` 조건 안에 있다. 따라서 이 분리 물질 단계에
  옛 MOM 복사장 변분이 섞였다고 단정하지 않는다. 전체 결합 Jacobian의
  검증은 여전히 별개이며, 이 확인을 온도 해제나 전체 결합 실행 승인으로 쓰지 않는다.
- 정본: [Si/S 묶음](BUNDLE_CMFGEN_SI_S_BF_CHANNELS_2026-10-19.md)의
  `연속 SC 수정본 직접 증거`, `428782 전주파수 완주`.

## P03 MOM 변화량 수렴과 실제 방정식 불일치

- 과거 확인: 78회 이전 주파수의 단일 호출은16회 뒤 변화량5.3231e-9로
  종료해도 J 대체 후 원식 잔차0.9273846이 남았다. formal 입력의 필요조건
  통과와 MOM 반환의 모멘트 실현 불가능성을 같은 주파수에서 구분했다.
- 현재 판정: 원 MOM 계열의 물리 수리는 미완료다. 10월30일 판독에서 노드/경계
  구분, formal closure 비율, PREV/SAVE의 복사는 확인했다. 단순한 주파수
  주소 혼동으로 설명되지 않으며 작은 내부 변화량을 물리 수렴으로 쓰지 않는다.
- 분석 경로 보완: `entry-only` 포획은 native MOM 호출 전에 의도적으로 멈춰 원 실행
  `MOM_TRACE.bin`을 만들지 않지만, 기존 분석기는 그 trace를 필수로 읽었다. 새
  `analyze_cmfgen_mom_capture.py --entry-only`는 trace-flux 재생 한 건과 `MOM_FINAL`
  을 받아 phase/주파수 및 최종 J/마지막 trace J를 대조하고, 원식 잔차와 formal
  4모멘트 지지행렬·반환 경계 필요조건을 기록한다. 재생 실행에는 반드시
  `LUMINA_MOM_CAPTURE_MODE=REPLAY`가 필요하다. 일반 runner 검증기는 target 지정 build를
  거부하므로, 이 분석 경로는 trace-flux build의 모드·부모 identity·생성기와 입력/출력
  해시를 별도로 확인한다. 직접시험10개 통과. 이는 분석 경로 수정이며 물리 수렴이나
  CMFGEN 동질성 판정은 아니다.
- 2026-11-01 첫 유의미 음수 주파수의 포획/replay 재현: outer78, 318.962458 Å의
  formal J/H/K/N은 저장된 angular-support PSD 검사에서 최소 고유값 `+0.02317`로
  허용영역 안에 있었다. 그런데 같은 주파수 MOM 재귀의 모든 14회에서 rows1·2의 raw J가
  음수였고, 마지막 raw 최소는 `−1.85037e−3`이다. 원 삼중대각 잔차는 `1.17e−16`인데
  음수행을 native 방식으로 대체한 뒤 잔차는 `0.82942`가 됐다. J 변화 기준은
  `5.52e−9`로 통과했지만 반환 J/H/K는 보수적 실현성 필요조건을 edges2·4·6에서
  위반했다(최대비 `2.03754`). 즉 작은 `MAX_ER`와 양수 대체 뒤의 J만으로는 MOM 해의
  물리 타당성을 보장하지 않는다.
- 같은 entry capture에서 직전 `9.39941245274896×10^15 Hz` SAVE의 저장 방정식 잔차는
  `3.92e−16`, fresh 비선형 잔차는 `3.93e−10`이었지만 H는 edges1·2에서 `MAX_VAL`
  대체를 받았다. 마지막 target 선형계 계수를 고정한 국소 반사실에서 이 H 대체만
  되돌리면 row2 우변은 `−2.08831e−3`에서 `−9.37322e−4`로 완화되지만, 재풀이한 rows1·2
  J는 여전히 음수(`−2.8895e−4`, `−2.8984e−4`)다. 따라서 H 대체는 이 고정 계수계에서
  음수화를 악화시키지만 단독 원인은 아니다. 이는 고정 행렬 반사실이지 새 물리 재생이
  아니다. 해당 포획/replay는 entry SHA `a1d87a12…` 및 trace/final SHA가 고정되고,
  운전자 재생 JSON이 기존 `entry-replay-analysis.json`과 byte-identical이다.
- 추가 항 분해에서 row2 RHS의 직전-H 기여는 `+5.5751e−4`와 `−2.8953e−3`으로
  순 `−2.3378e−3`이며, source·직전 J·frequency-coupling의 순 양수 기여는 약
  `+2.4954e−4`다. entry coefficient와 trace RHS 사이 차는 절대 `2e−8` 이내다.
  따라서 row2 음수 RHS는 직전 H 전달항이 지배하지만, raw-H 국소 반사실에서도 RHS와
  J가 음수라 H 제한 하나만으로 설명되지는 않는다.
- 같은 마지막 trace M의 성분해 `solve(M,b_a)`를 합해 raw J를 전체90점 최대
  절대차 `1.07e-14`로 재현했다. 외곽 node1·2에서 이전 H 기여는 각각
  `−0.00552538/−0.00553125`, 나머지는 `+0.00367755/+0.00368088`이다.
  H 제한을 받지 않은 경계3도 경계2만큼 큰 음수 기여를 주므로, 다음 predecessor
  포획 판독은 제한된 두 경계에만 집중하지 않는다. 행렬의 전역 양수 보장 부재와
  외곽 두 행의 부호 원인은 구분한다. 고정 계수 분석의 수치·한계는 아래 정본의
  `같은 마지막 trace 행렬에서 실제 J의 항별 기여`를 참조한다.
- 같은 capture의 직전 SAVE J/H/K에 `H² ≤ max(J_left,J_right)K`를 적용하면
  edges1·2·4·6이 위반되고 최대비 `3.64775`(edge2)다. 따라서 `MAX_VAL` H 제한 뒤에도
  K와 결합한 직전 주파수 장은 실현 불가능했고, Fortran의 주파수 전환은 그 J/H/K를
  PREV 입력으로 복사한다. 이는 318.962 Å MOM 호출이 이미 비실현 가능한 직전 장을
  받았다는 직접 상태 증거다. 후속 `1252379`와 Lageunha 단일 replay가 아래 생성
  경로를 재현했다. 독립 이력 사이 배열은 비트 동일하지 않아 차이를 정본에 명시했다.
- `1252379`는 `COMPLETED 02:21:04`·예정 STOP96으로 직전 주파수의90점을 포착했다.
  Lageunha CPU1·4GiB·5분 한도에서 해당 MOM만 한 번 재생했다(native 실측0.014초).
  formal 노드/중간점 필요조건은 통과했지만 MOM은26회 뒤 `MAX_ER=2.91e-9`를
  만족하면서도 반환 J/H/K 필요조건을 경계1·2·4·6에서 위반했다(최대비3.64775).
  전26회에 **J 대체는0회**였다. 따라서 J 대체보다 먼저 실현 불가능한 MOM 반환
  상태가 생기는 이 주파수 사례가 직접 확인됐다.
- 중요한 공동 방정식 판독: 제한 전 H를 넣은 내부88점 원식 잔차는 최대 `4.32e-16`이나,
  저장된 제한 후 H를 넣으면 node2에서 `0.229354`, node3에서 `0.078999`다.
  H 방정식 잔차도 최대 `0.131387`이다. `completed_operator()`의 작은 잔차는
  **H를 소거한 J 방정식**의 검사이므로 실제 저장 J/H 쌍의 잔차를 대신하지 못한다.
  H 사후 제한이 공동 방정식을 깨는 경로는 확정됐지만, 제한 전 raw H도 각도 실현성이
  보장되지 않으므로 제한 삭제만으로 수리하지 않는다. 원래 MOM 해의 비실현성 원인과
  경계·closure를 함께 만족하는 해법은 미해결이다. 기존 affine/formal-primary 수리를
  재구현하기 전 정본을 재사용하며 이 진단판본을 valid CMFGEN anchor로 쓰지 않는다.
- 사용자 명시 요청으로 Astra 단회 자문을 받았다(11월1일, agent 종료). H 제한은
  반환 원식 불일치의 직접 원인이면서 비실현 가능한 raw 해의 증상이라는 해석을
  지지했고, 기존 수리·66069 반증을 재사용해 동일 수송장→물질·열 응답 연결을
  다음 묶음으로 권고했다. 자문은 재현/실행 승인/최종 동질성 판정이 아니다.
  원문과 별도 production 정적 추적은 정본의 `Astra 단회 자문과 운전자 통합 판단` 참조.
- 기존 318.962 Å capture의 중간점 변환도 재구성했다. formal 4모멘트의
  Steffen 보간 뒤 angular-support 최소 고유값은 `+0.0256830`, 최대 `H²/(J*K)`는
  `0.899068`이고, 재구성 K/J 중간점 closure와 SAVE의 최대 상대차는 `3.30e-15`다.
  따라서 이 target의 raw formal K 보간만으로 MOM 반환 위반을 설명할 근거는 없다.
  고정 closure는 고정된 **비율**이며 오래된 절대 K 복사와 구별한다. 새 native 호출·
  수송은 없고 다른 closure 제한·경계식·직전 주파수는 미해결이다. 재구성식과 범위는
  아래 정본의 `기존 318.962 Å capture의 중간점 보간 확인`을 참조한다.
- 정본: [78회 실패 재현](CMFGEN_MOM78_FIRST_FAILURE_2026-10-30.md),
  [동일 주파수 대조](CMFGEN_FORMAL_MOM_LINK_2026-10-30.md).

## P04 원자모형 접두부와 PHOT 도착 초준위

- 과거 처리: 전체 osc/F_TO_S 길이가 아니라 실행 `MODEL_SPEC`의 활성 NF/SL을
  사용하고 run-local S III·Co II PHOT 선택을 실제 변환 경로에도 연결했다.
  v4 route는 첫 일치 목적지와 F_TO_S의 target SL, 합산 g·확률을 보존한다.
- 입증 범위: 당시 현대 모델의 S III256 FL/79 SL과 전체 파일380/127을 구별했다.
  구형 덱·현대 덱·공개 toy06 원자모형은 서로 같지 않다.
- 재사용: 새 입력의 `MODEL_SPEC`, 실제 링크·PHOT override, route schema를 먼저
  확인한다. 예전 덱 생성 성공이 현재 producer의 활성 자료 일치를 보장하지 않는다.
- 정본: [Si/S 묶음](BUNDLE_CMFGEN_SI_S_BF_CHANNELS_2026-10-19.md)의
  `운전자 구현·직접시험 기록`, `F_TO_S photo-route v4 일치`.

## P05 원본 PHOT가 같아도 유효 단면적이 달랐던 문제

- 과거 처리: CMFGEN `SM_PHOT_V3/CUT_POINTS_V3` 평활화가 Lumina 덱 생성에서
  누락돼 있었다. `scripts/cmfgen_phot_smoothing.py`와 변환기·provenance에
  같은 실행 MODEL의 평활화 처리를 연결했다.
- 입증 범위: 예시 Si II 광율 비가0.3942563에서1.0000683으로 개선됐다.
  뒤의 전체 동결상태율 비교까지 확인했지만 자기일관 분포·스펙트럼 증명은 아니다.
- 재사용: PHOT 파일 hash뿐 아니라 실제 평활화 폭·MODEL·생산 덱 provenance를
  대조한다. 라이브러리 미설정으로 skip된 시험을 실자료 통과로 표시하지 않는다.
- 정본: [Si/S 묶음](BUNDLE_CMFGEN_SI_S_BF_CHANNELS_2026-10-19.md)의
  `PHOT 평활화 누락 원인 재현과 생산 덱 수정`;
  [10월30일 인계](HANDOVER_2026-10-30.md)의 `실제 Si/S BF 채널 시험 독립 재현`.

## P06 Milne 역율과 상위 SL 열적 분배함수

- 과거 처리: PHOT 분기용 정적 합산 g를 역 Milne/3체 재결합의 Saha 분모에도
  쓰던 것을, 같은 T의 `GION(T)=g_anchor/f_anchor`로 수정했다. terminal 단일
  가상 준위와 활성 target SL의 규약은 구분했다.
- 입증 범위: 평활화만 고친 뒤 남았던 S III RR median0.0314938 dex 오프셋을
  추적했고, 수정 뒤27180 SL-depth 셀의 당시 양방향율 기준을 통과했다.
- 재사용: 실제 도착 SL·에너지·T 및 현재 C/Python 소비 경로가 같아야 한다.
  정적 g만 맞춘 fixture나 구형 route 계약을 현재 열적 정규화 증거로 쓰지 않는다.
- 정본: [Si/S 묶음](BUNDLE_CMFGEN_SI_S_BF_CHANNELS_2026-10-19.md)의
  `상위 superlevel 열적 분배함수 수정과 양방향 율 판독`.

## P07 정방향과 역방향이 다른 복사장을 읽던 문제

- 과거 처리: 명시 PHOT 역재결합에만 걸린 ground-core `REC_SPINGATE`를 제거하고,
  자극 재결합도 정율과 같은 세대검증 `RadiationFieldView`를 읽도록 연결했다.
  legacy `nlte->J_nu`가 자동으로 같은 장이라는 가정을 제거했다.
- 입증 범위: explicit channel pair/EW 경로와 그 시험에 해당한다. 구형
  ground-only 경로까지 같은 변경이 적용됐다고 간주하지 않는다.
- 재사용: 현재 정/역율 producer와 J/Jbar/Jblue 소비자, 유효성·세대·단위가
  실제로 같은지 관련 부분만 확인한다. 과거와 다른 알고리즘이면 적용 범위를 기록한다.
- 11월1일 현 전원소 응답 경로를 정적으로 추적했다. `ModelResponse`는 새 J의
  6성분 point BF율과 같은 Jbar의 BB율을 `MaterialBridge`에 전달하고,
  `nlte_model_element_bf_assemble`은 `point_bf_rates`를 소비한다. 열 장부는 같은
  trial/율/Jbar와 새 J의 FF를 사용한다. 옛 BF-bin→canonical-bin 번호 오용 경로와
  구별되며, 현재 이 경로에서 옛 J 혼용이 확인된 것은 아니다. Lageunha CPU1·
  4GiB·300초 내 관련 기존13시험 통과(21.23초). 실제90점 해·모든 과정·외부
  공동수렴 증거는 아니며, 자세한 범위는 10월31일 인계의 `공통장 소비 경로 추적` 참조.
- 정본: [Si/S 묶음](BUNDLE_CMFGEN_SI_S_BF_CHANNELS_2026-10-19.md)의
  `운전자 구현·직접시험 기록` 중 `REC_SPINGATE`, `bf_rate_view_jnu`.

## P08 열 정지기준과 후보 척도 및 불필요한 replay

- 과거 처리: 전하1e-10과 물리 RE1e-3을 분리했다(`4ac9f889`). 수렴 판정은
  초기 척도가 아닌 실제 후보의 ne·가열/냉각 척도를 쓰게 고쳤다(`5565479d`).
  기준 상태가 이미 수렴하면 사용하지 않은 섭동 replay를 요구하지 않게 했다(`1fa0688c`).
- 입증 범위: 현행 소스와 관련 회귀에 대한 수리다. 소스044df9ea에 봉인된
  1243823에는 소급되지 않는다. 실행 중인 봉인본을 바꾸거나 중간점을 최종 선택으로
  재명명하지 말고 종료 후 실제 선택 상태를 판독한다.
- 재사용: `scripts/lumina_coupled_material.py`, `lumina_fixed_field_thermal.py`,
  `prepare_lumina_accepted_material.py`의 실제 실행 판본을 구별한다.
- 후속 전달 수리(10월31일): 1243823은 정상 종료하고 generation22를 선택했지만,
  전달기가 `charge_projection`만 읽고 통상 결합 solver의
  `charge_projected_trials`를 읽지 않아 실제 수용 이력을 거부했다.
  두 모드를 각각 인식하도록 수정했다. 원 계산·결과는 변경하지 않았고
  전하/SE/열 문턱과 마지막 거부 후보 차단도 유지했다. 실제 solver 출력의
  소비 시험을 포함해 수정 전3실패를 재현했고, 관련148시험을 통과했다.
- 저장 선형계 재판독(10월31일): 세 방향의 상대 역오차는약1e-18~1e-17이다.
  원 정규화 행렬 조건수는약1e9지만, 전하 제거 후 열행을 크기 정규화한
  Schur 계의 조건수는7~8이었다. 이 자료로 단순 특이행렬/선형 풀이 실패를
  온도 이동 원인으로 삼거나 새 regularization을 넣지 않는다. secant 두 개를
  새 FD Jacobian으로 표시하지 않으며, 비선형/자기일관 수렴 증거도 아니다.
- 무callback 전달 호환(10월31일): 1248335는 온도응답 callback이 0회여서
  `thermal_evaluation_order` 키가 생략된 정상 수렴 기점이었다. 전달기가 이를
  빈 순서로 처리하도록 하고 비어 있지 않은 history와 길이 불일치는 계속
  거부한다. 56개 관련시험과 실제 C full 전달에서 재현했다. 수리 전 실패는
  출력 생성 전이며 원 계산을 재실행하지 않았다.
- 현재 입력 재판독(11월1일): F444706에 전달된 generation15의 생산→준비→
  수용 manifest→실제 응답/장부 SHA를 대조했다. 전하 `4.85e-15`, 후보 RE
  `2.8996283e-6`으로 **온도 보정0회에서 고정장 수렴**으로 기록됐다.
  이는 `4ac9f889`에서 의도한 전하/RE 분리 문턱의 동작이고 수리 소실이 아니다.
  `numerical_tolerance=1e-10` 기본값이 있어도 명시한 `thermal_tolerance=1e-3`이
  실제 열 종료를 결정한다. 이 표기는 더 엄격한 열 근 또는 공동수렴의 증거가 아니다.
  관련 기존2시험을 Lageunha CPU1에서 재현했다. 문턱·봉인 계산은 변경하지 않았다.
  정확한 계보·수치·재현식은 10월31일 인계의 `현재 입력의 열 종료와 물리 일치 구분` 참조.
- 정본: [10월30일 인계](HANDOVER_2026-10-30.md)의 `고정장 수렴 판정의 후보 척도 수정`,
  `실행 봉인본과 현행 열수용 기준 대조`.
- Astra 단회 코드 자문(11월1일)은 이 구조가 T를 허용 영역 안의 입력값에
  머물게 할 수 있음을 재확인했다. 이것만으로 잘못된 온도 근/Si/S 원인을
  확정하지 않았으며, 수용 문턱과 근 탐색 목적의 구분을 권고했다. 실행/시험
  재현은 하지 않은 자문이다. [자문 정본](ASTRA_CODE_REVIEW_2026-11-01.json) 참조.
- 새 J7d257602의 실제 응답1253439는11월1일12:06 완료했다. 선택 generation14,
  RE2.90462e-6·전하8.92497e-11이며 열 callback0회,90점 T가 입력C와 비트 동일했다.
  따라서 같은 정지 기전이 다시 작동한 것은 실자료로 확인됐으나 온도가 틀린
  근인지/근으로부터의 거리는 미확정이다. 새 기준이나 수리를 억지로 채택하지
  않고 같은 J의 온도 방향을 확인하는 후속 조사로 연결한다.
  [완료 판독](LUMINA_RETURN2_PHYSICAL_COMPARISON_2026-11-01.json) 참조.
- 방향 조사 준비: 수용 문턱은 그대로 두고 명시한 방향 모드에서만 같은 J의
  centered T/ne 네 응답으로 선형 방향을 구한다.0회 온도 수정의 완료 기점은
  같은 물리 producer·정책·실제 응답/장부 SHA를 확인해 재사용하며 기본
  수용/계수 경로의 TNE 거부는 유지한다. 관련283시험과1253439의 읽기 전용
  기점 복원을 통과했으나 추가 실물 callback/Slurm 제출은0회다. 실행 승인과
  실제 결과를 구현 통과로 대신하지 않는다. 위 판독의 준비 절 참조.
- 11월1일1254013 완료: 같은 J의4응답을 실제 계산해 최대|δlnT|1.7116,
  원점34 +0.2008/42 +0.1881을 얻었다. 기존 RE 통과만으로 온도 근 근접을
  주장할 수 없다. 저장 응답에서 Jacobian·잔차·방향을 비트 동일 재현했고
  Schur 조건수7.29/역오차2.08e-18을 확인했다. 단순 선형 풀이 실패가 아니다.
  이는 선형 방향이며 실제 T/물질은 수용하지 않았다. 기존0회 반환 수리의
  소실로 부르지 않고 물리 허용 조건과 실제 온도 근 탐색 목적을 분리한다.
  [실물 방향 결과](LUMINA_THERMAL_DIRECTION_RESULT_2026-11-01.json) 참조.
- 후속 명시적 유한 수정 경로를 추가했다(11월1일22:35). 기본 조기 반환은
  유지하고 `--refine-temperature-direction`에서만 저장 네 응답을 재사용한다.
  제한 T 제안의 실제 전하/SE/후보 자체 RE와 기점 척도 열감소를 함께 요구하며,
  RE 허용 영역에서도 실제 열이 악화되면 거부한다. 관련334시험과 원90점
  선형계/제안 재현은 통과했다. 새 실물 계산/물질 수용은0회이고 후속 단회는
  승인 대기다. 위 정본의 `temperature_refinement_preparation` 참조.
- 11월2일 승인1257134는02:20:45/COMPLETED0:0로 완료했다. 같은 J의
  실제 새 T1개·전하 보정 포함4응답·수용 수정1회이며 후보 자체 RE1.36162e-6,
  전하1.98e-15·SE9.74e-17을 재현했다. 기존 문턱 안에서도 T가 실제 수정되고
  열잔차가 감소해 이 상태의 온도 수정 연결은 확인됐다. 그러나 CMFGEN70 대비
  Si III/T는 악화, S III/Si IV는 개선해 최종 목표는 미달이다. 새 수송은0회다.
  [완료 판독과90점/C 전달 증거](LUMINA_THERMAL_REFINEMENT_RESULT_2026-11-02.json) 참조.
- 11월2일 사용자가 새 수용 generation14 물질에서 시작하는 바깥 왕복1회를
  승인했다. 최신 물질의 계수·경계 재생산 작업1258895는Grammar CPU16/32GiB,
  할당1시간/GPU0/재시도0에서29:29로 완료했다. 이어 F job447806을 Syntax
  H100(syn09)에 GPU1·CPU1·16GiB·최대10시간·재시도0으로 한 번 제출했고,
  계산노드의 Grammar SSH/host-key preflight를 통과했다. 마지막 확인은 첫 F
  sweep 진행 중이며 완성 sweep/반환장 응답은 아직 없다. 이후
  새 F1회(GPU1·10시간), 반환장 응답(16 CPU·3시간), 새-J 방향 네 응답(16 CPU·6시간),
  유한 T/ne 수정1회(16 CPU·3시간)가 같은 목적의 승인 묶음이다. 이 한 번은
  수렴이나 CMFGEN 동질성 선언이 아니다. 단계별 완료 후 이 인덱스와
  [최신 인수인계](HANDOVER_2026-11-02.md)를 갱신한다.
- 11월3일 방향 전용 job `1260551`은 Grammar에서65초 후 exit1로 끝났고 실제 callback,
  BF 적분, 물질응답은 모두0회다. 보존 결과의 `evaluations=[]`,
  `full_BF_integrations=0`, `RE_solved=false`, `new_transport=0`으로 확인했다.
  seed `1260171`은 고정온도(`temperature_evaluations_max=0`) 결과이며
  `thermal_implementation_sha256`가 없다. 현 결과 생성기는 온도평가 한도가0이면
  producer hash를 기록하지 않지만 방향 seed loader는 이 해시를 요구한다. 따라서
  확인된 직접 원인은 물리 구현 변경이 아니라 **seed provenance 누락**이다. 완료 run의
  보존 source archive에 있는 열 producer 9개 해시가 source commit
  `cf147be68eef17eb1a35cf74f924f09cc6c5b717`의 Git tree와 전부 일치하고, 실패 job도
  같은 commit을 사용했다. 원본 결과 SHA `7d2d5248ee5ab2706728310a82c18eb68104792eeabf5b77c42d64bdc4c467f5`를 보존한 채
  별도 `result.thermal-provenance-attested.json`(SHA `4d6c52d0b79eed824602c6dc643cf7c74f660449da0717ff8f1321d4b9df1e72`)을 만들었다.
  현재 source의 loader로 계수6종·열 기준·producer9종을 read-only 복원했고 새 물리 callback/BF/수송은0회다. 해시 검사를 우회하지 않았다.
  결과 생성기는 이제 고정온도 평가0회에도 producer hash를 기록하며, loader는 누락과 불일치를 구별한다. 관련125시험 통과. 실패한 단회는
  재시도0이므로 새 방향 계산은 제출하지 않았고, 후속 물리 단회는 새 승인이 필요하다.
  승인된 BF 응답 `1260171`이 이미 완료되어 중복 제출하지 않은 사실과 검증 범위는
  [11월3일 인수인계](HANDOVER_2026-11-03.md)의 최신 갱신을 참조한다.

## P09 작은 열잔차와 BB 왕복 상쇄 및 에너지 정책

- 과거 확인: 작은 gross 정규화 잔차와 순채널 잔차가 거의1인 상태가 함께 있었다.
  physical/native 계열은 BB 선 가중·FF·내부 에너지/단열에서 구분한다.
  외부열의 저장0 붕괴열과 국소 순간붕괴 시나리오는 별도의 정책 선택이다.
- 11월3일 generation14 같은 새-J 응답을 선택 정책 `native_instantaneous_scenario`로
  읽기 전용 재집계했다. 안쪽 node80–89에서 net은 모두 양수, 중앙
  `+2.6870e-3` erg cm⁻³ s⁻¹이고, `|net|/channel_scale`은 중앙0.4213/p95 0.5885/
  최대0.7169다. 현재 packet의 gross Newton 척도 `0.5*(gross+|net|)`로 나눈 열행은
  중앙 `3.569e-7`/최대 `4.683e-7`로 작다. 큰 gross는 중앙 `1.660e4`이며 큰 상반 BB
  흐름이 포함된다. net 성분 중앙은 즉시 붕괴 외부가열 `+4.360e-3`, native BB
  `−1.769e-3`, BF `+5.012e-4`, FF `−1.613e-4`다. 이는 gross 정규화와 net 열평형
  해석을 분리해야 한다는 기존 P09 경고의 현 응답 사례다. channel-scale 비율은
  진단이며 새 합격선을 만들지 않는다. 고정장 방향 진단은 후보를 수용하지 않고,
  동상태 CMFGEN `STEQ_T` parity도 아직 미확인이다. result SHA256
  `7d2d5248ee5ab2706728310a82c18eb68104792eeabf5b77c42d64bdc4c467f5`, balance SHA256
  `ccc1951d323819a2880e423a3a37971d679101f8c89132df4430844fa9d50271`;
  자세한 집계와 response SHA는 [11월3일 인계](HANDOVER_2026-11-03.md)의
  `generation14 기준 열장부 재판독` 참조.
- 11월3일 구현·계약 대조: v2 RE 행의 사전 합의 척도는
  `max(total heating, total cooling, |deposition|)`이다. 현 `packet()`은
  `0.5*(gross+|net|)`을 사용하며 `gross=H+C`, `net=H-C`이면 `max(H,C)`와 같다.
  이번 generation14에서 gross 중앙은 `1.660e4`, 별도 기록된 즉시 붕괴 외부가열은
  `4.360e-3`이다. 이 한 항은 큰 H/C 척도를 정하지 않지만, 이것만으로 모든
  deposition 항의 포함이나 native 식과의 동등성을 증명하지 않는다. 따라서 현재
  gross 기반 척도 자체를 오류로 단정하거나 P09의 net-channel 비율로 교체하지 않는다.
  반면 이 척도에서 작은 잔차만으로 열평형의 물리적 동등성을 선언하지도 않는다.
  결론은 같은 물질·J에서 CMFGEN native `STEQ_T` 열행과 합의된 residual/Jv를
  대조해야 하며, 현재 direction-only 진단은 그 증거가 아니다.
  근거: `CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md`의 RE scaling 규약과
  `scripts/lumina_coupled_material.py`의 `packet()`/`residual()`.
- 비교 신원 주의: 11월2일에 준비된 native `STEQ_T` 비교는 다른 J·response·balance
  해시에 고정돼 있다. 11월3일 generation14의 새 J와 일치하지 않으므로 현재 결과의
  동상태 CMFGEN parity 증거로 재사용하지 않는다. 현재 입력에 맞춘 native 열행 평가는
  별도 실행 범위이며, direction-only 진단의 증거로 간주하지 않는다.
- 최근 재판독: generation22의 저장 장부를 항별 합산해 두 정책의 net을 정확히
  재현했다. 정책 차이는 주로 BB 가중이며, 인공 가열/physical 가열냉각 척도의
  최대 비는 약4.4e-203이다. 이는 해당 저장 상태에 대한 확인이다.
  원 자료는 `lumina_second_field_thermal_replacement_20261030-fz3u4uV0`의
  `run/evaluation-12/balance.npz`, SHA256
  `d78df23c89193c81e2fb04e9175e95fe313d97e3bac7ef0beae95f4ae8ccea92`다.
- 재사용: absolute net·사전 RE 척도·정책·T/ne/이온분포·장 갱신을 함께 해석한다.
  순채널 척도를 임의의 새 합격선으로 만들거나, 작은 gross 잔차를 최종 물리
  일치로 확대하지 않는다. SL 에너지 proxy 분석은 전자 열해 증명과 구별한다.
- 현재 generation15(11월1일): 원점42의 native 순열은 `+3.8464472e-5 erg cm^-3 s^-1`,
  병진 운동에너지 밀도는 `2.4514373e-4 erg cm^-3`다. 고정 인구에서의
  `Ekin/|Q|=6.37325초`는 잔여 에너지 규모의 비교일 뿐 실제 NLTE 열완화시간이나
  시간발전 예측이 아니다. RE 통과와 동시에 T의 CMFGEN 대비 최대차32.97%가
  남아 있다. 옛 generation22의 `fixed_field_numerical_converged=false`와 달리
  이 generation15는 위 분리 기준에서 true다. 둘을 혼동하지 않는다. 추가 물리
  실행이나 새 수용 기준 없이 저장 장부로 확인했다.
- 좌표계 주의(10월31일): native J/H/K는 공이동계 각도로 적분한다.
  F2 외곽 원점의 공이동계 H 적분 광도 `-1.879e44 erg/s`는 같은 저장
  모멘트의 국소 실험실계 변환에서 `+3.979e43 erg/s`다. signed H의 음수를
  음수 J나 관측자 탈출 광도의 음수로 오인하지 않는다. 이 변환 자체는
  에너지 보존·확장 외부 경계 스펙트럼·CMFGEN 광도 동등성의 증거가 아니다.
  재현 범위와 수식은 아래 인계의 `끝점 광도와 좌표계 구분`에 있다.
- 정본: [10월30일 인계](HANDOVER_2026-10-30.md)의 `열잔차 정규화의 해석 한계`;
  [SL 에너지 계약](LUMINA_SL_ENERGY_CONTRACT_2026-10-27.json).
- 같은 자문 후 운전자 정적 대조: 독립 C photon 검사의 BB는 비가중이고
  선택 native 열행은 line_scaling 가중이다. 전자 열행 전체의 동등성 증명으로
  확대하지 않는다. 기존 SL 계약도 이미 실제 native EHB 미입증을 명시했으므로
  과거 검증 소실/새 회귀로 부르지 않는다. CMFGEN의 실제 선택 열행과 같은
  후보를 연결하는 문제로 유지한다. 자문만으로 열 정책/계수를 변경하지 않았다.
- 11월1일 후속 정적 대조: J71 기준 빌드 manifest와 SHA가 일치하는 봉인
  `cmfgen_sub.f`에서 SL 에너지차/주파수 비·허용폭 fallback·`POP_ATOM` 고밀도
  해제가 현 `line_scaling`과 같은 식임을 확인했다. 현재 native의 g-평균 정의도
  같다. 가중치 자체를 Lumina만의 오류로 보고 삭제하는 방향은 비채택이다.
  전체 열행/적분 및 보조 객체의 실제 링크 동일성까지 입증한 것은 아니다.
  정확한 source SHA와 범위는 위 자문 JSON의 운전자 후속 기록에 보존했다.
- 단열 재조사 시 먼저 [8월8일 대응표](CMFGEN_ADIABATIC_V3_MAPPING_2026-08-08.md)의
  3·6·7절을 재사용한다. `STEQ_T`의 INT_EN 주경로와 진단 AD_CR_DT의 COL_EN,
  EHB interior의 host INT_EN 혼재는 당시 이미 구분했다. 이 진단 출력 차이를
  새 Lumina 회귀로 처리하거나 GENCOOL 단열 출력만으로 주 열행을 인증하지 않는다.
  현재 native 호환 내부에너지 영점/SL 정의는 위10월27일 기록이 우선한다.
- 11월2일 실제 RE 비교 준비: 기존 fixed-J native의 열 출력 제외는 유지한다.
  봉인 J71 caller에서 `USE_FIXED_J`가 continuum COMP_OPAC을 건너뛰므로,
  기존 STEQ_T를 바로 같은 상태의 열행으로 쓰지 않는다. private 전용 경로는
  COMP_OPAC을 평가하고 최종 STEQ_T/입력 상태를 물질 갱신 직전에 저장한 뒤
  종료한다. 지원 BLANK 설정의 선 기여는 continuum loop 안에서 처리하며
  뒤의 별도 LINE LOOP를 건너뛴다. J71 실제 archive/객체로 빌드와 관련36시험을
  확인했지만 실제 열평가·상태 주입·열행 일치는 아직 없다. 기존의 출력 한계를
  구체화한 것이며 새 production 회귀나 에너지 정책 수용으로 부르지 않는다.
  [11월2일 인계](HANDOVER_2026-11-02.md)의 `후속 준비`와 기존 완료 JSON의
  `native_integral_re_preparation`을 재사용한다.
- 이어서 수용90점·1800변수의 native 입력을 만들고 실제 Fortran 로더의
  162000개 double 전달을 바이트 동일 재현했다(관련58시험). 입력은 원래
  POPTOION/SUP_TO_FULL 전에 주입하며 누락/오류 시 예전 상태로 실행하지 않는다.
  아직 native full 인구 복원·J 소비·실제 열행은 미평가다. J71 archive 그대로인
  붕괴 구현과 `native_instantaneous_scenario`의 차이를 먼저 명시하며 기존 순간
  붕괴 수리를 재사용한다. 같은 인계의 `수용 상태의 native 입력 연결` 및
  완료 JSON의 `native_state_preparation` 참조. 새 물리 실행은0회다.
- 현재 J의1115020주파수/90점을 native EDD로 전수 보존했다. 원 RD_CONT_J
  선형 보간 시험을 포함해69시험을 확인했고,10월27일의 검증된 순간 붕괴 수리를
  private 명시 object로 재사용하는 옵션을 추가했다. 실제 물리 평가는0회다.
  NCF_MAX800000은 외부 장 record 수가 아니라 native 자체 격자의 용량이다.
  native 적분 격자/가중치와 Lumina fine 격자가 같다는 뜻은 아니며, 실제 열행
  차이의 해석에서 이를 분리해야 한다. 위 인계의18:30 준비 및 완료 JSON의
  `native_current_field_preparation` 참조. 설치본/옛 앵커를 새 수리판으로 바꾸지 않았다.
- 11월2일 full-level 동일 상태 비교 준비: `EVAL_LTE_V5` 직후 native FL 인구·LTE
  분율·점유확률·주소 및 실제 주입 상태를 저장하는 private observer와 비교기를
  추가했다. MODEL_SPEC 27개 이온/24542 FL이 C `levels.csv`와 일대일 대응하고,
  C 점유 배열은 수용 응답과 정확히 같다. 동일 입력 바이트 및 같은 평가번호의
  열행을 확인한 뒤 `(Z, stage, FL)`로 정렬해 C full과 비교하도록 했다. 비교 신원도
  보강해 실행파일을 private build manifest와, 실행 출력의 EDD 두 파일을 field
  manifest와 대조하며, 유일한 J 입력을 generation15의 고정 SHA256에 묶는다. 관련
  시험은76개 통과·1개 건너뜀이고 private 링크 빌드는 통과했지만 실제 native
  평가/물리 비교/Slurm은0회다. 합성 end-to-end 및 신원 불일치 거부 시험도 통과했다.
  새 물리 평가 권한이나 합격선을 뜻하지 않는다.
  capture는 `EVAL_LTE_V5` 뒤, 원자율/열 계산 전이다. 따라서 이 비교는 수용된
  Lumina superlevel 인구의 full-level 재구성 진단이며, CMFGEN이 독립적으로 푼
  NLTE 이온분포가 아니다. full-level 대응이 맞아도 자기일관 해나 최종 동질성은
  입증되지 않는다.
  추가 정적 추적에서 native `COMP_J_BLANK`는 `SECTION='CONTINUUM'`일 때
  `CONT_VEL`을 참으로 설정하고 `USE_FIXED_J`도 참이면 `RD_CONT_J`로 EDDFACTOR J를
  읽는다. 현재 VADAT의 `USE_FIXED_J=T`, `GLOBAL_LINE=BLANK`는 이 분기에 맞는다.
  그런데 VADAT가 기존 상태-input 해시에 포함되지 않았으므로, 비교기가 실제 run-root
  VADAT SHA를 고정값 `80e08003…6d144ec`와 대조하도록 보강했다. 이는 예정 경로 확인이며
  실제 실행의 J 소비 증거는 아니다. 아직 물리 평가·Slurm 0회다.
  [11월2일 인계](HANDOVER_2026-11-02.md)의 20:54 `same-state full-level 비교 준비`와 완료 JSON의
  `native_full_comparison_preparation` 참조. 최종 목적은 계속 **NOT_MET**다.
- 같은 실행에서 native `STEQ_T` 수치와 Lumina `native_instantaneous_scenario_net`을
  비교하도록 후처리를 보강했다. 선택 balance의 해시를 고정하고 90점별 절대·
  RMS·Lumina gross 기준 차이를 기록하며, 사전 합격선은 없다. 최초 구현은
  `STEQ_T`의 양수=냉각 부호와 Lumina balance의 양수=가열 부호를 정규화하지
  않고 빼는 결함이 있었다. 이를 정적 대조로 찾아 native 원값은 보존하고
  비교용으로만 native 부호를 뒤집도록 고쳤다. 합성시험에서 열/냉각 부호가
  물리적으로 대응하면 오차0, 불일치면 오차가 남음을 확인했다. 실제 native
  실행·열행 parity는 여전히 미실행/미평가이며, native/Lumina 주파수 quadrature
  차이도 해석에 남는다.
- **2026-11-03 정정 — 위 sign 판정·v4의 순간열 비교는 superseded.** 실제 고정-J
  열행 job `1261748`은 parent `COMPLETED 0:0`이고 실행단계는 의도된 `STOP96`
  (`96:0`)으로 물질 갱신 전에 끝났다. 같은 J·물질의 native 누적
  `STEQ_T`를 unit-corrected v6로 다시 비교했다. 봉인 native source
  `cmfgen_sub.f` SHA256 `84e3b5ae0c6f44f531b4c1cd6aa83cde9cd51bc051a0f98cddd8fcd7fd17d54c`
  에서 복사 연속항은 `CHI_NOSCAT*RJ-ETA_NOSCAT`를 `STEQ_T`에 더하고 선 냉각항은
  빼며, 직접 `dE_RAD_DECAY`는 별도 `STEQ_T_EHB`에 더한다. 따라서 독립 `LINEHEAT`
  성분의 양수=냉각 관례를 누적 `STEQ_T`에 옮긴 이전 sign flip은 잘못이다.
  CMFGEN 내부 단위는 `4*pi*1e-10`으로 cgs 환산한다.
- Lumina 장부에서 `native_instantaneous_scenario_net`은
  `native_stored_net + instantaneous_scenario`로 재현되며 최대 절대 재조립오차는
  `4.34e-19`, 이 실행의 `stored_decay`는0이다. 직접 붕괴항이 없는 native
  `STEQ_T`에는 순간가열 포함 정책보다 저장 정책이 더 가까운 비교 후보지만,
  동등성은 아니다. 그 후보의 90점 절대차 최대 `1.5183e-3`, RMS `5.3641e-4`
  erg cm⁻³ s⁻¹, 비영점 부호 불일치29점, 대칭 net 척도 중앙 상대차0.9878/p95
  `1.1623`이다. gross 척도 p95 `2.968e-5`는 큰 상쇄 흐름을 분모로 한 값일 뿐
  net 열 일치 증거가 아니다. 순간가열 포함 v4 비교는 정책 차이로 부호 불일치가
  더 커져(59점) source-policy 판정에 쓰지 않는다.
- 격자도 일치하지 않는다. Lumina J 입력은1,115,020점, CMFGEN `NCF`는197,291점이며
  native는 선형 주파수 보간을 사용한다. v6 보고서는 두 격자 수와 보간법을
  함께 기록하며 field manifest가 `native_quadrature_equal=false`로 명시하므로
  v6 수치는 같은 물질/J 계보의 **서로 다른 적분 격자 진단**이지
  parity 합격/실패나 회귀 원인 확정이 아니다. 다음 판정에는 native 격자에서의
  Lumina 재적분 또는 term별 native/Lumina 대응이 필요하다. v2 원단위·v3 잘못된
  부호·v4 순간열 비교 원본은 포렌식으로 보존하며 덮어쓰지 않는다. 현재 최종목표는
  계속 `NOT_MET`이다. 상세는 [11월3일 인계](HANDOVER_2026-11-03.md)의 최신 진행상태.

## P10 공통장 인구 일치와 자기일관 해 및 CMFGEN 기준해

- 과거 확인: 실제71의 같은 T/ne/J에서 Si/S 전90점 조건부 C1 비교를 수행했다.
  그러나 물질·장 반복의 자체 이동과 감쇠 분기 관찰 목표는 별도로 미달했고,
  단순 반-step/복원 방법은 채택하지 않았다.
- 재사용: 조건부 인구 일치는 독립 장·온도·스펙트럼 일치가 아니다. J71은 물질70의
  장이라는 생산 phase를 확인한다. 물질→그 물질의 계수/경계→새 J→반환장 응답의
  고리를 닫아야 한다. fixed-T 비수렴 CMFGEN70은 인증 oracle이 아니다.
- 반경 표본·공간 평균 질문의 적용 범위(2026-11-03 재대조): 기준
  `native_material_nodes.bin`은 CMFGEN `SCRTEMP`의 같은 90개 깊이점 노드값이며,
  manifest는 `NOT transfer-cell averages`로 명시한다. 현재 반환장 물질응답은
  material 반경과 `source.radius`를 점별 정확 비교하고, 고정인구 bound-photo도
  CMFGEN field와 material 반경의 정확 일치를 요구하며 주파수축에만 log-ν 적분
  가중치의 adjoint 보간을 한다. 따라서 이 특정 90점 고정인구 율 차이는 radial
  bin-average/center 오프셋으로 설명되지 않는다. 다만 production 수송 경로는
  별도다. `prepare_lumina_material_cells.py`는 90점에서 89개 구간의 rho·n·rhoX·
  rhoT를 재구성해 부피/질량 평균을 만들며, 그 manifest도 물질 평균만으로 J의
  공간 평균 기능을 추정하지 말라고 명시한다.
- 과거 10월22일 89셀 고정-J 대조는 Lumina 셀중심 F를 CMFGEN native 양 끝점 J의
  산술평균과 비교했다. 셀37/39의 100–400Å 겉보기 비 `0.647/1.438`은 실제
  끝점 평균 자체가 아니라 비선형 source의 계수중간 대리값 `0.631/1.477`과
  가깝다. 따라서 표본/공간 source 차이는 해당 두꺼운 대역의 겉보기 잔차 일부를
  설명하지만, 대리값은 CMFGEN의 실제 셀중심 J가 아니다. 같은 공간계수 세분화
  P0-4/native-2/native-4도 EUV 차이를 제거하지 못했다. 셀39 native49/50 출력점의
  F/J는 대역별로 크게 달랐고, 동일 위치라고 전체 대역 동등성이 성립하지 않았다.
- 별도 1450Å 중앙광선 분석에서는 물질 `∫χds≈0.044`여도 주파수 이류를 포함한
  CMFGEN 유효 감쇠가 `τ≈41.67`, source의 `98.7–99.96%`가 직전 주파수 강도였다.
  그러므로 `η/χ`는 중심 J 대리값이 아니며, 이는 단일 광선 결과라 전각도/전대역
  결론으로 확대하지 않는다. CMFGEN 이산 upwind와 Lumina 연속 특성선은 수치
  연산자가 다르며 한 광선의 주파수 이동량 일치만으로 전체 장 parity가 증명되지
  않는다. 원자료와 수치는 `LUMINA_NATIVE_SPATIAL_FIXED_2026-10-22.md`,
  `LUMINA_UV_SOURCE_LOCATION_2026-10-22.md`,
  `LUMINA_FORMAL_FREQUENCY_COUPLING_2026-10-22.md`에 있다.
- 따라서 현재 90점 bound-photo 차이를 공간 평균 문제로 무효화하지 말고, 반대로
  89셀 transport의 F/Jref 잔차도 실제 같은 위치 J와 곧바로 동일시하지 않는다.
  공간 차이를 닫으려면 실제 CMFGEN 셀중심 formal J 또는 동등한 중심 formal
  평가를 같은 각도·주파수·공간 functional로 확보해야 한다.
  `prepare_lumina_material_seed.py`, `prepare_lumina_outer_input.py`,
  `lumina_new_field_response.py`, `compare_cmfgen_bound_photo_profiles.py`의 blob은
  현재 방향 job source `0824950d2de24af90c1a47456c58f5632c35fcd2`와 byte-level로 일치한다.
- 현 고정장 진단의 과정 계약 재확인(2026-11-03): 봉인 `MODEL`/`MODEL_SPEC` 해시는
  각각 `e8ec1835355869c3b8fbaf761fc8b6eeb516b527aa660aea768b8cbc19e2f6c9`/
  `3e6f53861f16e2dac822d68e1412f91813c7a80f939fdbd0c4b1394e9055f2fb`다. `INC_CHG=F`,
  `TRT_NON_TE=F`, `GAMRAY_TRANS=LOCAL`, `INS_DEP=T`이며 활성 27이온은 Si/S/Ca/Fe/Co/Ni다.
  `INC_PEN=T`지만 H I/He I가 없어 확인된 native Penning 경로는 비활성이다.
  그러므로 이 입력에서 CE·비열전자 처리가 빠진 것을 새로운 Lumina/CMFGEN 불일치로
  세지 않는다. 즉시 감마 침적 열은 별도 활성 입력이다. 이는 이 특정 roster에 한정한다.
  과정 범위와 H I/He I 조건의 정본은 [전모형 물질응답 계획 §3](PLAN_CMFGEN_FULL_MATERIAL_RESPONSE_2026-10-27.md)을 참조한다.
- 10월31일 현재 CMFGEN source `dd1660af`의 정적 추적: `comp_j_blank.f`는 formal
  `FG_J_CMF_V13` 다음 `MOM_JREL_V9`를 호출한다. MOM은 앞서 `GET_MOMS_REL`에서
  얻은 closure 배열을 사용·저장하면서 MOM 갱신 J/H도 별도로 저장한다. 음수 J의
  `FORM_VAL` 대체와 H 제한(`MAX_VAL`) 뒤 종료조건 `MAX_ER`는 수정된 J의 반복
  변화량만 검사하며, 원 삼중대각식 잔차나 J/H/K 공동 각도 실현성은 검사하지 않는다.
  이는 포획 재생의 `5.3231e-9` 반복변화와 `0.9273846` 원식 잔차가 함께 나온 것과
  맞는다. 완료 job의 build manifest/build log 및 부모 identity 해시와 실제 실행파일
  해시를 재대조해 진단 capture/replay binary와 링크 입력을 고정했다. manifest는
  `entry_only=true`, `first_repair=false`; 소스 비교는 계산문 삭제 없이 trace·snapshot·
  선택 정지 계측의 추가만 보인다. 다만 source snapshot에 Git metadata가 없어 파일
  해시를 특정 CMFGEN commit에 연결하지 못했다. 진단 binary/링크 신원은 고정됐으나
  유효한 물리 기준해는 아니며, 최초 원인이나 Lumina 결함을 확정하지 않고 CMFGEN
  source를 임의 변경하지 않는다. Slurm 영수증은 전용 `submit_cmfgen_moment_failure`
  wrapper와 driver commit `38466bf9`를 가리키며, run 기록은 CPU16/OMP16·128GiB·
  GPU0·13,680초·retry0 및 STOP96 진입포획으로 이를 확인한다. 기존 승인은 이 target
  단회에 한정되고 298–319 Å snapshot은 없다.
  후속 계보 대조(2026-11-01): parent build report가 가리키는 CMFGEN
  `dd1660af039e10f8f8c61d892c4e226e1cd6438c`와 포획 native의 직접 입력 6개 중
  5개(`mom_jrel_v9.f`, `mod_ray_mom_store.f`, `comp_j_blank.f`, `GET_J_CHI.INC`,
  `set_kind_module.f`)는 바이트 동일했다. 여섯째 `cmfgen_sub.f`는 기준 SHA256
  `092f8526…`에서 포획 SHA256 `a9b2544b…`로 달랐지만 기준본에 실제 주파수
  `EMHNUKT` 갱신(`prepare_cmfgen_fixedj_exponent.py`), BA 초기화
  (`build_cmfgen_instantaneous_reference.py`), dZ lazy 행 초기화
  (`prepare_cmfgen_lazy_dz.py`)를 생성기 순서대로 적용하면 포획본과 바이트
  동일했다. 따라서 이 여섯 파일에는 미설명된 소스 차이가 없다. snapshot에
  Git metadata가 없고 이 private patched 진단본은 stock/released CMFGEN이나
  수렴 물리 기준해가 아니다. 이 계보 확인은 물리적 동질성을 증명하지 않는다.
  원본 checkout에 대한 기존 `tests/test_native_lazy_dz.py`도 5/5 통과했다
  (source identity, lifecycle/활성 consumer projection, 두 음성대조). 이는
  full-model peak memory 또는 실제 물리 동등성 검증이 아니다.
  추가 source audit: `MOM_JREL_V9`는 100회 반복 시 `excessive iteration count`를
  기록한 직후 `ACCURATE=.TRUE.`로 강제해 반복문을 끝낸다. 호출부
  `comp_j_blank.f`는 상태코드 없이 subroutine을 호출하므로 일반 후속 계산을
  계속한다. 실제 318.962 Å 재생이 이 경로를 밟는지는 미확인이고, 기존 490.873 Å
  판독은 16회 반복이라 이 cap의 사례가 아니다. 반복한도 종료는 수렴 증거로
  인정하지 않는다.
  후속 판독기 변경은 저장된 `INCL_ADVEC_TERMS`·반복 수·`MAX_ER`를 사용해
  비반복, 변화 허용치 충족, 비수렴 cap 종료와 예상 밖 trace 종료를 명시적으로
  구분한다. 관련 두 MOM 분석 시험 48/48 통과; 이는 재생 해석만 보완하며
  새 물리 실행·동질성 증거가 아니다.
  실제 490.873 Å replay에도 적용해 1호출·16회·마지막 `MAX_ER=5.3230787363e-9`를
  재현했고 `CHANGE_TOLERANCE_MET`로 분류했다. 이는 cap 분류 회귀 점검이며,
  FORM_VAL 후 원식 잔차가 큰 기존 실패를 해소하지 않는다.
  같은 포획 sweep의 `MOM_J_ERRORS`는 `INIT`에서 rewind되며 음수 원시 J를 대체하기
  전에 기록한다. 해시 `00a8658790a83bdb87e10aa6461c015dc27a162261d746478926b348afd3acd4`인
  로그에는 38,565행/1,537주파수가 있다. 첫 기록은 node7, 298.404 Å,
  `J=-2.4755e-6`(±2점 이웃 양의 최댓값 대비 `7.71e-4`); 318.962 Å에는 처음으로
  그 비가 1을 넘어 `2.62`가 된다. 따라서 490.873 Å 포획점은 음수 J 기록의 시작이
  아니라 뒤쪽 사례다. 최초 음수의 미소 크기만으로 최초 물리 실패를 정하지 않는다.
  음수 기록 중심은 전부 바깥쪽 node1–10/90에 한정된다. 318.962 Å의 node1/2에서
  원시 J는 약`−1.84e-3`, node3은 `+7.03e-4`; node1의 `ETA`, `CHI`, `ESEC`, `K/J`는
  모두 양수다. 포획 인수는 `OUTER_BND_METH=HONJ`, `J_CHK=FORM_VAL`,
  `H_CHK=MAX_VAL`, `XM_CHK=NONE`을 확인한다. target node1의 `PSIPREV×J_PREV`도
  `+4.2803e5`여서 음수 이전주파수 RHS 하나만으로 설명되지 않는다. outer-row
  operator/closure spectral coupling이 선도 후보지만 319 Å의 전체 계수가 없어
  인과는 미확정이다. 추가 재현에서는 target 490.873 Å trace 해시를 맞춰 읽었고,
  raw 선형계 잔차`1.4188e-16`에서 `FORM_VAL` 교체 뒤`0.927384637`로 악화됨을
  확인했다. 최대 잔차는 node3이며, node2 J 대체가 node3 하부결합항에 만든
  변화`−8765.2134`와 node3 방정식 잔차 증분이 일치한다. 이는 후반 capture 점의
  clipping/수렴판정 실패 메커니즘만 확정하며 298–319 Å 최초 원인은 아니다.
  2026-11-01 318.962458 Å 진입포획·동일 MOM 재생에서는 row2 우변을 항별로
  재구성해 이전 H 공간 결합`−2.33783e-3`이 총 우변`−2.08831e-3`의 음수
  부호를 결정함을 확인했다. 직전 주파수 H 원식 재구성은 두 외곽 경계에서
  raw H`−3.13783e-3/−4.32491e-3`를 native `MAX_VAL`이 저장 H
  `−5.57623e-4/−2.89592e-3`로 바꾼 것과 정확히 맞는다. 저장 closure에 대한
  보수적 각도 실현성 비는`2.181/3.648`로 필요조건을 위반한다. 전체89경계
  census에서 비율>1인 곳은 직전 상태4개(1,2,4,6), 두 단계 전5개
  (1,2,3,6,7)로 외곽에 집중됐다. H를 그 필요조건 한도까지 낮춘 반사실에도
  node1/2 J는 음수로 남으므로, H 클립
  초과만이 유일 원인은 아니다. formal 입력은 각도 실현 가능하고, MOM raw
  해의 잔차는`1.17e-16`인데 `FORM_VAL` 뒤 원식 잔차는`0.82942`다. 이건
  특정 비수렴 CMFGEN 경로의 증거이지 valid CMFGEN anchor나 Lumina의 전반적
  결함 판정이 아니다. 상세 수치·해시·재현 범위는 [첫 MOM 실패](CMFGEN_MOM78_FIRST_FAILURE_2026-10-30.md)의
  `2026-11-01: 318.962 Å 음수 J의 이전 H 결합 추적`을 참조한다.
  같은 capture-build의 호출부/루틴 정적 추적(11월1일)은 formal 뒤 MOM 호출,
  MOM 진입 시 `GET_MOMS_REL` 1회, recurrence 중 closure 재계산 없음, 종료 시
  closure SAVE 순서를 확인했다. 즉 고정 closure 아래에서 J/H를 갱신하는 경로이며,
  이것만으로 구현 결함이나 충분한 outer 수렴을 판정하지 않는다. 직전 주파수
  포획 `1252379`가 formal 입력과 반환 상태를 가른다. 상세 범위는 위 정본의
  `직전 주파수 formal→MOM 연결 확인`을 참조한다.
  정본: [2026-10-31 인계](HANDOVER_2026-10-31.md).
  정본: [formal/MOM 연결](CMFGEN_FORMAL_MOM_LINK_2026-10-30.md),
  [첫 MOM 실패](CMFGEN_MOM78_FIRST_FAILURE_2026-10-30.md).
- 과거 CMFGEN relT2/relT3 진단의 재사용: `MAXCH`는 실제 correction scaling 전
  raw `STEQ`에서 산출되고, 감소 correction `c≥0.99999`는 `10^7%` sentinel로
  치환된다. 그러므로 거대 `MAXCH`만으로 native factor cap 미작동이나 bulk 물리
  발산을 판정하지 않는다. 반대로 독립 population-weighted 지표에는 감쇠하지 않는
  진동/S drift가 있었고 full BA 뒤 광도도 약2.007배 이동해, 이력 전체가 단순한
  계측 아티팩트였던 것도 아니다. relT2의 `MAX_LIN/MAX_LAM=10`은 단계당 최대
  10배 변화까지 허용했다. relT3 설계 메모는 native trust-radius를 더 좁히는 방안을
  제안했지만, 그 설계만으로 실행·수렴이나 released-T 앵커를 입증하지 않는다.
  재사용 전 현재 CMFGEN 소스·입력·실행 계보를 확인한다. 정본:
  [relT3 trust-radius 설계](CODEX_RELT3_TRUSTREGION_DESIGN.md),
  [relT3 사후분석](CODEX_RELT3_ANALYSIS.md),
  [2026-08-01 보고](MORNING_REPORT_2026-08-01.md).
- 후속 실행 확인: job `1244808`의 고정T LAMBDA 5회는 정상 종료했지만,
  iteration59→60에서 Si III·S III 최대 분율 변화가 각각 `2.789e-3`,
  `1.316e-3`였고 correction p95도 컸다. T는 계속 고정됐으므로 이는
  수렴 released-T 기준해가 아니다. `CODEX_RELT_RECIPE.md`의 10월18일
  판정은 source/config/restart 신원과 고정T 안정성·rollback 조건을 갖추기 전
  실행 No-Go다. 완료코드 0 또는 A/B/C 후보를 실행·앵커 승인으로 해석하지 않는다.
  결과 정본: [10월30일 인계](HANDOVER_2026-10-30.md)의
  `CMFGEN 고정T 감쇠 진단 5회 완료 — 수렴 미달`.
- 10월31일 새 J 고정장 응답 generation15의 Si/S 사후 비교를 read-only helper로
  재현했다(result/response/balance SHA는 [10월30일 인계](HANDOVER_2026-10-30.md)의
  `저장 응답 이온 비교의 후처리 재현` 참조). 온도변화0, 최대 ne변화0.974%이며,
  분율-only 결과와 비인증 CMFGEN70 기준의 한계는 그대로다. 이는 새 수송이나
  공동수렴·최종 스펙트럼 판정이 아니다.
- 같은 비교의 점별 CSV에서 Si III·S III 최대 분율차는 반경점33에 함께 나타났다.
  새 J 응답에서 두 이온은 모두 III→IV로 이동했으나 Si III는 기준과의 차가 커지고
  S III는 조금 줄었다. 기준은 비수렴 fixed-T라 종별 원인으로 단정하지 않으며,
  새 물질의 다음 수송 비교가 필요하다(수치·CSV SHA는 인수인계서 참조).
- 같은 점의 저장 BF photo/역흐름은 각각 거의 상쇄됐다. Lumina 내부 BF 부분 장부
  확인일 뿐 CMFGEN율·비-BF 과정·전체 SE의 동등성은 아니다(정확한 값·balance SHA는
  인수인계서 참조).
- 11월1일 같은 점34(1-based)의 실제 단계 분해: 옛 F2에서 열보정 전→후
  Si III 분율은 `0.705294→0.702269`, 그 물질의 **다른 새 장** 응답 뒤는
  `0.664854`였다. 뒤 단계는 온도 동일·ne 최대0.974% 이동이고 동일-J 시험이 아니다.
  열보정 후 인구를 고정해 두 장의 저장 bound PHOT율만 바꿔도 Si III Gamma는
  **18.64965% 증가**, S III는0.39987% 감소했다. Si 증가분의67.29%는 문턱
  약1650 Å의 FL31/32/33 세 경로다(FL 번호는 CSV의0-based 주소).
  실제 새/옛 J 비도1650/1600 Å에서1.319/1.357이다. 따라서 열 종료 조건만을
  고치거나600–750 Å만 비교해 Si 불일치를 설명하지 않는다. 이는 동일 인구의
  bound 부분 율 반사실이며, CMFGEN율·전체 원소 분포·스펙트럼의 동등성이나
  새 J의 정확성 증거는 아니다. 해시·단계·재현식은 10월31일 인계의
  `Si III 차이 증가를 새 장의 들뜬 준위 광이온화까지 추적` 참조.
- 같은1650 Å 표본의 소스 연결 확인(11월1일): F442848은 직전 F2 J를 실제
  입력으로 사용하며 `CompiledInput.bind_view`→dense J→device Reader의
  `eta_true+ES*J_input` 연결이 유지된다. 원70/0장 재주입 근거는 없었다.
  점34의 ES/총소광은0.997742이며, **동일 물질**에서 반환 J로 산란 방출항만
  갱신해도 사용 소스보다31.87585% 커진다. 이것은 한 번의 F가 산란 고정점을
  닫지 않았다는 구체적 증거이지 수송 kernel 오동작 판정이 아니다. 열잔차1e-6
  통과와 구별한다. 관련 기존 CPU14시험 통과; 전체 수송 잔차/수렴률을 이 표본으로
  대신하지 않는다. 10월31일 인계의 `UV 산란 소스의 미닫힌 고리` 참조.
- CMFGEN J 직접 비교(11월1일): 같은 점34·B 인구·같은σ/전역 격자로
  Si III147/S III256개의 **전체 bound 경로**를 합하면 현재/CMFGEN 장의
  Gamma 비는 각각1.17614175/0.87463838이다. Si 과이온화·S 저이온화의
  방향과 부합하나 다른 물질에서 생성된 비수렴 장 비교라 유일 원인 판정은 아니다.
  Si 차이의68.83%는 앞의1650 Å 세 경로, S 차이의94.29%는631–634 Å의
  세 경로다. 같은 점에서 부호가 반대여서 공통 회색 J 배율로 두 율을 동시에
  맞출 수 없다. 기존 Sobolev/MC ALI나 설계 전용 메모를 현재 full-SR ES 수리로
  바로 이식하지 않는다. 전모형 BF6/SE/수송 없이 CPU1·29.60초에 선택 photo
  적분으로 확인했으며 상세는 아래 인계의 `CMFGEN J로 Si III/S III bound율 직접 비교`다.
- 전90점 확장(11월1일): 같은 정의를 Si II/III·S II/III의975경로·90점으로
  확장했다. 보간의 전치 작용으로 fine 격자 구적을 그대로 원 CMFGEN 격자에
  합산하는 비교기를 구현했고, 18시험 및 이전 직접 적분과 상대2.22e-16로 재현했다.
  Lageunha CPU1·4GiB에서38.83초에360행을 얻었다. S III의 현재/참조율은
  점34/50/61에서0.875/1.430/0.997로 부호·크기가 깊이에 따라 달라진다.
  [전90점 bound photo 프로파일](CMFGEN_BOUND_PHOTO_PROFILES_2026-11-01.csv)은
  고정B 인구의 photo율이며 이온분율·수용 판정은 아니다. 구현/재현은
  `compare_cmfgen_bound_photo_profiles.py` 및 10월31일 인계의
  `90점 bound photo 비교의 재사용 구현`을 참조한다.
- 실행 판본 재대조(11월1일): 실제444706의 control/stage는
  `lumina_f1_generation15_sources_20261031/dispatch_bundle_a100pcie.ZrU1ai` 아래다.
  `lumina_feedback_generation15_20261031`의 미제출 구성을 현 실행으로 대용하지 않는다.
  실제 stage76파일의 해시가 봉인값 및 HEAD와 전부 같고 두 실행 binary도 일치했다.
  이 범위에서 코드 소실/앱·터미널의 서로 다른 물리 판본은 확인되지 않았다.
  Slurm 출력 발견 후 단회 확인에서는 H200에서 RUNNING이었다. 최신 상태는
  인계서 맨 위와 `실제 제출 코드와 현재 코드의 신원 대조`를 따른다.
- generation15 새 계수/경계 job `1250253`은 완료됐지만 상태는
  `PREPARED_NEW_COEFFICIENT_BOUNDARY_NOT_RUN`이다. 원 주파수1,115,020에서 총소광
  음수표본7,985, signed line 소광 음수표본10,278,542이며 producer는 native zero
  replacement/`SRCE_CHK`를 적용하지 않았다. 10월27일 이전 generation의 유사 총소광
  음수표본은8,275였으며, raw 개수 비교만으로 새 회귀를 판정하지 않는다. 귀속 job
  `1250462` 결과는 SHA256
  `60b0cc6482f45c0c8daae70e06748c85826a63a9d68ad91a9d8619183b4498a1`; 표본 수는
  producer와 per-node 합계가 일치하고 46점에 분포한다. 가장 음수인 12개 대표점
  전부에서 Si II FL124→125가 최대 음수 선 기여자였고, 같은 반경의 original70
  재구성 C-full에서도 역전 상태로 표기됐다. 이는 최대 사례의 역전이 generation15에
  새로 생긴 것만은 아님을 보이지만, 나머지 7,973개 표본의 개별 원인이나 native
  `SRCE_CHK`·연속소광 하한의 새 J/열 영향은 판정하지 않는다. 수송은 0회이며
  물리 수용·CMFGEN 동등성을 뜻하지 않는다. 수치와 검증 범위는
  [10월30일 인계](HANDOVER_2026-10-30.md)의 generation15 귀속 절 참조.
- Si II FL124→125의 Δm 의심은 이미 10월28일 원자료·Lumina 변환기·CMFGEN 경로와
  허용 E1 superlevel 성분 합산을 대조했다. generation15의 두 importer 파일 SHA도
  그때 기록한 SHA와 일치하고 현재 선 목록 `f=2.975`도 같다. 독립 hydrogenic 가중합
  `f=2.974565825179`는 원행과 0.0146% 이내다. 이 한 묶음 전이에서 Lumina만
  Δm/허용 성분을 누락했다는 증거는 없으므로 임의 필터를 되살리지 않는다. 11월1일
  현 저장소도 확인했다: importer의 이후 변경은 FF 정책으로 한정되고 `gf_selection`
  본문은 유지됐으며, 관련 기존 시험 44개가 통과했다. 이는 다전자 원자료 전체나
  모든 Si 전이의 정확성 증명이 아니다. 정본:
  [음수 소광·FL124→125 이력](LUMINA_NEGATIVE_NODAL_SOURCES_2026-10-27.md)의
  `Si II FL124→125 Δm 선택규칙과 두 코드의 원자료 경로` 및
  `FL124→125의 허용 성분 합산 추가 검증`.
- 부호 있는 소광의 기존 처리(10월29일): Lumina production fullSR reader와 양의
  scaled-moment core는 signed `chi`/gain을 처리하므로 raw 음수 개수만으로 실행
  실패라 단정하거나 `chi=0`으로 clamp하지 않는다. 반면 원 CMFGEN
  `NEG_OPAC_OPT=SRCE_CHK` 및 연속소광 하한은 다른 수송/열 정책이다. 동일한
  과거 계수에서 raw 총소광 음수8,910개, 첫 native `SRCE_CHK` 조건 적용17,288개,
  `ES×10%` 조건381개였지만 이는 반사실 계수 판독이지 native 재실행이 아니며 새 J/
  수렴해 영향도 미확정이다. 그러므로 generation15의7,985개는 그 자체로 새 회귀도,
  CMFGEN parity도 아니다. 정본: [10월29일 인계](HANDOVER_2026-10-29.md)의
  `후속 조사 범위 — 부호 있는 소멸계수와 native 교체 규칙` 및
  `실제 읽기 결과와 판단`; [현재 plan](CURRENT_PLAN.md)의 deterministic fine producer.
- 10월31일 배선 확인: generation15 producer `1244ca46`와 현재 `HEAD` 양쪽에서
  full-spectrum 경로(`scripts/lumina_fullsr_sc.py` → `cmf_fullsr_sc.cpp`)는
  aggregate `chi, eta`를 읽어 signed `chi`로 수송한다. 이 경로는
  `line_net_sobolev_material`을 호출하지 않는다. 반면 그 helper는
  `lumina_cmfgen.c`/`lumina_plasma.c`의 선별 Sobolev 경로에서 개별
  `tau_sobolev`에 적용되며, `tau < -0.5`일 때만 `SRCE_CHK` 유효값으로 바꾼다.
  코드 배선 차이만으로 현재 물질의 line-opacity 차이를 판정하지 않도록
  1250253의 동일 C-full 인구·선 목록·팽창시간에서 line `tau`를 재구성했다.
  790,493×90 쌍 중 `tau<0`은5,572,747개지만 `tau<-0.5`는0개이며, 최솟값은
  Si II FL124→125/node54의`−0.1562763`이다. 따라서 표준 Sobolev `tau`가 native
  trigger라는 전제하에 **개별 Sobolev helper**의 치환은 이 generation에서 발동하지
  않는다. 이는 CMFGEN native 전체 소광 aggregate `SRCE_CHK`의 발동 여부를 판단하지
  않는다. committed `comp_opac.f`에서는 continuum χ floor가 확인되지 않았다. FL 수와
  부호 있는 총 `chi` 음수표본은 `tau<-0.5` 수를 대신하지 않는다. 코드 경로 차이가
  미래 상태에 중요할 수 있음을 지우지 않고,
  현재 line helper 결과를 aggregate native 정책으로 일반화하지 않는다. 새 계수공간
  반사실 분석은 다음 census를 참조한다. 전체 CMFGEN parity는 미해결이며 aggregate
  `chi`를 production에서 clamp하지 않는다.
  재현 결과/범위와 F 계보 구분은 [10월30일 인계](HANDOVER_2026-10-30.md)의
  `generation15 물질 전달과 C full 일치` 절 참조.
- 10월31일 generation15 소광 counterfactual 정정 및 재계산: 이전 census는 CMFGEN source
  commit `dd1660af`의 committed `comp_opac.f`가 아니라 uncommitted working-tree의
  `LUMINA neg-opacity floor` 변경을 반영했다. committed file에는 continuum χ floor가
  없고 ETA 하한만 있다. 수정한 분석기로 committed-stock 정책을 재계산한 결과,
  raw 총소광 음수는 7,985개이며 source/scattering branch는 8,659/381개, 총 변경은
  9,040개다.
  600–750 Å 변경은0개이고 최대 `|Δχ|=4.16249e-14 cm^-1`이다. 반면
  27,888,567개 floor 표본, patched continuum 아래서 계산한 16,764/359개 branch 및
  600–750 Å의189,819개는 local patch counterfactual이지 stock CMFGEN 수치가 아니다.
  이전 `generation15-native-opacity-policy.json`은 stock 판정에 사용하지 않는다.
  재계산은 `generation15-opacity-policy-variants.json`에 기록했다(보고서 SHA256
  `14c9c955465071d5dae35ed3ff3b14fb7c1fe7393c5a7dcfef72009a630a947b`). producer
  `sources.npz` SHA256은 `a6828fc343e59b910e77c8ad9a619885a1de5db1f08c43e5ba907116bbae8793`이다.
  해당 source diff는 이 운전자가 수정하지 않았고 작성자나 실제 CMFGEN binary 적용 여부는
  미확인이다. 추가 대조로 job1244808 manifest의 실행파일 SHA
  `f2b9afcc…`가 현재 설치 binary와 일치함을 확인했다. 같은 설치의 `libsubs.a`
  SHA256은 `f719c7393dcb49fca6dd603c43db8616533fc9b757c980ebe7588eb82184048a`,
  `comp_opac.o` SHA256은 `afc5464efedf2c8976410694c43c14bf72da27f293fc0a9a962f1b24fdd1ac08`이며
  바이너리에 `comp_opac_` symbol이 있다. 이는 floor patch object 사용 가능성을 높이지만,
  manifest에 연결 archive/object hash나 link map이 없어 실제 링크를 확정하지 않는다.
  그러므로 비교 실행 정책은 미확정이며, stock-source census와 patch 반사실을 모두
  별도 후보로 보존한다. 한편 같은
  committed `cmfgen_sub.f`에서 확인한 native `NEG_OPAC_FAC=0`의 downstream 의미는
  유효하다: `DO_VAR_CONT`의 line extinction/stimulated-opacity J/H 응답 도함수와 T
  미분에 곱해지지만, direct `ZNET_SIM`/SE line rate와 line emissivity tangent에는
  적용되지 않는다. 즉 BA radiation-response mask이지 원자전이 제거가 아니다.
  aggregate χ 교체는 별도로 수송 opacity를 바꾼다. J/수렴해/이온분포 영향은 아직
  측정하지 않았다. 정본: [10월30일 인계](HANDOVER_2026-10-30.md)의
  `generation15 CMFGEN 전체 소광 정책의 계수공간 대조`.
- 11월1일 J71 실행파일을 특정한 후속 확인: 원429062 runtime→입력→빌드 manifest→
  실제 executable81397292→보존 libsubs.e4d5ec0d를 연결했다. 이 archive의
  `comp_opac.o`는43872288이며 현재 설치 archive의afc5464e와 다르다.
  기준 객체2함수와 호출부8함수의 relocation 제외 코드, 해석되는 named 주소,
  참조 수치 상수가 실제 executable과 일치했다. 기준 객체에는1e-280이 있으나
  1e-3은 없고 현재 설치 primary 함수는 크기/코드 음성대조에서 거부됐다.
  따라서 현재 설치된 continuum-floor patch 객체를 J71에 귀속하지 않는다.
  J71의 저장 MODEL/VADAT는 NEG_OPAC_OPT=SRCE_CHK이며 봉인 호출부도 확인했다.
  다른 binary인 job1244808의 링크 불명 이력은 그대로 유지한다. 모든 보정 부재나
  완전 stock/released 실행을 인증한 것은 아니며, generation15 census를 물질70
  J71의 실제 발동 개수로 바꾸지 않는다. 소스/계수 clamp·새 물리 실행은0회다.
  [J71 binary 정본](CMFGEN_J71_OPACITY_BINARY_IDENTITY_2026-11-01.json) 참조.
- 같은 묶음의 F2→F1 저장장 비교에서 `max|ΔJ|/max(J_F1)=0.127354`였다.
  입력·출력 배열 SHA와 상세 의미는 [10월30일 인계](HANDOVER_2026-10-30.md)의
  `F2 입력장과 새 F1 수송장의 변화` 참조. 단일 수송 변화이지 외부 고정점
  잔차나 수렴 증거가 아니다.
- 범위 구분: 주 비교는 matched-physics steady self-run이다. 공개 StaNdaRT의
  시간의존·비국소 침적·비열·다른 원자모형은 과학 교차검사로 구분한다.
- 구형 `LUMINA_GPH_JTABLE` 경로는 CMFGEN 기준장 동등성이나 production 수리가
  아니다. `run_common_cmfgen_field_response.py`는 고정장 Si/S 국소 응답 진단이며,
  full transport·전 원소 물질·전자밀도 갱신을 수행하지 않는다. 그 실행기는 이전
  `jnu4` CMFGEN 자료를 읽는다. `build_cmfgen_jtable.py`는 비양수 J 표본을 log 보간
  결측으로 취급하고 한쪽 깊이 표본만으로 보간할 수 있으며, C 주입부는 양수 표만
  덮어쓰고 나머지는 Lumina 장을 유지한다. 따라서 그 진단 결과를 공통 J의 증거나
  released-T 기준해로 승격하지 않고, 현재 production 원인으로 재해석하지 않는다.
- 기존 연결: `run_lumina_resident_observer.py`와 관측자 NPZ 비교기는 이미 있다.
  공통 `InputField`가 동일 물질의 완료장 계보를 검사한다. 새 observer나 중복
  계보 검사를 만들기 전에 [10월30일 인계](HANDOVER_2026-10-30.md)의
  `최종 스펙트럼 deliverable의 실제 경로`와 그 뒤의 바이너리/소스 대조를 읽는다.
- 최종 formal metric 범위(10월31일 재확인): v2 수치 허용폭은 정해져 있지만
  EW·특징속도의 line ID, 파장창, continuum 및 blend 규약은
  [C 리뷰](CODEX_EQUIV_PLAN_C_REVIEW.md)의 acceptance 결함으로 남아 있다.
  `plot_spectrum_vs_cmfgen.py`는 공개 StaNdaRT와의 정규화 진단 그림이고,
  `compare_spectra_v2.py`는 TARDIS 비교다. SN 2011fe/HST 등 다른 모델·목표의
  창을 matched-physics toy06 최종 acceptance에 그대로 승계하지 않는다.
  수렴 released-T 기준과 해당 모델 identity가 고정된 뒤 실제 스펙트럼을 보기
  전에 manifest를 확정해야 한다. v2 문턱은 변경하지 않았고 새 observer 실행도 없다.
- 정본: [Si/S 묶음](BUNDLE_CMFGEN_SI_S_BF_CHANNELS_2026-10-19.md)의
  `실제71 완료`, `감쇠/복원 분기 종료`;
  [동등성 계획 v2](CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md),
  [공개 모델 차이](PUBLIC_TOY06_COMPARISON_2026-10-30.md).
- 11월2일 실행시간 정정: 처음의 약10:07 직렬 추정은 새 J에서 필요한 centered
  T/ne 방향 네 응답을 빠뜨렸다. 1254013의 네 응답은 실제4:20:28이고,
  `lumina_thermal_replay.load()`가 저장 방향의 field 신원을 현재 복사장과
  일치시킨다. 한 새 물질→계수/경계→F→응답→새 방향→T/ne 수정 iteration은
  관측 유사 단계 합 약14:27:45이며 큐 대기·추가 수렴 iteration·matched steady
  CMFGEN 기준해·observer spectrum은 별도다. 444706은 F worker 완주 뒤 dispatcher
  DNS로 FAILED였으므로 해당 H200 시간은 성공 bundle이 아닌 worker 처리량이다.
  상세 산술과 권한 경계는 [11월2일 인계](HANDOVER_2026-11-02.md)의
  `후속 실행시간 정정`을 따른다. 새 실행은 아직 승인되지 않았다.
- 11월3일 새-J 반환장 응답 `1260171`(source `cf147be68eef17eb1a35cf74f924f09cc6c5b717`)
  의 고정온도 CSV/NPZ를 재집계했다(result SHA `7d2d5248ee5ab2706728310a82c18eb68104792eeabf5b77c42d64bdc4c467f5`).
  안쪽 node80–89의 Si III `log10(response/CMFGEN70)` 중앙값은 `+0.2759 dex`
  (범위 `+0.1391…+0.2872`), S III는 `+0.1708 dex`
  (`+0.0770…+0.1818`)다. 같은 점의 T 차 중앙/최대는 `0.0243/0.1683%`, ne는
  `0.1546/0.1620%`였고, 새-J 응답/채택 분율의 중앙 로그비는 각각 `+0.2495`,
  `+0.1600 dex`다. 작은 T/ne 차이만으로 이 분율 패턴을 설명하기 어렵다는 것은
  후속 율 분석의 우선순위를 정하는 **추론**일 뿐이다. 이 비교는 fixed-T·비수렴
  CMFGEN70 진단과 분율-only 활성점에 한정되어 순수 J 인과, 전체 v2 수용, released-T
  동질성이나 새 수송 수렴을 증명하지 않는다. 11월1일 점34의 PHOT/J 비교와 입력장·
  상태가 다른 별도 사례로 보존하며 새 회귀라고 부르지 않는다. 수치·파일 SHA 및
  현재 실행 상태는 [11월3일 인수인계](HANDOVER_2026-11-03.md)의 `추가 판독`을 참조한다.
- 같은 현재 반환 인구로 저장된 bound photo율만 직접 비교한 결과(4이온×90점,
  44.79초, report SHA `a48426e695632ebafe423f8fea1abad600d19ed8104f96d73fba54026d31062c`,
  profiles SHA `f7867644de25ab2cce03a87dd4fae4561a2c31da2c57688ce586672869e05410`),
  안쪽 node80–89의 새/CMFGEN J 이온당 율 중앙값은 Si II/III `0.9449/0.9324`,
  S II/III `0.9107/0.9308`이고 이전 장은 `0.9918/0.9968/0.9930/0.9969`다.
  즉 새 장에서 네 stage bound photo율은 모두 감소했지만 Si III/S III 분율은 상승했다.
  따라서 직접 BF 광이온화 증가만으로 III 분율 상승을 설명할 수 없으며, 역 Milne·
  자극 재결합·BB·충돌 등은 별도 같은 상태 비교 없이는 미확정이다. 이 고정 인구
  율 진단은 CMFGEN 총율이나 원인 확정이 아니며, 11월1일 점34 사례와도 구별한다.
  입력·범위·출력 신원은 [11월3일 인수인계](HANDOVER_2026-11-03.md)의 `추가 율 분해`를 따른다.
- 같은 generation14 반환 인구에서 저장 총 BF photo와 bound-only Gamma를 결합했다
  (11월3일, node80–89). pseudo photo 분율 중앙값은 Si II/III·S II/III 순서로
  0.080/0.213/0.281/0.405%, 최대도0.408%였다. BF spontaneous/photo는
  0.984/0.981/0.989/0.996, stimulated/photo는0.016/0.019/0.011/0.004이며
  저장 BF net/photo 중앙값은 Si II `6.3e-5`, Si III `7.6e-6`, S II `7.1e-8`,
  S III `3.3e-7`이다.
  따라서 이 inner 구간에서 pseudo photo가 큰 III 과잉의 주원인이라는 설명은
  지지되지 않는다. 이 net은 BF 장부만의 비율이며 전체 SE 잔차가 아니다. CMFGEN의
  대응 역율·bound-bound/충돌 및 열·수송 공동수렴은 미비교다. 수치와 해시는
  [11월3일 인수인계](HANDOVER_2026-11-03.md)의 `동일 인구의 총 BF 유량과 pseudo photo 기여` 참조.
  generation14 응답은 inverse율 cache 경로24426개와 직접 재적분241개를 사용했다.
  직접 계수가 별도 저장되지 않았으므로 기준 `bound-rates.npy`만으로 전체 응답의
  bound/pseudo 역율을 분해하지 않는다.
- Astra가 코드 read-only 검토에서 제기한 `ModelResponse.evaluate()`의 seed 기반
  occupation/fraction과 SE 반환 population 불일치를 generation14 실자료로 대조했다
  (2026-11-03). 선택 response SHA `216347a338d3adb66337cd46180135c71dec42581c18f48f0351bcb5055eb2a8`,
  result SHA `7d2d5248ee5ab2706728310a82c18eb68104792eeabf5b77c42d64bdc4c467f5`에 대해
  해당 MODEL/OSC/준위 입력과 최종 population·T·ne로 `material_occupation` 및 LTE
  분율을 재계산했다. 24,542×90 occupation, fraction, population×fraction이 저장값과
  정확히 일치했고, seed/최종 총 양이온 밀도 합의 상대차는 `3.91e-15`였다. 따라서
  코드 흐름은 향후 상태에 대한 잠재 위험으로 남지만 이 generation14 출력의 stale
  occupation 원인설은 반증된다. 이는 다른 모형·상태의 보증이나 독립 CMFGEN 인구
  일치가 아니며, 코드 수정·새 계산·추가 게이트를 만들지 않았다. [11월3일 인계](HANDOVER_2026-11-03.md)의
  `점유율 stale 가능성 재검사`를 참조한다.

## O01 장시간 git add와 대형 미추적 계산 출력

- 과거 처리: 10월30일 syntax의 Codex 자식 git add가 미추적8GB
  `lumina_events.bin`을 읽으며4시간 넘게 임시 pack을 생성했다. 그 PID만 TERM으로
  종료하고 미사용 미완성 pack 약4.57GB를 삭제했다. 원본 데이터와 실제 인덱스는 보존했다.
- 재사용: 이 checkout의 `.git/info/exclude`에 `/lumina_events.bin`을 추가했다.
  **로컬 제외 규칙은 커밋·다른 checkout으로 전파되지 않는다.** 새 checkout에서는
  해당 파일을 다시 확인한다. git add는 변경한 정확한 경로만 지정한다.
- 구분: CPU를 쓰는 실행 상태였지 Z 상태의 좀비가 아니었다. 별도 임시 인덱스는
  실제 index.lock 충돌과 다르며, 다른 작업의 지연 원인이라는 인과관계도 별도다.
- 근거: 해당 세션의 종료 전후 PID·cwd·fd·인덱스/pack 확인 및 현재 로컬 제외 규칙.
  다른 Codex 부모 프로세스나 Slurm 작업까지 종료한 것으로 기록하지 않는다.

## O02 GPU 완료 뒤 CPU 제출의 DNS와 구성 전달 누락

- 과거 수리: 10월29일 계산 노드의 `ssh grammar` DNS 실패를 확인해
  `submission_host_addresses`와 원 alias/엄격한 호스트키를 유지하는 숫자 IP
  연결을 dispatcher에 추가했다. 이 수리 자체는 현재도 있었다.
- 11월1일 재발 원인: 일반 `prepare_lumina_bundle_dispatch.prepare`가 주소를
  입력받아 새 config에 보존하지 않아, 새 구성2bfe7832에는 주소가 빠졌다.
  F444706은 완주했지만 callback이 DNS 단계에서 실패해 batch FAILED가 됐다.
- 처리: 원격 명령 미실행·큐/회계0·미사용 parent를 확인하고 첫 CPU1253439를
  grammar 로그인에서 제출했다. 원 실패 marker를 보존했으며 GPU 재실행은0이다.
  생성기에 명시 주소 입력/직렬화를 연결하고, 누락된 config는 GPU 제출 전에
  거부하도록 수정했다. DNS 추측·키 검증 해제·자동 재시도·실행 한도 증가는 없다.
- 재현: 관련63시험 통과(Lageunha CPU1,4.42초). 현1253439는 기존860ea692
  봉인 소스이며 수정의 소급 적용을 주장하지 않는다. 원 정본은
  [10월29일 인계](HANDOVER_2026-10-29.md)의 `새 실제 운영 결함`과
  [10월31일 인계](HANDOVER_2026-10-31.md)의 `F 완주와 DNS 실패 복구`다.
- 11월2일 후속 fail-fast: 주소가 config에 있고 IP 형식이 맞는 것은 실제
  GPU 계산 노드에서 연결된다는 증명이 아니다. `submit_lumina_accepted_feedback.slurm`은
  이제 해당 노드에서 strict host-key 검증을 유지한 `ssh grammar hostname -s`를
  먼저 실행하고 성공할 때만 F worker를 시작한다. 실패 시 재시도 없이 그 자리에서
  중단하므로, callback 연결 오류가 장시간 전주파수 작업 뒤에야 드러나는 일을
  막는다. 직접/묶음 순서시험48개 통과. 실제 계산 노드 handshake는 아직 실행하지
  않았으므로 연결 성공 자체를 주장하지 않는다. [11월2일 인계](HANDOVER_2026-11-02.md)의
  `후속 실행 보호`를 따른다.

## 이력 추가 형식

기존 항목을 갱신하는 것을 우선한다. 새 원인은 증상, 원인·해결 방식, 당시
입증 범위, 현행 적용 여부, 원 정본/코드·입력 판본, 재사용 한계만 짧게 남긴다.
세부 해시·명령·raw 결과는 기존 묶음 정본에 두고 여기에는 찾는 경로를 연결한다.
현재 상태를 재확인하지 않은 항목에는 과거 수리 사실만 적고 현행 재현으로 승격하지 않는다.
