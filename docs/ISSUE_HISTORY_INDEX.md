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
- 정본: [10월30일 인계](HANDOVER_2026-10-30.md)의 `고정장 수렴 판정의 후보 척도 수정`,
  `실행 봉인본과 현행 열수용 기준 대조`.

## P09 작은 열잔차와 BB 왕복 상쇄 및 에너지 정책

- 과거 확인: 작은 gross 정규화 잔차와 순채널 잔차가 거의1인 상태가 함께 있었다.
  physical/native 계열은 BB 선 가중·FF·내부 에너지/단열에서 구분한다.
  외부열의 저장0 붕괴열과 국소 순간붕괴 시나리오는 별도의 정책 선택이다.
- 최근 재판독: generation22의 저장 장부를 항별 합산해 두 정책의 net을 정확히
  재현했다. 정책 차이는 주로 BB 가중이며, 인공 가열/physical 가열냉각 척도의
  최대 비는 약4.4e-203이다. 이는 해당 저장 상태에 대한 확인이다.
  원 자료는 `lumina_second_field_thermal_replacement_20261030-fz3u4uV0`의
  `run/evaluation-12/balance.npz`, SHA256
  `d78df23c89193c81e2fb04e9175e95fe313d97e3bac7ef0beae95f4ae8ccea92`다.
- 재사용: absolute net·사전 RE 척도·정책·T/ne/이온분포·장 갱신을 함께 해석한다.
  순채널 척도를 임의의 새 합격선으로 만들거나, 작은 gross 잔차를 최종 물리
  일치로 확대하지 않는다. SL 에너지 proxy 분석은 전자 열해 증명과 구별한다.
- 좌표계 주의(10월31일): native J/H/K는 공이동계 각도로 적분한다.
  F2 외곽 원점의 공이동계 H 적분 광도 `-1.879e44 erg/s`는 같은 저장
  모멘트의 국소 실험실계 변환에서 `+3.979e43 erg/s`다. signed H의 음수를
  음수 J나 관측자 탈출 광도의 음수로 오인하지 않는다. 이 변환 자체는
  에너지 보존·확장 외부 경계 스펙트럼·CMFGEN 광도 동등성의 증거가 아니다.
  재현 범위와 수식은 아래 인계의 `끝점 광도와 좌표계 구분`에 있다.
- 정본: [10월30일 인계](HANDOVER_2026-10-30.md)의 `열잔차 정규화의 해석 한계`;
  [SL 에너지 계약](LUMINA_SL_ENERGY_CONTRACT_2026-10-27.json).

## P10 공통장 인구 일치와 자기일관 해 및 CMFGEN 기준해

- 과거 확인: 실제71의 같은 T/ne/J에서 Si/S 전90점 조건부 C1 비교를 수행했다.
  그러나 물질·장 반복의 자체 이동과 감쇠 분기 관찰 목표는 별도로 미달했고,
  단순 반-step/복원 방법은 채택하지 않았다.
- 재사용: 조건부 인구 일치는 독립 장·온도·스펙트럼 일치가 아니다. J71은 물질70의
  장이라는 생산 phase를 확인한다. 물질→그 물질의 계수/경계→새 J→반환장 응답의
  고리를 닫아야 한다. fixed-T 비수렴 CMFGEN70은 인증 oracle이 아니다.
- 10월31일 현재 CMFGEN source `dd1660af`의 정적 추적: `comp_j_blank.f`는 formal
  `FG_J_CMF_V13` 다음 `MOM_JREL_V9`를 호출한다. MOM은 앞서 `GET_MOMS_REL`에서
  얻은 closure 배열을 사용·저장하면서 MOM 갱신 J/H도 별도로 저장한다. 음수 J의
  `FORM_VAL` 대체와 H 제한(`MAX_VAL`) 뒤 종료조건 `MAX_ER`는 수정된 J의 반복
  변화량만 검사하며, 원 삼중대각식 잔차나 J/H/K 공동 각도 실현성은 검사하지 않는다.
  이는 포획 재생의 `5.3231e-9` 반복변화와 `0.9273846` 원식 잔차가 함께 나온 것과
  맞는다. source 정적 확인이며 capture binary의 정확한 링크 신원은 별도다. 최초
  원인이나 Lumina 결함을 확정하지 않고 CMFGEN source를 임의 변경하지 않는다.
  같은 포획 sweep의 `MOM_J_ERRORS`는 `INIT`에서 rewind되며 음수 원시 J를 대체하기
  전에 기록한다. 해시 `00a8658790a83bdb87e10aa6461c015dc27a162261d746478926b348afd3acd4`인
  로그에는 38,565행/1,537주파수가 있다. 첫 기록은 node7, 298.404 Å,
  `J=-2.4755e-6`(±2점 이웃 양의 최댓값 대비 `7.71e-4`); 318.962 Å에는 처음으로
  그 비가 1을 넘어 `2.62`가 된다. 따라서 490.873 Å 포획점은 음수 J 기록의 시작이
  아니라 뒤쪽 사례다. 최초 음수의 미소 크기만으로 최초 물리 실패를 정하지 않는다.
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
  Δm/허용 성분을 누락했다는 증거는 없으므로 임의 필터를 되살리지 않는다. 이는
  다전자 원자료 전체나 모든 Si 전이의 정확성 증명이 아니다. 정본:
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
- 같은 묶음의 F2→F1 저장장 비교에서 `max|ΔJ|/max(J_F1)=0.127354`였다.
  입력·출력 배열 SHA와 상세 의미는 [10월30일 인계](HANDOVER_2026-10-30.md)의
  `F2 입력장과 새 F1 수송장의 변화` 참조. 단일 수송 변화이지 외부 고정점
  잔차나 수렴 증거가 아니다.
- 범위 구분: 주 비교는 matched-physics steady self-run이다. 공개 StaNdaRT의
  시간의존·비국소 침적·비열·다른 원자모형은 과학 교차검사로 구분한다.
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

## 이력 추가 형식

기존 항목을 갱신하는 것을 우선한다. 새 원인은 증상, 원인·해결 방식, 당시
입증 범위, 현행 적용 여부, 원 정본/코드·입력 판본, 재사용 한계만 짧게 남긴다.
세부 해시·명령·raw 결과는 기존 묶음 정본에 두고 여기에는 찾는 경로를 연결한다.
현재 상태를 재확인하지 않은 항목에는 과거 수리 사실만 적고 현행 재현으로 승격하지 않는다.
