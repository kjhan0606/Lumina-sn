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
- 정본: [10월30일 인계](HANDOVER_2026-10-30.md)의 `열잔차 정규화의 해석 한계`;
  [SL 에너지 계약](LUMINA_SL_ENERGY_CONTRACT_2026-10-27.json).

## P10 공통장 인구 일치와 자기일관 해 및 CMFGEN 기준해

- 과거 확인: 실제71의 같은 T/ne/J에서 Si/S 전90점 조건부 C1 비교를 수행했다.
  그러나 물질·장 반복의 자체 이동과 감쇠 분기 관찰 목표는 별도로 미달했고,
  단순 반-step/복원 방법은 채택하지 않았다.
- 재사용: 조건부 인구 일치는 독립 장·온도·스펙트럼 일치가 아니다. J71은 물질70의
  장이라는 생산 phase를 확인한다. 물질→그 물질의 계수/경계→새 J→반환장 응답의
  고리를 닫아야 한다. fixed-T 비수렴 CMFGEN70은 인증 oracle이 아니다.
- 범위 구분: 주 비교는 matched-physics steady self-run이다. 공개 StaNdaRT의
  시간의존·비국소 침적·비열·다른 원자모형은 과학 교차검사로 구분한다.
- 기존 연결: `run_lumina_resident_observer.py`와 관측자 NPZ 비교기는 이미 있다.
  공통 `InputField`가 동일 물질의 완료장 계보를 검사한다. 새 observer나 중복
  계보 검사를 만들기 전에 [10월30일 인계](HANDOVER_2026-10-30.md)의
  `최종 스펙트럼 deliverable의 실제 경로`와 그 뒤의 바이너리/소스 대조를 읽는다.
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
