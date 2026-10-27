# CMFGEN fixed-T 기준 상태 추가 반복 계획

## 목적과 경계

목적은 미수렴 CMFGEN reference의 Si II/III 및 S 이온분율을 같은 물리 조건에서
계속 계산하여 기준의 반복 안정성을 평가하는 것이다. 최종 목표인 Lumina와의
물리량·관측 스펙트럼 동질성에는 직접 필요한 비교 기준을 개선하지만, 이것만으로
Lumina와 동등하거나 released-T 물리 앵커를 얻는 것은 아니다.

이전 작업 `1225397`은 record70에서 시작해 내부 `NUM_ITS=4`에 따라 record74에서
정상 종료했다. Slurm 12시간을 다 쓰지 않았다. 73→74 최대 이온분율 절대변화는
Si II 6.32e-7, Si III 0.159574, S II 1.35e-4, S III 0.114642였으므로 74를
수렴 기준으로 채택하지 않는다. 실제 추출은
`lumina_native_long_20261027-9RyXe2/run/material`에서 90점 전체, 다섯 상태를
보존했고 출력 SHA는 인수인계서에 남겼다.

## 실행 설계

- 시작점은 검증된 record74만 사용한다. 원70 material identity를 재검증하고,
  이전 실행의 완료 결과·native/build 실행파일 SHA, record70~74 계보,
  POINT1/POINT2=74, RVTJ 완결성과 고정 T를 확인한다.
- 이전 실행을 수정하지 않는다. 기존 원 seed에서 새 run 복제본을 만들고
  record74의 SCRTEMP, 포인터, RVTJ, STEQ_VALS, EDDFACTOR, EDDFACTOR_INFO,
  CUR_MODEL_DATA만 명시 restart overlay로 전달한다. 출력 디렉터리는 신규·배타적이다.
- 이전과 같은 corrected native 실행파일, 순간 침적열, FIX_T=T, FIX_NE=F,
  USE_FIXED_J=F, COMP_BA=T, N_FIX_BA=0, 자동 T/Lambda 전환 금지, 기존 수치
  제어를 유지한다. `NUM_ITS=16`은 record74 이후의 최대 추가 횟수이며 목표
  반복 번호가 아니다. 변경 전 기본 4회는 그대로 보존한다.
- 자원은 CPU16/OMP16/128GiB/GPU0, 내부 최대42600초·Slurm 최대12시간,
  재시도0이다. 이는 수렴 보장이 아니다. 첫 반복의 비용이 선형 확장과 다를 수
  있으므로 시간 완료를 예상하지 않는다. 제한이 끝나면 마지막 완결 record/RVTJ와
  부분 출력을 보존하고 불완전으로 분류한다.

## 관측값과 판정

90점 Si/S 모든 charge-stage 분율, 각 반복 개체수 및 `ne` 변화, raw CMFGEN 보정,
fresh BA 로그, 최종 RVTJ 일관성, 입력·실행파일 신원과 실제 경과/최대 RSS를
보존한다. 각 연속 반복에서 최대 절대 분율 변화를 계산하며 최대 위치도 기록한다.
원소 총수 및 terminal closure는 기존 추출기와 POP 출력으로 대조한다.

반복 수 상한 도달이나 정상 프로세스 종료만으로 수렴을 선언하지 않는다. 최종
동질성 판단에는 사전 물리 수용기준에 따른 active population 통계, 최종 반복들의
분율 변화, 유효 동일세대 RVTJ, Lumina의 동일 물질·공통장 물리량 및 스펙트럼
비교가 모두 필요하다. 이번 묶음은 fixed-T 안정성 조사이며 온도 해제/새 BA 설정/
원자자료/물리식을 바꾸지 않는다.

## 실행 안전과 재현

운전자는 봉인된 저장소 commit에서 전용 Slurm staging을 수행한다. 동일
fixed-T continuation 결과 디렉터리가 이미 있으면 제출하지 않는다. 제출 전
grammar의 같은 job-name 활성 작업 부재, 예상 입력 경로/해시, Slurm CPU·메모리·
시간 한도를 확인한다. 원 1225397과 모든 원본/부분 자료는 보존한다. 재현은
`scripts/submit_cmfgen_fixedt_continuation.slurm`의 인수를 사용하며, 제출 뒤
새 job ID와 모든 출력 해시를 본 문서 및 당일 인수인계서에 기록한다.

검토 역할은 계획·평가 담당 Astra, 구현·실행·재현 담당 운전자다. Astra의
read-only 설계 자문을 받은 뒤 운전자가 최종 실행 판단을 기록한다. 결과는 반드시
운전자가 정확한 source/build/material·명령을 다시 확인하고, `PASS`, `REWORK` 또는
`BLOCKED`가 아닌 과학적 최종목표 `NOT_MET`을 분리해 판정한다.

## Astra 계획·설계 평가

Q-GOAL=`PASS`: Si III/S III가 73→74 사이에도 각각 최대0.159574/0.114642
변했으므로 미수렴 기준을 계속 푸는 것이 코드 간 물리 비교에 직접 필요하다.
단 fixed-T 안정화는 필요 단계일 뿐 최종 물리량·스펙트럼 동질성의 충분조건이
아니다.

Q-LEAN=`APPROPRIATE`: 단일 이어계산에서 재시작 identity·계산 안전·전90점
Si/S 분율과 인구/ne/raw 보정·RVTJ를 함께 본다. 원70 재계산이나 반복별 외부
감사는 불필요하다. 기존 상태 배열과 이온분율 추출기를 재사용한다.

설계에서 지적한 위험을 반영했다. 완료 당시 추출 JSON의 SCRTEMP/RVTJ/POPSIL/
POPSUL 해시와 실행 제어 SHA를 재검증하고, 원70 계보·POINT1/2·동일 T까지
확인한다. 각 신규 반복 종료 후 fresh BA 위반을 재검사하고, 최종 완료는
POINT1=POINT2=마지막 SCRTEMP record 및 같은 세대 RVTJ가 모두 맞아야 한다.
Slurm 12시간에서 후처리 여유를 남기도록 외부 timeout을43,000초로 제한했다.

이 반복 설계는 같은 fixed-T 수식의 record74 초기값에서 계속 푸는 조건부
과학적 재시작으로 타당하다. 재시작은 내부 이력을 초기화하므로 record70부터
한 번에 계산한 것과 비트 단위 동일성을 주장하지 않는다. 16회는 최대 추가
횟수다. 기존4회에2시간33분41초가 걸린 단순 비례 추정은 약10시간15분이므로
12시간은 합리적인 자원 한도이나 완료 보장이 아니다. 끝 반복 수/작은 변화만으로
수렴을 선언하지 않는다.

Astra 자문 모델은 `gpt-6-astra`, read-only 평가였으며 작업·계산은 하지 않았다.
재현과 독립 확인은 운전자가 맡는다.
