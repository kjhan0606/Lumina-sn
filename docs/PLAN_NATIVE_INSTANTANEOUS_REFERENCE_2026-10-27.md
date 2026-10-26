# 같은 순간 침적열을 사용하는 CMFGEN 물리 reference 준비

## 목적과 현재 판정

최종 목적은 v2 계약의 Si II/III·S 분포, 물리량과 스펙트럼 동질성이다.
검증한0시간 순간 침적열 수정을 **실제 full native 실행 경로**에 연결하고,
고정온도 안정화 후보를 얻어 온도 해제 가능 여부를 판단한다. 빌드 성공이나
fixed-T 정상 종료를 certified anchor·물리 동등성으로 바꾸지 않는다.

직전 턴은 기존 커밋·원격 일치만 확인했으므로 과학적 goal 진전은 없었다.
이번에는 별도 full build 경로를 구현하고 실제 링크 결과를 검증한다.

계획 평가는 Astra `01a178b2-657c-7ba1-ab5a-637a898d5198`, 구현·실행은
운전자다. Astra는 최소패치 full build를 지지했으며, source70이 안정 seed가
아니므로 실제 계산은 fixed-T 후보 안정화까지만 계획하고 온도 해제는
자동 연결하지 말라고 평가했다. 평가를 자원 한도 확대나 계산 제출 자체의
승인으로 표시하지 않는다. 사용자의 자율 진행 권한과 기존 한도를 따른다.

## 범위·입력과 변경

원 native와 원본·기존0열원 계산을 보존한다. 검토한 SHA의 native main
object·라이브러리·include·모듈만 새 출력의 inputs에 봉인 복사한다.
별도 소스의 공통 actual-nu fixed-J 지수 수정은 두 실행파일에 동일하게
적용하며, 원분기/수정분기 decay object를 각각 archive 앞에 명시 링크한다.
단독시험의 열·출력 stub과 포획용 코드는 full executable에 들어가지 않는다.
baseline도 공통 fixed-J 수정을 포함하므로 과거 원본 binary 재현이라 부르지 않는다.

`build_cmfgen_instantaneous_reference.py`는 compile/link만 수행한다. 실제
CMFGEN 계산·설치·원 자료 수정·온도 해제·새 Lumina 수송은 이 빌드에 없다.
원/교정 executable·공통 object·decay source/object·link.map과 모든 원
입력 해시를 기록한다. 양쪽 링크 입력은 공통이며 decay object만 다르다.
기존 출력이 있거나 입력 해시가 변하면 덮어쓰지 않는다.

## 같은 물리 계약과 후속 실행 경계

같은 Ni/Co 수밀도·나이·붕괴자료·LOCAL/thermal-only 조건을 유지한다.
작은 시간 간격, 핵종량 조정, 광도 맞춤으로 차이를 숨기지 않는다.
SCL_LN=T를 유지하는 비교에는 native 호환 열장부·정적 단열을 잔차와
Jv에 일관되게 적용하고 물리 광자 장부는 별도로 보존한다.
PREVENT_LOW_T/T_MIN 인공열은 표시하며 최종 앵커를 온도 보호항에
의존한 수렴으로 인증하지 않는다.

source70은 raw 개체수 보정 약88/76%, 광도1.513e15 Lsun, MAX_LAM=1e6이며
안정화된 seed가 아니다. 열원 교정은 fixed-T·thermal-only에서 이 불안정을
해결하는 수단이 아니다. iteration 번호나 cap으로 작아진 적용 보정만으로
후보를 선택하지 않는다. 같은 세대 restart의 active raw 보정·실제 인구 변화·
J/moment·광도 연속 기록으로 선택한다. 증가 cap, LAMBDA/full 운용, 시작
checkpoint와 보존/중단 규칙을 구체화한 뒤 실제 후보 계산을 제출한다.
현재 문서는 임의의 후보나 온도 해제 레시피를 새로 승인하지 않는다.

## 증거·비용·보존·재현

빌드 수용은 원본/봉인 입력 불변, 출력 미덮어쓰기, 두 executable 링크
성공, 의도한 private decay object 선택과 정의1개, archive의 구형 decay
object 미포함이다. 이는 빌드 신원 확인이지 물리 수용 기준이 아니다.
직접시험은 같은 보존/거부·심볼 검사를 다루고 물리 열 검증은 완료한
`LUMINA_NATIVE_DECAY_FIX_2026-10-27.json`을 인수한다.

빌드 두 개의 합계 한도는 CPU1·4GiB·10분·GPU0·retry0이다. 실행 위치는
grammar Slurm 또는 소형 grammar-debug이고 로그인 노드에서 빌드 묶음을
돌리지 않는다. 실제 후보는 기존 CPU16·OpenMP16·32GiB·2시간·GPU0·
retry0 이내이며 아직 이 문서의 빌드 제출과 함께 실행하지 않는다.
실패/부분 출력과 마지막 유효 checkpoint는 보존하고 자동 재시도·온도
해제·자원 확대를 하지 않는다. baseline 장시간 계산을 자동 추가하지 않는다.

재현 진입점: `sbatch scripts/submit_cmfgen_instantaneous_reference_build.slurm
NEW_PARENT SEALED_COMMIT`. 원본 경로는 runner가 명시하고 새 출력만 쓴다.
결과·실제 job ID·해시와 남은 안정화 작업은 당일 인수인계서에 기록한다.
최종 판정은 여전히 released-T의 v2 전체 잔차·마지막3회 변화·에너지 보존,
실제 격자 사다리와 Si/S·물리량·스펙트럼 증거가 필요하다.
