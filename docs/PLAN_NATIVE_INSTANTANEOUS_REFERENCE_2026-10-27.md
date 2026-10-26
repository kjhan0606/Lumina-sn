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

현재 입력에는 USE_EHB/COMP_EHB가 없고, 원 rd_control_variables.f:979–984는
두 값을F로 정한다. 원70의 실제 STEQ_VALS 마지막 구간도 `Electron Energy
Balance Equation (not computed)`와 복사평형 행을 출력한다. 따라서 현재
열방정식이 전자 열평형으로 몰래 바뀌었다는 후보는 지지되지 않는다.
광자 RE에 CI/C3 전자 열량을 추가해 차이를 맞추지 않는다. 이것은 현재
입력의 경로 확인이며 EHB를 켠 다른 모델의 지원이나 native 전체 잔차/Jv
동등성을 증명한 것이 아니다. 실행 입력/스위치를 바꾸지는 않았다.

source70은 raw 개체수 보정 약88/76%, MAX_LAM=1e6이며 안정화된 seed가
아니다. 원70의 로그 광도1.513e15 Lsun은 USE_FIXED_J=T의 대용 H 결과다.
`comp_j_blank.f:161–165`는 RSQHNU=1/HFLUX_AT_OB=0.001/HFLUX_AT_IB=0으로
놓으므로 그 숫자를 실제 광도나 발산 근거로 사용한 앞선 설명은 철회한다.
새 자체장 실행의 실제 광도를 별도로 판단한다. 열원 교정은 fixed-T·thermal-only에서 이 불안정을
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

## 자체장 fixed-T 후보 실행 구체화 — Astra 보완 반영

full build1223879는11초에 완료했다. 후속은 같은 원70 완료 restart의 새
복제본에서 corrected executable만 사용하는 최대4회 자체장 응답이다.
`USE_FIXED_J=F`로 native가 매번 새 장을 계산하고, 첫 반복은 full BA를
요청하되 원래 `LAM_VAL=400/NUM_LAM=2` fallback을 보존한다. Astra는
`LAM_VAL=1e100`이나 `NUM_LAM=0`으로 fallback을 없애는 안을 반대했다.
DO_LAM_AUTO=F가 fallback 금지라는 과거 주석도 실제 코드와 다르다.

원 MODEL의 COMP_BA=F/N_FIX_BA=2는 과거 행렬을 재사용할 수 있으므로
이번에는 **COMP_BA=T/N_FIX_BA=0**으로 새 자체장의 BA를 재조립한다.
STORE_BA/STORE_BA_INV/WR_PRT_INV는F로 두어 재사용과 대형 중복 출력도
피한다. IT_SPECIFIER를 복제하지 않으며 복제본에 있으면 실행을 거부한다.
MAX_LIN=3/SCALE_OPT=LOCAL은 유지하고 MAX_LAM은1e6 대신 기존 finite10을
이번 staging 수치 설정으로 명시한다. 적용 보정 cap을 raw 수렴으로 보지 않는다.

FIX_T=T/FIX_T_AUTO=F/FIX_NE=F/DO_DDT=F/DO_NG=F 및 실제 SCL_LN·LOCAL·
순간침적·thermal-only·핵종/나이는 유지한다. NUM_ITS=4, DO_LAM_IT=F,
DO_LAM_AUTO=F, DO_T_AUTO=F다. CPU16/OMP16/32GiB/2시간/GPU0/retry0,
자식6600초·파일8GiB·실행출력16GiB 한도이며 오류 시 해당 자식만 중단한다.
원본 및 마지막 유효 checkpoint·부분 출력은 보존하고 자동 재시도하지 않는다.

기존 SCRTEMP의 실제 인구/ne/T 변화·전90점 Si/S stage와 OUTGEN의 raw
보정·광도·실제 BA/LAMBDA 선택, RVTJ 순간열을 함께 남긴다. native는 최종
J/H만 저장하므로 마지막3회 J 안정성을 이 실행만으로 인증하지 않는다.
4회 완주는 앵커나 SE/RE 공동 Newton 수렴이 아니다. 결과는 다음 안정화
방향 선택에 사용한다. 재현: `sbatch scripts/submit_cmfgen_instantaneous_fixedt.slurm
NEW_PARENT SEALED_COMMIT`. 중복 같은 job-name이 없을 때만 한 번 제출한다.

### 첫 BA 초기화 실행 결함 교정 — 최신

1223908의 첫71에서 실제 MODEL은 COMP_BA=T였으나 `BA matrix **NOT**
computed`와 과거 BAMAT 재읽기가 관측됐다. 원 소스는 NLBEGIN=0 뒤
CHK=F로 과거 BA를 읽고, 성공하면서 NLBEGIN=-999가 아니면 WRBAMAT만
정해 실제 COMPUTE_BA를 입력값으로 초기화하지 않는다. 이번 실행에서
그 상태는F였다. 온도·SCL·침적열의 물리식 문제가 아니라 첫 Jacobian
재조립 요청을 수행하지 못한 초기화 결함이다. 큰 raw 보정이나 광도의
원인을 전부 이 하나로 단정하지 않는다.

이 자체 step1223908.0만 TERM으로 중단했고 batch는FAILED1:0/17분06초,
step은CANCELLED0:15다. 원본·완료71·부분72·실패 로그를 보존했다.
원 실패 JSON은 마지막 RVTJ가70에 머물러71 SCRTEMP와 연결되지 않음을
보고한다. 이것을 열원 실패나 후보 파일 유실로 부르지 않으며, 새 판독은
부분 checkpoint와 최종 RVTJ 연결 실패를 분리한다. 실패71은 새 seed로
쓰지 않는다. 새 실행은 같은 원70에서만 시작한다.

Astra는 최소 초기화 수정과 동일 한도의 교정 실행을 지지했다. private
cmfgen_sub의 CHK=F 한 위치를 `COMPUTE_BA=COMPUTE_BARDIN; CHK=COMPUTE_BA;
SUCCESS=F`로 바꾼다. 명시T는 과거 BA 읽기를 건너뛰고 기존 분기에서
fresh BA로 들어가며, 명시F는 정상 cache를 재사용하고 cache 누락/읽기
실패나 새 모델은 fresh BA로 간다. 과거 미정 상태의 byte 재현이 아니라
명시 설정의 의도를 보존하는 수정이다. 실제 native READ_BA와 원/수정
초기화 블록을 컴파일한 시험은 원 결함을 재현하고 수정16조건을 통과했다.
단독시험만으로 실제 첫 물리 반복의 재조립을 입증했다고 하지는 않는다.

새 full build는 두 실행파일에 공통으로 이 수정만 더한다. 이후 새 build와
보고서 SHA를 명시 인수로 전달하고, 실제 첫 갱신 전 cache 성공-read가
없으며 BA 계산이 수행됨을 확인한다. 같은 미조립이 나오면 첫 반복에서
중단한다. partial 판독/BA 위반 시험4개도 통과했다. 이전 solver의 terminal을
확인했으며 관측 timeout 때문에 재실행하는 것이 아니다. 원래 작은 빌드
CPU1/4GiB/10분, 단일 corrected 실행 CPU16/32GiB/2시간/최대4회/retry0와
기존 fallback·cap·T고정·원본 보존을 유지한다. 자동 반복 제출은 없다.

최신 재현 진입점은 `sbatch scripts/submit_cmfgen_instantaneous_fixedt.slurm
NEW_PARENT SEALED_COMMIT NEW_BUILD BUILD_REPORT_SHA256`이다. 앞 절의
2인수 명령은 교정 전 이력이며 새 실행에는 사용하지 않는다.
