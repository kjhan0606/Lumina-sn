# 물리 항을 유지하는 native 기준 계산의 메모리 쓰기 절감

## 목적과 판단

최종 목표는 원 v2의 Si II/III·S 이온화, 물리량과 스펙트럼 동질성이다.
이 수정의 목적은 fresh BA 기준 계산을 기존32GiB 안에서 다시 시도할 수
있는지 확인하는 것이다. 작은 시험이나 빌드 완료를 기준 계산·온도 해제·
물리 수렴의 완료로 표시하지 않는다. 직전 턴은 원격 일치 확인만 했으므로
과학적 goal에는 no progress였고, 이번에는 실제 메모리 쓰기 원인을 수정한다.

128GiB 제안은 사용자 미응답 상태다. 추가 메모리를 신청하지 않는다.
원1224011은 첫 갱신 전 OOM으로 권위 종료했으며 실패71을 seed로 쓰지 않는다.
현재 별개 Lumina 온도·전하 Newton1224060은 실제 RUNNING51분04초에서
확인했고 중단·중복·입력 변경하지 않는다.

계획 평가는 Astra `01a178b2-657c-7ba1-ab5a-637a898d5198`, 구현·시험·실행은
운전자다. Astra는 아래 수명주기 불변조건의 직접 재현을 전제로 이 방식을
지지했으며, 누적된 strong 슬롯의 페이지는 반환되지 않으므로 실제32GiB
완주를 보장할 수 없다고 명시했다. 추가 호출을 정례 게이트로 만들지 않는다.

## 고정 입력과 최소 변경

원 native·모델70·선 목록·실제 물리식·NM16006·MAX_SIM8000·ND90·NUM_BNDS1과
NEW_VST_METH 기본T를 유지한다. 변수 공유방식F 전환이나 NM 축소는 하지 않는다.
원 dZ의 가상 크기는92194560000바이트(85.8629GiB)다. 배열 크기 자체는
변하지 않으며, 읽히지 않는 메모리의 선행 쓰기만 제거한다.

`ZERO_dZ`의 행 초기화는 LINE_STORAGE_USED이며 weak가 아닌 **모든** 컬럼에
수행한다. 현재 resonance 안의 컬럼으로 줄이지 않는다. main의 하위/상위
행 해제 두 위치도 같은 helper로 바꾸며 WORKSHARE 종료 barrier 밖에서
호출한다. mask 갱신과 helper를 동시에 실행하지 않는다.

새 strong 선에 대한 원 `ZERO_2D_MAT` 전체 컬럼 초기화와 선 종료의 원
`DP_ZERO`는 그대로다. weak/미사용 컬럼의 stale 값은 소비되지 않으며,
다음 strong 생성 전에 전체 초기화되어야 한다. 강제 슬롯 회수와 다음
주파수 sweep의 포인터·refcount reset도 이 불변조건에 포함한다.

실제 archive/main object의 nm 검사에서 dZ 참조자는 INIT_LINE_OPAC_VAR_V2,
UPDATE_BA_FOR_LINE, cmfgen_sub뿐이었다. 소비자는 resonance/end-resonance와
not-weak guard 안에서만 현재 선 컬럼을 읽는다. SET_LINE_OPAC은 사용 중인
선에만 두 resonance flag를 설정한다. nm 목록만으로 순서 증명을 대신하지 않는다.

## 직접시험·증거와 실행 경계

실제 native 종류/선 module·helper·메모리 함수와 실제 main 해제/SET_LINE_OPAC
회수·sweep reset 블록을 작은 driver에 넣어 원본/수정의 활성 strong 전체
투영과 합산 소비 결과를 byte 비교한다. OMP1/2, 비resonant strong, 행 재사용,
weak→strong 재사용, 정상 해제/강제 회수, 다음 sweep을 포함한다. 신규 컬럼
reset 누락과 한 strong 컬럼만 지우는 주입 결함은 비교에 실패해야 한다.
이 driver의 합산은 실제 BA 물리 산술의 대체가 아니다.

별도 프로세스에서 원 크기의85.9GiB 가상 dZ를 예약하고4 strong 컬럼만
초기화한 다음 실제 원/수정 helper로 행7 하나를 지운다. 전체 배열 poison이나
검사는 하지 않는다. 물리 메모리는 Slurm CPU1·4GiB·10분·GPU0·retry0로
제한한다. 두 결과 합산 일치 및 원 RSS>1GiB/수정 RSS<256MiB는 광범위
row-touch 제거를 판단하는 기준이지 실제 모델 peak 예측이 아니다.

빌드 묶음은 같은 CPU1/4GiB/10분 한도 안에서 수행한다. 원본/봉인 입력
불변·실제 함수 정의 각1개·private INIT/ZERO_dZ 선택·구형 object 미포함을
확인하며, 두 비교 executable은 공통 메모리 수정과 첫BA/actual-ν 수정이
동일하고 decay 분기만 다르다. 단독시험 stub은 full 실행파일에 넣지 않는다.

위 직접시험과 링크 신원이 충족된 뒤만 원70에서 corrected executable의
단일 fresh BA 후보를 기존 CPU16/OMP16/32GiB/2시간/GPU0/추가최대4회/retry0
범위로 제출할 수 있다. fallback·cap·T고정·출력 상한·기존 중단 조건은
유지한다. 재OOM이면 부분 결과와 RSS를 보존한 뒤 멈추며 자동 재제출·
128GiB 증액·온도 해제를 하지 않는다. 실제 모델의 전체 BA/상태 증거는
별도로 판단하고 이후 released-T 및 원 v2의 잔차/Jv·격자·스펙트럼이 남는다.

원 설치와 과거 출력은 읽기 전용이다. 새 고유 parent만 쓰며 원복은 이
선택 빌드를 사용하지 않는 것으로 충분하다. 실패 증거를 삭제하지 않는다.
재현 진입점은 `submit_cmfgen_instantaneous_reference_build.slurm NEW_PARENT
SEALED_COMMIT lazy-dz`와 기존4인수 fixed-T runner다. 완료 계산을 중복
제출하지 않는다. 실제 job ID·해시·실측은 별도 완료 기록으로 남긴다.

## 직접시험 착지

grammar-debug에서 CPU 최대2개·4GiB·180초 상한으로5개 시험이1.205초에
통과했다. 원/수정 OMP1/2의 관측 투영·합산이 byte 일치했고 두 주입 결함은
검출됐다. 첫 시험은 물리 실행 전에 sweep 블록 추출의 공백 위치가 달라
실패했고, 원 소스를 바꾸지 않고 정확한 경계를 수정한 뒤 다시 시험했다.
아직 원 크기 메모리 실측·full 링크·새 실제 BA 계산은 이 기록에서 미수행이다.
