# 사전등록 — DET-OBS-CAPTURE: E′·R′ 동일 view 원천별 탈출 분해

2026-10-06 · 기안  
정본 예정 경로: `docs/PREREG_DET_OBS_CAPTURE_2026-10-06.md`

## §0. 목표·승인·주장 범위

### Q-GOAL — 최종목적에 직접 기여하는가?

이 단의 목표는 기존 E′·R′의 최종 관측자계 작성기 N이 소비한 상태를 캡처하고, **같은 감쇠 아래 원천별 탈출 스펙트럼과 대역 광도를 구하여 저장 N에 폐합하는 것**이다. 이를 통해 E′의 FUVb·UV2500 출력과 E′−R′ 차이를 어느 원천이 지배하는지 판정한다.

diag5는 저장 원장의 무수송 규모와 부분 선 표본까지 조사했지만, 최종 view의 개별 선 τ·원천값과 실제 연속체 입력 결손 때문에 전체 탈출 분해를 완료하지 못했다. 이 단을 생략하면 큰 국소 원천과 실제 탈출 기여를 구별하지 못한 채 후속 수리 대상을 고르게 된다.

**기안자 자기답: PASS.** Fable의 사전감사 판정은 별도 집행 기록으로 받는다.

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

캡처는 최종 view 한 번, 실제 소비 선·셸 쌍, 연속체·J·기하에 한정한다. 전체 반복 dump, 전 level 인구, 비소비 dense 선 배열, 새 W5 표본, ray별 이력은 저장하지 않는다. bf와 ff는 합친 한 성분으로 다룬다.

유지하는 검사는 입력·계보, 캡처 완전성·무변경, 수송 폐합 세 묶음이다. digest·census·음성대조는 이 세 판단에 직접 소비한다. 기존 작성기의 물리 fixture 전체를 새 감사 단위로 반복하지 않는다.

**기안자 자기답: APPROPRIATE.** 사전감사는 Q-GOAL·Q-LEAN에 한정한다.

### 승인과 분장

실행 자체는 다음 user 결정으로 승인됐다.

`/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/evidence/USER_DECISION_2026-10-06.txt`

> user 2026-10-06, 원문 「진행」

본 계약은 승인 범위를 **캡처 구현·시험·코드 감사, E′·R′ 각 한 번의 새 캡처 런, 오프라인 원천 분해**로 고정한다. 정상적인 내부 작업 사이에 별도 실행 승인을 반복 요청하지 않는다. 추가 GPU 런, 물리 변경, 비용 상한 증가는 이 승인에 포함되지 않는다.

| 업무 | 담당 |
|---|---|
| 기안·계약 개정·종료감사 | astra |
| 사전감사: Q-GOAL·Q-LEAN | Fable, 실패 시 현행 대행 규약 적용 |
| 구현·직접시험·기계 게이트·빌드·실행·판독·판정 | 운전자 |
| 코드 감사 | gpt-6-sol |
| 범위·비용·후속 과학 결정 | user |

운전자는 구현자와 판정자를 겸한다. astra는 계약 기안자가 종료감사를 수행한다. 각각의 이해관계를 장부에 남기며 완전히 독립된 감사로 표시하지 않는다. 대행은 실제 호출 실패·대행 근거·실측 모델 ID를 기록하고 원 담당자의 판정으로 재명명하지 않는다.

## §1. 기준선과 보존

```text
P  = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
W  = /gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002
D  = /gpfs/kjhan/lumina/cmfgen_map/diag5_20261006
C  = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006

T1 = W/source
B1 = W/cuda_build2
E0 = W/Eprime/run1
R0 = W/Rprime/run2

T2 = C/source
B2 = C/cuda_build1
EC = C/Eprime/run1
RC = C/Rprime/run1
```

근거는 다음과 같다.

- `docs/PREREG_DET_OBS_WRITER_2026-10-02.md`, 개정 1–5.
- `docs/VERDICT_DET_OBS_WRITER_2026-10-02.md`, E′ 실행 후 두 상태 판정 포함.
- `D/P2_DIAGNOSIS.md` §4와 `D/astra_audit_p2.md` (4).
- `D/DIAGNOSIS_r2.md`, `D/astra_audit_r2.md`, `D/LOOP_CLOSED.md`.

`LOOP_CLOSED.md`의 새 런 필요성 미확정 기록은 1단계 당시의 결론이다. 이후 P2의 결손 판정과 10월 6일 user 결정이 이번 실행의 근거다. 과거 문서를 소급 수정하지 않는다.

운전자는 T1의 실제 파일과 B1의 빌드 소스 manifest를 연결하고, E0·R0의 바이너리·입력·exports·러너·N/L·작성기 manifest 신원을 봉인한다. T1이 B1의 빌드 소스와 다르면 현재 T1을 그대로 복사하지 않고 해당 판본을 먼저 복원한다.

T2는 이 기준선의 사본이며 T1·B1·E0·R0·diag5는 변경하지 않는다. C는 결정문 때문에 이미 존재할 수 있지만, 새 source/build/run/capture/readout 대상이 충돌하면 덮어쓰지 않는다. 해시뿐 아니라 변경 전 소스·등록 patch·빌드 명령을 보존하여 기준선을 재구성할 수 있게 한다.

## §2. 물리 계약과 산출물

### 2.1 성분 정의

각 상태에서 다음 일곱 성분을 수송한다. 분류는 **고정된 최종 view에서의 주입 위치**에 따른다.

| 성분 | 실제 주입 |
|---|---|
| 매핑 선 | 권위 mapped mask가 참인 소비 선의 실제 fallback 선택 후 \(S_{\rm used}\) |
| 미매핑 열 | 소비 미매핑 선의 \(\epsilon B_\nu(T_e)\) |
| 미매핑 산란 | 같은 선의 \((1-\epsilon)J_b\) |
| bf·ff | 기존 capture의 연속체 분자 \(\eta_c-\eta_{\rm dep}\); builder ff와 추가 writer ff 포함 |
| \(\chi_eJ\) | 순수 전자산란 계수와 최종 coarse J의 빈별 곱 |
| 내경계 | 기존 core 교체 규칙과 \(D^3B_\nu(T_{\rm inner})\), `INNER_BB_SCALE` |
| \(\eta_{\rm dep}\) | 기존 capture의 직접 침적 분자; 이 실행에서는 정확한 0을 확인 |

모든 성분은 동일한 전체 연속체 소광과 동일한 선 τ로 감쇠된다. 특정 성분을 계산할 때 다른 성분의 opacity를 제거하거나 J·인구·온도를 재계산하지 않는다.

매핑 선 성분에는 그 \(S_{\rm used}\) 안에 이미 들어 있는 물리 과정이 함께 포함된다. 이를 원자 미시 과정별 분해로 확대하지 않는다. \(\chi_eJ\)와 미매핑 산란도 고정 J를 소비한 주입 분류이며, 산란 광자의 최초 에너지 기원 추적은 아니다.

### 2.2 고정하는 수송

다음은 선행 N과 동일하다.

- 소비 predicate: \(\tau>10^{-12}\), 엄격한 주파수 범위, 유효 coarse bin.
- 원래 선 ID와 descending 주파수 순서, 동주파수 선의 기존 순서.
- 반열린 공명 구간 소유와 선 앞·뒤 연속체 분할.
- 256 ray, 연속체 최대 속도 간격 30 km/s, 기존 4096 substep cap.
- 기존 관측 격자, Doppler 인자, \(D^2\eta\), 선·경계의 \(D^3S\), 단위변환.
- 기존 source fallback, Planck 함수, 보간·끝점 처리, 선별 ε helper.
- 기존 N/L 호출 순서와 R7·후속 dump·footer 흐름.

음·영·문턱 이하 τ의 새 방출을 추가하지 않는다. 그 영역의 비소비 census를 보존하되 N의 지원 영역 밖 방출이 물리적으로 0이라고 해석하지 않는다.

침적 setter 연결, NRB·인구·온도·opacity 생산자, 솔버·반복 수·난수·원장 변경은 제외한다. 직접 \(\eta_{\rm dep}=0\)은 gamma 물질 가열의 간접 효과가 없다는 뜻이 아니다.

### 2.3 필수 산출

상태별로 다음을 낸다.

1. 유효 committed capture와 manifest.
2. 기존 N/L에 대한 무변경 대조.
3. 같은 capture의 scalar N 재생 결과.
4. 일곱 성분의 전 출력 격자 탈출 스펙트럼과 성분합.
5. 대역 광도·성분 분율·E′−R′ 차이·수치 폐합 보고서.
6. 운전자 판정문과 astra 종료감사.

대역 적분은 기존 판독기의 파장 경계 보간·사다리꼴 적분 규약을 재사용한다.

| 대역 | 파장 [Å] | 용도 |
|---|---:|---|
| FUVb | 1500–2000 | E′ FUV 출력의 원천 |
| UV2500 | 2500–3000 | 기존 +0.61 dex 보고와 같은 대역 |
| OPT | 4000–6000 | 기존 색비의 정규화 |
| TOTAL | 저장 N의 전체 파장 범위 | 전체 폐합 |

UV2500을 단색 \(L_\lambda(2500)\)로 사용하지 않는다. 다른 대역은 저장 성분 스펙트럼에서 추후 읽을 수 있으므로 필수 판정량으로 늘리지 않는다.

## §3. 변경집합

새 생산 env는 **`LUMINA_DET_OBS_CAPTURE` 하나**다. 미설정과 `"0"`은 OFF, `"1"`만 ON이다. 다른 값은 캡처 설정 실패로 기록하고 캡처하지 않으며 모델 흐름은 유지한다.

OFF에서는 추가 배열 순회·할당·digest·파일 생성·캡처 로그를 수행하지 않는다. 기존 env 목록 변경에 따른 신원 차이는 별도 기록하되 기존 과학 출력의 바이트 보존을 요구한다. 기존 `LUMINA_DET_OBS_WRITER`와 `LUMINA_A210_SPRODUCER_CAPTURE`의 의미는 바꾸지 않는다.

캡처 위치는 런 work 아래 `det_obs_capture`로 고정하며 별도 경로 env를 만들지 않는다. 장애 주입은 시험 빌드의 adapter로만 제공한다.

### 변경집합 — DET-OBS-CAPTURE (표)

신규 심볼은 `det_obs_capture_` 접두를 사용한다. 기존 동명 계열과의 충돌은 피하며 기존 함수 이름의 참조를 계산식 수정 허가로 해석하지 않는다.

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c` | `det_obs_write`의 최소 hook; 신규 `det_obs_capture_*` | 동일 view 읽기, sparse 직렬화, 추가 digest, 거래·실패 처리. 기존 작성기 계산식·출력 형식 불변 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/env_universe.h` | `LUMINA_DET_OBS_CAPTURE` | 새 키 하나 등록, 목록·COUNT 정합 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/tests/det_obs_capture_selftest.c` | `det_obs_capture_*` fixture·I/O adapter | OFF/ON·저장 실패·reader 거부·성분 수송 직접시험 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/tools/det_obs_capture_replay.c` | `det_obs_capture_*` reader·scalar/vector replay | 오프라인 전용. 생산 바이너리의 수송 경로에 연결하지 않음 |
| `scripts/det_obs_capture_readout.py` | `det_obs_capture_*` staging·preflight·판독 | 기존 판독기·`runner_rev5` 재사용, 이번 계보·캡처·폐합·성분 분기 검사 |

### 변경집합 끝

`lumina_cmfgen.h`, `lumina_plasma.c`, `line_net_rate.c`, CUDA 커널, 원자 입력과 기존 판독기는 변경하지 않는다. 이 집합으로 구현할 수 없으면 필요한 정확한 변경을 계약 개정으로 넘긴다.

별도 실행 산출은 EC·RC의 `job.slurm`, 입력 이식 기록, B2 빌드 스크립트다. 러너는 기존 `scripts/det_obs_readout.py`의 `runner_rev5`로 생성하고 작업명·새 경로·바이너리 신원·새 capture export만 적용한다. 기존 자원 검사·모듈·실행·footer를 유지한다. 생성된 러너 전체와 허용 diff를 봉인한다.

기존 판독기의 T1/B1 고정 경로 검사를 우회하여 B2를 B1으로 표시하지 않는다. 새 판독기는 **B1→T2 patch→B2→EC/RC** 연결을 검사하고, 재사용 가능한 기존 파싱·적분·러너 검사를 호출한다.

## §4. 캡처 명세

### 4.1 위치·세대

최종 assemble의 기존 `DetObsCapture`를 그대로 사용한다. 마지막 radiation commit 뒤 `det_obs_write`에 전달되는 view가 완성되고 기존 view 검사가 통과한 시점, **N 행진 전·R7 물질 갱신 전**에 캡처한다.

기대값은 다음과 같다.

```text
material_generation = 7
rad_generation      = 7
iter                = 5
asm_serial          = 7
mode                = 1
src_nlte            = 1
NS                  = 50
NB                  = 1234
NL                  = 2588798
```

τ의 computed/required generation, atom/NLTE committed generation은 각각 별도 필드로 기록하고 선행 manifest·실행 증거와 대조한다. population7과 τ generation14처럼 서로 다른 카운터를 하나의 “gen7”로 합치지 않는다.

dump 함수가 기존 `DetObsView`의 실패 플래그·jump counter를 건드리지 않도록 읽기 전용 별도 context를 사용한다. ε 조회에 필요한 helper 호출도 생산 상태·전역 상태를 바꾸지 않는 것을 직접시험한다.

### 4.2 sparse 선 payload

스키마는 `DET_OBS_CAPTURE_V1`이다. 소비 쌍마다 다음 **33 byte packed record**를 저장한다.

```text
line_id : uint32 little-endian
shell_id: uint32 little-endian
tau     : IEEE754 binary64 little-endian
S_used  : IEEE754 binary64 little-endian
epsilon : IEEE754 binary64 little-endian
mapped  : uint8, 0 또는 1
```

native struct padding은 쓰지 않는다. 순서는 원래 `line_id` 증가, 그 안에서 `shell_id` 증가다. ID는 0-based다.

- mapped: `S_used`는 실제 fallback 선택 후 값, ε 레인은 0으로 저장하고 미사용으로 명시한다.
- unmapped: ε는 실제 helper의 raw 값, `S_used`는 실제 선택된 열 원천값이다.
- τ·ε는 무차원, \(S_{\rm used}\)는 공이동계 `erg s^-1 cm^-2 Hz^-1 sr^-1`이다.
- mask는 `S>0` 같은 추정이 아니라 기존 capture의 권위 mapped byte에서 읽는다.
- ε helper 실패, 비유한 필수 값, 범위 밖 ε는 캡처 부적격이며 보정·clamp하지 않는다.

`unmapped_line_source_positive=0`을 필수 gate로 둔다. 미매핑 `S_used`도 저장하지만, 양의 미매핑 raw source가 발견됐다고 기존 작성기의 `UNMAPPED_SOURCE_CONFLICT`를 해제하지 않는다. 그런 상태는 이 계약의 E′·R′ 재현이 아니므로 승인하지 않는다.

mapped B fallback도 기존 값 0과 대조한다. fallback 자체를 삭제하거나 raw S로 대체하지 않는다.

### 4.3 연속체·J·기하 payload

| 배열·값 | 저장 내용 |
|---|---|
| `chi_a`, `chi_e`, `eta_c`, `eta_dep` | 기존 capture의 각 `NS×NB` binary64 배열 |
| `J` | 실제 소비 coarse J, `NS×NB` binary64 |
| `Te`, `ne` | capture의 셸별 실제 값 |
| 격자 | 실제 `nu`, `dnu`, `nu_min`, `nu_max`, `d_log_nu`; edge 정의와 재구성식 |
| 선 목록 | 전체 `line_nu` binary64, 원래 ID·순서 |
| 기하·경계 | `r_inner`, `r_outer`, `time_explosion`, 실제 `inv_ct`, `T_inner`, `INNER_BB_SCALE` |

\(\chi\)는 `cm^-1`, \(\eta\)는 `erg s^-1 cm^-3 Hz^-1 sr^-1`, J는 강도 단위, 온도 K, 밀도 `cm^-3`, 길이 cm, 시간 s, 주파수 Hz다. frame은 `SHELL_COMOVING`이다.

여기서 저장 `eta_c`는 기존 capture의 **bf·ff+직접 침적 분자**다. `DetObsView.eta_c`처럼 전자산란까지 합친 배열과 이름을 혼동하지 않도록 manifest에 생산 필드와 의미를 함께 적는다. 순수 `chi_e`를 `cs->chi_es`로 대체하지 않는다.

### 4.4 coverage와 추가 live digest

소비 쌍의 중복 0·누락 0을 확인한다. sparse 파일의 정렬·ID 범위·각 행 predicate와 live 소비 순회의 count·mapped/unmapped·fallback census가 일치해야 한다. 이 검사는 파일 자체의 행수만 확인하는 것으로 대신하지 않는다.

기존 전체 τ/S FNV, view digest, 비유한·음·영·작은 양의 τ·주파수 제외 census를 유지한다. 비소비 영역을 sparse 파일에서 0으로 복원해 전체 FNV를 재현했다고 주장하지 않는다.

추가 digest는 shape·순서·원시 값 bit를 포함하여 다음 domain별로 계산한다.

- full authoritative mapped 배열.
- consumed 쌍 ID·τ·선택 후 S·raw ε.
- 분리된 \(\chi_a,\chi_e,\eta_c,\eta_{\rm dep}\).
- J·Te/ne·격자·선 주파수·기하·경계.

dump 직전·직후 각각 **live 입력에서 독립적으로** 계산하여 일치를 요구한다. 같은 직렬화 버퍼를 두 번 해시한 결과는 live 불변 검사로 인정하지 않는다. full mask digest와 sparse source digest의 domain을 구분한다.

파일 SHA256은 저장 내용의 무결성에 사용하며 live digest와 역할을 구분한다.

### 4.5 manifest와 원자적 기록

manifest는 다음을 포함한다.

- schema, dtype/endian/layout/shape, 단위·frame, ID·selection predicate·순서.
- 모든 세대·callsite·mode·src_nlte·수송 설정.
- 파일별 bytes·SHA256, domain별 live digest 전후값, 기존 FNV·census와 coverage 대조.
- T1/B1 기준선, T2 patch·소스, B2 바이너리·컴파일러·옵션·모듈, 실제 입력·exports·러너 신원.
- 새 N/L 파일의 bytes·SHA256 및 기존 writer manifest와의 연결.
- capture 상태·실패 단계·오류 코드.

거래 순서는 다음과 같다.

1. 같은 work 파일시스템에 `det_obs_capture.partial`을 exclusive 생성한다.
2. payload를 쓰고 반환값·short write·`fflush`·동기화·`fclose`를 검사한다.
3. 파일 재독출로 크기·해시·schema·coverage를 검사한다.
4. N/L이 기존 경로로 완료된 뒤 출력 신원을 연결한다.
5. 최종 manifest를 완성하고 모든 검사 성공 뒤 최종 디렉터리로 원자적 rename한다. 기존 최종 대상은 교체하지 않는다.
6. reader는 최종 경로의 `COMMITTED` manifest와 모든 해시·크기가 맞을 때만 수용한다.

`.partial`, payload만 존재하는 디렉터리, 최종 manifest가 없는 경로는 성공 자료가 아니다. 캡처 commit은 캡처 거래의 성공을 뜻하며, GPU 런 전체 유효성과 오프라인 폐합은 별도 확인한다.

### 4.6 저장 실패와 모델 지속

할당·open·쓰기·flush·close·검증·rename 실패는 **캡처 실패만** 기록한다. 캡처 소유 자원을 해제하고 기존 N/L, R7, 후속 출력과 footer를 계속한다. 모델 입력을 복구 명목으로 다시 쓰지 않는다. 기존 작성기가 독립적으로 실패한 경우 그 실패를 성공으로 바꾸지 않는다.

이 보장은 프로세스가 생존하고 기존 출력 경로가 사용 가능한 캡처 국소 실패에 한정한다. 파일시스템 전체 장애·프로세스 강제 종료·OOM에서도 모델 완주를 보장한다고 주장하지 않는다. 실패 기록 자체를 쓰지 못하면 stderr에 최소 상태를 남기고 reader는 commit 부재로 거부한다.

## §5. 실행·자원·무변경 gate

### 5.1 새 런

E0·R0 각각의 봉인 초기 입력에서 EC·RC를 시작한다. 최종 인구 warm start는 사용하지 않는다.

허용 차이는 새 경로 이식, B2 신원, 작업명, `LUMINA_DET_OBS_CAPTURE="1"`이다. `LUMINA_DET_OBS_WRITER="1"`, MGPU=2, 6회 반복, FIXED_TE와 E′/R′의 기존 NRB 차이를 유지한다. exports는 중복 키·부재·빈 문자열·`"0"`을 구분하여 전체 비교한다.

E′·R′는 같은 B2와 모듈을 사용한다. GPU는 h200x2를 사용하고 형식을 지정한다.

```text
--partition=h200
--nodes=1
--ntasks=1
--gres=gpu:H200:2
--cpus-per-task=16
--mem=64G
--time=30:00:00
--no-requeue
--exclude=syn102
```

기존 h200x2 러너의 GPU 종류·개수·서로 다른 UUID·GPU당 최소 메모리·CPU·MGPU 검사를 유지한다. H200이 비어 있지 않으면 대기하며 형식 없는 GPU 요청이나 다른 종류로 자동 대체하지 않는다.

수동 CPU 시험·판독·오프라인 수송은 **lageunha**에서 수행한다. 로그인 노드에서는 파일 조회만 한다. CPU 작업은 동시에 한 묶음, 최대 16 thread·64 GB로 제한하고 선행 작업 부하를 확인한다. 새 GPU 시험 런은 만들지 않는다.

### 5.2 비용

선행 H200×2 실측은 다음과 같다.

| 상태 | 경과 | GPU·시간 |
|---|---:|---:|
| E′ | 09:58:46 | 약 19.96 |
| R′ | 10:29:27 | 약 20.98 |
| 합계 | 20:28:13 | 약 40.94 |

이는 캡처 추가 전 실측이며 새 완주시간 보장이 아니다. 캡처 I/O 비용은 아직 미측정이다.

이번 실행 상한은 선행과 같은 **상태별 30시간·60 GPU·시간, 합계 120 GPU·시간**이다. 예상 비용과 상한을 구분한다. 각 상태 한 번이며 자동 재제출·재시작은 금지한다. 명확한 캡처·무변경 실패가 첫 런에서 확인되면 미제출 둘째 런은 보류하여 같은 결함에 비용을 반복 지출하지 않는다.

오프라인 대형 작업은 상태별 scalar/vector를 함께 계산하는 한 번의 전체 replay로 제한한다. 실패 원인 확인 뒤 필요한 범위만 다시 계산하며 격자 수렴 연구·다른 물리 재수송으로 확대하지 않는다.

### 5.3 저장량

33 byte 레코드와 선행 consumed count로 계산한 비압축 예상은 다음과 같다.

| 항목 | E′ | R′ |
|---|---:|---:|
| sparse 선 payload | 670,663,257 byte | 650,504,085 byte |
| 전체 선 주파수 | 20,710,384 byte | 20,710,384 byte |
| 연속체 4배열 | 1,974,400 byte | 1,974,400 byte |
| J | 493,600 byte | 493,600 byte |

기하·격자·manifest를 포함한 짝 payload 예상은 약 **1.37 GB**, committed payload 상한은 **짝 2 GB**로 둔다. 캡처 예상 크기가 상한을 넘으면 잘라 저장하지 않고 실패시킨다.

일반 런 출력·T2·B2·임시 파일은 이 2 GB에 포함되지 않는다. 제출 전 기존 E0/R0 산출 manifest의 파일 크기와 T1/B1 복제 대상 목록으로 총 필요량을 계산한다. 일반 출력과 source/build는 각각 기준선 크기의 두 배를 예약하고, capture는 committed와 부분 파일 보존을 합쳐 4 GB를 예약한다. 이 식으로 산출한 총량·가용 quota를 `resource_plan.json`에 봉인한다. 이는 저장 예약 규칙이지 실측 크기를 주장하는 숫자가 아니다.

범위를 제한한 파일 목록만 사용하며 전체 홈·GPFS 재귀 계측을 하지 않는다. 이전 실패 산출을 지워 공간을 확보하지 않는다.

### 5.4 기존 N/L과의 무변경

각 상태에서 다음 파일을 기준선과 바이트 비교한다.

```text
work/lumina_spectrum_det_obs.csv
work/lumina_spectrum_det_obs_legacy.csv
```

세대, 전체 τ/S FNV, 기존 view digest, census, jump count, source reconciliation과 χ_e 상한 판정도 대조한다. 벽시계·경로·바이너리 해시처럼 예상 변경되는 메타데이터는 동일성을 요구하지 않는다.

**바이트 불일치의 사전 고정 처분은 다음과 같다.**

1. 원본·새 파일·manifest를 보존하고 `BASELINE_MISMATCH`로 표시한다.
2. 격자·행수·유한성, 첫 불일치와 최대 절대차를 보고한다.
3. 상대차는 \(\max(|x|,|y|)>0\)인 셀에서만 계산하고 둘 다 0이면 0으로 둔다. 임의 floor는 넣지 않는다.
4. 입력·빌드·세대·τ/S·view 순으로 최초 신원 불일치를 좁힌다.
5. 차이가 작아도 **과거 E′·R′의 동일 view 분해로 승인하지 않는다.**

이 계약에는 수치 유사성으로 바이트 gate를 대신하는 허용치가 없다. 새로운 상태의 참고 분해는 구별하여 보존할 수 있으나 목표 완료로 판정하지 않는다. 기준 변경이나 추가 GPU 실행이 필요하면 증거를 가지고 계약 개정으로 넘긴다.

## §6. 오프라인 분해 하네스

### 6.1 입력과 scalar 재생

하네스는 committed capture만 읽는다. 원자 물리·인구·ε를 다시 계산하거나 diag5 후보 NPZ로 누락 값을 채우지 않는다.

같은 binary64 상수·Planck 함수·보간·공명 순서·경계 처방으로 scalar N을 먼저 재생한다. sparse 접근은 같은 소비 순서를 보존한다. 비소비 선을 새 0 배열로 만들어 다른 알고리듬에 넘기는 방식은 사용하지 않는다.

scalar 레인은 성분합으로 계산하지 않고 기존 N의 연산 결합 순서를 따른다.

- 빈별 \(\chi_c=\chi_e+\chi_a\).
- 빈별 \(\eta_{\rm total}=\eta_c+\chi_eJ\).
- 미매핑 선 원천은 기존 순서의 \(\epsilon S_{\rm used}+(1-\epsilon)J_b\).

같은 출력 함수·정밀도로 쓴 scalar CSV가 캡처 런의 저장 N과 바이트 일치해야 한다. 계산 호스트의 compiler/libm 차이로 실패하더라도 허용치를 사후 추가하지 않고 재현 환경 문제로 분류한다.

### 6.2 성분 벡터 행진

연속체 한 구간의 기존 scalar 갱신을

\[
I_{\rm out}=aI_{\rm in}+g\,e,\qquad
a=e^{-\tau},\quad
g=\frac{1-e^{-\tau}}{\tau}
\]

로 쓰면, 각 성분은 같은 \(a,g\)를 사용한다. 실제 구현은 기존 `expm1`·영 극한 계산 순서를 보존한다.

\[
\tau=q(\chi_a+\chi_e)\,ds,\qquad
e_k=D^2\eta_k\,ds.
\]

선 공명에서는 모든 성분을 같은 \(e^{-\tau_l}\)로 감쇠한 뒤 해당 성분에만 \(D^3S_{k,l}(1-e^{-\tau_l})\)를 더한다. 미매핑 열과 산란은 같은 공명에서 각각 주입한다.

전자산란은 반드시 **빈별 \(q_{eJ}[i]=\chi_e[i]J[i]\)를 먼저 만든 뒤 기존 보간 함수로 보간**한다. 보간된 \(\chi_e\)와 보간된 J를 곱하지 않는다. 연속체 다른 성분도 같은 생산 보간 규약을 적용한다. 선 산란의 J는 기존 선 bin 선택을 유지한다.

내경계가 강도를 교체하는 위치에서는 기존 scalar와 똑같이 벡터를 교체한다. 경계 성분에 경계 강도를 넣고 나머지는 0으로 시작한다. 이후 모든 성분이 같은 감쇠를 받는다.

### 6.3 폐합 기준

세 단계가 모두 필요하다.

1. **성분합 ↔ 같은 capture의 scalar N.**
2. **scalar N ↔ 캡처 런 저장 N:** CSV 바이트 동일.
3. **캡처 런 저장 N/L ↔ E0/R0 저장 N/L:** §5.4 바이트 동일.

첫 단계는 성분 분리·합산 때문에 생기는 부동소수 결합 순서 차이만 허용한다. 스펙트럼에 경험적인 상대오차 문턱을 붙이지 않는다.

수치 상한은 이진64 최근접 반올림을 전제로

\[
u=2^{-53},\quad a_{\min}=2^{-1074},\quad
\rho(x)=\frac{u|x|}{1-u}+a_{\min}
\]

를 사용하여 실제 덧셈·곱셈·나눗셈의 반올림 오차를 전파한다. 공통 감쇠·기하 계수는 scalar/vector가 같은 계산값을 공유한다. 원천의 분리·재결합, 보간, 각 구간·선 갱신, ray 합산·단위변환·대역 적분의 오차를 포함하고 상한 계산은 바깥쪽으로 반올림한다.

생산 결과를 보기 전에 구현의 오차 전파식과 연산 순서를 시험 증거에 봉인한다. 결과 편차에 맞춘 배수·floor·허용치 완화는 금지한다. 폐합은 모든 출력 점과 필수 대역에서 해당 상한 이내여야 한다. 상한의 유한성·전제 실패는 폐합 실패다.

이 상한은 물리 정확도나 격자 수렴 오차가 아니다. scalar 재생과 성분 벡터의 유한정밀도 분해를 검사한다. 성분 순위·과반 판단에도 이 상한을 전파하여 수치적으로 구별되지 않는 결과를 강제 분류하지 않는다.

## §7. 판정량과 분기

상태 \(X\), 성분 \(k\), 대역 \(b\)의 광도를 \(L_{Xkb}\)라 둔다.

\[
L_{Xb}=\sum_k L_{Xkb},\qquad
f_{Xkb}=L_{Xkb}/L_{Xb}.
\]

총광도가 양수이고 수치적으로 구별될 때만 분율을 정의한다.

### 7.1 E′ 출력의 지배 성분

FUVb·UV2500 각각에 대해 모든 성분의 광도와 분율을 보고한다.

- 한 성분의 분율 하한이 \(1/2\)를 넘으면 **그 성분 과반 지배**.
- 과반 성분은 없으나 한 성분의 하한이 다른 모든 성분 상한보다 크면 **혼합·최대 성분 식별**.
- 순위 또는 과반 경계가 수치 구간과 겹치면 **혼합·수치 경계**.

미매핑 열·산란을 합치거나 bf·ff를 선에 합쳐 사후에 “지배” 범주를 만들지 않는다. 묶음 합은 참고값으로만 표시한다.

FUVb에는 공개 CMFGEN 직접 기준이 없으므로 이 판정은 **큰 E′ FUVb 출력의 원천**을 뜻한다. UV2500의 기존 공개 대비 과잉도 총 색비와 나란히 보고하되, 공개 스펙트럼의 원천 분해가 없으므로 잔차를 특정 성분에 임의 배분하지 않는다.

### 7.2 E′−R′ 차이의 귀속

절대광도 차이는

\[
D_{kb}=L_{E'kb}-L_{R'kb},\qquad D_b=\sum_kD_{kb}
\]

로 정의한다. 부호를 유지하고

\[
P_b=\sum_k\max(D_{kb},0),\qquad
M_b=\sum_k\max(-D_{kb},0)
\]

를 함께 보고하여 상쇄를 드러낸다.

\(D_b>0\)가 수치적으로 확정되면 양의 차이를 만드는 성분의 \(D_{kb}/P_b\)에 §7.1의 과반·최대 규칙을 적용한다. \(D_b<0\)이면 음의 차이 크기와 \(M_b\)에 같은 규칙을 적용한다. 총차이 구간이 0을 포함하면 **순차이 부호 미확정**으로 둔다. 작은 \(D_b\)로 나눈 큰 기여율을 지배 근거로 쓰지 않는다.

기존 OPT 정규화 응답과 연결하기 위해 다음도 계산한다.

\[
Q_{kb}=
\frac{L_{E'kb}}{L_{E',OPT}}
-\frac{L_{R'kb}}{L_{R',OPT}}.
\]

그 합은 두 상태의 총 색비 차이에 폐합한다. 기존 dex 색 변화 자체를 성분 dex들의 합으로 분해하지 않는다.

### 7.3 배선·물리·알고리듬 귀속

| 증거 | 허용 귀속 |
|---|---|
| committed payload가 live 소비 값·세대·분류와 다름 | 캡처 또는 연결 배선 결함 |
| 올바른 payload에서 scalar가 저장 N을 재현하지 못함 | replay 구현·환경·알고리듬 불일치; 원인 조사 전 생산 writer 결함으로 단정하지 않음 |
| scalar는 일치하나 성분합이 수치 상한 밖 | 분해 하네스 결함 |
| 원천 소비식·변환·공명 소유가 등록 수송식과 다름 | 위치·재현 반례가 있는 알고리듬 결함 |
| 모든 폐합 통과 후 특정 성분이 큼 | 고정 view에서 그 원천의 큰 탈출 기여 |
| 큰 성분과 고정 T·원장 비폐합·J 등이 연관됨 | 물리 상태·근사에 관한 후속 조사 후보 |
| 직접 \(\eta_{\rm dep}=0\) | 기존 DET setter 우회와 일치하는 배선 사실; 비영 FUV 직접 원천에서 제외 |

큰 \(S/B\), 큰 산란 기여, \(L>L_{\rm inner}\)만으로 물리 오류나 에너지 보존 위반을 확정하지 않는다. 이 단은 원천 귀속을 완료할 수 있지만 물리 모형의 최종 타당성까지 자동으로 확정하지 않는다.

## §8. 최소 게이트와 음성대조

| 묶음 | 막는 실제 실패 | 통과 증거 |
|---|---|---|
| G-ID | 잘못된 기준선·입력·patch·러너·자원 | B1→T2→B2 계보, exports 허용 diff, 실행 신원·비용 |
| G-CAP | 누락·세대 혼합·잘못된 원천·live 변경·부분 저장 승인 | schema·coverage·digest·거래 검사, OFF/ON·저장 실패 시험, 기존 N/L 바이트 일치 |
| G-CLOSE | 다른 수송·보간·성분 분리·판독 | scalar 바이트 재생, vector 폐합, 적분·분기 시험 |

음성대조는 작은 CPU fixture와 저장 fixture 복사본에서 수행한다. 생산 E0/R0나 실제 캡처를 변조하지 않는다.

| 시험 | 주입 결함 | 필수 결과 |
|---|---|---|
| 신원·reader | 다른 세대, 파일 bit 변조, sparse 쌍 누락·중복 | 지정 사유로 거부 |
| 원천 분류 | 미매핑 양의 raw source, ε helper 실패 | 캡처 부적격; 기존 writer gate 유지 |
| live 불변 | 시험 adapter에서 J 또는 mask 한 값 변조 | 추가 digest 실패 |
| 저장 지속 | exclusive/open 실패, 중간 short write, flush 실패, close 실패, commit rename 실패 | COMMITTED 없음; reader 거부; 모델 후속 경로 지속 |
| scalar/vector | 빈별 곱과 보간 후 곱이 다른 χ_e/J fixture, 한 성분 누락·공명 중복 | 폐합 또는 독립 기대값 검사 실패 |
| 판독 | 과반 경계·동률·상쇄·0 총차이·입력 결손 | §7의 등록 분기 |

저장 실패 시험은 각각 OFF·정상 ON과 같은 입력에서 비교한다. N/L 바이트, 세대·기존 FNV·live digest·census·jump가 같아야 한다. 기존 후속 R7 호출 지점까지 실제 제어 흐름이 이어지는 adapter 시험을 포함한다. R7 물리 계산을 stub으로 대체한 경우 전체 GPU 종단 시험이라고 표시하지 않는다.

정상 scalar/vector fixture는 경계만·연속체만·매핑 선·미매핑 ε=0/1/혼합을 작은 같은 시험 묶음에서 다룬다. 기존 정적 해석해 허용오차를 재사용하며, 생산 스펙트럼 폐합에 그 fixture 허용오차를 전용하지 않는다.

코드 감사는 등록 patch·직접시험·위 세 묶음만 대상으로 한다. 차단 사항은 항목별 수정 신원·시험·검토 처분을 연결하며 과거 감사 PASS를 새 코드에 승계하지 않는다.

## §9. 완료·실패 처분과 재현

### 완료 기준

다음이 모두 충족되면 운전자는 `PASS`를 판정할 수 있다.

- EC·RC가 각각 유효하게 완료되고 기존 E0·R0 N/L과 바이트 일치.
- 양쪽 committed capture의 신원·coverage·무변경 검증 통과.
- scalar 재생과 일곱 성분 폐합 통과.
- FUVb·UV2500의 성분 광도·분율·E′−R′ 및 색비 차이 귀속 보고.
- 직접 \(\eta_{\rm dep}=0\) 확인.
- 코드 감사 차단 사항과 집행 이탈의 처분 기록.
- astra가 판정문·목표 달성·주장 상한을 종료감사.

과반 성분이 없다는 결과도 유효한 완료 결과다. 폐합 실패·기준선 불일치를 “혼합 원천”으로 처리하지 않는다. 캡처 확보만으로 완료하지 않는다.

코드·하네스의 범위 내 수정은 `REWORK`, 추가 GPU 런·물리 변경·새 비용 결정이 필요한 경우는 해당 미달 목표를 적어 `BLOCKED`로 기록한다. 기존 산출은 보존한다.

### 구현할 재현 인터페이스

다음은 운전자가 구현할 CLI 계약이며 아직 존재·통과한다고 주장하지 않는다. 모든 writer-capable 명령은 새 출력 디렉터리만 수용한다.

```bash
python3 -B scripts/det_obs_capture_readout.py selftest \
  --baseline-source "$T1" --source "$T2" \
  --out "$C/stage1"

python3 -B scripts/det_obs_capture_readout.py preflight \
  --baseline-e "$E0" --baseline-r "$R0" \
  --source "$T2" --build "$B2" \
  --run-e "$EC" --run-r "$RC" \
  --out "$C/evidence/preflight"

python3 -B scripts/det_obs_capture_readout.py replay \
  --run "$EC" --baseline "$E0" \
  --out "$C/readout/E"

python3 -B scripts/det_obs_capture_readout.py replay \
  --run "$RC" --baseline "$R0" \
  --out "$C/readout/R"

python3 -B scripts/det_obs_capture_readout.py paired \
  --e "$C/readout/E" --r "$C/readout/R" \
  --out "$C/readout/paired"
```

`preflight`는 아직 없는 실행 후 산출을 요구하지 않는다. 실행 후 검사는 실제 Slurm 상태·restart 수·footer·writer 상태·캡처·출력까지 연결한다.

운전자는 lageunha에서 기존 `check_prereg_preflight.py`, `check_gate_referents.py`, `check_doc_canonical.py`, `check_dispatch_preflight.py`를 현행 CLI로 실행하고 명령·rc·해시를 기록한다. 기계 게이트 결과를 Fable 사전감사나 gpt-6-sol 코드 감사로 표시하지 않는다.

## §10. 집행 기록·분장 장부

빈칸은 미집행이다. 뒤에 기재하되 계약 수용 기준을 집행 기록으로 바꾸지 않는다.

| 집행 항목 | 담당 | 상태·경로·신원·결과 |
|---|---|---|
| user 실행 결정 | user | 2026-10-06 「진행」, §0 경로 |
| 본문 기안 | astra | 본 문서; read-only 기안 |
| 정본 파일화·기준선 봉인 | 운전자 | 미집행 |
| 기계 프리플라이트 | 운전자 | 미집행 |
| Q-GOAL·Q-LEAN 사전감사 | Fable 또는 기록된 대행 | 미집행 |
| T2 구현·직접시험·음성대조 | 운전자 | 미집행 |
| 코드 감사·차단 처분 | gpt-6-sol / 운전자 | 미집행 |
| B2 빌드·러너·총 저장 예약 | 운전자 | 미집행 |
| E′ 캡처 런·Slurm 비용 | 운전자 | 미집행 |
| R′ 캡처 런·Slurm 비용 | 운전자 | 미집행 |
| 기존 N/L 무변경·capture 검증 | 운전자 | 미집행 |
| scalar/vector 폐합·짝 판독 | 운전자 | 미집행 |
| 목표 달성 판정 | 운전자 | 미집행; 구현자 겸임 명시 |
| 종료감사 | astra | 미집행; 계약 기안자 이해관계 명시 |
| 이탈·미해결·후속 결정 | 실제 담당 | 발생 시 기록 |

## §11. 기계 선언

아래 분기는 대역별 판독 상태를 검사한다. `v`는 판독기가 실제 수치 구간 검사 후 만드는 단일 코드다.

- `0`: 과반 지배.
- `1`: 혼합·최대 성분 식별.
- `2`: 혼합·수치 경계 또는 순차이 부호 미확정.
- 그 밖·결손: 부적격.

실제 성분 번호·광도·비율·오차 구간은 판독 산출에 별도로 기록한다. 이 선언만으로 수송·과반 계산을 검증했다고 주장하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-OBS-CAPTURE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/(?:src|tests|tools)/[A-Za-z0-9_/.]+\\.(?:c|h)|scripts/det_obs_capture_readout\\.py)",
    "symbol": "det_obs_capture_",
    "roots": [
      "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/env_universe.h",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/tests/det_obs_capture_selftest.c",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/tools/det_obs_capture_replay.c",
      "scripts/det_obs_capture_readout.py"
    ],
    "expected_extra": [
      "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/env_universe.h",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/tests/det_obs_capture_selftest.c",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/tools/det_obs_capture_replay.c",
      "scripts/det_obs_capture_readout.py"
    ]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [9.0, 9.0]],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2) for x in v)",
      "majority": "sum(1 for x in v if x == 0)",
      "largest": "sum(1 for x in v if x == 1)"
    },
    "rules": [
      {"name": "X", "predicate": "not valid"},
      {"name": "MAJORITY", "predicate": "valid and majority == 1"},
      {"name": "MIXED_LARGEST", "predicate": "valid and largest == 1"},
      {"name": "UNRESOLVED", "predicate": "valid and majority == 0 and largest == 0"}
    ],
    "adversarial_fixtures": [
      {"name": "majority", "mix": [[0,1]]},
      {"name": "mixed-largest", "mix": [[1,1]]},
      {"name": "boundary-or-zero-difference", "mix": [[2,1]]},
      {"name": "invalid", "mix": [[9,1]]},
      {"name": "missing", "mix": []},
      {"name": "extra", "mix": [[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path": "CLAUDE.md"},
    {"path": "docs/PREREG_DET_OBS_WRITER_2026-10-02.md"},
    {"path": "docs/VERDICT_DET_OBS_WRITER_2026-10-02.md"},
    {"path": "scripts/check_prereg_preflight.py"},
    {"path": "scripts/check_gate_referents.py"},
    {"path": "scripts/check_doc_canonical.py"},
    {
      "path": "scripts/det_obs_readout.py",
      "flags_existing": [
        "def runner_rev5(",
        "def band_table(",
        "gpu:H200:2",
        "cuda_build2"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/src/lumina_cmfgen.c",
      "flags_existing": [
        "det_obs_consumed",
        "det_obs_line_source",
        "det_obs_cont_build",
        "det_obs_view_digest",
        "UNMAPPED_SOURCE_CONFLICT"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag5_20261006/P2_DIAGNOSIS.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag5_20261006/astra_audit_p2.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag5_20261006/DIAGNOSIS_r2.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag5_20261006/astra_audit_r2.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag5_20261006/LOOP_CLOSED.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/evidence/USER_DECISION_2026-10-06.txt"
    }
  ]
}
```

