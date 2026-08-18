# MC-EVT 사전감리 — 2026-08-08

성격: **Codex read-only 판정 감리의 사전 의견서**.  운전석 판정 초안이 아직 없으므로
이 문서는 판정문을 대신하지 않으며, 코드 적용분을 폐합하지 않는다.

대조 대상:

- 사전등록: `docs/RUNG_EVENT_MEASURE_LANE_AGREEMENT.md`
- 적용 코드: 현재 미커밋 `Makefile`, `src/*`, `scripts/check_event_measure_access.py`
- 실행 증거: `validation/evt/evt_mc.log`
- 당시 발주/제출 원문: `/tmp/claude-10396/codex_evt/{TASK_EVT.md,OUT_EVT.md}`
- 빌드 원문: `/tmp/claude-10396/{evt2_cpu.log,evt2_gpu.log}`

---

## 1. 총평

**[판정] 현재 자격은 `부분 증거 확보`다.  MC-EVT PASS 또는 폐합 자격은 없다.**

핵심 구현 방향은 맞다.  CPU T03, CPU A208, GPU T03, GPU virtual-packet이
상태를 내는 사건측도 접근자를 통하고, legacy 경로도 이름과 provenance를 얻었다.
CPU MC는 수리 전 800/800 차단에서 수리 후 11/200 차단으로 살아났고 A2-10 PRE까지
도달했다.

그러나 사전등록 E1~E4를 기준으로 보면 미실행 항목이 남아 있고, 등록된 ME2와 코드가
출력한 `[E-ME2]`는 서로 다른 측정이다.  따라서 인수인계서의 "게이트 완료"는 증거가
요구 강도에 도달했다는 뜻으로 사용할 수 없다.

---

## 2. 사전등록과 발주문 사이의 ME2 변조

### [실측]

저장소 사전등록 §6의 ME2는 다음이다.

> GPU 생산 구성에서 event ON vs OFF 스펙트럼 차이

반면 당시 `TASK_EVT.md`의 "측정 계측"은 ME2를 다음으로 바꾸었다.

> 같은 구성에서 두 생산자 측도 배열의 셸별 최대·중앙 상대차

사전등록 파일, 당시 Codex에 제공된 `ref/` 사본, 감사용 사본은 모두 SHA-256
`1706ed9b4e86d1c867d69e4cf1f3ae82b41cb15721ee1b6f182c0b8c534b3f51`로 동일하다.
즉 사전등록 파일이 바뀐 것이 아니라 **작업문이 등록된 측정을 다른 측정으로 재서술**했다.
Codex 패치는 작업문을 따라 배열 차이를 `[E-ME2]`로 출력했다.

### [판정]

배열 차이는 유용한 보조 계측이지만 end-to-end 스펙트럼 차이를 대체하지 않는다.
전자는 입력 측도의 차이이고, 후자는 수송·상호작용·재방출을 지난 관측량의 차이다.
비선형 수송에서 배열 차이 0이 스펙트럼 parity를 자동으로 증명하지도 않는다.

현재 코드에서는 이 오인을 막기 위해 배열 출력을 `[E-AUX-MEASURE-ARRAY]`로 강등했다.
등록된 **ME2 이름은 GPU ON/OFF 스펙트럼 비교용으로 비워 두었다.**

---

## 3. 게이트별 증거 감사

| 게이트 | 현재 증거 | 판정 |
|---|---|---|
| NE1 수리 전 3종 불일치 | 코드와 800/800 CPU 차단 실측. GPU의 silent fallback은 구 코드로 확인 | 수리 전 음성대조는 성립 |
| NE1 수리 후 세 소비자 동일 판정 | CPU 런만 존재. GPU 수리 후 소비 로그 없음 | **미실행** |
| NE2 음수 legacy 셸·빈 | ME1은 기준 덱에서 음수 0/50000. 주입 런 없음 | **미실행** |
| NE3 event ON | 현 로그 producer는 `LEGACY_ARGMAX`; T03 차단 11 | **미실행** |
| NE4 접근자 우회 | 정적 PASS는 존재했으나 기존에는 주입 음성대조가 없었음 | 검사기 보강 후 로컬 음성대조 PASS, 운전석 재실행 필요 |
| E2 CPU/GPU 배열 | 런타임 `memcmp` 코드는 있음. `[E-E2] ... match=1` GPU 런 로그 없음 | **미실행** |
| E3 ME1~ME3 | ME1 존재. 등록 ME2 없음. ME3는 event OFF 한 구성만 존재 | **부분** |
| E4 event ON GPU byte parity | 비교 대상 두 런·산출 해시 없음 | **미실행** |
| CPU/GPU 빌드 | 두 빌드 로그와 바이너리는 존재 | 성공 관측. 정본 경로에 timestamp/sha/rc 묶음 없음 |

따라서 고정 감리 질문 1(미실행 게이트)과 2(증거 강도)에서 모두 PASS 자격이 없다.

---

## 4. 새로 드러난 NE3 선행 문제

### [실측]

`evt_mc.log`에서 수리 후 200패킷 중 11건이
`EVENT_MEASURE_OUT_OF_GRID`로 차단됐다.  복사장 빈 채움률은 0.1075였다.
사전등록 NE3는 event ON에서 `t03_blocks=0`을 요구한다.

### [판정]

producer를 SPONTANEOUS로 바꾸어도 주파수 격자 경계는 같으므로, event ON만으로
11건이 사라진다고 사전 단정할 수 없다.  반드시 event ON 런으로 측정해야 한다.

OUT_OF_GRID를 "정당한 BF exact-zero"로 읽을지 "계약 위반"으로 차단할지는 물리·계약
판단이다.  현재 패치는 차단을 택했다.  이 판단을 몰래 바꾸어 NE3를 통과시키지 않는다.
event ON에서도 재현되면 Fable 판단 또는 사전등록 보정이 선행되어야 한다.

---

## 5. 이번 사전감리에서 한 보정

1. 배열 차이 로그를 `[E-ME2]`에서 `[E-AUX-MEASURE-ARRAY]`로 변경했다.
2. NE4 검사기가 네 소비자 각각에 직접 배열 접근을 **메모리 안에서 주입**하고 모두
   검출하는 음성대조를 추가했다.  작업트리는 주입 과정에서 수정되지 않는다.
3. GPU opacity owner가 provenance를 발행하지 않고 상위 래퍼가 반환 복사본에 사후
   대입하던 소유권 결함을 고쳤다.  이제 canonical owner가 배열과 provenance를 함께
   발행하고 상위 래퍼는 둘의 일치만 검사한다.
4. 등록 ME2와 E4가 같은 CSV를 서로 다른 강도로 판정하도록
   `scripts/compare_event_measure_spectra.py`를 추가했다. ME2는 차이를 측정만 하며,
   E4는 전체 파일 byte parity가 아니면 실패한다.
5. CPU/CUDA가 같은 `bf_event_measure_lookup_raw()` 분류기를 쓰도록 중복 구현을
   합쳤고, `tests/mc_evt_access_selftest.cu`가 GPU에서 OK·NEGATIVE·OUT_OF_GRID·
   UNAVAILABLE 네 상태를 직접 만든다. 테스트 오브젝트 컴파일은 성공했으며 로그인
   노드 GPU 실행은 규칙상 하지 않았다.
6. 로컬 정적 확인 결과:

```text
[E-NE4][PASS] all CPU/GPU event consumers use the status accessor, block non-OK status, and retain no chi_bf event fallback
[E-E2][STATIC][PASS] canonical GPU opacity owner publishes event-measure provenance
[E-NE4][NEGATIVE-CONTROL][PASS] injections=4 detected=4
[E-SPECTRUM-COMPARE][SELFTEST][PASS]
```

이 로컬 확인은 코드 저자의 자가검사이며, 개정11의 운전석 검사·실행을 대체하지 않는다.

---

## 6. 폐합 전에 필요한 최소 증거

1. event ON CPU MC 런: producer=SPONTANEOUS, ME3, T03 사유별 계수.
2. event OFF/ON GPU 생산 런: 같은 덱·패킷·seed·바이너리에서 E2 로그와 스펙트럼 산출.
3. 등록 ME2: ON/OFF 스펙트럼 byte 비교와 수치 차이 지도.
4. NE2 음수 주입: 주입 실재 확인 후 CPU/GPU 소비자의 이름 있는 차단.
5. E4: pre-MC-EVT와 post-MC-EVT 바이너리를 모두 event ON으로 돌린 byte parity.
6. 두 타깃 강제 재빌드의 rc, timestamp, SHA-256.
7. 위 자료를 인용한 운전석 판정 초안과 그 뒤 Codex 감리.

이 중 하나라도 미실행이면 판정문은 `부분 폐합 — 미실행: ...`로 적어야 한다.
