## 고정질문 1/2/3

1. [실측] `evidence/rows.txt`에는 1,282행 모두 `phase=REQUESTED_TE`, `ion=3`이고 `target_ion=2` 음성대조 행은 없다 (`evidence/rows.txt:1-1282`). 초안이 이를 §3에서 미실행으로 공개한 것은 정직하다 (`draft.md:33-37`).

   다만 이는 전체 Stage PASS가 아니다. `V1-V2 판정을 바꿀 수 없다`는 문장은 근거가 없다. `docs/RUNG_CLOSURE_PROTOCOL.md` 자체도 스테이징에 없어 프로토콜상 PASS 요건은 판정 불가하다. 독립 캡처 PASS는 캡처에만 한정해야 한다 (`draft.md:15`, `src/lumina_cmfgen_capture.c:6521-6528`).

2. [실측] `python3 evidence/repro.py evidence/rows.txt` 결과는 실측 3, 5, 6의 수치와 일치한다 (`evidence/repro.py:8-17`, `evidence/numbers.txt:1-4`). 식 자체도 일치한다.

3. [판정] 단정 과다 항목은 있다. 특히 `완전 단조`, `항등`, `LTE형 실증`, V2 `[확정]`, V4의 인과적 재정의가 그렇다.

## A 산술

[실측] rank corr의 정확한 값은 `−0.9999998917890933`이다. `−1.000`은 3자리 반올림으로는 가능하지만, 초안의 “1,282행 완전 단조”는 틀리다. 행을 주파수순으로 재검사하면 인접 비단조 구간이 15개 있다 (`evidence/rows.txt:1-1282`, 계산식 `evidence/repro.py:12-17`).

[실측] 추가 재계산한 Wien 적합은 계수 `0.60636`, `T=21,988.66 K`, ln 잔차 rms `0.03905`로 초안 수치와 맞는다. 반면 `B(19kK)/B(10kK)=10⁴~10⁶`은 전체 444–1,884 Å 범위에는 맞지 않는다. 실제 범위는 약 `37.8~4.55×10⁶`이다.

## B S_probe=B 항등

[실측] 표시된 값은 모두 `1.0000`이지만 실제 `S_probe/B(T_req)` 범위는 `0.9999914~0.9999940`, `stdev(log10)=2.28×10⁻⁷`이다. 즉 표시 정밀도에서의 근사이지 항등은 아니다.

[실측] 소스상 `S_probe`는 독립 측정값이 아니라 기존 `source_function`을 그대로 복사한 값이다 (`src/lumina_plasma_saturation_add.c:14008-14012, 14060-14063`). `REQUESTED_TE`라는 표지도 trial 온도의 균일성을 뜻할 뿐 LTE를 증명하지 않는다 (`src/lumina_plasma_phase.c:13619-13635`).

[판정] NLTE 물질이 우연히 또는 계약상 `S→B`로 수렴했을 가능성은 배제되지 않는다. 따라서 “LTE형의 실증”은 [추정]으로 낮춰야 한다. V1의 핵심인 “서로 다른 온도 잣대를 비교했다”는 진단은 유지 가능하지만, V2는 이 구분 때문에 확정할 수 없다.

## C V2 강도

[실측] 재현된 값은 `Jbar/[βJ_cont+(1−β)B(T_e)]`: 중앙 `0.9992`, q10 `0.7131`, q90 `1.0000`이다 (`evidence/repro.py:11-15`).

[판정] 이는 생산자의 실제 `S`가 아니라 `B(T_e)` 대용값과의 일치다. 생산자 `S`를 직접 캡처한 근거는 없다. 따라서 “생산 Jbar 폐합 만족” 및 `[확정]`은 과대주장이다. 적절한 표현은 “`B(T_e)` 대용 폐합과 대체로 일치하나, q10 꼬리와 실제 생산자 S는 미확정”이다.

## D 독립성 한정

[실측] Sobolev 모드에서는 두 line-deposit 경로가 모두 `!sobolev_operator` 조건으로 차단된다 (`src/lumina_cmfgen_deposit_gate.c:5524, 5592-5593`). 캡처는 line deposit을 뺀 두 번째 solve다 (`src/lumina_cmfgen_capture.c:6378-6380`).

[판정] bit-identical은 독립 검증의 성공이라기보다 두 경로가 같은 line-free/common-mode를 탔다는 증거다. 캡처는 선 폐합 독립성의 구현 점검에는 가치가 있지만, 연속체 물리의 독립적 검증에는 거의 가치가 없다. 초안의 범위 한정은 방향은 맞지만, “독립 캡처 PASS”와 “independent J_cont”라는 명칭은 더 강등해야 한다.

또한 초안의 소스 경로 `src/lumina_cmfgen.c`는 스테이징의 실제 파일명과 불일치한다. 정확한 근거는 `src/lumina_cmfgen_deposit_gate.c`다 (`draft.md:17`).

## E V4 자격

[실측] 데이터는 `Jbar`와 trial-material source가 서로 다른 상태의 조합과 양립한다. 그러나 `Jbar`가 실제 NO_BRACKET 경로에서 frozen 되었는지, 그 결합이 선 순냉각 폭주를 일으켰는지는 이 스테이징에서 직접 검증되지 않는다. 행은 진단 전용이고 (`evidence/rows.txt:1-1282`), 소스 발췌도 caller/R7 인과경로를 보여주지 않는다 (`src/lumina_plasma_saturation_add.c:13951-14012`).

[판정] “가능한 메커니즘 후보”로는 적을 자격이 있다. 다만 “물리 원인을 재정의했다”, “J 생산 결함이 아니다”는 표현은 추측을 넘는다. [추정] 후보로만 유지해야 한다.

## 총평 — 초안 채택/수정/기각 + 다음 단 하나

**수정. 현 상태 채택 불가.** V1의 잣대 불일치 진단과 주요 비율 계산은 유효하지만, V2 확정, LTE 실증, 독립 캡처의 증거 가치, 완전 단조, V4 인과 주장이 증거보다 앞선다.

다음 단 하나: **생산자 상태에서 실제 line-material `S`를 직접 캡처해 `Jbar/[βJ_cont+(1−β)S_producer]`를 재계산하라.**