# Fable 판정 — MC-EVT OUT_OF_GRID·NE2·ME2

날짜: 2026-08-08

## 호출 provenance

- CLI: Claude Code `2.1.226`
- 요청 모델: `--model fable`
- 응답 canonical model: `claude-fable-5`
- session id: `da83f3f6-87da-450d-936b-3787573d51d9`
- 입력: `docs/QUERY_FABLE_MC_EVT_2026-08-08.md` 원문을 stdin으로 전달
- tools: disabled
- CLI status: success, `stop_reason=end_turn`, permission denial 0

## Fable 원문 응답

```text
Q1: 1 — 근거
Q2: 통합주입필수 — 근거
Q3: 원사전등록유지 — 근거
MC-EVT 현재 상태: BLOCKED
```

### Q1 — 판정 1: 정당한 BF exact-zero, 단 조건부

Fable이 인용한 실측 사실: 격자 `[1.5e14, 3.0e16) Hz`, 차단 11건 전부 하단 밖,
구 CPU/GPU 모두 격자 밖에서 수치 0 반환, 사전등록 NE3는 event ON에서
`t03_blocks=0`을 요구한다.

Fable 원문:

> 사건측도 격자 하단 밖 주파수에서 정의되지 않는 것은 BF 채널뿐이다. 전자산란과 선
> 수송은 해당 주파수에서 물리적으로 완전히 정의되며 BF 격자를 필요로 하지 않는다.
> BF 미정의를 이유로 패킷 전체를 재흡수/차단하는 것은 에너지를 비물리적으로 제거하는
> 수송 개입이다. 올바른 조치는 OUT_OF_GRID 상태 표면화·카운터는 유지하고,
> `chi_bf=0`으로 수송을 계속하며 차단 사유에서만 제외하는 것이다.

조건:

> `NLTE_NU_MIN=1.5e14 Hz` 아래에 모델 원자데이터의 bf edge가 존재하지 않음은 제시된
> 증거로 확인되지 않았다. edge가 하단 밖에 존재하면 판정은 3(격자 확장 의무)으로
> 격상되어야 한다. 이 확인 전까지 판정 1은 조건부다.

### Q2 — 통합주입필수

Fable 원문:

> 공유 raw classifier의 GPU NEGATIVE fixture와 네 소비자 정적 음성대조는 분류기와 가드
> 존재의 증거다. 그러나 실제 커널 실행 경로에서 가드가 발화했다는 동적 실측이 없다.
> publication을 시험에서만 우회하여 실제 transport kernel 경로에 음수를 주입하고
> 차단이 발화함을 관측하는 통합 음성대조가 NE2 자격의 필요조건이다.

### Q3 — 원사전등록유지

Fable 원문:

> 정본 ME2는 GPU event ON/OFF 스펙트럼 차이다. 배열 차이를
> `[E-AUX-MEASURE-ARRAY]`로 강등하고 ME2 이름을 복원한 조치는 옳다. 배열 차이 0은
> 필요조건일 뿐이며, 수송 분기·상태 정책·RNG 소비 순서가 달라질 수 있으므로 영향 0은
> end-to-end ON/OFF 스펙트럼 비교로만 판정할 수 있다.

### 현재 상태

Fable의 최종 판정은 `BLOCKED`다.

- NE3: 현재 `t03_blocks=11`, 조건부 Q1 조치와 저주파 BF edge census 미완.
- NE2: 실제 GPU transport 통합 음수 주입 미실행.
- ME2: GPU ON/OFF end-to-end spectrum 미측정.

## 즉시 효력

- OUT_OF_GRID를 곧바로 exact zero로 구현하지 않는다. 먼저 `NLTE_NU_MIN` 아래 BF edge
  census를 수행한다.
- edge 0개가 입증되면 OUT_OF_GRID는 이름·카운터를 보존한 BF exact zero이고 packet
  transport는 계속한다.
- edge가 하나라도 있으면 SH-GRID를 다시 열고 하단 격자 확장을 검토한다.
- NE2는 실제 GPU transport kernel 통합 주입 전까지 미폐합.
- ME2는 원 사전등록의 GPU ON/OFF spectrum 비교를 유지한다.

