# Fable 질의 — MC-EVT의 OUT_OF_GRID 물리와 게이트 자격

작성 2026-08-08.  요청: 아래 사실·선택지 밖을 추측으로 채우지 말고 물리 판정을 내려 달라.

응답 수신: `docs/FABLE_VERDICT_MC_EVT_2026-08-08.md` — `MC-EVT 현재 상태=BLOCKED`.

## 배경

MC-EVT는 bound-free packet event measure를 한 정의로 만들고 CPU T03·CPU A208·GPU
T03이 같은 상태 정책을 쓰게 하는 MC 전용 단이다.  현재 패치는
`OK / UNAVAILABLE / NEGATIVE / OUT_OF_GRID`를 구분하고, 모든 non-OK를 차단한다.

기준 덱 CPU MC 수리 후 실측:

```text
packets=200
T03 blocks=11
reason=EVENT_MEASURE_OUT_OF_GRID
producer=EVENT_MEASURE_LEGACY_ARGMAX
radiation_bins_filled=5375/50000
```

사건측도 격자는 `[NLTE_NU_MIN, NLTE_NU_MAX) = [1.5e14, 3.0e16) Hz`다.  차단된
패킷에는 `7.884542e13`, `1.032178e14`, `1.497605e14 Hz`처럼 하단 밖 주파수가 있다.
구 `bf_get_chi()`와 구 GPU lookup은 격자 밖에서 수치 0을 반환했다.  새 접근자는 이를
`OUT_OF_GRID` 상태로 표면화하고 소비자가 패킷을 재흡수/차단한다.

사전등록 NE3는 event ON에서 `t03_blocks=0`을 요구한다. producer를 SPONTANEOUS로
바꾸어도 격자 경계는 같으므로, 이 11건이 producer 선택만으로 사라진다는 근거는 없다.

## Q1 — OUT_OF_GRID의 물리 의미

다음 중 어느 계약이 맞는가.

1. **정당한 BF exact-zero**: 격자 밖에서는 BF 사건만 0이고 전자산란·선 수송은 계속한다.
   OUT_OF_GRID는 관측 카운터로 남기되 패킷 차단 사유가 아니다.
2. **fail-closed 결함**: 모델 주파수 영역 밖이므로 패킷을 차단한다. 이 경우 NE3의
   `t03_blocks=0`은 격자 확장 전에는 요구할 수 없으며 사전등록을 정정해야 한다.
3. **격자 확장 의무**: 패킷이 방문하는 영역까지 사건측도를 정의해야 한다. 이 경우
   shared NLTE/BF grid 계약과 SH-GRID에 미치는 범위를 함께 지정해 달라.

특히 구 코드의 0 반환을 물리 계약으로 계승할지, 결함 은폐로 볼지 판정해 달라.

## Q2 — NE2 증거 강도

상류 A208/GPU publication이 음수 사건측도를 먼저 거부하므로 정상 생산 경로에서 음수가
GPU transport kernel까지 도달하지 않는다. 현재 준비한 증거는 다음 두 조각이다.

- CPU/CUDA가 동일한 raw classifier를 공유하며 GPU fixture가 NEGATIVE 상태를 직접 시연
- 정적 음성대조가 CPU/GPU 소비자 네 곳의 직접 배열 우회를 주입해 모두 검출하고,
  각 소비자가 non-OK를 차단하는 코드를 확인

이 결합으로 NE2 자격이 있는가, 아니면 publication을 시험에서만 우회해 실제 transport
kernel에 음수를 넣는 통합 음성대조가 반드시 필요한가.

## Q3 — ME2 정본

저장소 사전등록의 ME2는 **GPU event ON/OFF 스펙트럼 차이**다. 당시 Codex 작업문은 이를
**두 생산자 사건측도 배열 차이**로 바꾸어 전달했고, 패치는 후자를 `[E-ME2]`로 출력했다.

현재 조치:

- 배열 차이는 `[E-AUX-MEASURE-ARRAY]` 보조 계측으로 강등
- ME2 이름은 원래의 GPU ON/OFF 스펙트럼 비교에 돌려놓음

사전등록 우선 원칙상 이 조치가 맞는지 판정해 달라. 배열 차이 0만으로 기존 GPU 결과의
영향 0을 판정할 수 있는지도 명시해 달라.

## 요청 판정 형식

```text
Q1: 1 / 2 / 3 / 판단불가 — 근거
Q2: 자격있음 / 통합주입필수 / 판단불가 — 근거
Q3: 원사전등록유지 / 배열측정으로대체 / 판단불가 — 근거
MC-EVT 현재 상태: PASS 가능 / 부분 폐합 / BLOCKED
```
