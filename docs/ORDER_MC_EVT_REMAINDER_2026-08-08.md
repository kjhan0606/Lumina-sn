# MC-EVT 잔여 실행 발주서 — 2026-08-08

상태: **부분 증거 / 폐합 아님**. 정본은
`docs/RUNG_EVENT_MEASURE_LANE_AGREEMENT.md`, 사전감리는
`validation/evt/CODEX_MC_EVT_PREAUDIT_2026-08-08.md`다. 이 문서는 정본의 ME2나
게이트를 바꾸지 않고, 운전석이 남은 증거를 생성하는 순서만 고정한다.

## 0. 금지와 고정 조건

- 로그인 노드에서 모델·GPU 실행을 하지 않는다. 빌드는 가능하고 실행은 계산 노드다.
- 기존 `validation/evt/evt_mc.log`를 덮어쓰지 않는다. 새 attempt 디렉터리를 쓴다.
- ON/OFF 비교는 덱, 패킷 수, seed, GPU, task/thread 수, 바이너리, 출력 후처리를
  고정하고 `LUMINA_FIX_BF_CONTINUUM_EVENT`만 바꾼다.
- 배열 계측 `[E-AUX-MEASURE-ARRAY]`를 ME2라고 부르지 않는다. ME2는 끝까지
  **GPU 생산 ON/OFF 스펙트럼 차이**다.
- Fable Q1은 조건부 BF exact zero를 택했다. 그러나 `NLTE_NU_MIN` 아래 active BF edge가
  0개인지 census하기 전에는 exact-zero 패치도 격자 확장도 금지한다.
- E4 baseline이 무엇인지 재현할 수 없으면 E4는 BLOCKED다. 현재 작업트리가 더럽기
  때문에 단순히 `HEAD`와 작업트리를 비교하여 MC-EVT만의 회귀라고 주장할 수 없다.

## 1. 빌드·정적 게이트

운전석은 강제 재빌드 원문, rc, 시작/종료 시각, 바이너리 SHA-256을 남긴다.

```sh
make -B OMP=1 lumina
make -B cuda
make event-measure-check
make -B selftest_mc_evt_access
```

`selftest_mc_evt_access`는 계산 노드에서 실행하여 CPU/CUDA 공용 분류기의
`OK/NEGATIVE/OUT_OF_GRID/UNAVAILABLE` 네 상태를 남긴다. 정적 게이트와 fixture 실행은
서로 대체하지 않는다.

## 2. 실행 행렬

| 순서 | arm | producer/config | 목적 | 필수 증거 |
|---|---|---|---|---|
| A | CPU MC | OFF=`LEGACY_ARGMAX` | NE1/현상 대조 | producer, 사유별 T03 block, ME3 |
| B | CPU MC | ON=`SPONTANEOUS` | NE3 | producer, `t03_blocks`, 채움률 |
| C | GPU | OFF=`LEGACY_ARGMAX` | NE1·ME2 오른팔 | E2, 사유별 block, spectrum CSV |
| D | GPU | ON=`SPONTANEOUS` | NE3·E2·ME2 왼팔 | E2 `match=1`, 사유별 block, spectrum CSV |
| E | CPU/GPU injection | negative measure | NE2 | 주입 위치·값·소비자별 이름 있는 차단 |
| F | pre-MC-EVT GPU | ON | E4 baseline | binary/source identity, spectrum CSV |
| G | candidate GPU | ON | E4 candidate | F와 MC-EVT 변경 외 모든 조건 동일 |

A~D는 같은 seed로 실행한다. B에서 OUT_OF_GRID가 한 건이라도 남으면 현재 구현은 NE3
FAIL이다. edge census가 0개를 증명한 뒤에만 OUT_OF_GRID를 이름·카운터가 있는 BF
exact zero로 소비하고 packet transport를 계속하도록 고친다. edge가 하나라도 있으면
SH-GRID를 다시 열고 하단 격자 확장을 검토한다.

Fable Q2에 따라 E의 실제 GPU transport kernel 통합 음수 주입은 **필수**다. 공용
classifier GPU fixture와 정적 부정대조는 부분 증거이며 NE2를 대체하지 않는다.

F의 baseline은 source tree/patch manifest로 “candidate와 MC-EVT 변경만 다름”이 입증되어야
한다. 기존 바이너리의 시각·SHA만 있고 대응 source를 복원할 수 없으면 E4 비교에 쓰지 않는다.

## 3. 정본 비교 명령

두 spectrum CSV의 header는 정확히 `wavelength_angstrom,flux`이고 wavelength 문자열 축이
byte-identical이어야 한다.

```sh
python3 scripts/compare_event_measure_spectra.py --gate ME2 \
  --left  <gpu_event_on.csv> --right <gpu_event_off.csv>

python3 scripts/compare_event_measure_spectra.py --gate E4 \
  --left  <pre_evt_event_on.csv> --right <candidate_event_on.csv>
```

ME2는 차이를 **측정**하고 임의의 PASS 임계값을 두지 않는다. E4는 파일 전체가
byte-identical할 때만 PASS다. 필터된 로그, 정렬한 행, flux 열만의 비교는 E4 증거가 아니다.

## 4. attempt에 반드시 둘 것

- `COMMANDS.txt`: 실제 명령과 환경, scheduler job id.
- `BUILD_CPU.log`, `BUILD_GPU.log`, `BUILD_SHA256.txt`.
- A~G 각 stdout/stderr 원문과 `rc`, 시작/종료 시각.
- ON/OFF 및 pre/candidate spectrum CSV 원문과 SHA-256.
- ME1, 보조 배열 계측, ME2 JSON, ME3, E2 원문.
- NE2 주입 diff 또는 fixture identity와 기대/실제 상태.
- Fable Q1~Q3 원문 판정 `docs/FABLE_VERDICT_MC_EVT_2026-08-08.md`.
- 운전석 판정 초안. 그 뒤 Codex read-only 감리를 요청한다.

## 5. 판정 규칙

- E1: NE1~NE4 전부 기대와 일치해야 한다.
- E2: event ON CPU publication과 실제 GPU transport device field가 같고 로그에
  `match=1`이 있어야 한다.
- E3: 원래 정의의 ME1~ME3가 모두 대장에 있어야 한다.
- E4: 재현 가능한 pre-MC-EVT ON baseline과 candidate ON 출력이 byte-identical이어야 한다.
- 하나라도 미실행이면 `부분 폐합 — 미실행: ...`; Fable 판단이 필요한 정책을 임의로
  선택하면 PASS가 아니라 계약 위반이다.
