# 과제 B — ARTIS 의 물리 배선도를 문서화하고 Lumina 와 **대조**하라

너는 읽기 전용 분석자다. **수정 제안·수리안을 쓰지 마라.** 차이를 적는 것이 전부다.

대상: `./artis/` (ARTIS 원본 발췌) 와 `./lumina/` (Lumina-sn src 발췌).

배경: Lumina 는 2 개월 전 전면 재구축 때 **ARTIS 의 배선도를 기준으로 삼았다.**
따라서 차이는 대개 (a) 의도적 변경이거나 (b) 이식 중 빠진 것이다. 둘을 구분하는 것이 목적이다.

## 산출물

1. **ARTIS 사슬 표** — 반복 1 회 안의 순서: 산출량 / 함수 / 입력 / `파일:행`.
   최소한: packet propagation, radfield estimators, T_e(thermal balance),
   partition functions, n_e(ionization balance), ion populations, level populations,
   opacity/tau, macro-atom transition probabilities.

2. **★ARTIS 의 반복 0(부트스트랩)** — 첫 수송 이전에 각 입력을 어떻게 얻는가.
   특히: LTE start 가 어디서 어떻게 일어나는가(`파일:행`), 무엇이 초기 T_e·n_e·population 을 준다.

3. **대조표** — 각 행에 대해 다음 중 하나로 분류:
   - `MISSING` : ARTIS 에 있고 Lumina 에 **없다**
   - `EXTRA`   : Lumina 에 있고 ARTIS 에 없다
   - `DIVERGENT`: 둘 다 있으나 물리·수식·순서가 다르다
   - `EQUIVALENT`
   각 행에 **양쪽 `파일:행`** 을 붙여라.

4. 다음 6 개는 반드시 다뤄라(우선순위 순):
   1. **부트스트랩/LTE start** — 복사장이 없을 때 population 을 어떻게 얻는가
   2. **최상단 이온(top ion) 처리** — 준위 수, 분배함수, 전이
   3. **이온화 균형** — Saha vs rate-SE(광이온·재결합·충돌), 어느 조건에서 무엇을 쓰는가
   4. **분배함수** — 무엇을 합산하는가, super-level 을 어떻게 다루는가
   5. **tau/opacity 갱신 시점** — 수송 전인가 후인가, 무엇으로부터
   6. **T_e 해법** — thermal balance 의 입력과 반복 내 위치

## 이미 확인된 사실 (재발견 말고 **검증**하라)

- ARTIS: `input.cc:1226` "optionally limit the top ion to one level and no transitions",
  `input.cc:153` "in case the top ion has nlevelsmax = 1" ⟹ 최상단 이온에 준위 **1 개**(바닥).
- Lumina: 로더가 전리에너지 n 개 → population n+1 개를 만들어 최상단 population 의
  준위가 **0 개**다(실측 15/74, 전부 원소 최상단).
- Lumina `lumina_plasma.c:2347` 주석: "Fail-closed to the B2 LTE-Saha pin otherwise
  (mirrors ARTIS's LTE start)" — 그러나 2644-2649 는 Saha 값을 `(void)phi_neb` 로 폐기하고
  `return POP_BF_STALE`.
- Lumina `parity_field_built` 는 **수송된** 복사장을 요구한다.

각각 **확인/반박**하고, ARTIS 쪽 대응 지점을 `파일:행` 으로 대라.

## 형식

한국어. 표 중심. `[실측]`/`[추정]` 구분. 근거 행 없는 문장 금지.
"ARTIS 가 정답" 이라고 전제하지 마라 — Lumina 의 최종 심판은 CMFGEN 이고 ARTIS 는 참고다.
차이를 **기술**하되 어느 쪽이 옳은지는 판정하지 마라(그 판정은 다음 단계에서 한다).
