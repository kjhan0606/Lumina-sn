# 과제 A — Lumina 의 **실제** 물리 배선도를 문서화하라

너는 읽기 전용 분석자다. **수정 제안·수리안을 쓰지 마라.** 있는 그대로의 배선을 적는다.

대상: `./lumina/` (Lumina-sn 의 src 발췌). 참고로 `./artis/` 도 있으나 과제 A 에서는 쓰지 않는다.

## 산출물

반복(iteration) 1 회 안에서 물리량이 **어떤 순서로 무엇으로부터** 계산되는지의 사슬.
문서가 말하는 것이 아니라 **코드가 하는 것**을 적는다. 모든 주장에 `파일:행` 을 붙인다.

1. **사슬 표** — 각 행: 산출량 / 계산 함수 / 입력 / 호출 위치(파일:행) / 반복 내 순서
   최소한 다음을 포함: 수송(transport), 복사장 추정자, T_e, 분배함수 Z(T_e),
   n_e, 이온 population, 준위 population, tau_sobolev, 전이확률.

2. **세대·신선도 계약 표** — 각 간선(edge)을 지키는 계약. 각 행:
   계약명 / 검사 함수(파일:행) / 무엇의 세대를 요구하는가 / 위반 시 상태코드.
   (예: `tau_sobolev_assert_fresh`, `population_partition_build`,
    `parity_field_built`, `a210_solve_transaction`, `population_transaction_begin`)

3. **★반복 0 표** — 위 사슬의 각 입력이 **첫 수송 이전에** 어디서 오는가.
   각 입력에 대해 정확히 하나로 분류하라:
   - `DECK` : 덱에서 읽어 그대로 쓴다
   - `COMPUTED` : 반복 0 안에서 계산된다 (어디서인지 파일:행)
   - `ABSENT` : 존재하지 않는다 (그리고 그것을 요구하는 계약이 무엇인지)

4. **끊긴 간선 목록** — `ABSENT` 인데 계약이 요구하는 간선을 전부 나열.

## 이미 확인된 사실 (재발견하지 말고, **맞는지 검증**만 하라)

- `lumina_main.c` 는 반복 루프 **앞에서** `lumina_prepare_solver_owned_tau` 를 부른다.
  그 안의 `compute_plasma_state` 는 발행된 T_e(generation ≥ 1)를 요구한다.
- 세대를 올리는 곳은 루프 **안**뿐이며, `compute_radiative_equilibrium_te` 가
  그 반복의 복사장으로 자격부여한 뒤에만 오른다.
- `parity_field_built`(lumina_plasma.c) 는 `radfield_view_status == RADIATION_FIELD_VIEW_OK`
  와 `J_nu` 를 요구한다 — 즉 **수송된** 복사장.
- `lumina_plasma.c:2347` 주석은 "Fail-closed to the B2 LTE-Saha pin otherwise
  (mirrors ARTIS's LTE start)" 라고 적혀 있으나, 2644-2649 는 Saha 값을
  `(void)phi_neb` 로 폐기하고 `return POP_BF_STALE` 한다.

이 4 개가 코드와 일치하는지 각각 **확인/반박**하고 근거 행을 대라.
반박이면 무엇이 실제인지 적어라.

## 형식

한국어. 표 중심. 추측과 실측을 반드시 구분해 표기하라(`[실측]` / `[추정]`).
근거 행이 없는 문장은 쓰지 마라.
