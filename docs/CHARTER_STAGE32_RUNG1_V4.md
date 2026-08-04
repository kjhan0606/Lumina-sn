# Stage 3.2 Rung 1 v4 — 측정 대상을 실제 반복 연산자로 교체

`patches/stage32_rung1_readonly_lambda_v3.patch`
(sha256 `b41e991c5ff947e488c85b678b96bfc1f5ba4f88d78146e3978ab262e1728c30`)를
기반으로 **v4**를 낸다. F5 수리는 확인했고 유지한다. 이번 변경은 결함 수리가 아니라
**측정 대상 교체**다.

## 0. 왜 바꾸는가

v3까지의 1차 측정량 `rho = (1−eps0)·(1−beta_Sobolev)`는 **코드가 실제로 수행하는
반복의 spectral radius가 아니다.** 유도와 근거는
`docs/AUDIT_STAGE32_RUNG1_EPSILON_DISCREPANCY.md`에 전문이 있다. 요지 둘:

- production의 `eps' = C/(C+A·beta)`는 Sobolev 사다리를 합산한 **순** 열화확률이며
  닫힌 형태다. `eps0 = C/(C+A)`가 이 연산자에 대해 틀린 ε다.
- 더 중요하게, 코드가 반복하는 것은 Sobolev 2-준위가 아니라
  `S = S_fixed + (chi_es/chi_tot)·Lambda_formal[S]`이다
  (`src/lumina_cmfgen.c:1641`). 그 대각은 production이 이미 `cs->lambda_star`에
  들고 있다(`:2147`, `:2179`).

**사전등록도 교체됐다.** 새 정본은 `patches/stage32_rung1_expected_changes_v2.txt`.
구 문안(`stage32_rung1_expected_changes.txt`, sha `e3c5c186…`)은 **은퇴**이며
수정하지 마라 — sha 계보를 보존해야 한다.

## 1. 1차 측정량 교체 (필수)

행/셀의 1차 값은 production 배열만으로 계산한다.

```
rho_local[idx] = (chi_es[idx] / chi_tot[idx]) * lambda_star[idx]
```

- `radeq_line_local_response`·log-domain logistic·line2k 테이블은 **1차 측정에서
  빠진다**. 2차 산출물(아래 §3)에서만 쓴다.
- `chi_tot`가 0이면 정의되지 않는다 — 대체값을 넣지 말고 정의되지 않았다고 기록하라.
- `lambda_star`가 `[0,1)` 밖이면 그대로 기록하고 FAIL하라. 자르지 마라.

**세대 정합 (중요)**: `chi_es`는 조립 때(`:1641`), `lambda_star`는 수송 solve
안에서(`:2147`, `:2179`) 채워진다. **둘은 같은 시점이 아니다.** 어느 시점의 짝을
기록하는지 명시하고, 세대가 어긋나면 **FAIL하라**. 서로 다른 세대의 두 배열을 곱한
값은 아무 연산자의 spectral radius도 아니다. 이 정합을 어떻게 보장했는지 보고서에
파일:줄로 제시하라.

## 2. 유지할 것

- **F1** branch-site disposition 기록과 독립 evidence 검산 (`acc_w` 포함).
- **F4** 세대 규율: `.iter%03d`, 필수 keyword-only `expected_iteration`,
  독립 `field_generation`.
- **F5** 행 에너지는 production이 `eta_line`에 실제로 더하는 `eta_l`을 재사용.
  다시 계산하지 마라.
- **F3** 제거한 가드는 계속 제거 상태로. 새 clamp/floor/cap/fallback 금지.
- 읽기 전용. 선원함수·불투명도·방출률·율·population·수송 상태 무변경. 2단 이후 금지.

## 3. 2차 산출물 — 선별 view는 예측 대상이 아니다

같은 `(line, shell, bin)`에 대해 **네 값을 모두** 기록하라.

| 열 | 정의 |
|---|---|
| `beta` | `(1−exp(−tau))/tau` |
| `eps0_raw` | `C/(C+A)` — 탈출인자 없음 |
| `eps_prime` | `C/(C+A·beta)` — production의 순 열화확률 (`src/lumina_plasma.c:8471`) |
| `eps_applied` | `eps_prime`에 production의 `eps_floor`/`eps_cap` 적용 후 (`src/lumina_cmfgen.c:795-797`) |

`eps_applied != eps_prime`인 행 수를 manifest에 실어라 — 클램프 대장 입력이다.
**이 열들에는 사전등록 예측을 걸지 않는다.**

## 4. 음성 대조 (필수)

주입 결함으로 FAIL을 시연해야 게이트 자격이 있다. 최소 셋:

1. 1차 측정을 구 정의 `(1−eps0)·(1−beta)`로 되돌리는 결함 → FAIL해야 한다.
2. `chi_es`와 `lambda_star`를 **서로 다른 세대**에서 취하는 결함 → FAIL해야 한다.
3. F5 결함(누적을 `w*Sl`로 되돌림) → 계속 FAIL해야 한다.

v3에서 "양쪽 동시 ε 제거는 검출 못 함"을 정직히 보고한 것은 옳은 처리다. 이번에도
못 잡는 것은 못 잡는다고 보고하라 — 억지 가드를 넣지 마라.

## 5. 산출물과 규율

- `patches/stage32_rung1_readonly_lambda_v4.patch`. v3/v4 sha256 모두 보고.
- v3를 덮어쓰지 마라. 트리에 적용하지 마라. commit 금지.
- **격리 복사본에서만 빌드**하라. 실제 작업 트리에서 빌드하지 마라.
- 모델 런·GPU 금지. 빌드와 fixture 자기검사까지다.
- fixture에서 1차 측정량의 값과, 사전등록 v2의 예측 1·2에 대응하는 요약량
  (가중 중앙값, `1/(1−rho)`)을 산출해 제시하라. **fixture 값을 production 판정으로
  쓰지 마라** — 배선이 맞는지만 보이는 것이다.
- 각 항목에 대해 무엇을 어떻게 했고 **어느 시험이 그것을 잡는가**를 파일:줄로 제시하라.
- 전체 보고는 `docs/CODEX_STAGE32_RUNG1_V4.md`.
