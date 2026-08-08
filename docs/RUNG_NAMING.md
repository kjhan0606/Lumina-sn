# 단(rung) 명명 규약 — 2026-08-08

user 지시: 단 이름에 **소속 접두**를 붙여 "이건 어느 팔인가" 를 매번 묻지 않게 한다.

## 접두

| 접두 | 뜻 | CMFGEN 발자국 규약 적용 |
|---|---|---|
| **DET-** | 결정론 팔 전용 | **예** |
| **MC-** | Monte Carlo 팔 전용 | **아니오** (CMFGEN 에 대응물 없음) |
| **SH-** | 두 팔 공유 기계 | CMFGEN 대응물이 있으면 **예** |

## 정본 표

| 새 이름 | 옛 호칭 | 소속 | 상태 | 정본 문서 |
|---|---|---|---|---|
| **SH-R0** | R0 | 공유(원자자료) | 폐합 | catalog 15이온 7,242준위 |
| **SH-R1** | R1 | 공유(부트스트랩) | **미결** | 발주서 `/tmp/.../codex_grey/ORDER_GREY.md` |
| **SH-R2** | R2 | 공유(물질) | 폐합(=L1-1) | `RUNG_L1_1_BOOTSTRAP_SUPPLIER.md` |
| **SH-R4** | R4 | 공유(물질) | **미결** | n_e 전하보존 잔차 1.586e-02 |
| **SH-R5** | R5 | **두 팔 barrier** | 미착수 | 배선도 |
| **DET-R6** | R6 | 결정론 전용 | 폐합 `3ca077d` | `RUNG_R6_DETERMINISTIC_LINE_JBAR.md` |
| **SH-PUB** | "R7" | 공유(두 lane) | 폐합 `9012995` | `validation/r7/R7_VERDICT.md` |
| **SH-GEN** | "R8" | 공유 | SH-PUB 안에서 폐합 | 위 판정문 §R8 |
| **SH-R9** | R9 | 공유(물질) | 미착수 | 배선도 |
| **SH-R10** | R10 | 세 열(MC·DET·CMFGEN) | 미착수 | 배선도 |
| **SH-GAMMA** | Γ | 공유 | 폐합 `5450518` | `RUNG_GAMMA_DEPOSITION_OWNER.md` |
| **SH-GRID** | 격자 단(안 B) | 공유 | 폐합 `2e26c2f` | `RUNG_GRID_CONTAINMENT_CONTRACT.md` |
| **MC-EVT** | E | **MC 전용** | 발주 중 | `RUNG_EVENT_MEASURE_LANE_AGREEMENT.md` |
| **SH-RADEQ** | (신규) | 공유 | **신규 전선** | `RADEQ_NO_BRACKET` — 미작성 |

## ★인용 정정 (2026-08-08)

배선도 `validation/layer1_replan/OUT_F_functions_and_wiring.md` 에는
**R3·R7·R8 이 존재하지 않는다.**  실제 항목 집합은
`R0 R1 R2 R4 R5 R6 R9 R10` 이고, 구현 순서표는 **숫자 1~7** 로 매겨져 있다.

- 발행 위상 = 배선도 **순서 7**  (a208+a209 를 물질 mutation 앞에)
- 결정론 line-J̄ = 배선도 **순서 6**
- n_e 전하보존 = 배선도 **순서 4**

운전석이 `TASK_R7.md` 에 *"배선도가 정한 수리(OUT_F R7·R8)"* 라고 적은 것은 **오기**다.
내용은 순서 7 과 정확히 일치했으므로 산출물에는 영향이 없으나, 출처 표기는 틀렸다.
⟹ 새 이름 `SH-PUB` / `SH-GEN` 으로 바꾸고, "R7·R8" 은 **폐어**로 둔다.

## 적용 범위

- **앞으로의** 문서·태스크·커밋 메시지·발주서는 새 이름을 쓴다.
- **이미 커밋된 메시지와 파일명은 바꾸지 않는다** — 링크가 깨지고 이등분 탐색이 어려워진다.
  이 표가 옛 호칭 ↔ 새 이름의 사전이다.
- `E` 는 파이프라인 단계 **Fable E**(D→E→F→G)와 글자가 겹쳤다.  `MC-EVT` 로 해소한다.
