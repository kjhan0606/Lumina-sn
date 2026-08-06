# live 클램프 5건 판정 — 판정런이 실제로 켜는 것

기계 분류 `SCRAP-CLAMP` 42건 중 **판정런이 실제로 넘기는 5건**. 이것들은 미설정 기본값으로
되돌리면 **동작이 바뀌므로** 기계적 스크랩에 섞을 수 없다. 개별 판정한다.

판별식(`feedback_clamps_are_not_physics_fix_the_solver`):
**정확해가 그 가드를 위반할 수 있는가.** 위반 가능하면 그것은 물리가 아니라 은폐다.

| 노브 | 값 | 판정 | 근거 |
|---|---|---|---|
| `LUMINA_HRESP_CLAMP` | 1.0 (27런처) | **NON-PHYSICAL (클램프)** | `lumina_plasma.c` H_resp 미분에서 `\|dB\| <= fac·max(blag, \|B\|/10)`. 주석이 자백: 은퇴한 lre 항이 "12 orders of meaningless extrapolation" 을 냈고 그것이 hot root 를 파괴해서 넣었다 — **미분 구성의 병리를 값으로 덮는다** |
| `LUMINA_NLTE_LTE_FLOOR` | 1 (81런처) | **NON-PHYSICAL (양쪽 다 floor)** | 주석: *"Off => legacy 1e-30 clamp (byte-identical)"*. ON 은 LTE-상대 floor + b_k cap. **두 경로 다 floor** 이고 ON 이 덜 나쁠 뿐 — "덜 틀린 승자" 를 수리로 부르지 않는다. 대장의 3단 사슬(1e-30 → b_k 1e8 → LTE floor)의 마지막 칸 |
| `LUMINA_NLTE_INV_CEIL` | 1e4 (130런처) | **UNDECIDED (클램프 아님, 기준이 물리적으로 위험)** | 값을 자르지 않는다 — **거부하고 Boltzmann fallback 으로 보낸다**(`lumina_plasma.c:14726`). 목적은 near-singular 행렬의 쓰레기해 탐지로 정당하다. 그러나 판별 기준이 *물리적 비개연성*(Boltzmann 천장×1e4)이라 **진짜 population inversion 을 거부한다**. 정본 수리 = 특이성을 조건수로 직접 탐지 |
| `LUMINA_TE_STEP_CLAMP` | 1 (27런처) | **PHYSICAL (globalization)** | `T_new ∈ [0.5·T_old, 2·T_old]` 로 **반복 스텝만** 제한한다. 수렴점에서 `T_new≈T_old` 이므로 비활성 — 고정점을 옮기지 못한다(주석의 주장이 성립). 감쇠 Newton/신뢰영역과 같은 부류. ⚠단 **다근 문제에서 어느 근에 착지하는지는 바꿀 수 있다** — 기재 |
| `LUMINA_CMF_EPAY_SMIN` | 5 (3런처) | **오분류(클램프 아님) — 그러나 ad-hoc** | 정수 **셸 인덱스 하한**: `if (epay && s >= epay_smin)`(`lumina_cmfgen.c:923`). 클램프가 아니라 **기전의 공간적 제한**이다. 왜 셸 5 인가에 대한 근거가 없다 — 임의 경계 |

## 처분

- `HRESP_CLAMP` · `NLTE_LTE_FLOOR` → **L1-5 로**. 값을 끄는 것이 수리가 아니다
  (끄면 그 아래 병리가 다시 드러난다). 수리는 각각 H_resp 미분의 구성과
  SL 해의 floor 의존성 자체다.
- `NLTE_INV_CEIL` → 목적(특이 행렬 탐지)은 유지하되 **기준을 조건수로** 바꾸는 것이 정본.
- `TE_STEP_CLAMP` → 존치. 다근 착지 문제만 기재.
- `CMF_EPAY_SMIN` → 클램프 대장에서 내리고, **임의 공간 경계**로 별도 기재.

## 방법 기재

기계 분류 `SCRAP-CLAMP` 는 **이름 정규식**이라 5건 중 1건을 오분류했다(EPAY_SMIN).
반대로 이름에 clamp 가 없는 클램프는 이 그물에 걸리지 않는다 —
`docs/CLAMP_CENSUS` 계열(88항목)과 대조해야 전수가 된다.
**수치를 "클램프 42건" 으로 읽으면 안 된다.**
