# 설계 ①: Line 에너지 폐쇄 = 2준위 SE (ETLA) — ε=1 열적 line_re 대체

2026-07-02. 근거: 4중 수렴 (물리 에이전트 정량 closure + ARTIS 소스 + CMFGEN 소스 + 코드 감사).
memory: project_farouter_5layer_audit_fixplan.

## 1. 문제 (감사 결함 #1)

champion(LINE_RE=1)의 line 에너지 항 `radeq_line_re` = 4π∫χ_line·(J−B(T_e))dν — **S_l=B(T_e) (ε=1) 암묵 가정**.
- thin 외곽(n_e~4e4)에선 충돌 공급(n_e·q)이 병목: 가스는 χ·B로 방출할 에너지가 없음.
  실측 과대: Λ(25000K, s=49) = 4.15e-7 vs 물리 정답 ~3e-11 (**10⁴×**).
- 결과: 가짜 냉근(7989K), hot root 부재, cap 해제 시 92000K 폭주 — 3증상 단독 설명.
- (별도 측면: cmfgen_assemble의 S_l=B 폴백이 J 색도 붕괴시킴 — **①의 범위 아님**, 수송은 별도 단계.)

## 2. 물리 목표 — 하나의 형태로 전 regime

라인당 가스↔복사 순 에너지 교환 (2준위 SE + Sobolev escape):

```
n_up = n_lo·(C_lu + R_lu)/(C_ul + R_ul),  R_lu=B_lu·J̄·β_esc, R_ul=(A_ul+B_ul·J̄)·β_esc
Λ_line = n_e·Σ ΔE·(n_lo·q_lu − n_up·q_ul)
```

극한 검증 (SE 항등식: 순 충돌 여기 = 순 복사 방출):
- **C-지배(thick 내부)**: n_up→Boltzmann(T_e) → 순 복사 방출 = χ(B(T_e)−J̄) 등가 → **기존 thermostat 보존**
- **R-지배(thin 외곽)**: n_up→복사 설정(산란) → Λ = 충돌-제한 thin 냉각 → 물리 에이전트 Q3: Λ(25690K)≈3e-11=H_γ, Λ∝T^3.5 안정 root
- ARTIS 등가(kpkt.cc 충돌냉각), CMFGEN 등가(Z_net).

## 3. 기존 기계 (코드 신작 불필요)

| 요소 | 위치 | 상태 |
|---|---|---|
| 2준위 SE 폐쇄 | `radeq_line_cool_etla` plasma.c:4001 | 구현·hybrid mode 2(부호 판별)·Boltzmann ceiling(무펌핑 가드) 포함 |
| radeq 치환 | `RADEQ_ETLA_DELTA` 4593 (lagged−ETLA 더해 치환) | bisection 3곳 배선 완료 |
| CN 치환 | `RADEQ_LINE_DELTA` 6722 | CN r1 배선 완료 |
| line 집합 | radeq_lines = **전체 bb census** (NLTE 미추적은 dilute-Boltzmann) | 완비 |
| 충돌계수 | vR/Upsilon (2ef9b54 trap fix 후) | 완비 |
| champion 정합 | COOL_ESCAPE=0(충돌형)·COOL_NONNEG=0(signed) 이미 설정 | ✓ |

**활성 config**: champion + `LUMINA_RADEQ_LINE_RE=0` + `LUMINA_RADEQ_LINE_RESPOND=2`.

## 4. Falsifier 사다리 (순서 고정)

- **F0 (곡선, 커밋 전 판정)**: RTRUTH 스캔, s=49: 새 폐쇄의 냉각(25000K)이 4.15e-7 → ~1e-11–1e-10로 **~10⁴ 하락**; 냉근 소멸 또는 hot root 출현. s=25/s=0 곡선의 thermostat 보존.
- **F1 (capped run)**: 안/골 회귀 게이트 |ΔT_e| ≤ 5% (20795/10705). 외곽은 상자(9402) 포화 예상 — 그 자체가 "root가 hot으로 이동" 증거.
- **F2 (uncapped, 결정)**: +WIDE_BRACKET=1 +CN_THI_ABS=140000 — **안정 hot root, 92000 폭주 없음**이 채택 기준.
- **F3 (closure 수치)**: 수렴 외곽에서 Λ_line ≈ H_γ=3.1e-11 (RADEQ-BAL/RTRUTH로 판독).

**기대치 명시 (flip-flop 방지)**: F2의 외곽 root는 1e4–5e4K 범위면 성공. **정확한 25690 일치는 ①의 목표 아님** — 이온화가 아직 tdep-IC(Si V 88%, line 희소)라 과열 overshoot 가능. 정확 착지는 ②(Γ_nt+평형 IC) 후 판정.

## 5. 리스크와 완화

- **R-a 내부 thermostat 회귀** (역사: line_re가 Phase-3 외곽 폭주를 치유했음, MC-era): hybrid mode 2가 lagged 진성 coolant 보존 + F1 게이트 5%. 실패 시 line_re를 thick 셸만 유지하는 분할은 **금지**(경계 불연속) — 대신 원인 규명.
- **R-b binned J̄ super-thermal 포켓의 펌핑 가열 재점화**: Boltzmann ceiling이 차단(4022-4023). 결정론 J는 MC보다 매끈 — 3-arm 시대보다 유리.
- **R-c 비용**: 전체 census 조립은 champion서 이미 매 iter 수행(LINE_RE=1이 skip했을 뿐) — pre-champion 시절 상시 경로였음.

## 6. 비목표 (별도 단계)

- ② Γ_nt 이온 balance + tdep IC→평형 IC (감사 #4)
- ③ cap soft화 (감사 #2/#3) — F2에서 진단용으로만 임시 해제
- ④ T_rad 빌더 수정 (감사 #5) / 수송 S_l=B 폴백 (G1)
