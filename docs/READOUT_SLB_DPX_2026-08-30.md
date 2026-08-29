# 판독 — DET-SUPERLEVEL-B (B-1·B-2) + d-proxy 예측 P-1~P-4 (운전석 계측, 2026-08-29)

계약 둘: `docs/RUNG_SUPERLEVEL_B_2026-08-28.md` (SLB-*) ·
`docs/PREREG_DPROXY_JBAR_ATTRIBUTION_2026-08-29.md` (DPX-*).
★**판정 아님** — 라벨 부여는 판정자 몫. 아래는 실측과 명령뿐이다.

## 0. 런 착지 — **두 팔 모두 FAILED**

| | B-1 `326688` | B-2 `326689` |
|---|---|---|
| 상태 | **FAILED 70:0** · elapsed **00:00:05** | **FAILED 70:0** · elapsed **02:39:08** |
| 노드 | syn102 | syn102 |
| 사인 | `DET_FLIGHT_FATAL Z2A raw producer capture is not armed` | `[R7][FATAL] lane=DET iter=0 rc=4` |
| 산출 | stderr·RUN_FOOTER **없음** | stderr 1,252행 · RUN_FOOTER 있음 |

### 0-1. B-1 — 계약 구성이 이 런처 경로에서 실행 불가

런처 `scripts/run_det_convergence_2026-08-08.slurm` 의 `A210_Z2A_PROBE` 분기가
capture 2노브 = 1 을 요구한다(`:229`·`:231`). 계약 §7-1 은 B-1 에 **ZO 값(0,0)** 을 얹으라
했고, 봉인 ZO 는 **다른 분기**(zero-opacity census)로 돌았기에 0 이 허용됐다.

| 런 | INDEPENDENT_CAPTURE | SPRODUCER_CAPTURE |
|---|---|---|
| 봉인 ZO | 0 | 0 |
| 봉인 R-II | 1 | 1 |
| **B-1** | **0** | **0** |
| B-2 | 1 | 1 |

⟹ 런처·구현·봉인 무죄. `die` 가 옳게 작동. **게이트 FAIL 아님**(게이트 전 사전 가드).

### 0-2. B-2 — ★승격이 방사평형 불일치를 드러냄

```
[A2-10][LINE-SATURATION-BLOCKED] reason=UPSTREAM_LINE_SCAN_INCOMPLETE complete=0
[A2-10][LINE-NET-BLOCKED] status=RADEQ_SIGN_MISMATCH line_status=UNRESOLVED_CANCELLATION
    line=1370680 shell=33 equation=FMA_ETA_MINUS_CHI_JBAR negative_policy=CMFGEN_SRCE_CHK
[A2-10][BLOCKED] reason=RADEQ_FIXED_T_BUNDLE_BUILD_FAILED material_update=BLOCKED rc=3
[A2-10][BLOCKED] event=R7_MATERIAL_UPDATE_BLOCKED iter=0 te_generation_before=1 after=1
[R7][FATAL] lane=DET iter=0 rc=4
```

**★봉인 대조 — 승격 전에는 0건**:

| 런 | SUPER_LEVELS | RADEQ_SIGN_MISMATCH | UNRESOLVED_CANCELLATION | R7 FATAL |
|---|---|---|---|---|
| 봉인 R-II | 미설정 | **0** | 0 | 0 |
| R-III (승격 전) | 미설정 | **0** | 0 | 0 |
| **B-2** | **=1** | **1** | **2** | **1** |

- **iter=0 에서 사망** — R7 phase 는 `view`·`a208`·`a209` 까지 도달, `te_generation 1→1`(갱신 차단)
- **위생 카운터 `clamp=0 floor=0 cap=0`** — 물리값 조작 없이 fail-closed
- **`LINE-SATURATION-ROW` 0행** — Stage-4 이전에 죽음

### 0-3. 배너 (계약 E-2 대상) — B-2 stdout 에 실재

```
ACTIVE (13015 FL -> 1173 SL across ions)
K=100: 21919 levels lumped
```
`off (identity)` 0건. ⟹ 승격은 **형상을 바꾸지 않았다**(계약 §1 문면대로).

## 1. DPX 예측 P-1~P-4 — **전부 측정 불가**

사전등록 §6 산법은 **한 런 안의 반복 0↔1 짝**을 요구한다. B-2 는 **iter=0 에서 사망**해
반복 1 이 없고 `LINE-SATURATION-ROW` 가 0행이다 ⟹ **짝 0**.

| # | 요구 | 승격 전 기준 | 승격 후 실측 |
|---|---|---|---|
| P-1 | Δn_upper 값 수 ≥ 800 | 73 | **측정 불가 (짝 0)** |
| P-2 | 축약비 ≤ 2.0:1 | 21.9 : 1 | **측정 불가** |
| P-3 | 선-수준 ρ > +0.4107 | +0.4176 관측 / +0.4107 등록 | **측정 불가** |
| P-4 | 73 ↔ 해당 준위들의 SL 수 | 73 (준위 1,599) | ★**부분 재료만** — 아래 |

**P-4 부분 재료**: 배너의 전역 비 = 13,015 FL / 1,173 SL = **11.1 : 1**.
승격 전 실측의 국소 비는 **21.9 : 1**(ion III shell 0 의 1,599준위 → 73값)로 **전역 평균의 약 2배**다.
★그러나 이 준위들이 매핑되는 **SL id 는 로그에 없다** ⟹ 「73 = SL 수」의 직접 확인은
**여전히 미실행**이다. 운전석 §2 주장은 검사되지 않은 상태로 남는다.

## 2. 승격 전 실측 (사전등록 §2·§2-1·§2-2 — 재게)

| 양 | 값 |
|---|---|
| 짝 (shell 0, ion III) | 2,420 |
| ΔJ̄ | 서로 다른 값 **2,420** · 범위 −3.1004e-01 ~ +2.9778e-01 |
| Δn_upper | 서로 다른 값 **73** · 준위 **1,599** · 축약비 **21.9:1** |
| 선-수준 ρ | **+0.4176** |
| 군-수준 ρ (73군) | **+0.6286** (행≥5 인 27군: +0.5733) |
| 군내 셔플 귀무 (200회) | 중앙 +0.4097 · 95% CI **[+0.3942, +0.4288]** ⟹ 관측이 구간 안 |

## 3. 운전석이 하지 않은 것

- **라벨 부여 0** — SLB-*·DPX-* 어느 것도 붙이지 않았다.
- **B-3 예비런 미제출** — 계약 §11 상 B-3 자격은 `SLB-F-D`(G-7 미러 실패)뿐이고 G-7 은
  집행조차 못 됐다. 자격 없음.
- 계약 개정 0 · src 0 · 봉인 접촉 read-only(사전·사후 `sha256sum -c` rc=0 둘 다).
- `RADEQ_SIGN_MISMATCH` 의 물리 기전 규명 — 이 단의 권한 밖.
- P-4 의 직접 확인(SL id 부재).

## 4. 집행 기록

게이트: G-0 PASS · G-S PASS(통제 19종) · **G-RC PASS**(앵커 4필드×2 bit-일치, 음성대조
1바이트 → `ANCHOR_FIELD_MISMATCH` 시연) · **G-1 PASS**(RUN_SEAL 135행 non_ok=0 ·
shared_hardlinks=0). **G-2~G-7 미집행**(런 산출 부재).
계약 정정 1건(§7-1 levels digest 오기, user 지시 운전석 정정, 커밋 `c19ccf4`).
