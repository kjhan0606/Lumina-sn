# 회수 — 부적격 판정을 무시한 CMFGEN 인용 (2026-08-20)

근거: `docs/ORACLE_NAN_AUDIT_2026-08-19.md`.
프로젝트 자신이 **2026-08-04** 에 `validation/a2_00_oracle/` 매니페스트로 판정해 두었다:

- `CMFGEN_PHYSICAL_ORACLE = **INELIGIBLE**`
- `CMFGEN_NONLINEAR_CONVERGENCE = **FAIL**` (마지막 3회 [13800, 8980, 3460]%)
- `CMFGEN_SNAPSHOT_REPLAY = ELIGIBLE` — 단 *"this does not authorize **cross-file physics gating**"*

## 실측 — 울타리가 넘어간 지점

| 산출물 | CMFGEN 참조 | 적격성 인용 |
|---|---|---|
| `validation/a2_10/A2_10_NONOVERLAP_K36_CMFGEN_ION_OWNER_COMPARISON_2026-08-17.json` | **144건** | **0건** |
| `docs/VERDICT_A210_STAGE4_JO_2026-08-18.md` | — | **0건**(보류 표기 없음) |
| `docs/VERDICT_DET_SPROD_IV_2026-08-18.md` | — | **0건** |

## 회수 대상 (수치를 CMFGEN 내부장부에서 가져온 주장)

| 주장 | 출처 | 처분 |
|---|---|---|
| 흡수 결손 **1/172**(Lumina/CMFGEN) | 08-17 성분 대조 | **회수** |
| net 비 **4.37e5** (일부 기록 5.75e5) | 〃 | **회수** |
| 방출 **1.41×** | 〃 | **회수** |
| CMFGEN 상쇄조건 **27.5–105.2** | 〃 | **정성만 유지** — "CMFGEN 은 양 부호를 갖는다"(상쇄조건 ≫1 이 그것을 함의). **수치 인용 금지** |
| `oracle_vs_cmfgen.md` 커버리지 **17.01%** | parity59 | **회수**(별도 확인 전까지) |

## 회수되지 **않는** 것 — 오라클 무의존 결론

| 주장 | 왜 살아남는가 |
|---|---|
| `lumina_cancellation_condition = 1.0` 정확, `absolute_signed_sum == signed_rate` 17자리 일치 ⟹ **1,282선 중 순-흡수 선 0개** | 순수 Lumina 내부 성질. CMFGEN 불요 |
| `S_producer/B(T_e) = 0.999993`, 1282/1282 행이 1±1% | 〃 |
| `J_cont ≤ B`, `Jbar ≤ S_p` 전 행 | 〃 |
| **⟹ `RADEQ_NO_BRACKET` 은 LTE 시드의 필연적 귀결** (`FINDING_NOBRACKET_LTE_SEED_2026-08-19.md`) | 사슬 전체가 오라클 무의존 |
| τ_소비자/τ_생산자 ≈ 2.02e5, S_c/S_p ≈ 1.9e5, η 비 ≈ 4.7e-11 | 두 Lumina 상태 간 비교 |

★**본선의 현 표적은 전부 이쪽에 있다.** 회수가 임계경로를 건드리지 않는다.

## 재인용 조건 (사전등록)

CMFGEN 내부장부(`LINEHEAT`·`NETRATE`·`CHI_DATA`·`ETA_DATA`)를 **물리 판정에** 다시 쓰려면
아래를 **인용과 함께** 만족해야 한다:

1. 해당 런의 `validation/a2_00_oracle/*.manifest.json` 에서
   `CMFGEN_PHYSICAL_ORACLE = ELIGIBLE`, 또는
2. 그 판정을 뒤집는 **새 수렴 증거**(MAXCH 계열이 아니라 `CORRECTION_SUM` 급 독립 지표), 또는
3. 판정을 "물리 아님·형식 대조만" 으로 **명시 강등**하고 결론에 그 한정을 붙일 것.

★게이트 권고: 수렴 판정에서 **`MAXCH == 0.0` 을 PASS 로 읽지 않는다**(NaN 서명).
NaN 종료 런 8개 목록은 `ORACLE_NAN_AUDIT_2026-08-19.md` §1.
