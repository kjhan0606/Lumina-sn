# O-0 부록 — n_e solve 의 경로 실측 (운전석, 2026-09-08, 런 0회)

base = `slb_20260906T234024Z_dc1_ne_closure_on` (333531). 계측 = 기존 로그 판독 + 소스 대조.

## 1. `compute_electron_density` 의 호출처는 둘이다
| 행 | 문맥 |
|---|---|
| `lumina_plasma.c:7302` | `compute_plasma_state` 안 — **부트스트랩 시대**의 물질 공급 |
| `lumina_plasma.c:21196` | 후보 파이프라인 안 — 그 뒤 `:21062–21064` 가 채택점에서 이온 분율 **재평가** |

## 2. 진단 인쇄는 1회뿐이나 그것이 호출 수는 아니다
stdout:176 의 `[A2-07][n_e] charge-conservation residual: max=1.586e-02 at shell 3 (bootstrap)` 이 유일하다. 그러나 그 인쇄는 `emit_diagnostics` 게이트 뒤에 있으므로 **다른 호출은 조용히 지나간다**. ⟹ 실제 solve 횟수는 기존 로그로 알 수 없다. 새 `[A2-07][NE-TRAJ]` 의 `call=` 카운터가 ON 에서 이를 처음으로 드러낸다.

## 3. ★사다리 경로가 시대에 따라 갈린다
`r1_use = r1_rate_se_enabled() && parity_field_built(nlte, s)` (`:2800` 부근).
- **부트스트랩 시대**(stdout:172 OPEN → 200 CLOSED, 공급 8,100회): 복사장 view 미구축 ⟹ `!r1_use` ⟹ **네뷸라 Saha `phi_neb` 경로**. 인쇄된 그 solve 가 여기 속한다.
- **부트스트랩 종료 후**: `!r1_use` 이면 `lumina_bootstrap_active()` 가 거짓이라 **즉시 실패**한다 ⟹ 살아남는 후보 solve 는 전부 **rate-SE 경로**(Γ_phot = 정준 view J 적분 + σ_bf, α = Milne/RR, Seaton 충돌항).

## 4. 폐합 장부 150쌍의 지위
`q_saha` 는 후보 재평가(`:21062–21064`)의 전하합이고 그 시점은 부트스트랩 이후다(NE-CLOSURE 첫 행 stderr:419 · SEED gen2, R7 gen3 2254, gen4 4073; 부트스트랩은 stdout:200 에서 닫힘). ⟹ **150쌍은 rate-SE 경로의 표본**이다.

## 5. 함의 (계약 사정거리)
- 문턱이 지배하는 반복 F 는 **두 경로**를 갖는다: 부트스트랩 `phi_neb` 와 이후 rate-SE.
- 개정 2·3 의 오프라인 독립 사다리는 **두 경로를 모두** 구현해야 하고, 150쌍 자격 검증은 그중 **rate-SE 쪽만** 덮는다 — `phi_neb` 경로의 자격 검증 표적은 별도로 지정해야 한다.
- 이 사실은 계약 §O-0 의 「보존되지 않은 상태」 구분에 해당하며, 개정 2 가 이미 `INPUT_PENDING` 으로 처리할 자리다. 운전석은 판정하지 않고 계획(astra)에 넘긴다.
