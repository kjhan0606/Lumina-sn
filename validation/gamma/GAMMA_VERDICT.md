# Γ단 판정 — 2026-08-08 03:5x

## 판정: **Γ1 FAIL** (NC3).  패치를 되돌리고 Codex 로 반송.

정적 검사는 전부 통과했다.  **런타임 첫 게이트에서 죽었다.**

| 검사 | 결과 |
|---|---|
| `git apply --check` 실제 트리 | PASS |
| CPU 빌드 · CUDA 빌드 | PASS (`static` 전환 `nm` 확인, 새 API `gamma_deposition_publish`/`require`) |
| Γ2-a 수식 불변 | **PASS** — `compute_gamma_deposition` 67행 중 `static` 한정자 2줄만 |
| 새 `getenv` / clamp·floor | 0 / 0 |
| **NC3 (Ni·Co 존비 0 덱)** | **FAIL** |

---

## NC3 가 잡은 것

```
=== T3_CPU_REPRO deck=..._nicozero bin=./lumina.postGamma sha=1ab6cdd31867 ===
[A2-10][SEED] bootstrap T_e published generation=1 ... manifest=25b03b07ae22
[GAMMA][BLOCKED] reason=GAMMA_MANIFEST_BUILD_FAILED epoch=1683072 provenance=INTERNAL_BATEMAN action=TERMINATE
[GAMMA][FATAL] lane=MC iter=0 rc=5
```

**차단이 아니라 발행 자체가 실패했다.**  근원:

```c
/* 발행자가 온도 매니페스트 함수를 재사용한다 */
if (population_te_manifest_sha256(gd->heating_rate, gd->n_shells, manifest) != POP_OK) { ... }

/* 그 함수 (population_contract.c:74-81) */
const char d[]="A2-07:T_e:K:IEEE754:shell-order:v1";       /* ← 도메인 문자열이 "T_e in K" 다 */
for(...) { if(!isfinite(te[i]) || te[i] <= 0.0) return POP_INVALID_TE; }   /* ← 양수 요구 */
```

결함이 둘이다.

1. **도메인 분리 위반** — erg/s/cm³ 인 침착률을 `"A2-07:T_e:K"` 도메인으로 해싱한다.
   도메인 문자열의 목적이 "무엇을 해싱했는가"를 못박는 것인데, 이 태그는 **거짓말을 한다.**
2. **다른 양의 불변식을 물려받았다** — 온도는 양수여야 하지만 **침착률은 0 일 수 있다.**

## ★표준 덱에서도 실패한다 (오프라인 판정)

NC3 만의 문제가 아니다.  수리 전 표준 덱 런의 실측:

```
[Gamma] heating_rate[0]=7.62e-04, [49]=0.00e+00 erg/s/cm3
```

외곽 셸은 감마선이 탈출해 침착이 **정확히 0** 이다 — 물리적으로 정상이고 **모든 덱에서** 그렇다.
`te[i] <= 0.0` 이 그것을 거부한다.  ⟹ **이 패치는 어떤 덱에서도 감마를 발행할 수 없다.**

판정 근거는 코드 + 실측이며 런을 더 돌릴 필요가 없다(offline-first).
Γ2-b·Γ3 남은 런은 그래서 중단했다.

---

## 이것이 캠페인의 표적인 이유

**빌려온 계약이 원래 양의 불변식을 함께 들고 왔다.**
감마 소유권 단은 "소비자가 0 을 '할 일 없음'으로 읽는 것"을 고치려는 단이었는데,
그 수리가 **"0 을 '무효'로 읽는" 새 결함**을 만들었다.  같은 축의 반대편 오류다.

★그리고 **NC3 가 없었으면 못 잡았다.**  NC1(발행자 제거 → 차단)만 봤다면
"차단이 잘 된다"로 읽고 통과시켰을 것이다.  **정당하게 0 인 경우를 시연하는 대조**가
게이트의 과잉을 가르는 유일한 수단이다.

---

## 반송 사항 (Codex)

`heating_rate` 는 **자기 매니페스트**를 가져야 한다.

- 도메인 문자열: 양·단위·순서를 정직하게 (`"GAMMA:q_dep:erg/s/cm3:IEEE754:shell-order:v1"`)
- 유효성: **유한 · ≥ 0** (양수 아님).  전 셸 0 도 유효한 발행이다.
- 같은 점검을 `nonthermal_ioniz_rate` 에도 (역시 0 가능)
- ⚠`population_te_manifest_sha256` 은 **건드리지 않는다** — T_e 의 양수 요구는 옳다.

나머지 설계(단일 발행자·도장·이중발행 차단·세 소비자·M1/M2)는 **유지**한다.
정적 검사와 Γ2-a 를 이미 통과했으므로 재작성이 아니라 **이 한 지점의 수리**다.
