# A2-18 — A-2 캠페인 종결 판정 (재판정)

2026-08-06 재판정(운전석). HEAD=`1502056`. 초판(2026-08-06, HEAD=`d47a596`, 16/18)을
이 문서가 대체한다.
**이 문서는 캠페인을 "완료"로 선언한다 — 단계 폐합에 한해서다.** 게이트는 별개이며
그 상태를 정직하게 적는다.

---

## 1. 단계별 폐합 상태 — **18/18**

| 단계 | 계약 | 커밋 | 상태 |
|---|---|---|---|
| A2-00~03 | 원장 자격·census·격자 4000빈·스키마 | ~`606f23a` | 폐합 |
| A2-04 | J_ν 정본 승격 (commit 초크포인트) | `bafd2bb` | 폐합 |
| A2-05 | CPU bf 광이온율 → canonical view 직적분 | `d8b9870` | 폐합 (L-1bf PASS) |
| A2-06 | CPU bb rate → J̄ selective estimator | `ece5aef` | 폐합 |
| A2-07 | population → 단일 Z(T_e)·transactional publish | `3ddd95c` | 폐합 (L-2 self PASS) |
| — | 잣대 복구 (census 앵커 + 회귀 편입) | `694d9cd` | 폐합 |
| A2-08 | CPU opacity → signed χ 게시·클램프 제거 | `8a9f861` | 폐합 |
| A2-09 | CPU emissivity → Planck 표본 제거 | `36b8426`+`bf2af37` | 폐합 |
| A2-10 | 복사평형 → 항별 J_ν heating/cooling | `540ebbd`+`068fb36` | 폐합 |
| A2-11 | 중간 마디 (전량 회귀 + 계보 감사) | `9b73c04` | 폐합 |
| A2-12 | GPU lifecycle (세대 결박·업로드) | `4077537`+`3e9e317`+`7f72c0d` | 폐합 · GPU 검증 완료 |
| A2-13 | production CUDA 가 canonical view 소비 | `d47a596` | 폐합 ⚠**단서 있음** |

> ### ⚠ A2-13 단서 — 봉인 없이 구현됐다 (2026-08-07 적발)
>
> `scripts/verify_seals.py`(C7 수리로 신설한 봉인 검증기)가 첫 실행에서 잡았다.
> `validation/a2_13_15/implementation_start_manifest.json` 의 **V1**(커밋 `50e21b6`)이
> 기록한 것:
> ```
> scope = A2-13 · source_edit_started = true
> seal_status = "BLOCKED_GIT_READ_ONLY"
> allowlist_blob_ids = 전부 UNAVAILABLE
> ```
> 캠페인 규율은 **첫 src 편집 전에 changed-output allowlist 를 봉인**하는 것이다.
> A2-13 은 그 봉인 없이 편집에 들어갔고, 그 기록마저 A2-14/15 의 V2 로 교체되며
> 현 트리에서 사라졌다(V2 는 A2-14/15 만 덮는다). 변조가 아니라 교체지만
> **실패 기록의 소실**이므로 append-only 규율 위반에 준한다.
>
> ⟹ **이 표의 A2-13 "폐합"은 봉인 근거가 없는 폐합이다.**
> 처분 미결 — 사후 검증 / 재봉인 / 단서 기재 유지 중 판정 필요.
> 증거 `validation/a2_13_15/A2_13_UNSEALED_IMPLEMENTATION.json`.

| **A2-14** | GPU signed opacity production 배선 | `19c8ab0`+`65498e1`+`c65800e` | **폐합** |
| **A2-15** | GPU emissivity CDF production 배선 | `1178878`+`84a1481`+`ae72804`+`841e88f` | **폐합** |
| **A2-16** | 스칼라 seed generation-0 격리 | `686a4a1` | **폐합** |
| **A2-17** | 스칼라 생산자·잔여 소비 철거 | `8f99e7f`…`5660352` (7커밋) | **폐합** |

**18/18 폐합.** 초판의 미완 2단계(A2-14/15와 그에 의존하는 A2-16/17)가 모두 닫혔다.

### 실측 확인 (재판정 시점)

- 원장 `docs/A2_01_DISPOSITION_LEDGER.md` **157행 · 미분류 0 · terminal verified 157**
- `src/` 전수 grep: `plasma->W[` / `plasma->T_rad[` **production 역참조 0**
  (유일한 잔여 1건은 `lumina_nlte_assemble.cu:105-112` 의 **완료 tombstone 주석**)
- A2-17 정적 추적: `classified_scalar_hits=15` 중 14는 주석·문자열·테스트,
  **실물 진단 파생 1건**(`lumina_atomic.c:728` row-local CONFIG-PREC 증인, 미게시)
- 음성대조 N17-1~5 전부 PASS (**N17-4 diagnostic-escape** 포함 — 진단이 상태로
  새지 않음을 증명)
- 원장 disposition 분포: `CLOSED_*` 111 · `DIAGNOSTIC_ONLY_CANONICAL_DERIVED` 23 ·
  `REMOVED_A2_17_*` 4 (합 157). **23행은 "제거"가 아니라 "정본 파생 진단 유지"다** —
  이 구분을 흐리지 않는다.

## 2. ★게이트 — O-PHYS 진리는 확보됐으나 **오라클로 인증되지 않았다**

초판 이후 O-PHYS가 진행돼 capture와 formal이 **정상 종료**했다
(`CMF_FLUX has finished`, 104 LS 루프, fatal 0). 진리 파일은 전부 존재한다:

| 파일 | 크기 | 검증 |
|---|---|---|
| `NETRATE` / `TOTRATE` | 1.09GB / 1.28GB | capture 4/4 |
| `CHI_DATA` / `ETA_DATA` | 각 346MB | **475,154 레코드 = 연속체 진동수 475,140 + 부기 14** |
| `EDDFACTOR`·`JH_AT_CURRENT_TIME`·`OBSFLUX`·`OBS_FREQ`·`GENCOOL`·`RVTJ` | — | 존재 |

**그러나 오라클 인증은 기계 계약이 거부했다.** 증거
`validation/a2_18/A2_18_OPHYS_ORACLE_REFUSAL.json`:

```
$ python3 scripts/package_cmfgen_ophys_attestation.py --root <ophys> --evidence <실측치>
REFUSE: expected exactly F [FIX_T], found ['T']
```

수렴 항목까지 갈 것도 없이 **첫 관문에서 거부**됐다. 그리고 수렴 실측치는 이렇다
(OUTGEN iteration 103):

| 지표 | 실측 | 계약 임계 |
|---|---|---|
| 최대 % 증가 (depth 19) | **9.03E+04** | — |
| 최대 % 감소 (depth 6) | **1.07E+06** | — |
| SOLVEBA_V13 최대 변화 | **1.0E+07** | — |
| `active_population_max_correction_fraction` | ~1e5 | ≤ 0.01 |
| `max_normalized_heat_residual` | 미달 | ≤ 0.001 |
| `Temperature held fixed` 발생 | **56회** | 0 |
| L(inner)/L(outer) | **4,158×** | ~1 |

### ★사전 합의했던 `PASS_UNCONVERGED_ORACLE` 명명을 **철회한다**

이 명명은 "불수렴"이 수 % 수준일 때를 상정하고 제안했던 것이다. 실측은 **10⁴–10⁶ %**다.
이 크기에서 "공시하면 PASS"는 방어할 수 없고, 프로젝트 자신의 기계 계약이 애초에
거부한다. 공시로 통과시키는 것은 PASS 세탁이다.

### 게이트 최종 집계

| 게이트 | 상태 | 막는 것 |
|---|---|---|
| L-1bf (A2-05) | **PASS** | — |
| L-2 self-check (A2-07) | **PASS** | — |
| A2-12 GPU lifecycle | **PASS** (H200, 음성대조 9종 `physical_launches=0`) | — |
| A2-13~15 마이크로 오라클 | **bf·bb·conjunction·opacity·emissivity 전부 CPU와 일치** | — |
| L-1bb (A2-06) | **BLOCKED** | `BLOCKED_ORACLE_NOT_CERTIFIED` |
| L-4 (A2-08) | **BLOCKED** | `BLOCKED_ORACLE_NOT_CERTIFIED` |
| L-3 · L-5 (A2-09) | **BLOCKED** | `BLOCKED_ORACLE_NOT_CERTIFIED` |
| L-6 (A2-10) | **BLOCKED** | `BLOCKED_ORACLE_NOT_CERTIFIED` + free-T 필요 |
| A2-16/17 read-trace | **PASS** | — (초판의 upstream 블로커 해소) |

**차단 사유가 바뀌었다**: `BLOCKED_MISSING_{RATE_EXPORT,CHI_DATA,ETA_DATA}`
(파일 부재) → **`BLOCKED_ORACLE_NOT_CERTIFIED`** (파일 존재, 런이 오라클 자격 미달).
진전이다 — 남은 것은 파일 생산이 아니라 **수렴한 free-T 런 하나**다.

**PASS 세탁 0건.**

### ★게이트 4종은 **실행하지 않았다** (user 결정 2026-08-06)

L-1bb·L-4·L-3·L-5 를 "측정으로만 기재"하고 돌리는 안과, 수렴한 free-T 런까지
BLOCKED 를 유지하는 안 중 **후자를 택했다.** 따라서 위 BLOCKED 는
*"돌려서 실패했다"* 가 아니라 **"돌리지 않았다"** 이다. 이 구분을 흐리지 않는다.

부수 상태: 게이트를 막던 두 사유 중 **vintage 불일치는 해소됐다** — 덱-런 종속
방침으로 `data/tardis_reference_toy06_19p48d_ophys` 를 만들었고 O-PHYS 런이 싣는
27 이온 전부가 덱과 동일 원본이다(`DECK_PROVENANCE.json`). 남은 사유는
**오라클 미인증 하나**뿐이다.

⚠ 그 결과 **게이트 코드 경로 자체가 한 번도 실행된 적이 없다.** 검증기의 작동
여부는 미확인 상태이며, 수렴 런이 나왔을 때 게이트 자신의 결함과 물리 판정이
동시에 드러날 위험을 안고 간다(전례: census 체커가 3커밋 동안 FAIL 방치).

## 3. 캠페인이 실제로 바꾼 것

셸당 스칼라 (W, T_rad) → **주파수분해 정본 `J_ν[4000빈]`**.

- **소유권**: 정본 `RadiationField` + `LineJbarCache`, commit 초크포인트 2곳,
  checked view 경유 소비만 허용, fallback(0·coarse·이전 세대) 전면 금지
- **rate**: bf Γ는 canonical view 직적분, bb J̄는 φ-가중 selective estimator
- **population**: 단일 `Z(T_e)` 정본, transactional publish(부분 갱신 0)
- **opacity/emissivity**: signed χ 게시 + 클램프 제거, Planck 표본 제거,
  단일 CDF. 음수를 처리 못 하는 소비자는 값을 바꾸지 않고 차단
- **GPU**: 세대 결박 lifecycle, fallback 금지, production 배선 완료(A2-13~15)
- **seed**: 스칼라는 generation 0 에서만 유효, 첫 commit 시 즉시 revoke

**측정된 물리 변화**: A2-05의 Γ_view/Γ_legacy = 0.987–0.998(전 이온·전 셸).

## 4. 남은 것 (정확히 두 가지)

### 4.1 수렴한 free-T CMFGEN 런
게이트 5종 전부가 이것 하나에 걸려 있다. STAGE-1(FIX_T=T)은 오라클이 될 수 없음이
기계 계약으로 확정됐다. `docs/CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md`도 같은 결론을
독립적으로 적어 뒀다("고정온도 산출물에 대한 Jν/rate/population oracle 주장은 전부 철회").

### 4.2 `full_nlte_integration_run = False` (A2-13/15)
GPU production 배선은 마이크로 오라클·lifecycle·정적 census로 폐합했으나
**전량 NLTE 통합런은 아직 수행되지 않았다**(`validation/a2_13_15/stage_status.json`).
산술은 검증됐고 남은 것은 통합 실행이다.

## 5. ★층 1 진입 판정 — **진입 완료**

초판은 "진입 가능"이었고, 재판정 시점에는 **이미 진입해 성과가 나왔다**:

- R-1~R-3(계보·덱 내부 항등식·원본 앵커링) 완료, Fable L3 검수 통과
- **R-5 → I20 공기파장 규약 결함 확정**: 45/58 이온·635,169선(덱 28.6%)이
  82–85 km/s 어긋나 있었다. 수리 완료(`64ae09f`), 판정런 인수 PASS(`e6987d9`)
- 파생: 충돌자료 vintage 자체선택 차단, 덱 파이프라인 완결(`1502056`)

층 1이 A-2의 BLOCKED 게이트와 **독립**이라는 초판의 판단은 실측으로 확인됐다 —
게이트가 하나도 안 풀린 상태에서 층 1이 확정 결함 하나를 찾아 닫았다.

## 6. 검수 체계 결산

L2 교차 검수(Codex)가 A-2에서 잡은 것은 초판 §6에 있다. **재판정 시점의 새 사실**:

- 층 1 R-5 국면에서 Codex L2를 2회 발주했으나 **두 번 다 최종 답변 없이 종료**했다
  (파일 탐색 중 예산 소진, exit 0). 발주 형태를 바꿔야 한다 —
  파일·행 범위를 운전석이 미리 잘라 붙이고 판정만 받는 방식.
- 그 국면의 결함 3건(덱 39파일 누락 · 충돌자료 vintage 자체선택 · stale 오염)은
  **전부 운전석이 "목록을 좁히지 않고 전수로 다시 세어" 찾았다.**
  L2가 잡아야 할 종류를 L1이 잡은 것이므로 L2 무용이 아니라 **L1의 전수 규율이
  유효**하다는 증거다. 둘 다 필요하다.
