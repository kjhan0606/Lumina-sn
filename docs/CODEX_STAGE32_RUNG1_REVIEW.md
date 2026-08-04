# Stage 3.2 Rung 1 독립 리뷰

## 범위와 결론

- 리뷰 대상: `patches/stage32_rung1_readonly_lambda.patch`
- 확인한 SHA-256: `db400a22907f32b126fa9007972de4be8dbf76ad3297b8ff5bee99c04bf65bb8` — 제시값과 일치한다.
- 정본: `docs/CODEX_STAGE32_ALI_DESIGN.md`, `docs/CHARTER_STAGE32_RUNG1.md`, `patches/stage32_rung1_expected_changes.txt`.
- **`docs/CODEX_STAGE32_RUNG1.md`는 읽지 않았다.** 구현자 보고서의 주장을 판정 근거로 쓰지 않았다.
- 패치는 적용하지 않았다. `git apply --check`만 읽기 전용으로 수행했고 성공했다. 빌드, GPU, 모델 런은 하지 않았다.
- 파일 접근 장애는 없었다.

**최종 판정: REJECT / 수정 후 재리뷰 필요.** Sobolev 식 자체와 read-only side-band 배선은 대체로 맞지만, 하드 게이트의 `disposition`이 실제 분기와 동일하지 않고, `rho_local` 분포에 쓰는 에너지가 실제 per-line 방출 에너지가 아니다. 또한 유효한 정확해를 누락·거부하는 가드와 R-N4 세대 규율의 차이가 있다. 이 상태로는 사전등록의 하드 게이트, 에너지 census, `rho_local` 예측을 만족했다고 판단할 수 없다.

## 중대 발견

### F1 — `disposition`은 실제 EPAY 처분을 기록한 값이 아니다

패치는 실제 처분 분기 안에서 값을 쓰지 않고, 분기 직전에 조건을 다시 계산한다 (`patches/stage32_rung1_readonly_lambda.patch:402-409`). 특히 실제 `rate_shape_replaced` 분기는

```c
if (epay >= 2 && acc_w > 0.0 && hot_regime)
```

이다 (`src/lumina_cmfgen.c:1704`). 패치의 분류식은 `acc_w > 0.0`을 누락하고 `epay >= 2 && hot_regime`만 검사한다 (`patches/stage32_rung1_readonly_lambda.patch:408`). 따라서 `acc_w == 0`인 정확히 허용된 상태에서 실제 코드는 scalar-rescale 경로(`src/lumina_cmfgen.c:1725-1734`)를 타지만 덤프는 `rate_shape_replaced`를 기록한다.

두꺼운-bin 판정식은 현재 코드와 같은 식을 재계산하지만, 처분 전체는 branch-site 관측이 아니라 근사 재구성이다. 질문의 판별 기준인 “진짜 그 셀의 처분인가”에는 **아니다**가 답이다.

체커도 이 오류를 잡지 못한다. manifest의 하드코딩된 문자열과 `output_discarded_rows: 0`을 신뢰하고 (`patches/stage32_rung1_readonly_lambda.patch:309-313,572-583`), disposition이 실제 EPAY branch에서 기록됐는지는 독립적으로 검증하지 않는다.

### F2 — 행별 `eta_energy`는 실제 선 방출 에너지가 아니다

패치는 bin 총 `eta_line`을 저장한 뒤 (`patches/stage32_rung1_readonly_lambda.patch:394-397`) 각 선에

```text
E_l(patch) = eta_bin * dnu_bin * w_l / chi_line_bin
```

으로 배분한다 (`patches/stage32_rung1_readonly_lambda.patch:199-213,264-280`). 그러나 실제 조립은 선마다 `eta_l = w_l S_l`을 더한다 (`src/lumina_cmfgen.c:1371-1395`). 따라서 같은 bin에 서로 다른 `S_l`을 가진 선이 둘 이상 있으면

```text
(sum_i w_i S_i) * w_l / (sum_i w_i) != w_l S_l
```

이다. 불투명도 share는 방출 에너지 share가 아니다. 이 값으로 계산한 `rho_local` 에너지 비율과 에너지 가중 median (`patches/stage32_rung1_readonly_lambda.patch:584-598`)은 사전등록의 “pre-EPAY line-emission energy” 통계가 아니다.

게다가 production은 `tau <= 1e-6`에서 expansion-opacity 분자로 `tau`를 쓴다 (`src/lumina_cmfgen.c:1369-1370`). 패치와 fixture는 항상 `-expm1(-tau)`를 쓴다 (`patches/stage32_rung1_readonly_lambda.patch:199-200,266-267,485-489`). 따라서 얇은 선에서는 numerator가 production `chi_line`을 만든 실제 share와도 동일하지 않다. fixture가 같은 비-production 식으로 `chi`를 만들기 때문에 이 불일치를 가린다.

manifest의 disposition별 합은 자기 행들의 합일 뿐이다. 독립적인 authoritative `sum(eta_pre_epay*dnu)` 총량, 선택-window 경계 bin의 비선택 선 기여, 또는 closure residual이 기록되지 않는다 (`patches/stage32_rung1_readonly_lambda.patch:300-318`). 그러므로 “row-count와 eta_pre_epay*dnu energy census를 닫는다”는 사전등록 항목도 증명하지 못한다.

### F3 — 정확해를 누락·거부하는 가드가 있다

1. `tau <= 1e-12`인 행을 조용히 건너뛴다 (`patches/stage32_rung1_readonly_lambda.patch:177-180,253-255`). 양의 유한 `tau`에 beta 정확해가 존재하므로 이는 단순 validity 검사 아니라 숨은 active-set floor/omission이다. manifest의 `floor: 0` 주장과도 맞지 않는다.
2. `lambda_star < 1`과 `rho < 1`을 강제한다 (`patches/stage32_rung1_readonly_lambda.patch:204-209`; reader에서도 `:542-547`). 수학적으로 유한 양의 `tau`에서는 둘 다 1보다 작지만, binary64에서는 예를 들어 `tau=2e16`일 때 `beta=5e-17`, `1-beta=1.0`으로 올바르게 반올림된다. 유한하고 정상인 입력을 실패시키므로 “정확해가 위반할 수 있는 가드”에 해당한다.
3. 유한 `C_ul`과 `A_ul`의 합이 overflow할 수 있는데 `denom`의 비유한성을 곧바로 거부한다 (`patches/stage32_rung1_readonly_lambda.patch:94-101`). 정확한 비율 `C/(C+A)`가 정의되는 유한 입력도 안정적인 비율 계산 대신 거부될 수 있다.

비유한값을 다른 물리값으로 대체하는 새 fallback은 없다. 그 점은 맞다. 문제는 위 유효행 누락과 표현 가능한 극한값 거부다.

### F4 — R-N4와 동일한 세대 규율이 아니다

좋은 부분은 payload header와 JSON manifest 양쪽에 `iteration`/`field_generation`을 쓰고 SHA-256을 검증하며, 기존 경로 덮어쓰기·payload tamper·generation mismatch를 음성 대조한다는 점이다 (`patches/stage32_rung1_readonly_lambda.patch:231-247,295-327,528-583,637-651`).

그러나 R-N4의 핵심 규율과 다음 차이가 있다.

- writer가 실제 경로에 `.iter%03d` 같은 세대 식별자를 붙이지 않고 사용자가 준 경로를 그대로 쓴다 (`patches/stage32_rung1_readonly_lambda.patch:341-352`). R-N4는 세대가 이름과 header 양쪽에 존재한다.
- 공용 reader의 `expected_iteration`이 필수 keyword가 아니라 기본값 10이다 (`patches/stage32_rung1_readonly_lambda.patch:528`). R-N4는 소비자가 기대 세대를 생략하면 `TypeError`로 막는 규율이다.
- 두 실행 경로 모두 독립적으로 얻은 generation metadata가 아니라 `iteration`을 두 인자에 그대로 복제한다 (`patches/stage32_rung1_readonly_lambda.patch:414-415,427-429`). header가 두 필드를 가진다는 것만으로 실제 field generation의 독립적 계보가 증명되지는 않는다.

따라서 SHA sidecar 무결성은 구현됐지만 “R-N4와 동일 규율”은 **부분 충족**이다.

## 요청 항목별 판정

### 1. 읽기 전용 계약

**기존 물리 소유자에 대한 mutation은 발견하지 못했다.** 새 쓰기는 전용 진단 배열 `stage32_eta_pre_epay`와 `stage32_epay_disposition` 및 writer-local/static cache뿐이다 (`patches/stage32_rung1_readonly_lambda.patch:26-30,68-89,387-409`). `S_fixed`, `J`, 기존 opacity/emissivity/rate/population 배열을 새 코드가 쓰는 경로는 없다. 새 artifact도 transport에 재주입되지 않는다.

단, 새 static `line2k` cache는 ON에서 영구 상태와 heap allocation을 만든다 (`patches/stage32_rung1_readonly_lambda.patch:68-89`). 물리 상태 mutation은 아니지만 순수한 무상태 observer도 아니다.

### 2. OFF 중립성

정적 코드상 gate OFF는 엄격한 조기 return이다 (`patches/stage32_rung1_readonly_lambda.patch:341-342`). 진단 배열 allocation, memset, snapshot은 모두 path-valued gate 또는 null pointer 뒤에 있다 (`:362-372,387-395,402-409`). OFF에서 RNG 소비, 부동소수 rounding-mode 변경, 새 heap allocation, 기존 물리 배열 쓰기는 보이지 않는다. 새 wrapper가 `getenv`를 한 번 호출하는 것 외의 실행 부수효과도 없다.

`CMFGENState` 크기와 뒤 멤버 offset은 header 변경으로 달라지지만, OFF에서는 새 포인터가 `cmfgen_init`의 전체 `memset`으로 null이고 새 heap buffer는 할당되지 않는다. 기존 값 기반 산출물을 바꾸는 직접 경로는 찾지 못했다.

따라서 **semantic OFF neutrality는 정적 PASS**다. 다만 selftest는 OFF/ON 기존 산출물 byte-compare를 하지 않으므로 “모든 기존 산출물 byte-identical”이라는 경험적 명제는 이 패치만으로 입증되지 않았다. 모델 런 금지 때문에 본 리뷰에서도 실측하지 않았다.

### 3. Sobolev beta와 Lambda-star

식 `beta=-expm1(-tau)/tau`, `Lambda_star=1-beta`는 설계문서 §2.2와 일치한다 (`patches/stage32_rung1_readonly_lambda.patch:96,197-198,264-265`). 계산되는 범위에서는 `tau -> 0+`에 `beta -> 1`, `Lambda_star -> 0`; `tau -> infinity`에 `beta -> 0`, `Lambda_star -> 1`로 옳다. `expm1` 선택도 작은 tau cancellation을 피한다.

그러나 구현은 `tau <= 1e-12`를 기록하지 않아 0 극한을 노출하지 않고, 큰 유한 tau에서 `Lambda_star`가 1로 반올림되면 실패한다. 따라서 **수식 PASS, 구현 도메인/극한 처리 FAIL**이다.

### 4. clamp/floor/cap/fallback/비유한값 대체

물리값 대체 fallback은 없고 비유한 입력은 대체하지 않고 실패한다. 그러나 F3의 `tau <= 1e-12` 누락, `lambda_star < 1`/`rho < 1` 거부, overflow 가능한 `denom` 거부는 정확해가 위반할 수 있는 가드다. 이 항목은 **FAIL**이다.

### 5. 하드 게이트

artifact 자체의 생존 경로는 구조적으로 맞다. 호출은 EPAY가 포함된 assemble 및 formal solve 뒤에 있고 (`patches/stage32_rung1_readonly_lambda.patch:411-418,423-430`), 파일은 `eta_line`/`S_fixed`로 돌아가지 않는 side-band다. 그러므로 Rung-1 값 자체가 EPAY에 의해 폐기되지는 않는다.

하지만 disposition은 F1처럼 실제 branch-site 값이 아니고, 에너지는 F2처럼 실제 per-line pre-EPAY 방출 에너지가 아니다. manifest의 `output_route`와 `output_discarded_rows`는 계산된 증거가 아니라 상수 문자열/숫자다. 따라서 **배선 생존 PASS, 요구된 disposition census 하드 게이트 FAIL**이다.

### 6. 세대 계약과 SHA-256 sidecar

SHA-256 payload binding, header/manifest의 iteration 일치, overwrite/tamper/mismatch 거부는 PASS다. 세대-stamped filename, 필수 expected-generation choke point, 독립 field-generation provenance가 없어 R-N4 동일 규율은 FAIL이다. 상세는 F4와 같다.

### 7. 사전등록 기대변경집합

- OFF의 기존 산출물 불변: 정적 경로상 가능하지만 byte-identity 대조가 없어 미입증.
- ON의 신규 payload+manifest 외 기존 산출물 불변: 기존 물리 배열 write는 없어 가능하지만 역시 byte-identity 대조가 없다.
- `rho_local` 예측 구간: F2의 opacity-share 에너지 배분 때문에 사전등록한 **line-emission-energy weighted** fraction/median을 측정할 수 없다.
- 하드 게이트: F1 때문에 마지막 열이 모든 허용 상태에서 실제 처분이라는 계약을 지키지 못한다.
- disposition별 count/energy closure: 자기 행 합만 확인하고 authoritative cell 총량에 대한 closure가 없어 지키지 못한다.
- clamp/floor/cap/fallback 0: `tau <= 1e-12` 누락과 정확해 거부 가드 때문에 지키지 못한다.
- R-N4 동일 세대 규율: F4의 차이 때문에 지키지 못한다.
- KA-3.2.3 beta 결함 음성 대조: 의도적 `beta *= 0.5`가 analytic judge에서 실패하도록 구성돼 있어 이 항목은 충족한다 (`patches/stage32_rung1_readonly_lambda.patch:629-636`).

결론적으로 사전등록 전체는 이 패치로 만족시킬 수 없다. 최소 수정 요건은 (1) disposition을 실제 EPAY 각 branch 안에서 기록하고 모든 branch predicate를 공유하는 것, (2) `w_l S_l`의 실제 per-line pre-EPAY 에너지를 관측하거나 에너지 가중 예측을 제거하는 것, (3) authoritative cell 총량과 count/energy closure residual을 manifest에 봉인하는 것, (4) 유효 tau 누락 및 큰-tau 반올림 거부를 제거하는 것, (5) R-N4식 이름·필수 소비자 기대세대·실제 generation provenance를 적용하는 것이다.
