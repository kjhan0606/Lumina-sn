# Codex A-S31 round 4 — rev3 스킴 개정 적용 + KA3 재판정

상태: **rung6 PASS, rung7 KA3 FAIL / STOP**  
정본: `docs/CODEX_STAGE31_DESIGN_REV3.md` §2, §4, §5 및 원설계 §6  
실행일: 2026-08-01

## 1. 결론

rev3의 최소 개정을 격리 작업본에 적용했다. `k=1`은 trapezoidal 시작으로 바꾸고, `k>=2`의 BDF2 계수는 유지한 채 core reset을 넘지 않는 3점 quadratic history와 quadratic-exact formal integral을 결합했다. 기존 `src`는 수정하지 않았고 결과는 patch로만 납품한다.

KA1 3격자 회귀는 세 optical-depth case 모두 PASS했다. `p_obs`, I/J 오차, max error, residual은 round 2의 수치와 전부 동일해 “KA1 실질 불변” 사전등록을 충족했다.

KA3는 승인된 `(256,1024)`, `(512,2048)`을 추가해 다섯 격자를 계산하고 최상위 `(128,512)`, `(256,1024)`, `(512,2048)` triple로 판정했다. 차수와 finest L2는 각각 `2.0011103392`, `9.5801490261e-5`로 rev3 예측과 기존 acceptance를 통과했다. 그러나 finest L1 `1.0565893300e-4 > 1e-4`, 공식 triple의 sign-uncertain 카운터가 모두 0이 아니며, finest에서 roundoff enclosure가 28회 전 실수 구간으로 넓어져 non-finite gate도 실패했다. acceptance는 완화하지 않았고 최종 KA3는 **FAIL**이다.

따라서 rung8 KA2와 rung9 무가속 coherent scattering은 구현·실행하지 않았다. 관련 patch도 gate 규율에 따라 만들지 않았다.

| rung | 내용 | 판정 |
|---:|---|---|
| 6 | trapezoidal 시작 + branch-local quadratic-exact SC, KA1 회귀 | **PASS** |
| 7 | KA3 5격자, 최상위 3격자 공식 판정 | **FAIL / STOP** |
| 8 | KA2 Nyström oracle | **NOT RUN / NOT IMPLEMENTED** |
| 9 | 무가속 coherent-scattering iteration | **NOT RUN / NOT IMPLEMENTED** |

## 2. rung6 구현

### 2.1 `k=1` trapezoidal 시작

설계식 그대로 다음 계수를 사용한다.

```text
chi_eff = chi_1 + 3a + 2a/dx
eta_eff = eta_1 + (2a/dx - 3a) I_0
```

기존 backward-Euler 시작의 `a/dx` 계수는 남기지 않았다. KA3 범위에서 `(2/dx-3)>0`이고 trapezoidal rule의 A-stability를 유지한다.

### 2.2 `k>=2` quadratic history formal integral

BDF2 계수 `3/2, -2, 1/2`는 변경하지 않았다. 각 ray branch에서 현재 segment의 두 node와 인접 node 하나를 택하며, core boundary reset의 양쪽 node를 한 stencil에 섞지 않는다. 비균일 `z` 좌표의 3점 Lagrange 재구성으로

```text
H(s) = eta + (a/dx) [2 I_(k-1)(s) - 0.5 I_(k-2)(s)]
```

를 만들고 다음 formal update를 쓴다.

```text
I_d = exp(-tau) I_u + ds (c0 J0 + c1 J1 + c2 J2)
Jn  = integral_0^1 exp[-tau(1-u)] u^n du
```

`tau<0.25`에서는 24항 급수를, 그 밖에서는 `expm1` 기반 폐쇄식을 사용한다. solve enclosure는 coefficient interval을 독립적으로 두 번 확장하지 않고 세 nodal interval에 최종 formal-integral 가중치를 한 번만 적용한다. residual probe도 같은 quadratic polynomial과 formal kernel을 사용한다.

quadratic reconstruction은 단조 스킴이 아니다. negative/sign-uncertain 분류, BDF 음수 source 진단, clamp 0 계약은 그대로 유지했다. finest에서 interval radius가 double 범위를 넘은 경우 중심해를 clamp하지 않고 enclosure를 `[-inf,+inf]`로 넓혀 `nonfinite_count`를 증가시키고 최종 `LCMF_ENONFINITE`로 fail closed했다.

### 2.3 회귀시험

self-test에 constant/linear/quadratic source의 vacuum 해석값, `tau -> 0`, negative interval, zero-straddling interval을 추가했다. 별도의 inner-core fixture는 `k=1` trapezoidal 해석값을 직접 대조하고, `k=2`가 core reset을 넘지 않는 branch-local stencil로 유한·양의 해를 내는지 검사한다. strict C11 빌드 옵션은 다음을 모두 통과했다.

```text
-std=c11 -O2 -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow
stage31 skeleton PASS
```

## 3. KA1 3격자 회귀

### 3.1 전체 수치

| tau R | grid `(Nr,Nmu)` | I relative L2 | J relative L2 | max scaled error | residual |
|---:|---:|---:|---:|---:|---:|
| `1e-3` | 128x32 | `5.28153727e-6` | `5.23053987e-6` | `1.24043580e-8` | `1.36837793e-7` |
|  | 256x64 | `1.32856264e-6` | `1.32183473e-6` | `3.09788710e-9` | `9.36365906e-7` |
|  | 512x128 | `3.33167152e-7` | `3.35714853e-7` | `1.42099237e-9` | `1.07003118e-5` |
| `1` | 128x32 | `5.55045481e-6` | `5.49763843e-6` | `6.18727102e-6` | `1.94709268e-7` |
|  | 256x64 | `1.39475566e-6` | `1.39011078e-6` | `1.72493884e-6` | `1.36583725e-6` |
|  | 512x128 | `3.49579176e-7` | `3.57357535e-7` | `1.31678797e-6` | `5.94620029e-6` |
| `100` | 128x32 | `6.92765346e-6` | `1.08499409e-5` | `1.14870263e-4` | `1.63401038e-9` |
|  | 256x64 | `1.71180097e-6` | `6.43534751e-6` | `1.16906114e-4` | `2.70963069e-8` |
|  | 512x128 | `4.24031719e-7` | `4.42865713e-6` | `1.17007671e-4` | `1.96790597e-7` |

| tau R | `p_obs(J)` | 사전등록 창 | 상태 |
|---:|---:|---:|---|
| `1e-3` | `1.9851779278` | 1.90--2.06 | **PASS** |
| `1` | `1.9874335795` | 1.90--2.08 | **PASS** |
| `100` | `2.0225322021` | 1.85--2.15 | **PASS** |

세 case 모두 finest I/J L2, max error, 차수, residual, clamp/solution-negative/sign-uncertain/non-finite 0 조건을 통과했다. 세 번의 determinism SHA-256도 새 결과 안에서 동일하다. round 2와 비교하면 표의 모든 부동소수 수치가 정확히 같고 JSON 차이는 rev2 진단 필드 분리뿐이다.

## 4. rung7 KA3 재판정

### 4.1 격자별 수치

| grid `(Ns,Nnu)` | profile L1 | profile L2 | centroid error | area error | residual |
|---:|---:|---:|---:|---:|---:|
| 32x128 | `2.68860857e-2` | `2.43136236e-2` | `8.11217954e-7` | `8.15191771e-7` | `4.07063781e-10` |
| 64x256 | `6.78867616e-3` | `6.14783204e-3` | `2.08554047e-7` | `2.09060862e-7` | `2.19199204e-11` |
| 128x512 | `1.69387067e-3` | `1.53592620e-3` | `5.28631728e-8` | `5.29271433e-8` | `2.80234304e-11` |
| 256x1024 | `4.22978524e-4` | `3.83501001e-4` | `1.33067390e-8` | `1.33148331e-8` | `3.21929809e-11` |
| 512x2048 | `1.05658933e-4` | `9.58014903e-5` | `3.33809778e-9` | `3.33902945e-9` | `2.98780412e-11` |

연속 refinement L2 차수는 다음과 같다.

```text
32->64    1.9836152683
64->128   2.0009688558
128->256  2.0018066472
256->512  2.0011103392
```

공식 판정값은 최상위 triple의 middle/fine 비, 즉 `p_obs=2.0011103392368392`다.

### 4.2 rev3 사전등록 대조

rev3 §4의 128x512 중심 예측 `1.53593e-3`을 두 번 factor-two refinement해 finest L2 중심을 다음처럼 외삽했다.

```text
L2(512x2048) = 1.53593e-3 / 4^2 = 9.59956e-5
등록 중심값  = 9.59e-5
등록 창      = [8.8e-5, 1.08e-4]
```

| 항목 | 사전등록 | 실측 | 판정 |
|---|---:|---:|---|
| 원 계열 64->128 `p` 중심 | `2.00097` | `2.0009688558` | **PASS** |
| 공식 triple `p` 창 | 1.96--2.04 | `2.0011103392` | **PASS** |
| finest L2 창 | `[8.8e-5,1.08e-4]` | `9.5801490261e-5` | **PASS** |

예측 적중은 acceptance PASS 선언이 아니다.

### 4.3 가드와 기존 acceptance

| grid | solution min | solution-negative | sign-uncertain | non-finite | solver status |
|---:|---:|---:|---:|---:|---:|
| 32x128 | `-2.47691711e-5` | 0 | 1,076 | 0 | `LCMF_ESIGNUNCERTAIN` |
| 64x256 | `+2.46657266e-17` | 0 | 7,460 | 0 | `LCMF_ESIGNUNCERTAIN` |
| 128x512 | `+4.60368189e-15` | 0 | 51,036 | 0 | `LCMF_ESIGNUNCERTAIN` |
| 256x1024 | `+8.05486952e-15` | 0 | 242,152 | 0 | `LCMF_ESIGNUNCERTAIN` |
| 512x2048 | `+9.05178375e-15` | 0 | 1,019,773 | 28 | `LCMF_ENONFINITE` |

공식 triple에서 PASS한 항목:

- finest profile L2 `<=1e-4`
- finest centroid와 invariant-area error `<=1e-4`
- `1.8<=p_obs<=2.2`
- residual `<=1e-4`
- blue/red boundary fraction `<1e-12`
- clamp 0, solution-negative 0

실패한 항목:

- finest profile L1 `1.05658933e-4 > 1e-4`
- 공식 triple sign-uncertain `51036 / 242152 / 1019773`, 요구값 0
- 공식 triple non-finite `0 / 0 / 28`, 요구값 0

따라서 acceptance 수치를 바꾸지 않은 최종 KA3 판정은 **FAIL / STOP**이다. finest의 28회는 중심 intensity의 non-finite가 아니라 독립 worst-case enclosure가 전 실수 범위로 넓어진 사건이다. 이를 tolerance, tail 제외, clamp 또는 counter 제거로 숨기지 않았다.

## 5. 격자 생성과 판정 규율

512x2048에서 독립 `exp(x_k)` 샘플은 double 나눗셈으로 복원한 `dln nu`가 기존 uniform-grid 검사 `1e-12`를 넘었다. 동일 `dx`의 고정 ratio `exp(-dx)`를 재귀 곱해 frequency grid를 만들도록 driver를 고쳤다. 물리 domain, nnu, acceptance, oracle, 제외 cell은 바꾸지 않았다.

runner는 다섯 격자를 모두 JSON에 남기되 `official_triple_grids`를 최상위 세 격자로 고정한다. profile, 보존량, residual, boundary, clamp 및 fail-closed 문턱의 숫자는 기존 코드와 동일하다.

## 6. 검증과 산출물

- 기존 `src` 수정 0; 모든 구현·실행은 `/tmp/s31_round4_work` 격리본에서 수행.
- strict compile PASS, 확장 skeleton self-test PASS, Python syntax PASS.
- rung6와 rung7 patch를 깨끗한 격리 snapshot에 순차 적용한 뒤 5개 대상 파일과 byte-identical.
- replay strict compile 및 self-test PASS.
- 신규 KA configuration/model/GPU 실행 0; 차터가 지시한 KA1과 승인된 KA3 격자만 실행.
- clamp/floor/tail 제외/1차 fallback 0, acceptance 변경 0, 커밋 0.

| 산출물 | SHA-256 |
|---|---|
| `patches/s31_rung6.patch` | `041e5c4ecc8b24f170c073f60f239ee97e9e6df32af14eb8ba6bb0aeb39ed3fc` |
| `patches/s31_rung7.patch` | `5171612278e008db3303099f8d14f1d48ae6989c545ff1bec63cc1a0b9e80bae` |
| `docs/s31_results/ka1_rev3.json` | `d294dd9d1d64ce68d5603d3905ef392ab2546241e9b4bec6232feafc063db6d5` |
| `docs/s31_results/ka3_rev3.json` | `f4f6e8a7813ef956fb3f650f23b05802eda1aece796c474932b480a97186b922` |
| `docs/s31_logs/rung6_ka1_rev3.log` | `d294dd9d1d64ce68d5603d3905ef392ab2546241e9b4bec6232feafc063db6d5` |
| `docs/s31_logs/rung7_ka3_rev3.log` | `f4f6e8a7813ef956fb3f650f23b05802eda1aece796c474932b480a97186b922` |

rung8/rung9 산출물은 KA3 gate 실패 때문에 존재하지 않는다.
