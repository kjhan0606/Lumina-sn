# Codex A-S31 round 3 — rev2 가드 재설계 적용 + KA3 실측

상태: **rung4 완료, rung5 KA3 FAIL / STOP**  
정본: `docs/CODEX_STAGE31_DESIGN_REV2.md` §3, §5, §6, §7  
실행일: 2026-08-01

## 1. 결론

rev2의 가드 재분류와 Gaussian fixture 수리를 적용했다. 유한한 `eta_eff<0`은 solve 종료 조건에서 제외해 카운터, 최소값, 최초 좌표, 두 history 값, 감쇠비, 이론 한계로만 기록한다. BDF effective source는 물리 source용 비음수 SC와 분리한 signed SC로 계산하고 누적 roundoff enclosure를 반환한다. 실제 해는 enclosure 상한이 음수이면 `LCMF_ENEGATIVE`, enclosure가 0을 포함하면 `LCMF_ESIGNUNCERTAIN`, non-finite이면 `LCMF_ENONFINITE`로 fail closed한다. clamp, floor, tail 제외, 1차 fallback은 없다.

KA3의 기존 세 격자는 모두 끝까지 계산했다. rev2 §5 사전등록 창 네 개는 전부 적중했지만 기존 acceptance는 그대로 **FAIL**이다. 특히 `p_obs=1.3335320198708085`, finest profile L2 `=2.9921707438074078e-3`로 각각 기존 `1.8–2.2`, `<=1e-4`를 통과하지 못한다. coarse에는 enclosure 상한까지 음수인 실제 이산해가 있고, 세 격자 모두 누적 enclosure가 0을 포함하는 내부 update가 있다.

따라서 KA2와 무가속 coherent scattering은 구현하거나 실행하지 않고 중단했다. 신규 KA configuration, 모델/GPU 실행, acceptance/스킴 변경, 커밋은 모두 0이다.

| rung | 내용 | 판정 |
|---:|---|---|
| 4 | rev2 가드 재설계 + stable Gaussian fixture | **PASS** |
| 5 | 기존 KA3 32x128, 64x256, 128x512 실측 | **FAIL / STOP** |
| 이후 | KA2 / coherent scattering | **NOT RUN / NOT IMPLEMENTED** |

## 2. rung4 구현

### 2.1 `eta_eff` 진단 강등

`LCMFResult`의 기존 generic `negative_count`를 다음 세 의미로 분리했다.

- `bdf_eta_negative_count`: 음의 BDF effective-source endpoint 평가 수, 기록 전용.
- `solution_negative_excess_count`: solution enclosure 상한까지 음수인 update 수, fail closed.
- `sign_uncertain_count`: solution enclosure가 0을 포함하는 update 수, fail closed.

추가로 `bdf_eta_negative_plane_count`, `bdf_eta_min`, 출력 profile의 `solution_min`, `bdf_eta_first`를 기록한다. 최초 `bdf_eta_first`에는 evaluation/frequency/ray/segment/substep/endpoint, `I[k-1]`, `I[k-2]`, decay ratio와 균일격자 BDF2 이론 한계 4를 저장한다. 유한한 음의 `eta_eff`는 그대로 signed SC에 전달하고 non-finite coefficient만 즉시 중단한다.

### 2.2 signed SC와 solution enclosure

기존 `lumina_cmf_sc_linear`는 물리 source의 nonnegative 계약을 유지한다. 별도 `lumina_cmf_sc_linear_signed`는 signed source를 허용하며 중심값과 `[lower,upper]`를 반환한다. advection solve는 각 주파수 plane과 공간 node의 enclosure를 보존하고 history interpolation, BDF effective source, SC update에 전파한다. 실제 연산 그래프의 항 절대값과 `gamma_m` roundoff 항, `nextafter` outward expansion을 사용했다.

가드 판정은 다음과 같다.

- `upper < 0`: `solution_negative_excess_count++`, 최종 `LCMF_ENEGATIVE`.
- `lower <= 0 <= upper`: `sign_uncertain_count++`, 최종 `LCMF_ESIGNUNCERTAIN`.
- 정확히 알려진 영 구간 `[0,0]`: vacuum의 해석적 zero이므로 uncertainty로 세지 않는다.
- non-finite coefficient/update: `nonfinite_count++`, 즉시 `LCMF_ENONFINITE`.

가드 실패가 있더라도 유한한 recurrence는 끝까지 진행해 측정 배열을 보존한다. public solver는 전체 계산 뒤 실패 status를 반환하므로 fail-closed 계약은 유지되고, KA 측정 드라이버만 해당 status를 기록한 뒤 profile을 출력한다.

coarse의 최초 확정 음수는 다음과 같다.

| 항목 | 값 |
|---|---:|
| frequency / segment / substep | 98 / 48 / 0 |
| 중심값 | `-2.2010974092479715e-8` |
| enclosure lower | `-2.2085116040793818e-8` |
| enclosure upper | `-2.1936832144165612e-8` |

상한도 0보다 작으므로 임의 tolerance 없이 실제 이산 음수로 판정된다.

### 2.3 Gaussian fixture 수리

같은 부호의 Gaussian 꼬리에서 `erf(high)-erf(low)`를 직접 빼지 않는다.

- 양의 꼬리: `erfc(low)-erfc(high)`.
- 음의 꼬리: `erfc(-high)-erfc(-low)`.
- 0을 가로지를 때만 원래 `erf` 차.

이는 동일한 cell-average 해석식의 cancellation-free 평가이며 floor나 tail 제외가 아니다. Python oracle도 같은 안정식을 사용하고 80-digit `mpmath` 값과 대조했다.

| grid | naive `erf-erf` zero cell | stable positive cell / total | 80-digit oracle 최대 상대오차 | 가짜 중·미세 `eta<0` plane |
|---:|---:|---:|---:|---:|
| 32x128 | 15 | 128 / 128 | `1.80887e-14` | 해당 없음; 실제 30 plane 유지 |
| 64x256 | 30 | 256 / 256 | `2.18383e-14` | 0 |
| 128x512 | 62 | 512 / 512 | `6.38508e-14` | 0 |

## 3. rung5 KA3 실측

### 3.1 profile 및 보존량

| grid | profile L1 | profile L2 | centroid error | invariant-area error | residual |
|---:|---:|---:|---:|---:|---:|
| 32x128 | `2.70066583e-2` | `2.41727337e-2` | `1.20586753e-4` | `1.82023124e-4` | `5.39848116e-10` |
| 64x256 | `8.21373923e-3` | `7.54083626e-3` | `6.14712142e-5` | `9.24965847e-5` | `1.63290531e-11` |
| 128x512 | `3.26870415e-3` | `2.99217074e-3` | `3.10323465e-5` | `4.66214644e-5` | `1.54927570e-11` |

기존 계산 정의대로 middle/fine profile L2로 얻은 Richardson 차수는

```text
p_obs = log2(7.540836255024896e-3 / 2.9921707438074078e-3)
      = 1.3335320198708085
```

이다.

### 3.2 rev2 사전등록 창 대조

| 항목 | 사전등록 창 | 실측 | 판정 |
|---|---:|---:|---:|
| `p_obs(profile L2)` | 1.25–1.45 | `1.3335320198708085` | **PASS** |
| finest centroid error | `(2.5–3.8)e-5` | `3.103234646721631e-5` | **PASS** |
| finest invariant-area error | `(4.0–5.5)e-5` | `4.662146438950384e-5` | **PASS** |
| finest profile L2 | `(2.5–3.5)e-3` | `2.9921707438074078e-3` | **PASS** |

사전등록 예측은 네 항목 모두 적중했다. 이것은 acceptance PASS가 아니다.

### 3.3 가드 카운터

| grid | 음의 eta endpoint | 음의 eta plane | eta 최소값 | 해 최소값 | solution-negative | sign-uncertain | nonfinite |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32x128 | 1054 | 30 | `-6.62799110e-14` | `-1.98223753e-5` | 193 | 591 | 0 |
| 64x256 | 0 | 0 | — | `+3.93198800e-17` | 0 | 3462 | 0 |
| 128x512 | 0 | 0 | — | `+5.94250369e-15` | 0 | 32933 | 0 |

coarse 최초 음의 `eta_eff`는 `k=98, segment=47, substep=0, downstream endpoint`이며 다음 기록을 남겼다.

```text
eta_eff             = -4.7557903111546924e-17
I[k-1]              =  1.1251344524451112e-7
I[k-2]              =  5.0547558932850906e-7
decay ratio         =  4.492579426663395
theoretical limit   =  4
```

중·미세 profile 중심값 자체는 비음수지만 누적 worst-case enclosure가 0을 포함하는 내부 update가 있어 `LCMF_ESIGNUNCERTAIN`이다. 이 카운터를 숨기거나 acceptance에서 제외하지 않았다.

### 3.4 기존 acceptance 판정

유지한 기존 acceptance 중 다음은 PASS다.

- finest centroid `<=1e-4`, invariant-area `<=1e-4`, residual `<=1e-4`.
- blue/red boundary fraction `<1e-12`.
- clamp 0, non-finite 0.

다음은 FAIL이다.

- finest profile L1 `3.2687e-3 > 1e-4`.
- finest profile L2 `2.9922e-3 > 1e-4`.
- `p_obs=1.3335`가 `1.8–2.2` 밖.
- coarse `solution_negative_excess_count=193`.
- 모든 격자에서 요구한 `sign_uncertain_count==0` 불충족.

따라서 최종 KA3 판정은 예상대로 **FAIL**이다.

## 4. 검증

- strict C11 compile: `-Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow` PASS.
- skeleton regression PASS.
- ASan/UBSan self-test 및 coarse KA3 PASS (`detect_leaks=0`).
- exact-history 양/음 사례, 관측 numeric history 음수 재현, signed SC 해석값, interval-negative/interval-straddling 분기 회귀 PASS.
- stable Gaussian tail을 80-digit oracle과 대조 PASS.
- rung4, rung5 패치를 rung3 snapshot에 순차 적용한 뒤 현재 다섯 구현 파일과 byte-identical, replay strict self-test PASS.

## 5. 산출물

| 산출물 | SHA-256 |
|---|---|
| `patches/s31_rung4.patch` | `cd2903ef78e73b38a4f74754653b0b4e5e0c6fab7c7958a3d7e1d3b4fb46fdbb` |
| `patches/s31_rung5.patch` | `19d2aa5826ea01615c901395deef9165277e318c48b62587be9291ccb9cdfe5d` |
| `docs/s31_results/ka3_rev2.json` | `389dd7fa41723120ac3028f0e82bcfc245532b13bc998da0eb9cdfd25f242db9` |
| `docs/s31_logs/rung5_ka3_rev2.log` | `389dd7fa41723120ac3028f0e82bcfc245532b13bc998da0eb9cdfd25f242db9` |

추가 회귀 로그는 `docs/s31_logs/rung4_guard_rev2.log`다. KA2/산란 산출물은 gate 규율에 따라 생성하지 않았다.

## 6. 재현 명령

```bash
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow \
  -Isrc tests/stage31_cmf_skeleton_selftest.c src/lumina_cmf_field.c -lm \
  -o /tmp/stage31_skeleton_round3
/tmp/stage31_skeleton_round3

python3 scripts/run_stage31_cmf_ka.py --ka ka3 \
  --work /tmp/stage31_round3_ka3 \
  --output docs/s31_results/ka3_rev2.json
# expected exit 1, status FAIL after all three grids complete
```
