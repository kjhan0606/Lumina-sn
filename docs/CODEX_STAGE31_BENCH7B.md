# Codex A-S31 round 7B — 로그 경계 외삽 재인증

날짜: 2026-08-01  
상태: **FAIL / STOP AT KA3**  
판별 벤치: **NOT RUN (KA gate closed)**

## 결론

`src/lumina_cmf_field.c`의 inner/outer face χ,η 재구성을 양끝 대칭 로그-공간 선형 외삽으로 교체했다. 기존 선형 외삽에서 음수가 되던 급경사 양수 stencil은 이제 양수로 유지되며, 혼합 0/양수 stencil은 로그 불가를 우회하지 않고 face·field·frequency 좌표를 기록한 뒤 `LCMF_ENEGATIVE`로 fail closed한다. KA3의 사전등록 진공 `χ=η=0`은 두 stencil 값이 모두 정확히 0인 동일 영장으로만 명시적으로 연장한다.

KA1 세 optical depth는 모든 원문 창과 문턱을 통과했다. 그러나 다음 순서의 KA3가 원문 창을 크게 벗어났다. 차터의 “하나라도 창 밖이면 FAIL 기재 후 중단” 규율에 따라 KA2 및 parity59 `J_det`/§7.2/§7.3 판별 벤치는 실행하지 않았다. acceptance 완화, clamp/floor, 신규 모델/GPU run, 커밋은 모두 0이다.

## 1. rung — 양끝 로그 face 외삽

shell-center 두 값 `(v0,v1)>0`과 center 좌표 `(r0,r1)`에 대해 face 값은

```text
t = (r_face-r0)/(r1-r0)
v_face = exp(log(v0) + t [log(v1)-log(v0)])
```

로 계산한다. inner face는 `t<0`, outer face는 마지막 두 center에 대해 `t>1`이며 같은 식을 쓴다. interior center 사이는 기존 선형 보간을 유지했다.

fail-closed 계약은 다음과 같다.

- stencil이 둘 다 양수: 위 기하 외삽.
- 한 값이라도 0이고 다른 값이 양수: `LCMF_ENEGATIVE`, inner/outer face와 좌표 기록.
- 음수/non-finite 입력: 기존 input validation 또는 reconstruction guard에서 실패.
- 계산 `exp`가 underflow 0 또는 overflow/non-finite: 실패; 0/floor로 대체하지 않음.
- `(0,0)`인 KA3 정확 진공장: 동일한 0장. 이는 양수 production bin의 fallback이 아니며 KA3 원문 `χ=η=0`을 바꾸지 않는다.

strict 회귀는 다음을 확인했다.

| check | result |
|---|---|
| 과거 outer 선형 외삽이 음수였던 `1 -> 0.1` stencil | 기하 외삽 solve **PASS** |
| outer 혼합 `1 -> 0` stencil | 좌표/메시지 포함 fail-closed **PASS** |
| inner 혼합 `0 -> 1` stencil | 좌표/메시지 포함 fail-closed **PASS** |
| strict C11 `-Wconversion -Wshadow` build | **PASS** |
| Stage 31 skeleton | **PASS** |
| production field driver strict build | **PASS** |

## 2. KA1 — PASS

runner의 세 τ, 세 nested grid `(128,32)`, `(256,64)`, `(512,128)`, 80-digit exact oracle, 사전등록 `p_obs` 창과 모든 수치 acceptance를 원문 그대로 사용했다.

| χR | prereg `p_obs` window | measured `p_obs` | finest I rel L2 | finest J rel L2 | finest max scaled | finest residual | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| `1e-3` | `[1.90,2.06]` | `1.9905731831` | `3.335975530e-7` | `3.362414859e-7` | `1.422942880e-9` | `1.070031154e-5` | **PASS** |
| `1` | `[1.90,2.08]` | `1.9912927418` | `3.498958543e-7` | `3.577613570e-7` | `1.318450184e-6` | `5.946114788e-6` | **PASS** |
| `100` | `[1.85,2.15]` | `2.0221547417` | `4.240575734e-7` | `4.431357905e-6` | `1.170793363e-4` | `1.967905971e-7` | **PASS** |

세 경우 모두 I/J relative L2 `<=1e-4`, max scaled error `<=3e-4`, generic `p_obs` `[1.8,2.2]`, transport residual `<=1e-4`, clamp/solution-negative/sign-uncertain/non-finite count 0을 만족했다. 결정론 3회 SHA-256은 모두

```text
6d00719782a87edd65fe4a67ab7648c416afdd909ba6933d8d6d2a81f3f83ddb
```

로 동일했다.

## 3. KA3 — FAIL / gate stop

전체 6격자를 실행했으며 공식 triple은 `(256,1024)`, `(512,2048)`, `(1024,4096)`이다. 인증 MPFR report는 기존 `docs/s31_results/mpfr_cert_rung8_fine.json`을 그대로 사용했다.

| grid | profile L1 | profile L2 | centroid error | invariant error | residual | certified neg/uncertain/nonfinite |
|---:|---:|---:|---:|---:|---:|---:|
| `256x1024` | `1.567062077e-3` | `1.412444355e-3` | `1.559064395e-5` | `2.340426519e-5` | `1.957072660e-11` | `0/0/0` |
| `512x2048` | `7.844203287e-4` | `7.035137827e-4` | `7.813975446e-6` | `1.172554586e-5` | `1.760984167e-11` | `0/0/0` |
| `1024x4096` | `3.946008629e-4` | `3.534552600e-4` | `3.911655570e-6` | `5.868630018e-6` | `1.822239313e-11` | `0/0/0` |

사전등록/acceptance 비교:

| item | window/threshold | measured | result |
|---|---:|---:|---|
| official triple profile-L2 `p_obs` | `[1.96,2.04]` prereg; `[1.8,2.2]` acceptance | `0.9930510762` | **FAIL** |
| finest profile L2 prereg | `[8.8e-5,1.08e-4]` | `3.534552600e-4` | **FAIL** |
| finest profile L2 acceptance | `<=1e-4` | `3.534552600e-4` | **FAIL** |
| finest profile L1 prereg | `[2.50e-5,2.80e-5]` | `3.946008629e-4` | **FAIL** |
| finest profile L1 acceptance | `<=1e-4` | `3.946008629e-4` | **FAIL** |
| centroid / invariant | each `<=1e-4` | `3.912e-6 / 5.869e-6` | PASS |
| transport / boundary / clamp / certified counters | original thresholds | all inside / zero | PASS |

### 실패 분해

로그 외삽 변경은 KA3에서 face stencil이 항상 정확한 `(0,0)`이므로 산술적으로 0장을 그대로 반환한다. 따라서 이번 KA3 차수/오차 변화는 양수 face 외삽값 때문이 아니다.

활성 `src/lumina_cmf_field.c`를 기존 PASS 산출물과 함께 남아 있는 `impl_s31_round5b/src/lumina_cmf_field.c`에 read-only 비교했다. 활성 파일에는 runner가 사전등록한 `trapezoidal-start + branch-local quadratic-exact SC`의 `quadratic_moments`, `branch_quadratic_stencil`, quadratic source integration 경로가 없다. 활성 경로는 `k=1`에 기존 선형 history 및 `a/dx` 계수를 사용한다. 반면 비교 사본에는 trapezoidal `2a/dx` 시작과 branch-local quadratic 경로가 존재한다.

이는 다음 기존/신규 결과와도 일치한다.

| source/result | `p_obs` | finest L1 | finest L2 | status |
|---|---:|---:|---:|---|
| archived `ka3_rev4.json` | `2.0005918699` | `2.640394380e-5` | `2.394054891e-5` | PASS |
| active `src` round 7B | `0.9930510762` | `3.946008629e-4` | `3.534552600e-4` | FAIL |

따라서 현재 분해는 **로그 face positivity rung 자체의 실패가 아니라, 활성 `src`와 KA3 사전등록/기존 인증 구현의 불일치**다. 이 보고서는 비교 사본을 활성 `src`에 병합하지 않았다. 그것은 face reconstruction 교체를 넘어서는 별도 대규모 solver 변경이며, KA FAIL 뒤 중단 규율상 운전석 재승인이 필요하다.

## 4. KA2와 판별 벤치

| stage | status | reason |
|---|---|---|
| KA2 full battery | **NOT RUN** | KA3 window FAIL 뒤 의무 중단 |
| `J_det` production-input CPU solve | **NOT RUN** | KA gate closed |
| §7.2 600–3000 Å six-band table | **NOT RUN** | `J_det` 미산출 |
| §7.3 Γ D-lane Fe III idx201 / S II SL4 | **NOT RUN** | `J_det` 미산출 |
| preregistered physical readout | **UNRESOLVED** | 판별 입력 미완성 |

round 7의 baseline `J_MC/CMFGEN` 또는 Γ B/C를 새 D-lane처럼 재사용하지 않았고, production payload에 clamp/floor를 적용하지 않았다.

## 5. 산출물 및 SHA-256

| artifact | SHA-256 |
|---|---|
| `src/lumina_cmf_field.c` | `2abd13c682d88798ead5c3c5ac90b39508b556a48530a6b1c430f9d4b3ea4c6c` |
| `tests/stage31_cmf_skeleton_selftest.c` | `dccec2661be17c2a450a8a15693107f62e3e4d7718a40cb977013114ada1a2c7` |
| `scripts/stage31_cmf_field_bench.py` (7B 준비, 미실행) | `333d9c952d8fc28fa136d646bfe2ee3bc20e0fdfa3b94c57885690596dd2ab93` |
| `docs/s31_results/ka1_round7b.json` | `1bd5215f0241227a7682969c3263c3fcb94c39e6506dacc905ac2712e480a1a1` |
| `docs/s31_results/ka3_round7b.json` | `d606cb965b565c3eb312fe29a96635ddc6d53e911522678ed6a5ef102f441f0d` |

## 6. 전 수치 재현 명령

실제로 실행한 순서다.

```bash
python3 -m py_compile scripts/stage31_cmf_field_bench.py

gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -Wconversion -Wshadow -Isrc \
  tests/stage31_cmf_skeleton_selftest.c src/lumina_cmf_field.c -lm \
  -o /tmp/stage31_skeleton_7b
/tmp/stage31_skeleton_7b

gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -D_POSIX_C_SOURCE=200809L -Isrc \
  scripts/stage31_cmf_field_driver.c src/lumina_cmf_field.c -lm \
  -o /tmp/stage31_cmf_field_driver_7b

python3 scripts/run_stage31_cmf_ka.py --ka ka1 \
  --work /tmp/stage31_round7b_ka1 \
  --output docs/s31_results/ka1_round7b.json

python3 scripts/run_stage31_cmf_ka.py --ka ka3 \
  --certificate docs/s31_results/mpfr_cert_rung8_fine.json \
  --work /tmp/stage31_round7b_ka3 \
  --output docs/s31_results/ka3_round7b.json
# expected round-7B result: exit 1, status FAIL
```

KA2와 판별 벤치 명령은 실행하지 않았다. 신규 모델/GPU run과 커밋도 수행하지 않았다.

