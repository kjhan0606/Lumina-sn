# Codex A-S31 round 2 — 설계 개정 적용 및 KA 사다리 재개 보고서

상태: **중단 — rung2R KA1 PASS, rung3 KA3 FAIL**  
개정 정본: `docs/CODEX_STAGE31_DESIGN_REV1.md` 전 절  
원설계 acceptance: `docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md` §6, §8  
실행일: 2026-08-01

## 1. 결론

개정안의 face 외삽, fail-closed radial reconstruction, SC path-length subcycling, nested-face 관측, 128/256/512 공통 격자를 적용했다. KA1은 acceptance를 완화하지 않고 세 optical-depth case 모두 통과했으며, 사전등록한 세 p 창에도 모두 들어왔다.

그 다음 발주 순서에 따라 KA3를 구현하고 첫 사전등록 격자 `(Ns,Nnu)=(32,128)`을 실행했다. BDF2 유효 방출률이 음수가 되어 `LCMF_ENEGATIVE`로 fail closed했다. 설계 §2.2가 이 경우 clamp 없이 즉시 실패하도록 구속하므로 acceptance 변경, tail floor, 음수 허용, 재시도 없이 중단했다. 따라서 rung4 KA2와 rung5 무가속 산란 반복은 실행하거나 구현하지 않았다.

| rung | 내용 | 판정 | 다음 단계 |
|---:|---|---|---|
| 2R | 개정안 적용 + KA1 재판정 | **PASS** | rung3 진입 |
| 3 | KA3 homologous redshift | **FAIL** | 정직 중단 |
| 4 | KA2 Nyström oracle | **NOT RUN / NOT IMPLEMENTED** | rung3 gate 차단 |
| 5 | 무가속 coherent scattering | **NOT RUN / NOT IMPLEMENTED** | rung3 gate 차단 |

모델/GPU 실행, 기존 production `src` 수정, clamp/floor, acceptance 변경, 커밋은 모두 0이다. 변경은 기존 Stage 3.1 신규 파일, 신규 test/runner, 문서·패치 산출물에 한정했다. 저장소의 선행 dirty worktree와 무관한 변경은 건드리지 않았다.

## 2. rung2R 구현

### 2.1 C solver 개정

- `radial_value`를 `double` 직접 반환에서 status + out parameter로 바꿨다.
- 안쪽과 바깥쪽 half-cell은 가장 가까운 두 shell-center 값을 이용한 one-sided linear extrapolation으로 평가한다.
- 외삽/재구성된 `chi` 또는 `eta`가 음수/non-finite이면 face, frequency, ray, segment, 원값을 `LCMFError`에 남기고 실패한다. limiter나 zero clamp는 없다.
- 실패 cleanup이 오류 record를 지우지 않도록 result array 해제와 result 초기화를 분리했다.
- path builder는 shell index가 아니라 임의 `target_r`를 받는다. target의 두 signed z-node를 boundary, tangent, shell-center node와 함께 정렬하고 정확히 중복 제거한다.
- `LCMFOptions.r_eval/n_r_eval`과 `LCMFResult.nr`를 분리해 source `input.nr`와 관측 radial count가 달라도 된다. 기본 경로는 기존 shell-center 관측이다.
- 각 기존 segment를 `n_sub=ceil(ds/h_loc)`으로 나누고 각 subnode에서 `chi/eta`를 다시 재구성한 뒤 기존 linear SC를 반복한다. residual도 실제 substep마다 계산한다.

### 2.2 KA 관측 개정

- source `chi/eta`는 계속 shell center에서 sample한다.
- KA1 출력 반지름은 `r_j=jR/Nr`, `j=0..Nr`의 nested faces다.
- restriction은 pair average가 아니라 `fine[2*j]` exact injection이다.
- coarse/fine 반지름은 bitwise equality 또는 relative `1e-14` 일치를 검사한다.
- 공통 격자는 `(Nr,Nmu)=(128,32),(256,64),(512,128)`이다.
- exact intensity와 80-digit `mpmath` J oracle, 원래 acceptance, 제외점 0을 유지했다.

### 2.3 fail-closed 회귀

입력 shell-center `eta`는 모두 양수지만 outer one-sided extrapolation만 음수가 되는 fixture를 추가했다. solve는 `LCMF_ENEGATIVE`, `radial_index=nr`, `frequency_index=0`, 음의 원값, `negative_count=1`, `outer face` 메시지를 보존하며 종료한다. strict compile/self-test와 ASan/UBSan(`detect_leaks=0`; sandbox ptrace 환경에서 LSAN 자체가 실행 불가)은 통과했다.

## 3. rung2R KA1 수치

모든 표의 L2는 exact 대비 상대 오차다. max는 원설계의 scaled max 값이고 residual 문턱은 `1e-4` 그대로다.

| chi R | grid | I rel L2 | J rel L2 | max scaled | residual |
|---:|---:|---:|---:|---:|---:|
| 1e-3 | 128x32 | 5.2815372669e-6 | 5.2305398652e-6 | 1.2404358035e-8 | 1.3683779264e-7 |
| 1e-3 | 256x64 | 1.3285626356e-6 | 1.3218347310e-6 | 3.0978871001e-9 | 9.3636590565e-7 |
| 1e-3 | 512x128 | 3.3316715156e-7 | 3.3571485305e-7 | 1.4209923737e-9 | 1.0700311770e-5 |
| 1 | 128x32 | 5.5504548091e-6 | 5.4976384263e-6 | 6.1872710173e-6 | 1.9470926759e-7 |
| 1 | 256x64 | 1.3947556573e-6 | 1.3901107803e-6 | 1.7249388420e-6 | 1.3658372465e-6 |
| 1 | 512x128 | 3.4957917639e-7 | 3.5735753503e-7 | 1.3167879712e-6 | 5.9462002859e-6 |
| 100 | 128x32 | 6.9276534649e-6 | 1.0849940948e-5 | 1.1487026264e-4 | 1.6340103756e-9 |
| 100 | 256x64 | 1.7118009727e-6 | 6.4353475057e-6 | 1.1690611429e-4 | 2.7096306877e-8 |
| 100 | 512x128 | 4.2403171921e-7 | 4.4286571326e-6 | 1.1700767102e-4 | 1.9679059714e-7 |

| chi R | 실측 p_obs(J) | 사전등록 기대 창 | 기대 창 | 원 acceptance 1.8–2.2 | finest I/J | finest max | residual | 판정 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-3 | 1.9851779278 | 1.90–2.06 | PASS | PASS | PASS | PASS | PASS | **PASS** |
| 1 | 1.9874335795 | 1.90–2.08 | PASS | PASS | PASS | PASS | PASS | **PASS** |
| 100 | 2.0225322021 | 1.85–2.15 | PASS | PASS | PASS | PASS | PASS | **PASS** |

세 case 모두 `outer_incoming_error=0`, `center_symmetry_error=0`, `clamp/negative/nonfinite=0/0/0`이다. 동일 512x128, chi R=1 출력의 3회 SHA-256은 모두 다음과 같다.

```text
ca35c32438977de62a5082a62a7649df196ffa08804f6767ae6e5e36eb10a3cd
```

기계판독 정본은 `docs/s31_results/ka1_rev1.json`, 전체 실행 로그는 `docs/s31_logs/rung2R_ka1_rev1.log`다.

## 4. rung3 KA3 구현과 실패

### 4.1 구현한 범위

- homologous `a=1/(c t_exp)` blue-to-red frequency sweep.
- k0 static inflow plane, k1 implicit first-order bootstrap, k>=2 BDF2 history.
- BDF history intensity를 각 SC subnode로 선형 재구성하고 path-length subcycling을 그대로 적용.
- `chi=eta=0`, radial p=0 core characteristic, inner Gaussian irradiation, outer zero.
- `sigma_x=0.04`, `A=0.1`, input/output profile 양쪽 8 sigma를 포함하는 uniform `ln nu` domain.
- cell-average Gaussian boundary와 profile L1/L2, centroid, invariant-area runner 자리.
- BDF 유효 방출률의 negative/non-finite와 해의 negative/non-finite를 clamp 없이 오류 record로 전파.

### 4.2 최초 실패

첫 격자 `(Ns,Nnu)=(32,128)`에서 solver가 profile acceptance를 계산하기 전에 다음 오류로 종료했다.

| 항목 | 값 |
|---|---:|
| status | `LCMF_ENEGATIVE` |
| evaluation/ray/segment | 0 / 0 / 47 |
| frequency index | 98 |
| x=ln(nu) | -0.2510236220472441 |
| nu | 0.7780039932948063 |
| I(k-1) | 1.1251344515546073e-7 |
| I(k-2) | 5.0547558929601476e-7 |
| 2 I(k-1) - 0.5 I(k-2) | -2.7710904337085918e-8 |
| BDF effective eta | **-4.7557903389322746e-17** |

이 값은 roundoff 부호 반전이 아니라, 기록된 두 history term을 설계 식에 그대로 대입하면 재현된다. coarse `dx=0.74/127=0.005826771653543307`, `a=A/L=1e-11 cm^-1`이므로

```text
(a/dx) * [2 I(k-1) - 0.5 I(k-2)]
= -4.755790338932314e-17.
```

설계 §2.2는 BDF2 유효 방출률이 음수가 되면 해당 좌표와 세 항을 기록하고 `LCMF_ENEGATIVE`로 종료하도록 명시한다. 따라서 다음 행위는 하지 않았다.

- 음수를 0으로 clamp하거나 작은 값으로 floor.
- Gaussian tail cell 제외 또는 domain 변경.
- positivity limiter, scheme 교체, acceptance 변경.
- 실패 뒤 64x256/128x512 실행.

KA3는 **FAIL**이며 machine-readable 결과는 `docs/s31_results/ka3.json`, 로그는 `docs/s31_logs/rung3_ka3.log`다.

## 5. rung4/rung5 상태

rung3 실패가 terminal gate이므로 KA2 Nyström oracle과 무가속 coherent-scattering fixed point에는 진입하지 않았다. 따라서 `patches/s31_rung4.patch`와 `patches/s31_rung5.patch`를 성공 rung처럼 생성하지 않았다. 빈 패치나 NOT-RUN marker를 적용 가능한 구현 패치로 위장하지 않는 것이 이번 사다리 규율에 맞는다.

## 6. 패치 사다리

| 단계 | 패치 | SHA-256 | 상태 |
|---:|---|---|---|
| 기존 rung1 | `patches/s31_rung1_skeleton.patch` | 기존 산출물 | 기준 |
| 기존 rung2 | `patches/s31_rung2_ka1_pure_absorption.patch` | 기존 산출물 | KA1 구 구현 |
| rung2R | `patches/s31_rung2R.patch` | `4cabfa4faed895a0c7a6a8b0f0082a8ebc23dcd56d826f3f0a98fc931e85318e` | **PASS** |
| rung3 | `patches/s31_rung3.patch` | `2d2b074c90cdec4222762678dce59d93a6e56200f9a9bbec04c2ab8bca6b12b6` | **FAIL / STOP** |
| rung4 | 생성 안 함 | — | NOT RUN |
| rung5 | 생성 안 함 | — | NOT RUN |

기존 rung1+rung2 상태를 임시 트리에 재구성한 뒤 rung2R, rung3를 차례로 `patch -p1` 적용했고, 최종 header/C/driver/runner가 작업 트리와 byte-identical임을 `diff -q`로 확인했다. 재구성 트리의 strict self-test도 PASS했다.

## 7. 재현 명령

### 7.1 현재 작업 트리

```bash
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow \
  -Isrc tests/stage31_cmf_skeleton_selftest.c src/lumina_cmf_field.c -lm \
  -o /tmp/stage31_skeleton_round2
/tmp/stage31_skeleton_round2

python3 scripts/run_stage31_cmf_ka.py --ka ka1 \
  --work /tmp/stage31_round2_ka1 \
  --output /tmp/stage31_round2_ka1.json
# expected exit 0, status PASS

python3 scripts/run_stage31_cmf_ka.py --ka ka3 \
  --work /tmp/stage31_round2_ka3 \
  --output /tmp/stage31_round2_ka3.json
# expected exit 1, status FAIL, first failure at k=98/ray=0/segment=47
```

### 7.2 패치 사다리 재생

```bash
# 기존 rung1+rung2가 적용된 tree에서
patch -p1 < patches/s31_rung2R.patch
python3 scripts/run_stage31_cmf_ka.py --ka ka1 \
  --work /tmp/stage31_rung2R --output /tmp/stage31_rung2R.json

patch -p1 < patches/s31_rung3.patch
python3 scripts/run_stage31_cmf_ka.py --ka ka3 \
  --work /tmp/stage31_rung3 --output /tmp/stage31_rung3.json
```

## 8. 산출물 목록

- 전체 보고서: `docs/CODEX_STAGE31_IMPL2.md`
- rung2R patch: `patches/s31_rung2R.patch`
- rung3 patch: `patches/s31_rung3.patch`
- KA1 JSON/log: `docs/s31_results/ka1_rev1.json`, `docs/s31_logs/rung2R_ka1_rev1.log`
- KA3 JSON/log: `docs/s31_results/ka3.json`, `docs/s31_logs/rung3_ka3.log`
- 수정된 Stage 3.1 신규 solver/API: `src/lumina_cmf_field.c`, `src/lumina_cmf_field.h`
- runner/driver: `scripts/run_stage31_cmf_ka.py`, `scripts/stage31_cmf_ka_driver.c`
- fail-closed regression: `tests/stage31_cmf_skeleton_selftest.c`
