# Codex A-S31 round 7C — S31 트리 감사·인증 상태 복원 + 판별 벤치

날짜: 2026-08-01  
상태: **KA1/KA3/KA2 PASS; 판별 벤치 UNRESOLVED / STOP AT `J_det`**  
물리 판독: **UNRESOLVED-SOLVER-GUARD**

## 0. 총결론

S31 패치 사다리를 clean-room에서 1→2→2R→3→4→5→6→7→8→9→10→11 순서로 전부 재생했고 모든 patch가 순차 적용됨을 확인했다. 감사 시점의 활성 트리는 rung1–5와 8–9의 산출물을 보유했지만, 생산 solver의 rung6, 독립 oracle 파일의 rung10, 생산/runner의 rung11이 빠져 있었다. rung7은 자기 hunk는 runner에 있으나 필수 선행 rung6 생산 solver가 없어 **부분/불일치** 상태였다.

승인대로 신규 설계 없이 누락분을 6→10→11 순서로 복원했다. 7B의 양끝 로그-face 외삽은 유지했다. strict C11, skeleton, KA1, KA3, KA2의 최종 10R 판정은 모두 PASS다. KA3는 5B의 `p=2.0005918699`, fine L1 `2.64039438e-5`를 그대로 재현했고, KA2는 로그-face 유지 영향으로 `p=1.757385`가 되어 5B의 `1.769543`에서 `-0.012158` 이동했지만 원문 `[1.7,2.3]`과 모든 수치 문턱을 통과했다.

판별 벤치는 로그 face 자체는 통과했다. 그러나 quadratic-exact 경로가 production payload의 첫 sweep `frequency=2, ray=1, segment=32`에서 값과 interval 상한이 모두 음수인 solution을 검출했다. acceptance 완화나 clamp 없이 fail closed했으므로 `J_det`가 생성되지 않았고, §7.2의 candidate 열과 §7.3 Γ D-lane 및 수송/χ,η 이분 판독은 UNRESOLVED다. 동일 실패 stderr는 독립 3회 SHA-256까지 일치했다. 이것이 7C의 정직한 최종 stop이다.

## 1. 트리 상태 전수 감사

### 1.1 판정 방법

각 patch에 대해 감사 전 활성 트리에서 `git apply --check`와 `git apply --reverse --check`를 모두 실행했다. `0`은 check 성공, `1`은 실패다. 후속 rung가 같은 hunk를 바꾸면 양방향 check가 모두 실패할 수 있으므로, clean-room 순차 재생, 단계별 SHA-256 계보, 고유 심볼/파일 grep을 함께 사용했다.

| rung | 실제 patch | 정방향/역방향 rc | 감사 전 핵심 증거 | 감사 전 판정 | 7C 조치 |
|---:|---|---:|---|---|---|
| 1 | `s31_rung1_skeleton.patch` | `1/1` | API, `LCMFCE01` reader, skeleton 존재; 후속 hunk로 exact reverse 불가 | **반영(후속 대체)** | 유지 |
| 2 | `s31_rung2_ka1_pure_absorption.patch` | `1/1` | KA1 driver/runner와 80-dps oracle 경로 존재 | **반영(2R로 대체)** | 유지 |
| 2R | `s31_rung2R.patch` | `1/1` | diffusion/irradiation BC, frequency-advection 계약과 KA1 창 존재 | **반영(후속 대체)** | 유지 |
| 3 | `s31_rung3.patch` | `1/1` | KA3 BDF/redshift driver 및 runner 존재 | **반영(후속 대체)** | 유지 |
| 4 | `s31_rung4.patch` | `1/1` | signed SC, `LCMF_ESIGNUNCERTAIN`, solution guard 존재; 활성 source 계보가 rung4 SHA `9a9e781...` | **반영** | 유지 |
| 5 | `s31_rung5.patch` | `1/1` | Gaussian cell-average oracle와 prereg window 판독 존재 | **반영(후속 대체)** | 유지 |
| 6 | `s31_rung6.patch` | `0/1` | `quadratic_moments`, `branch_quadratic_stencil`, quadratic SC prototype/시험 전부 부재 | **누락** | **복원** |
| 7 | `s31_rung7.patch` | `1/1` | 5-grid/official triple/rev3 runner hunk는 존재하나 선행 rung6 생산 solver 부재 | **부분/불일치** | rung6 복원으로 정합화 |
| 8 | `s31_rung8.patch` | `1/1` | 두 MPFR 파일 존재; rung9 최종본과 byte-identical | **반영(9가 수정)** | 유지 |
| 9 | `s31_rung9.patch` | `1/0` | 역체크 exact PASS; 4096 grid/certificate merge 존재 | **반영** | 유지 |
| 10/10R | 실제 파일은 `s31_rung10.patch` | `0/1` | patch가 추가하는 `stage31_ka2_oracle.py`, `run_stage31_ka2_oracle.py` 부재. 별도 10R HP script/result는 존재 | **patch-managed rung10 누락** | **복원**, 기존 10R HP 인증 유지 |
| 11 | `s31_rung11.patch` | `1/1`; rung6 복원 뒤 `0/1` | KA2 driver/runner, plain source iteration, coherent selftest 부재; 선행 rung6 복원 뒤 clean apply | **누락(선행 의존으로 최초 check 가림)** | **복원** |

`patches/s31_rung10R.patch`라는 파일은 실제로 존재하지 않는다. 차터의 10R은 `patches/s31_rung10.patch`와 후속 `scripts/s31_ka2_oracle_hp.py`/`scripts/s31_ka2_judge.py` 및 HP 결과를 합친 인증 상태로 해석했다. 존재하지 않는 patch를 만들어내지 않았다.

### 1.2 누락 시점/원인 추정

강한 증거는 “후대 삭제”가 아니라 “격리 구현의 활성 승격 누락”을 가리킨다.

- clean-room source SHA는 rung4와 rung5에서 `9a9e781602ed...`, rung6 뒤 `49084880604c...`, rung11 뒤 `6e364e7f9a53...`다.
- 5B 보고서는 당시 활성 `src/lumina_cmf_field.c`가 작업 전후 `9a9e781602ed...`이고 production `src/` 수정이 0이라고 명시한다.
- 같은 보고서는 rung6 이후 구현을 `impl_s31_round5b/` 격리본에서 검증하고 patch로 납품했다고 명시한다.
- 감사 전 활성 source는 7B 로그 hunk를 제외하면 rung4/5 계보였고, `impl_s31_round5b/`에는 rung6·11 최종본이 남아 있었다. 반면 runner/MPFR 일부는 후속본이었다.

따라서 유실 지점은 round 4의 rung6 격리 구현 시점부터 5B patch 납품 사이의 **활성 트리 승격 단계**로 추정한다. 즉 rung6/10/11이 활성에서 삭제됐다기보다 처음부터 활성에 통합되지 않았고, 일부 runner만 따로 전진해 split-brain이 생겼다.

## 2. 인증 상태 복원

복원 순서는 `rung6 → rung10 → rung11`이다. 각 단계는 기존 patch 내용만 적용했으며 신규 solver 수식은 추가하지 않았다. 복원 후 rung10과 rung11은 exact 역체크가 통과했고, rung6은 후속 rung11이 같은 말단 hunk를 수정하므로 독립 exact reverse 대신 고유 심볼·strict build·clean-room 최종 diff로 확인했다.

7B 로그 외삽 보존 확인:

- `src/lumina_cmf_field.c`에는 inner/outer `exp(log(v0)+...)` 및 양수 stencil fail-closed 메시지가 남아 있다.
- 7B의 steep-positive PASS, mixed-zero inner/outer fail-closed selftest가 남아 있다.
- 복원 source와 `impl_s31_round5b` source의 diff는 `radial_value()` 로그-face hunk뿐이다(`3 insertions / 34 deletions`, current→5B 방향).
- 복원 source SHA-256: `cae7d1208a919efb42b1076ad641c7f4c62c443f1b7269d31a6997a4611b3753`.

| 검증 | 결과 |
|---|---|
| Python runner/harness `py_compile` | PASS |
| strict C11 `-Wconversion -Wshadow` | PASS |
| 확장 skeleton + 로그-face + quadratic + coherent selftest | PASS |
| clamp/floor/acceptance 변경 | 0 |
| 신규 모델/GPU run | 0 |
| commit | 0 |

## 3. KA 전 배터리 재인증

### 3.1 KA1 — PASS

| χR | 원문 `p_obs` 창 | 7C `p_obs` | fine I rel L2 | fine J rel L2 | fine max scaled | fine residual | 판정 |
|---:|---:|---:|---:|---:|---:|---:|---|
| `1e-3` | `[1.90,2.06]` | `1.9905731831` | `3.335975530e-7` | `3.362414859e-7` | `1.422942880e-9` | `1.070031154e-5` | PASS |
| `1` | `[1.90,2.08]` | `1.9912927418` | `3.498958543e-7` | `3.577613570e-7` | `1.318450184e-6` | `5.946114788e-6` | PASS |
| `100` | `[1.85,2.15]` | `2.0221547417` | `4.240575734e-7` | `4.431357905e-6` | `1.170793363e-4` | `1.967905971e-7` | PASS |

모든 I/J L2, max, generic p, residual, boundary 및 guard 문턱을 통과했다. clamp/negative/sign-uncertain/non-finite는 모두 0이다. 3회 결정론 SHA-256은 `6d00719782a87edd65fe4a67ab7648c416afdd909ba6933d8d6d2a81f3f83ddb`로 동일하다. JSON SHA는 7B와 같은 `1bd5215f...`로 수치와 직렬화까지 일치한다.

### 3.2 KA3 — PASS

| official grid | profile L1 | profile L2 | centroid error | invariant error | residual | certified neg/uncertain/nonfinite |
|---:|---:|---:|---:|---:|---:|---:|
| `256x1024` | `4.229785243e-4` | `3.835010006e-4` | `1.330675452e-8` | `1.331483328e-8` | `2.655015175e-11` | `0/0/0` |
| `512x2048` | `1.056589331e-4` | `9.580149042e-5` | `3.338085808e-9` | `3.339029448e-9` | `3.103209724e-11` | `0/0/0` |
| `1024x4096` | `2.640394380e-5` | `2.394054891e-5` | `8.359514886e-10` | `8.360132333e-10` | `3.578321929e-11` | `0/0/0` |

공식 `p_obs(L2)=2.000591869903567`, fine L1 원문 창 `[2.50e-5,2.80e-5]`, L1/L2 `≤1e-4`, centroid/area/residual, boundary, clamp 및 MPFR certificate 문턱을 전부 통과했다. 이는 5B의 인증 중심 `p=2.0006`, L1 `2.64e-5`와 계산값까지 일치한다.

### 3.3 KA2 + 10R — 최종 PASS

| grid | J oracle rel L2 | max error | iterations | source residual | transport residual | energy closure |
|---:|---:|---:|---:|---:|---:|---:|
| `128x32` | `1.274737109e-5` | `2.827346507e-6` | 32 | `4.082756580e-13` | `3.051425430e-7` | `9.892940604e-6` |
| `256x64` | `3.738333544e-6` | `8.928692654e-7` | 32 | `4.088829746e-13` | `1.225514452e-6` | `3.695223921e-6` |
| `512x128` | `1.073274278e-6` | `2.801350461e-7` | 32 | `4.088816629e-13` | `8.127498024e-6` | `1.232214348e-6` |

수송 수치의 `p_obs(J)=1.7573850206761803`이며 원문 `[1.7,2.3]` 안이다. 모든 수치 gate와 counter 0을 통과했다. 구 rung10 runner JSON은 의도적으로 `binary64 matrix_storage` 때문에 status FAIL을 기록한다. 최종 판정은 기존 full-80-digit 10R oracle과 strict judge를 결합한 결과다.

| 10R 항목 | 실측 | 판정 |
|---|---:|---|
| end-to-end mpmath 80 dps arithmetic audit | 전 항 true | PASS |
| Nref 2048/4096 relative L2 | `3.6453307657433484e-10 < 1e-9` | PASS |
| 새 7C solver numeric battery | 전 항 true | PASS |
| strict judge overall | `PASS` | **PASS** |

5B 정합도: `p`는 `1.7695427052 → 1.7573850207`(`-0.687%`), fine L2는 `1.079129704e-6 → 1.073274278e-6`(`-0.543%`)이다. 7C source가 5B source와 다른 유일한 생산 hunk는 유지 승인된 로그-face 외삽이므로, 이 작은 이동은 그 경계 처리의 영향으로 해석한다. 이는 source diff와 수치의 결합에 근거한 추정이다.

## 4. 판별 벤치

### 4.1 결론

요청된 수송 결함 대 χ,η 내용 결함의 이분 판정은 내릴 수 없다. 인증 payload와 로그 radial-face 외삽은 정상이나, 정본 solver가 첫 sweep의 solution guard에서 fail closed했다. 이를 clamp하거나 tolerance로 무시하면 acceptance를 바꾸므로 수행하지 않았다.

실제 첫 실패는 `deterministic solve failed: LCMF_ENONFINITE: solution interval upper bound is negative radial=0 frequency=2 ray=1 segment=32 substep=0 value=-1.5420218010268406e-68 interval=[-1.5420218010334412e-68,-1.54202180102024e-68]`이다. 실행 시간은 4.118 s로, 장시간 계산 때문에 중단한 것이 아니다.

`record_solution_guard()` 자체는 이 경우 `LCMF_ENEGATIVE`를 기록하고 `solution_negative_excess_count`를 올린다. 최종 solver 반환은 누적 `nonfinite_count`를 negative보다 먼저 검사하므로 status string은 `LCMF_ENONFINITE`, 보존된 첫 상세 메시지는 negative solution guard다. 둘 중 어느 것도 acceptance상 허용되지 않는다. 독립 3회 stderr는 모두 SHA-256 `8451c3e1dd520b520da0e78fbcc439788bc9d5ac671463dff00fd74cb0079d0f`로 동일했다.

### 4.2 입력 독립 검증

- checker: `PASS: iteration=10 field_generation=10 post_damp=1 bytes=2416472`
- payload SHA-256: `94d75988034454f55fb6b130f04521f01c56f875cb22ef3a711850d7382ffa2f`; sidecar와 일치
- schema: 50 shell × 1000 bin, iter=10, generation=10, post_damp=1
- candidate/input ν grid max relative identity error: `1.221e-15`
- inner boundary: producer `cmf_solve_J`와 같은 explicit `Bν(T_inner=10020 K)` irradiation, amplitude scale 1.0; diffusion gradient를 추정하지 않음

#### radial face 사전검사

| field | face | log invalid/negative | 600–3000 Å invalid/negative | log minimum | legacy linear negative | first bad bin / Å |
|---|---|---:|---:|---:|---:|---:|
| chi_total | inner | 0 | 0 | 3.634926484e-14 | 0 | — |
| chi_total | outer | 0 | 0 | 2.330578153e-20 | 40 | — |
| eta_total | inner | 0 | 0 | 1.499574103e-34 | 0 | — |
| eta_total | outer | 0 | 0 | 1.505617270e-54 | 37 | — |

### 4.3 §7.2 s8 Jν 3중 대조

J_MC는 계약대로 sidecar payload의 `J_producer`다. CMFGEN은 RVTJ의 9610.017–10163.506 km/s 사이 log-J velocity interpolation 후 공통 1000-bin edge에 적분보존 평균했다. point interpolation은 쓰지 않았다.

| band [Å] | J_det/J_MC | J_det/J_CMFGEN | J_MC/J_CMFGEN | log10(det/MC) | log10(det/CMFGEN) | log10(MC/CMFGEN) | toward CMFGEN |
|---|---:|---:|---:|---:|---:|---:|---|
| B0 600–1000 | UNRESOLVED | UNRESOLVED | 33.764 | UNRESOLVED | UNRESOLVED | 1.52845 | UNRESOLVED |
| B1 1000–1500 | UNRESOLVED | UNRESOLVED | 32.3231 | UNRESOLVED | UNRESOLVED | 1.50951 | UNRESOLVED |
| B2 1500–2000 | UNRESOLVED | UNRESOLVED | 7.37579 | UNRESOLVED | UNRESOLVED | 0.867808 | UNRESOLVED |
| B3 2000–2500 | UNRESOLVED | UNRESOLVED | 6.91223 | UNRESOLVED | UNRESOLVED | 0.839618 | UNRESOLVED |
| B4 2500–3000 | UNRESOLVED | UNRESOLVED | 16.2922 | UNRESOLVED | UNRESOLVED | 1.21198 | UNRESOLVED |
| BALL 600–3000 | UNRESOLVED | UNRESOLVED | 11.9771 | UNRESOLVED | UNRESOLVED | 1.07835 | UNRESOLVED |

#### spectral norm

| band | pair | median log10 ratio | p10 | p90 | positive pairs | zero/excluded |
|---|---|---:|---:|---:|---:|---:|
| B0 | det/MC | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B0 | det/CMFGEN | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B0 | MC/CMFGEN | +1.489393 | +1.224499 | +1.658454 | 97 | 0 |
| B1 | det/MC | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B1 | det/CMFGEN | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B1 | MC/CMFGEN | +1.547098 | +1.077093 | +1.821144 | 76 | 0 |
| B2 | det/MC | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B2 | det/CMFGEN | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B2 | MC/CMFGEN | +0.828737 | +0.772125 | +1.060727 | 55 | 0 |
| B3 | det/MC | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B3 | det/CMFGEN | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B3 | MC/CMFGEN | +0.780508 | +0.739546 | +0.858911 | 42 | 0 |
| B4 | det/MC | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B4 | det/CMFGEN | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B4 | MC/CMFGEN | +0.984745 | +0.711919 | +1.507408 | 34 | 0 |
| BALL | det/MC | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| BALL | det/CMFGEN | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| BALL | MC/CMFGEN | +1.258953 | +0.771584 | +1.689085 | 304 | 0 |

candidate가 들어가는 spectral quantile은 두 양수 field가 모두 양수인 bin만 사용했으며, 제외 수를 그대로 기록했다.

### 4.4 §7.3 Γ D-lane

기존 `w3_gamma_triple_compare.py`의 grid/C1/C2 loader, EDDFACTOR/RVTJ 적분보존 평균, within-SL fraction·σ·threshold·route 및 `4πσJ/(hν)` quadrature를 import해 재사용했다.

| target | Γ_MC B [s⁻¹] | Γ_CMFGEN C [s⁻¹] | Γ_det D [s⁻¹] | D/B | D/C | log10(D/C) | log10(B/C) | toward CMFGEN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fe III C48 lump (idx 201) | 4.363858095e+02 | 2.807174861e+01 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | +1.191601 | UNRESOLVED |
| S II SL4 (idx 4) | 3.310150073e+01 | 4.748076950e-01 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | +1.843330 | UNRESOLVED |

Fe III와 S II의 D-lane은 동일한 σ·threshold·route·within-SL fraction에 `J_det`만 대입하도록 준비됐지만, `J_det` 부재로 실제 D 값은 계산하지 않았다. B/C baseline을 D처럼 재사용하지 않았다.

### 4.5 Acceptance와 사전등록 판독

- sidecar/schema/checksum/epoch: **PASS**
- CMFGEN integral conservation: `1.000000000000000` (**PASS**)
- 6 band row와 2 Γ row의 baseline provenance: **PASS**
- candidate transport residual ≤1e-4, finite/nonnegative, clamp=0: **UNRESOLVED** (solver guard가 장 생성 전에 fail closed)
- candidate 성공-output SHA-256: **NOT RUN** (장 생성 전 fail closed); 실패 stderr 3회 identity: **PASS** (`8451c3e1dd520b520da0e78fbcc439788bc9d5ac671463dff00fd74cb0079d0f`)
- 최종 bench acceptance: **UNRESOLVED**

최종 사전등록 판독은 **UNRESOLVED-SOLVER-GUARD**이며 방향 자체는 acceptance를 변경하지 않는다.

## 5. 전 수치 재현 명령

```bash
# 감사 전 양방향 check
for f in patches/s31_rung1_skeleton.patch \
         patches/s31_rung2_ka1_pure_absorption.patch \
         patches/s31_rung2R.patch patches/s31_rung3.patch \
         patches/s31_rung4.patch patches/s31_rung5.patch \
         patches/s31_rung6.patch patches/s31_rung7.patch \
         patches/s31_rung8.patch patches/s31_rung9.patch \
         patches/s31_rung10.patch patches/s31_rung11.patch; do
  git apply --check "$f"; echo "forward=$? $f"
  git apply --reverse --check "$f"; echo "reverse=$? $f"
done

# 감사 전 tree에서 인증 상태 복원 재생 순서
git apply --check patches/s31_rung6.patch
git apply patches/s31_rung6.patch
git apply --check patches/s31_rung10.patch
git apply patches/s31_rung10.patch
git apply --check patches/s31_rung11.patch
git apply patches/s31_rung11.patch

python3 -m py_compile scripts/run_stage31_cmf_ka.py \
  scripts/run_stage31_ka2_oracle.py scripts/stage31_ka2_oracle.py \
  scripts/stage31_cmf_field_bench.py

gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -Wconversion -Wshadow -Isrc \
  tests/stage31_cmf_skeleton_selftest.c src/lumina_cmf_field.c -lm \
  -o /tmp/stage31_skeleton_7c
/tmp/stage31_skeleton_7c

python3 scripts/run_stage31_cmf_ka.py --ka ka1 \
  --work /tmp/stage31_round7c_ka1 \
  --output docs/s31_results/ka1_round7c.json

python3 scripts/run_stage31_cmf_ka.py --ka ka3 \
  --certificate docs/s31_results/mpfr_cert_rung8_fine.json \
  --work /tmp/stage31_round7c_ka3 \
  --output docs/s31_results/ka3_round7c.json

# expected exit 1: 구 rung10의 binary64-arithmetic 자격 항목만 false
python3 scripts/run_stage31_cmf_ka.py --ka ka2 \
  --work /tmp/stage31_round7c_ka2 \
  --output docs/s31_results/ka2_round7c.json

# expected exit 0: full-80-digit 10R 자격을 결합한 최종 KA2 판정
python3 scripts/s31_ka2_judge.py \
  --oracle docs/s31_results/ka2_oracle_hp.json \
  --solver docs/s31_results/ka2_round7c.json \
  --binary64-oracle docs/s31_results/ka2_oracle_rung10.json \
  --out docs/s31_results/ka2_round7c_judge_hp.json

sha256sum /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10
python3 scripts/cmf_chieta_check.py /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -D_POSIX_C_SOURCE=200809L -Isrc scripts/stage31_cmf_field_driver.c \
  src/lumina_cmf_field.c -lm -o /tmp/stage31_cmf_field_driver
/tmp/stage31_cmf_field_driver /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10 /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10.manifest.json 8 16 10020 1 /tmp/stage31_jdet.tsv
# expected exit 3: 보고서/JSON 생성 완료, J_det solver guard UNRESOLVED
python3 scripts/stage31_cmf_field_bench.py --frozen /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10 \
  --driver-table docs/s31_results/stage31_jdet_s8_round7c.tsv \
  --report docs/CODEX_STAGE31_BENCH7C_REPLAY.md \
  --status-json docs/s31_results/stage31_bench_round7c_replay.json
```

## 6. 산출물과 SHA-256

| 산출물 | SHA-256 | 상태 |
|---|---|---|
| `src/lumina_cmf_field.c` | `cae7d1208a919efb42b1076ad641c7f4c62c443f1b7269d31a6997a4611b3753` | rung6/11 + 7B 로그 face |
| `src/lumina_cmf_field.h` | `960b02c6924c0341ad4c11a88f98ceb3dee3b8d77ca6e8b332aa8c2035748bfd` | restored |
| `scripts/run_stage31_cmf_ka.py` | `b5345c4310be763954f0cca3334bde8776c2487e15c92ee45f056014dbbee14b` | restored |
| `scripts/stage31_cmf_field_bench.py` | `267e1f3c8372776f4f3436b5c48ea0ab6486561b37d8aa134e1d05bc95025cb2` | unresolved 렌더 fail-closed 수리 |
| `docs/s31_results/ka1_round7c.json` | `1bd5215f0241227a7682969c3263c3fcb94c39e6506dacc905ac2712e480a1a1` | PASS |
| `docs/s31_results/ka3_round7c.json` | `37830436e3dd92f137b1b936ff1f1757d194d3a85e4322656994f3cd8419fb6d` | PASS |
| `docs/s31_results/ka2_round7c.json` | `a01a146a0de4cdcf517b71987cb56f23a809d9aa4ff402b36706c031dfaa9002` | numeric PASS / legacy oracle flag FAIL |
| `docs/s31_results/ka2_round7c_judge_hp.json` | `8acd9afb31a5ac05b7d69aaac90c4429970e5c7e8286fdd809d4561b59205fc0` | final PASS |
| `docs/s31_results/stage31_bench_round7c.json` | `6c9111d092aea9d7d6724b5d8d6a69f45e0dc0b7593bbd0870db1560478882e4` | UNRESOLVED-SOLVER-GUARD |

`stage31_jdet_s8_round7c.tsv`는 solver가 장 직렬화 전에 fail closed했으므로 존재하지 않는다. 빈/부분 파일을 `J_det` 산출물로 남기지 않았다.

신규 모델/GPU run, acceptance 변경, clamp/floor, 커밋은 없었다.
