# Codex A-S31 round 7D — 절단오차 계층화 및 판별 벤치

날짜: 2026-08-01  
상태: **가드·KA1·KA3·KA2·판별 bench PASS**  
물리 판독: **UNRESOLVED-MIXED — 대역·Γ별 분해**

## 0. 총결론

7C를 막은 `frequency=2`의 인증 음수는 값 수정 없이 sub-truncation으로 기록됐고, 정본 `J_det` solve가 완료됐다. 가드는 실제 비유한 해/스케일을 즉시 fail하고, 인증 음수의 `|I|`가 `B_trunc=C h² S_loc`보다 클 때만 fail-closed한다. 그 이하에서는 원래 음수 bit pattern을 그대로 다음 연산에 전달한다. clamp/floor는 0이다.

`C=27.80641753160013`은 7D에서 재실행한 KA 수렴 자료의 `max(E_relative/h²)`이며 안전배수는 정확히 1.0이다. seeded 대규모 음수 `-0.1`은 `LCMF_ENEGATIVE`, seeded sub-truncation `-1e-4`는 `LCMF_OK`와 raw-bit identity를 재현했다. KA1·KA3의 모든 물리 수치와 KA2 수송 수치는 7C와 동일했고 최종 상태는 세 battery 모두 PASS다.

판별 bench도 PASS했다. 다만 `J_det`는 600–3000 Å 적분에서 `J_MC`의 97.7181%로 사실상 MC UV 과잉을 재현한다. 6개 대역 중 B2만 CMFGEN 반대 방향이고, Γ는 Fe III가 반대(`D/B=1.5713`), S II가 CMFGEN 방향(`D/B=0.521386`)이다. 따라서 사전등록 판독은 한쪽 단일 원인이 아니라 **파장·이온 의존 혼합 원인**이다. 수송 단독 결함 판정은 기각되고, χ,η 내용 결함이 전체 UV 수준을 지배하되 B2·Fe III에는 수송/원자경로 성분이 따로 남는다.

신규 모델/GPU run, acceptance 수치 문턱 변경, clamp/floor, 커밋은 없었다.

## 1. 해-부호 가드 계층화

### 1.1 정의와 C의 비임의 유도

`h=max(Δr_loc/(r_out-r_in), Δlnν/ln(ν_max/ν_min))`로 KA3와 같은 무차원 격자 폭을 쓴다. `S_loc`는 같은 ray의 인접 BDF 주파수면(`k-2,k-1`) 전체 path와 현재 substep 이웃의 `max|I|`이며, 전역 fallback/floor는 전 주파수의 inner-boundary와 `η/χ` 최대 해 스케일이다.

| battery | 계수 정의 | 실측 최대 |
|---|---|---:|
| KA1 | `I relative L2 × n_mu²` | `0.007085203760632297` |
| KA2 | `J oracle relative L2 × n_mu²` | `0.017584525768328887` |
| KA3 | `profile relative L1 × n_s²` | **`27.80641753160013`** |

최댓값은 KA3 `64×256`의 `0.006788676155175813 × 64²`이다. 따라서 `C=27.80641753160013`; 별도 여유계수는 곱하지 않았다. 전체 표와 산출식은 `docs/s31_results/guard_calibration_round7d.json`에 있다.

가드 순서는 다음과 같다.

1. 실제 solution value 또는 `h/S/B_trunc`가 비유한이면 즉시 `LCMF_ENONFINITE`.
2. 구간 상한까지 음수이고 `|I|>B_trunc`이면 즉시 `LCMF_ENEGATIVE`.
3. 인증 음수이나 `|I|≤B_trunc`이면 count/min/first/min-coordinate/interval/scale/h/bound를 기록하고 raw 값을 수정하지 않고 진행.
4. 0을 가로지르는 구간도 중심값이 `B_trunc` 아래일 때만 별도 sign-indeterminate sub-truncation으로 기록한다. 그 밖의 sign-uncertain은 기존처럼 fail-closed다.

유한 중심값의 binary64 구간 외피만 의존성 과대평가로 `±inf`가 될 때는 중심값의 양 옆 1 ulp에서 외피를 재시작하고 횟수를 공개한다. 실제 값이 비유한 경우에는 이 경로를 타지 않고 즉시 fail한다. 이는 solution 값 수정이나 clamp가 아니다.

### 1.2 seeded 음성 시험

| seed/class | I | h | S | B_trunc | status/count | 값 수정 |
|---|---:|---:|---:|---:|---|---|
| `0x7d01` excess | `-0.1` | `1/64` | `1` | `6.788676155175813e-3` | `LCMF_ENEGATIVE`, excess=1 | 없음 |
| `0x7d02` sub-truncation | `-1e-4` | `1/64` | `1` | `6.788676155175813e-3` | `LCMF_OK`, sub=1 | **bit-identical** |

별도 `NaN` probe도 `LCMF_ENONFINITE`, nonfinite=1로 즉시 종료한다. 좌표 seed는 `evaluation/frequency/ray/segment/substep=7/11/13/17/19`로 고정했다.

## 2. KA 배터리 재실행

### 2.1 KA1 — PASS, 수치 불변

| χR | p_obs 7C | p_obs 7D | fine I rel L2 | fine J rel L2 | fine residual |
|---:|---:|---:|---:|---:|---:|
| `1e-3` | `1.9905731831359303` | `1.9905731831359303` | `3.335975530e-7` | `3.362414859e-7` | `1.070031154e-5` |
| `1` | `1.991292741751914` | `1.991292741751914` | `3.498958543e-7` | `3.577613570e-7` | `5.946114788e-6` |
| `100` | `2.0221547417133876` | `2.0221547417133876` | `4.240575734e-7` | `4.431357905e-6` | `1.967905971e-7` |

모든 guard/sub-truncation/nonfinite counter는 0이다. 3회 테이블 SHA는 진단 열 추가 때문에 7C와 달라졌지만 7D 내부에서는 `225e537aa5468e14bd9329c8d0d05e5557fb998e5bfa99b47593297a623df560`으로 동일하다.

### 2.2 KA3 — PASS, 수치 불변

| official grid | L1 | L2 | p_obs | certified neg/uncertain/nonfinite | sign-indet sub / enclosure restart |
|---:|---:|---:|---:|---:|---:|
| `256×1024` | `4.229785243e-4` | `3.835010006e-4` | — | `0/0/0` | `242152 / 0` |
| `512×2048` | `1.056589331e-4` | `9.580149042e-5` | — | `0/0/0` | `1019770 / 3` |
| `1024×4096` | `2.640394380e-5` | `2.394054891e-5` | `2.000591869903567` | `0/0/0` | `3708285 / 436329` |

profile, centroid, invariant, residual과 MPFR certificate는 7C와 계산값까지 동일하다. binary64 진단 계층의 counter만 새 의미로 노출됐다.

### 2.3 KA2 + 10R — 최종 PASS, 수치 불변

| grid | J oracle rel L2 | max error | iterations | source residual | transport residual | energy closure |
|---:|---:|---:|---:|---:|---:|
| `128×32` | `1.274737109e-5` | `2.827346507e-6` | 32 | `4.082756580e-13` | `3.051425430e-7` | `9.892940604e-6` |
| `256×64` | `3.738333544e-6` | `8.928692654e-7` | 32 | `4.088829746e-13` | `1.225514452e-6` | `3.695223921e-6` |
| `512×128` | `1.073274278e-6` | `2.801350461e-7` | 32 | `4.088816629e-13` | `8.127498024e-6` | `1.232214348e-6` |

`p_obs=1.7573850206761803`, 모든 수송 counter=0이며 7C와 동일하다. 구 binary64 runner의 의도된 oracle 자격 false 때문에 raw JSON은 FAIL이지만, full-80-digit 10R의 `3.6453307657433484e-10<1e-9`와 strict judge를 결합한 최종 `overall_status`는 **PASS**다.

## 3. 판별 bench

### 3.1 입력 독립 검증

- checker: `PASS: iteration=10 field_generation=10 post_damp=1 bytes=2416472`
- payload SHA-256: `94d75988034454f55fb6b130f04521f01c56f875cb22ef3a711850d7382ffa2f`; sidecar와 일치
- schema: 50 shell × 1000 bin, iter=10, generation=10, post_damp=1
- candidate/input ν grid max relative identity error: `1.221e-15`
- inner boundary: producer `cmf_solve_J`와 같은 explicit `Bν(T_inner=10020 K)` irradiation, amplitude scale 1.0; diffusion gradient를 추정하지 않음

### radial face 사전검사

| field | face | log invalid/negative | 600–3000 Å invalid/negative | log minimum | legacy linear negative | first bad bin / Å |
|---|---|---:|---:|---:|---:|---:|
| chi_total | inner | 0 | 0 | 3.634926484e-14 | 0 | — |
| chi_total | outer | 0 | 0 | 2.330578153e-20 | 40 | — |
| eta_total | inner | 0 | 0 | 1.499574103e-34 | 0 | — |
| eta_total | outer | 0 | 0 | 1.505617270e-54 | 37 | — |

### 해-부호 가드 통계

- certified-negative sub-truncation: `90761`; minimum `-0.00016493418433066037`
- minimum coordinate: frequency `709`, ray `15`, segment `47`, substep `0`
- first value/coordinate: `-1.5384093822697655e-68` at frequency `2`, ray `0`, segment `32`, substep `0`
- first scale/h/B_trunc: `0.022285160912469657` / `0.02` / `0.00024786819563641049`
- sign-indeterminate sub-truncation: `931122`
- finite-value enclosure restarts: `1661`
- excess/sign-uncertain/nonfinite/clamp: `0/0/0/0`

## s8 Jν 3중 대조

J_MC는 계약대로 sidecar payload의 `J_producer`다. CMFGEN은 RVTJ의 9610.017–10163.506 km/s 사이 log-J velocity interpolation 후 공통 1000-bin edge에 적분보존 평균했다. point interpolation은 쓰지 않았다.

| band [Å] | J_det/J_MC | J_det/J_CMFGEN | J_MC/J_CMFGEN | log10(det/MC) | log10(det/CMFGEN) | log10(MC/CMFGEN) | toward CMFGEN |
|---|---:|---:|---:|---:|---:|---:|---|
| B0 600–1000 | 0.999326 | 33.7413 | 33.764 | -0.000292704 | 1.52816 | 1.52845 | yes |
| B1 1000–1500 | 0.995926 | 32.1914 | 32.3231 | -0.00177286 | 1.50774 | 1.50951 | yes |
| B2 1500–2000 | 1.00189 | 7.38976 | 7.37579 | 0.000822005 | 0.86863 | 0.867808 | no |
| B3 2000–2500 | 0.993085 | 6.86443 | 6.91223 | -0.00301362 | 0.836604 | 0.839618 | yes |
| B4 2500–3000 | 0.956326 | 15.5806 | 16.2922 | -0.019394 | 1.19258 | 1.21198 | yes |
| BALL 600–3000 | 0.977181 | 11.7038 | 11.9771 | -0.010025 | 1.06833 | 1.07835 | yes |

### spectral norm

| band | pair | median log10 ratio | p10 | p90 | positive pairs | zero/excluded |
|---|---|---:|---:|---:|---:|---:|
| B0 | det/MC | +0.001305 | -0.002894 | +0.004364 | 97 | 0 |
| B0 | det/CMFGEN | +1.489588 | +1.230418 | +1.658189 | 97 | 0 |
| B0 | MC/CMFGEN | +1.489393 | +1.224499 | +1.658454 | 97 | 0 |
| B1 | det/MC | -0.002322 | -0.021740 | +0.046673 | 76 | 0 |
| B1 | det/CMFGEN | +1.543567 | +1.092337 | +1.822300 | 76 | 0 |
| B1 | MC/CMFGEN | +1.547098 | +1.077093 | +1.821144 | 76 | 0 |
| B2 | det/MC | +0.001039 | -0.002880 | +0.003042 | 55 | 0 |
| B2 | det/CMFGEN | +0.829530 | +0.773637 | +1.063737 | 55 | 0 |
| B2 | MC/CMFGEN | +0.828737 | +0.772125 | +1.060727 | 55 | 0 |
| B3 | det/MC | +0.001108 | -0.022961 | +0.006054 | 42 | 0 |
| B3 | det/CMFGEN | +0.782536 | +0.746877 | +0.884767 | 42 | 0 |
| B3 | MC/CMFGEN | +0.780508 | +0.739546 | +0.858911 | 42 | 0 |
| B4 | det/MC | -0.000731 | -0.116476 | +0.185559 | 34 | 0 |
| B4 | det/CMFGEN | +1.120529 | +0.732713 | +1.497224 | 34 | 0 |
| B4 | MC/CMFGEN | +0.984745 | +0.711919 | +1.507408 | 34 | 0 |
| BALL | det/MC | +0.001018 | -0.016683 | +0.020631 | 304 | 0 |
| BALL | det/CMFGEN | +1.262029 | +0.775331 | +1.688322 | 304 | 0 |
| BALL | MC/CMFGEN | +1.258953 | +0.771584 | +1.689085 | 304 | 0 |

candidate가 들어가는 spectral quantile은 두 양수 field가 모두 양수인 bin만 사용했으며, 제외 수를 그대로 기록했다.

## Γ D-lane

기존 `w3_gamma_triple_compare.py`의 grid/C1/C2 loader, EDDFACTOR/RVTJ 적분보존 평균, within-SL fraction·σ·threshold·route 및 `4πσJ/(hν)` quadrature를 import해 재사용했다.

| target | Γ_MC B [s⁻¹] | Γ_CMFGEN C [s⁻¹] | Γ_det D [s⁻¹] | D/B | D/C | log10(D/C) | log10(B/C) | toward CMFGEN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fe III C48 lump (idx 201) | 4.363858095e+02 | 2.807174861e+01 | 685.692816 | 1.5713 | 24.4264 | 1.38786 | +1.191601 | no |
| S II SL4 (idx 4) | 3.310150073e+01 | 4.748076950e-01 | 17.2586497 | 0.521386 | 36.3487 | 1.56049 | +1.843330 | yes |

Fe III와 S II의 D-lane은 동일한 σ·threshold·route·within-SL fraction에 J_det만 대입해 계산했다.

## Acceptance와 판독

- sidecar/schema/checksum/epoch: **PASS**
- CMFGEN integral conservation: `1.000000000000000` (**PASS**)
- 6 band row와 2 Γ row의 baseline provenance: **PASS**
- candidate transport residual ≤1e-4, finite/nonnegative, clamp=0: **PASS** (`9.420355153e-07`, clamp `0`)
- candidate 3-run SHA-256 identity: **PASS** (`f8a46cfd25e01bc863ebdbded0ef779935eec5a5448630996da6b9a73fc7a025`)
- 최종 bench acceptance: **PASS**

최종 사전등록 판독은 **UNRESOLVED-MIXED**이며 방향 자체는 acceptance를 변경하지 않는다.

## 4. 전 수치 재현 명령

```bash
# strict build + skeleton + seeded hierarchy
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -Wconversion -Wshadow -Isrc tests/stage31_cmf_skeleton_selftest.c \
  src/lumina_cmf_field.c -lm -o /tmp/stage31_skeleton_7d
/tmp/stage31_skeleton_7d
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -Wconversion -Wshadow -DLCMF_TEST_HOOKS -Isrc \
  tests/stage31_cmf_guard_selftest.c src/lumina_cmf_field.c -lm \
  -o /tmp/stage31_guard_7d
/tmp/stage31_guard_7d

# C calibration (safety multiplier 1.0)
python3 scripts/stage31_guard_calibrate.py \
  --ka1 docs/s31_results/ka1_round7d.json \
  --ka2 docs/s31_results/ka2_round7d.json \
  --ka3 docs/s31_results/ka3_round7d.json \
  --out docs/s31_results/guard_calibration_round7d.json

# KA1 / KA3
python3 scripts/run_stage31_cmf_ka.py --ka ka1 \
  --work /tmp/stage31_round7d_ka1 \
  --output docs/s31_results/ka1_round7d.json
python3 scripts/run_stage31_cmf_ka.py --ka ka3 \
  --certificate docs/s31_results/mpfr_cert_rung8_fine.json \
  --work /tmp/stage31_round7d_ka3 \
  --output docs/s31_results/ka3_round7d.json

# KA2 raw runner expected rc=1: binary64 matrix-storage qualification only false
python3 scripts/run_stage31_cmf_ka.py --ka ka2 \
  --work /tmp/stage31_round7d_ka2 \
  --output docs/s31_results/ka2_round7d.json
# final 10R judge expected rc=0 / PASS
python3 scripts/s31_ka2_judge.py \
  --oracle docs/s31_results/ka2_oracle_hp.json \
  --solver docs/s31_results/ka2_round7d.json \
  --binary64-oracle docs/s31_results/ka2_oracle_rung10.json \
  --out docs/s31_results/ka2_round7d_judge_hp.json

# discriminator input and J_det / bands / Gamma
sha256sum /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10
python3 scripts/cmf_chieta_check.py /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -D_POSIX_C_SOURCE=200809L -Isrc scripts/stage31_cmf_field_driver.c \
  src/lumina_cmf_field.c -lm -o /tmp/stage31_cmf_field_driver
/tmp/stage31_cmf_field_driver /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10 /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10.manifest.json 8 16 10020 1 /tmp/stage31_jdet.tsv
python3 scripts/stage31_cmf_field_bench.py --frozen /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10 \
  --driver-table docs/s31_results/stage31_jdet_s8_round7d.tsv \
  --report docs/CODEX_STAGE31_BENCH7D_REPLAY.md \
  --status-json docs/s31_results/stage31_bench_round7d.json
```

## 5. 산출물 SHA-256

| artifact | SHA-256 |
|---|---|
| guard calibration JSON | `ba0e7bd44c41e725a260c8643a5aa09e5fe69e2fbcbd648b074b4ef627596733` |
| KA1 JSON | `75b3b6501da05e7308d53107c6fecac6422fcd91b67ce7cbb4539c469703f796` |
| KA3 JSON | `53acc3ebe124b03f8df9ec58122563507f0f1aa7622205a79a955f3551e706ae` |
| KA2 raw JSON | `cc7e67dede42b61d78747166bb36b10a071732d6e618c2812d4ff8731387bd3e` |
| KA2 10R judge JSON | `3af3387a91ca7b99c62717a8c680d9c617d19b90dfac1723020e68dc9d1187aa` |
| discriminator JSON | `f4912e507aaddd0b828e0d3873fbe4017ddb5b37516255b1641b95834e4d4eb1` |
| J_det TSV | `f8a46cfd25e01bc863ebdbded0ef779935eec5a5448630996da6b9a73fc7a025` |

신규 모델/GPU run, acceptance 변경, clamp/floor, 커밋은 없었다.
