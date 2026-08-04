# E10 — 형광 재분배 행렬의 형상 효과 검증

판정일: 2026-08-02 (Asia/Seoul)  
범위: 동결 `emiss_ab2_capture_188766` 계열 payload, E9에서 복원한 s8
energy-redistribution matrix, 기존 stage31 CPU formal solver만 사용. 생산 코드 수정,
신규 모델/GPU run, population/scattering 재수렴, clamp/floor/fallback, commit 없음.

## 1. 결론

**형상 gate는 FAIL이다.** 재분배는 B1과 광학을 원하는 방향으로 움직였지만 B0를 크게
악화했다.

| 대역 | E9 `J_det/CMFGEN` | E10 `J_det/CMFGEN` | E10/E9 | 판독 |
|---|---:|---:|---:|---|
| B0 600--1000 Å | 8.29055106 | **20.90950168** | **2.52208828** | 악화, gate FAIL |
| B1 1000--1500 Å | 4.91614286 | **3.58482097** | **0.72919382** | 개선, -27.08% |
| B2 1500--2000 Å | 1.83988084 | 1.49060727 | 0.81016512 | 개선 |
| B3 2000--2500 Å | 0.20836087 | 0.86053383 | 4.13001657 | 1 쪽으로 개선 |
| B4 2500--3000 Å | 0.33680469 | 0.74457002 | 2.21068784 | 1 쪽으로 개선 |
| BALL 600--3000 Å | 0.93228813 | 1.17171673 | 1.25681824 | 진폭 closure에서 이탈 |
| optical 3000--10000 Å | 6.92103893 | **7.28493778** | **1.05257864** | +5.26%, 방향 적중 |

직접 원인은 B0로의 source-weighted 교차대역 유입이다. B0 재분배 유입의 **68.2869%가
B2**에서 왔다. 반면 3000 Å 경계 빈의 B0 기여는 0.02148%뿐이다. 따라서 이 frozen
probe에서 보인 B0 악화는 빈 폭 누락이나 경계 한 빈의 산술 문제가 아니라 복원된
`R_prefix`의 분기 형상과 E9 source 형상의 결합 결과다.

차터의 판독에 따라 **“형상까지 CMFGEN 쪽으로 이동”은 성립하지 않으며 구조 수리 설계
근거는 아직 완성되지 않았다.** 다만 이 행렬이 iteration-11의 비무작위 41.2134% event
prefix이고 iteration-10 source에 적용됐으므로, 같은 악화가 동시대 full matrix에서도
유지되는지는 **UNRESOLVED**다.

## 2. 사전등록 — exact-bin 적용 전에 고정

exact 1000-bin matvec와 stage31 측정 전에
`validation/emiss_e10/preregistration.json`을 먼저 생성하고 SHA-256
`4484e9be7c9cc3034c9bdf9e7f95d2c4fead6a6c7bf0c38195955d76786148b4`로
고정했다. predictor는 E9 event matrix를 B0--B4/optical로 먼저 축약한 뒤, 대역별 E9
동일-빈 선 반환 power에 곱한다. exact 빈별 source--matrix covariance와 formal transport는
보지 않는다.

CMFGEN 형상 목표 `J/CMFGEN=1`까지 필요한 감소는 다음과 같이 등록했다.

| 대역 | E9 잔여 | CMFGEN까지 필요한 감소 | coarse 사전 예측 | 최소 형상 gate |
|---|---:|---:|---:|---:|
| B0 | 8.29055106 | **-87.9381%** | 12.96845673, **+56.42%** | E9 대비 최소 -10% |
| B1 | 4.91614286 | **-79.6588%** | 3.63937055, **-25.97%** | E9 대비 최소 -10% |
| optical source | 기준 1 | 증가 필요 | **1.25196457, +25.20%** | source와 `J_det` 모두 증가 |

즉 측정 전부터 이 prefix matrix는 B1은 낮추지만, 강한 B2 source의 B0 유입 때문에 B0는
오히려 올릴 것으로 예측됐다. 이를 사후에 “둘 다 감소” 예측으로 바꾸지 않았다. 구조
성공 gate는 물리 목표에 맞춰 B0/B1이 둘 다 최소 10% 감소하는 것으로 별도 고정했고,
coarse magnitude의 보조 적중 창은 ±25%로 두었다.

exact source 측정은 이 사전 예측을 다음처럼 재현했다.

| 대역 | coarse source ratio | exact source ratio | exact/coarse | ±25% |
|---|---:|---:|---:|---|
| B0 | 1.56424545 | **1.74265926** | 1.11405743 | HIT |
| B1 | 0.74028983 | **0.73845397** | 0.99752008 | HIT |
| optical | 1.25196457 | **1.24443379** | 0.99398483 | HIT |

stage31에서는 B1과 optical 방향은 유지됐지만 B0의 비국소 반응이 source 증가보다 더
커서 E10/E9가 2.5221가 됐다. 따라서 B0 stage31 magnitude는 coarse ±25% 창 밖이다.

## 3. 재분배 연산자 적용

### 3.1 source 분해와 적용식

E9 frozen source에서 같은 빈으로 되돌아가던 비열적 선 반환만 교체했다.

```text
eta_E9 = eta_fixed + chi_es,proxy J_E9
                   + (1-eps_MC) chi_line,proxy J_E9

a_i = (1-eps_MC) chi_line,proxy,i J_E9,i Delta_nu_i

eta_E10,j = eta_E9,j - a_j/Delta_nu_j                 [observed input columns]
                       + sum_i R^E[j,i] a_i/Delta_nu_j
```

`eps_MC=0.0024368222433042742`에 해당하는 열 파괴분은 E9에서 이미 line return 밖에
있으므로 다시 재분배하지 않았다. 전자 산란 반환과 `eta_fixed`도 그대로 유지했다.

`R^E[j,i]`는 count probability가 아니라 event terminal output energy를 해당 input
column의 terminal energy로 나눈 확률이다. event input energy로 정규화하면 관측된
`7.5e-7` float-energy 차이를 source에 인위적으로 증폭/감쇠시키므로 사용하지 않았다.
grid 밖 확률은 source에 재주입하지 않고 별도 side ledger에 보존했다.

LCMFCE01의 `eta_coherent` slot에는 stage31 계약상 `eta_total-eta_fixed`인 동결 variable
source가 들어간다. E10에서는 이것이 물리적으로 frequency-coherent하다는 뜻이 아니다.
생산 schema나 source owner를 바꾼 것이 아니라 기존 frozen formal-solve payload를 위한
진단 조립이다.

### 3.2 구성 단계와 적용 단계의 에너지 보존

| gate | 값 |
|---|---:|
| raw event paired input energy | 14.402992095820537 |
| raw event terminal output energy | 14.403002902669414 |
| **구성 단계 `(output/input)-1`** | **+7.5031971e-7** |
| input-column raw closure 최대 절대오차 | 8.1505136e-5 |
| `R` column sum(+outside) 최대 `abs(sum-1)` | 1.5543122e-15 |
| 적용 시 제거한 동일-빈 선 반환 power | 0.006043561079152001 |
| on-grid 재주입 power | 0.006043496188588920 |
| outside-grid side ledger | 6.4890563e-8 |
| **적용 후 `(grid+outside)/removed-1`** | **+2.2204460e-16** |
| 전체 shell source closure(+outside) | 0.0 |

따라서 event 기록 자체의 input/output 차이는 `7.5e-7`로 그대로 보고하되, 복원 확률을
source에 적용한 산술은 roundoff에서 보존된다. `eta*Delta_nu`를 이동하고 출력 빈에서
`Delta_nu_j`로 나눴으므로 빈 폭 누락은 없다.

### 3.3 coverage, 결손, 무폴백 장부

| 항목 | 카운트 |
|---|---:|
| observed input columns | 305 |
| center가 600--3000 Å인 grid bins | 304 |
| 그중 미커버 bins | **0** |
| band 경계를 가로지르는 active bins | 2 (`357`, `661`) |
| 1000-bin 전체 중 matrix 미지원 bins | 695 |
| sparse observed edges | 92,287 |
| 지원 열의 미관측 edges `305*1000-nnz` | **212,713** |
| 한 번이라도 쓰인 output bins / 미사용 bins | 676 / 324 |
| outside-grid terminals | 43 |
| prefix tail에서 terminal 미쌍 activations | 89 (70 bins) |
| matrix 미지원 shells | 49 |

600--3000 Å의 center-selected target bins는 모두 덮이며, 3000 Å를 가로지르는 한 빈을
추가로 적용했다. 나머지 695개 빈과 s8 이외 shell은 operator의 물리적 identity라고
추정하지 않았다. 그곳은 명시적으로 **미지원**이며 이름 붙은 frozen E9 baseline을
그대로 두었다. 212,713개 미관측 edge에는 smoothing, nearest-bin, identity, renormalized
fill을 넣지 않고 정확히 0을 사용했다. 유한·capped prefix에서 “미관측”은 정본 물리에서
확률 0이라는 뜻이 아니다.

구성 source의 음수/nonfinite는 각각 `eta_total=0/0`, `eta_coherent=0/0`,
redistributed power `0/0`이다. clamp와 fallback 카운트도 모두 0이다.

## 4. stage31 formal solve

E9와 같은 C driver, `shell=8`, `nmu=16`, `T_inner=10020 K`, `bb_scale=1` 및 같은
guard로 frozen source formal solve를 수행했다. population, matrix, opacity, scattering
source의 반복 갱신은 없다. 출력 3회는 byte-identical이었다.

```text
outer file SHA-256              = 2654d566991f...ec6549 (3/3 identical)
payload SHA-256                 = e64a59c4b2a6...e79fd
transport_residual              = 8.180569987551006e-7
source_residual                 = 0
source_iterations               = 1
trip count                      = 0
1208 A trip                     = 0
clamp                           = 0
bdf_eta_negative                = 363700
solution_negative_excess        = 0
solution_subtruncation          = 121365
sign-indeterminate subtruncation= 974103
roundoff enclosure restart      = 1624
sign_uncertain                  = 0
nonfinite                       = 0
```

trip=0은 driver exit 0, 완전한 1000-row table, `nonfinite=0`,
`solution_negative_excess=0`으로 판독했다. 별도 trip counter를 새로 생산 코드에 넣지
않았다.

raw table에는 음수 `J_det` bin이 E10 1개(E9 2개) 있다. E10 최소값은 ascending-frequency
bin 213의 `-1.2605513992e-5`다. producer `J` 음수는 0이다. 이 raw 음수는 기존 인증
guard의 sub-truncation 안에 있어 `solution_negative_excess=0`이며, 값을 clamp하거나
bit pattern을 고치지 않았다. `bdf_eta_negative=363700`도 숨기지 않았으며 E9의
358,662보다 5,038 많지만 최종 solution-negative excess는 0이다.

## 5. B0 원인 분해

B0로 재주입된 총 power는 `3.8110395833e-4`다.

| 입력 대역 | B0 유입 power | B0 유입 점유율 |
|---|---:|---:|
| B0 | 3.0467282e-5 | 7.9945% |
| B1 | 7.5192799e-5 | 19.7303% |
| **B2** | **2.6024391e-4** | **68.2869%** |
| B3 | 9.8870248e-6 | 2.5943% |
| B4 | 5.2310824e-6 | 1.3726% |
| 3000 Å boundary bin | 8.1860665e-8 | 0.02148% |

두 straddling input bin이 제거 power에서 차지하는 비율도 0.15530%뿐이다. 따라서
관측된 B0 +152% stage31 변화의 즉시 원인은 **B2→B0 branch와 B2 source power의 결합**이다.

요청된 잔여 원인 후보를 다음처럼 판독한다.

- **빈 폭:** 원인으로 지지되지 않는다. `eta*Delta_nu` matvec와 roundoff closure를
  통과했다.
- **경계:** 측정 가능하며 비지배적이다. 두 straddling bin과 B0 기여를 위에 수치화했다.
- **EPAY:** 독립 물리 owner는 **UNRESOLVED**다. LCMFCE01에 EPAY가 직렬화되지 않아
  별도 대조할 수 없다. 다만 capture의 terminal/input energy 차이 `7.5e-7`은 B0의
  +152%를 산술적으로 설명하기에 너무 작다.
- **정본성:** **UNRESOLVED**다. 관측 즉시 원인은 명확하지만, `R`이 iteration-11
  400,000,000/970,557,187 비무작위 prefix이고 source는 iteration 10이다. full 동시대
  matrix의 B2→B0 분기는 이번 규율 안에서 확정할 수 없다.

## 6. emergent 연결 — 간접 에너지 장부

이번 stage31은 내부장 formal solution이며 emergent flux를 직접 계산하지 않는다.
신규 coupled/GPU run 금지 때문에 source-weighted destination energy로만 간접 평가했다.

exact source 가중 destination은 다음이다.

```text
UV 600--3000 A retention = 0.9634874677
optical 3000--10000 A   = 0.0356761963
EUV                     = 0.0006698641
IR                      = 0.0001557347
outside                 = 0.0000107371
sum-1                   = 6.66e-16
```

현재 emergent UV 42.9% 전부가 이 s8 operator를 정확히 한 번 겪는다는 강한 가정에서

```text
UV:   42.9% -> 41.3336%   (-1.5664 point; 목표 23.8%)
blue:  5.8% -> <=7.3305%  (+1.5305 point; 목표 14.5%)
```

이다. blue 값은 3000--10000 Å로 빠진 에너지를 전부 역사적 blue diagnostic에 넣은
상한이므로 실제 기대는 더 작을 수 있다. 방향은 UV 감소/광학 증가로 기지 결함과
정합하지만, 한 번 적용한 크기는 UV에 필요한 -19.1 point와 blue에 필요한 +8.7
point에 크게 부족하다. 같은 UV retention이 반복된다는 단순 모델에서는 UV 23.8%에
도달하려면 **15.84회의 유효 상호작용**이 필요하다.

escape, shell 이동, population feedback, 반복 상호작용, iteration mismatch, prefix bias가
모두 빠졌고 optical 범위가 blue diagnostic보다 넓다. 따라서 실제 emergent UV/blue
점유율은 **UNRESOLVED**다.

## 7. 최종 판독

1. 재분배 구성과 적용 에너지 보존은 각각 `7.5032e-7`, `2.22e-16`으로 검증됐다.
   clamp/fallback 없이 대상 UV center bins 304/304를 덮었다.
2. B1은 4.916→3.585, B3/B4도 1 쪽으로 움직였고 optical source/J는 각각
   +24.44%/+5.26%다. frequency-off-diagonal source가 실제 형상을 바꾼다는 방향성은
   확인됐다.
3. 그러나 B0는 8.291→20.910으로 악화해 사전등록 형상 gate를 실패했다. BALL도
   0.932→1.172가 되어 E9의 scalar amplitude closure를 유지하지 못했다.
4. 즉시 원인은 B0 유입의 68.29%를 차지한 B2→B0 branch다. 빈 폭과 경계는 지배 원인이
   아니며, 독립 EPAY와 동시대 full-matrix 정본성은 UNRESOLVED다.
5. emergent 방향은 맞지만 single-pass 크기는 UV/blue 목표에 불충분하다. 따라서
   **구조 수리 설계 근거 완성=NO**다. 다음 필요한 증거는 생산 구현이 아니라 먼저
   cap 없는 iteration-10 동시대 `R`, 독립 EPAY ledger, 경계 bin 정의를 가진 동일한
   offline 재검증이다.

## 8. 재현 명령

순서가 판정의 일부다. `preregister`를 exact apply보다 먼저 실행하고 SHA를 고정한다.
아래는 모델/plasma/GPU transport를 실행하지 않는다.

```bash
E10_RUN=/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766
E10_CMF=/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4

python3 -m py_compile \
  scripts/emiss_e10_preregister.py \
  scripts/emiss_e10_apply_redistribution.py \
  scripts/emiss_e10_jdet_measure.py \
  scripts/emiss_e10_diagnose.py

# 1. 측정 전 사전등록
python3 scripts/emiss_e10_preregister.py \
  --e9-payload validation/emiss_e9/emiss_e9_effective_iter10 \
  --source-payload "$E10_RUN/emiss_ab_iter10.A" \
  --matrix validation/emiss_e9/redistribution_matrix_s8_sparse.csv \
  --normalization validation/emiss_e9/redistribution_input_normalization_s8.csv \
  --e9-stage31 validation/emiss_e9/stage31_measurement.csv \
  --out-dir validation/emiss_e10 \
  > validation/emiss_e10/preregistration.stdout
sha256sum validation/emiss_e10/preregistration.json

# 2. exact 1000-bin offline source 구성
python3 scripts/emiss_e10_apply_redistribution.py \
  --e9-payload validation/emiss_e9/emiss_e9_effective_iter10 \
  --source-payload "$E10_RUN/emiss_ab_iter10.A" \
  --matrix validation/emiss_e9/redistribution_matrix_s8_sparse.csv \
  --normalization validation/emiss_e9/redistribution_input_normalization_s8.csv \
  --matrix-summary validation/emiss_e9/redistribution_summary.json \
  --preregistration validation/emiss_e10/preregistration.json \
  --out-dir validation/emiss_e10 \
  > validation/emiss_e10/redistribution_application.stdout

python3 scripts/cmf_chieta_check.py \
  validation/emiss_e10/emiss_e10_redistributed_iter10

# 3. E9와 같은 stage31 CPU formal solve; 3회 결정론 확인
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -D_POSIX_C_SOURCE=200809L -Isrc \
  scripts/stage31_cmf_field_driver.c src/lumina_cmf_field.c -lm \
  -o /tmp/stage31_cmf_field_driver_e10

/tmp/stage31_cmf_field_driver_e10 \
  validation/emiss_e10/emiss_e10_redistributed_iter10 \
  validation/emiss_e10/emiss_e10_redistributed_iter10.manifest.json \
  8 16 10020 1 validation/emiss_e10/jdet_redistributed_s8.tsv

/tmp/stage31_cmf_field_driver_e10 \
  validation/emiss_e10/emiss_e10_redistributed_iter10 \
  validation/emiss_e10/emiss_e10_redistributed_iter10.manifest.json \
  8 16 10020 1 /tmp/e10_jdet_repeat2.tsv

/tmp/stage31_cmf_field_driver_e10 \
  validation/emiss_e10/emiss_e10_redistributed_iter10 \
  validation/emiss_e10/emiss_e10_redistributed_iter10.manifest.json \
  8 16 10020 1 /tmp/e10_jdet_repeat3.tsv

sha256sum \
  validation/emiss_e10/jdet_redistributed_s8.tsv \
  /tmp/e10_jdet_repeat2.tsv /tmp/e10_jdet_repeat3.tsv
cmp validation/emiss_e10/jdet_redistributed_s8.tsv /tmp/e10_jdet_repeat2.tsv
cmp validation/emiss_e10/jdet_redistributed_s8.tsv /tmp/e10_jdet_repeat3.tsv

# 4. 대역 판정, guard, 원인·emergent 간접 장부
python3 scripts/emiss_e10_jdet_measure.py \
  --payload validation/emiss_e10/emiss_e10_redistributed_iter10 \
  --jdet validation/emiss_e10/jdet_redistributed_s8.tsv \
  --e9-jdet validation/emiss_e9/jdet_effective_s8.tsv \
  --preregistration validation/emiss_e10/preregistration.json \
  --source-measurement validation/emiss_e10/source_band_measurement.csv \
  --cmf-run "$E10_CMF" --out-dir validation/emiss_e10 \
  > validation/emiss_e10/stage31_measurement.stdout

python3 scripts/emiss_e10_diagnose.py \
  --e9-payload validation/emiss_e9/emiss_e9_effective_iter10 \
  --source-payload "$E10_RUN/emiss_ab_iter10.A" \
  --matrix validation/emiss_e9/redistribution_matrix_s8_sparse.csv \
  --normalization validation/emiss_e9/redistribution_input_normalization_s8.csv \
  --preregistration validation/emiss_e10/preregistration.json \
  --application-summary validation/emiss_e10/redistribution_application_summary.json \
  --stage31-summary validation/emiss_e10/stage31_summary.json \
  --out-dir validation/emiss_e10 \
  > validation/emiss_e10/diagnosis.stdout

jq '{construction_energy,operator_normalization,application_energy,
     coverage_and_missing,source_guards}' \
  validation/emiss_e10/redistribution_application_summary.json
jq '{bands,driver_metadata,raw_table_negative_counts,
     raw_table_nonfinite_counts,trip_count,shape_gate,optical_gate}' \
  validation/emiss_e10/stage31_summary.json
jq '{B0_redistributed_inflow,boundary_audit,
     emergent_indirect,residual_cause_readout,verdict}' \
  validation/emiss_e10/diagnosis_summary.json

sha256sum \
  scripts/emiss_e10_preregister.py \
  scripts/emiss_e10_apply_redistribution.py \
  scripts/emiss_e10_jdet_measure.py \
  scripts/emiss_e10_diagnose.py \
  validation/emiss_e10/preregistration.json \
  validation/emiss_e10/emiss_e10_redistributed_iter10 \
  validation/emiss_e10/source_band_measurement.csv \
  validation/emiss_e10/jdet_redistributed_s8.tsv \
  validation/emiss_e10/stage31_measurement.csv \
  validation/emiss_e10/redistribution_application_summary.json \
  validation/emiss_e10/stage31_summary.json \
  validation/emiss_e10/band_flow_measurement.csv \
  validation/emiss_e10/diagnosis_summary.json
```

규율 장부: 생산 코드 수정 0, 신규 모델/GPU run 0, clamp/floor 0, missing-bin/edge
fallback 0, commit 0.
