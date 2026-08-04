# T5 — 순수 rank-1 대리 연산자로 E10–E13 종결 판정

판정일: 2026-08-02 (Asia/Seoul)  
실행 성격: 기존 capture만 읽은 CPU 오프라인 분석 및 stage31 formal solve. 생산 코드 수정,
신규 모델/GPU run, clamp/floor/fallback, commit 없음.

## 1. 결론

**수치 시험은 완결했지만, 무조건적인 노선 종결/유지 판정은 `UNRESOLVED`다.** 이유는
계산 실패가 아니라 **판정 계약이 서로 다르기 때문**이다.

- 정본 사전등록 `docs/FABLE_UV_T3T4.md` §9.4의 세 항목
  **B0·B1·B2→B0**는 대각 포함/제외 rank-1 판본이 모두 15% 이내로 재현했다.
  이 계약만 따르면 **T5 PASS → E10–E13 형광행렬 노선 종결**이다.
- 이번 과업 본문은 세 번째 항목을 **광학 변화량**으로 바꿨다. E12 정식 artifact의 실제
  광학 변화 `−7.901687%`를 기준으로 하면 rank-1은 `−5.941596%`/`−5.887430%`,
  상대오차 24.81%/25.49%로 **FAIL → 노선 유지**다. 과업에 적힌 `−0.79%`를 문자
  그대로 써도 FAIL이다.
- 반면 광학 **수준비** `E12/E9=0.920983`을 비교하면 rank-1의
  `0.940584`/`0.941126`은 2.13%/2.19% 오차로 PASS다. 작은 변화량을 분모로 삼았을
  때만 15%를 넘는다.

따라서 결과를 본 뒤 세 번째 metric을 B2→B0, 광학 변화량, 광학 수준비 중 하나로
고르는 것은 사후 선택이다. 본 보고서는 이를 숨기지 않고 최종 노선 처분을
`UNRESOLVED`로 둔다. **이번 과업문의 광학-변화량 계약이 정본을 명시적으로
대체한다고 해석할 경우의 조건부 판정은 FAIL/노선 유지**이고, **기존 §9.4가 계속
정본이면 PASS/노선 종결**이다.

## 2. 입력 세대 계약 — N4 선처리

과업 시작 시 원파일과 사이드카를 다시 읽었다.

| 항목 | E12가 실제 사용한 세대 | T5 시작 시 현재 파일 | 일치 |
|---|---:|---:|---|
| 파일명 | `fluor_matrix_iter10` | `fluor_matrix_iter10` | 이름만 같음 |
| header iteration | **10** | **11** | 아니오 |
| sparse edge | **473,045** | **468,330** | 아니오 |
| SHA-256 | `2b65dba6…d01c99b` | `08ff3312…735af6` | 아니오 |
| absorbed energy | 3065.503198962503 | 2975.9287956931835 | 아니오 |
| events total | 509,203,774 | 483,936,781 | 아니오 |

현재 전체 SHA-256은
`08ff331222f5ddfadac62e1c716963c22aaf0986e0dbd3615b46145e46735af6`이고
`sha256sum -c`는 PASS다. 이는 현재 payload/sidecar 쌍의 무결성만 보증하며 E12 세대가
복구됐다는 뜻이 아니다. 현재 reader 실측은 active input 752개,
`column_closure_max_abs=1.86033e-13`, 전역 에너지 오차
`(reemitted/absorbed)-1=2.80091e-8`이다.

지시에 따라 **iteration-11 현재 행렬에서 q를 만들고 T5를 수행**했다. 다만 동일 소스
계약을 지키기 위해 E12의 iteration-10 `emiss_ab_iter10.A`와 same-capture E9 frozen
payload를 썼다. 따라서 matrix header 11/source header 10이라는 교차 세대는 남는다.
기존 applicator에는 iteration guard가 없지만, T5 builder는 시작 시
`expected_iteration=11`과 위 전체 SHA를 강제하여 파일이 다시 바뀌면 fail-closed한다.

## 3. q와 R*의 정확한 정의

현재 행렬의 on-grid edge energy를 `A_ij`(입력 `i`, 출력 `j`)라 두었다. 두 판본은

```text
q_inc[j] = sum_i A_ij       / sum_ik A_ik
q_exc[j] = sum_{i != j} A_ij / sum_{i != k} A_ik
R*[j,i]  = q[j]              (모든 관측 active input i)
```

이다. 이는 각 입력의 conditional output SED를 그 입력의 실측 on-grid terminal
energy로 가중 평균한 것과 같다. `q_exc`에서는 exact-bin 대각 `i=j`만 평균 전에
제외했다. 두 q 모두 binary64 `fsum`으로 정확히 1이 되게 **에너지 보존 정규화**했고,
부호 있는 마지막 roundoff 보정은 포함판 `+1.11022e-16`, 제외판 `0`이다. 이는 q의
정의상 정규화이며 clamp/floor/fit이 아니다.

E10과 동일하게 관측 active column 752개만 교체하고, 미관측 248 bin과 다른 49 shell은
frozen E9 그대로 유지했다. 결측을 identity로 추정하거나 채우지 않았다. 순수 q가
on-grid에서 합 1이므로 fixture의 outside probability는 0이다.

### 3.1 q 대역표

| 출력 대역 | 대각 포함 q | 대각 제외 q |
|---|---:|---:|
| EUV 100–600 Å | 0.693821% | 0.711294% |
| **B0 600–1000 Å** | **10.529739%** | **10.549059%** |
| **B1 1000–1500 Å** | **25.220383%** | **25.207928%** |
| **B2 1500–2000 Å** | **40.076572%** | **40.008762%** |
| B3 2000–2500 Å | 9.736700% | 9.656115% |
| B4 2500–3000 Å | 7.140703% | 7.206183% |
| optical 3000–10000 Å | 6.492474% | 6.548044% |
| IR 10000–20000 Å | 0.109609% | 0.112615% |
| **합** | **100.000000%** | **100.000000%** |

원행렬 on-grid energy 중 exact diagonal은 2.86774%이고, 두 q 사이 TVD는
**0.00517345**다. 따라서 q 자체는 대각 처리에 둔감하다.

## 4. E10 applicator와 source guard

fixture는 기존 fixture 전용 canonical writer로 썼고 production writer나 생산 코드는
수정하지 않았다.

| 항목 | 대각 포함 | 대각 제외 |
|---|---:|---:|
| fixture SHA-256 | `ba11c447…0ef95bd` | `59ae45b7…551db80` |
| sparse edge | 606,112 | 606,112 |
| fixture reader 열 폐합 최대 | 2.83167e-15 | 3.07524e-15 |
| E10 applicator 열 합 오차 | 6.66134e-16 | 1.11022e-16 |
| 제거 line-return power | 0.006765195301256633 | 동좌 |
| on-grid 재주입 | 0.006765195301256632 | 0.006765195301256634 |
| 적용 상대 에너지 오차 | −1.11022e-16 | +2.22045e-16 |
| full source 에너지 오차 | +2.22045e-16 | +2.22045e-16 |
| 음수/nonfinite/clamp/fallback | 0/0/0/0 | 0/0/0/0 |

두 판본 모두 E12의 완화값이 아니라 **E10 기본 엄격 열 가드 `2e-13`**를 통과했다.
E9 component reconstruction relative error는 양쪽 `1.61026e-16`이다.

## 5. stage31 J_det 대역표

shell 8, `nmu=16`, `T_inner=10020 K`, `bb_scale=1`로 E10/E12와 같은 CPU formal
solve를 썼다. 각 판본 3회 J_det와 stdout이 byte-identical이다.

| 대역 | E12 `J_det/CMFGEN` | R* 대각 포함 | R* 대각 제외 | 포함 `R*/E9` | 제외 `R*/E9` |
|---|---:|---:|---:|---:|---:|
| **B0** | **26.432495** | **29.466800** | **29.670659** | 3.554263 | 3.578852 |
| **B1** | **5.658865** | **6.006075** | **6.002118** | 1.221705 | 1.220900 |
| B2 | 1.691298 | 1.613570 | 1.611091 | 0.876997 | 0.875650 |
| B3 | 0.609357 | 0.554897 | 0.549488 | 2.663152 | 2.637195 |
| B4 | 1.482722 | 1.716596 | 1.734059 | 5.096709 | 5.148559 |
| BALL 600–3000 Å | 1.528677 | 1.602007 | 1.605825 | 1.718361 | 1.722455 |
| **optical** | **6.374160** | **6.509819** | **6.513568** | **0.940584** | **0.941126** |

광학 source ratio는 포함 0.792326, 제외 0.796601이고, stage31 변화량은 각각
`−5.941596%`, `−5.887430%`다.

stage31 guard는 포함/제외 각각 `transport_residual=8.18057e-7/1.04362e-6`,
`source_residual=0`, `source_iterations=1`; trip, clamp, nonfinite,
sign-uncertain, solution-negative-excess는 전부 0이다. R* J_det 음수 bin은 0이고 E9
baseline의 기존 음수 bin 2개는 숨기지 않았다. `bdf_eta_negative=366471/366429`,
subtruncation `123648/123659`, sign-indeterminate subtruncation
`974048/974031`, enclosure restart 1624도 그대로 남겼다.

## 6. 15% 재현 판정

### 6.1 정본 §9.4 계약

| metric | E12 target | 대각 포함 | 상대오차 | 대각 제외 | 상대오차 | 판정 |
|---|---:|---:|---:|---:|---:|---|
| B0 `J_det/CMFGEN` | 26.432495 | 29.466800 | **11.48%** | 29.670659 | **12.25%** | 양쪽 PASS |
| B1 `J_det/CMFGEN` | 5.658865 | 6.006075 | **6.14%** | 6.002118 | **6.07%** | 양쪽 PASS |
| B2→B0 source 점유율 | 54.924532% | 60.917882% | **10.91%** | 60.917882% | **10.91%** | 양쪽 PASS |

순수 rank-1에서는 B0 출력 q가 모든 입력에 공통이므로 B2→B0 점유율은 active
line-return 입력 power 중 B2 몫과 정확히 같다. **정본 판정은 두 판본 모두 PASS**다.

### 6.2 이번 과업문의 광학 계약

E12 artifact가 기록한 값은 `E12/E9=0.9209831266`, 즉
`fractional_change_from_E9=−0.0790168734 = −7.901687%`다. 과업문의 `−0.79%`와
10배 다르다.

| 광학 metric 해석 | target | 포함 결과/오차 | 제외 결과/오차 | 15% |
|---|---:|---:|---:|---|
| E12 artifact 변화량 | −7.901687% | −5.941596% / **24.81%** | −5.887430% / **25.49%** | FAIL |
| 과업문 문자값 | −0.790000% | −5.941596% / **652.10%** | −5.887430% / **645.24%** | FAIL |
| E12/E9 수준비 | 0.920983 | 0.940584 / **2.13%** | 0.941126 / **2.19%** | PASS |
| `J_det/CMFGEN` 수준 | 6.374160 | 6.509819 / **2.13%** | 6.513568 / **2.19%** | PASS |

즉 광학 **스펙트럼 수준은 rank-1과 구별되지 않지만**, 작은 E9 대비 변화량을 직접
상대 비교하면 15%를 넘는다.

### 6.3 같은 세대 full-R 대조군

세대 차이와 R* 잔차를 분리하기 위해 현재 iteration-11 full R도 같은 source/guard로
한 번 적용했다. 이는 판정 기준을 바꾸기 위한 세 번째 모델이 아니라 read-only
동일세대 대조군이다.

| metric | E12 it10 | current full R it11 | E12 상대오차 | R* 포함의 full-R 상대오차 | R* 제외의 full-R 상대오차 |
|---|---:|---:|---:|---:|---:|
| B0 `J/CMF` | 26.432495 | 27.930222 | 5.67% | 5.50% | 6.23% |
| B1 `J/CMF` | 5.658865 | 5.740905 | 1.45% | 4.62% | 4.55% |
| optical `J/CMF` | 6.374160 | 6.314278 | 0.94% | 3.10% | 3.16% |
| optical 변화량 | −7.901687% | −8.766909% | 10.95% | — | — |

current full R 자체는 E12의 B0/B1/광학 변화량을 모두 15% 안에서 재현한다. 따라서
rank-1의 광학-변화량 FAIL을 세대 교체 하나로 설명할 수는 없다. 다만 rank-1과 full R의
광학 **수준** 차이는 3.2% 이하라서, 변화량 상대오차 25%는 작은 분모에 의해 확대된다.

## 7. 비대각 정보량 — SVD와 TVD

1000-bin operator는 각 active 입력 row를 on-grid 확률로 정규화해 SVD했다. 이 SVD는
입력 energy를 동일 가중하므로, 매우 작은 에너지의 희귀 row도 큰 row와 같은 비중을
갖는다. 그래서 실제 source 영향에는 energy-weighted 잔차와 TVD를 함께 봐야 한다.

| 잔차 metric | 대각 포함 | 대각 제외 |
|---|---:|---:|
| `sigma_1` | 2.140371 | 2.153421 |
| `sigma_2` | 1.000039 | 1.000029 |
| `sigma_3` | 0.606654 | 0.600735 |
| `sigma_4` | 0.596649 | 0.555768 |
| `sigma_5` | 0.529831 | 0.528080 |
| 최적 rank-1 Frobenius energy | **59.803%** | **62.457%** |
| 최적 rank-2 누적 energy | 72.858% | 75.927% |
| 최적 rank-1 상대 Frobenius 잔차 | 0.6340 | 0.6127 |
| q-proxy 무가중 상대 Frobenius 잔차 | 0.6536 | 0.6346 |
| q-proxy 입력-energy 가중 잔차 | **0.4534** | **0.3440** |
| row→q energy-weighted 평균 TVD | **0.1263** | **0.1173** |
| row→q energy-weighted p50/p95 TVD | 0.1052 / 0.2484 | 0.0979 / 0.2212 |
| row→q 최대 TVD | 0.99999 | 0.99999 |

최대 TVD는 거의 무게가 없는 희귀 row 때문에 1에 가깝다. 실질 입력 밴드별
energy-weighted output SED의 q 대비 TVD는 다음과 같다.

| 입력 대역 | 대각 포함 TVD | 대각 제외 TVD |
|---|---:|---:|
| B0 | 0.1301 | 0.1131 |
| B1 | 0.0680 | 0.0520 |
| B2 | **0.0517** | **0.0418** |
| B3 | 0.1964 | 0.1768 |
| B4 | 0.1031 | 0.0929 |
| optical | 0.1173 | 0.1083 |

SVD는 원행렬이 수학적으로 정확한 rank-1은 아님을 보여 주고, B3/B4에는 남은 구조가
있다. 실제 stage31에서도 R*와 same-generation full R의 B4 차이는 20.56–21.78%로
유의미하다. 반면 사전등록 핵심 B0/B1은 6.3% 이내, 광학 수준은 3.2% 이내다. 따라서
**“모든 대역에서 R의 정보가 rank-1과 구별 불가”는 지지되지 않지만, 원 사전등록이
찍은 B0/B1/B2→B0 관측량에서는 구별되지 않는다.**

## 8. 최종 처분

1. **수치 산출: RESOLVED.** q 두 판본, R* 적용, stage31, B0–B4/광학 대역표,
   SVD/TVD, 동일세대 full-R 대조까지 완료했다.
2. **정본 `FABLE_UV_T3T4.md` §9.4: PASS.** 이 계약을 유지하면
   “R의 정보 내용은 지정 관측량에서 rank-1과 구별 불가”이고 E10–E13을 종결한다.
3. **이번 과업문 광학 변화량 계약: FAIL.** 이 문구가 정본을 대체하면 비대각 구조가
   광학 변화량에 유의미하므로 노선을 유지한다. B4의 20%대 차이도 유지 사유다.
4. **무조건적인 노선 판정: UNRESOLVED.** 두 계약이 반대 결론을 내고, 과업문의
   `−0.79%`는 E12 artifact의 `−7.901687%`와도 불일치한다. 어느 metric이 지배하는지
   결과 확인 후 임의 선택하지 않았다.

추가 GPU/model run이나 생산 수리가 필요한 미해결은 아니다. 필요한 것은 오직 판정권자가
세 번째 정본 metric을 **B2→B0**, **광학 변화량**, 또는 **광학 수준비** 중 하나로
명시하는 계약 정정이다.

## 9. 전 수치 재현 명령

아래 명령은 기존 capture와 CMFGEN reference를 읽고 workspace의
`validation/emiss_t5`만 다시 만든다. 생산 실행과 GPU를 사용하지 않는다.

```bash
T5_RUN=/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828
T5_OUT=validation/emiss_t5
T5_CMF=/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4
T5_SHA=08ff331222f5ddfadac62e1c716963c22aaf0986e0dbd3615b46145e46735af6

# N4: 현재 세대를 먼저 실측/고정
sha256sum "$T5_RUN/fluor_matrix_iter10"
sha256sum -c "$T5_RUN/fluor_matrix_iter10.sha256"
python3 scripts/emiss_e11_fluor_matrix.py "$T5_RUN/fluor_matrix_iter10"

# q 두 판본, R* fixture, SVD/TVD, application contract
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python3 scripts/emiss_t5_rank1.py build \
  --matrix "$T5_RUN/fluor_matrix_iter10" \
  --expected-iteration 11 --expected-sha256 "$T5_SHA" \
  --e9-payload validation/emiss_e12/e9_same_capture/emiss_e9_effective_iter10 \
  --source-payload "$T5_RUN/emiss_ab_iter10.A" \
  --base-preregistration validation/emiss_e12/preregistration.json \
  --out-dir "$T5_OUT" > "$T5_OUT/build.stdout"

# 순수 rank-1 두 판본: E10 기본 2e-13 guard를 그대로 사용
for T5_VARIANT in diagonal_inclusive diagonal_exclusive; do
  mkdir -p "$T5_OUT/$T5_VARIANT"
  python3 scripts/emiss_e10_apply_redistribution.py \
    --e9-payload validation/emiss_e12/e9_same_capture/emiss_e9_effective_iter10 \
    --source-payload "$T5_RUN/emiss_ab_iter10.A" \
    --matrix "$T5_OUT/rank1_${T5_VARIANT}.lfmat" --matrix-format formal \
    --preregistration "$T5_OUT/preregistration_${T5_VARIANT}.json" \
    --out-dir "$T5_OUT/$T5_VARIANT" \
    > "$T5_OUT/$T5_VARIANT/redistribution_application.stdout"
done

# 동일 세대 current-full-R 대조군
mkdir -p "$T5_OUT/current_full_R"
python3 scripts/emiss_e10_apply_redistribution.py \
  --e9-payload validation/emiss_e12/e9_same_capture/emiss_e9_effective_iter10 \
  --source-payload "$T5_RUN/emiss_ab_iter10.A" \
  --matrix "$T5_RUN/fluor_matrix_iter10" --matrix-format formal \
  --preregistration "$T5_OUT/preregistration_current_full_R.json" \
  --out-dir "$T5_OUT/current_full_R" \
  > "$T5_OUT/current_full_R/redistribution_application.stdout"

# E10/E12 동일 CPU stage31 driver
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -D_POSIX_C_SOURCE=200809L -Isrc scripts/stage31_cmf_field_driver.c \
  src/lumina_cmf_field.c -lm -o /tmp/stage31_cmf_field_driver_t5

for T5_VARIANT in diagonal_inclusive diagonal_exclusive current_full_R; do
  /tmp/stage31_cmf_field_driver_t5 \
    "$T5_OUT/$T5_VARIANT/emiss_e10_redistributed_iter10" \
    "$T5_OUT/$T5_VARIANT/emiss_e10_redistributed_iter10.manifest.json" \
    8 16 10020 1 "$T5_OUT/$T5_VARIANT/jdet_redistributed_s8.tsv" \
    > "$T5_OUT/$T5_VARIANT/stage31_run.stdout"

  # 각 명령을 두 번 더 실행하고 cmp하면 3/3 byte-identical
  /tmp/stage31_cmf_field_driver_t5 \
    "$T5_OUT/$T5_VARIANT/emiss_e10_redistributed_iter10" \
    "$T5_OUT/$T5_VARIANT/emiss_e10_redistributed_iter10.manifest.json" \
    8 16 10020 1 "/tmp/t5_${T5_VARIANT}_repeat2.tsv" \
    > "/tmp/t5_${T5_VARIANT}_repeat2.stdout"
  cmp "$T5_OUT/$T5_VARIANT/jdet_redistributed_s8.tsv" \
      "/tmp/t5_${T5_VARIANT}_repeat2.tsv"
  /tmp/stage31_cmf_field_driver_t5 \
    "$T5_OUT/$T5_VARIANT/emiss_e10_redistributed_iter10" \
    "$T5_OUT/$T5_VARIANT/emiss_e10_redistributed_iter10.manifest.json" \
    8 16 10020 1 "/tmp/t5_${T5_VARIANT}_repeat3.tsv" \
    > "/tmp/t5_${T5_VARIANT}_repeat3.stdout"
  cmp "$T5_OUT/$T5_VARIANT/jdet_redistributed_s8.tsv" \
      "/tmp/t5_${T5_VARIANT}_repeat3.tsv"

  python3 scripts/emiss_e10_jdet_measure.py \
    --payload "$T5_OUT/$T5_VARIANT/emiss_e10_redistributed_iter10" \
    --jdet "$T5_OUT/$T5_VARIANT/jdet_redistributed_s8.tsv" \
    --e9-jdet validation/emiss_e12/jdet_e9_same_capture_s8.tsv \
    --preregistration "$T5_OUT/preregistration_${T5_VARIANT}.json" \
    --source-measurement "$T5_OUT/$T5_VARIANT/source_band_measurement.csv" \
    --cmf-run "$T5_CMF" --out-dir "$T5_OUT/$T5_VARIANT" \
    > "$T5_OUT/$T5_VARIANT/stage31_measurement.stdout"
done

python3 scripts/emiss_t5_rank1.py judge --out-dir "$T5_OUT" \
  > "$T5_OUT/verdict.stdout"
```

주요 SHA-256:

```text
T5 tool                         1d4f19bb7e1a17688d35eedac5c18c76c794bb900bf0fb93c058fbde5326a4b1
rank1 inclusive payload         0caac045326e4b556cb82c143b8878487f3d8c4ff2062600368789b84000cad2
rank1 exclusive payload         4ab5cd7812f87c951cc4433697a4ca61a2d85e9b272a72b2da196e8620058609
rank1 inclusive J_det           f1fbe2df3a643751c120c716ad844af3b315559eee6edad7414c9041b690764c
rank1 exclusive J_det           8e2c3f8e0cb14b40090494533ac4f35f00e4e3b4908fa173b8e39523fc712fd7
current full-R J_det             0e795fac48004c747544cd7d84f1f49617a9bfdcb61b82285fea796c8ee02f1f
```

정량 원자료는 `validation/emiss_t5/rank1_residual_summary.json`, `q_by_bin.csv`,
`svd_spectrum.csv`, `input_row_tvd.csv`, `input_band_tvd.csv`, 각 판본의
`stage31_measurement.csv`, 그리고 기계 판정 `verdict.json`에 있다.
