# E12 — 무편향 형광 행렬로 형상 gate 재판정

판정일: 2026-08-02 (Asia/Seoul)  
입력: `/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828`의
iteration-10 LFMAT001, chieta, emiss A/B2와 E9 frozen probe. 생산 코드 수정, 신규
모델/GPU run, clamp/floor/fallback, commit 없음. 변경은 오프라인 판독/재현 도구와 이
보고서뿐이다.

## 1. 결론

**엄격한 차터 판정은 UNRESOLVED, LFMAT001 정식 허용오차 내 보조 물리 판독은 명백한
FAIL이다. 구조 수리 설계 근거는 완성되지 않았고 생산 구현 상신 근거도 성립하지
않는다.**

엄격 판정이 UNRESOLVED인 이유는 단 하나다. 정식 파일 판독 계약의 열 폐합 허용오차는
`2e-12`이고 측정값 `2.04503e-13`은 이를 넉넉히 통과하지만, E10 prefix 소비자가 쓰던
더 엄격한 `2e-13`을 2.25% 초과한다. 같은 guard의 기본 실행은 exit 2로 fail-closed
됐다. 값을 맞추는 column renormalization, clamp 또는 terminal ledger 수정은 하지
않았다.

그와 별도로 LFMAT001 계약 허용오차 `2e-12`를 명시한 보조 실행에서는 사전등록 네
가설이 모두 틀렸다.

| 사전등록 가설 | 문턱 | 측정 | 판정 |
|---|---:|---:|---|
| H1 B2→B0 지배 소멸 | B0 유입의 ≤2.04205% | **54.9245%** | FAIL |
| H2 B0 악화 소멸 | `J_det/CMFGEN ≤ 8.290551` | **26.432495** | FAIL |
| H3 B0/B1 하락 | E12/E9 <1, <1 | **3.18827, 1.15108** | FAIL |
| H4 optical 상승 | source와 J 모두 >1 | **0.74650, 0.92098** | FAIL |

따라서 E10의 B0 악화를 prefix 절단 하나로 설명하는 가설은 기각된다. edge가
92,287개에서 473,045개로 5.1258배 늘자 B2→B0의 *점유율*은 68.2869%에서
54.9245%로 13.3623%p 낮아졌지만, B0 총 유입은 1.5871배, B2→B0 절대 유입은
1.2765배가 됐다. 즉 절단 편향은 있었으나 부호를 뒤집는 편향이 아니었다.

## 2. 입력과 행렬 인증

### 2.1 동시대 입력

| 파일 | SHA-256 | 계약 |
|---|---|---|
| `fluor_matrix_iter10` | `2b65dba6...d01c99b` | LFMAT001-v1, iteration 10 |
| `chieta_iter10` | `894a4ee8...863d5dc` | LCMFCE01-v1, iteration/generation 10 |
| `emiss_ab_iter10.A` | `894a4ee8...863d5dc` | chieta와 byte-identical, A-production |
| `emiss_ab_iter10.B2` | `62159e14...60d1d` | B2 Aul-nu, A-undefined retention |

A와 B2는 같은 `common_assembly_state_sha256=03cb5121...5f8c5`이며 geometry, grid,
opacity, coherent source와 producer J가 같다. B2는 정의된 transition에 Aul-nu 식을 쓰고
미정의 40,708 transition/501,462 line-shell은 A를 명시적으로 유지한다. A와 B2의
`eta_fixed/eta_total`만 달라지고 최대 절대차는 `4.7095023e-12`다. 형광 재분배에는
E10과 같이 production A를 사용했고 B2는 동시대 companion 입력 인증에 사용했다.

새 A는 E10 때 A와 약 `9e-12` 상대 수준에서만 다르지만 byte-identical은 아니다. E10의
component reconstruction guard `2e-15`를 완화하지 않기 위해 같은 E9 오프라인 식으로
동시대 A 기반 frozen baseline을 재직렬화했다. 새 baseline stage31 B0는
`8.29055105658781`로 원 E9 `8.29055105658763`과 15자리까지 같으며, 이 rebase가 판정을
움직이지 않는다.

### 2.2 checksum, 에너지와 열 폐합

sidecar `sha256sum -c`는 PASS이고 reader가 magic/endian/version/flags, exact length,
중복/음수/nonfinite edge, 대표 shell-group 합=global을 모두 통과했다.

| 장부 | 값 |
|---|---:|
| events total / classified | 509,203,774 / 509,047,721 |
| sparse global edges | **473,045** |
| absorbed / reemitted energy | 3065.503198962503 / 3065.503292285673 |
| `(reemitted/absorbed)-1` | **+3.0443018e-8** |
| 최대 input-column relative closure | **2.0470962e-13** |
| shell absorbed 합 - header | +1.63709e-11 (5.34e-15 relative) |
| shell reemitted 합 - header | +6.91216e-11 (2.25e-14 relative) |
| k-packet events | 10,440,714 |
| k-packet absorbed / total absorbed | **0.02042051545 (2.04205%)** |
| k-packet energy closure | +7.6467328e-7 |

미분류 원시 카운트는 input 83,438, output 73,140, invalid energy **0**, unresolved route
9다. 이 범주는 서로 배타적이라고 schema가 보증하지 않으므로 합을 total-classified와
억지로 맞추지 않았다. 요청된 미분류 에너지는 0이며 energy를 대각 edge로 숨긴 흔적도
없다.

형상 적용에서 제거한 line-return power는 `0.006765195756787215`, on-grid 재주입은
`0.006764368489028963`, outside side ledger는 `8.2726775821e-7`이다.
`(grid+outside)/removed-1=-6.77e-15`, 전체 source closure는 `-6.11e-15`다. 빈 폭은
`eta*Delta_nu`로 이동했고 보정 없이 출력 `Delta_nu_j`로 나눴다. negative/nonfinite,
clamp, fallback은 모두 0이다.

## 3. E10 prefix와 직접 비교

### 3.1 edge와 B2→B0

| 항목 | E10 prefix | E12 full | 변화 |
|---|---:|---:|---:|
| sparse edge | 92,287 | **473,045** | 5.1258배, +380,758 |
| active input bins | 305 | **753** | UV center bins는 양쪽 모두 304/304 |
| B0 총 source-weighted 유입 | 3.81104e-4 | **6.04844e-4** | 1.5871배 |
| B2→B0 절대 유입 | 2.60244e-4 | **3.32208e-4** | 1.2765배 |
| B2의 B0 유입 점유율 | **68.2869%** | **54.9245%** | -13.3623%p |
| UV 입력만 놓은 B2 점유율 | 68.3015% | 63.4783% | -4.8232%p |
| B2 terminal 중 B0 조건부 비율 | 6.3147% | **8.0609%** | +1.7462%p |

full B0 유입의 나머지는 B1 21.3165%, optical 11.3642%, B0 6.5449%, B3 2.2429%,
EUV 1.8770%, B4 1.4960%, IR 0.2340%다. B2는 여전히 최대 채널이며 k-packet energy
규모 2.04205%의 26.90배다.

대표 shell-group의 raw on-grid B2 output 중 B0 비율은 deep 0--4가 8.6942%,
photospheric 5--12가 8.3476%, envelope 13--49가 25.4430%다. shell 8 source에는 E11
계약대로 global operator를 적용했으며, 이 group 값들은 정규화 분모가 없는 독립 구조
감사다.

### 3.2 상위 채널 순위

동일한 동시대 line-return source로 두 matrix를 다시 가중했다. 비율은 UV input에서
제거한 전체 power에 대한 점유율이다.

| 채널 | prefix 순위/점유율 | full 순위/점유율 | 변화 |
|---|---:|---:|---:|
| B2→B2 | 1 / 36.4044% | 1 / 32.6006% | 유지 |
| B2→B1 | 3 / 9.9434% | **2 / 14.1536%** | +1 순위 |
| B2→B3 | 2 / 13.3119% | 3 / 8.7474% | -1 순위 |
| B1→B2 | 4 / 9.5724% | 4 / 8.2015% | 유지 |
| B1→B1 | 5 / 5.3768% | 5 / 7.3201% | 유지 |
| **B2→B0** | **6 / 4.3066%** | **6 / 5.4975%** | 유지, 점유율 증가 |
| B2→optical | 9 / 1.7926% | 8 / 3.1715% | +1 순위 |
| B1→B0 | 13 / 미표시 | **9 / 2.1336%** | +4 순위 |

상위 구조는 일부 순위 교환만 있고 B2→B0 제거는 없다. 오히려 B1→B0가 상위권으로
올라 B0 유입이 넓어졌다. 전체 90개 channel 표는
`validation/emiss_e12/channel_rank_comparison.csv`에 있다.

## 4. 사전등록과 exact stage31

exact-bin 적용 전에 사용자 차터의 네 가설과 문턱, 입력 SHA를
`validation/emiss_e12/preregistration.json`에 고정했다. SHA-256은
`08faec80865986934ed4edd32633acb0dbfb8cdd98f068099b181277baa4741e`다.
E10처럼 broad-band collapse로 만든 사전 coarse source ratio는 B0 2.86206, B1
1.22128, optical 0.82744였다. 즉 exact 적용 전부터 무편향 matrix가 원하는 방향과
반대일 가능성을 숨기지 않았다.

shell 8, `nmu=16`, `T_inner=10020 K`, `bb_scale=1`로 동일 CPU formal solve를 했다.
same-capture E9와 E12 모두 3회 byte-identical이다.

| 대역 | E9 `J_det/CMFGEN` | E12 `J_det/CMFGEN` | E12/E9 J | source E12/E9 | 판독 |
|---|---:|---:|---:|---:|---|
| B0 600--1000 Å | 8.29055106 | **26.43249460** | **3.18826751** | 2.76312905 | 악화 |
| B1 1000--1500 Å | 4.91614286 | **5.65886463** | **1.15107815** | 1.15914087 | 악화 |
| B2 1500--2000 Å | 1.83988084 | 1.69129789 | 0.91924317 | 0.71730518 | 개선 |
| B3 2000--2500 Å | 0.20836087 | 0.60935745 | 2.92452928 | 3.69486824 | 1 쪽 이동 |
| B4 2500--3000 Å | 0.33680469 | **1.48272165** | 4.40231896 | 3.16799625 | 1을 넘어 과상승 |
| BALL 600--3000 Å | 0.93228813 | 1.52867739 | 1.63970487 | 1.04654970 | closure 이탈 |
| optical 3000--10000 Å | 6.92103893 | **6.37416008** | **0.92098313** | 0.74649657 | 목표 반대 |

E10 prefix의 B0 20.90950보다도 E12 full 결과가 26.43249로 26.41% 높다. B1도 하락이
아니라 15.11% 상승하고 optical J는 7.90% 하락했다. B2와 B3의 국소 개선만으로 형상
gate를 통과했다고 볼 수 없다.

stage31 guard는 `transport_residual=8.3213064e-7`, `source_residual=0`,
`source_iterations=1`, clamp/nonfinite/sign_uncertain/solution_negative_excess=0이다.
완전한 1000-row table과 exit 0으로 trip=0, 1208 Å 재발=0으로 판독했다.
`bdf_eta_negative=365482`, subtruncation=123247,
sign-indeterminate subtruncation=974101, enclosure restart=1624는 숨기지 않았다. E12 raw
`J_det` 음수 bin은 0(E9 baseline은 guard 안의 2개)이다.

## 5. emergent 연결

UV input만 한 번 full operator를 겪는 source-weighted destination은 UV 0.94202583,
optical 0.05274040, EUV 0.00431680, IR 0.00082402, outside 0.00009295이며 합 closure는
0이다. 이를 현재 emergent 점유율에 한 번만 적용하면 다음과 같다.

```text
UV:    42.9% -> 40.4129%  (-2.4871 point)
blue:   5.8% -> <=8.0626% (+2.2626 point, optical 전부를 blue로 놓은 상한)
```

UV 목표 23.8%까지 필요한 -19.1 point의 **13.02%**, blue 목표 14.5%까지 필요한 +8.7
point의 **최대 26.01%**에 해당한다. 같은 retention이 매번 독립적으로 반복된다는
비물리적 단순 모델에서도 UV 목표에는 9.87회의 유효 상호작용이 필요하다.

이는 shell-8 단일-pass source 장부일 뿐 emergent flux 계산이 아니다. escape, shell
migration, population/opacity feedback, 반복 interaction을 포함하지 않고 optical
3000--10000 Å도 역사적 blue diagnostic보다 넓다. 따라서 실제 emergent UV/blue 값은
**UNRESOLVED**이며 위 blue 값은 상한이다.

## 6. 잔여 원인 순위와 판독

1. **EPAY/activation-owner 재형상.** 무편향 conditional branch 자체가 UV를 94.2%
   유지하고 B2/B1에서 B0/B1으로 상당한 power를 보낸다. 다음 설계에서 어느 activation
   energy와 owner가 matrix에 들어가는지를 우선 재형상해야 한다.
2. **선 투영과 source--matrix covariance.** global all-shell operator를 shell-8 frozen
   line-return proxy에 적용하는 진단 계약은 유지했지만, 실제 line/shell-resolved
   projection은 없다. coarse→exact 차이와 stage31 비국소 증폭은 이 항목을 2순위로 둔다.
3. **빈 폭.** `eta*Delta_nu` 적용, side ledger, `~1e-14` source closure로 산술 누락은
   지지되지 않는다. 다만 엄격 `2e-13` 열 guard를 4.50e-15 초과했으므로 형식적 마지막
   잔여로 남고 차터의 최종 엄격 판정을 UNRESOLVED로 만든다.

형상이 CMFGEN 쪽으로 이동하지 않았으므로 “구조 수리 설계 근거 완성”은 거부한다.
LFMAT 계약 내 물리 결과는 생산 상신에 반대하는 FAIL 증거이며, 동일 E10 열 guard를
고집한 차터 최종값은 UNRESOLVED다.

## 7. 재현 명령

아래는 생산 실행 없이 기존 capture만 읽는다.

```bash
E12_RUN=/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828
E12_OUT=validation/emiss_e12
E12_CMF=/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4

sha256sum -c "$E12_RUN/fluor_matrix_iter10.sha256"
python3 scripts/emiss_e11_fluor_matrix.py "$E12_RUN/fluor_matrix_iter10"
python3 scripts/cmf_chieta_check.py "$E12_RUN/chieta_iter10"
python3 scripts/cmf_chieta_check.py "$E12_RUN/emiss_ab_iter10.A"
python3 scripts/cmf_chieta_check.py "$E12_RUN/emiss_ab_iter10.B2"

# same-capture E9 frozen baseline 재직렬화. plasma CSV는 payload 구성에는 쓰이지 않고
# 기존 E9 script의 sensitivity metadata를 만족시키는 같은 deck companion이다.
E12_STAGE=$(mktemp -d /tmp/emiss_e12_rebase.XXXXXX)
ln -s "$E12_RUN/emiss_ab_iter10.A" "$E12_STAGE/emiss_ab_iter10.A"
ln -s "$E12_RUN/emiss_ab_iter10.A.manifest.json" \
  "$E12_STAGE/emiss_ab_iter10.A.manifest.json"
ln -s "$E12_RUN/lumina_ma_line_destruct.csv" \
  "$E12_STAGE/lumina_ma_line_destruct.csv"
ln -s /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/lumina_plasma_state.csv \
  "$E12_STAGE/lumina_plasma_state.csv"
python3 scripts/emiss_e9_prediction_design.py --run "$E12_STAGE" \
  --out-dir "$E12_OUT/e9_same_capture"

# exact application/stage31 전에 고정
python3 scripts/emiss_e12_preregister.py \
  --matrix "$E12_RUN/fluor_matrix_iter10" \
  --e9-payload "$E12_OUT/e9_same_capture/emiss_e9_effective_iter10" \
  --source-payload "$E12_RUN/emiss_ab_iter10.A" --out-dir "$E12_OUT"
sha256sum "$E12_OUT/preregistration.json"

# 엄격 E10 column guard: expected exit 2 / UNRESOLVED
python3 scripts/emiss_e10_apply_redistribution.py \
  --e9-payload "$E12_OUT/e9_same_capture/emiss_e9_effective_iter10" \
  --source-payload "$E12_RUN/emiss_ab_iter10.A" \
  --matrix "$E12_RUN/fluor_matrix_iter10" --matrix-format formal \
  --preregistration "$E12_OUT/preregistration.json" \
  --out-dir /tmp/emiss_e12_strict_guard

# LFMAT001 정식 reader 계약 허용오차를 명시한 보조 실행; 보정/renormalization 없음
python3 scripts/emiss_e10_apply_redistribution.py \
  --e9-payload "$E12_OUT/e9_same_capture/emiss_e9_effective_iter10" \
  --source-payload "$E12_RUN/emiss_ab_iter10.A" \
  --matrix "$E12_RUN/fluor_matrix_iter10" --matrix-format formal \
  --column-closure-tolerance 2e-12 \
  --preregistration "$E12_OUT/preregistration.json" --out-dir "$E12_OUT"
python3 scripts/cmf_chieta_check.py "$E12_OUT/emiss_e10_redistributed_iter10"

gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -D_POSIX_C_SOURCE=200809L -Isrc scripts/stage31_cmf_field_driver.c \
  src/lumina_cmf_field.c -lm -o /tmp/stage31_cmf_field_driver_e12

/tmp/stage31_cmf_field_driver_e12 \
  "$E12_OUT/e9_same_capture/emiss_e9_effective_iter10" \
  "$E12_OUT/e9_same_capture/emiss_e9_effective_iter10.manifest.json" \
  8 16 10020 1 "$E12_OUT/jdet_e9_same_capture_s8.tsv"
/tmp/stage31_cmf_field_driver_e12 \
  "$E12_OUT/emiss_e10_redistributed_iter10" \
  "$E12_OUT/emiss_e10_redistributed_iter10.manifest.json" \
  8 16 10020 1 "$E12_OUT/jdet_redistributed_s8.tsv"

# 각 명령을 두 번 더 실행하고 cmp; 두 payload 모두 3/3 byte-identical
sha256sum "$E12_OUT/jdet_e9_same_capture_s8.tsv" \
  "$E12_OUT/jdet_redistributed_s8.tsv"

python3 scripts/emiss_e10_jdet_measure.py \
  --payload "$E12_OUT/emiss_e10_redistributed_iter10" \
  --jdet "$E12_OUT/jdet_redistributed_s8.tsv" \
  --e9-jdet "$E12_OUT/jdet_e9_same_capture_s8.tsv" \
  --preregistration "$E12_OUT/preregistration.json" \
  --source-measurement "$E12_OUT/source_band_measurement.csv" \
  --cmf-run "$E12_CMF" --out-dir "$E12_OUT"

python3 scripts/emiss_e12_diagnose.py \
  --formal "$E12_RUN/fluor_matrix_iter10" \
  --e9-payload "$E12_OUT/e9_same_capture/emiss_e9_effective_iter10" \
  --source-payload "$E12_RUN/emiss_ab_iter10.A" \
  --preregistration "$E12_OUT/preregistration.json" \
  --application-summary "$E12_OUT/redistribution_application_summary.json" \
  --stage31-summary "$E12_OUT/stage31_summary.json" --out-dir "$E12_OUT"
```

주요 산출물 SHA-256은 same-capture E9 J_det
`933710b1...e7bd79`, E12 redistributed payload `f6ac52e0...7265e7`, E12 J_det
`f96ec0aa...278eec`, 최종 diagnosis JSON `5d5818e5...3a44bf`다.
