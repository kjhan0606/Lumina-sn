# A2-02C estimator 게이트 개정 2호 — 구현 보고

- 작성일: 2026-08-05
- 계약: 개정 2호 저작과 판정부 구현을 한 승인 발주로 집행
- 변경 파일: `docs/ORDER_A2_GRID_AMENDMENT_BY_CODEX.md`,
  `scripts/a2_02c_segment_replay.py`, 이 보고서
- 불변 확인: `src`, 덱, `/gpfs`, 적분, estimator, capture 판독은 변경하지 않음
- commit/push: 하지 않음

## 결과

종전 line별 `delta<=1%`, 중앙값 `<=0.2%`, `invalid-eligible=0` 판정을 통계적 z 게이트,
flow-가중 집계 게이트 및 사유별 coverage 보고로 교체했다. 기존 line별 delta는 결과 JSON의
`legacy_per_line_delta_diagnostic`에 남지만 `judgment=false`다. 결과 schema는
`lumina-a2-02c-segment-replay-gate2-v2`, 권장 artifact 이름은
`a2_02c_estimator_effort_gate2_result.json`이다. 종전 v1 artifact는 읽거나 덮어쓰지 않는다.

## 개정 부록 전문

아래는 `ORDER_A2_GRID_AMENDMENT_BY_CODEX.md` 말미에 append한 규범문 전문이다.

### 효력, 근거와 변경 경계

이 부록은 §4.4의 estimator effort 및 독립 재현 판정만 대체하는 개정 2호다. §3.1의
적분식, segment trajectory 적분, packet normalization, variance estimator, raw capture
판독, canonical projection 및 fine histogram 작성법은 바꾸지 않는다. §4.3 전역 격자
사다리와 그 최대 1%·중앙값 0.2% 합격선도 바꾸지 않는다. 이 개정 2호에 한해 문서
머리말의 구현 제외 범위를 대체하고, 같은 승인 계약 안에서
`scripts/a2_02c_segment_replay.py`의 판정부와 self-test 구현을 허가한다. `src`, 덱,
`/gpfs` 및 estimator/capture 생산 경로는 여전히 변경 대상이 아니다.

production replay `P=200,000`, `2P=400,000`, 사전등록 cohort 74선의 `rc=3`을 다음 네
독립 판별로 재분해했다.

1. same-measure는 PASS했고 canonical projection closure `δ`는 정확히 `0.0`이었다.
2. 두 fine 게이트의 유효 수치는 최대 `0.0074`, `0.0048`로 통과 범위였고, 종전
   `invalid-eligible=17 (UNSAMPLED)` 처분만 FAIL을 만들었다.
3. prefix `z=(Jhat_P-Jhat_2P)/(sigma_P/sqrt(2))`는 50개 유효선에서 평균 `-0.21`,
   표준편차 `0.89`, `|z|>3` 0건이었고, 부호는 `+21/-29 (p=0.32)`였다.
4. 관측 `δ/[1/sqrt(2 count)]` 중앙값은 `1.00`이며, 독립 분산 예측 중앙값 `10.2%`와
   독립 replay 실측 중앙값 `9.9%`가 일치했다.

line별 표본수는 중앙값 66, 최솟값 0, 최댓값 5,048이었다. 고표본 13선에서 계통 성분
검출 한계는 약 2% 이하였다. 따라서 종전의 line별 `δ<=1%`, 중앙값 `<=0.2%`는
estimator 편향보다 포아송 표본 잡음의 크기를 판정한 잘못된 검정이다. 이 수치는 이후
진단 열에는 보존하지만 PASS/BLOCKED 판정에서는 제외한다.

### line별 z 게이트

고정 RNG prefix 쌍은 모든 유효선에 대해 다음을 계산한다.

\[
 z_i={\widehat{\bar J}_{i,P}-\widehat{\bar J}_{i,2P}
       \over \sigma_{i,P}/\sqrt{2}},\qquad |z_i|\le3.
\]

`sigma_P`는 replay가 이미 기록하는 `standard_error`이며 새 분산 추정이나 재조정은
금지한다. 분모와 차이가 모두 0이면 `z=0`; 분모만 0이고 차이가 비0이면 무한 outlier로
처분한다. 모든 유효선을 분류하고 `|z|>3`이 없어야 하며, 동시에 outlier 수가 양측 정규
꼬리확률 `p=erfc(3/sqrt(2))=0.002699796...`인 이항 기대와 양립해야 한다. 양립 검사는
단측 exact upper-tail `P[X>=k]>=0.05`로 한다.

동일 effort 독립 레인에는

\[
 z_i={\widehat{\bar J}_{i,a}-\widehat{\bar J}_{i,b}
       \over\sqrt{\sigma_{i,a}^2+\sigma_{i,b}^2}},\qquad |z_i|\le3
\]

를 적용한다. z의 부호·평균·표준편차, 유효선 수, outlier 수·비율 및 exact-binomial
p-value를 기록한다. 종전 line별 `δ`는 진단 전용으로 계속 기록한다.

### flow-가중 집계 게이트

값을 보기 전에 cohort에 등록된 비음수 유한 가중 `w_i`로

\[
 J_{agg}={\sum_i w_i\widehat{\bar J}_i\over\sum_{i\in cohort}w_i},\qquad
 \delta_{agg}={|J_{agg,a}-J_{agg,b}|\over|J_{agg,b}|}
\]

를 만든다. 등록 aggregate의 최대 `δ<=1%`, 중앙값 `<=0.2%`를 모두 요구한다. 현재
등록은 전체 cohort aggregate 하나이므로 한 값이 최대와 중앙값 양쪽에 기록된다. prefix와
독립 비교가 각각 통과해야 한다.

명시 가중 cohort는 모든 active 행에 `flow_weight` 또는 `cohort_weight`가 있어야 한다.
cohort-level `flow_weights_frozen_before_values=true` 또는 동형 contract도 필요하며,
부분 등록이나 freeze 선언 누락은 schema FAIL이다. 명시 필드 전의 frozen cohort v1은 각 등록 행 자체를
단위가중 `1.0`의 사전등록으로 해석한다. replay 값, count 또는 validity로 가중을 만들거나
바꿀 수 없다. 결과에 가중 출처, 등록 행 수와 전체 가중합을 기록한다.

### `UNSAMPLED` 및 저표본 처분

어느 비교 레인에서든 `sample_count<10`이거나 `validity=UNSAMPLED`인 line은 line별 z와
fine 수치 게이트에서 사유를 붙여 제외한다. 이를 `EXACT_ZERO`로 승격하지 않으며
`invalid-eligible`로 PASS를 죽이지 않는다. 결과에는 74개 등록선 중 유효·제외 선 수,
제외 record 전량, 원소/이온 및 `100–450`, `450–918`, `918–1290`, `1290–2000`,
`2000–10000`, `10000–20000 Å` 파장대별 coverage를 기록한다.

flow 집계에서는 제외선의 numerator 기여를 넣지 않되 전체 등록 가중 분모를 줄이지
않는다. 각 레인의 included weight·record·sample count와 전체 등록 가중 대비 coverage를
기록한다.

### production 잡음 자산과 하류 입력

production의 `sample_count=66` 부근 line별 `δ` 중앙값 약 12%를 검증된 MC 잡음 자산으로
등재한다. A2-06은 이를 damping, iteration averaging, 수렴 감시 및 packet effort 설계의
정량 입력으로 사용한다. 표본수를 무시한 1% line별 정지조건, variance inflation 또는
결과 후 cohort/가중 변경으로 잡음을 숨기면 FAIL이다.

### 음성대조

종전 7종은 유지한다. 독립 레인 상수 편향 3% 주입은 z 게이트가 FAIL해야 한다. 기록
`sigma`만 10배 부풀린 fixture는 z가 통과하더라도 flow 집계 `1%/0.2%` 게이트가 FAIL해야
한다. 총 9종 모두 비0 종료와 `A2_02C_REPLAY_NEGATIVE_FAIL` marker를 내야 한다.

### artifact, schema와 최종 판정식

종전 `a2_02c_estimator_effort_result.json`/`lumina-a2-02c-segment-replay-v1` BLOCKED
artifact는 보존한다. 개정 2호는 `a2_02c_estimator_effort_gate2_result.json` 및
`lumina-a2-02c-segment-replay-gate2-v2`를 쓰고 `amends_after`로 종전 artifact/schema를,
`historical_amends_after`로 `43ffe31`을 참조한다.

```text
selective_jbar_effort_ladder_PASS =
  prefix_z_gate_PASS
  AND prefix_flow_weighted_aggregate_gate_PASS
  AND independent_z_gate_PASS
  AND independent_flow_weighted_aggregate_gate_PASS
  AND same_measure_commit_gate_PASS
  AND canonical_projection_closure_PASS
  AND fine_histogram_resolution_convergence_PASS
  AND fine_diagnostic_closure_PASS
  AND estimator_negative_controls_9_of_9_PASS
```

이 사유 제외 및 coverage 보고가 `invalid-eligible=0`의 종전 문구를 대체한다. 나머지
A2-02 최종 판정항과 하류 보류 조건은 그대로다.

## 구현 diff 요약

| 영역 | 변경 | 불변 |
|---|---|---|
| schema/artifact | replay v2 schema, `amends_after`와 새 출력 이름 | 종전 v1 artifact |
| prefix 판정 | `sigma_P/sqrt(2)` z, 전 유효선 `|z|<=3`, exact binomial | P가 2P의 literal packet prefix인 선택법 |
| 독립 판정 | `sqrt(sigma_a^2+sigma_b^2)` z | capture 및 estimator 계산 |
| flow 집계 | 등록 가중 분모, 1%/0.2%, lane별 coverage | 합격선 수치 |
| validity | count<10/UNSAMPLED 사유 제외, 이온/파장 coverage | validity 원값과 line record |
| fine 게이트 | 저표본 사유 제외 후 기존 수치 판정 | fine histogram 및 적분 |
| 진단 | 종전 per-line delta 유지, 판정 제외 | delta 계산식 |
| self-test | z, 이항, σ 팽창 backstop, denominator 보존 | segment split/profile 적분 검사 |

병렬 worker 수, `spawn` pool, cohort 진행률 로그, capture/global chunk 진행률 로그는 그대로다.

## 음성대조 표

| # | fixture | 기대 검출 | 관측 marker |
|---:|---|---|---|
| 1 | legacy Jbar schema | schema | `NEGATIVE_FAIL` |
| 2 | cohort/Q hash swap | binding | `NEGATIVE_FAIL` |
| 3 | mandatory Fe II 제거 | census | `NEGATIVE_FAIL` |
| 4 | label-only fake P→2P | prefix ledger | `NEGATIVE_FAIL` |
| 5 | line/profile swap | identity | `NEGATIVE_FAIL` |
| 6 | median-only false PASS | 1% maximum | `NEGATIVE_FAIL` |
| 7 | stale union edge | edge hash | `NEGATIVE_FAIL` |
| 8 | 독립 레인 +3% 상수 편향 | z/binomial | `NEGATIVE_FAIL` |
| 9 | `sigma` 10배 팽창, 값 0.5% 편향 | z 통과 후 aggregate 0.2% | `NEGATIVE_FAIL` |

실행 결과는 `A2_02C_REPLAY_NEGATIVE_SUMMARY passed=9 total=9`, wrapper rc=0이었다.
self-test marker는 `z_gate=1 binomial=1 aggregate=1 unsampled_coverage=1`, rc=0이었다.

종전 BLOCKED JSON에 이미 보존된 P/2P line record를 새 판정부로만 읽기 전용 재분류한
결과는 다음과 같다. capture 재생이나 산출물 쓰기는 하지 않았다.

| 항목 | 값 | 판정 |
|---|---:|---|
| prefix z 유효/제외 | 50 / 24 | coverage 보고 |
| z 평균 / 표본표준편차 | -0.210753 / 0.890793 | 순수 표본잡음과 정합 |
| `|z|>3` | 0 | PASS |
| cohort-v1 등록 단위가중 aggregate `δ` | 0.805079% | 최대 1% PASS, 중앙값 0.2% FAIL |

따라서 개정은 종전 per-line 포아송 잡음 오판을 제거하지만 현재 보존 자료를 억지로 PASS로
바꾸지는 않는다.

## lageunha 운전석 명령 — production 재판정 1회

아래의 `INDEPENDENT`에는 seed `62519117`로 이미 생산해 보존 중인 generation-2
독립 capture의 실제 경로를 넣는다. 두 capture는 읽기 전용이며 출력은 새 artifact다.

```bash
PRIMARY=/gpfs/kjhan/lumina_runner2/scratch/a2_02c/a2_02c_segments_g2_2P400000.bin
INDEPENDENT=/gpfs/kjhan/lumina_runner2/scratch/REPLACE_WITH_EXISTING_SEED62519117_CAPTURE.bin

ssh lageunha 'bash -s' -- "$PRIMARY" "$INDEPENDENT" <<'EOF'
set -euo pipefail
PRIMARY=$1
INDEPENDENT=$2
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
OUT=$REPO/validation/a2_02c
cd "$REPO"
test -r "$PRIMARY" -a -r "$INDEPENDENT"
test ! -e "$OUT/a2_02c_estimator_effort_gate2_result.json"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02c_segment_replay.py run \
  --capture "$PRIMARY" \
  --independent-capture "$INDEPENDENT" \
  --cohort "$OUT/A2_02C_ESTIMATOR_COHORT.json" \
  --union "$OUT/A2_02C_FREQUENCY_UNION.json" \
  --global-bins 4000 \
  --effort 200000 --double-effort 400000 \
  --chunk-records 1000000 \
  --output "$OUT/a2_02c_estimator_effort_gate2_result.json"
EOF
```

로그인 노드에서는 위 replay를 직접 실행하지 않는다. 현재 cohort v1 단위가중과 보존된
P/2P 값에서는 prefix aggregate가 중앙값 0.2%를 넘으므로 production 재판정의 기대 rc는
`3/BLOCKED`다. 새 네 판정(prefix z/aggregate, 독립 z/aggregate)과 기존
same-measure/canonical/fine가 모두 PASS하는 다른 정당한 사전등록 입력이면 rc는 `0`이고,
schema/hash/generation/identity 오류는 `2`다. 결과 후 가중을 바꾸어 rc를 0으로 만드는
행위는 금지한다.
