# A2-06 구현 명세 v5 (최종) — V4 잔여 2건 확정

기준 HEAD=d8b9870. 우선순위 V5 > V4 > V3 > V2. 4차 검수 잔여만 대체한다.

## 1. 처분표 행번호 정정 (V4 §3 표의 3개 오기)

- census 4556 W 의 현행 = **4611** (4610 아님; 4610 에는 W 읽기 없음 — W/T_rad 둘 다
  4611 에서 읽음)
- census 13739 W 의 현행 = **13791** (13789 는 NLTE 분기; fallback W 는 13791)
- 진단 3행의 심볼 읽기 행 = **13972 / 13992 / 14132** (V4 의 13970/13990/14131 은
  호출 시작행 — 원장에는 심볼 읽기 행을 기재)

나머지 13행·실체·처분(A2-06 6 / A2-07 6 / A2-08 4)·census-밖 이관 9지점·잔류
허용목록은 V4 §3 그대로. 구현 diff 에서 전 행 재실측 후 원장 기재.

## 2. fine closure cohort (V4 §5 대체)

cohort = `validation/a2_02c/A2_02C_ESTIMATOR_COHORT.json` 의 동결 audit 레코드
(활성 74건), 결박 = `validation/a2_02c/A2_02C_RESOLUTION_INPUT.json` 의 SHA-256
`0c029c…c3e69` (전문 인용). `A2_02C_LINE_CENSUS.json` 은 BB census 원장이지 audit
cohort 가 아니다(V4 오기).

## 3. f_cov·E_1·P95·동률 규약 (V4 §5 L-1bb 절 대체)

- **f_cov (coverage)**: `F_truth(line,shell) = n_l(POP)·B_lu·J̄_truth` (J̄_truth =
  EDDFACTOR fine 격자 φ 직적분, 오프라인). 활성집합 = F_truth 내림차순 누적
  ≥ 0.999·ΣF_truth 가 되는 **최소 접두사**; 경계 동률(같은 F_truth 값)은 전부 포함.
  `f_cov = Σ_{active∩usable} F_truth / Σ_active F_truth` (usable = view
  MEASURED/EXACT_ZERO). CMFGEN 값 C 는 coverage 에 등장하지 않는다.
- **E_1 (채널별)**: ORDER §6.3 규약 그대로 — 채널 X ∈ {J̄, R_lu, R_ul^stim} 각각
  `E_1(X) = Σ_active w|X_lum − X_cmf| / Σ_active w|X_cmf|`, w = F_truth (활성집합
  가중 freeze). 분모의 C = 해당 채널의 CMFGEN 값이다.
- **E_sym P95**: 활성집합 (line,shell) 항별 `E_sym = 2|L−C|/(|L|+|C|)` 값들의
  95백분위 — numpy `percentile(…, 95)` 기본(선형 보간) 규약 명시. 양쪽 0 = 0,
  한쪽 0 = false-positive 로 별도 FAIL 행(ORDER 규약).

이외 전 조항 = V4/V3/V2 유효. 구현 순서·제약 = V3 §5.
