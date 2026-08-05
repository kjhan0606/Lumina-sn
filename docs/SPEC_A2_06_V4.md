# A2-06 구현 명세 v4 (최종) — V3 잔여 5건 확정

기준 HEAD=d8b9870. V2 §1·§2-B1(노선)·V3 전체를 승계하고, 3차 검수 잔여 5건만 대체한다.
우선순위: V4 > V3 > V2.

## 1. CI/SE 규약 (V3 §1.1·§3.3 의 M2 규약 대체) — 선례 정합

A2-02C gate2 선례(`scripts/a2_02c_segment_replay.py:476-503,947`)를 그대로 승계:
분산 모집단 = **패킷 단위, 무기여 패킷 포함**.

- accumulator per (line, shell): `line_sum = Σ_p y_p`, `line_sumsq = Σ_p y_p²`,
  `line_count = 기여 패킷 수` (y_p = 패킷 p 의 φ-가중 세그먼트 적분 합 — 패킷 종료
  시점에 sumsq 반영; transport thread-local 에 현 패킷 부분합 보관).
- N = 그 generation 의 총 패킷 수(무기여 포함; commit request 의
  `contribution_count` 와 별개로 `line_n_packets` 로 전달).
- `s² = (Σy² − (Σy)²/N) / (N−1)`, `Var(Ĵ̄) = norm²·N·s²`,
  `norm = 1/(4π V_s Δt)`, `SE = sqrt(Var)`, CI 반폭 = 1.96·SE.
- commit line 블록 필드명 갱신: `line_raw`→`line_sum` + `line_sumsq` +
  `line_n_packets` (M2/Welford 삭제). view 조회 반환의 se 도 이 식.

## 2. q_set_hash 타입 (V3 §1.1/§1.2 대체)

스키마(`radiation_field.h:126` `const char *q_set_hash`) 그대로 **SHA-256 hex
문자열**. commit 입력·view 인자 모두 `const char *`, 비교는 strcmp. uint64 변환
규약은 폐기.

## 3. census 16행 1:1 처분표 (V3 §2 표 대체 — 현행 행번호 실측 정정)

| # | census 행(bafd2bb) | 심볼 | 현행(d8b9870) | 실체 | 처분 |
|---|---|---|---|---|---|
| 1 | lumina_plasma.c:4556 | W | 4610 | MA 상향률 W/T_rad 읽기 | A2-06 이관 |
| 2 | lumina_plasma.c:4556 | T_rad | 4611 | 〃 | A2-06 이관 |
| 3 | lumina_plasma.c:4596 | W | 4651 | LTE 비교장 진폭 | A2-06 이관 |
| 4 | lumina_plasma.c:4596 | T_rad | 4651 | LTE 비교장 색 | A2-06 이관 |
| 5 | lumina_plasma.c:4701 | W | 4756 | 선 상향 복사율 | A2-06 이관 |
| 6 | lumina_plasma.c:4701 | T_rad | 4756 | 〃 | A2-06 이관 |
| 7 | lumina_plasma.c:4879 | T_rad | 4934 | population fallback 지수 | A2-07 재배치 |
| 8 | lumina_plasma.c:4880 | W | 4935 | population fallback 희석 | A2-07 재배치 |
| 9 | lumina_plasma.c:11908 | W | 11960 | line-source fallback | A2-08 재배치 |
| 10 | lumina_plasma.c:11908 | T_rad | 11960 | 〃 | A2-08 재배치 |
| 11 | lumina_plasma.c:11915 | W | 11967 | blanketed heating 빈 장 | A2-08 재배치 |
| 12 | lumina_plasma.c:11915 | T_rad | 11967 | 〃 | A2-08 재배치 |
| 13 | lumina_plasma.c:12093 | W | 12145 | lower-level population fallback | A2-07 재배치 |
| 14 | lumina_plasma.c:12100 | W | 12152 | upper-level population fallback | A2-07 재배치 |
| 15 | lumina_plasma.c:13739 | W | 13789 | coupled lower population fallback | A2-07 재배치 |
| 16 | lumina_plasma.c:13743 | W | 13795 | coupled upper population fallback | A2-07 재배치 |

census-밖 A2-06 이관(ADDENDUM 신설, 1차 검수 전수): 4633 그룹(jbar_line_det/
jbar_line/coarse 선택) · 4661(jblue_line) · 4731(B_lu−B_ul 의 W/T_rad) · 10827 ·
12182 · 13823 · 15238 · 15292 · 15361. 잔류 허용목록: 진단 shadow 15457·15591
(행 신설 KEEP_DIAGNOSTIC_READ) · 진단 3행 현행 13970/13990/14131 · cmfgen
3153/3159(A2-08) · GPU(A2-12/13) · falsifier. 구현 diff 에서 각 행 재실측 확정.

## 4. A_ul crosswalk 연결 규약 (V3 §3.7 대체)

연결 = ORDER §6.2 그대로 **원소·이온·level label(원자자료 로드 시 원본 config
label 의 해시를 레벨 인덱스에 결박해 적재)·에너지·통계가중치**. 에너지 최근접은
label 일치 후보가 복수일 때의 타이브레이커로만. label 해시 결박이 로더에 없으면
이 단계에서 추가한다(진단 전용 테이블 — 런타임 물리 불변). 판정
`|A_lum−A_cmf| / max(|A_lum|, |A_cmf|) ≤ 1e-10`; 양쪽 0=일치, 한쪽 0=FAIL 행,
UNMATCHED 는 목록 보고(삭제 금지).

## 5. cohort·모집단 사전등록 (V3 §3.5/§3.8 보강)

- projection closure 8대역: canonical 격자 [1.4402928950097124e12,
  4.032418413741097e16] Hz 를 로그 균등 8분할(경계 = geomspace 9점, 사전등록 수치
  게이트 산출물에 기재). φ^(N)_k = k번째 대역 내 상수 1/(대역 Hz 폭), 대역 밖 0.
- fine closure cohort = `A2_02C_LINE_CENSUS.json` 감사 라인 집합(변경 금지, hash 인용).
- L-1bb 모집단·가중(도착 시 판정용 사전등록): 모집단 = truth-측 활성집합(V3 §3.6
  과 동일 구성 — n_l·B_lu·J̄_truth 누적 99.9%), flow coverage 분모 = Σ w|C|
  (w=truth flow, C=CMFGEN 값; ORDER §6.3 부호 규약), E_sym P95 = 그 모집단 내
  (line,shell) 항별 E_sym 의 95백분위.

이외 전 조항은 V3 유효. 구현 순서·제약 = V3 §5.
