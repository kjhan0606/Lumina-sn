# A2-06 폐합 기록 — CPU bound-bound rate: J̄ selective estimator + LineJbarCache 이관

2026-08-06. 명세 `docs/SPEC_A2_06_V5.md`(승계 V5>V4>V3>V2; Codex 검수 5라운드
BLOCK 7→4→4→2→APPROVE). 체제=개정 9(저작·구현=Codex / 검수=fable 운전석).
기준 HEAD=d8b9870.

## 계약과 구현
- production J̄ = 전역 4000빈 재적분이 아니라 **같은 raw measure 의 φ-가중
  selective estimator**(Q_g SHA 결박·Gaussian v_D=10km/s ±4D 닫힌형 세그먼트 적분·
  패킷-모집단 분산 s²=(Q−S²/N)/(N−1)) — `src/line_jbar.{h,c}` (운전석 구현).
- **dual-view 원자 commit**: J_ν 와 LineJbarCache 가 한 commit 으로만 전이;
  partial-commit 주입 양방향 불변성 selftest PASS — `src/radiation_field.{h,c}`.
- transport 훅 = 정본 accumulator 인접 단일점(ν′ 양끝점 전달), 적립 오류 latch →
  commit 거부.
- 소비 이관 6+9지점(Codex 구현): 공통 소비 함수 `nlte_bb_jbar_canonical`
  (`lumina_plasma.c:514-561`) — R_lu=B_lu·Ĵ̄ / R_ul^stim=B_ul·Ĵ̄ / R_ul^sp=A_ul 분리,
  legacy 는 전량 진단 shadow 강등(rate 에 무기여 — 이중계상 없음 검수 실측),
  비-OK/MISS/UNSAMPLED = 무기여+원인별 카운터(fallback 0건). population fallback
  6행(A2-07)·line-source 4행(A2-08)은 불변 — 원장 ADDENDUM 재배치.

## 게이트 실측
- 정합 3종 PASS: same-measure 해시 · projection closure 8대역 · fine cohort
  74건(A2_02C_ESTIMATOR_COHORT, SHA 결박).
- 음성대조 9종 전부 기대 FAIL 관측 (개정 §3.5 7종 + §5.4 static read-trace·
  partial-commit).
- **A_ul crosswalk PASS**: 정본 ftos 덱 2,220,953선 전량 match(truth 3,617,414 전이),
  unmatched 0·한쪽-0 행 0·최대 상대오차 1.72e-16 ≤ 1e-10. label SHA 결박은 진단
  전용(부재 시 경고+비활성, crosswalk=BLOCKED_MISSING_LABEL_BINDING — fail-closed 는
  게이트에서).
- **L-1bb 최종 = `BLOCKED_MISSING_RATE_EXPORT`** (NETRATE/TOTRATE 부재; PASS 승격
  금지, 도착 시 판정 스키마 사전등록: coverage≥0.95·E_1≤0.10 각각·E_sym P95≤0.25).

## 검수 (fable 운전석)
- 결함 1건 적발·수리: label 결박 로더 fail-closed 오배치 → D 배터리 전멸
  ('0 != 25620' FATAL). V4 §4 위반 판정 → Codex 수리(경고+비활성+게이트 BLOCKED) →
  배터리 재검 PASS.
- 회귀 전판 PASS: 배터리 36(lageunha) · selftest 5종(a2_03/04/05/06×2) ·
  L-0 replay · L-1bf(A2-05) 게이트+음성대조 · make lumina.

## 미결 (등재)
- L-1bb 물리 판정 = O-PHYS NETRATE/TOTRATE 도착 후.
- MC estimator 통계 수렴 증거(P→2P 사다리)는 estimator 표본 확충 후(A2-05 CHAIN 과
  동일 사유).
- replay lane 의 line 블록(CMFGEN_REPLAY provenance) production 배선은 A2-08 접점.
