# A2-05 폐합 기록 — CPU bound-free 광이온율의 canonical J_ν view 이관

2026-08-06. 명세 `docs/SPEC_A2_05_V2.md` (Codex 검수 반박 7+공백 6 전부 계약화, 개정 8 체제:
저작·구현=운전석 / 검수=Codex). 기준 HEAD=bafd2bb.

## 계약 (단일)

CPU 생산 물리의 bf 광이온율이 `bf_rate_estimator`(C2)+dilute-BB fallback(C1) 대신
**정본 `RadiationField.J_nu` 의 checked read-view 를 직접 적분**한다.

## 구현 실측

- `radiation_field_read_view()` (R5): enabled·units/frame·epoch/shells·세대 삼중일치·격자
  유한성 전수 검사, 실패 시 view 미제공+구별 오류코드 (rate 0 반환 금지). 갱신 지점은
  commit 초크포인트 2곳뿐 (MC=lumina_main.c, replay=lumina_cmfgen.c).
- `src/bf_rate_jnu.c`: 보존형 Γ 적분기 — σ 구간별 선형 해석해(빈·σ 노드 양쪽에서 분할,
  threshold 부분빈 `[max(ν_th,ν_lo),ν_hi]` 정확), R6 validity(w_miss ≤1e-3→VALID,
  우선순위 STALE>UNSAMPLED>OOG), §6.3 CI 용 포아송 delta-method 분산(결정론 commit 은 0).
- `bf_rate_gamma_legacy_grid()`: 레거시 1000빈 σ 행(또는 Kramers 빈중심 평가)을 bin-상수
  계단 표(중복 edge 노드)로 재인코딩 — 생산과 게이트 픽스처가 **동일 산술** 공유.
- 소비 이관 **7지점** (스펙 6 + 실측 발견 1 = 스테이지-IV 들뜬준위 R_bf_hl): 원장
  `docs/A2_01_DISPOSITION_LEDGER.md` ADDENDUM. JEQB(source 2)는 falsifier 장치로 존치,
  GPU lookup 은 R4 허용 잔류(A2-13). CPU 생산 물리의 estimator 소비 = **0**.
- R6 downstream: 비-VALID 항은 값 대입 없이 무기여+카운터
  (`bf_view_blocked_{stale,unsampled,out_of_grid}` / `bf_view_rate_terms`).

## 게이트 실측

1. **selftest** `tests/a2_05_bf_rate_selftest.c` PASS — 해석해 4종(상수σ 닫힌형 1e-12 ·
   수소형 ν⁻³ 1e-6 · threshold 이동 1e-9 · 계단σ 1e-12) + R6 validity 전수 + read_view
   오류코드 전수(사전-commit 거부, 세대/epoch/shells/disabled).
2. **L-1bf 게이트** `scripts/a2_05_l1bf_gate.py` + `tests/a2_05_l1bf_fixture.c`
   (ORACLE_INPUT lane, EDDFACTOR→canonical 4000 결정론 commit, PRRR 대조,
   한계선=기존 인증 사전등록치 [0.5,2.0] 전셸+[0.8,1.25] s6–s8, bandmask 라이브 산출
   — S II s8 자동 제외): **PASS 전 셀**. 원장 `validation/a2_05/L1BF_GATE_LEDGER.json`.
3. **음성대조 3종 전부 기대 FAIL 관측** (runner rc=0): (a) W·B_ν(14172.549K) 주입 →
   31셀 한계 밖 (b) witness Fe III·Co III threshold +1빈 → ΔΓ/Γ(s0)=5.3%/5.6% ≫ 0.5%
   (c) α 밀도 1회 추가곱 → 등록 항등식 검출. α 실비교는
   `BLOCKED_MISSING_RATE_EXPORT`(재결합 이관 후).
4. **회귀**: 배터리 36케이스 PASS · A2-03 radfield selftest PASS · A2-04 commit selftest
   PASS · L-0 replay PASS(음성대조 5종 포함).

## 캠페인 첫 물리 수치 — 이관 전후 Γ 변화

Γ_view/Γ_legacy1000 (동일 EDDFACTOR 장·동일 σ·동일 population; J 소유권만 교체):

| ion | s0 | s4 | s8 |
|---|---|---|---|
| Co III | 0.9924 | 0.9890 | 0.9883 |
| Fe III | 0.9933 | 0.9893 | 0.9898 |
| S II | 0.9976 | 0.9952 | 0.9960 |
| S III | 0.9960 | 0.9943 | 0.9960 |

전 이온·전 셸 0.2–1.3% 하향의 일관 시프트 — 근원=threshold 부분빈의 정확 적분(레거시는
빈중심<ν_th 전체 드롭/전체 포함) + 4000빈 해상도. PRRR 대비 비율은 인증 표와 동등
(Co III 1.004–1.155 · Fe III 0.911–1.006 · S II 0.988–1.023 · S III 0.914–0.994).

## Codex 구현 검수 (1차 BLOCK 6건 → 2차 BLOCK 4건 → 전건 처분, 2026-08-06)

1차 판정 BLOCK(6 BLOCKER) 처분 후 2차 재검수가 4건을 STILL-BLOCKED 로 유지
(read_view 내부 edge·EXACT_ZERO+missing·Kramers 분기 편측·f_cov 순환/CHAIN
mech 미판정). 4건 전부 재수리(아래 5~9 항)·재실측 완료. 1·2차 처분 통합:
1. read_view: 실패 시 `*out` 전체 무효화 + canonical edge **앵커 비트일치**
   (NU_MIN/NU_MAX 상수 ==) 검사 추가 — 수리.
2. 비-VALID rate 소비: 값 대입 없이 무기여(수치 0)+카운터가 실행 가능한 유일 형태
   ("작은 값 대입 금지"의 의미). 관측성 요구는 수리 — `nlte_free` 종료 보고
   `[A2-05][BF-VIEW]` 라인(rate_terms/blocked_{stale,unsampled,oog}), 게이트의
   blocked population share 기재·분모 제외.
3. EXACT_ZERO+missing: 1차 처분은 반박이었으나 2차 검수가 기각 — 항목 6 으로 수리.
4. Kramers threshold 부분빈: **수리(2회)** — 계단 상수를 닫힌형 등가
   s\* = σ₀ν_th³(ν_th⁻³−hi⁻³)/(3 ln(hi/ν_th)) 로 정확화하되, 2차 검수가
   ν_c 기준 편측 적용을 적발 → 부분빈 판정을 (lo<ν_th<hi) 로 정정.
   selftest 는 양측 배치(위/아래) 모두 **물리 닫힌형 독립 기대값**으로 1e-11.
5. read_view 내부 edge(2차): **수리** — 4001개 edge 전수를 owner-init 과 동일식
   재계산 **비트 일치** 비교(변조 즉발).
6. EXACT_ZERO+missing(2차, 반박 기각됨): **수리** — w_miss∈(0,tol] 이면 상태=VALID
   (값 0·w_miss 기록), EXACT_ZERO 는 전 구간 관측·0 확정일 때만. selftest `zm-*`.
7. f_cov 순환성(2차): **수리** — 활성집합·분모를 truth-측 기여(인증의 1000빈
   CMFGEN-장 직교화 p·Γ_lev, view 상태 무관)로 재정의. ORACLE lane f_cov=1.0000 은
   이제 실측(비자명), CHAIN lane 은 0.000–0.061 로 정직 폭로.
8. CHAIN mech/자격(2차): **수리** — mech 체크 4종 실판정 + 자격 = CI ∧ f_cov≥0.999,
   미달 사유 명기(BLOCKED_INSUFFICIENT_SAMPLING/UNDERPOWERED 구분).
9. α 원장 원자료(2차): **수리** — ledger 에 α·n_e·n_ion·depth 기재.
10. 게이트 metric(1차): **수리** — E_1(A2-04 정의의 rate-공간 이식) 본선/독약 기록
    (본선 ≤0.134, 독약 0.9–1.0 > 한계 0.5 — undershoot 포화 반영), E_sym 대칭형,
    α 왕복은 실 PRRR 데이터(Co III s0: α=5.871e-12, n_e=4.848e9).
11. Makefile/러너 배선(1차): **수리** — SOURCES/HEADERS·CUDA·전 CPU 하니스 타깃·
    `run_zinert_selftest.sh` 에 `bf_rate_jnu.{c,h}` 배선, `make lumina` 실빌드 확인.

## CHAIN lane 실측 (2026-08-06 재판정, lageunha 32-worker, 46s)

고정 seed 캡처 `a2_02c_segments_g2_2P2400000.bin`(6.29억 레코드, 55GB) →
생산 MC commit → view → Γ 사슬. **기전 판정 PASS**(DONE marker·출력 완전성
27,702행·상태 전건 유효·STALE 0·fallback 0). **판정 자격 0/36** — 전 셀
`BLOCKED_INSUFFICIENT_SAMPLING`: truth-측(CMFGEN 기여 99.9% 활성집합) f_cov 가
최대 0.061(S II s0)에 불과. 물리 실체=2.4M-패킷 캡처의 EUV 이온화-edge 대역
미표본(확립된 MC EUV 기근과 일치). 판정력 있는 CHAIN 비교는 estimator 표본
확충 이후의 일로 등재.

⚠ 자기정정: 1차 CHAIN 원장의 "CI-qualified 21/36" 은 **순환 잣대**(활성집합을
판정가능 기여로 구성 → blocked 항이 분모에서 소거)의 허상 — Codex 2차 검수가
적발, f_cov 를 truth-측 기여로 재정의한 재판정이 본 원장이다.

## 미결 (등재)

- σ 는 레거시 1000빈 표본 그대로(데이터 무죄 — J 만 이관). 덱 원본 σ(ν) 점표 직적분은
  별도 census 항목.
- Lumina 쪽 α coefficient 실비교 = `BLOCKED_MISSING_RATE_EXPORT`(재결합 이관 후).
