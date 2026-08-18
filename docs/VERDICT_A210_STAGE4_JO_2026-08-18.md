# A2-10 Stage-4 J/O 귀속 판정 — 2026-08-18 [감리 반영 확정본]

작성=운전석, 감리=Codex read-only(의견서 `docs/CODEX_AUDIT_A210_STAGE4_JO_VERDICT_2026-08-18.md`,
결론 "수정 — 현 상태 채택 불가"). 본 확정본은 감리의 강등 지시를 전부 반영했고,
감리 수치 주장 3건은 운전석이 독립 재계산으로 확인 후 채택했다.

## 0. 봉인

- run root: `/gpfs/kjhan/lumina/a210_line_saturation_independent_jcont_a100x2_nonoverlap_sobolev_k36/manual_20260818T132600Z_stage4` (syn101 A100 GPU6/7 수동, tripwire)
- binary SHA `a655b2c6d2ee842973cbcdc4adcc067b1a3ea17c66131ab00b4cb6a6db6cd930`, sigma SHA `90d04042…5cc3ad`
- 최종 stderr SHA `7aa9fc5647a7e746d107b3caffc03ef0b132736a49c23735f4f321a8b07e4ee0` (3,338,538 bytes)
- 종료: 자연 `RADEQ_NO_BRACKET`(owner 4셸, 4개 국면 동일부호) → 물질 갱신 BLOCKED(te_generation 1→1,
  manifest preserved) → rc=4, child_rc=70. 사전등록 예상 경로 그대로.
- offline 판정: `validation/a2_10/A2_10_STAGE4_JBAR_OFFLINE_2026-08-18.json` = `prediction_status=READY`,
  1,282/1,282 rows, floor/cap/clamp/jitter/repair=0 (08-17 `INSUFFICIENT_INDEPENDENT_FIELDS` 해소).
  ※이 PASS는 **필드 완비에 한정**되며 Stage 전체 PASS가 아니다(감리 고정질문 1).

## 1. 실측 체인 (재현: `validation/a2_10/a210_stage4_state_mismatch_repro.py` + 봉인 stderr)

1. [실측] J_cont 캡처 완료: cells=109,014,300, error_envelope=1, refinements=36, 3,073s.
2. [실측] 캡처 연속체 solve가 R2와 bit-identical(52회, residual 8.1222406993212508e-09).
   원인: fine 격자 선 침착 두 분기 모두 `!sobolev_operator` 게이트(`src/lumina_cmfgen.c:5524,5593`)
   ⟹ Sobolev 모드 생산 fine solve는 원래 line-free. **함의(감리 D 반영): bit-identical은 독립 검증의
   성공이 아니라 두 경로가 같은 line-free 공통모드를 탔다는 증거다. 이 캡처는 선 폐합 단계의 구현
   점검에는 유효하나 연속체 물리의 독립 검증 가치는 없다. "independent J_cont" 명칭은 과칭.**
3. [실측] IV 1,282행: `Jbar/[β·J_cont+(1−β)·S_probe]` 중앙 5.484e-6 (q10 9.6e-7, q90 9.2e-3).
   예측은 (1−β)·S_probe 지배(중앙 1.000).
4. [실측] `S_implied=(Jbar−β·J_cont)/(1−β)`의 `S_implied/S_probe`는 ν에 대해 **거의-완전 단조**
   (rank corr −0.9999999, ν-정렬 인접 비단조 15~16쌍/1,281 — 집계 규약 차). Wien 적합
   `0.606·exp(−hν/kT)`, T=21,989K, ln 잔차 rms 3.9%.
5. [실측] `S_probe/B(T_req=19059.411196903675K)` = 0.9999914~0.9999940 (산포 stdev(log10)=2.3e-7)
   — **표시 정밀도의 근사이지 항등이 아니다**(체계적 ~6.0~8.6e-6 결손 존재, 성격 미규명).
6. [실측] `Jbar/[β·J_cont+(1−β)·B(T_e=10020K)]` = 중앙 0.9992, q90 1.0000, q10 0.7131.
7. [실측] REQUESTED_TE 국면 = 전 셸 trial_te=요청 T(`src/lumina_plasma.c:13618-13636`); 물질은 시행
   후보 계약(`nlte_population_candidate`)으로 재구축. S_probe는 별도 측정이 아니라 row의
   `source_function`(=시행 물질 η/χ) 복사값(`src/lumina_plasma.c:14060-14063`).
   [추정] 5의 근사-항등은 시행 물질이 이 선들에서 S≈B(T_trial)로 행동함을 시사하나, LTE의
   실증은 아니다(NLTE가 계약상·우연히 S→B로 수렴했을 가능성 미배제 — 감리 B).

## 2. 판정

- **V1 [유지] 잣대 불일치 진단**: "선택 IV 1,282선 전부 J 후보"의 원 근거(Jbar/S ≪ 1−β)는
  서로 다른 상태의 비교였다 — 분자(Jbar)는 생산 상태 산물, 분모(S)는 시행온도(19,059K) 물질.
  Wien 꼬리(444–1,884Å)에서 B(19kK)/B(10kK)=**37.8~4.6e6**이며, 관측 격차 규모·ν 준-단조·합성온도가
  이 하나로 재현된다(실측 3-5). ⟹ **기존 잠정 분류(J 후보 1,282)는 이 잣대로는 지지되지 않는다.**
- **V2 [강등: 확정→대용 일치]** 생산 Jbar는 **B(T_e) 대용 폐합**과 중앙 0.08%로 일치한다(실측 6).
  단 이는 생산자의 실제 S가 아닌 대용값과의 일치다. 생산자 S 직접 캡처 전에는 "폐합 만족"을
  확정하지 않는다. q10=0.7131 꼬리(~29% 하회)의 성격도 미확정.
- **V3 [미확정]** 실측 5의 체계적 결손(6.0~8.6e-6)과 실측 6의 꼬리는 모두 미규명 항목으로 대장 기재.
- **V4 [추정 후보]** NO_BRACKET의 기전 후보: frozen-Jbar(생산 상태) × 시행-T 물질의 선 항 결합.
  이 데이터와 양립하나 **인과는 미검증**(row는 진단 전용; R7 경로에서의 결합 방식 미추적).
  "물리 원인 재정의"·"J 생산 결함 아님"의 단정은 하지 않는다. CMFGEN 동종 처리와의 대조가 선행 조건.
- **V5 [유지]** source edit·K-final 권한 미부여. 이 판정은 귀속 시도이지 수리가 아니다.
- **V6 [기재]** 매니페스트의 `independent_S_probe=LINE_MATERIAL_SOURCE_FUNCTION` 및
  `independent_J_cont` 명칭은 실측 2·7의 한정을 달아 읽어야 한다.

## 3. 미결 (사전등록 후보, 우선순위=감리 총평)

1. **생산자 상태의 실제 line-material S 직접 캡처** → `Jbar/[β·J_cont+(1−β)·S_producer]` 재계산
   (감리가 지정한 다음 단 하나). V2를 확정 또는 기각한다.
2. III 음성대조(target_ion=2) 실행.
3. 실측 5의 체계적 결손 성격, 실측 6의 q10 꼬리.
4. V4 기전의 R7 경로 추적 + CMFGEN 동종 루틴 대조.
