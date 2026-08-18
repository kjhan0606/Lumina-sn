# Fable 총괄계획 정정 — diagnostic completion과 final non-census PASS 분리

- 요청 모델: `fable`
- CLI: Claude Code `2.1.233`
- 질의: `docs/QUERY_FABLE_A210_PLAN_ANALYSIS_TRANSFER_CORRECTION_2026-08-17.md`
- 권한: Read/Grep/Glob only; 코드·파일·process·작업 변경 없음

아래는 Fable 응답 원문이다.

# CORRECTION VERDICT

**정정 인정 — 선행 응답의 FINAL GATE EVIDENCE 3항과 5항 후반부는 실제 auditor 계약과 충돌하며 철회한다.**

근거 (실측):

- `audit_run`은 line 416에서 `model.rc == 0`을, line 436-438에서 `r7_material_commit`과 `physics_comparison_commit` **둘 다 true**를 fail-closed로 강제한다. K36 diagnostic run의 의도된 자연 결과(`model.rc=1` + `REQUESTED_TE RADEQ_NO_BRACKET`)는 이 두 요건을 **원리적으로** 충족할 수 없다 — no-bracket 종료는 R7 commit 지점에 도달하지 못한 종료이기 때문이다.
- 따라서 선행 EVIDENCE 3항("자연 종료 증거 = rc=1 + NO_BRACKET")은 **범주 오류**였다: 그것은 Stage 1-4의 *diagnostic completion* 증거이지 Stage 5 *final non-census PASS* 증거가 아니다. 두 completion 은 별개 대상이며 별개 원장 항목이다.
- 같은 binary SHA 사용은 R1 계보 연결(sealed reference 비교의 정합성)에만 기여하며, R7 commit 의 대체물이 아니다 — auditor는 commit 을 boolean 필드로 직접 검사하지 line 계보로 추론하지 않는다.
- **판정 1번 항목: K36 diagnostic rc=1 은 final gate PASS 증거에서 명시적으로 제외한다.** K36 은 K24/K30과 같은 지위(diagnostic completion, `docs/CURRENT_PLAN.md:3121-3124`의 K24/K30 판정과 동일 범주)로만 기재한다.
- **판정 3번 항목: "미결도 gate를 막지 않는다" 문장은 철회한다.** 형식 논리로는 auditor 에 Stage 4 귀속 필드가 없으므로 미결 자체가 검사 대상은 아니다. 그러나 그 문장이 참이 되는 전제 — rc=1 종료로도 gate 증거가 성립한다 — 가 틀렸으므로 문장 전체가 무효다. 실제 계약 하에서 final PASS 는 rc=0 도달을 요구하고, rc=0 도달은 NO_BRACKET 원인의 해소를 사실상 전제하므로, **미결 상태에서는 rc=0 재실행 발주 자체가 offline-first 3요건(기전 특정·수리안 오프라인 검증·기대치 사전등록)을 충족할 수 없어 final PASS 에 도달할 경로가 없다.** 미결은 gate 를 "규정상" 막는 게 아니라 "발주 자격 상실"로 막는다 — 결과는 동일하게 PASS 불가.

# REVISED STAGE 4

입력·규칙 J/O·null control·혼동 금지 조항은 선행 응답 그대로 유지한다. **출구 분기만 다음으로 교체한다:**

- **귀속 확정(규칙 J 또는 O가 단일 경로 지목 + III-stage null control 무결)** → 물리 구현 트랙 개시. 선행 응답의 허가 요건 1-4(파일:라인 국소화 → sealed K36 state 위 offline recomputation 으로 정합 복원 + null control 안정 시연 → 기대치 사전등록 + preregistered negative control → clamp/repair 전무·pre-core refresh 금지·generation barrier 보존)를 **전부** 통과한 뒤에만 `model.rc=0` non-census 재실행이 정당화된다. 이때 offline recomputation 이 시연해야 하는 기대치에 **"radeq bracketing 회복 ⟹ rc=0 자연 도달"이 명시적으로 포함**되어야 한다 — IV-stage tau/Jbar 정합 복원만으로는 재실행 자격이 없다.
- **미결(산술상 실재하나 tolerance 미등록, 또는 J/O 신호 혼재)** → rc=0 재실행 발주 **금지**. 다음 read-only 단계만 허용: (a) 미결 항목별 물리 tolerance 사전등록 후 재판정, (b) sealed K36 이벤트로그·덤프에서 NO_BRACKET 직접 원인의 국소화 확장(RADEQ 장부의 항별 bracketing 실패 지점 — 이는 J/O 와 독립일 수 있다), (c) 필요 시 CMFGEN 동종 루틴 출력과의 추가 대조표 추출. 이 단계들이 귀속을 확정시키면 위 귀속 분기로 합류하고, 끝내 미결이면 **미결인 채로 diagnostic 원장에 기재하고 Stage 5 는 개시하지 않는다.**
- **null control 오염** → 선행 응답 유지(판정 철회, 잣대 감사 선행, 코드 변경 불허).

# REVISED STAGE 5

Stage 5 는 K36 이 아니라 **별도의 final non-census 판정런(K-final) 1회**를 대상으로 한다. 개시 전제와 PASS 조건:

- **개시 전제(전부 충족 전 발주 금지):** Stage 4 귀속 확정 + 물리 구현 트랙 완료(negative control FAIL 시연 포함) + offline 검증이 rc=0 자연 도달을 예측 + 기대치 사전등록. 미결 상태에서는 개시 불가.
- **PASS = `audit_run` 전 필드 충족.** 요건을 auditor 실측 그대로 열거하면:
  1. `model.rc` 파일 내용 = `0` (line 416)
  2. `TARGETED_GATE_VERDICT.txt`에 `A210_TARGETED_GATE_ACCEPT status=PASS` (418-420)
  3. `a210_targeted_gate_report.json`: schema `LUMINA_A210_TARGETED_GATE_V3`·status PASS·`expected_devices=2`·refinements 일치·`exact_publications=2`·`r6_radiation_generations=[1,2]`·`seed_material_predictor_commit=true`·`line_operator=CMFGEN_NONOVERLAP_HOMOLOGY_SIGMA0`·`sobolev_jbar_cells=109,014,300`·**`r7_material_commit=true` 그리고 `physics_comparison_commit=true`**·`cancellation_census_present=false`·`physical_values_modified_by_numerical_repair=false`·R2 signed-material 정확 일치(signed 22,866,166 / exact-zero-τ 86,148,134 / raw-neg 4,246,581 / mild-neg 4,246,577 / srce_chk 4) (422-451)
  4. snapshot report: `LUMINA_DET_CONVERGENCE_V1`·CONVERGED·iterations 1·tail_transitions 0·bins 1234·transitions [] (453-460)
  5. `r1_k24_reference_comparison.json` PASS: refinements=24면 bit-exact, 아니면 PROOF_REFINEMENT_ONLY + proof bounds 비증가; occurrence 0/0; candidate stderr = 이 run 의 `stderr.log`이고 비교 후 SHA 불변; sealed reference SHA 불변 (462-501)
  6. tripwire: `manual_control/COMPLETED` 존재·FAILED/YIELDED 부재·`child.rc=0`·supervisor.log 생애주기 카운트(START 1회·CHILD_STARTED 1회·`COMPLETED child_rc=0` 1회)·YIELD/FAILED/FATAL 전무·A100 gpu_preflight 정확히 2행 (503-519)
  7. RUN_FOOTER: `diagnostic_mode=A210_TARGETED_GATE`·`outer_iterations=1`·refinements 일치·`LUMINA_CMF_FINE_MGPU_DEVICES=2` (521-526)
  8. `input/precore_tau_refresh.txt` = `0` (527-528)
- **fail-closed:** `require` 1건이라도 위반 → PASS 주장 금지, 위반 필드 명시. 폐합 시 Codex read-only 판정 감리 스탬프 + V0-V5 원장 기재 + 커밋 1건 = 계약 1건 유지.

# REVISED FINAL GATE EVIDENCE

선행 목록에서 3항·5항을 교체하고 나머지는 유지한다:

1. K18 comparison JSON (SHA `b6a0be0b…`) — census closure 증명. (유지)
2. K36 diag dir + binary SHA `f9c2d1b8…` — **diagnostic completion 증거로 재분류.** K36 의 rc=1 + `RADEQ_NO_BRACKET` 은 Stage 4 귀속의 입력이지 gate 증거가 아님을 원장에 명시(K24/K30 과 동일 범주). 같은 binary SHA 는 R1 계보 연결에만 인용.
3. **(교체)** K-final run 의 `audit_run` 전 필드 충족 기록 — 위 REVISED STAGE 5 의 1-8 전항, 특히 `model.rc=0`과 `r7_material_commit=true ∧ physics_comparison_commit=true`.
4. V2/V3/V4 산출물 SHA — V4 PASS 또는 UNDERCOVERED→union 재실행 후 PASS. (유지)
5. **(교체)** Stage 4 귀속 확정 문서 + 물리 구현 트랙 완료 증거(파일:라인 국소화, offline recomputation 시연, preregistered negative control FAIL 기록, 불변조건 준수) — **귀속 확정이 K-final 발주의 전제 조건이며, 미결이면 이 항목이 결손이고 gate 는 폐합 불가.** 미결 기재는 diagnostic 원장으로만.
6. 불변조건 스탬프: floor/cap/clamp/jitter/repair 0, `physical_values_modified=false`, pre-core refresh 0, generation barrier(te 1→1, pop 1→2), tripwire 무발동, 외부 충돌 0. (유지 — auditor 의 `audit_repair_fields` 3개소와 대응)
7. Codex read-only 판정 감리 스탬프 + V0-V5 원장 기재 + K24/K30/**K36** 제외 범위 명시. (K36 추가)

요약하면: 선행 응답은 "진단 완결"과 "gate 폐합"을 하나의 완결로 합쳐 썼고, 이는 auditor 가 rc=0·R7 commit 을 boolean 으로 검사한다는 사실과 충돌한다. K36 은 Stage 4 의 증거 생산자로 격하하고, final PASS 는 귀속 확정 → 물리 수리 → 별도 K-final 판정런 1회의 경로로만 도달 가능하다. 파일 수정·발주는 하지 않았다 — 이 판정문의 봉인 여부는 운전석 처분에 맡긴다.
