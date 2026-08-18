# Fable 중요 통합 재심 — SH-RADEQ `REVISE` 폐합 (2026-08-08)

- model: `claude-fable-5`
- Claude Code CLI session: `4ce73c4f-5d45-450b-b545-fabd9e25f73d`
- 질의: `docs/QUERY_FABLE_SH_RADEQ_PREFLIGHT_CLOSURE_2026-08-08.md`

아래는 Fable 응답 원문이다.

---

## 감리 결과 — 5개 조건 전건 실측 폐합 확인

문서 주장을 그대로 믿지 않고 소스·보존 산출물에서 독립 재확인했다. 전 항목이 실물과 일치한다.

**1. Raw tau slab 계약 — 폐합.** Writer 3개 census를 grep으로 독립 재현: `compute_tau_sobolev`(lumina_plasma.c:2907/3018), `nlte_update_tau_sobolev`(15751/15872), `apply_overlap_corrections`(16900/16963) 각각 첫 write 전 `require_refresh`·마지막 write 후 `mark_computed` bracket 실물. CUDA 측 `tau_sobolev[...]=` 대입 0건(lumina_cuda.cu:1926은 공용 host writer 호출로 대체됨). post-NLTE 진단은 NaN→0 변이 없이 `[TAU-DIAG][FATAL] ... action=TERMINATE`로 종료(lumina_cuda.cu:10850). A2-09 소비 양끝 bracket은 `a209_line_generation_view_valid` begin/end 등식으로 `EMISS_STALE_OPACITY` fail-closed(emissivity_publication.c:55-68). 산출물 `[TAU-WRITER-CENSUS][NEGATIVE-CONTROL][PASS] injections=4 detected=4` 확인.

**2. NLTE authority 단일화 — 폐합.** `nlte_tau_line_authority`/`nlte_tau_line_shell_authorized`/`nlte_tau_line_uses_nlte`가 writer 경로(15784/15837)와 reader 경로(8601/8508) 양쪽에서 동일 predicate로 소비됨을 실측. 별도 사본 없음.

**3. LTE/NLTE population 공유 — 폐합.** `population_line_level_number_density`(population_contract.c:167)가 bulk writer(2996-3000, `POP_LINE_VIEW_LTE_TE`)와 A2-09 reader(8519-8520, `use_nlte` 분기)의 단일 routine. A2-07 selftest가 LTE/committed-NLTE/음수 거부 branch를 덮음.

**4. Signed tau 등록 — 폐합.** `docs/SIGNED_TAU_EMISSIVITY_REGISTER_2026-08-08.md` 실물: 지수 증폭 명시, clamp/floor/fallback 부재, nonfinite→`EMISS_DIRECT_LINE_INVALID` candidate 전체 폐기, `tau=-0.25` analytic·`tau=-800` abort 자가검사. CMFGEN inversion 처리 동등성을 **선언하지 않고** 대조를 사전등록한 것이 규약(잣대부터 감사·기대결과 드라마화 금지)에 정확히 부합한다.

**5. SH-GRID 등록 — 폐합.** `docs/SH_GRID_REOPEN_CONTRACT_2026-08-08.md` 실물: dlog·상한 보존, 1000→1178 bins, 하한 `5.84127859196e13 Hz`(최저 활성 edge `5.84852771e13` 아래), padding/첫-bin 대입 금지, 저주파 band level별 CMFGEN 대조를 MC-EVT 재개의 선행조건으로 고정. 실측 census(707 default-active, CMFGEN sigma 707, Kramers 0, 의도된 `REOPEN_SH_GRID rc=3`)가 보존 산출물과 일치.

보존 산출물(`validation/a2_09/A2_09_SELFTEST.json` N1–N8 8/8, `validation/a2_10/A2_10_SELFTEST.json` L6=BLOCKED_INCOMPLETE_ADIABATIC, 폐합 정본 `validation/a2_09/SH_RADEQ_FABLE_REVISE_CLOSURE_2026-08-08.md`)도 주장과 등치.

**경계 유지 확인:** 완전 CMFGEN 단열항 부재로 A2-10 L6이 `BLOCKED_INCOMPLETE_ADIABATIC`으로 남아 있고, 폐합 문서 스스로 구현 폐합과 flight 허가를 분리하고 있다. 이 상태에서 flight를 READY로 승격할 근거는 없다.

```text
IMPLEMENTATION_CLOSURE = ACCEPT
FLIGHT_STATE = BLOCKED_INCOMPLETE_ADIABATIC
```

## Codex 의견

동의한다. A2-09 direct line emissivity 및 raw τ generation ownership은 이번 rung에서
폐합됐다. 대용량 compact τ copy를 지금 추가하지 않는 조건부 token 방식도 writer census와
양끝 bracket으로 충분히 강제됐다. 그러나 완전 CMFGEN 단열항과 atomic trial transaction이
없으므로 A2-10 production flight는 계속 금지한다.
