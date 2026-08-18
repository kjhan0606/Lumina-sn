# Fable Stage-4 J/O 귀속 판정 질의 — A2-10 per-ion union K36 final diagnostic

당신은 Fable이다. Read/Grep/Glob만 사용하고 코드·파일·프로세스·작업을 변경하지 말라.
아래 봉인된 evidence를 직접 읽고 Stage-4의 J/O 귀속만 판정하라. 물리 원인을 강제로
선택하지 말고, 증거가 섞이거나 물리 tolerance가 사전등록되지 않은 경우 `UNRESOLVED`
를 명시하라. floor/cap/clamp/jitter/repair/abs-fix/음수 삭제·물리값 scaling은 절대
허용하지 않는다.

## Objective

K18 census와 K12→K18 closure 이후 K36 diagnostic의 finite per-line evidence를 사용해
다음 K-final 물리 구현 트랙이 J(radiation/Jbar)인지 O(opacity/lower-population/line-universe)인지,
아니면 아직 미결인지 판정한다. 이번 K36 run 자체는 `model.rc=1` 자연
`RADEQ_NO_BRACKET` diagnostic completion이며 final non-census PASS가 아니다.

## Sealed run and baseline

- Candidate run: `/gpfs/kjhan/lumina/a210_line_saturation_per_ion_union_a100x2_nonoverlap_sobolev_k36/diag_20260817T042955Z_14b199f12d29`
- Candidate binary SHA256: `14b199f12d29246e1b7c7173d0ef9e8a9254ba4e8614cc0a838e85c200ad8825`
- Final candidate stderr SHA256: `07dc0366951dce4bc19d2832acb503ab74981a31db94914ba3d10822343e173c`
- Sealed guard3 baseline stderr SHA256: `55004df4b525068ecdf2ba9910059de9aa7b638461ef1a08cacbd225ff461bc5`
- Final R1 report: `validation/a2_10/A2_10_NONOVERLAP_K36_PER_ION_UNION_R1_FINAL_REFERENCE_COMPARISON_2026-08-17.json`
- Final R2 report: `validation/a2_10/A2_10_NONOVERLAP_K36_PER_ION_UNION_R2_FINAL_REFERENCE_COMPARISON_2026-08-17.json`
- Final phase report: `validation/a2_10/A2_10_NONOVERLAP_K36_PER_ION_UNION_PHASE_BASELINE_STREAMS_FINAL_2026-08-17.json`

## Required evidence to inspect

1. Candidate `stderr.log`, `RUN_FOOTER.txt`, and `manual_control/supervisor.log`.
2. `a210_line_saturation_summary_v2.json`, `a210_cmfgen_line_saturation_comparison_v2.json`,
   `a210_cmfgen_line_saturation_comparison_roundoff_v3.json`.
3. `a210_line_ion_owner_report_coverage_v4.json`, `a210_line_saturation_per_ion_coverage_v4.json`,
   and `a210_line_saturation_intersection_v4.json`.
4. `validation/A2_10*` final reports above and the CMFGEN finite comparison records in the
   candidate diagnostic bundle.
5. Existing branch contract and null-control rules in
   `docs/FABLE_PLAN_ANALYSIS_TRANSFER_A210_2026-08-17.md` and
   `docs/FABLE_PLAN_ANALYSIS_TRANSFER_CORRECTION_A210_2026-08-17.md`.

## Branch rules

- J: per-line Lumina tau is optically thick, CMFGEN `1-ZNET` indicates saturation, and
  Lumina `Jbar/S` is below the line's trapping expectation `1-beta(tau)` by more than the
  V3 arithmetic proof bound.
- O: per-line Lumina tau is optically thin while CMFGEN `1-ZNET` indicates saturation;
  tau is independent of Jbar, so this is an opacity/lower-population/line-universe signal.
- Aggregate ratios alone cannot trigger either rule.
- Fe/Co/Ni III is the null control. If the same rule flags III, the scale is contaminated.
- Undercoverage must not be interpreted as a physical signal; the per-ion union run already
  passed Fe/Co/Ni IV coverage >=0.9 and shared printed rows are byte-identical.

## Output format

Return exactly:

1. `VERDICT: J | O | UNRESOLVED | NULL_CONTROL_CONTAMINATED`
2. Per-ion/per-line evidence table: tau, beta, Jbar/S, `1-ZNET`, arithmetic bound, selected rule.
3. III-stage null-control result.
4. Whether a specific file:line physical expression is localized.
5. The only permitted next action: physical implementation/offline recomputation, or read-only
   localization/tolerance preregistration. Do not authorize K-final code or execution unless
   all four Fable correction prerequisites are satisfied, including offline prediction of
   natural `rc=0` bracketing recovery and a preregistered negative control.
