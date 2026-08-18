# Fable audit request: A2-INIT seed-material predictor

Role: independent architecture and implementation auditor. Read only. Do not
edit files, generate patches, run GPU jobs, or broaden the scope. Report only
critical/high-confidence correctness findings; conserve tokens.

## Physical rule

- No numerical floor, cap, clamp, jitter, repair, absolute-value fix, or
  deletion of negative physical values.
- A physical negative/nonfinite/no-bracket result must fail closed and be
  root-caused.
- `LUMINA_A210_PRECORE_TAU_REFRESH` was an A/B diagnostic that was rejected;
  it must never enter this production initialization path.

## Intended state sequence

1. Bootstrap publishes seed electron temperature T1 and LTE/Saha material P1.
2. Exact deterministic transport publishes paired continuum/line radiation R1.
3. Exactly once, solve a private NLTE material bundle at byte-identical T1
   against R1 and atomically commit material P2 only. T_e array, T_e generation,
   and T_e publication must remain byte-identical.
4. Recompute exact deterministic transport to publish R2 from P2.
5. Only then run the ordinary A2-10 RE-integral solve and publish its T2/P3
   transaction. There is no fallback if any stage fails.

The initialization boundary is sealed as R1 generation=1, T_e generation=1,
public atom material generation=1, public NLTE material generation=0, followed
by the single transition P1->P2. This also rejects re-entry structurally.

## Files to audit

- `src/lumina_cmfgen.c`: initialization pass and R1 -> predictor -> R2 -> R7
  ordering.
- `src/lumina_plasma.c`: `lumina_init_seed_material_predictor`,
  `candidate_material_commit_preflight`, `candidate_seed_commit_preflight`, and
  `nlte_population_candidate_commit_seed_material`.
- `src/nlte_population_candidate.h` and `src/lumina.h`: public contracts.
- `tests/a2_10_seed_commit_selftest.c` and
  `tests/nlte_candidate_tau_selftest.c`: atomicity/provenance controls.
- `scripts/check_a210_targeted_gate.py` and
  `tests/a2_10_targeted_gate_selftest.py`: two-publication gate and repair audit.
- `Makefile`: seed selftest wiring.

Relevant Codex verification already passed:

- CPU and CUDA fat-binary builds.
- seed material commit selftest.
- general candidate tau/commit and candidate/adiabatic selftests.
- A2-04, A2-06, A2-08, A2-09, A2-10 selftests.
- targeted gate judge positive and 13 fail-closed controls.

## Required response

Return exactly these compact sections:

1. `VERDICT: APPROVE | REVISE | REJECT`
2. `ARCHITECTURE`: whether the causal ordering and exactly-once boundary are
   correct.
3. `CODE FINDINGS`: only correctness defects, each with file:line and impact.
4. `REQUIRED CHANGES`: minimal required changes, or `NONE`.
5. `GPU GATE`: `READY` or `NOT READY`.

