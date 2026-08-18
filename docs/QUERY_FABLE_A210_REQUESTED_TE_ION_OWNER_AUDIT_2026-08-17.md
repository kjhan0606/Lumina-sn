# Fable critical audit request: requested-Te and ion-owner diagnostics

Read-only, narrow correctness audit. Do not edit files or launch jobs. Tokens
are scarce: inspect only the cited code/evidence and return a compact verdict.

## Non-negotiable invariants

- No physical floor/cap/clamp/jitter/repair or absolute-value sign repair.
- A diagnostic may never alter a bracket, publication, solver result, physical
  value, or coevolution generation authority.
- A partial line traversal may never be printed or interpreted as an ion total.
- Invalid/nonfinite/unresolved physical input must fail closed.

## Change 1: requested private diagnostic temperature

- `src/radeq_publication.c:21-31`
- `src/radeq_publication.c:430-535`, especially 473-532
- `src/radeq_publication.h:90-118`
- `tests/a2_10_radeq_selftest.c:80-100`
- production phase identification:
  `src/lumina_plasma.c:9065-9110` and `src/lumina_plasma.c:13605-13645`

`LUMINA_RADEQ_DIAG_TE_K` is parsed only under the existing
`LUMINA_RADEQ_DIAG` no-bracket lane. If one positive finite uniform value lies
inside every open bracket, it gets one callback labeled `REQUESTED_TE`. The
function still returns the original no-bracket code 4 and frees the private
candidate. Existing output Te/ne and publication must remain unchanged. It is
not a root pin, fallback, neighbor, floor, clamp, or solver trial.

## Change 2: complete-only signed line ownership

- `src/lumina_plasma.c:13645-13820`
- initialization and allocation call: `src/lumina_plasma.c:14010-14110`
- accepted-cell accumulation: `src/lumina_plasma.c:14270-14415`
- complete-only logging/publication boundary:
  `src/lumina_plasma.c:14445-14575`
- `scripts/summarize_a210_line_ion_owners.py`
- `tests/a2_10_line_ion_owner_summary_selftest.py`

`LUMINA_A210_LINE_ION_OWNER_SHELLS=N` allocates diagnostic long-double bins
for the first N shells. Only cells already accepted by the existing physical
line kernel are grouped. Per-ion records and shell summaries are printed only
after the whole line universe and all shell totals pass existing finite/sign
proofs. The postprocessor rejects incomplete, duplicate, grouped-total
mismatch, blocked callback, invalid requested scan, or any repair marker.

## Why K36 is running

- `validation/a2_10/A2_10_NONOVERLAP_K30_SOBOLEV_BOUND_DECOMPOSITION_2026-08-17.json`
- prior condition: `docs/FABLE_VERDICT_A210_K30_PROOF_RUNG_2026-08-17.md`

K30 proved both physical endpoints and produced a genuine no-bracket result.
The separate geometric-mid callback failed at line 894169/shell11 because the
certified uncertainty exceeded the finite signed value. The new decomposition
artifact says the propagated input contribution is exclusively
`beta * continuum_j_absolute_uncertainty`, with beta < 1, and no additional
non-contracting propagated component. K36 changes only proof refinement and is
being used to complete the explicitly requested-temperature callback, not to
turn the no-bracket result into a root or relax any physics.

## Questions

1. Can the new requested-temperature callback change bracket/publication,
   output Te/ne, solver return code, or generation authority? Identify a
   concrete side-effect path if yes.
2. Can the ion-owner implementation emit a partial total, alter the physical
   line sum, double count a cell, mis-map ion ownership, or pass an invalid
   grouping? Cite exact lines for any defect.
3. Does the K30 bound decomposition satisfy the prior prerequisite for this
   single K36 proof rung, with no physical-value repair?
4. Return exactly `VERDICT: APPROVE`, `VERDICT: REVISE`, or
   `VERDICT: BLOCK`. List only blocking/required fixes first; keep optional
   diagnostics separate. Do not broaden into a project-wide review.
