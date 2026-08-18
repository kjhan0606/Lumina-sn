# Fable audit — A2-10 requested-Te and ion-owner diagnostics

- Date: 2026-08-17 KST
- Model requested: `fable`
- Scope: read-only critical code audit; no job launch or code modification
- Query: `docs/QUERY_FABLE_A210_REQUESTED_TE_ION_OWNER_AUDIT_2026-08-17.md`

## Verdict

`VERDICT: APPROVE`

Blocking or required fixes: none.

## Findings

1. The requested-temperature parser is a pure environment read that requires a
   finite positive value.  Its callback is reachable only inside the existing
   diagnostic no-bracket branch.  It writes private `mid`/`lmid` scratch,
   preserves `outte`, `outne`, publication state and generation authority, and
   returns the original no-bracket code 4.  Only diagnostic counters change.
   The selftest proves unchanged outputs and publication generation.

2. Ion-owner records are emitted only after `status == RADEQ_OK`, following the
   complete line-universe traversal, census reconciliation and all per-shell
   finite/sign proofs.  Accepted cells enter the physical shell sum and the
   separate diagnostic owner bin once at the same acceptance point.  The ion
   slot is range-checked and owner allocation is overflow-checked.  The
   postprocessor rejects duplicate/incomplete/blocked/unsafe records and
   grouped-total mismatches.

3. The K30 bound decomposition satisfies the prior prerequisite for a single
   K36 proof rung: its propagated input term is
   `beta * continuum_j_absolute_uncertainty` with `beta < 1`, it identifies no
   additional non-contracting propagated component, does not change the
   physical no-bracket result, and has zero repair markers.  K36 is justified
   only for completing the requested-temperature callback.

## Optional defense-in-depth notes

- Invalid owner scope or allocation failure deliberately changes the callback
  return to a fail-closed schema/nonfinite error; this is not a physical repair.
- The postprocessor already closes printed owner rows against grouped summary
  fields, but can additionally bound grouped-vs-line-order deltas.
- A requested temperature identical to an endpoint or geometric midpoint can
  inherit that higher-precedence phase label; the postprocessor then fails
  closed rather than misidentifying it as `REQUESTED_TE`.

No floor, cap, clamp, jitter, repair, root pin, fallback or tolerance relaxation
was approved.
