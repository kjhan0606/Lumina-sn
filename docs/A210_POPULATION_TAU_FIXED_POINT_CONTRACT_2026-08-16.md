# A2-10 private population--Sobolev-tau fixed-point contract

Status: **REJECTED / DO NOT IMPLEMENT (2026-08-16).**  This document is kept
only as a record of a falsified branch.  The actual production SE population
path overwrites the preceding legacy mode-1/2/3 diagnostic-shadow rates with
the A2-06 canonical line view and `jd_beta=1.0`; it therefore does not consume
the proposed pre-core Sobolev-tau seed.  The premise of this fixed point is
false for the production gate.

The opt-in probe confirmed the boundary instead of producing an A/B result:
after exact/R6 it terminated with
`PRECORE-TAU-SEED-BLOCKED reason=UNSUPPORTED_RATE_CONSUMER jbar_mode=UNSET`
and `POP_FORBIDDEN_FALLBACK`.  The matched runs were then operator-stopped and
all selected GPUs were returned.  No result from those incomplete runs may be
used as a physical comparison.  Any future proposal to couple population and
tau requires new evidence that the production rate consumer actually depends
on tau; this rejected contract is not authorization to add that coupling.

## Scope and invariant

The fixed point exists only inside an `NLTEPopulationCandidate`.  It must not
mutate public population, ionization, electron-density, partition, tau, source,
opacity, or emissivity arrays before the enclosing candidate transaction
commits.  The ordinary/public `nlte_solve_all()` path remains byte-identical
when the gate is off.

The supported rate consumer is production mode 3,
`LUMINA_NLTE_JBAR_POPS=3`, in which the SE matrix consumes
`beta(tau) * J_inc`.  Deterministic line-response/source-lag consumers are
rejected because a tau-only inner iteration would otherwise create a mixed
material state.

No physical value may be repaired.  In particular, the implementation may not
floor, cap, clamp, take an absolute value of, add jitter to, or replace a
population, Sobolev tau, source, rate, or radiative-equilibrium term.  A
negative or nonfinite value is terminal evidence and aborts the private
candidate.

## Iteration map

Let `n_k` be the complete private NLTE level-population slab and `tau_k` the
complete private line/shell Sobolev slab consumed by the next SE solve.

1. Start from the already tested trial-temperature LTE/ionization tau seed.
2. Solve every active SE/CE pair using `tau_k`.
3. Apply the existing population damping, if configured, to obtain `n_(k+1)`.
4. Rebuild the complete LTE/unmapped tau slab from the same private trial
   thermodynamic and ionization state.
5. Apply overlap corrections, when physically enabled by the sealed run
   configuration.
6. Overlay mapped NLTE tau and line source using `n_(k+1)` and the same
   element-wide authority map that the candidate will publish.
7. Validate every tau/source status and every numerically required value.
   Failure aborts and rolls back the candidate.
8. The next SE pass consumes this refreshed `tau_(k+1)`.

Convergence cannot be declared before at least two SE passes: one pass must
consume a tau slab produced from a previous NLTE population iterate.  Reaching
the finite iteration limit without convergence is a terminal
`CE_NONCONVERGED` result; it does not fall back to the one-shot seed or the
public tau slab.

## Convergence observable

Ion totals alone are insufficient because ion lock/per-ion rescaling can keep
them unchanged while the level distribution and therefore the line tau change.
For each NLTE ion and shell, use

```
d = sum_l abs(n_new[l] - n_old[l])
s = sum_l (abs(n_new[l]) + abs(n_old[l]))
level_change = 0                 when d == 0 and s == 0
level_change = d / s             when s > 0
invalid                          otherwise
```

The fixed-point population residual is the maximum `level_change` over all
ion/shell blocks.  The exact-zero branch is algebraic, not a numerical floor.
All operands and partial sums must be finite and populations must be
nonnegative before evaluating the metric.  Record the worst ion, shell, pass,
level residual, and the existing ion-total residual.  Convergence requires the
level residual to meet its declared dimensionless tolerance and the refreshed
tau slab to have a valid current generation.  The tolerance controls an
iteration decision only; it must never alter a physical value.

## Ownership and generations

- Every complete tau rebuild advances `tau_required_generation`, and every
  successful LTE/overlap/NLTE-authority producer leaves
  `tau_computed_generation == tau_required_generation`.
- The matrix pass records the exact tau generation it consumed.  Consecutive
  fixed-point passes must show the previous pass's produced generation as the
  next pass's consumed generation.
- Element-wide status is candidate-private.  A committed `(Z,shell)` owner is
  not overwritten by a pair solve, damping, or a different tau writer.
- A failed pass leaves all public generations and arrays unchanged.

## Required diagnostics

One machine-readable record per pass must include: candidate population
generation, pass/max-pass, tau generation consumed and produced, maximum
level-distribution residual and owner, maximum ion-total residual and owner,
tau/source invalid counts, and
`public_mutation=0 floor=0 cap=0 clamp=0 jitter=0 repair=0`.

The final record must state exactly one of `CONVERGED`, `NONCONVERGED`, or a
typed physical/operational failure.  A comparator PASS is not a physical gate
PASS; the R7 LOWER/UPPER sign bracket remains the gate.

## Minimum tests before a production gate

1. Gate-off candidate and public paths remain byte-identical to their sealed
   references.
2. Unsupported rate consumers are rejected before the first inner solve.
3. A same-ion-total/two-level fixture changes its level residual and cannot
   falsely converge on the old ion-total metric.
4. An exact-zero population block evaluates to exactly zero without a floor.
5. A known contractive population--tau fixture consumes the newly produced tau
   generation on its next pass and converges only after at least two passes.
6. A noncontractive fixture reaches the iteration limit, fails closed, and
   proves byte-identical public arrays/generations.
7. Negative/nonfinite population, tau, source, and generation-staleness
   controls each fail at their point of origin with no fallback attempt.
8. The A100x2 non-census run preserves the four exact/R6 records, reports zero
   repair attempts, and must restore a finite R7 bracket before any 4-iteration
   or CMFGEN same-state comparison is accepted.
