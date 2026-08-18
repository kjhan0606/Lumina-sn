# Fable verdict — A2-10 K30 proof rung

- Date: 2026-08-17 KST
- Model requested: `fable`
- Canonical CLI alias: `claude-fable-5`
- Scope: architecture verdict only; no code or file modification

## Evidence submitted

- The componentwise error refinement applies residual plus `K(e)` to an already
  verified supersolution and does not change the converged physical `J`.
- With identical R1 `J` and domain hash, the maximum certified bound decreased
  from `5.58322805e-8` at K12 to `3.96908703e-9` at K18 and
  `2.66993389e-10` at K24. All repair counters were zero.
- The R2 K24 non-overlap Sobolev solve produced 109,014,300 finite `Jbar` cells,
  then failed closed at two endpoint witnesses. Their current-bound/required-bound
  ratios were `1.246158927` (LOWER) and `1.783855481` (UPPER).

## Verdict

`YES — K30 is the justified minimum next rung.`

K30 changes proof strength only. It does not relax a tolerance or declare a
smaller bound, and it does not alter any physical value. The two measured
deficits require less than a factor of two while the previous six-refinement
rungs reduced the global bound by factors of 14.1–14.9.

Required evidence is: R1 physical/exact/R6 identity remains bit-exact, exact
residual remains below `1e-8`, refinements are exactly 30, both local witness
bounds actually fall below their required bounds, all 109,014,300 Sobolev
`Jbar` cells remain finite, and all repair counters remain zero.

If K30 fails, do not proceed mechanically to K36. Split the local bound into
the refinement-contracting and non-contracting terms first. A persistent local
floor would require proof-arithmetic precision analysis, not tolerance
relaxation, floor/cap/clamp/jitter, or a repaired physical value.
