# Codex read-only audit — A2-10 Stage4 J/O attribution (2026-08-17)

## Disposition

The Fable Stage4 verdict is structurally supported as a **J candidate**, but it is
not yet an implementation authorization and it is not a K-final gate.  No source
or model output was modified by this audit.

## Source-chain findings

1. The deterministic CMFGEN line producer builds `jbar_line_det` in
   `src/lumina_cmfgen.c:6238-6274`.  It first bulk-fills a private
   `upper_population_cache` through the registered
   `lumina_line_upper_population_fill_for_tau()` callback (`lumina_cmfgen.c:6188-6204`,
   callback implementation `lumina_plasma.c:9017-9063`).  It then evaluates
   `line_net_sobolev_material()` and `line_net_sobolev_radiation()` and publishes
   `radiation.jbar`.
2. The Sobolev radiation expression in `src/line_net_rate.c:175-215` is the
   physical expression `Jbar = beta * J_cont + local_emission_term`; no numeric
   floor, cap, jitter, or repair is applied in this path.  The finite and
   non-negative checks are fail-closed validation only.
3. `radiation_field_line_jbar_energy_view()` (`src/radiation_field.c:1191-1229`)
   binds the line view to the radiation epoch/generation, profile, and Q_E hash.
   The view carries a radiation generation, but no population-generation field.
4. The A2-10 consumer (`src/lumina_plasma.c:14481-14740`) validates the line view
   against `nlte->radfield_view.generation` and the opacity publication
   generation, then reads `Jbar` from that view.  It separately recomputes
   `n_upper` with `a209_upper_population_for_tau()` for the trial candidate.
5. The producer is wired once from `lumina_main.c:384-387`, before trial bundles
   are built.  Trial bundles are constructed later in
   `a210_production_bundle_ledger()` (`lumina_plasma.c:15246-15296`) from
   `candidate->trial_te` and a private candidate population.  Thus the current
   contract proves radiation-view freshness, but does **not** prove that the
   producer's line-material population and the consumer's trial population are
   the same physical state.  This is the concrete structural basis for the J
   hypothesis.  It does not by itself prove that J, rather than O, is the sole
   defect.

## Fable result recorded

Fable classified all 1,282 selected IV lines as rule **J** (Fe IV 482, Co IV
426, Ni IV 374) and none as O.  The selected lines are optically thick while
`Jbar/S` is far below `1-beta(tau)`; the representative Co IV line 233521 has
`tau=6.119e6`, `beta=1.634e-7`, Lumina `Jbar/S=1.3359e-5`, and the CMFGEN
interpolated value near unity.  This is a finite-value discrepancy, not a
zero-output comparison.  The aggregate III null is retained, but per-line III
stability evidence is absent because the selector is IV-only
(`a210_line_saturation_target()`, `lumina_plasma.c:13912-13914`).

## Allowed next action (no K-final yet)

Perform a sealed-state offline recomputation using
`Jbar_probe = beta * J_cont + (1-beta) * S_probe`, with the exact K36 line,
population, tau, and CMFGEN records.  The preregistered test must include:

- per-line IV prediction and a per-line III negative-control/stability table;
- explicit population/radiation generation provenance;
- finite physical comparisons and zero repair/floor/cap/clamp/jitter counters;
- a prediction of whether the natural `RADEQ_NO_BRACKET` becomes an `rc=0`
  bracket before any source change or K-final run.

Until that offline prediction and the negative control pass, source edits and
the final non-census A100x2 gate remain prohibited.
