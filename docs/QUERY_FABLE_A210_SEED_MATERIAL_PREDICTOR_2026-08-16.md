# Fable critical verdict + implementation — A2-10 seed material predictor

You now own the coding. Inspect the named repository files, decide the one
architectural ambiguity below, and implement the accepted minimal design in this
working tree. Do not merely propose pseudocode. Do not commit or push Git.

## Non-negotiable physics rules

- Never add a physical-value floor, cap, clamp, jitter, absolute-value repair,
  negative deletion, fallback, or retry that changes a value.
- A failed population solve, invalid generation, or no bracket terminates. It
  must never fall back to LTE/Saha or the old material.
- No new environment knob. The initialization rule must be unconditional,
  exactly once, and explicit in provenance.
- Preserve deterministic line-Jbar arithmetic/order and the normal A2-10
  vector-root contract. Do not revive `LUMINA_A210_PRECORE_TAU_REFRESH`; that
  population--tau seed branch was experimentally rejected.
- Keep ordinary logical outer iterations numbered `0..n_iter-1` for convergence
  snapshots. Initialization is a separately labelled pass, not a hidden user
  iteration.

## Measured cause, not a numerical symptom

The sealed k24 exact/R6 solve is healthy: 45 iterations, residual
`9.6662782724980344e-09 < 1e-8`, 109,014,300 valid line cells, zero repair.
The current non-census root has no bracket in shells 0--3: at the public seed
`T=10020 K`, residuals are `-7.24206748, -5.64886329, -4.40180827,
-3.43092743`; line cooling alone is `7.24430364, 5.65057056, 4.40308484,
3.43186180`. Both endpoints and the geometric midpoint are same-sign.

Mapped finite line: Lumina line 233521 / shell 0 is CMFGEN line 76887,
Co IV 606.784334 Angstrom. CMFGEN depth 67/68 has finite
`q=4.1740456633e-5 / 7.0614206393e-5` and
`J/S=0.999998687164 / 0.999998559733`. Lumina at 10020 K has finite
`J/S=0.999995818345449`, but `eta=1.6365876244e-9` and signed rate
`8.5999761692e-14`; interpolated CMFGEN eta is `3.1281367602`, implying an
upper-population scale difference about `1.91e9`. At the geometric midpoint
`T=22135.943621178667 K`, the same line has finite `eta=20.9434284982` and
signed rate `263.182241415`, while the total shell-0 residual remains cooling
at `-5012.57579760`. Thus coefficient and J/S sign are not the primary cause;
the initial material population scale is. The complete four-point comparator
passed and is stored as
`validation/a2_10/A2_10_CMFGEN_MAPPED_LINE_COMPARISON_2026-08-16.json`.
The causal ledger is
`validation/a2_10/A2_10_SEED_MATERIAL_GENERATION_CAUSAL_AUDIT_2026-08-16.json`.

Source ordering is:

1. `src/lumina_cuda.cu` publishes seed Te generation 1, opens the exactly-once
   LTE/Saha bootstrap, and creates material P1/tau before transport.
2. `src/lumina_cmfgen.c::cmfgen_run` iteration 0 computes and commits exact R1
   from P1.
3. `lumina_r7_publish_and_solve_te` immediately evaluates A2-10 trial bundles:
   the trial material is NLTE P2 at each candidate Te but radiation remains R1
   from LTE/Saha P1. The first thermal root can therefore fail before the
   ordinary lagged Picard map has coevolved once.

## Critical verdict

Choose and state one:

- `ACCEPT_ONE_PREDICTOR`: after R1, perform exactly one fixed-seed-Te private
  NLTE material predictor P2, atomically commit it without publishing a fake
  radiative-equilibrium Te, compute R2 from P2, then run the normal A2-10 root;
  or
- `REJECT`: explain the exact violated invariant and do not implement a warmup.

Do not implement an arbitrary number of warmup passes. A fixed-Te R<->M
convergence loop would require a separately specified physical norm and is out
of scope. If one predictor still gives no bracket, the production run must
terminate and return evidence for another causal diagnosis.

## Required implementation if accepted

Read at minimum:

- `src/lumina_cmfgen.c::cmfgen_run`
- `src/lumina_plasma.c`: `a210_production_bundle_ledger`,
  `a210_production_solve`, candidate build/commit helpers
- `src/nlte_population_candidate.[ch]`
- `tests/nlte_candidate_tau_selftest.c`
- `scripts/check_a210_targeted_gate.py` and its selftest
- `scripts/run_det_convergence_2026-08-08.slurm`

Implement a distinct seed-only material commit API. It may reuse the private
bundle builder, with required Te generation equal to current seed generation
and population generation `m+1`, but must not fabricate or consume an A2-10
ledger. Its preflight must prove:

- public seed Te array, publication bytes, manifest, and generation are
  unchanged across the commit;
- candidate NLTE ion/ne/level populations, tau/source, BF, A2-08 and A2-09 are
  complete and generation-consistent;
- no public byte changes before all fallible work/preflight has succeeded;
- failure preserves all public owners byte-for-byte and terminates;
- successful provenance logs `INIT_SEED_MATERIAL_PREDICTOR`, R1 generation,
  `Te 1->1`, `population m->m+1`, and repair fields all zero.

Make `cmfgen_run` perform R1 -> predictor -> R2 -> normal R7/A2-10 for logical
iteration 0. Use a monotonic radiation-generation counter rather than
`iter+1`. The init pass must not create a physics-comparison snapshot or consume
one of `n_iter` ordinary iterations. For the one-iteration targeted gate, expect
exactly two exact/R6 publications, exactly one predictor commit, then one
ordinary R7 and comparison record at logical iter 0. Extend the gate checker and
positive/negative fixtures accordingly.

Add focused C negative controls for wrong generation/provenance and for failed
preflight preserving public Te bytes/generation; add a positive control proving
the material changes while Te bytes/generation do not. Run the relevant CPU
selftests and build checks available locally. Do not launch GPU or Slurm jobs.

End with: verdict, files changed, exact tests run/results, and remaining risk.
