# Fable audit request: A2-10 R2 negative-opacity operator

Read-only architecture audit. Do not edit files, launch jobs, or broaden scope.
Tokens are scarce: inspect only the cited evidence and return a compact verdict.

## Non-negotiable constraints

- Preserve every signed physical value. No floor/cap/clamp/jitter/repair/abs-fix.
- A physical negative or nonfinite result must be root-caused and fail closed.
- Preserve the coevolution generation barrier.
- The rejected pre-core tau refresh must not return.
- Fable owns architecture/code audit; Codex owns implementation and execution.

## New finite A100x2 evidence

Authoritative report:

`validation/a2_10/A2_10_A100X2_SEED_MATERIAL_R2_NEGATIVE_OPACITY_FORENSIC_2026-08-16.json`

Run root:

`/gpfs/kjhan/lumina/a210_targeted_gate_a100x2_seed_material_k24/det1234_20260816T133014Z_2f4dc0175a0a`

- R1 exact/R6 is bit-exact with the sealed k24 reference.
- The new seed-material predictor commits exactly once: population generation
  1->2, Te generation remains 1, Te bytes/publication are preserved.
- The next R2 census has 4,246,581 negative line-shell cells: 4,246,577 in
  `-0.5 <= tau < 0`, four at `tau < -0.5`.
- The most negative cell is Fe III line 2164811, shell 0, tau=-0.9581055493.
  It is the 1100(g=3, SL96) -> 1296(g=1, SL100) transition at 19111.918823 A.
  Its full-level populations are both 0.13653029676205097, hence
  `(n_u/g_u)/(n_l/g_l)=3`: a finite population inversion, not roundoff.
- The first invalid fine bin is a distinct witness: shell 0, bin 187,
  chi_cont=2.1585306358e-15, chi_line=-2.3443262375e-15, hence
  chi_total=-1.8579560170e-16. No value was changed; the solver failed closed.

## External same-transition evidence

The sealed CMFGEN O-PHYS deck has `ALLOW_OL=F`, `CHK_L_POS=T`,
`NEG_OPAC_OPT=SRCE_CHK`:

`/gpfs/kjhan/cmfgen_runs/toy06_19p48d_ophys/VADAT`

In its finite `FeIIIOUT`, the same zero-based full levels 1100 and 1296 are
population-inverted at 41/90 depths. Example depth 1:

- T=16493.527 K
- b_lower=0.025177219
- b_upper=0.056036312
- `(b_u/b_l) exp[-(E_u-E_l)/kT] = 1.4100551514 > 1`

Thus negative opacity for this exact transition also exists in CMFGEN and must
not be deleted.

CMFGEN source:

- `/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:3535-3590`
- `/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sobjbar_sim.f:1-260`
- `/gpfs/kjhan/cmfgen_src/cur_cmf/subs/exponx.f:1-80`

For `ALLOW_OL=F` Sobolev transport, CMFGEN consumes each line through the
escape-probability operator `EXPONX(tau)`. It does not sum millions of signed
Gaussian profiles into one shared frequency-bin extinction. `tau < -0.5` is
handled by the declared `SRCE_CHK` effective-material branch while raw signed
populations/opacity remain diagnostic facts.

Lumina currently does the opposite in `src/lumina_cmfgen.c:4430-5605`:
it deposits every signed direct-bracket line as a Gaussian into shared
`fs.chi_line`, adds continuum, then requires `fs.chi_tot > 0`.

## Codex provisional verdict

The population inversion is physical/model-level and is corroborated by
CMFGEN. The failure is a structural transport mismatch: an overlapping-profile
Gaussian total-extinction operator is being used for a sealed non-overlap
Sobolev deck. Making the populations/tau positive or clamping total opacity
would be wrong.

## Questions requiring a compact Fable verdict

1. Is the provisional root cause correct? If not, name the exact missing fact.
2. Choose the correct implementation architecture:
   - A: replace the Q_E deterministic line-Jbar producer with a
     CMFGEN-equivalent non-overlap Sobolev operator for the sealed parity lane;
   - B: route only negative cells through Sobolev while positive cells remain
     in the shared Gaussian solver;
   - C: retain signed Gaussian overlap and implement a true maser/saturation
     transfer model;
   - or specify another precise architecture.
3. Under the no-repair rule, may CMFGEN `SRCE_CHK` exist only as an explicitly
   typed benchmark-policy view with raw tau/populations immutable, or must the
   A100 gate stop at any `tau < -0.5`?
4. State the smallest proof obligations/tests before another A100x2 run.
5. Give `VERDICT: APPROVE`, `REVISE`, or `BLOCK`, followed by required changes
   only. Separate correctness requirements from optional diagnostics.
