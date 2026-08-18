# Fine-Jbar solver audit — 2026-08-08

## Verdict

The current sealed DET configuration cannot produce a qualified full-domain
line-Jbar field by increasing the window or ALI count alone.

1. The CPU branch performs a causal blue-to-red frequency sweep, so it does not
   have the GPU lag-convergence defect.  However, its `ADV_SPLIT` discretisation
   is only a first-order operator split with a production segment Courant number
   near 874 fine bins.  The existing production-resolution offline harness found
   a 17.2% full-field L2 difference from a true drifting-characteristic solve,
   with median line-Jbar biases up to 27.7% in shells 10–29.
2. The GPU branch lags the blue-neighbour intensity by one ALI iteration.  With
   advection enabled, information advances roughly one frequency bin per
   iteration.  Twenty-four iterations cannot qualify a 498,721-bin mesh, much
   less the bare BB-support 1,906,171-bin mesh (or the 2,013,113-bin causal
   production mesh described below).
3. The GPU layout stores four `(frequency, ray, segment)` intensity arrays.  At
   the full BB-support mesh it would require about 195.478 GiB for the guarded
   `4*segment + 6*cell` allocation alone, exceeding the 139.8-GB H200.
4. `LUMINA_CMF_ALAM=0` removes the homologous frequency-advection physics.  It
   is a static-limit model change, not a convergence repair.
5. `cmfgen_fine_jbar` ignores the return value of `cmf_solve_J` and itself
   returns `void`.  A GPU lifecycle failure can therefore leave the warm-start
   Planck field in place and still proceed to line extraction/publication.  The
   producer must become status-returning and fail closed.

No Fable call was needed: the source, the July production-resolution harness,
and the current H200 memory limit give a consistent local verdict.

## Existing validated route

`validation/cmfgen_toy06_19p48d/analysis/detjbar_convergence/detjbar_conv.c`
already contains two independently checked drifting-characteristic operators:

- direct cell march, `O(beta)` per output bin;
- sliding-window recurrence, `O(1)` per output bin.

Their line-Jbar values agreed exactly at 800/3,200 bins and to `1.93e-9` at
51,200 bins.  At the production 498,721-bin 1000–4000 A mesh, the sliding form
converged in 25 ALI iterations to max relative change `8.297e-10`; the existing
CPU operator-split reference also took 25 iterations but differed from the
drifting-characteristic field by L2 `1.724e-1`.  Harness timings were 62.8 s for
the exact sliding solve and 4,249.6 s for the old CPU reference.  These timings
are an implementation guide, not a production flight result: the harness uses
its documented reduced opacity/source reconstruction.

The exact CPU layout needs two ray-segment intensity fields plus source/work
cell arrays. The BB line-ID domain is the amended closed 100–20000 A window,
not the wider canonical union-owner edges. Including the registered ±4-Doppler
profile support alone gives 99.986659–20002.668869 A and about 1,906,171 bins.
The homologous characteristic is causal blue-to-red, however, so the production
mesh retains the canonical owner's 74.274847-A blue edge as an upstream
reservoir. This gives 2,013,113 bins for the current default geometry. For 50
shells and 66 rays, the implementation allocates:

- two segment fields + five exact-solver cell fields: about 104.722 GiB;
- including the fine producer's seven existing host cell fields: about
  109.972 GiB.

The selected H200 node `syn104` reports 4,128,416 MiB real memory and over
4,032,607 MiB free during the audit, so the host layout is feasible.  It is not
device-feasible in the present GPU representation.

## Required production contract

The production repair now ports the independently validated sliding
drifting-characteristic CPU operator rather than preserving the inaccurate
operator-split fixed point. It:

- covers the complete registered profile support of the explicit 100–20000 A
  `BB_IN_DOMAIN` rate graph, while the wider canonical owner continues to serve
  its other consumers;
- exposes final max-relative residual, tolerance, iterations used and iteration
  cap;
- returns nonzero if allocation, arithmetic validity, domain coverage, or
  convergence fails;
- forbids extraction and R6 commit after any solve failure;
- rejects negative sliding-recurrence source roundoff instead of silently
  clamping it (the harness clamp counter was zero in validation);
- retains the current GPU solver only for separately declared static-limit tests,
  not the homologous production flight.

The canonical production tolerance is initially `1e-8`, stricter than the
existing hardcoded `1e-4` and supported by the 25-iteration harness result at
`1e-9`. The default cap is 64; exhaustion is an explicit blocked state, never
an implicit success. The production module's direct/sliding selftest agrees to
maximum relative difference `9.152e-16` and converges in 17 iterations with
residual `6.557e-12` on its compact fixture.
