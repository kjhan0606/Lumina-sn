# Fable code audit request: A2-10 non-overlap Sobolev operator

Read-only, compact correctness audit. Do not edit files or launch jobs. Tokens
are scarce; inspect only the cited ranges and answer the five questions below.

## Non-negotiable invariants

- No physical floor/cap/clamp/jitter/repair/abs-fix.
- Raw signed tau and populations remain immutable and are censused.
- Any invalid/nonfinite input or census mismatch fails before publication.
- Preserve the coevolution generation barrier and the rejected pre-core tau
  refresh exclusion.

## Architecture already approved

Your prior verdict is
`docs/FABLE_AUDIT_A210_R2_NEGATIVE_OPACITY_OPERATOR_2026-08-16.md`:
CMFGEN-equivalent non-overlap Sobolev operator, not sign-routed Gaussian and
not a maser/saturation expansion.

## Implementation to audit

1. Pure material/radiation kernel and tests:
   - `src/line_net_rate.h`
   - `src/line_net_rate.c:89-215`
   - `tests/line_net_rate_selftest.c:50-190`

   `line_net_cmfgen_exponx` copies CMFGEN's three branches. The companion
   `(1-beta)/tau` uses the shared polynomial at `|tau|<1e-3`. The homologous
   sigma=0 operator evaluates

   `Jbar = beta*J_cont + eta*(c*t/nu)*(1-beta)/tau_eff`

   without constructing `eta/chi`. The J-cont error is propagated upward as
   `nextafter(beta*error, +inf)`; this changes only a proof bound, not Jbar.
   The fixed Fe III line 2164811/shell0 witness and mild-negative/zero/positive
   CMFGEN direct-bracket identities pass, and injected NaN/negative-J defects
   fail closed.

2. Explicit operator selection and publication provenance:
   - `src/lumina_cmfgen.h:268-290`
   - `src/lumina_cmfgen.c:3476-3837`
   - `src/radiation_field.h:90-110`
   - `src/radiation_field.c:578-608`
   - `src/lumina.h:276-302`

   R1 is explicitly the sealed initialization shared-Gaussian operator so its
   seed-predictor input stays bit-identical. R2+ is explicitly CMFGEN
   non-overlap Sobolev. This is generation/pass selection, never sign
   selection. The canonical commit records a distinct static producer string.

3. Census, continuum-only assembly, and per-line application:
   - `src/lumina_cmfgen.c:4896-5685`
   - `src/lumina_cmfgen.c:6160-6345`
   - call site `src/lumina_cmfgen.c:7060-7090`

   Both passes independently validate every Q_E line/shell material. The R2
   shared line arrays remain exact zero, so the fine exact solver sees only
   continuum. Its Gaussian profile quadrature is retained solely to sample
   the continuum with the same registered R6/MC profile identity; the sampled
   J_cont is then passed through the per-line Sobolev response. The runtime
   `srce_chk` count must reproduce the pre-solve census exactly.

4. Fail-closed log judge:
   - `scripts/check_a210_targeted_gate.py:105-360`
   - `tests/a2_10_targeted_gate_selftest.py`

   It pre-registers R1 `(27748410,81265890,0,0,0)` and R2
   `(22866166,86148134,4246581,4246577,4)`, requires 109014300 finite Sobolev
   Jbar cells, exact policy reproduction, and zero repair fields.

## Local results

- `selftest_line_net_rate`: PASS, including direct-bracket identity and defect
  injection.
- A2-06 dual commit, A2-08, A2-09, A2-10 N1--N8, cancellation/refinement and
  targeted checker controls: PASS.
- CPU/OpenMP build: PASS.
- CUDA sm80/sm86/sm90 build: PASS.
- CUDA binary SHA256:
  `8f9d2865647a568f4c9367de4fd67726a54d64f98519032bf9fa8b20f2b0cccf`.

## Questions

1. Is the stable Jbar formula exactly equivalent to CMFGEN `SOBJBAR_SIM` for
   non-overlap, homology sigma=0, including mild negative tau and typed
   tau<-0.5 material?
2. Is Gaussian averaging of the continuum-only J acceptable as the registered
   finite-profile evaluation of CMFGEN's sigma=0 BETAC/J_cont, or is a distinct
   line-centre interpolation required before A100?
3. Is the explicit R1 sealed-init/R2+ Sobolev split consistent with your prior
   simultaneous requirements that R1 remain bit-exact and production parity
   remove shared line deposition?
4. Find any data race, leak, incomplete fail-closed path, error-envelope error,
   double counting, or provenance mismatch in the cited code.
5. Return `VERDICT: APPROVE`, `REVISE`, or `BLOCK`; list only required changes
   before A100 and separate optional diagnostics.
