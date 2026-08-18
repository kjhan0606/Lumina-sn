# Fable audit: A2-10 non-overlap Sobolev implementation

- Date: 2026-08-16 KST
- Model: `claude-fable-5`
- Mode: read-only code audit
- Query: `docs/QUERY_FABLE_A210_NONOVERLAP_SOBOLEV_CODE_AUDIT_2026-08-16.md`
- Verdict: **APPROVE**
- Required changes before A100: **none**

## Verdict

> **VERDICT: APPROVE**
>
> **Q1 — Equivalent: YES.** `line_net_cmfgen_exponx` reproduces CMFGEN's
> three EXPONX branches. Evaluating the shared small-tau polynomial before the
> subtraction gives the finite `(1-beta)/tau` limit at zero. The stable form
>
> `Jbar = beta*J_cont + eta*(c*t/nu)*(1-beta)/tau_eff`
>
> is algebraically identical to `beta*J_cont + (1-beta)*S` without constructing
> the singular `eta/chi`. Mild negative tau stays finite and positive under the
> per-line operator. The typed `tau<-0.5` material maps CMFGEN's internal
> `CHIL=1.0` to `1e-10` cgs, divided by the simultaneous-line count.
>
> **Q2 — Gaussian continuum averaging is acceptable.** It preserves the
> registered GAUSS_VD10/R6/MC profile identity. A separate line-centre
> interpolator is not required before A100. A sharp bound-free edge inside a
> four-Doppler support is an optional later diagnostic, not a blocker.
>
> **Q3 — The explicit R1/R2 split is consistent.** Selection is pass-indexed,
> never sign-indexed: R1 retains the sealed shared-Gaussian initialization
> input; R2 and later use the non-overlap Sobolev operator. In R2 the shared
> line-deposit loops are disabled, so the exact fine solve sees continuum only.
> Distinct static producer strings and the generation/order checks preserve
> provenance and the coevolution barrier.
>
> **Q4 — No race, leak, fail-closed gap, double counting, or provenance
> mismatch found.** The material and runtime `srce_chk` counts are independently
> reconciled; `jbar_line_det_operator` is zero until successful completion;
> the population cache is freed; sentinel pairing, positive finite beta, and
> finite Jbar are enforced. Exact-zero tau cells may correctly publish finite
> Jbar because beta=1 and the zero-tau source limit remains finite.
>
> `nextafter(beta*err_J,+inf)` is a valid upper bound on propagated continuum
> input error. It does not claim to include the few-ulp arithmetic rounding of
> the Sobolev operator itself.
>
> **Q5 — Required changes before A100: NONE.**

## Optional diagnostics

1. Keep the propagated-input-error-only meaning of the Jbar uncertainty
   explicit; operator floating-point rounding is not included.
2. Later, compare profile-averaged and line-centre continuum J for the subset
   whose support straddles a sharp bound-free edge.
3. A future enum rename may replace the historical `SRCE_CHK` label with the
   exact CMFGEN `CHECK_LINE_OPAC && SOBOLEV` branch name; naming only.

## Codex disposition

Accepted. No optional item is promoted into a pre-A100 requirement. Physical
material remains immutable; no floor, cap, clamp, jitter, or repair was added.
