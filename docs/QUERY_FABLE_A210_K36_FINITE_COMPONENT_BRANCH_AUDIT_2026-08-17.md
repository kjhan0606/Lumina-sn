# Fable audit request — A2-10 K36 finite-component branch

Audit only. Do not edit code, launch jobs, or broaden the search. Give a
concise physical verdict and one next discriminating branch.

## Non-negotiable contract

- No physical-value floor, cap, clamp, jitter, absolute-value repair, deletion,
  or replacement.
- Negative/nonfinite values fail closed until their cause is proved.
- Rejected pre-core tau refresh stays rejected; the coevolution generation
  barrier stays intact.
- The present comparison is explicitly state-unmatched and is not a parity
  claim.

## Sealed evidence

- K36 run root:
  `/gpfs/kjhan/lumina/a210_line_owner_a100x2_nonoverlap_sobolev_k36/diag_20260816T194201Z_3ec4239f2d76`
- Strict REQUESTED_TE owner report: SHA-256
  `460b0dfd19ef727254165386ebefd217b12647150a3aba23b7d87bb450dd08e4`;
  4 shells, 108 owner records, complete PASS, no mutation/repair.
- Corrected component comparison:
  `validation/a2_10/A2_10_NONOVERLAP_K36_CMFGEN_ION_COMPONENT_COMPARISON_2026-08-17.json`,
  SHA-256 `d8f65d4c0b8214fae2d047421b0dfb4065291887bd606b39b25a58f682f5775e`.
- CMFGEN component owner source:
  `validation/a2_10/A2_10_CMFGEN_LINE_COMPONENTS_ION_OWNERS_2026-08-17.json`,
  SHA-256 `aab5d3ea13504b255f700be95268bc22306e3d73e5b254a6b8076bf0457ea66e`;
  789,775 paired LINEHEAT/NETRATE records; depth totals remain bit-exact.
- The CMFGEN label bug is resolved from native conventions:
  `SIX=VI`, `SEV=VII`, `Sk=Si`, `Nk=Ni`. All 27 ion owners now match.

At exactly `T_e=19059.411196903675 K`:

| Quantity (erg cm^-3 s^-1) | Lumina | interpolated CMFGEN | L/C |
|---|---:|---:|---:|
| signed net | 720.427669987422 | 0.0016469023429593025 | 437444.07 |
| scaled emission | 723.428290410326 | 514.271420109047 | 1.406705 |
| scaled absorption | 3.00062042290404 | 514.269773206705 | 0.00583472 |

`n_e(CMFGEN)/n_e(Lumina)=1.08202684`; electron density, ion/level
populations, and radiation field/Jbar are not otherwise matched.

Dominant Lumina owners:

| Owner | emission L/C | absorption L/C | net L/C |
|---|---:|---:|---:|
| Co IV | 1.48859 | 0.00303097 | 5.754e5 |
| Fe IV | 1.44995 | 0.00267484 | 7.169e5 |
| Ni IV | 1.38167 | 0.00252187 | 3.668e5 |

CMFGEN components are diagnostic reconstructions from serialized LINEHEAT and
NETRATE; authoritative signed LINEHEAT remains independently bit-exact.

## Questions

1. Does this evidence justify prioritizing the radiation-absorption/Jbar path,
   or is that inference too strong because `chi_line` and populations are also
   unmatched?
2. Choose exactly one minimal, non-mutating diagnostic branch that best
   separates a Jbar deficit from a line-opacity/population or line-universe
   mismatch. State its required observables and pass/fail criterion.
3. State what evidence would be sufficient to authorize a later physical code
   change. Do not recommend any numerical repair.
