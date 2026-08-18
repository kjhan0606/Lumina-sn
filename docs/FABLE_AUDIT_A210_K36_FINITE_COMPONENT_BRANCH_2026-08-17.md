# Fable audit verdict — A2-10 K36 finite-component branch

Fable verified the two named artifacts against their sealed SHA-256 values.
The component closures, CMFGEN bit-exact depth totals, and zero repair markers
were internally consistent. The state-unmatched comparison remains diagnostic,
not a parity claim.

## Verdict

The total absorption deficit alone cannot prove a Jbar fault because line
absorption also contains lower-level populations and line opacity. However,
the owner-resolved saturation ratios make the radiation-absorption/Jbar path
the priority hypothesis:

| Owner | Lumina absorption/emission | CMFGEN absorption/emission |
|---|---:|---:|
| Co IV | about 0.0020 | about 0.999997 |
| Fe IV | about 0.0018 | about 0.999997 |
| Ni IV | about 0.0018 | about 0.999997 |
| Fe/Co/Ni III | 0.897/0.913/0.984 | about 0.99999 |

The dominant IV-stage emission differs by only 1.38--1.49 times, whereas its
absorption is deficient by roughly 330--400 times. This makes a pure population
normalization explanation unlikely, but does not exclude a Sobolev-opacity or
lower-population defect. Thus Jbar is a priority hypothesis, not yet a finding.

## Exactly one authorized diagnostic branch

For the shell-0 lines cumulatively carrying at least 90% of the Lumina Co IV,
Fe IV, and Ni IV emission, cross-match each transition to CMFGEN depths 67/68
and record:

- Lumina Sobolev tau, which is independent of Jbar;
- Lumina `Jbar/S_line`;
- the escape probability `beta(tau)`;
- CMFGEN `1-ZNET`, equivalent to its serialized absorption/emission ratio.

This is read-only extraction. If Lumina tau is optically thick but `Jbar/S` is
far below the line's own trapping expectation `1-beta(tau)`, the FUV Jbar
self/local coupling is defective. If Lumina tau is optically thin where
CMFGEN `1-ZNET` indicates saturation, the opacity/lower-population/line-universe
path takes priority instead.

## Threshold for a later physical code change

A code change is not authorized by aggregate ratios. It requires:

1. localization of a specific incorrect physical expression in the Jbar or
   opacity path;
2. offline recomputation on the sealed K36 state showing that the corrected
   expression restores the IV-stage tau/Jbar consistency while the III-stage
   null control remains stable;
3. preregistered expectations and a negative-control failure in one verdict
   run.

No floor, cap, scaling, deletion, or numerical repair is acceptable. The
rejected pre-core tau refresh and coevolution generation barrier remain intact.
