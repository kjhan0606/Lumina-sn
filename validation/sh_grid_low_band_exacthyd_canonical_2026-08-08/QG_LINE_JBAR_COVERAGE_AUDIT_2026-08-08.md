# Q_g line-Jbar coverage audit — 2026-08-08

## Scope and source

- active deck: `data/tardis_reference_toy06_19p48d_sivcaiv_active`
- source: `line_list.csv`
- runtime selection reproduced exactly: every line whose `(Z, ion_number)` is
  present in the base 31-slot `NLTE_TARGET_Z/NLTE_TARGET_ION` table
  (`nlte_line_map >= 0`)
- pre-repair runtime call: `line_jbar_qset_build(..., bb_in_domain=NULL)`
- current fine-Jbar producer window: strict `1000 < wavelength_A < 4000`
- amended A2-02C BB line-centre domain: closed `100 <= wavelength_A <= 20000`,
  equivalently `[1.49896229e14, 2.99792458e16] Hz`, domain-contract hash
  `3278062cf80281ffdcc4eb74ffc37e743cbdc51a128da5a319bfba7d3a6416c4`
- the wider option-B canonical `RadiationField` is the union owner for BB, BF,
  and other consumers.  Its outer edges are not the BB line-ID selection rule.

This is a static census. No model or CUDA binary was executed on the login
node.

## Exact census

| wavelength cohort | lines | fraction of pre-repair Q_g |
|---|---:|---:|
| below 100 A | 0 | 0.000000% |
| 100–1000 A | 140,368 | 7.895339% |
| current fine window, 1000–4000 A | 533,172 | 29.989555% |
| 4000–20000 A | 717,591 | 40.362650% |
| beyond 20000 A | 386,728 | 21.752456% |
| total pre-repair Q_g | 1,777,859 | 100.000000% |

Thus the repaired runtime Q_g is 1,391,131 lines (78.247544% of the pre-repair
set). Of those, 857,959 lines are physically in the BB domain but remain
`UNSAMPLED` while the fine producer is restricted to 1000–4000 A. The other
386,728 mapped NLTE lines are `BB_EXCLUDED_OUTSIDE_DOMAIN`: they retain atomic
bookkeeping, spontaneous `A_ul`, and collision terms, but do not own a J-driven
BB rate edge.

The current 533,172-line window count exactly reproduces smoke #5 R6
`valid_lines=533172`, so the observed `UNSAMPLED` count is completely explained
by the producer window; it is not a shell-dependent extraction failure.

## Ion census

| Z | ion | total | 100–1000 A | 1000–4000 A | 4000–20000 A | outside BB domain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | 1 | 1,840 | 37 | 489 | 643 | 671 |
| 14 | 2 | 1,639 | 163 | 669 | 586 | 221 |
| 20 | 1 | 701 | 0 | 161 | 282 | 258 |
| 20 | 2 | 3,497 | 620 | 1,077 | 1,180 | 620 |
| 26 | 1 | 488,840 | 10,139 | 131,371 | 214,969 | 132,361 |
| 26 | 2 | 136,263 | 29,763 | 59,026 | 39,842 | 7,632 |
| 16 | 1 | 8,506 | 450 | 2,779 | 3,538 | 1,739 |
| 16 | 2 | 4,534 | 811 | 1,809 | 1,333 | 581 |
| 27 | 1 | 507,723 | 11,473 | 120,983 | 221,238 | 154,029 |
| 27 | 2 | 505,993 | 73,246 | 170,332 | 190,625 | 71,790 |
| 28 | 1 | 51,812 | 3,213 | 14,565 | 21,693 | 12,341 |
| 28 | 2 | 66,511 | 10,453 | 29,911 | 21,662 | 4,485 |

Only these twelve target ions have lines in the active deck. Global wavelength
extrema of the selected set are 250.088844 A and 1.5625e9 A.

## Structural finding

The amended A2 grid contract says `Q_g` is the enabled bound-bound rate graph
intersected with the closed 100–20000 A `BB_IN_DOMAIN`. The pre-repair callers
passed a null domain mask and therefore put 386,728 excluded lines into Q_g.
Widening the fine mesh to the full raw Q_g would extend it to 156.25 m. That is
not a valid coverage repair.

The next implementation must do both:

1. restore an explicit runtime `BB_IN_DOMAIN` selection bound to domain-contract
   hash `3278062c...` and keep structural exclusions distinct from invalid cache
   requests;
2. solve and publish every in-domain Q_g line, with no 1000–4000 A window
   sentinel and with a quantitative convergence residual.

At 10 km/s and 12 points per Doppler width, the current 1000–4000 A mesh has
498,721 bins. The complete registered support of 100–20000 A line centres spans
99.986659–20002.668869 A and needs about 1,906,171 bins. That is the minimum
profile-support mesh, not the complete causal transfer mesh: blue-to-red
characteristics require the canonical 74.274847-A blue reservoir. The resulting
production mesh has 2,013,113 bins. For 50 shells, the seven existing host field
arrays occupy about 5.250 GiB; with the current exact solver workspace the peak
estimate is about 109.972 GiB. This is feasible on the sealed H200 node's host
RAM. Convergence and complete Q coverage are now mandatory commit gates.
