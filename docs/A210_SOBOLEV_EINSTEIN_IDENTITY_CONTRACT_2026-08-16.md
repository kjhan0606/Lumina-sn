# A2-10 Sobolev--Einstein line-energy identity contract

Status: **DIAGNOSTIC CONTRACT — no production physical fix has been selected**  
Date: 2026-08-16

## 1. Question being decided

The k=24 non-census gate closes the numerical cancellation census but fails
closed with `RADEQ_NO_BRACKET` in shells 0--3.  The present question is whether
the finite negative LOWER residual is physical, or whether it is the residual
of two large line terms built from inconsistent representations of the same
atomic transition.

No negative value is clipped and no floor, cap, clamp, jitter, diagonal shift,
bracket enlargement, or post-hoc repair is permitted.  A counterfactual in
this investigation is diagnostic only and must not be published as a material
state.

## 2. Required identity

The canonical A2-06 SE matrix consumes

```text
R_lu       = B_lu Jbar
R_ul,stim  = B_ul Jbar
R_ul,spont = A_ul
```

while the R7 direct line bracket consumes, per steradian,

```text
eta = n_u A_ul h nu / (4 pi)
chi_int = tau nu / (c t_exp)
net = eta - chi_int Jbar .
```

For a signed Sobolev depth

```text
tau = K_S f_lu wavelength_cm t_exp
      * (n_l - n_u g_l/g_u),
```

the absorption pieces are the same physical rate only if every represented
line satisfies

```text
K_S f_lu (wavelength_cm nu/c) = h nu B_lu/(4 pi),
B_ul = B_lu g_l/g_u,
A_ul = (2 h nu^3/c^2) B_ul.
```

`wavelength_cm*nu/c` is kept explicit.  Assuming it is exactly one would hide
serialization error in the present deck.

## 3. Static evidence already established

The runtime header contains

```text
SOBOLEV_COEFF = 0.026540281
```

whereas the constants already adopted by the I20 atomic-data contract give

```text
e_esu = (1.602176634e-19 C) c/10
pi e_esu^2/(m_e c) = 0.026540088545744744.
```

Thus the runtime coefficient is larger by
`7.251454904544374e-06` relative, or `7.2514549 ppm`.

The streaming audit of all 2,588,798 lines in the sealed k=24 gate deck found
zero invalid rows:

| quantity | minimum | maximum | signed mean |
|---|---:|---:|---:|
| `h nu B_lu/(4 pi f_lu K_exact) - 1` | `-1.4082516531788158e-6` | `+1.4145372606577666e-6` | `-6.091009234811596e-11` |
| `wavelength_cm nu/c - 1` | `-4.985068036145179e-7` | `+4.992908417822406e-7` | not used as a correction |
| Einstein opacity / actual runtime transport opacity `- 1` | `-9.145003031618693e-6` | `-5.364827160070362e-6` | `-7.251441834474044e-6` |

The common-mode offset is therefore the stale runtime constant; the smaller
line-dependent spread comes from separately serialized `f`, `nu`, wavelength,
and `A/B`.  The current generator writes several of these columns with only
six digits after conversion, and the finalizer derives `B` from the rounded
`A`.

There is also a separate generator-stage convention defect:
`write_line_list_csv()` and the static macro-atom internal-up weights use
`A c^2/(8 pi h nu^3)`, while the I20 contract and the finalizer use
`A c^2/(2 h nu^3)`.  The finalizer rewrites the line-list `B` columns but does
not rebuild the already-written static macro-atom weights.  The current gate
sets `LUMINA_DYNAMIC_TRANSPROB=1`, so this dormant static fallback is not the
direct explanation of the present R7 residual; nevertheless a regenerated
source-of-truth deck must not preserve it.

The reproducer is `scripts/check_sobolev_einstein_identity.py`; its independent
known-answer test is `tests/sobolev_einstein_identity_selftest.py`.  Both are
read-only and report `physical_mutation=0`, `repair=0`.

## 4. Dynamic discriminator

`a210_private_line_energy_build()` retains the existing signed physical sum
unchanged and, only when `LUMINA_RADEQ_DIAG` is set, also accumulates a second
sum in which each ordinary raw-opacity cell is evaluated with the exact
Einstein-implied opacity.  CMFGEN `SRCE_CHK` cells remain under their declared
negative-opacity policy and are not silently reinterpreted.

The diagnostic ratio includes the serialized `wavelength_cm*nu/c` factor.  An
earlier diagnostic binary omitted that factor; all of its runs are rejected
and may not be cited as physical evidence.  The final diagnostic accumulates
three sums in the same cell traversal: current runtime opacity, exact global
Sobolev constant with the existing serialized line columns, and per-line
Einstein opacity.  This separates the common constant error from deck
serialization without rescaling `SRCE_CHK` cells after the fact.

The dynamic cause is established only if the same candidate state and same
`Jbar`, populations, and atomic rows give a finite counterfactual change that
accounts for the LOWER/MID/UPPER residual behavior.  Merely observing that the
constants differ is not sufficient to change production physics.

## 5. Prohibited pseudo-fixes

- Do not multiply a published R7 result by a fitted per-line ratio.
- Do not replace a negative residual by zero or its absolute value.
- Do not add a diagonal jitter or exponentiate/reparameterize the matrix to
  hide this identity violation.
- Do not enlarge the temperature bracket or accept a same-sign endpoint.
- Do not silently regenerate or overwrite the production atomic deck.

## 6. Conditions for a root repair

If the dynamic discriminator confirms causality, the repair must establish one
atomic source of truth by construction, not tune the residual.  Before editing
production physics, register the exact changed set.  At minimum:

1. use one declared set of `e`, `m_e`, `c`, and `h` constants for Sobolev and
   Einstein forms;
2. preserve CMFGEN's authoritative oscillator strength and level-energy
   lineage;
3. serialize or derive `nu`, wavelength, `A_ul`, `B_lu`, and `B_ul` with enough
   precision that the three identities above close at binary64 roundoff;
4. derive both line-list and static macro-atom `B` weights with the declared
   `c^2/(2 h nu^3)` convention before normalization;
5. reject inconsistent active decks at validation time instead of repairing
   individual runtime cells;
6. pass known-answer, negative-control, full-deck identity, A100x2 k=24 gate,
   H200 cross-device, and same-state finite CMFGEN comparisons with every
   floor/cap/clamp/jitter/repair field equal to zero.
