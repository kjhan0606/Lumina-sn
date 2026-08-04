# toy06 CMFGEN composition alignment contract

## Verdict before generation

The requested production geometry cannot pass all composition gates without a
forbidden extrapolation or a geometry change.  The canonical `SN_HYDRO_DATA`
contains 700 zone-centred samples at 1025–35975 km/s, with measured uniform
spacing 50 km/s and physical cell edges 1000–36000 km/s.  The certified Lumina
deck geometry contains 50 shells with edges 3900–40300 km/s.

The overlap is therefore 3900–36000 km/s.  CMFGEN material at 1000–3900 km/s
has no Lumina shell.  Lumina shell 44 (35932–36660 km/s) is only partially
covered, and shells 45–49 are wholly outside the canonical source.  The mapper
does not evaluate, clamp, repeat, or substitute a value there.  It records
shells 44–49 as incomplete and deliberately omits runnable composition files.
Consequently the full-ejecta `SPECIES_MASSES` total, 0.99393 Msun, cannot be
represented on this geometry.  This is an unresolved input-domain conflict,
not a tunable tolerance.

## Coordinate and conservative mapping rule

Both coordinates are measured, not assumed:

- CMFGEN radius is read from `Radius grid (10^10cm)` and checked against the
  separately read `Velocity (km/s)` using `r = v t` at 19.48 d.  The source
  cell edges are the midpoints between its uniform centres; the measured end
  edges must be exactly 1000 and 36000 km/s.
- Lumina `geometry.csv` radius and velocity edges are independently read.  Each
  edge must satisfy the same homologous relation, and adjacent velocity edges
  must be contiguous.

For source cell `j`, Lumina shell `s`, and species `k`, the exact spherical
intersection volume is used:

```text
DeltaV_js = 4 pi/3 [r_hi^3 - r_lo^3]
X_k,s = sum_j rho_j X_k,j DeltaV_js / sum_j rho_j DeltaV_js
rho_s = sum_j rho_j DeltaV_js / DeltaV_s
```

This is mass weighting.  It preserves both total and per-species mass under a
finite-volume restriction when the target domain is fully covered.  Pure
volume weighting would not preserve species mass because toy06 density varies
strongly with radius.  Partial or zero coverage is a hard failure.

The canonical six elemental blocks are `SIL`, `SUL`, `CAL`, `IRON`, `COB`, and
`NICK`, written as Z=14,16,20,26,27,28.  `VADAT` and the `MOD_SUM` abundance
table are not read.

## Isotope decision

`NICK56`, `COB56`, and `IRON56` are mapped with the same `rho*dV` rule and are
written separately to `isotopes.csv`; elemental rows remain total elemental
mass fractions used for opacity and ionization.

The present source code cannot consume that file.  In
`compute_gamma_deposition`, it treats all elemental Ni and Co as initial
radioactive material and applies a Bateman evolution for another 19.48 d.  The
canonical isotope blocks are already at 19.48 d, so this is double decay.
`patches/toy06_cmfgencomp_isotopes.patch` is prepared but intentionally not
applied.  It:

1. makes shortened abundance rows fail instead of silently leaving shells
   zero;
2. loads exactly Ni56, Co56, and Fe56 for all shells; and
3. computes gamma production from current-epoch Ni56 and Co56 directly, with
   no elemental fallback and no second Bateman evolution.

No Lumina model run is valid with this new deck until that patch (or an
equivalent reviewed implementation) is applied and verified.  The existing
external-deposition path is not used as a substitute in this preparation.

## Six gates and implementation

`scripts/verify_toy06_cmfgencomp_deck.py` is read-only and implements:

1. exact abundance row set `{14,16,20,26,27,28}`;
2. 50 per-shell sums with absolute tolerance `5e-9`, plus exact three-isotope
   identity, subset, and remapping checks;
3. finite-volume per-species and total mass against `SPECIES_MASSES` (total
   tolerance `5e-6 Msun`, species tolerance `5e-5 Msun`, matching the printed
   precision of that file);
4. direct canonical comparison of an inner and outer fully covered shell with
   absolute tolerance `5e-10`;
5. exactly 50 populated, nonzero, fully source-covered shells; and
6. byte identity of certified non-composition inputs followed by the existing
   R1/R4 verifier, including its OFF-control byte gate.

Any failure is accumulated, printed, and returned as a nonzero process status.
No normalization, retry, or repair is performed by the verifier.

## Prepared files and execution boundary

- `scripts/toy06_cmfgen_composition.py`: canonical parser and conservative map
- `scripts/build_toy06_cmfgencomp_deck.py`: refuse-overwrite candidate builder
- `scripts/verify_toy06_cmfgencomp_deck.py`: final six-gate verifier
- `scripts/selftest_toy06_cmfgencomp.py`: positive and injected-failure fixtures
- `scripts/sbatch_toy06_cmfgencomp_deck.sh`: CPU-only generation/verification job
- `patches/toy06_cmfgencomp_isotopes.patch`: unapplied runtime isotope patch

The builder creates only the new
`data/tardis_reference_toy06_19p48d_cmfgencomp/` path.  It never modifies or
deletes the five existing toy06 decks.  With the current geometry it writes an
audit report and `COMPOSITION_INVALID`, omits abundance/isotope/density output,
and leaves the final verifier to fail the Slurm job.

The preregistered direction is Co-dominated deep material, Si/S/Ca outer
material, and zero abundance for C/O/Mg/Al/Sc/Ti/V/Cr/Mn.  Spectra, ionization,
and temperature are expected to change globally.  Post-run before/after metrics
remain pending because GPU/model execution and `validation/regression_ledger/`
changes are outside this task.
