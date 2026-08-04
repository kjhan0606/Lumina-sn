# StaNdaRT toy06 composition-to-physics audit (2026-08-03)

Scope: read-only audit of the current dirty working tree.  `src/` was not
modified.  Proposed source changes exist only in
`patches/standart_abundance_path_audit.patch`.

## 1. 30/50 defect

The blanket statement "the current toy06 abundance file has 30 shell columns"
needs a path qualifier:

- `data/tardis_reference_toy06_19p48d/abundances.csv`: 50 shell columns.
- `data/tardis_reference_toy06_19p48d_sivcaiv/abundances.csv`: 50.
- `data/tardis_reference_toy06_19p48d_sivcaiv_ftos/abundances.csv`: 30.
- `data/tardis_reference_toy06_19p48d_sivcaiv_fullcov/abundances.csv`: 30.
- `data/tardis_reference_toy06_19p48d_sivcaiv_links/abundances.csv`: 30.

All five geometries have 50 shells.  The certified `ftos` deck selected as the
new deck's immutable base is therefore affected.

The defect mechanism is confirmed.  `src/lumina_atomic.c:818-819` allocates
`n_elements*n_shells` zeroes.  Lines 823-838 ignore the CSV header width and
call `strtod` exactly `n_shells` times.  At end-of-line `strtod(p,&p)` returns
0.0 without advancing `p`; shells 30-49 therefore retain/receive zero for every
element.  `tests/abundance_loader_short_row_fixture.c` executes this exact
pointer pattern against libc and requires 30 nonzero parses, 20 implicit zeroes,
and 20 pointer stalls.  This proves the runtime language/library behavior
without executing Lumina.

Consequence: in each affected deck, all elements in shells 30-49 are zero.  No
loader warning or counter exists in the unpatched tree.

## 2. Loader to physics path

Current-tree line references:

1. `src/lumina_main.c:110-120` selects the deck, loads reference geometry/plasma,
   then calls `load_atomic_data`; CUDA entry is `src/lumina_cuda.cu:6880-6890`.
2. `src/lumina_atomic.c:287-294` reads geometry and establishes `n_shells`;
   `src/lumina_atomic.c:386-388` reads density into `plasma->rho`.
3. `src/lumina_atomic.c:808-819` gets the 15-element topology from
   `atom_masses.csv` and allocates `atom->abundances`; lines 820-842 parse it.
   Independently, `src/lumina_atomic.c:465-491` loads the deck's precomputed
   electron-density/Sobolev-opacity tables.  Those tables were made for the
   copied deck's old composition; the unpatched iteration-0 path does not
   invalidate them.
4. `src/lumina_atomic.c:844-902` builds every element's ion ladder and level
   offsets independent of abundance.  Lines 980-983 allocate per-ion/per-shell
   population and partition arrays, also independent of abundance.
5. `src/lumina_main.c:579-590` (CPU) and `src/lumina_cuda.cu:10634` (GPU path)
   call `compute_plasma_state`, then BF and NLTE.  The master sequence is
   `src/lumina_plasma.c:6340-6391`: partition functions -> electron density ->
   ion populations -> optional frozen-in overwrite -> electron opacity ->
   Sobolev opacity.
6. Mass fraction first becomes an element number density at
   `src/lumina_plasma.c:2260-2262` and again inside the iterative electron solve
   at lines 2453-2460.  Ion populations are normalized to that element total at
   lines 2365-2381 / 2522-2529.
7. Line opacity consumes `ion_number_density` at
   `src/lumina_plasma.c:2582-2681`; BF consumes it at lines 7086-7151 and level
   populations at 7174-7200.  Transport consumes electron and BF continuum
   opacity at `src/lumina_transport.c:525-544` and the previously built Sobolev
   line table in `trace_packet`.
8. NLTE targets are a fixed 31-slot table containing all 15 elements at
   `src/lumina_plasma.c:7677-7684`; `nlte_init` builds the projection without an
   abundance predicate at lines 14256-14311.  Conservation RHS totals come from
   `nlte_pair_total_density` (1808-1830) and are written at 16607-16632.
9. Dynamic macro-atom topology is global: `compute_transition_probabilities`
   starts at `src/lumina_plasma.c:3498`, iterates `n_macro_levels`/blocks, and
   writes the existing ion-indexed arrays.  BF sigma and collision tables are
   likewise loaded by atomic identity, not by abundance.

There is a second path defect at iteration 0: the copied `tau_sobolev.npy` is
transport-visible before the first composition-derived plasma refresh (the
ordinary refresh is guarded by `iter > 0`).  The proposed patch adds the
explicit `LUMINA_REBUILD_INITIAL_PLASMA=1` gate immediately after initial
`T_e` construction in both entry points and prints
`[ABUNDANCE-INITIAL-PLASMA]`.  This gate is required on any later physical run
of the new deck; without it, iteration 0 uses stale opacity even though the new
CSV was parsed.

There is no abundance clamp, default substitution, or abundance
renormalization between CSV and `n_element`.  Partition functions have their
own `1e-300` numerical floor (`src/lumina_plasma.c:1927-1929`), but that is not a
mass-fraction normalization.

## 3. Exact-zero downstream behavior

Unpatched verdict: zero species are not cleanly excluded.

- Every ion and macro-atom slot remains allocated because topology follows the
  15-row `atom_masses.csv`, not the six abundance rows.
- The ordinary and iterative ion solvers set upper stages to `1e-300` when the
  computed value is zero (`src/lumina_plasma.c:2374-2381`, 2522-2529).
  Frozen-in and coupled charge writebacks repeat this behavior at 6321-6324 and
  12603-12605.
- `LUMINA_NLTE_SKIP_DEAD` detects pair totals below `1e-10`
  (`src/lumina_plasma.c:8173-8189`, 16699-16718), but deliberately routes them
  to the Boltzmann fallback.  In the fallback, scaling occurs only when
  `n_total>0`; for exactly zero total the scale remains 1 and positive LTE-shape
  populations remain (`src/lumina_plasma.c:16925-16957`).  The analogous GPU
  branches are `src/lumina_cuda.cu:1493-1525` and 1733-1741.  Thus the skip is
  not an exclusion proof and can manufacture population for an absent element.
- BF is protected downstream by `n_ion < 1e-30` at
  `src/lumina_plasma.c:7144-7152` (GPU GEMM mirror
  `src/lumina_bf_gemm.cu:81-89`), so the `1e-300` ion floor does not create BF
  opacity there.
- Sobolev opacity instead forces a generic `1e-100` floor at
  `src/lumina_plasma.c:2678-2681`, so absent lines are not exact zero.
- Macro-atom/sigma/collision topology continues to occupy memory/index slots.
  With exact zero opacity and zero k-packet weights it is unreachable, but the
  unpatched NLTE/floor behavior prevents using topology presence as proof of
  physical exclusion.
- Searches for division by abundance/`n_element` found no unguarded production
  division.  `src/lumina_element_wide.c:1778-1784` explicitly returns before
  `density/n_elem` when `n_elem<=0`; diagnostic ratios at 2206, 2211, 2265,
  2364-2365 are guarded.  Total-atom divisions in plasma paths are also guarded
  by `>0` checks.
- `LUMINA_SIMUL_CAP_TOPION` is abundance-independent once a ladder is admitted,
  but the simultaneous solver skips elements with `nel<=0` at
  `src/lumina_plasma.c:10040-10048`; `simul_ladder` therefore has no zero-element
  ladder to cap.  Global slots still remain.

The proposed patch makes short/long rows fatal, prints one per-element line and
an `[ABUNDANCE-SUMMARY]`, preserves missing abundance rows as exact calloc zero,
clears zero-species ion/NLTE populations after both CPU and GPU solves, and
writes exact zero line opacity/source for absent elements.  Its
`[ABUNDANCE-ZERO-NLTE]` counter reports both excluded slots and any nonzero
residual found before clearing.  It also provides the explicit iteration-0
plasma rebuild above.  Topology is retained, but physical population and
opacity ownership becomes exact zero.

The model-free negative-control fixture applies the patch only to a disposable
source copy, seeds nonzero ion/NLTE populations behind exact-zero abundances,
and observed:

`[ABUNDANCE-ZERO-NLTE] excluded_ion_shell_slots=3 cleared_level_cells=5 residual_nonzero_before_clear=10 policy=EXACT_ZERO`

followed by `ZERO_NLTE_FIXTURE ... PASS`.  The one active shell's sentinels
remain unchanged, so the counter is an exclusion test rather than a blanket
population wipe.

## 4. Mass normalization

Lumina does not renormalize `atom->abundances`; it will run with any finite sums
the current permissive parser accepts.  The proposed loader reports min/max
shell sums and explicitly labels `runtime_renormalization=NONE`; it still does
not alter inputs.

StaNdaRT ASCII rounding makes the raw six-species sums differ from unity by up
to order `1e-5`.  The new builder preregisters and reports one explicit
six-species normalization per native source cell before conservative
restriction.  This mirrors the supplied CMFGEN policy and is not a runtime
clamp or hidden repair.  The verifier recomputes it independently.

## 5. Mapping and gates

Measured coordinates:

- source centres: 100..40,300 km/s in 200 km/s increments;
- inferred cell edges: 0..40,400 km/s;
- Lumina geometry: 50 contiguous shells, 3,900..40,300 km/s;
- both source and target radii satisfy `r=v*t` at their declared epochs.

For every target/source cell intersection, the source cell mass contribution is

`dmass[j] * (v_hi^3-v_lo^3)_intersection / (v_hi^3-v_lo^3)_source`.

Element and isotope numerators use that same mass.  Division by target-shell
mass gives abundance; shell mass divided by target spherical volume gives
density.  This conserves each normalized species over the exact represented
domain.  Coverage below 1 is fatal and recorded; there is no extrapolation,
clamp, or replacement.

Gate implementation in `scripts/verify_toy06_standart_deck.py`:

1. exact six abundance rows and exact 50-column header: `gate1`;
2. canonical Ti/O/C 202/202 exact-zero census: `gate2`;
3. shell sum, exact remap, density, and species/isotope mass conservation:
   `gate3` (`2e-12` sum and mass tolerances are printed);
4. analytic primary decay plus common-coverage comparison to CMFGEN
   `SN_HYDRO_DATA`: `gate4` (`5e-5` absolute, preregistered for printed/regridded
   source precision);
5. all 50 shells covered and nonzero: `gate5`;
6. Co-dominant core and 0.55/0.35/0.10 Si/S/Ca exterior: `gate6`;
7. immutable base bytes and the existing R1/R4 verifier: `gate7`.

## UNRESOLVED

- The external CMFGEN run directory is unavailable in this workspace, so the
  real secondary gate and R1/R4 gates cannot be executed here.
- GPU/model execution is forbidden.  The proposed patch was syntax-context
  checked with `git apply --check` and applied/linked successfully only in a
  disposable CPU build copy.  CUDA compilation and the driving-seat dump-only
  runtime remain unexecuted.
- `isotopes.csv` is provenance only: no source loader consumes it.  Moreover
  `compute_gamma_deposition` treats elemental Ni/Co as t=0 and decays them again
  (`src/lumina_plasma.c:17626-17658`).  In the unpatched CUDA entry point,
  `LUMINA_DEPOSITION_FILE` is read at `src/lumina_cuda.cu:7743-7767`, but the
  result is overwritten by unconditional internal deposition at lines
  10594-10599 on later iterations.  The proposed patch makes the external file
  fail closed (all 50 unique shells required), locks it against that overwrite,
  and recomputes only its non-thermal derivative.  That change is not active
  until the patch is applied and runtime-tested by the driving seat.
- Atomic and macro-atom topology remains globally allocated for the nine absent
  elements.  The patch makes those slots physically unreachable per shell; it
  does not compact/reindex certified R1/R4 atomic data.
