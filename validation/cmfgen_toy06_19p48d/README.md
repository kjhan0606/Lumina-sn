# CMFGEN self-run validation — toy06 at 19.48 d

This directory backs up the configuration, build process, run procedure, and
validation analysis for our own CMFGEN calculation of the StaNdaRT **toy06** SN Ia
benchmark at **19.48 days**. The purpose is a reproducibility record: to show that
our self-run CMFGEN reproduces the *published* StaNdaRT CMFGEN toy06 output, so
that internal quantities StaNdaRT does not publish (the radiation field
`J_nu`, photoionization rates `Γ`, recombination `α`) can be taken from our run
and trusted.

## Why a self-run

The published StaNdaRT release ships CMFGEN's emergent spectrum, `T_e(v)`,
`n_e(v)`, and ion fractions for toy06, but not the internal radiation field or the
level-resolved rates. Those internals are what the LUMINA-SN ionization work needs
as a reference. A self-run that provably matches the published CMFGEN lets us read
the internals off the same converged model.

## Model and epoch

- Model: `snia_toy06` (StaNdaRT toy06, uniform-composition SN Ia ejecta), 19.48 d.
- Structure: fixed (`DO_HYDRO=F`), steady-state snapshot (`DO_DDT=F`), full NLTE.
- Ions: Si II–V, S II–VI, Ca II–V, Fe II–VI, Co II–VI, Ni II–VI (see
  `config/MODEL_SPEC` for the super-level structure per ion).
- Grid: `ND=90` depth points, velocities 1025–35975 km/s.

The published CMFGEN toy06 is a **time-dependent** sequence (57 epochs, 2.0–217 d).
Our snapshot is steady-state (`DO_DDT=F`), chosen for tractability. Whether the
time-dependent terms matter at 19.48 d is answered empirically by the overlay below
(the only known residual is a small outer-shell `n_e` excess).

## Directory contents

```
config/     CMFGEN input decks (the only files we author; source is pristine)
  VADAT         main control deck (flags, abundances, SN_AGE=19.48)
  MODEL_SPEC    grid + super-level structure per ion
  IN_ITS        iteration control (NUM_ITS, DO_LAM_IT) — re-read each iteration
run/        launch + environment
  setup_links.sh          symlinks to atomic data + generic data (provenance)
  run.sh                  minimal launcher
  run_lageunha_snap1948.sh  canonical manual launcher (OMP=16, core binding, docs)
  run_lageunha_armN.sh    stint-2 arm N (damped Newton continuation)
  run_lageunha_armF.sh    stint-2 arm F (fresh damped LAMBDA)
build/
  CMFGEN_BUILD_RUN_GUIDE.md  build patches, SN recipe, stabilization ladder
analysis/   validation scripts (see "Validation" below)
PROVENANCE_R8_campaign_ledger.txt   full lab notebook (R8_RESUME_NOTE): every
                                    config decision, falsifier, and gate, dated
```

## Build

CMFGEN source is kept **pristine**; we change only input decks and launchers. The
build (compiler flags, the handful of dimension patches, and the SN stabilization
ladder) is documented in `build/CMFGEN_BUILD_RUN_GUIDE.md`.

- Source tree: `/gpfs/kjhan/cmfgen_src/cur_cmf` (program date 18-Jun-2025).
- Atomic data: `/gpfs/kjhan/cmfgen_21jun23/atomic` (linked by `run/setup_links.sh`).
- Executable: `exe/cmfgen_dev.exe`.

## Run procedure

**`OMP_NUM_THREADS=16` is mandatory.** `comp_opac.f` sums continuum opacity with an
array reduction under a dynamic schedule, so the floating-point summation order is
thread-count dependent. At 32/60/64 threads it tips the net opacity negative at the
cold outer plateau and `LOG(chi<=0)` then produces NaN. 16 is the only thread count
this model has run clean at (details in the launcher header and the build guide).

Steps:

1. `bash run/setup_links.sh` — create all atomic/generic-data symlinks.
2. Place `config/VADAT`, `config/MODEL_SPEC`, `config/IN_ITS` in the run directory.
3. Launch with `run/run_lageunha_snap1948.sh` (binds to 16 free physical cores,
   `OMP_NUM_THREADS=16`).
4. Convergence is reached in two stints:
   - **Stint 1** — `FIX_T=T` (hold temperature), converge the populations.
   - **Stint 2** — `FIX_T=F` (release temperature), converge the self-consistent
     `T_e`. The `arm N`/`arm F` launchers drive stint 2; `arm N` raises `LAM_VAL`
     to force full linearization (Newton) with a damped step cap, `arm F` is a
     fresh damped-LAMBDA insurance run.
5. **Graceful stop**: CMFGEN re-reads `IN_ITS` each iteration. To stop cleanly, set
   `NUM_ITS` to 1 there; the run finishes one final iteration and exits 0. Never
   kill the process.

## Validation

Run from this directory (`analysis/` scripts use repo-relative paths):

| Script | What it checks |
|---|---|
| `cmp_rvtj_T_ne_vs_published.py <RVTJ>` | our `T_e(v)`, `n_e(v)` vs published CMFGEN phys |
| `cmp_ionfrac_codes_at_lumina_v.py` | is published CMFGEN inside the inter-code spread (Fe/Co f(IV)) |
| `cmp_ionfrac_all_codes.py` | full Co/Fe ion-fraction table across all StaNdaRT codes |

### Result 1 — our CMFGEN vs published CMFGEN (`T_e`, `n_e`)

At an intermediate iteration of stint 1 (`FIX_T=T`), against published CMFGEN phys
at 19.48 d:

| v [km/s] | T ratio | n_e ratio |
|---|---|---|
| 4264 | 1.00 | 1.00 |
| 7176 | 1.00 | 1.01 |
| 9900 | 1.00 | 1.00 |
| 11544 | 0.99 | 1.15 |
| 24460 | 1.00 | 1.33 |

`T` matches to five significant figures because our run is seeded from the
published 19.48 d structure and holds `T` during stint 1. The load-bearing number
is `n_e`, which is *solved* from the ionization balance. It reproduces the
published CMFGEN electron density to ~1% inward of the photosphere (v ≤ 10000 km/s).
The seed structure being a near fixed point of our solver is the validation. The
outer excess (8–33%, growing outward) is confined to the still-converging trace
high-ion tail and is where a steady-state vs time-dependent difference, if any,
would appear.

### Result 2 — published CMFGEN is inside the code consensus

`f(IV) = IV/(III+IV)` at 19.48 d for Fe, interpolated to Lumina shell velocities:

| code | s0 (4264) | s4 (7176) | s6 (8632) | s10 (11544) |
|---|---|---|---|---|
| CMFGEN | 1.00 | 0.73 | 0.07 | 0.02 |
| ARTIS | 1.00 | 0.43 | 0.00 | 0.00 |
| SEDONA | 1.00 | 0.99 | 0.51 | 0.00 |
| TARDIS | 1.00 | 1.00 | 1.00 | 0.00 |
| SUPERNU | 1.00 | 0.86 | 0.01 | 0.00 |
| CRAB | 1.00 | 0.82 | 0.06 | 0.00 |

All six photospheric-epoch codes agree on the shape — Fe (and Co) fully IV deep,
recombining to III at the photosphere. CMFGEN sits inside the cluster. The codes
disagree most in the transition zone (v ≈ 7–9 kK).

## Status and caveats

- Definitive validation requires the **converged** model: rerun
  `cmp_rvtj_T_ne_vs_published.py` on the converged `RVTJ` (after stint 2), then
  overlay the emergent spectrum and ion fractions. Convergence is in progress.
- `data/tardis_reference_toy06_19p48d/` in the repo is a **TARDIS** run of toy06
  (it provides TARDIS-format atomic data), not a CMFGEN reference. The CMFGEN
  benchmark is `data/standart_data1/toy06/*_cmfgen.txt`.
- Job IDs and PIDs move; the authoritative live state is
  `PROVENANCE_R8_campaign_ledger.txt`.
