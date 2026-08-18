# Stage-IV atomic-data inventory + importer design (toy06 @ 19.48d)

Offline analysis. No commits, no source edits, no runs. Read-only.

CMFGEN atomic tree (authoritative, toy06-linked): `/gpfs/kjhan/cmfgen_21jun23/atomic`
Local mirror used by the importer: `data/atomic/cmfgen` (same content, `_pick_latest` picks 19apr23 dirs).
Lumina dataset under audit: `data/tardis_reference_toy06_19p48d` (levels/lines/ioniz are symlinks to
`data/tardis_reference_cmfgen_superlev_ionfix_ddc15strat`).

---

## Headline corrections to the task framing (must read first)

1. **"Sc IV (CMFGEN species 'Sk')" is a mislabel.** In CMFGEN, `Sk` = **Silicon**
   (`Sk2_*`→`SIL/II`, `SkIV_*`→`SIL/IV`). toy06's `SkIV_ISF`/`SkV_ISF` are **Si IV / Si V**,
   not Scandium. **Si IV is already present in Lumina** (`levels.csv` has `14,3` = 66 levels),
   so it is *not* a missing ion. True **Scandium IV does not exist in the CMFGEN tree**
   (`SCAN/` has only I, II, III).

2. **toy06 solves ONLY Si, S, Ca, Fe, Co, Ni ions.** `MODEL_SPEC` has ISF blocks for exactly
   these six elements. There is **no C, O, Mg, Mn, V, Sc** ion in the model at all. Confirmed by
   composition: `abundances.csv` is non-zero only for Z = 28,27,26,20,16,14 (Ni,Co,Fe,Ca,S,Si);
   the O (Z=8) and C (Z=6) rows are **all zeros**, and Mg/Sc/V/Mn are not in the file.

3. **Therefore C IV, O IV, Mg IV, Mn IV, V IV, Sc IV have exactly zero impact on toy06**
   (zero element abundance). V IV and Sc IV are additionally **absent from the CMFGEN tree**.
   The only genuinely relevant missing stage-IV ions are **S IV, S V, Ca IV, Ca V**.

4. **The "deep S IV = 0.951" fact is real but misleading.** That 95% ionization is at v≈4183 km/s,
   which is Lumina **shell 0 (the IGE core) where X_S = 0**. Where sulfur actually lives (outer,
   X_S=0.35) it is **S II / S III**, not S IV. Mass-weighted, only **0.8 % of all S** and **3.7 %
   of all Ca** is in stage IV at 19.48d (Task 2). This downgrades the urgency substantially.

---

## Task 1 — Inventory

Per-ion CMFGEN files (importer/19apr23 resolution). Full machine table: `inventory_ions.csv`.

| ion | Z,ion₀ | in toy06? | toy06 solves | in Lumina? | levels | lines | σ_bf | f_to_s | E_ion (eV) | files (toy06-linked) |
|-----|--------|-----------|--------------|------------|-------:|------:|:----:|:------:|-----------:|----------------------|
| **S IV** | 16,3 | yes | **yes** `[SIV_ISF] 65sl/176fl` | **NO** | 194 | 3765 | Y | Y | 47.222 | `SUL/IV/3oct00/{sivosc_fin.dat,f_to_s_69.dat,phot_sm_3000.dat,col_siv.dat}` |
| **S V** | 16,4 | yes | **yes** `[SV_ISF] 39sl/163fl` | **NO** | 307¹ | 9968¹ | Y | Y | 72.594 | `SUL/V/3oct00/{svosc_fin.dat,f_to_s_50.dat,phot_sm_3000.dat,col_sv.dat}` |
| **Ca IV** | 20,3 | yes | **yes** `[CaIV_ISF] 41sl/375fl` | **NO** | 378 | 8533 | Y | Y | 67.100 | `CA/IV/10apr99/{osc_op_sp.dat,f_to_s.dat,phot_smooth.dat,col_guess.dat}` |
| **Ca V** | 20,4 | yes | **yes** `[CaV_ISF] 70sl/528fl` | **NO** | 613 | 18272 | Y | Y | 84.408 | `CA/V/10apr99/{osc_op_sp.dat,f_to_s.dat,phot_smooth.dat,col_guess.dat}` |
| Si IV | 14,3 | yes | yes `[SkIV_ISF] 50sl/61fl` | **YES (66 lev)** | 61 | — | Y | Y | 45.142 | `SIL/IV/5dec96/*` — already imported |
| C IV | 6,3 | **no** | no | NO | (in tree) | — | Y | Y | — | `CARB/IV/19apr23/*` — X_C=0 in toy06 |
| Mg IV | 12,3 | **no** | no | NO | (in tree) | — | Y | Y | — | `MG/IV/19apr23/*` — Mg absent from toy06 |
| O IV | 8,3 | **no** | no | NO | (in tree) | — | Y | Y | — | `OXY/IV/19apr23/*` — X_O=0 in toy06 |
| Mn IV | 25,3 | **no** | no | NO | (in tree) | — | Y | Y | — | `MAN/IV/19apr23/*` — Mn absent from toy06 |
| Sc IV | 21,3 | no | no | NO | **absent** | — | — | — | — | `SCAN/IV` **does not exist** |
| V IV | 23,3 | no | no | NO | **absent** | — | — | — | — | `VAN/` has only neutral V I |

¹ S V: 19apr23 has 307 lev / 9968 lines; the **toy06-linked 3oct00** file has **216 lev / 3462 lines**.
Pin to 3oct00 for an exact toy06 level-count match; otherwise 19apr23 is a richer superset. S IV, Ca IV,
Ca V have **identical** level counts in 19apr23 and the toy06-linked files (194 / 378 / 613).

`MODEL_SPEC` ISF triplet = `N_super, N_super, N_full`. toy06 truncates the full files slightly
(S IV 176/194, S V 163/216, Ca IV 375/378, Ca V 528/613) and collapses to super-levels for the SE solve.

Lumina already carries stage-IV for Al(13,3), Si(14,3), Ti(22,3), Cr(24,3), Fe(26,3)+V(26,4),
Co(27,3), Ni(28,3). It **lacks** stage-IV for C(6), O(8), Mg(12), S(16), Ca(20), Sc(21), Mn(25),
and lacks V beyond neutral.

**σ_bf source per ion:** photoionization is one file per ion (`phot_*`), matched to levels by
term-config name (J-collapsed) inside `expand_atomic_data_cmfgen.py` → `cmfgen_sigma_bf.bin`.
S IV `phot_data_A` has 124 entries spanning cs_type {20:89 (OP tabulated), 1:11 (Seaton), 2:4, 3:20}.

---

## Task 2 — Cross-check the need (mass-weighted, not fraction-weighted)

Method: for each Lumina shell, weight the CMFGEN 19.48d ion fraction f(stage,v) by the shell's element
mass fraction X(Z,shell) and shell mass ρ·dV. Full per-shell table: `task2_per_shell_profile.txt`;
ranking: `task2_need_ranking.csv`.

Integrated over the 19.48d ejecta (v = 3900–39900 km/s, Lumina 50-shell grid):

| element | total mass (g) | in stage **III** | in stage **IV** | in stage **V** |
|---------|---------------:|-----------------:|----------------:|---------------:|
| **S**  (Z=16) | 2.79e32 | 43.1 % | **0.8 %** (2.28e30 g) | 0.01 % |
| **Ca** (Z=20) | 7.96e31 | 96.2 % | **3.7 %** (2.98e30 g) | 0.06 % |

Deep (v≈4183, shell 0) vs photosphere (v≈9901, shell ~8) fractions:

| ion | deep III/IV/V | photosphere III/IV/V | X at that shell |
|-----|---------------|----------------------|-----------------|
| S   | 0.003 / **0.951** / 0.040 | 0.841 / 0.007 / 1.3e-5 | deep X_S=0 · phot X_S=0.27 |
| Ca  | 0.856 / 0.161 / 6e-4 | 0.975 / 0.025 / 1.3e-4 | deep X_Ca=0 · phot X_Ca=0.08 |

**Ranking (by capacity to distort deep opacity / ionization budget):**
1. **Ca IV** — largest *absolute* stage-IV ion column (2.98e30 g); non-trivial across the transition
   and outer zones; still a minority behind Ca III (96 %).
2. **S IV** — 2.28e30 g absolute (0.8 % of S). The dominant-stage headline is an artifact of the
   S-free IGE core. Its residual weight is worth importing mainly because its **resonance lines
   (657/748/810/1063 Å) sit in the FUV window** the current campaign is fighting over — even a small
   S IV population adds FUV line opacity where the model is FUV-starved.
3. **S V, Ca V** — negligible (<0.1 % of element mass). Import only as recombination anchors for IV.
4. **C IV, O IV, Mg IV, Mn IV, Sc IV, V IV** — **zero** (zero abundance in toy06; two absent from tree).
   Do not import for the toy06 benchmark.

Bottom line: the stage-IV gap is a **second-order** correction for toy06, dominated by Ca IV≈S IV in
absolute terms, with S IV's value concentrated in the FUV. It will not, by itself, move the ionization
budget of the S/Ca-bearing zones (those are II/III), but it plugs the FUV-opacity and top-of-ladder
recombination-anchor holes.

---

## Task 3 — Importer design

**Key finding: the importer already exists and is proven.** `scripts/expand_atomic_data_cmfgen.py`
(+ `scripts/cmfgen_parser.py`) is exactly this converter and it built the deployed dataset. Adding the
four ions is a **configuration change**, not new code:

```
# scripts/expand_atomic_data_cmfgen.py  ION_LEVEL_CAPS  (stage is 1-based; 4 = 'IV')
(16, 4): None,   # S IV   (194 lev < 200 -> full, matches toy06 count)
(16, 5): None,   # S V    (pin 3oct00 for 216-lev toy06 match; see below)
(20, 4): None,   # Ca IV  (378 lev; None matches toy06's 375; or cap 200)
(20, 5): 528,    # Ca V   (613 lev; toy06 uses 528 -> cap 528, or None for full)
```

Then re-run: `CMFGEN_SUPER_LEVELS=1 CMFGEN_OUT_SUFFIX=... python3 scripts/expand_atomic_data_cmfgen.py`,
followed by the `finalize_cmfgen_ref_npy.py` step and the ddc15strat abundance/density/ionfix overlay
that produced the current deployed variant. **All** downstream tables regenerate together.

### File formats — both sides (with line examples)

**CMFGEN `F_OSCDAT` (e.g. `SUL/IV/3oct00/sivosc_fin.dat`)**
Header tags: `194 !Number of energy levels`, `380870.0000 !Ionization energy` (cm⁻¹),
`4.0 !Screened nuclear charge`, `3597 !Number of transitions`.
Level row (6 cols): `config  g  E[cm⁻¹]  ν_ion[10¹⁵Hz]  λ_ion[Å]  ID`
```
3s2_3p_2Po[1/2]     2.0    0.00000000   11.418195   2.62557E+02   1
```
Transition row: `cfg_lo -cfg_up   f_lu   A_ul[s⁻¹]   λ[Å]   i-j   trans#`, with the banner
`Wavelengths in air for lambda > 2000 Ang, else vacuum`:
```
3s2_3p_2Po[1/2] -3s2_3d_2De[3/2]   9.0359E-01  6.9748E+09   657.319   1-  11   7
```

**CMFGEN `PHOT_*` (`phot_sm_3000.dat`)**: per-level blocks `config / Type / Npts / (E_ratio, σ_Mb) pairs`.
E in units of threshold ν; σ in **Megabarns**; type 20 = OP tabulated, 1 = Seaton fit, 2/3 = hydrogenic.

**CMFGEN `F_TO_S` (`f_to_s_69.dat`)**: adds super-level column (header `6 !Entry number of link to super
level`) and trailing full-level ID; maps 176→65 super-levels for S IV.

**Lumina TARDIS side** — `levels.csv`:
`atomic_number,ion_number,level_number,energy_eV,g,metastable,super_level`
```
16,3,0,0.0000000000,2,1,0
```
`line_list.csv`:
`atomic_number,ion_number,level_number_lower,level_number_upper,line_id,wavelength,f_ul,f_lu,nu,B_lu,B_ul,A_ul,wavelength_cm`
`ionization_energies.csv`: `atomic_number,ion_number,ionization_energy_eV` — append `16,3,47.2218…`,
`16,4,72.5945…`, `20,3,67.1002…`, `20,4,84.4084…` (currently ends at `16,2` / `20,2`).
`cmfgen_sigma_bf.bin`: header `magic 'CMFD' / version 1 / int32 n_levels,n_freq / double ν_min,ν_max /
int8 has_cmfgen[n_levels] (8-byte pad) / double σ_cm2[n_levels·1000]`, level-major on a 1000-bin log-ν
grid (1.5e14–3.0e16 Hz). **`n_levels` in the header must equal `atom->n_levels`** or the C loader rejects it.

### Unit conversions (constants from the script; worked examples)

| quantity | CMFGEN | TARDIS | formula | worked (S IV g.s.→3d ²D₃/₂) |
|----------|--------|--------|---------|------------------------------|
| level E | cm⁻¹ (above g.s.) | eV | `E_eV = E_cm × 1.239841984e-4` | 152133.2 cm⁻¹ → 18.862 eV |
| ion E | cm⁻¹ | eV | same const | 380870 → **47.2219 eV** |
| wavelength | Å (air>2000, vac<2000) | cm | `λ_cm = λ_Å × 1e-8`; carried verbatim | 657.319 Å → 6.57319e-6 cm |
| ν | — | Hz | `ν = c/(λ_Å·1e-8)`, c=2.99792458e10 | 657.319 Å → 4.561e15 Hz |
| f | f_lu | f_lu, f_ul | `f_ul = f_lu·g_lo/g_up` | 0.9036·2/4 = 0.4518 |
| A | A_ul [s⁻¹] | A_ul | verbatim | 6.9748e9 s⁻¹ |
| B | — | B_lu,B_ul | `B_lu = A_ul·c²/(8πhν³)·g_up/g_lo`, `B_ul = B_lu·g_lo/g_up` | h=6.62607015e-27 |
| σ_bf | Megabarns | cm² | `σ_cm2 = σ_Mb × 1e-18` | 0.388 Mb → 3.88e-19 cm² |
| bf threshold | — | Hz | `ν_th = (E_ion−E_lev)[eV]·1.602176634e-12/h` | ground: 47.22 eV → 1.142e16 Hz |

### Level-indexing pitfalls
- **Global reindex, not append.** `build_global_levels` sorts by `(Z,stage)`, so inserting S IV (16,4)
  shifts the global index of **every level from Ca upward** (Z≥20). This invalidates every prebaked
  array keyed on global index (`tau_sobolev.npy`, `transition_probabilities.npy`, `line2macro*.npy`,
  `macro_atom_*.csv`, `cmfgen_sigma_bf.bin`). **The whole dataset must be rebuilt in one pass** — the
  pipeline already does this; just never hand-append rows to a live dataset.
- **Version drift via `_pick_latest`.** It selects 19apr23 (newest), which matches toy06 counts for
  S IV/Ca IV/Ca V but **not S V** (307 vs 216). Add a per-ion date/file override to pin `SUL/V/3oct00`
  for an exact toy06 reproduction.
- **CMFGEN ID ≠ row order in old files.** The parser keys transitions on the printed `i-j` IDs, not row
  position; keep that (some 10apr99/3oct00 files skip the explicit trans# column).
- **σ_bf name matching is term-level.** `phot` configs are J-collapsed (`3s2_3p_2Po`); the importer maps
  them to all J-split levels via `_term_cfg`. Levels with no phot match get σ=0 → Kramers fallback.
- **super_level column.** With `CMFGEN_SUPER_LEVELS`, only the listed II-ions keep f_to_s grouping; for a
  new top ion each full level is its own super-level (identity) unless you add it to `SUPER_LEVEL_IONS`.

### Counts that land per ion (importer, lam≠0 filter)

| ion | full lev | full lines | lines if cap≤100 | lines if cap≤200 | phot entries |
|-----|---------:|-----------:|-----------------:|-----------------:|-------------:|
| S IV | 194 | 3765 | 1008 | 3765 | 124 |
| S V | 307 | 9968 | 973 | 4010 | 259 |
| Ca IV | 378 | 8533 | 768 | 2872 | 338 |
| Ca V | 613 | 18272 | 635 | 2482 | 596 |

Recommended caps to mirror toy06 while matching the existing top-ion pattern (Fe IV=Cr IV=Ni IV=200):
**S IV None (194, full), Ca IV None (378) or 200, S V pin-3oct00 (216) or 200, Ca V 528.** A "K=100"
super-cutoff would drop ~73 % of S IV lines and starve the FUV opacity it is being imported for — not
recommended; keep full levels for these low-N ions.

### Effort & risk
- **Effort: LOW** (~½ day). 4 config lines + optional date-pin patch + one pipeline re-run + rebuild of
  the ddc15strat overlay + a validation pass. No C changes (the C loader already reads variable
  `n_levels` and re-checks the sigma header).
- **Risks:** (R1) global reindex invalidates all prebaked npy/bin — must regenerate the full set atomically;
  (R2) S V version drift — pin 3oct00 or accept 307-level superset; (R3) incomplete σ_bf term-matching
  → some levels Kramers-fallback (acceptable, same as existing ions); (R4) growth of `n_levels` enlarges
  every per-shell rate matrix — S IV+S V+Ca IV+Ca V add ≈1.5k levels, trivial vs the current ~30k.

### Priority order
**S IV → Ca IV → (S V, Ca V as anchors) → stop.** Do not import C/O/Mg/Mn/Sc/V IV for toy06.

---

## Task 3-B — VALUE-RIGOR VALIDATION LAYER (driver addendum)

Fail-closed: any REJECT blocks the dataset from being written. The pass emits a machine report
(`stage4b_import_validation.json` + human summary) that the driver reviews before any run consumes the
data. **Never silently coerce** — flag or reject. Implement as a `validate_import()` gate between
`build_*` and `write_*` in the pipeline.

### Checklist (rule → check → action)

| # | rule (quantity, both sides) | concrete check | tol / bound | on violation |
|---|-----------------------------|----------------|-------------|--------------|
| **D1** | level E: cm⁻¹(above g.s.) → eV | recompute `E_eV=E_cm·1.239841984e-4`; assert CMFGEN col-3 is energy-above-ground (col-4 `ν_ion` must **decrease** with level index) | exact (f64) | REJECT ion |
| **D2** | ion E: cm⁻¹ → eV | `E_ion_eV` vs NIST (S IV 47.222, S V 72.594, Ca IV 67.27, Ca V 84.34) | ≤0.5 % | FLAG |
| **D3** | wavelength air/vac | establish existing dataset convention from a known line **before importing** (do not assume); CMFGEN banner = air>2000Å/vac<2000Å; assert new rows use the same convention | — | REJECT if mismatch |
| **D4** | A↔f↔gf round-trip | `A_pred = 6.6702e15·(g_lo/g_up)·f_lu/λ_Å²` vs file A_ul | ≤3 % (else independent A) → FLAG | FLAG list |
| **D5** | E_up−E_lo ↔ λ | `λ_pred=1e8/(E_cm_up−E_cm_lo)` vs file λ; **vacuum lines <2000Å tight, optical air ~0.03 % offset expected** | UV ≤0.05 %, optical ≤0.1 % | FLAG (bad level xref) |
| **D6** | B coefficients | `B_ul·g_up==B_lu·g_lo`; `B_lu=A_ul c²/(8πhν³)(g_up/g_lo)` reproduces to f64 | exact | REJECT row |
| **D7** | σ_bf unit + threshold | Mb→cm² (×1e-18); each level's edge `ν_th=(E_ion−E_lev)/h` lands in a grid bin; σ=0 below edge; assert edge within Rydberg-scale sanity (`0<E_th≤E_ion`) | edge bin ±1 | REJECT level's σ |
| **D8** | σ high-ν falloff | far above edge σ decreasing, ~ν⁻² … ν⁻³ envelope; no rising tail | monotone-ish | FLAG |
| **D9** | g = 2J+1 | CMFGEN g is float (2.0,4.0…); `round(g)` integer, `abs(g-round)<1e-6`, g≥1 | exact | REJECT level |
| **D10** | f_lu > 0 | strictly positive | >0 | drop line + FLAG |
| **D11** | A_ul range | `1e-3 ≤ A_ul ≤ 1e11` s⁻¹ | bound | FLAG outliers (keep) |
| **D12** | E monotonic / non-neg | per ion E_cm≥0 and ascending in ID order | — | FLAG (reorder) |
| **D13** | no duplicate levels | unique `(Z,ion,level_number)` and unique `(Z,ion,i,j)` lines | — | REJECT dup |
| **D14** | partition function | `U(T)=Σg_i e^{-E_i/kT}` at T=5000/10000/15000 K vs direct sum on written rows | ≤1e-9 | REJECT (dropped levels) |
| **D15** | sig-figs / precision | f64 end-to-end; record source decimal precision per column; flag values with < source sig-figs (e.g. `2.0` g is exact, but a `1.0E+02` A is 3 sig-fig) | — | record in sidecar |
| **D16** | provenance harvest | pull `[Reference]` header lines + date-dir + `!`-tagged accuracy notes into a per-ion sidecar (`data_quality.json`): source (OP/TOPbase/Mendoza83), vintage, known caveats | — | required field |
| **D17** | NIST spot-check (top-10) | for the 10 strongest lines/ion (by g·f) compare λ and A/f to NIST ASD. **Offline** → emit the exact line list as a driver-side manual step | driver ✓ | block-until-checked |

### Worked NIST spot-check list for S IV (D17 — driver action)
| λ_CMFGEN (Å) | transition | f_lu | A_ul (s⁻¹) | check vs NIST ASD |
|-------------:|-----------|-----:|-----------:|-------------------|
| 657.319 | 3s²3p ²P°₁/₂ – 3s²3d ²D₃/₂ | 0.9036 | 6.975e9 | λ, gf |
| 748.393 | 3s²3p ²P°₁/₂ – 3s3p² ²P₁/₂ | 0.3758 | 4.476e9 | λ, gf |
| 809.656 | 3s²3p ²P°₁/₂ – 3s3p² ²S₁/₂ | 0.1489 | 1.515e9 | λ, gf |
| 1062.664 | 3s²3p ²P°₁/₂ – 3s3p² ²D₃/₂ | 0.04074 | 1.203e8 | intercombination-adjacent; λ, gf |
| (also 1073, 1406, 1416 Å UV multiplet) | — | — | — | classic S IV λ1063 / λ1406 doublet |

### Acceptance protocol
1. Import → `validate_import()` runs D1–D16 over every new ion; counts **checked / passed / flagged /
   rejected** per rule.
2. Any REJECT ⇒ dataset **not written**; report lists offending (ion, level/line, rule).
3. D17 NIST list emitted for driver sign-off (offline).
4. Reproduce toy06 counts: **S IV = 194 levels** (19apr23 or 3oct00), Ca IV = 378, Ca V = 613;
   S V = 216 (3oct00) or 307 (19apr23) — must match the pinned source exactly.
5. Report + `data_quality.json` sidecar are artifacts the driver reviews **before** any run loads the data.

---

## Provenance of numbers in this doc
- ISF/level counts: `/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MODEL_SPEC`, `setup_links.sh`.
- Ion fractions: `data/standart_data1/toy06/ionfrac_{s,ca}_toy06_cmfgen.txt` `#TIME 19.480`.
- Composition/geometry: `data/tardis_reference_toy06_19p48d/{abundances,geometry,density}.csv`.
- Headers/line counts: parsed live from `data/atomic/cmfgen/**` via `scripts/cmfgen_parser.py`.
- Importer behavior: `scripts/expand_atomic_data_cmfgen.py`, loader `src/lumina_atomic.c:594,940`.
