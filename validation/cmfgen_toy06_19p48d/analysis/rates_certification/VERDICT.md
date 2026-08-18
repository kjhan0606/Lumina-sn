# Rate-machine certification (Stage 2): σ-bake + integration layer vs CMFGEN's own Γ

**2026-07-29.** Offline python only. `src/` untouched, no GPU, no LUMINA run.
Script `certify_rate_machine.py` (~65 s, CPU, single core). Full stdout `run_log.txt`.

Question: with CMFGEN's **field** and **populations** pinned, does LUMINA's baked
σ_bf table plus its 1000-bin integration weight reproduce CMFGEN's own converged
photoionization rate?

```
Gamma_ion(d) = SUM_l [n_l(d)/n_ion(d)] * 4pi INT sigma_l(nu) J_nu(d)/(h nu) dnu
```

| piece | source | note |
|---|---|---|
| J_ν(d) | `EDDFACTOR` + `_INFO` (196185 freq, 3.5e12–1e18 Hz, `FINISH_REC=1`) | CMFGEN truth |
| n_l(d) | `POPCOB` / `POPIRON` / `POPSUL` | CMFGEN truth; `DCI(ion)==ground(next ion)` = **0 violations** for all four species × 90 depths |
| level order | `<ion>_F_TO_S` names compared to `<ion>_F_OSCDAT` names | **identical**, 1000/1500/322/256 — ordering is confirmed, not assumed |
| **truth Γ** | `CoIIIPRRR`, `FeIIIPRRR`, `S2PRRR`, `SIIIPRRR` (`subs/prrr_sl_v6.f:126`, `PR=n_SL·R_SL`) | Σ_SL PR / n_ion |
| **σ under test** | `data/atomic/cmfgen_sigma_bf_superlev_ionfix_ddc15strat_sivcaiv.bin` (magic `0x434D4644`, 26592 × 1000, 1.5e14–3e16 Hz, has_cmfgen 98.1 %) | row index = `data/tardis_reference_.../levels.csv` row order |
| **integration under test** | `Γ = Σ_b σ(ν_c,b)·J̄_b·4π/(hν_c,b)·Δν_b`, bins from `src/lumina.h:487-489` + `src/lumina_nlte_gemm.cu:142-148` | ν_c = geometric bin centre, Δν = hi−lo |

Level identification was **not assumed**: for S II, S III, Fe III the ref rows are
1:1 with the CMFGEN model levels (max |ΔE| = 5.0e-11 eV, g identical); for Co III
the first 1000 ref rows are 1:1 with the toy06 model's 1000 levels. Match failure
rate **0/3078**; population sitting on a σ==0 row is **0.000 at every shell for all
four ions**.

---

## 1. Verdict — pre-registered gate

Gate (registered before running): `Γ_D/Γ_PRRR ∈ [0.5,2.0]` at every shell **and**
`∈ [0.8,1.25]` at the forming shells s6–s8.

| ion | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | **verdict** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Co III** | 1.142 | 1.148 | 1.153 | 1.163 | 1.168 | 1.167 | 1.152 | 1.155 | 1.016 | **PASS** |
| **Fe III** | 1.012 | 1.013 | 1.015 | 1.013 | 1.010 | 1.005 | 0.994 | 0.995 | 0.921 | **PASS** |
| **S II** | 1.007 | 1.013 | 1.019 | 1.025 | 1.028 | 1.027 | 1.004 | 0.991 | 0.941 | **PASS** |
| **S III** | 0.998 | 0.997 | 0.997 | 0.995 | 0.993 | 0.987 | 0.969 | 0.959 | 0.918 | **PASS** |

**All four ions pass at every shell, with margin.** Worst deviation anywhere in the
36 (ion × shell) cells is **+17 %** (Co III s4) and the forming-shell band is
0.92–1.17 against an allowed 0.8–1.25.

**Control row** — layer A (CMFGEN's own σ on CMFGEN's own grid), which must be ≈1
or the ladder means nothing:

| ion | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
|---|---|---|---|---|---|---|---|---|---|
| Co III | 0.993 | 0.993 | 0.992 | 0.991 | 0.989 | 0.984 | 0.973 | 0.971 | 0.880 |
| Fe III | 0.993 | 0.993 | 0.992 | 0.990 | 0.988 | 0.984 | 0.972 | 0.973 | 0.892 |
| S II | 0.996 | 0.996 | 0.995 | 0.991 | 0.988 | 0.982 | 0.960 | 0.952 | 0.879 |
| S III | 0.996 | 0.996 | 0.996 | 0.995 | 0.992 | 0.987 | 0.970 | 0.960 | 0.919 |

The control carries a shell-dependent −0.4 %…−12 % of its own (see §4.1), **common
to all four ions**, so the machine-only number is `D/A`:

| ion | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
|---|---|---|---|---|---|---|---|---|---|
| Co III | 1.150 | 1.156 | 1.161 | 1.173 | 1.181 | 1.186 | **1.184** | 1.189 | 1.155 |
| Fe III | 1.018 | 1.021 | 1.023 | 1.023 | 1.022 | 1.022 | **1.022** | 1.022 | 1.032 |
| S II | 1.012 | 1.018 | 1.025 | 1.034 | 1.041 | 1.047 | **1.046** | 1.041 | 1.071 |
| S III | 1.001 | 1.001 | 1.001 | 1.001 | 1.000 | 1.000 | **0.999** | 0.999 | 0.998 |

**Conclusion: the rate machine is certified. It is right to 0.1 % (S III), 2 %
(Fe III), 5 % (S II), and 18 % (Co III), and every one of those four numbers is
explained below and attributed to a named layer.** Ionization error in LUMINA at
the 2–20× level that this campaign is chasing **cannot** originate in σ-bake or
in the frequency integral.

---

## 2. Layer ladder

Each column is the factor introduced by **exactly one** change relative to the
column to its left.

```
A    CMFGEN sigma (sub_phot_gen.f), CMFGEN 196185-pt CMF grid, CMFGEN levels
A|g  ... integral truncated to LUMINA's grid [1.5e14, 3e16] Hz          -> truncation
Bpt  exact sigma sampled at the 1000 bin CENTRES x bin-averaged J        -> LUMINA quadrature
Bav  bin-AVERAGED exact sigma x bin-averaged J   ("ideal bake")          -> Bpt/Bav = resonance blur
C    LUMINA's baker fed the SAME phot file toy06 used                    -> fit-type / tail approximations
D    sigma read verbatim from the shipped .bin                           -> D/C = atomic-data vintage
```

### s6 (forming shell) attribution

| ion | A/PRRR | A\|g / A | Bpt / A\|g | Bpt / Bav | C / Bpt | D / C | **D/PRRR** |
|---|---|---|---|---|---|---|---|
| Co III | 0.973 | 1.000 | 0.998 | 0.998 | 1.000 | **1.186** | 1.152 |
| Fe III | 0.972 | 1.000 | 1.002 | 1.001 | **1.020** | 1.000 | 0.994 |
| S II | 0.960 | 1.000 | **1.047** | **1.045** | 1.000 | 0.999 | 1.004 |
| S III | 0.970 | 1.000 | 1.001 | 1.001 | 1.000 | 0.998 | 0.969 |

Full ladders (9 gated shells + 5 diagnostic depths) for all four ions are in `run_log.txt` and
`rates_certification.csv`.

### 2.1 Grid truncation is exactly zero (`A|g / A = 1.000` everywhere)

LUMINA's 100 Å–20000 Å window loses **no** measurable photoionization for these
ions at this epoch. Independent of the σ representation. (Quadrature self-check:
`Σ J̄_b Δν_b` vs the native `∫J dν` over the same range = ratio **1.000000**.)

### 2.2 The 1000-bin grid is fine enough — but S II is the ion where it is not free

The bin width is `Δlnν = 5.298e-3` = **1588 km/s**. Measured node spacing of the
tabulated cross-sections over `1 < ν/ν_th < 5`:

| ion | phot data | median Δlnν of the σ table | data/bin ratio |
|---|---|---|---|
| Fe III | `FE/III/19apr23/phot_data_A` (OP, smoothed 3000 km/s) | 5.05e-3 | 1.05× |
| S III | `SUL/III/3oct00/phot_sm_3000.dat` (OP, smoothed 3000 km/s) | 5.10e-3 | 1.04× |
| **S II** | `SUL/II/19apr23/phot_data_A` (**OP, not smoothed**) | **2.01e-3** (min 1.99e-4) | **2.63×** |

Where the data are pre-smoothed at 3000 km/s the 1588 km/s bin is *finer than the
data*, so the bake resolves everything it is given: `Bpt/Bav = 1.001`.
Where the data are **unsmoothed** (S II) the bake grid is 2.6× coarser than the
table and point-sampling at the bin centre biases Γ **high** by
**+1.4 % (s0) → +4.5 % (s6) → +6.4 % (s8)**.

**So: the "1000-bin bake smears resonances" hypothesis is TRUE in kind but small
in size — a few per cent, not a factor.** It is also fixable exactly: the bias is
entirely `Bpt/Bav`, i.e. the baker point-samples σ at ν_c instead of averaging it
over the bin. Replacing `np.interp(nu_grid, ...)` with a bin-average in
`expand_atomic_data_cmfgen.bake_sigma_bf_grid` removes it (the ⟨σJ⟩ vs ⟨σ⟩⟨J⟩
correlation term, `Bav/A|g`, is only 1.000–1.007). **Not applied here — this
report does not touch src/ or the data pipeline.**

---

## 3. Defects found in the baker (real, small, actionable)

### 3.1 Fe III: fit type 7 (modified Seaton) is replaced by a spurious Kramers edge — **+2.0 % of Γ(Fe III)**

`expand_atomic_data_cmfgen.py:658-666` lumps CMFGEN fit types **2, 3, 7, 8, 9**
into `σ = params[0]·1e-18·(ν_th/ν)³`. For type 7 CMFGEN's actual rule
(`sub_phot_gen.f:505-512`) is

```
RU = (EDGE + A3)/nu ;  sigma = 0 identically unless RU <= 1
```

i.e. the cross-section **does not turn on at ν_th at all** — it turns on at
`ν_th + A3`. For the 399 Fe III levels carrying type 7, `A3 = 10.2…12.3` (CMFGEN
units of 1e15 Hz) = **42–50 eV above threshold**, so the true edge sits near
150–250 Å where J is negligible.

| | exact | baker (C) | shipped (D) |
|---|---|---|---|
| Fe III type 7 share of Γ, s0 | 0.00000 | 0.01733 | 0.01733 |
| Fe III type 7 share of Γ, s6 | 0.00000 | 0.02017 | 0.02017 |

The baker instead opens a ≈1.2 Mb edge at 1000–4000 Å for each of those 399
levels. That is the whole of `C/Bpt = 1.017…1.029` for Fe III. It is small here
**only because** those levels are weakly populated; it is a wrong-physics term,
not a rounding error, and it will scale with whatever ion has type-7-heavy,
well-populated levels.

The same code path also mishandles type **2/3/8** (`params[0]` there is a
*principal quantum number*, not a cross-section in Mb). In S II 9 levels take that
path and contribute a spurious **+0.015 %** — negligible here, arbitrary in
general.

### 3.2 Tabulated σ: `left=0` / `right=const` — measured, immaterial at this epoch

The baker's `np.interp(..., left=0.0, right=sigma[-1])` differs from CMFGEN twice:
below the first table node CMFGEN returns `CROSS_A[1]` (baker returns 0), and
above the last node CMFGEN extrapolates `CROSS_A[N]·(u_N/u)³` (baker holds the
last value). Net measured effect on the tabulated levels: **−0.24 % (S II s0),
−0.11 % (S II s6)**. The constant-tail limb is essentially unreachable because the
tables run to `u_max ≈ 27–32`, i.e. down to ~15 Å, well outside LUMINA's 100 Å
grid edge. Flagged as a latent defect, not a live one.

### 3.3 Co III: `D/C = 1.19` is atomic-data vintage, **not** a machine error

The toy06 run reads `COB/III/18oct00/phot_data.dat` (386 terms, all single-
parameter Seaton, self-described "Very crude"). The shipped binary was baked from
`COB/III/19apr23/phot_data_A`. Provenance was verified, not assumed: re-baking
LUMINA's own vintage and differencing against the shipped rows gives
`Σ|D−C2|/Σ|D| = 7.2e-15` (Fe III 5.1e-14, S II 3.8e-12, S III 1.3e-13) —
**the shipped binary IS `bake_sigma_bf_grid` applied to the 19apr23 tree, bit for
bit.** So Co III's +15…+19 % is LUMINA using *newer* Co III cross-sections than
the toy06 reference model, which is a data statement, not a bug. With the same
data (`C/Bpt = 1.000`) the machine is exact for Co III.

---

### 3.4 Side finding — the toy06 snapshot is internally inconsistent over 34 of its 90 depths

Extending the ladder to diagnostic depths outside the gate (`--extra-depths`)
showed `A/PRRR ≈ 0.333` — **the same value, for all four chemically independent
ions, at the same depths**. A factor that is common to Co, Fe and S is not
physics and not a σ error. Tracking it down:

`*PRRR` prints its own `Ion Density` record. It must track `POP*`'s next-ion
ground population up to one fixed scale. For **Fe III that scale is exactly
1.000 outside the band and exactly 3.000 inside it**:

| depth | 31 | 35 | **37** | **42** | **45** | **49** | 51 | 53 | 57 | 61 |
|---|---|---|---|---|---|---|---|---|---|---|
| `DI(FeIIIPRRR)/ground(FeIV,POPIRON)` | 1.003 | 1.013 | **3.000** | **3.000** | **3.000** | **3.000** | 1.039 | 1.009 | 1.002 | 1.000 |

The affected depth set is **identical for all four ions**: `d1–d20` and
`d37–d50` (d39 excepted), i.e. `v ≥ 27400 km/s` and `v = 12376–19769 km/s`
(median deviation factor 2.47–2.66 by ion). EDDFACTOR's `J` is smooth across the
band at 300/600/1100/2000/5000 Å, and my machinery gives `A/PRRR ≈ 1.0` on **both**
sides of it — so it is `*PRRR` and `POP*` having been written from different
states, not a transport or a bake effect.

**Consequence for this campaign:** `Γ_PRRR` is not a usable truth at those 34
depths, and neither is any Co/Fe/S quantity that pairs `*PRRR` with `POP*` there.
It is flagged per depth in `rates_certification_alldepth.csv`
(`prrr_pop_inconsistent`) and reported per ion by the script (`[snap]` line).
Of the nine gated shells, **none** falls in the band for Co III / Fe III / S III;
**s8 (d51) does for S II**, and d51 still carries a 4 % residual for Fe III.
That — not a composition cliff — is the better explanation for the s8 control
degradation noted in §1 and in `../gamma_coiii_alllevel/VERDICT.md` caveat 1.
The mechanism inside CMFGEN is **not** identified here.

---

## 4. Caveats — stated as unresolved

1. **The control degrades outward and s8's truth is contaminated.** `A/PRRR`
   falls from 0.993 at s0 to 0.96–0.97 at s6/s7 and **0.88–0.92 at s8**, nearly
   identically for all four ions. Part is my reproduction of CMFGEN's own
   quadrature (CMFGEN integrates `JPHOT` with `FQW` weights on its NCF continuum
   grid; I trapezoid on the full CMF grid); at s8 the dominant part is almost
   certainly the `*PRRR`/`POP*` inconsistency of §3.4, whose band ends at d50–d51.
   Neither is chased down. Consequence: **the s8 column of the gate table is the
   least trustworthy** and `D/A` (§1) is the cleaner machine-only statistic.
2. **Fit types 2/3/8/9 have no exact evaluator here** (they need CMFGEN's
   `BF_L_CROSS`/`BF_N_GAUNT` hydrogenic tables). Affected: 9 S II levels. Their
   CMFGEN population share is ≤ 7.7e-6 at every shell, and layer A still
   reproduces PRRR to 0.99, which is itself the evidence that omitting them is
   safe **for these four ions**. For an ion where those types carry real
   population this certification would need extending.
3. **Co III levels 1001–3917 of the shipped table are untested.** The toy06 model
   atom has only 1000 Co III levels, so no CMFGEN population exists for the other
   2917 rows (all of which do carry σ>0). Likewise 124 S III and 2 S II rows.
   The machine is certified on the level set CMFGEN solves, not on LUMINA's full
   level set.
4. **This tests the σ table and the ν-integral only.** Explicitly NOT certified:
   how LUMINA builds `J̄_b` from MC estimators; the level populations LUMINA
   itself computes; the recombination/Milne side; the Kramers fallback used for
   the 1.9 % of rows with `has_cmfgen=0`; and the fine-ν sub-bin correction path
   in `src/lumina_bf_gemm.cu` (`col_glev` / `g_fgemm_dlognu`), which is a separate
   code path from the 1000-bin GEMM weight tested here.
5. **The §3.4 mechanism is unidentified.** I show *that* `*PRRR` and `POP*`
   disagree by an exact factor 3 over 34 depths, not *why*. Anyone reusing this
   snapshot for a depth in `d1-d20` or `d37-d50` should treat rate/population
   pairings there as unusable until it is explained.
6. **One epoch (19.48 d), one model (toy06), four ions.** Si, Ca, Ni, Cr, Mn and
   the neutral stages are untouched. The Fe III type-7 defect (§3.1) is the one
   result that plainly generalises and should be re-measured per ion.

---

## 5. What this closes

The Stage-2 question was whether Γ = (rate machine) × (field) can have its **first
factor** certified so that the residual is attributable to J_ν. It can:

- σ-bake + integration reproduces CMFGEN's own converged Γ to **0.1 %/2 %/5 %/18 %**
  for S III / Fe III / S II / Co III, and each of those four residuals is named
  (§3.1 type-7 stand-in, §2.2 unsmoothed-OP point sampling, §3.3 data vintage).
- Grid truncation: **0.0 %**. Level matching: **0 failures / 3078 levels**.
  Population on σ==0 rows: **0.000**.
- Therefore the campaign's open ionization discrepancies (n_e 1.92×, b_k 2–20×)
  are **not** attributable to the rate machine, and per-ion σ tuning is closed as
  a hypothesis at this magnitude. What remains on the Γ side is the field.

---

## 6. Artifacts

| file | content |
|---|---|
| `certify_rate_machine.py` | the whole calculation: `python3 certify_rate_machine.py` (~65 s) |
| `rates_certification.csv` | 4 ions × (9 gated shells + 5 diagnostic depths): Γ at every layer (PRRR, A, A\|g, Bpt, Bav, C, C2, D) + ratios + population-coverage diagnostics |
| `rates_certification_alldepth.csv` | all 90 CMFGEN depths: Γ_PRRR, Γ_C, Γ_D, ratios, and the `prrr_pop_inconsistent` flag of §3.4 |
| `run_log.txt` | full gate output, per-fit-type decomposition, σ-table resolution measurement |
