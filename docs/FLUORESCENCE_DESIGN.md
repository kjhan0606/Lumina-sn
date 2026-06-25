# Fluorescence Axis Design — UV→optical-green redistribution (2026-06-25)

## Problem (established by converged CONTONLY falsifier, 169757/758, commit 06012d7)

The freq-resolved emergent's remaining color error is a **green deficit / NIR excess**:
grn/nir = (5000-7000Å)/(7000-12000Å) — **model FULL 0.41, CONTONLY 0.49, gold 0.58.**
Removing lines (CONTONLY) *reduces* NIR and *raises* grn/nir → the model's optical lines
use a **thermal** source `S_l=B(T_e)` that re-emits cold-shell line photons at local Planck
(NIR-peaked). Gold instead fluoresces: UV photons absorbed by Fe/Co/Ni II lines pump upper
levels that cascade down through *different optical lines*, redistributing UV/blue → green.
Our lines absorb blue (blanketing, present) but don't re-emit green (fluorescence, absent).

**Architectural fact (lumina_cmfgen.c:161-288, 1416):** green photons can reach the emergent
ONLY through optical Fe/Co/Ni II lines having **super-thermal `S_l > B(T_e)`** in the
green-carrier shells. `χ_abs·B` is the cold NIR continuum; `r·J` is scattered ambient.
So *fluorescence in this architecture ≡ optical-line `line_source_S` exceeding local Planck.*

## What already exists (code audit, both agents convergent)

- **Producer** `cmfgen_fine_jbar` (lumina_cmfgen.c:1484): line-resolved `J̄_l` per line/shell on a
  vdop-resolved mesh over `[LAMLO,LAMHI]`, via frequency-coupled `cmf_solve_J` + e-scatter ALI.
  Output `jbar_line_det[line,shell]`, sentinel −1 outside window.
- **Consumer LIVE** (gate `LUMINA_CMF_LINERES_CONSUME`, plasma.c:6951-6967, 7022-7050):
  for in-window lines, replaces grey binned J in the bb up-rate with **mode-2 differenced pump**
  `bJext = J̄_l − (1−β)·S_lag`, `R_absorb = B_lu·bJext` → into NLTE rate matrix → solved pops →
  `line_source_S` → read by fine emergent. **The J̄_l→pops→S_l→emergent chain is wired (1-iter lag).**
- **Mode-2 is the only stable algebra** (removes the line's own-`S_l` self-term). Mode-1/3 SEALED (explode).
- **macro-atom is MC-only** (THEN_MC); does NOT touch the deterministic emergent. Its energy-weight
  concern is irrelevant here.

## Four breaks (ranked by how load-bearing for the green deficit)

1. **(A) WINDOW MISMATCH — DOMINANT.** UV pump lines (Fe II/Co II resonance forest ~2300-2900Å)
   sit at sentinel −1 when the window is optical (my 3000-12000 runs) → consumer feeds **grey binned J**
   to exactly the pump transitions → optical upper levels stay ~Boltzmann@cold-T_e → `S_l(green)→B` →
   no green. **Necessary & likely largely sufficient fix.**
2. **(B) CLAMP forces thermal.** `FINE_SL_CLAMP=1.0` caps `S_l≤B` — forbids the genuine super-thermal
   emission we want. Currently *inert* (optical window produces no super-thermal to clamp); becomes a
   hard blocker the moment (A) is fixed.
3. **(C) ORTHODOX FLOOR vs genuine pump.** `subres=1e-12·xmax` relative floor (cuda.cu:955-993)
   thermalizes UV-pumped upper levels whose *absolute* pop is tiny (deep cold shells, ground ≫12 dex).
   Attenuates inner-shell green; mid-velocity carriers (above subres) survive.
4. **(D) mode-1 explodes.** Not a deficit cause — a stability constraint: never re-enable mode-1.

## Single best hypothesis (dominant missing piece)

**Fe II self-fluorescence is severed at the window boundary.** The UV Fe II/Co II resonance forest
(2300-2900Å) is not line-resolved in the pump rates, so optical Fe II upper levels (z⁶P°, z⁴D°
feeding **4924/5018/5169** a⁶S–z⁶P° and **4555/4583/4629** b⁴P–z⁴D°) and Co II (~5900-6200Å) are
populated near Boltzmann at cold local T_e instead of cascade-over-populated. Their `S_l` collapses
to `B(T_e)` (NIR) → green deficit. Fe II is the primary optical-green agent in SN Ia at this epoch.

## Fixes (physical discriminators, not scalars)

### Fix-A (window): line-resolve the UV pump
- **(a) wide window 1000-12000Å** — physically safest, removes window confounds. Cost ~10× fine freqs
  (~5M; producer already OpenMP per-ray). Recommended for the FIRST physics test.
- **(b) two windows** UV 1000-3000 ∪ optical 3000-12000 merged into jbar_line_det — cheaper, valid
  (pump rates are frequency-local), use after (a) proves physics. Seam at 3000Å benign for rates.
- **(c) pump-lines-only** — sufficient ONLY if a small UV line set supplies ≥80% of the in-rate to
  green upper levels; Fe II forest-overlap pumping argues this is *marginal* → measure first.

### Fix-B (clamp → b-ceiling discriminator)
Replace fixed `S_l≤1.0·B` with a **per-level departure cap** `b_j ≤ b_ceiling`, trusted only when:
1. **pump provenance**: level `j` has ≥1 in-window line-resolved pump with `J̄_pump > 1.5·B(T_e,ν_pump)`
   (distinguishes pumped from ill-conditioned garbage — garbage has NO super-thermal in-rate);
2. **resolution**: above subres (not floored);
3. **bounded**: `b_ceiling ~ B(T_phot,ν_pump)/B(T_e,ν_pump)` (dilution bound; finite ~10²-10⁴, not 10⁷⁰).
Cap the *level departure*, not `S_l` directly (`S_l` mixes both levels).

### Fix-C (floor): exempt pumped levels
In the orthodox floor loop, **exempt levels that are the upper level of a live super-thermal
line-resolved pump** (`jbar_line_det≥0` AND `J̄>1.5·B`). Garbage levels (no real in-rate) still
floored → conditioning cure preserved (T_e 0.98, n_e dex 0.18 must be unchanged). Do NOT switch the
global floor to a conditioning/residual criterion (re-opens the hole).

### Stability (loop convergence)
Same loop that exploded before. Minimal stable scheme:
1. **Staged**: converge THERMAL first (clamp=1.0, the 6785 fixed point), THEN enable pump as a
   perturbation on converged T_e/n_e. (Most important stabilizer.)
2. **Under-relax** `S_l^new=(1−ω)S_l^old+ω·S_l^solved`, ω≈0.25.
3. **b-ceiling** trust region (Fix-B) as hard NaN/Inf safety net.

## Staged experiment plan

### Step 0 — test break (A) ALONE, cheapest, NO new code
Producer **UV window 1000-3000** (cheap, narrow) + `LINERES_CONSUME=1` + champion + thermal clamp=1.0.
The emergent clamp is irrelevant here — read the **plasma** `lumina_sl_vs_B.csv` (all lines, independent
of emergent clamp): do **optical Fe II green lines (4924/5018/5169, 4555/4583/4629) become
super-thermal `S_l/B>1`**? Upper-level pops are global, so a UV-only pump window over-populates the
optical upper levels even though the optical lines aren't in-window.
- **PASS** (green S_l/B rises >1, traced to UV pump): break (A) confirmed dominant → build Fix-B/C to
  let it through to the emergent.
- **FAIL** (green S_l/B stays ≈1): pump insufficient → either forest-overlap underestimated (need wide
  coupled solve) or a *second root* (green Fe II opacity/ionization too low — not a source-function fix).
- Gate: plasma T_e/n_e must hold (compare_plasma_vs_gold). If CONSUME destabilizes plasma → under-relax.

### Step 1 — full emergent fluorescence
Wide window 1000-12000 (or two-window) + CONSUME=1 + Fix-B (b-ceiling, remove clamp=1.0) + Fix-C
(pump-exemption) + warm-start from thermal champion + ω=0.25 + 8 iters. **Control: CONSUME=0** (=current 0.41).
- **Primary falsifier**: grn/nir 0.41 → toward 0.58.
- **Secondary (must move together)**: green-shell `S_l/B>1` for Fe II 4924-5169; in-rate provenance =
  in-window UV pumps (not floor, not grey-J); NIR thermal tail drops.
- Outcomes: confirmed (grn/nir≥0.50 + S_l/B>1 traced to pumps) / refuted-insufficient (S_l/B>1 but
  grn/nir flat → second root = opacity/ionization) / ambiguous (grn/nir up but S_l/B≈1 → continuum
  re-coloring artifact, distrust).

## Skeptic caveats
- **T_phot dependence**: b_ceiling uses the pump color temp. The MC blue-tilt T_inner bug (memory
  168221) could mis-set it → validate T_phot independently first.
- **UV mesh resolution**: Fe II lines denser per Å in UV; if `cmf_solve_J` under-resolves the overlapping
  UV forest, pump magnitude underestimated → green too weak (refuted-insufficient branch). Check UV PPD.
- **Second root**: if green Fe II *opacity* is also too low (Fe II under-abundant / T_e mis-set in green
  layer), no source-function fix recovers green. Step-0 secondary diagnostics catch this — do NOT
  over-rotate onto "it's all fluorescence."

## Links
[[project_autonomous_stage2_2026-06-25]] (CONTONLY verdict), [[project_toored_ir_fluorescence_rootcause]],
[[project_cmf_lineres_build]] (P7 producer/consumer build), [[project_known_issues_gold_checklist]] #3.
