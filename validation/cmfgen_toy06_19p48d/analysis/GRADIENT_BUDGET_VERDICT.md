# Gradient-Budget Verdict — CMFGEN vs Lumina B-run, toy06 @19.48d

**Question (Fable):** of the missing deep→photosphere Fe IV/III ionization gradient,
how much is carried by (a) the FIELD, (b) the T_e profile, (c) rate normalization?

**Verdict (one line):** The missing gradient is carried **almost entirely by the FIELD**,
through the field-folded photoionization rate Gph. The recombination axes (n_e and
α(T_e)) match CMFGEN to within ≤0.3 dex and are **not** the culprit. Lumina's Gph is
flat (**+0.27 dex** s0→s8) where CMFGEN's spans **+6.67 dex** — a **6.4-dex** deficit in
the ionizing-rate gradient. The field flatness is **TRANSPORT-REAL, not a T_rad-pin
artifact** (the Gph-driving field is the transported MC field `mc_J`, and even the
deterministic `cs_J` is flat in the dominant 918–1290 Å band).

Span for every number below: **s0 (4264) → s8 (10088 km/s)**, +dex = deep more ionized.
Analysis: `gradient_budget.py`; per-shell data: `gradient_budget_shells.csv`.

---

## 0. Yardstick audit (which field feeds Gph — cite file:line)

Config of the B-run (`logs/coevolve_consume_a10_kx_gphall/stdout.log` header):
`LUMINA_GPH_ALLLEVEL=1`, `LUMINA_GPH_SIGMA_CMFGEN=1`, `LUMINA_COEVOLVE_PHOTOION_MC=1`,
`LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0` (no `GPH_ALLLEVEL_NLTE`, no `PHOTOION_OCC`).

The Gph rate integral blends the field at three sites in the all-level + Kramers paths:
`src/lumina_plasma.c:5261-5269`, `:5304-5312`, `:5359-5365`:

```
J = alpha * g_photoion_mc_J[s*nfb+bb] + (1-alpha) * nlte->J_nu[s*nfb+bb];   // alpha=1.0
```

With **α=1.0** (confirmed `[COEVOLVE-PHOTOION] … alpha=1.00`, stdout.log:3563) this is
**J ≡ g_photoion_mc_J**. That array is a memcpy of the transported MC shadow field
`nlte_Jmc` (`src/lumina_cuda.cu:5188`), dumped as the **`mc_J`** column of
`lumina_coevolve_field.csv` (`:5324`). `db_photoion_calc.field()` reads exactly this
column (mc_J if >0 else cs_J; floor=1e-30>0 ⇒ mc_J used everywhere) — run-faithful.

**Definitional-flatness check (required):** T_rad is pinned at **10470.093 K in all 50
shells (uniq=1)**. But the field that drives Gph is `mc_J` = the MC-transported field,
**not** a dilute-Planck W·B(T_rad). Independent confirmation: the deterministic `cs_J`
UV/opt color *varies* shell-to-shell (det color 6.02 @s10 vs 3.96 @s25, stdout.log
[COEVOLVE-COLOR]) — a fixed-shape W·B(T_rad) would be shell-invariant. **⇒ the field
flatness is transport-real, not definitional.**

---

## 1. THE GRADIENT BUDGET TABLE (Fe III→IV, s0→s8)

Framework — photoionization equilibrium, **exactly additive in dex**:
`Δlog[n(IV)/n(III)] = Δlog(Gph) − Δlog(n_e) − Δlog(α(T_e))`

| Axis (dex, s0→s8) | CMFGEN | Lumina B | Missing (C−L) | Carries the defect? |
|---|--:|--:|--:|:--|
| **TOTAL Fe IV/III gradient** (measured, ionfrac/ion_pops) | **+5.09** | **+0.65** | **4.44** | — |
| **FIELD → Gph_boltz** (field-folded rate, run's all-level scheme) | **+6.67** | **+0.27** | **6.40** | **YES ← dominant** |
|  ↳ Gph_gnd (pure field, ground-only) | +6.87 | +0.30 | 6.57 | (field) |
|  ↳ T_e-in-weights (Boltzmann) | −0.20 | −0.03 | 0.17 | no |
| **n_e** (−Δlog n_e) | −0.85 | −0.77 | 0.08 | **no (matches)** |
| **recomb** (−Δlog α(T_e), Milne) | +0.38 | +0.10 | 0.28 | **no (matches)** |
| — predicted (Gph−n_e−α) | +6.20 | −0.40 | — | — |
| — residual / NLTE closure | −1.11 | +1.05 | — | (see caveats) |

**Read-off:** of the 4.44-dex missing total, the **field-folded Gph accounts for 6.40 dex**
of leverage difference (it *over*-carries and is clawed back by n_e). The n_e axis differs
by 0.08 dex, the recombination-coefficient axis by 0.28 dex — **neither is the culprit.**
Rate normalization is not implicated: both codes use the identical real-σ / Milne kernel.

---

## 2. Pre-registered discriminators D1–D4

| # | Discriminator (s0→s8) | CMFGEN | Lumina | Note |
|---|---|--:|--:|:--|
| **D1** | J(918–1290 Å) decline, geom-mean | **+2.41** | **−0.19** | Lumina flat/inverted; peaks mid-envelope (s5–s7) |
| **D2** | J(300–450 Å) decline, geom-mean | +7.21 | — | Lumina mc_J **floor-dominated** at photosphere |
| **D2** | J(300–450 Å) decline, arith-mean | +6.15 | −0.83 | 44–45 of 77 bins at 1e-30 floor for v≥8600 |
| **D3** | Γ(Fe III) decline (field-folded) | +6.67 | +0.27 | = the Gph row above |
| **D3** | Γ(Co III) decline (field-folded, all-level) | +7.37 | +1.39 | Co total: C +4.09 vs L +1.53 (trigger ion) |
| **D4** | Saha T_e-alone (n_e fixed@s0) | **+7.23** | **+2.65** | T_e 18900→10345 (C) vs 13120→10811 (L) |

- **D1:** CMFGEN's 918–1290 Å field (the band that dominates the *all-level* Gph via
  excited-level thresholds) declines +2.41 dex; Lumina's `mc_J` there is **flat (−0.19)**.
  Floor-bin count = 0 (all 64 bins sampled) ⇒ this flatness is a genuine transported field,
  not undersampling.
- **D2:** CMFGEN's EUV ionizing continuum (Fe III/Co III ground threshold ~370–404 Å)
  collapses +6–7 dex from deep-hot to photosphere-cool. Lumina's `mc_J` there is
  **floor-dominated** at the photosphere (MC delivers ~no EUV photons past v~8600) — so it
  neither declines cleanly nor carries a gradient; it is simply absent/noise.
- **D4 (T_e axis, bound):** CMFGEN's T_e(v) alone would *source* +7.23 dex of LTE
  ionization gradient; Lumina's shallower T_e(v) only +2.65 dex — a **4.58-dex T_e-leverage
  gap**. This is the ROOT that starves the field (see §3), but it is **not additive** with
  the field row (in reality the field mediates it; the equilibrium accounting in §1 is the
  non-overlapping decomposition, where T_e's *direct* effect via α+weights is only ~0.45 dex).

---

## 3. Proximate cause vs root cause

- **Proximate (equilibrium accounting, §1):** the **FIELD** (via Gph) is the axis. n_e and
  α(T_e) match CMFGEN within 0.3 dex. Verdict: *"the missing ~4.4 dex is carried by the
  field-folded photoionization rate."*
- **Root of the field flatness — two coupled defects:**
  1. **Deep T_e deficit** (13120 K vs CMFGEN 18900 K at s0). Cooler deep gas under-powers
     the deep EUV ionizing continuum; D4 quantifies this as a 4.58-dex loss of T_e-driven
     leverage that manifests *through* the field.
  2. **MC EUV transport starvation.** In the ground-threshold band (300–450 Å), the
     deterministic `cs_J` carries +2.76 dex but the MC `mc_J` (which α=1.0 selects) is
     floor-dominated and flat (+0.30). So α=1.0 **discards** the deterministic field's real
     EUV gradient. HOWEVER the run's *all-level* scheme is dominated by excited levels
     (918–1290 Å), and **there both cs_J (+0.64) and mc_J (+0.27) are flat** — so switching
     α→0 alone would not restore the gradient; the 918–1290 flatness is intrinsic to
     Lumina's radiation field, upstream of the MC/deterministic choice.

**Photospheric localization:** Lumina's *deep* ionization is fine (s0 f(IV)=0.79, CMFGEN
fully IV). The entire defect is **photospheric over-ionization**: Lumina holds f(FeIV)=0.46
at s8 where CMFGEN has 0.02. The flat field fails to let Fe recombine to III at the
photosphere. (Co III mirrors this: missing 2.56 dex total.)

---

## 4. Caveats / provenance

- CMFGEN thermodynamics (T_e, n_e, ionfrac) = **published converged** state
  `data/standart_data1/toy06/{phys,ionfrac_fe,ionfrac_co}_toy06_cmfgen.txt` @19.480d.
- CMFGEN field = **self-run 4-iter snapshot** `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/
  EDDFACTOR` (FINISH_REC=1, ND=90, 196185 freqs), interpolated to the σ grid. Non-converged
  field folded with converged T_e ⇒ the −1.11-dex CMFGEN closure residual (also absorbs
  Boltzmann-vs-true-NLTE weights). The **relative** Gph gradient (6.67 vs 0.27) is robust to
  this; the absolute closure is not claimed.
- Both Γ sides use the identical real-σ kernel (`scripts/db_photoion_calc.py`, Fe III
  tabulated CMFGEN σ_bf + Co III real-σ patch `data/coiii_real_sigma_patch.npz`), so the
  Fe/Co comparison is data-symmetric (rate-normalization axis controlled out).
- Lumina forming shells = geometry indices {0,2,4,5,6,7,8,9,10}; CMFGEN matched by nearest
  velocity depth.
