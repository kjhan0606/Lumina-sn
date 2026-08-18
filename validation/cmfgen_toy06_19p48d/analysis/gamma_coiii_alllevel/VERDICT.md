# Γ_all(Co III) from CMFGEN's own data — "G보정 차용증" trigger ①

**2026-07-29.** Source = one converged CMFGEN snapshot, `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/`
(toy06 @ 19.48 d). No Lumina data enters the calculation. Script: `gamma_coiii_alllevel.py`
(offline, CPU, ~2 s). Full stdout: `run_log.txt`.

```
Γ_all(d) = Σ_l [n_l(d)/n_CoIII(d)] · 4π ∫_{ν_l}^∞ σ_l(ν) J_ν(d)/(hν) dν
Γ_gnd(d) = the same integral for the single ground level 3d7_a4Fe[9/2] (no population weight)
```

| quantity | file | how |
|---|---|---|
| n_l (1000 F-levels × 90 depths) | `POPCOB` | `rite_asc.f`: `CI(NCI,ND)` column-major → `reshape(90,1000)`, then `DCI(ND)` |
| σ_l | `PHOTCoIII_A` | 386 term entries, **all** Type-1 Seaton; `sub_phot_gen.f:412-417`, `CONV_FAC=1e-8` |
| ν_l, g_l, χ | `CoIII_F_OSCDAT` | χ = 270200 cm⁻¹ = 33.500530 eV; ν_l = (χ−E_l)·c |
| J_ν | `EDDFACTOR` | 196185 frequencies, 3.499e12–1.0e18 Hz (3 Å – 8.6e5 Å), `FINISH_REC=1` |
| v(d), T_e(d) | `RVTJ` | |
| **CMFGEN's own rate** | `CoIIIPRRR` | `PRRR_SL_V6:126` `PR(I,J)=Σ_ν WSE·HN·JPHOT` = n_SL·R_SL [cm⁻³s⁻¹] |

σ form actually in the file (**every one of the 386 terms**, no exceptions):
`A1 = 1.0`, `A2 = 2.0`, `A0 ∈ [1.6826, 4.3755] Mb` ⟹ `σ_l(ν) = A0·1e-18·(ν_l/ν)²  cm²`.
Only the threshold ν_l and the scale A0 differ between levels. The file self-describes as
"**Very crude** photoionization cross-sections for CoIII" — see the caveat section.

---

## 1. Verification gates

| gate | result |
|---|---|
| **V1** POPCOB round-trip | **PASS.** `DCI(ion)/ground(next ion) = 1.00000000` for **all 4 ion pairs × 90 depths**, 0 depths off by >1e-6. (Co2→CoIII, CoIII→CoIV, CoIV→CoV, CoV→CoSIX.) |
| **V2** Γ_gnd anchor | **PASS, 3 digits.** All 9 shells reproduce the repo anchor with ratio **1.011–1.023** (uniform +1–2 %). s0 0.4438 vs 0.4344; s6 1.928e-8 vs 1.884e-8; s8 1.758e-7 vs 1.727e-7. The uniform +2 % is the anchor's coarse 1000-bin log-ν grid vs this work's native 196185-point CMFGEN grid — it is a quadrature offset, not a physics difference. |
| **V3** σ name matching | **PASS, 100 %.** 1000/1000 F-levels matched to a phot term by name (`osc name minus [J]` → phot config); all 386 terms in the file are used; **zero** levels dropped, `frac_pop_without_sigma = 0.000` at every depth. 0 levels lie above the ionization limit. |
| **V4** population closure | **PARTIAL (explained).** Σ_ions Σ_l n_l (+top DCI) / `POP_SPECIES(Co)` = 1.000033 / 1.000226 / 0.999290 / 0.995057 at s0–s3, but 0.986 / 0.984 / 0.910 / 0.882 / **1.072** at s4/s5/s6/s7/s8. Cause: the model carries Co II–Co VI plus only the *ground* level of Co VII, so where Co is more ionized the closure must under-count — a property of the model atom, not of the parse. **s8 (depth 51) is a 7 % over-shoot sitting exactly on the composition cliff** (n_Co jumps 1e5× between depths 50 and 51); s8 numbers carry that caveat. Γ_all/Γ_gnd is a ratio *internal* to Co III and is unaffected by the missing stages. |
| **V5** vs CMFGEN's own converged rate | **PASS, ~1 %.** `Σ_SL PR(SL,d)/n_CoIII` from `CoIIIPRRR` vs this work: **0.993, 0.993, 0.992, 0.991, 0.989, 0.984, 0.973, 0.971, 0.880** for s0…s8. |
| **V5b** ground-*term* rate, anchor-independent | R_SL1 (the 4 `3d7_a4Fe` J-levels) mine/CMFGEN = 0.992…0.975, 0.879 at s8. |

Additional cross-checks (all in `run_log.txt`):
- **Kernel self-check**: brute-force `4π∫σJ/(hν)dν` for the ground level at s0 vs the cumulative-moment kernel: **+0.00000 %**.
- **Grid convergence** (2× refined ν grid, log-log J interpolation): top-5 levels change by **−0.004 % … −0.014 %**; Γ_all(s0) changes by **−0.002 %**.
- **J units**: ∫J dν at the innermost depth = 4.789e12 vs σT⁴/π = 5.069e12 (ratio 0.945) — thermalized inner boundary, correct units.
- **Level ordering**: `CoIII_F_TO_S` reproduces the `CoIII_F_OSCDAT` names in identical order 1…1000 — independent confirmation of the ordering assumed for POPCOB.
- **Super-level assumption**: within every super level, `n_l/(g_l e^{−E_l/kT_e})` is constant to **3e-8 (median), 9e-8 (max)** — CMFGEN's Boltzmann-at-T_e distribution inside a super level is confirmed exactly.

**Gate order was respected**: Γ_all was not computed/reported until V1 and V2 passed.

---

## 2. Result — Γ_all/Γ_gnd (Co III), per forming shell

| shell | v_target | depth | v_CMFGEN | T_e | n_CoIII | **Γ_gnd** | **Γ_all** | **Γ_all/Γ_gnd** | Γ_boltz/Γ_gnd |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 4264 | 67 | 4394 | 18536 | 1.484e6 | 4.4377e-01 | 1.4903e+01 | **33.6** | 48.0 |
| s1 | 5720 | 63 | 5841 | 16154 | 1.600e7 | 1.6330e-02 | 5.9794e-01 | **36.6** | 55.4 |
| s2 | 7176 | 60 | 7019 | 13925 | 1.857e8 | 3.9358e-04 | 1.5625e-02 | **39.7** | 52.4 |
| s3 | 7904 | 58 | 7983 | 12560 | 2.639e8 | 1.7138e-05 | 7.2948e-04 | **42.6** | 55.2 |
| s4 | 8632 | 57 | 8572 | 11987 | 1.813e8 | 3.7530e-06 | 1.6775e-04 | **44.7** | 56.4 |
| s5 | 9360 | 56 | 9133 | 11387 | 1.118e8 | 6.8029e-07 | 3.2534e-05 | **47.8** | 56.8 |
| s6 | 10088 | 54 | 10164 | 10323 | 3.212e7 | 1.9284e-08 | 1.1000e-06 | **57.0** | 60.4 |
| s7 | 10816 | 53 | 10706 | 10118 | 1.391e7 | 1.9904e-08 | 1.4914e-06 | **74.9** | 44.1 |
| s8 | 11544 | 51 | 11815 | 10173 | 1.124e6 | 1.7578e-07 | 2.3885e-05 | **136** ⚠ | 26.7 |

Units s⁻¹. Γ_boltz = the same sum with LTE(T_e) populations instead of CMFGEN's NLTE ones,
shown to separate "level structure" from "NLTE excitation". Depth = 1-based, chosen as the
CMFGEN depth nearest the target velocity (same convention as the anchor CSV). All 90 depths
are in `gamma_coiii_alllevel.csv`.

**The campaign's ~10× expectation is not what the data give: the correction is 34–57× across
s0–s6, rising to 75× at s7 and 136× at s8.** Reported as computed.

Additional observations, not assumptions:
- The ratio **increases outward monotonically** through s0→s6 (33.6→57.0) — the correction is
  not a constant factor and cannot be absorbed into a single scalar.
- Γ_boltz/Γ_gnd ≈ 48–61 at s0–s6 is *larger* than Γ_all/Γ_gnd, i.e. CMFGEN's NLTE Co III is
  **less** excited than LTE there. At s7/s8 the sign flips (Γ_all/Γ_gnd 75/136 vs Γ_boltz 44/27):
  the outer NLTE population is **more** excited than LTE.

---

## 3. Which levels drive Γ_all

**No single level dominates.** The largest single-F-level share is **0.8 %**.

| shell | N levels for 50 % | for 90 % | for 99 % | ⟨E_l⟩ weighted | ⟨λ_th⟩ weighted |
|---|---|---|---|---|---|
| s0 | 169 | 651 | 925 | 17.7 eV | **1012 Å** |
| s6 | 173 | 637 | 920 | 18.0 eV | **1017 Å** |
| s8 | 148 | 573 | 892 | 18.8 eV | **1033 Å** |

Contribution share by excitation energy band (s0 / s6 / s8):

| E_l band | 0–1 | 1–3 | 3–6 | 6–10 | 10–14 | 14–18 | 18–40 eV |
|---|---|---|---|---|---|---|---|
| s0 | 0.019 | 0.044 | 0.030 | 0.099 | 0.125 | 0.205 | **0.479** |
| s6 | 0.016 | 0.040 | 0.025 | 0.089 | 0.107 | 0.229 | **0.494** |
| s8 | 0.007 | 0.035 | 0.012 | 0.032 | 0.076 | 0.344 | **0.493** |

**≈ 50 % of Γ_all comes from levels above 18 eV, and ~70 % from above 14 eV.** The
population-weighted mean threshold is **≈ 1010–1030 Å**, i.e. the all-level Co III
photoionization is driven by the **FUV**, *not* by the 370 Å ground edge. Only ~2 % rides on
the ground term. The sum converges inside the model atom (99 % reached by level 925 of 1000;
the highest level sits at 28.56 eV, λ_th = 2510 Å).

Top-10 F-level contributors (share of Γ_all, from `gamma_coiii_level_breakdown.csv`):

- **s0 / s6** (same cast): `3d7_a2He[11/2]` (E=2.82 eV, λ_th 404 Å, 0.81 %/0.74 %),
  `3d6(3H)4s_a4He[13/2]` (8.88 eV, 504 Å), `3d7_a4Fe[9/2]` **ground** (0.69 %/0.58 % — rank 3 at
  s0, rank 5 at s6), `3d7_a2Ge[9/2]` (2.11 eV), `3d7_a2He[9/2]` (2.91 eV),
  `3d6(1I)4s_a2Ie[13/2]` (10.60 eV, 541 Å).
- **s8**: the cast changes to the 3d⁶4p odd levels — `3d6(1I)4p_y2Io[13/2]` (17.59 eV,
  λ_th 779 Å, R_l = 21 s⁻¹ with n_l/n = 8e-9), `3d6(3H)4p_4Io[15/2]` (15.64 eV, 694 Å),
  `3d6(a1G)4p_x2Ho[11/2]`, `3d6(1I)4p_z2Ko[15/2]`. The ground level drops out of the top 10.

**Super-level view** (the F-levels are *not* independent unknowns — 1000 F → 52 super levels,
Boltzmann-at-T_e inside each): 50 % / 90 % of Γ_all is carried by **11 / 31 super levels** at s0
(10/31 at s6, 8/27 at s8). Leading solved degrees of freedom, s0: SL20 (E_min 15.27 eV, 43 F
levels) 9.4 %, SL45 (25.61 eV, 148 F) 6.7 %, SL10 (8.79 eV, 15 F) 6.1 %, SL21 (15.62 eV) 5.3 %.
At s8 SL20 rises to 14.8 % and SL21/SL23 to 8.8 %/7.7 %.

---

## 4. Caveats — stated as unresolved, not resolved

1. **s8 (depth 51) is the weakest row.** It sits on the Co composition cliff (n_Co jumps 1e5×
   from depth 50 to 51), its species closure over-shoots by 7 %, and it is the only shell where
   V5 lands at 0.88 rather than ~0.99. Its Γ_all/Γ_gnd = 136 should be treated as **order-of-
   magnitude, not 3-digit**. s7 (75) is the outermost row I would quote with confidence.
2. **The σ data are "Very crude" by CMFGEN's own header.** Every one of the 386 terms is a
   single-parameter ν⁻² Seaton fit with no resonance structure, A0 varying only 1.68→4.38 Mb
   across 33 eV of excitation energy. So Γ_all/Γ_gnd here is dominated by *threshold shift*
   (λ_th 370 Å → ~1000 Å, where J is orders of magnitude larger) times *population*, **not** by
   any real level-dependent cross-section physics. Whether the true (resonance-resolved) Co III
   cross-sections would give the same 34–57× is **undetermined by this exercise**. This is the
   single largest open uncertainty in the number.
3. **Half of Γ_all rides on E_l > 18 eV levels whose individual populations are an LTE
   distribution inside their super level.** The *super-level totals* are solved; the split
   within a super level is assumed. This is CMFGEN's own assumption, so Γ_all computed here is
   by construction "CMFGEN's Γ" (V5 confirms it to 1 %) — but it is **not** independent evidence
   that the true high-level populations are right.
4. **The residual ~1 % vs `CoIIIPRRR` is not chased down.** It is uniform across shells and the
   likely cause is CMFGEN's continuum-band quadrature weights (`FQW` on the NCF continuum grid)
   vs the trapezoid on the full CMF grid used here. Not investigated further — it is far below
   any campaign-relevant threshold.
5. **This is Γ at a single epoch (19.48 d) for one model (toy06).** No claim is made about
   other epochs, other ions, or Fe III (the same machinery would run for Fe III, whose phot data
   are *tabulated*, i.e. genuinely resolved — that comparison has not been done).
6. **Nothing here says what Lumina should do.** The number is a CMFGEN measurement. Whether
   Lumina's ground-only Γ should be multiplied by it depends on whether Lumina's Co III level
   populations and FUV field match CMFGEN's — both are separately contested in this campaign
   (b_k 2–20×, s12+ FUV starvation). Applying 34–57× blind would be an unvalidated patch.

---

## 5. Artifacts

| file | content |
|---|---|
| `gamma_coiii_alllevel.py` | the whole calculation, reproducible: `python3 gamma_coiii_alllevel.py` |
| `gamma_coiii_alllevel.csv` | all 90 depths: depth, v, T_e, shell, n_CoIII, Γ_gnd, Γ_all, Γ_all(CMFGEN PRRR), Γ_boltz, ratio, frac_pop_without_sigma |
| `gamma_coiii_level_breakdown.csv` | top-20 F-level contributors at each of the 9 shells |
| `run_log.txt` | full gate output |
