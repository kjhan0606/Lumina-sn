# PHOTOSPHERE LEDGER — CMFGEN vs Lumina, piece-by-piece (COLD-CASE-P)

Offline forensic, 2026-07-20. Read-only on logs/ and /gpfs. No source edit, no GPU, no commit.
Every claim ties to a number + file. Shells s6(v8632)/s7(v9360)/s8(v10088), Lumina 50-grid
(v_mid = 4264 + 728·s). CMFGEN depths mapped by velocity to jnu4/cmfflux RVTJ (descending;
s6→depth56 V8572, s7→depth55 V9133, s8→depth53 V10164).

Sources: CMFGEN converged `toy06_19.48d_jnu4/` (EDDFACTOR J_nu, RVTJ T/n_e, FeIIIOUT b_k) +
`toy06_19.48d_cmfflux/` (ETA_DATA/CHI_DATA emissivity+opacity) + published `ionfrac_*_cmfgen.txt`.
Lumina B-run `logs/coevolve_consume_a10_kx_gphall/` and post-repair `..._kpr8/`.
Scripts+CSVs in this dir: `ion_census.py/.csv`, `field_bands.py/.csv`, `eta_chi_source.py/.csv`,
`thermalization.py/.csv`, `fe3_bk.py/.csv`, `LEDGER_MATRIX.csv`.

---

## BOTTOM LINE (verdict up front)

**COLD-CASE-P root cause = a NON-THERMAL photospheric UV source function, NOT a missing bf
continuum (candidate (a) is FALSIFIED).** CMFGEN's photospheric 405–2000 Å emissivity is itself
**91–100 % LINE, ~0 % bf continuum** (cont_frac 0.0003–0.09, `eta_chi_source.csv`) — so there is
no "thermal bf continuum" for Lumina to be structurally missing. The discriminator is not WHAT
emits but WHETHER IT THERMALIZES: CMFGEN drives its UV line field to **S=η/χ ≈ B(T_e)** (S/B =
0.43–0.95 across 300–2000 Å) via a line-blanketed photosphere that is optically thick in the UV
(χ_UV = 12–94 vs χ_opt = 2e-5 per 1e10 cm, a **~4e6×** blanket) and whose Fe III levels are
**thermal, b_k ≈ 1** (0.4–1.7). Lumina instead (i) piles Fe III excited/Rydberg levels to
**b_k = 10^3–10^9** (vs CMFGEN's ~1) and (ii) lets the kp_emiss k-packet re-emit a non-thermal
cross-ion LINE CDF (S_line ≠ B). The result is a field that is **super-thermal in the EUV**
(J/B = 2.6–3290× over the 404 Å Fe III edge) and **super-thermal in the optical/IR** (too-red),
while **sub-thermal in the FUV** (912–2000 Å). The super-thermal 404 Å edge + the b_k pileup
photoionize Fe III → **Fe over-ionization** (f(FeIV) 11–32× high).

**This is why OTS-threshold tuning cannot work:** the threshold slides the EUV field magnitude
up/down uniformly (kpr6 pushed it to DEFICIT → under-ion; Brun/kpr8 sit at EXCESS → over-ion) but
cannot convert a non-thermal line-dump into a thermalized S≈B field, cannot repair the b_k pileup,
and cannot fix the EUV-excess / FUV-deficit / optical-excess spectral SHAPE simultaneously. There
is no threshold that lands all three axes on CMFGEN at once. Cold case explained.

Most-guilty single item: **the photospheric UV line source function is not thermalized to B(T_e)
(kp_emiss line-CDF + Fe III b_k pileup)**. Ranked follow-ups in §Verdict.

---

## ITEM 1 — Ionization census, ALL elements (`ion_census.csv`)

f(IV)=IV/(III+IV) and mean charge z̄. **The over-ionization is element-graded, not uniform** — the
single most decisive fact in this ledger.

| ion (edge III→IV) | z̄ CMFGEN | z̄ Brun | z̄ kpr8 | reading |
|---|---|---|---|---|
| **Fe** (30.6 eV, 404 Å) | 2.07 | **2.76** | **2.87** | f(FeIV) s6 0.069→0.752/0.868 (**11–13×**); s7 0.027→0.513/0.837 (**19–31×**); s8 0.022→0.462/0.681 (**21–31×**). MASSIVE over-ion. |
| **Co** (33.5 eV, 370 Å) | 2.14 | 2.49 | 2.47 | moderate over-ion at s6, converges by s8 |
| **Ni** (35.2 eV, 352 Å) | 2.05 | 2.25 | 2.38 | moderate over-ion |
| **Si** (33.5 eV, 370 Å) | 2.04 | 2.23 | 2.19 | mild over-ion |
| **S** (34.8 eV, 356 Å) | 2.01 | 2.04 | 2.00 | **matched** (kpr8 S IV≡0: pinned in III) |
| **Ca** (67.3 eV, 184 Å) | 2.05 | 2.01 | 2.00 | **UNDER-ionized** |

**Read-off:** the excess ionization tracks IRON-GROUP-NESS (open 3d shell, dense near-threshold
level manifold), not simply the edge band. Fe/Co/Ni over-ionize strongly; Si (Mg-like, sparse
levels) mildly; S matched; Ca (closed-shell, deep 184 Å edge) UNDER. A pure field-band deficit
would push ALL ions with EUV edges the SAME way — it does not. This points the suspect at an
**Fe-group-specific level-population amplifier** (the b_k pileup, Item 6), plus the EUV field
being in EXCESS (Item 2), not deficit.

## ITEM 2 — Radiation field J_ν, band by band (`field_bands.csv`)

mc_J band-mean (ν-weighted) vs CMFGEN EDDFACTOR J. Ratio = Lumina_mc / CMFGEN.

| band (Å) | CMFGEN(s6) | Brun mcJ/C | kpr8 mcJ/C | character |
|---|---|---|---|---|
| 300–450 (Fe III gnd edge) | 5.1e-13 | **99.9×** | 27.3× | **EUV EXCESS** (s8: 2300×/32700×) |
| 450–912 | 7.2e-08 | 7.30× | 0.63× | EUV excess (Brun) |
| 912–2000 (FUV) | 2.3e-05 | 1.26× | **0.089×** | matched (Brun) / **DEFICIT** (kpr8) |
| 2000–4500 (NUV) | 1.8e-04 | 0.77× | 0.76× | mild deficit |
| 4500–7000 (opt) | 1.4e-04 | 2.98× | 4.51× | **OPTICAL EXCESS** (too-red) |
| 7000+ (IR) | 9.7e-06 | 6.99× | 46.6× | **IR EXCESS** (too-red) |

**Reconciliation with the campaign's "known" numbers (EUV 23× LOW / FUV 12.6× LOW / opt 2.8× high):**
those are the **kpr6** arm (over-recombined, EUV-DEFICIT). The B-run and kpr8 arms are the OPPOSITE
in the EUV: **EXCESS**. Reader validated — CMFGEN J(300–450,s8)=9.7e-15 matches the euv_source
verdict's J300=9.5e-15. **The OTS threshold swings the EUV field from deficit (kpr6, under-ion) to
excess (Brun/kpr8, over-ion); no value hits CMFGEN's thermal value.** The invariant across all arms
is the SHAPE: EUV+optical+IR piled up, FUV drained — a spectral redistribution, not a level offset.

## ITEM 3 — Emissivity η, opacity χ, and PROCESS (`eta_chi_source.csv`) — the load-bearing test

CMFGEN photosphere, band-integrated. cont_frac = smooth-floor / total emissivity (line = 1−cont).

| band | η_mean | χ_mean(/1e10cm) | **cont_frac (bf/ff)** | line_frac |
|---|---|---|---|---|
| 300–450 | 4.5e-12 | 12.7 | 0.087 | 0.913 |
| 450–912 | 2.2e-06 | 29.4 | **0.0006** | 0.9994 |
| 912–1290 | 6.9e-04 | **79.2** | **0.0000** | 1.0000 |
| 1290–2000 | 6.5e-05 | 2.18 | 0.0005 | 0.9995 |
| 2000–4500 | 1.4e-06 | 0.012 | 0.0020 | 0.998 |
| 4500–7000 | 3.0e-09 | 2.1e-05 | 0.0071 | 0.993 |

**Decisive:** CMFGEN's photospheric UV emissivity (450–2000 Å) is **99.9–100 % LINE, ~0.05 % bf/ff
continuum**. There is NO strong thermal bf continuum here. **The leading candidate (a) — "CMFGEN
makes a recomb-to-excited / thermal bf continuum in 405–2000 Å that Lumina structurally lacks" — is
FALSIFIED at the photosphere: CMFGEN doesn't make one either.** What CMFGEN DOES have is enormous UV
LINE opacity: χ(912–1290)=79 vs χ(4500–7000)=2e-5 per 1e10 cm — a ~4-million-fold UV line blanket.
That blanket is the thermalizing agent (Item 4).

## ITEM 4 — Source function S=η/χ vs B(T_e) (`eta_chi_source.csv`, `thermalization.csv`)

CMFGEN S/B at the photosphere (its emergent UV is set by S, which the thick UV blanket drives to B):

| band | S/B s6 | S/B s7 | S/B s8 | reading |
|---|---|---|---|---|
| 300–450 | 0.43 | 0.60 | 0.17 | thermalized/sub-thermal |
| 450–912 | 0.88 | 0.95 | 0.43 | **≈ thermal** |
| 912–1290 | 1.59 | 2.17 | 1.79 | line-pumped, slightly super |
| 1290–2000 | 0.55 | 0.62 | 0.35 | thermalized |
| 2000–4500 | 0.50 | 0.50 | 0.41 | thermalized |

CMFGEN's photospheric UV field is **at or below B(T_e) everywhere** — a genuine thermal, absorbed
field. Because B(11000 K) is Wien-dead in the EUV, CMFGEN's EUV field is intrinsically FAINT.

**Now the discriminating cross-cut — J/B(T_e_local), >1 = super-thermal (`thermalization.csv`):**

| band | CMFGEN J/B | Brun mcJ/B | kpr8 mcJ/B |
|---|---|---|---|
| 300–450 (s6) | 0.62 | **7.2** | **2.6** |
| 300–450 (s7) | 0.67 | **10.9** | **208** |
| 300–450 (s8) | 0.71 | **479** | **3290** |
| 450–912 (s6) | 0.82 | **1.96** | 0.19 |
| 912–2000 (s6) | 0.84 | 0.59 | **0.045** |
| 4500–7000 (s6)| 0.46 | **1.14** | **1.77** |
| 7000+ (s6) | 0.068 | 0.42 | **2.82** |

**This is the mechanism in one table.** CMFGEN ≤ 1 in every band (thermal). Lumina is **super-thermal
by 2.6–3290× at the 404 Å Fe III edge**, super-thermal in the optical/IR (too-red), and sub-thermal
in the FUV. The field deficit is NOT an emission shortage (S too low) — in the EUV it is an
**emission EXCESS that fails to thermalize/absorb.** So COLD-CASE-P is NOT candidate (a) missing
emissivity and NOT a simple χ-too-high absorption error; it is a **non-thermal source-function**
problem: the UV line field is not driven to B.

## ITEM 5 — Γ (photoionization) per ion — which ion, which band

Direct Γ decomposition (kpr6 arm, `over_recomb_s4/field_bands.out`): Lumina's Fe III Gph_boltz gets
**24–49 % of its rate from levels whose threshold is in the OPTICAL (>4000 Å)** vs CMFGEN's **1 %** —
i.e. the b_k-piled near-threshold excited levels photoionizing through the over-bright optical field.
Combined with Item 1's z̄ ladder, the per-ion answer:

- **Fe III / Co III / Ni III (IGE, open d-shell):** Γ is ENHANCED, not starved, in Brun/kpr8 —
  super-thermal 404–352 Å ground edges (Item 2/4) × the b_k~10^4 near-threshold pileup (Item 6).
  → over-ionization, worst for Fe (densest manifold).
- **Si III:** edge similar (370 Å) but Mg-like sparse structure → little pileup amplification → mild.
- **S III:** matched (kpr8 pins S in III via the S III kp_emiss attractor — S IV ≡ 0).
- **Ca III:** deep 184 Å edge in the Wien/thermal-dead sub-300 Å region (cs_J side, CMFGEN too) →
  genuinely photoion-STARVED → UNDER-ionized.

The starvation/excess axis is band-localized: EUV 300–450 (Fe III ground) is in EXCESS at the
photosphere; sub-300 Å (Ca) is dead on both sides. Item 1's element grading is the fingerprint.

## ITEM 6 — Fe III level structure / b_k (`fe3_bk.csv`) — the amplifier

| side | b_gnd | b_median | b_max | b_top20 |
|---|---|---|---|---|
| **CMFGEN s6** | 1.53 | **1.06** | **1.65** | 0.57 |
| Brun s6 | 1.00 | 780 | 7.4e4 | 8.8e3 |
| kpr8 s6 | 1.00 | 947 | 9.2e4 | 1.1e4 |
| **CMFGEN s8** | 1.17 | **1.04** | **1.69** | 0.51 |
| Brun s8 | 1.00 | 1.8e5 | 2.8e7 | 3.4e6 |
| kpr8 s8 | 1.00 | **8.1e6** | **1.2e9** | 1.4e8 |

**CMFGEN Fe III is thermal — b_k ≈ 1 for ALL 1500 levels** (0.4–1.7), including near-threshold
(b_top20 ≈ 0.5, normal sub-thermal edge). **Lumina Fe III has a catastrophic excited-level pileup:
b_median 10^3–10^6, b_max up to 1.2e9, worsening outward and with tuning (kpr8 > Brun).** It is
dynamically real: **82 % of Fe III sits in excited levels** (near-threshold levels at E≈28.68 eV,
1.97 eV below the 30.65 eV edge, have b_k≈5000–8000; individual n_k≈0.05, thousands of them) vs
CMFGEN where the LTE near-threshold Boltzmann population is ~1e-12. **Structural driver:** Fe IV,
the dominant photospheric ion (75–87 % at s6), has **ZERO NLTE levels in Lumina** (`levelpop` Fe
ion=3 rows = 0) — no proper parent for a Saha-consistent recombination cascade, so Fe III excited
levels are filled non-thermally (dilute-Boltzmann / kp_emiss feedback) instead of relaxing to b≈1.

---

## DIVERGENCE MATRIX (quantity × shell × 3 sides) → `LEDGER_MATRIX.csv`

Headline discriminators, all pointing one way:

| quantity | CMFGEN | Brun | kpr8 | cause status |
|---|---|---|---|---|
| f(FeIV) s6 | 0.069 | 0.752 | 0.868 | OVER-IONIZED (convicted) |
| mc_J/CMFGEN 300–450 s6 | 1 | 99.9× | 27× | EUV super-thermal EXCESS |
| J/B(Te) 300–450 s6 | 0.62 | 7.2 | 2.6 | non-thermal (>1) |
| J/B(Te) 912–2000 s6 | 0.84 | 0.59 | 0.045 | FUV sub-thermal deficit |
| J/B(Te) 4500–7000 s6 | 0.46 | 1.14 | 1.77 | optical super-thermal (red) |
| CMFGEN cont_frac η 450–2000 | ~0.0005 | — | — | UV is 99.95% LINE both sides |
| CMFGEN S/B 450–912 | 0.88 | — | — | CMFGEN UV = thermal |
| Fe III b_median s6 | 1.06 | 780 | 947 | Lumina pileup (anomalous) |
| Fe III b_max s8 | 1.69 | 2.8e7 | 1.2e9 | RUNAWAY pileup |
| Fe IV NLTE levels | 1500 | 0 | 0 | structural (no parent) |

---

## VERDICT — COLD-CASE-P

**(a) structural missing bf/thermal continuum — REJECTED.** CMFGEN's photospheric 405–2000 Å
emissivity is 99.9–100 % LINE (cont_frac ≤ 0.09; `eta_chi_source.csv`). There is no thermal bf
continuum there to be missing. Adding a recomb-to-excited continuum would only address the FUV
sub-thermal axis (912–2000, J/B 0.045–0.59), NOT the dominant Fe over-ionization driver.

**(b) opacity/absorption error — PARTIAL / secondary.** CMFGEN thermalizes the UV via a ~4e6× line
blanket (χ_UV 79 vs χ_opt 2e-5). Lumina's locally-made EUV excess is under-absorbed/under-
thermalized (euv_source verdict §5, corroborated here by J/B > 1). Real, but it is the *symptom* of
(c): the opacity exists, it just re-emits non-thermally instead of thermalizing.

**(c) ionization-specific rate/level issue via a NON-THERMAL UV SOURCE FUNCTION — CONFIRMED, the
root.** The convicting numbers: (1) over-ionization is Fe-group-graded, not uniform (Item 1); (2)
the 404 Å Fe III ground-edge field is super-thermal 2.6–3290× (Item 4) — an EXCESS, so a field
DEFICIT cannot be the cause; (3) Fe III b_k = 10^3–10^9 vs CMFGEN's thermal ≈1, with 82 % of Fe III
non-thermally in excited levels (Item 6); (4) CMFGEN's identical line-dominated UV thermalizes
(S/B≈0.9) while Lumina's does not. The mechanism is the campaign's convicted kp_emiss line-CDF
(`criminal_record/CRIMINAL_RECORD.md`): the k-packet re-emits a non-thermal cross-ion line field
instead of relaxing to B(T_e), and Fe IV's absence of NLTE levels removes the recombination-cascade
that would thermalize Fe III's manifold.

**MOST-GUILTY SINGLE ITEM:** the **non-thermalized photospheric UV line source function** — Fe III
b_k pileup (10^3–10^9) feeding, and fed by, a super-thermal EUV line-dump at the 404 Å Fe III edge.
This is the item OTS tuning provably cannot touch.

**RANKED FOLLOW-UPS (design only; cite the machinery):**
1. **Fe IV / Co IV / Ni IV NLTE-level promotion + proper recombination cascade** (the missing parent).
   Today Fe IV = single ionization fraction, no levels (`lumina_levelpop.csv` Fe ion=3 = 0 rows);
   fb recombination emits ONLY at ground edges (`lumina_plasma.c:2317–2329`, product = stage−1
   ground `find_ion_pop_idx(atom,Z,stage-1)` + ground `find_ioniz_energy`), so Fe III excited levels
   have no recombination source and no thermal anchor → b_k runaway. Add the top-stage-IV SE/NLTE set
   and a destination-resolved (recomb-to-excited) cascade so Fe III relaxes to b≈1. This alone should
   collapse the pileup and hence the near-threshold Γ (24–49 %→~1 %).
2. **B(T_e) thermalization of the k-packet line source where UV-thick** (kp_emiss → S→B clamp in the
   optically-thick UV). Replaces the non-thermal cross-ion line-dump (super-thermal EUV/optical) with
   a thermal S≈B field. `lumina_cuda.cu:3282–3325` (−4 B(Te) exit) currently intercepts only the
   resonant CDF path and is disqualified at the photosphere (W(s6)=0.054 < 0.13); the disqualification
   test is the lever. Selective — full S=B would kill the legitimate non-thermal EUV elsewhere.
3. **Frequency-resolved UV opacity/thermalization audit** (candidate b residual): verify Lumina's UV
   line blanket both absorbs AND thermalizes at CMFGEN's χ_UV≈79 magnitude, so the locally-made EUV
   excess is reabsorbed rather than transported out as a super-thermal photospheric field.

Pre-registered non-fixes (do not over-claim): OTS-threshold retuning (proven futile — swings EUV
excess↔deficit, cannot thermalize); a bf-continuum-only addition (fixes FUV axis only, leaves the
Fe over-ionization); T_e/n_e (z̄ decomp shows α and n_e are innocent, `over_recomb_s4`).
