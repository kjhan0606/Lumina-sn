# RADEQ T_e audit — why the deep gas solves 2000-2600K below its own bath

Offline source + run-state analysis, 2026-07-19. Read-only on logs/ and /gpfs. No source edits, no commit, no job touched.
Shell s0 (v=4264 km/s). Run: `logs/coevolve_consume_a10_kx_gphall` (B-run: RADEQ_TE=1 RADEQ_SIMUL=1 VR_STD=1
DAMP=0.5 FB_RATE=1 GPH_ALLLEVEL=1, MC_COEVOLVE consume, PHOTOION_ALPHA=1.0). CMFGEN toy06 @19.48d benchmark.

---
## Headline
The radeq balance **formulas are faithful**: fed CMFGEN's own J_nu they reproduce CMFGEN's deep T_e to 2.6%
(coupled root **18277 K** vs truth **18760 K**). The deep gas lands cold because **the field the balance
consumes for heating is starved**, not because a balance term is wrong. Two field-side starvations, quantified below,
account for the full 13120→18760 gap. Same γ-deposition enters both codes verbatim, so deposition is exonerated.

## FORK (bottom line, with margins)
**Balance FAITHFUL → mastermind = the deep field (bath reddening + EUV starvation + an un-pumped cooling channel).**
Decomposition of the 5640 K deficit (13120 → 18760), all at fixed s0, coupled ionization re-solve:

| lever | ΔT_e | dex | attribution |
|---|---|---|---|
| gas at zero-pump root vs its OWN cs.J line-cooling root (16617) | **+3400 K** | dominant | deep line cooling receives ~no radiative pumping (see caveat) |
| own cs.J root (16617) vs CMFGEN-field root (18277) | +1660 K | | spectral color: EUV/FUV −2 dex, NUV/opt pile-up |
| CMFGEN-field root (18277) vs truth (18760) | +480 K | 2.6% | residual formula/atomic-data error (balance is faithful) |

The single largest component (+3400 K) is that the deep gas commits the **zero-pump** root even though its own
line-cooling field (deterministic `cs.J`, dumped super-thermal 1.55× at 2500 Å) would, if fully coupled, pump the
Fe/Co/Ni III UV-resonance forest to net heating and lift the root to 16617 K. This is a **radiation-field-coupling**
symptom, not a wrong cooling coefficient (the same coefficient, pumped by B(T_e), gives Λ≈0 exactly — detailed
balance holds).

---
## 1. Is T_e(s0)=13120 a genuine root or a fallback?
- **It is the run's stable committed fixed point:** T_e[0] climbs 11463→12364→13120 (iters 0-1-2) then is
  **identical to 6 sig figs (13119.874754) for iters 2-11** (`stdout.log:242,3599,6995,…,29938`), while the mid-shell
  field J[mid,500] grows **20×** over the same span. A field-responsive root would drift; digit-identical stability
  under a 20×-growing field is the **fingerprint of a HOLD (pin_lo)** — `radeq_simul_all` HOLDs the previous T_e when
  `f_lo=simul_r1(3500)≤0` (`lumina_plasma.c:5639-5640`). The run reports 9-14 lo-pins of 50 shells every iteration
  (`stdout.log:217,3574,…`); s0 (deepest, strongest deposition, 2.14 M lines) is a prime lo-pin candidate.
- **BLOCKER (precise):** per-shell pin flags are not printed (only aggregate `pins hi=N lo=M`), and s0's per-iteration
  field is not dumped (only the final `lumina_coevolve_field.csv`). So "genuine converged root" vs "pin_lo HOLD" cannot
  be separated from the run's own logs. A one-line runtime probe (`printf` of held/root + f_lo at s0 in
  `radeq_simul_all`) would settle it. **Either way** the committed 13120 coincides with the reconstructed **zero-pump
  root (13214 K)** (`radeq_coupledroot.py`), so the physical statement is unaffected: the run's deep T_e is the
  temperature at which [deposition = recombination + UN-pumped collisional line cooling].
- s1=13592, s2=13911 behave identically (smooth, stable) — same class. NOT the T_rad=10470.093 pin that s9-s12 sit on
  exactly (those are permanent no-root HOLDs at the initial T_e=T_rad).

## 2. Decomposition of the implemented balance at s0 (T_e=13120, run's own field)
`simul_r1` (lumina_plasma.c:4889-4931): `H = H_dep + Σ nion·Hex`  ;  `C = C_ff + C_ad + C_fb + Λ_line`.
The **only radiative heating channels are (a) bf photoheating above the ionization threshold and (b) the signed ETLA
line term** (which can heat via radiative pumping). The bath energy density `u` never enters as a heating term except
through these two. Exact/near-exact terms (erg/cm³/s), see `radeq_ledger_s0.csv`:

| term | value | % of H_dep | which J it consumes |
|---|---|---|---|
| H_dep (γ deposition) | 1.507e-3 | 100 | none — injected CMFGEN edep, `LUMINA_DEPOSITION_FILE` (`stdout.log:97,192`) |
| H_photo (bf) | 7.19e-7 | 0.05 | mc_J, **EUV<912 Å only** (Σσ_bf J(hν−χ)) |
| C_ff | 3.84e-6 | 0.25 | — |
| C_ad | 2.14e-8 | ~0 | — |
| C_fb (recomb) | 3.91e-4 | 26 | — (α·(χ+kT)) |
| Λ_line (un-pumped) | 1.064e-3 | 71 | Jb via `nlte_get_J_at_nu` = deterministic `nlte->J_nu` |

**Which term forces cold:** heating is 100% deposition (bf is 0.05%); the balance is `H_dep ≈ C_fb + Λ_line`. The gas
is cold because **line cooling reaches the deposition rate at a low T** — i.e. the deposition is radiated away by
un-pumped collisional line cooling, receiving no offsetting radiative line-pump heating.

**(2a) Does the heating integral see the super-thermal RED part of the bath? — NO.**
- The super-thermal excess (176 erg/cm³ above a·T_e⁴) is **not red — it piles up in the NUV 1290-3000 Å (227 vs 98
  Planck = 2.31×) and optical (1.43×)**, while the heating-relevant bands are STARVED: **FUV 0.34×, EUV 0.65× even a
  cold 13120 K Planck** (`radeq_ledger.py`). Bolometrically only −0.24 dex (u 400 vs 695), but the EUV band that bf
  taps is **−2 dex**.
- **Band-resolved absorptive heating fraction:** bf photoheating is 100% from EUV<912 (7.19e-7); the entire NUV+optical
  super-thermal excess (362 erg/cm³) contributes **0** to bf heating (below the abundant-ion thresholds / Kramers ν⁻³
  cutoff). Net radiative heating actually delivered to the gas ≈ H_photo = 7e-7 = **<0.05 % of the heating budget**.
  The 400 erg/cm³ bath is a **scattering/trapped reservoir that does not thermally couple** — consistent with the
  gas sitting 2046 K BELOW its own bath-equivalent T = (u/a)^0.25 = 15166 K.

**(2b) Deposition at s0:** present and dominant — 1.507e-3 (identical to CMFGEN's edep, injected verbatim), ≈2100×
the bf heating. Deposition is not the difference between the codes.

**(2c) Band truncation:** the ν-grid is 1.5e14-3e16 Hz (2 µm – 100 Å). Lines outside get Jb=1e-30 from
`nlte_get_J_at_nu` (`lumina_plasma.c:9290`) → far-IR (>2 µm) forbidden lines and hard-UV (<100 Å) lines cool
**un-pumped** (pumping truncated, not the collisional cooling). 99.9 % of the bath energy is inside the grid
(trapping audit), so the energy budget is intact; the effect is a mild over-cool via un-pumped far-IR forbidden lines.

## 3. Counterfactual root with CMFGEN's J (same balance formulas)
Coupled model (ionization re-solved vs T, LOWEST-root march = code; `radeq_coupledroot.py`), swapping ONLY the field:

| pump field into Λ_line | coupled root | anchor |
|---|---|---|
| zero-pump | 13214 K | ≈ run committed 13120 |
| thermal B(T_e) | 14818 K | detailed-balance floor |
| run mc_J | 15460 K | |
| **run cs.J** (own cooling field) | **16617 K** | |
| **CMFGEN jtable** | **18277 K** | ≈ CMFGEN truth 18760 (2.6 %) |

bf channel alone: H_photo 7.2e-7 (run) → 3.7e-4 (CMFGEN), ×520, entirely from the EUV band the run starves.

**Interpretation:** the balance formulas are responsive and correct — given CMFGEN's field they hit 18277 ≈ 18760.
So this is case **(a): faithful response to a too-red / heating-starved bath**, NOT case (b) a defective balance term.
The proximate mastermind is whoever reddens & EUV-starves the deep field (the emission / DIFFUSE_INNER_BC color
machinery — consistent with the campaign F3-T finding), plus the coevolve **split-field** architecture: the SAME
`nlte->J_nu` that is super-thermal for cooling (cs.J 1.55×) is bypassed for photoionization, which consumes the
UV-SUB-thermal mc_J (0.185× at 2500 Å, `radeq_diag.py`) — an internal inconsistency that both keeps IGE in III AND
leaves the deep gas at the un-pumped (coldest) root.

## Caveat on the +3400 K "un-pumped" lever
Whether that component is "faithful" or a residual defect hinges on one runtime fact I cannot read offline: does the
`nlte->J_nu` the T_e solve consumed at s0 actually carry cs.J's super-thermal NUV (dumped in the final CSV), or a
weaker/lagged field? If it truly is super-thermal, a correct two-level treatment WOULD pump the thick Fe/Co/Ni III UV
resonance forest to net heating (reconstruction: Λ flips −1.07e-3) and the physical root is 16617 K, making the run's
13120 K commit a ~0.5 dex under-heating (radeq-side). The delicate collisional-vs-radiative balance of τ~10⁴-10⁵ lines
makes the SIGN of that pumping sensitive to the exact rate coefficients, so I do not over-claim it. The robust,
assumption-light result stands regardless: **the balance reproduces truth given truth's field; the gas is cold because
its consumed field is heating-starved.**

---
## FINAL FORK
**balance faithful → mastermind = bath reddening / EUV-starved deep field.** Quantitative margin: the same
`simul_r1` formulas, fed CMFGEN's J, solve **18277 K vs truth 18760 K (2.6 %, +0.03 dex)** — so the formulas are not
the crime; the deep field is. The 13120→18760 deficit is field-side: ≥ +1660 K from spectral color (EUV/FUV −2 dex),
and up to +3400 K from the deep line cooling running un-pumped despite a nominally super-thermal cooling field (a
coevolve split-field / cold-branch-HOLD symptom, sign-sensitive — flagged, not asserted as a term defect).

## Artifacts (this directory)
- `radeq_ledger_s0.csv` — term-by-term ledger + band decomposition + root ladder
- `radeq_ledger.py` — bf/ff/ad/fb terms + band-resolved u and H_photo (run vs CMFGEN field)
- `radeq_cooling.py` — full 2.14 M-line ETLA Λ_line(T,Jb) reconstruction (exact VR_STD/beta_esc formulas)
- `radeq_diag.py` — per-line pump breakdown, cs.J/B(T_e) super-thermal check, exact-bin lookup
- `radeq_coupledroot.py` — coupled (ionization-re-solved) LOWEST-root ladder = the counterfactual table

## Source / data relied upon
- `src/lumina_plasma.c`: 4889-4931 (`simul_r1` balance), 4867-4887 (`simul_line_term`, NO no-pump guard →
  pumping allowed), 4580-4612 (VR_STD coeff, gbar=0.2), 5795-5799 (`radeq_beta_esc`), 5270-5519 (Gph mc_J blend),
  5562 (line-cooling Jb=`nlte_get_J_at_nu`), 5626-5667 (LOWEST-root + HOLD), 9176-9296 (J_nu norm + `nlte_get_J_at_nu`)
- `src/lumina_cuda.cu`: 4560-4574 (edep injected), 4798 (`cmfgen_write_jnu`→nlte.J_nu=cs.J), 4859
  (`compute_radiative_equilibrium_te` after write), 5250-5253 (nlte.J_nu never overwritten by MC), 5387-5397
  (field CSV: cs_J=cs.J, mc_J=nlte_Jmc)
- `logs/coevolve_consume_a10_kx_gphall/`: stdout.log (:97,108,192,216-217,242,4891-trace), lumina_plasma_state.csv,
  lumina_coevolve_field.csv, lumina_ion_pops.csv
- `data/tardis_reference_toy06_19p48d/deposition_cmfgen.csv` (H_dep), `data/cmfgen_jtable_toy06_19p48d.bin/.json`
  (CMFGEN J), `data/.../line_list.csv` (2.565 M lines), levels.csv, ionization_energies.csv
