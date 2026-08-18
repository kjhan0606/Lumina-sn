# Photospheric warm-loop bistability discriminator — kpr5

Offline reconstruction 2026-07-20. Read-only on `logs/`, no rerun/edit/GPU.
Run: `logs/coevolve_consume_a10_kx_kpr5/` (= kpr4 + **both** repairs: FLOORM mode=1 BKMAX=1000
[`LUMINA_NLTE_FLOOR_MODE=1`] clamping 161735 photospheric NLTE levels/iter, **and** pump-fallback
`LUMINA_RADEQ_PUMP_FALLBACK=1`). No `GPH_JTABLE`, no `TINNER_COLOR` — kpr5 uses its OWN mc field.
Benchmark: CMFGEN toy06 @19.48d. Every claim = number or `file:line`.
Ledger machinery ported verbatim from the validated `../residual_offset_budget/residual_ledger_kpr4.py`
+ `../te_bias_budget/recon_terms.py`, extended with the code's `simul_ladder` ionization re-solve
(`gph_of`, `alpha_full`=RR+DR, `resolve_ladder`) and the Boltzmann-weighted Gph the code actually
consumes at W<0.13. Shared 1000-bin log-ν grid (1.5e14–3e16) — all field swaps bin-exact.

---
## HEADLINE — **NO COLD ROOT. NOT BISTABLE.** The repair did not converge into the wrong basin;
**a cold self-consistent state does not exist with kpr5's field.** Fix ≠ initialization/annealing;
fix = the FIELD (the T-insensitive EUV excess). And the mission's *specific* warm-loop mechanism
(**optical** excited-level photoionization) is **REFUTED**: the photoionization is 100% EUV.

Across s8/s6/s2, both modes give a **single** residual root ≈ kpr5's own committed T_e, and the
coupled ionization stays **Fe IV-locked (f(FeIV) 0.998–0.9997) at EVERY T from 9000 to 22000 K**.
Cooling the gas — even 1400 K *below* CMFGEN — does not flip Fe back to III. There is no second basin.

---
## 0. Wiring verification (the mission's precondition) — **CONFIRMED**

At s8 `W=0.0389` (state CSV) < `LUMINA_STAGE4_GPH_WTHR=0.13`, so the depth gate
`plasma->W[s] > g_stage4_gph_wthr` (**`lumina_plasma.c:5763`**) is **FALSE** ⇒ `want_nlte_w=0`,
`use_nlte=0`. Control falls to the Boltzmann `else` loop (**`5870-5924`**), where the level weight is
```
double pop_l = (double)atom->level_g[l] * exp(-x_l) / U_ion;   // 5880
double x_l   = E_l / kT;   kT = K_BOLTZMANN * plasma->T_e[s];   // 5878, 5716
```
i.e. **pure LTE-Boltzmann at T_e — the NLTE populations are never read.** The FLOORM fix acts on
`nlte->nlte_level_populations` (the `use_nlte` path, `5808`); at s8 that path is skipped, so
**clamping 161735 NLTE levels/iter cannot move Γ(Fe III) at the photosphere.** This is exactly why
f(FeIV) s8 = 0.982 survived FLOORM (and every level-population fix). Same at s6 (W=0.054). At **s2
(W=0.134 > 0.13)** the NLTE path *does* fire (the one shell where FLOORM can touch Gph); the
discriminator uses Boltzmann weighting there per the mission's coupled-mode spec (physical
fixed-point question) — flagged in the round-trip below.

**Round-trip validation (Boltzmann Γ reproduces the committed ladder):**
| shell | committed r34 = n(IV)/n(III) | reconstructed r34 | ratio | gnt/Gph | α = RR + DR |
|---|---|---|---|---|---|
| s8 | 591.7 | 841.8 | **1.42×** | 0.00% | 2.84e-11 + 4.69e-12 |
| s6 | 598.4 | 829.6 | **1.39×** | 0.00% | 2.53e-11 + 4.54e-12 |
| s2 | 1946.7 | 8395 | 4.31× (W>0.13, code uses NLTE-weighted Γ here; Boltzmann over-ionizes — directional) |

s8/s6 reproduce the committed balance to <1.5× from scratch → the reconstruction is faithful. NT
ionization (gnt) is **0.00% of Gph** for Fe — the balance is pure radiative Gph/(n_e·α).

---
## 1. The discriminator r(T)=H−C at s8 (primary), s6/s2 (profile controls) — `rT_curves.csv`

`r(T) = H_dep + Σ_p n_p·Hex_p − C_ff − C_ad − C_fb(DBFB) − Λ_ETLA`, mirroring `simul_r1`
(`lumina_plasma.c:5076-5142`). Mode (a) freezes ion fractions at kpr5-committed (T only in Boltzmann
weights / C_fb Wien / C_ff / Λ); mode (b) re-solves the full `simul_ladder` at each T.

| shell | kpr5 T_e / CMFGEN | mode a root | mode b root | r(10383) | r(9000) | cold root ~10-11kK? |
|---|---|---|---|---|---|---|
| **s8** | 12208 / 10383 | 12240 | **12212** | **+4.89e-5** | +6.35e-5 | **NONE** |
| s6 | 13362 / 11929 | 13348 | **13336** | +1.61e-4 | +1.91e-4 | **NONE** |
| s2 | 19480 / 16351 | 19160 | **19387** | +1.24e-3 | +1.33e-3 | **NONE** |

The mode-b root lands on kpr5's **committed** T_e to <30 K at s8/s6 (r(T_e)=+1.7e-7, −2.3e-6) — the
ledger closes at the run state, validating faithfulness. **r(T) is monotone: positive for every
T below the warm root, negative above — one crossing.** No second (cold) zero at any shell.

**Coupled f(FeIV)(T) — the decisive, normalization-independent result:**
| shell | T=9000 | T=10383 | T=12208 | T=22000 |
|---|---|---|---|---|
| s8 | 0.9984 | 0.9986 | 0.9988 | 0.9997 |
| s6 | 0.9979 | 0.9983 | 0.9986 | 0.9996 |
| s2 | 0.9996 | 0.9996 | 0.9997 | 0.9999 |

**Fe stays 99.8–99.97% ionized to IV at ALL trial T.** Cooling to 9000 K (1.4 kK below CMFGEN)
changes f(FeIV) by <0.001. The self-consistent cold III-rich basin **does not exist** with kpr5's field.

**Why (Gph decomposition, s8 Fe III, kpr5 field):** Gph is essentially **T-insensitive** and **100% EUV**:
| T | Gph_tot [/s] | ground | excited | EUV(<912Å) | opt(>912Å) |
|---|---|---|---|---|---|
| 12208 | 27.37 | 6.56 | 20.82 | 27.37 | 2.4e-3 |
| 10383 | 26.30 | 7.34 | 18.96 | 26.30 | 1.3e-4 |
| 9000  | 25.50 | 7.96 | 17.55 | 25.50 | 6.9e-6 |

Gph drops only **7%** over 12208→9000 K, and the optical channel is **0.009%** — the "excited"
contribution is from levels whose *binding* energy is still >13.6 eV (they photoionize in the EUV),
NOT the near-threshold optical levels. **The mission's hypothesised optical excited-level warm loop
is refuted.** The needed Gph for CMFGEN's f(FeIV)=0.022 is ~7e-4/s (=r34_crit·n_e·α); kpr5's field
delivers **27/s ≈ 39000× too much**, and that excess is EUV and T-flat ⇒ no thermal escape route.

---
## 2. Which term keeps r(10383) > 0 — TWO faces of ONE field defect (mode b, T≈CMFGEN)

| shell | H_dep | H_photo(bf) | C_fb | Λ_line | C_ff+C_ad | **r** | radiated / (H_dep+H_ph) |
|---|---|---|---|---|---|---|---|
| s8 (10500) | 4.57e-5 | **+2.48e-5** | 4.5e-8 | +2.31e-5 | 1.7e-7 | **+4.72e-5** | 2.33e-5 / 7.05e-5 = 0.33 |
| s6 (12000) | 1.51e-4 | +4.09e-5 | 6.6e-7 | +9.46e-5 | 3.4e-7 | +9.65e-5 | 9.56e-5 / 1.92e-4 = 0.50 |
| s2 (16500) | 8.40e-4 | +1.77e-4 | 4.1e-6 | +3.84e-4 | 1.9e-6 | +6.27e-4 | 3.90e-4 / 1.02e-3 = 0.38 |

At cold T the gas radiates only ~⅓–½ of its heating input. Two guilty terms, both downstream of the
IV-lock:
- **(A) bf photo-heating that should not be there:** `H_photo = +2.48e-5` at s8 (54% of H_dep),
  injected by the EUV-excess field photoionizing Fe/IGE. In CMFGEN's own balance this term is 8.6e-6
  (§3) — a **~1.6e-5 excess**.
- **(B) III line coolant that fails to appear:** with Fe (and Co/Ni) IV-locked, the Fe/Co/Ni III
  resonance+forbidden line cooling is burned out; the surviving Λ cannot reach H_dep. CMFGEN at
  10383 K sits at f(FeIII)≈0.97 and radiates H_dep on exactly that III coolant.

Both are caused by the single T-insensitive EUV-excess field. **No ledger term is missing at cold T;
the ledger cannot reach a cold root because the FIELD holds the ionization at IV regardless of T.**

---
## 3. CMFGEN-state residual — is CMFGEN's own state a root of OUR ledger? — `cmfgen_state_residual.csv`

r evaluated at (T=CMFGEN T_e, CMFGEN jtable J, CMFGEN Fe/Co/Ni/Si/S/Ca fractions):
| shell | T | f(FeIV) [in] | H_photo | C_fb | Λ | **r** | as % H_dep |
|---|---|---|---|---|---|---|---|
| s8 | 10383 | 0.0219 ✓ | 8.57e-6 | 1.01e-5 | 1.05e-5 | +3.37e-5 | **+73.6%** |
| s6 | 11929 | 0.0694 | 7.37e-6 | 9.33e-6 | 4.16e-5 | +1.07e-4 | +71.1% |
| s2 | 16351 | 0.9935 | 6.30e-5 | 9.70e-5 | 2.09e-4 | +5.96e-4 | +70.9% |

CMFGEN's state is **not a clean root** in this offline reconstruction — a residual **net heating of
+0.71–0.74·H_dep**, remarkably uniform across three very different shells. The uniformity, and the
positive sign (cooling < heating even with III restored and CMFGEN's own EUV-quiet field), point to a
**systematic ~2.6× under-count in the offline radiative cooling** (dominantly the van-Regemorter
`gbar=0.2` ETLA line term `Λ`), NOT a per-shell physical term. **Caveat (honest, per kpr2 VERDICT):**
the offline reconstruction's *absolute* cooling magnitude is uncertain by ~2× (that verdict κ-pins it
to the committed root); +0.7·H_dep is inside that band, so **this test is inconclusive on its own** —
it does not confirm the 2.6% closure the `radeq_ledger_audit` found at s0 (inner), nor refute it. The
authoritative, normalization-independent result is the **IV-lock ratio** (§1), which needs no cooling
calibration: Gph/(n_e·α) = 39000× is a pure ionization-side number.

---
## 4. FORK VERDICT — **NO-COLD-ROOT** (not the bistable branch)

- (b) has **no** cold root at ~10-11 kK; the warm root sits at **12212 K (s8), 13336 K (s6), 19387 K
  (s2)** = kpr5's own converged T_e. ⇒ the +1.4…+3.1 kK offset is a **unique** fixed point, not a
  wrong-basin convergence.
- **Fix is NOT initialization/annealing.** A cold seed (or `LUMINA_TE_TABLE` pin) would relax straight
  back: at any T the frozen EUV-excess field re-photoionizes Fe→IV (Gph 39000× critical, T-flat),
  the III coolant stays burned, r>0, and the gas re-warms to ~12.2 kK. Seeding cannot create a basin
  that does not exist.
- **Guilty term = the FIELD's EUV excess** (the Gph/Hex consumer field, kpr5 mc `gphJ`), which
  simultaneously over-ionizes Fe→IV (burning term B) and over-heats bf (term A). This is the
  standing-campaign root (MC line-transport Co IV fluorescence funnel → EUV/FUV deficit is the
  *emergent* face; here it presents at depth as an EUV **excess** in the photoion-consumer field).

### Falsification / next-step probes (env-gated; spec only — offline)
1. **Direct field lever (already built):** `LUMINA_GPH_JTABLE=<cmfgen jtable bin>` (#33
   GRADIENT-TRANSPLANT, the `if (g_gph_jtable)` overrides at `lumina_plasma.c:5856-5859 / 5912-5915 /
   5981-5984`). Predicted: Gph(FeIII) s8 27/s → O(1e-3/s), f(FeIV) → CMFGEN 0.02, III coolant
   returns, root → ~10.5-11 kK. This is the discriminating test that the disease is the field.
2. **Pin-then-release falsifier (small code change) to CONFIRM no-cold-root:** the machinery to *pin*
   T_e from a table exists (`LUMINA_TE_TABLE` loader, `compute_electron_temperature`); a variant that
   pins T_e = CMFGEN profile for the first N iterations then releases (`LUMINA_TE_TABLE_RELEASE_ITER=N`
   guard around the T-solve call) would show — as this discriminator predicts — the gas **re-warming
   to ~12.2 kK within a few post-release iters** while f(FeIV) stays ~0.98. If instead it *stayed*
   cold, that would refute this verdict (revealing a genuine hysteresis/bistability the offline r(T)
   missed). Cheap, and it directly tests the fork.

**Do NOT pursue more ledger surgery or a T-annealing schedule as a fix** — both are downstream of a
field the local T-solve cannot correct.

---
## Artifacts (this dir)
- `VERDICT.md` — this file
- `bistability.py` — discriminator: Boltzmann Γ (code-faithful, W<0.13), simul_ladder re-solve,
  α=RR(Milne)+DR(parsed DR_TABLE), full simul_r1 energy ledger, modes a/b, CMFGEN-state residual
- `rT_curves.csv` — r_a(T), r_b(T), coupled f(FeIV)(T), all terms, 3 shells × 27 T (9000–22000 K)
- `validation_roundtrip.csv` — committed vs reconstructed r34 + α split + gnt
- `cmfgen_state_residual.csv` — r at CMFGEN's own (T, J, pops)
- `rT_bistability.png` — r(T) + f(FeIV)(T), s8/s6/s2
- `plot_rT.py`

## Source / data
- `src/lumina_plasma.c`: 5716/5763/5870-5924 (Boltzmann Gph consumer + WTHR gate), 5880 (pop_l LTE),
  5808 (NLTE path, skipped at W<0.13), 4992-5049 (`simul_ladder`), 5076-5142 (`simul_r1`),
  2827-2958 (`frozenin_alpha_rr` RR+DR), 4046-4327 (`DR_TABLE`; Fe IV→III = `{26,3,6,…}`:4133),
  5996-6009 (gnt), 6215-6217 (simul writes committed ion pops), 5856-5984 (#33 GPH_JTABLE overrides)
- `logs/coevolve_consume_a10_kx_kpr5/`: plasma_state, coevolve_field (cs_J,mc_J), ion_pops, levelpop,
  stdout (env footer: FLOOR_MODE=1/BKMAX=1000, PUMP_FALLBACK=1, GPH_WTHR=0.13, FROZENIN_DR=1; no GPH_JTABLE)
- `data/tardis_reference_toy06_19p48d/`: cmfgen_sigma_bf.bin, levels.csv, line_list.csv,
  ionization_energies.csv, deposition_cmfgen.csv; `data/cmfgen_jtable_toy06_19p48d.bin`;
  `data/standart_data1/toy06/ionfrac_{fe,co,ni,si,s,ca}_toy06_cmfgen.txt`
- prior: `../te_bias_budget/VERDICT.md` (kpr2, offline cooling ~2× caveat), `../residual_offset_budget/VERDICT.md`
  (kpr4 floor-lock + EUV bf-heat), `../radeq_ledger_audit/VERDICT.md` (CMFGEN-field root 18277 K @ s0, 2.6%)
