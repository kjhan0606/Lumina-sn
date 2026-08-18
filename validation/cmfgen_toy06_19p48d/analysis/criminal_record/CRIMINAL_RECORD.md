# THE CRIMINAL RECORD — kp_emiss / kpacket_cdf (the k-packet global re-emission CDF)

Offline forensic compilation, 2026-07-20. Read-only on logs/ and /gpfs. No source edit, no commit,
no GPU. Convicted mechanism established in `mastermind_test/VERDICT.md` (read first). Every attribution
below ties to a number or file:line. CMFGEN-divergence framing throughout.

**The convict.** `kp_emiss[dst] += n_lower·C_up·dE` (built plasma.c:2132-2135 & 2226-2229 from
non-SE dilute-Boltzmann populations incl. excluded stage-IV IGE), sampled as `kpacket_cdf`
(cuda.cu:3071-3079). p_kpacket(s0)=0.943 (94% of deep line activations thermalize into k-packets),
but re-emission is **99.98% LINES / 0.02% ff-fb continuum**, cross-ion (overall same-ion 17.2%),
concentrated into the locally-strongest emissivity complex. Signature classes S1 (per-shell pile),
S2 (cross-ion abs->emit), S3 (suppressed ff/bf continuum), S4 (cold radeq roots), S5 (mis-fed Gph),
S6 (escape-spectrum distortion).

---

## NEW EVIDENCE — corpse spot-check proves the crime is ARM-INVARIANT (serial), attractor floats with ionization

Ran the same-ion / attractor / continuum-share battery (`corpse_signature.py`, reuses the validated
mastermind pairing kernel) on two physically CONTRASTING corpses vs the canonical gphall:

| arm | ionization | ff+fb % of exits (S3) | deep s0-2 same-ion (S2) | DEEP attractor = Co IV | PHOT attractor = S III |
|---|---|--:|--:|--:|--:|
| **gphall** (arm B) | deep over-ionized (Co IV dominant) | **0.02%** | **37.1%** | **83%** of deep emit-E | **84-88%** |
| **gphground** (arm A) | deep UNDER-ionized (held in III) | **0.009%** | **38.1%** | **6.9%** | 26.6% |
| **evlog** (earliest, 2026-07-13) | (early) | **0.000%** | **38.4%** | 6.7% | 15.3% |

**Read-off (decisive):**
1. **The mechanism fingerprints are invariant across three arms with different ionization balances:**
   the k-packet re-emits **~0% ff/fb continuum** (S3) and the **deep cross-ion re-routing is ~62%
   (same-ion ~38%)** in *every* arm. The channel is architectural — it predates and transcends the
   gphall config (present already in the earliest evlog corpse). Serial crime, confirmed.
2. **The attractor IDENTITY is a slave to the local peak-emissivity ion, not a Co-IV-specific bug.**
   Arm A holds the deep gas in III (Co IV emission is only 6.9% of deep emit-E, self-recycle 0.8% vs
   arm B's 84.6%) — so when you move the ionization, the Co IV pile *moves with it*. This is exactly
   the VERDICT thesis ("the local peak-emissivity ion is the attractor"), now demonstrated by
   intervention: the CDF re-samples whatever ion locally dominates the emissivity.
3. The "overall same-ion" differs between arms (17.2% gphall vs 37.8% gphground) **only** because arm
   A's transport is deep-shell-dominated (s0-2 = 40.2M of 64M pairs vs gphall's 8.5M/63.9M) — a
   population-weighting shift, not a mechanism difference. The per-group deep/phot numbers are the
   physical ones and they cluster tightly.

Artifacts: `corpse_signature.py`, `corpse_signature.csv`, `corpse_signature.out`.

---

## TASK 1 — current-campaign damage inventory (12 rows)

Grades: CONVICTED (this criminal, with linking evidence) / AGGRAVATED (criminal contributes + other
factors) / INDEPENDENT (proven not this criminal). Full numbers + file:line in `crime_table.csv`.

| # | crime | victim | CMFGEN vs Lumina | grade | key linking evidence |
|---|---|---|---|---|---|
| C1 | deep FUV collapse | mc_J(918-1290,s0) | 83.60 vs 1.922 (-1.54dex) | **CONVICTED** | 42% of deep u in one 1508A Co IV bin; bolometric only -0.24dex => 1.30dex is spectral redistribution (trapping_audit); funnel = kp_emiss Co IV lines |
| C2 | dead Fe recomb gradient | f(FeIV) slope | +5.09 vs +0.65 dex | **CONVICTED** | field-folded Gph +6.67 vs +0.27 (n_e,alpha innocent <=0.3dex); flat field = kp_emiss EUV starve(deep)+S III(phot) |
| C3 | deep T_e cold | T_e(s0) | 18760 vs 13120 K | **CONVICTED** | radeq balance FAITHFUL (CMFGEN field->18277K=2.6%); cold bc field heating-starved; bf heating 0.05%; funnel-kill self-heats to 14585K |
| C4 | deep u low | u_bol(s0) | 694.8 vs 400.2 (ForkB->264) | **CONVICTED** | funnel tau~1e4 traps AND miscolors (double duty); CMFGEN u=695 = real bf/ff reservoir kp_emiss emits 0.02% of; ForkB drains 400->264 |
| C5 | EUV starvation | mc_J(300-450,s0) | 2.05e-2 vs 4.60e-6 (-3.65dex) | **CONVICTED** | EUV emitted deep but locally reabsorbed (F0a); ff/fb only 0.02% of exits |
| C6 | phot FUV excess + over-ion | mc_J(s8)+f(FeIV,s8) | 0.022 vs 0.461 | **CONVICTED** | 84-88% S III line; removing S III lands on CMFGEN +/-0.05dex; cross-ion 94% (donors Co III/Fe III) |
| C7 | Co III Gph deficit | Gph(Co III) | 22x low (twin 17.26x) | **AGGRAVATED** | twin(pinned field) residual = threshold cliff 33.5 vs 30.65eV x LTE b_k weighting (INDEP of kp_emiss); real-run kp_emiss EUV -3.65dex aggravates |
| C8 | unfilled deep valley | mc_J(1650-2100,s0) | mc/CMFGEN -1.55dex | **CONVICTED (downstream)** | valley absorbers Co III/Fe III scatter a field 1.5dex too faint; B-run 89% up-pumped into Co IV pile |
| C9 | T_rad pin uniform | T_rad all shells | uniq=1 @10470K | **INDEPENDENT (instrument)** | MC estimator collapse; does NOT feed Gph(mc_J) nor radeq s0-2; BUT = the temp freezing stage-IV pops kp_emiss samples (aggravating coupling) |
| C10 | Co rate ~10x low | f(Co IV) twin | Co 5-20x under-ion w/ T+J pinned | **INDEPENDENT (of kp_emiss)** | twin reproduces Fe fully but Co still low => Co rate-side (threshold cliff+weighting); survives correct field |
| C11 | deep Co IV 1500A pile | mc_J(1290-2000,s0) | Co IV=80.9% of deep emit-E | **CONVICTED (crime scene)** | mc/cs 39x @1526.17A (lid 391357); level144 eweight e_low/hnu=9.3x; NO thermal exit (p_kpkt=8.1e-10, MACROATOM_BF off) |
| C12 | split-field mc_J vs cs_J | Gph(mc_J)/cooling(cs_J) | band 7-77x divergence | **MECHANISM signature** | mc_J = MC-transported reddened image of cs_J; the divergence IS kp_emiss recycling(mc) vs B(Te) fallback(cs) |

**Special calls the driver flagged:**
- **C7 Co III Gph** — the trace's threshold+LTE-weighting causes are real and INDEPENDENT (they survive
  in the twin with CMFGEN's own field). But kp_emiss is upstream of the *observed flat* Gph in the real
  run via EUV starvation (C5). Verdict: **AGGRAVATED** — kp_emiss supplies the −3.65 dex field starvation
  at the Co III edge; the residual (measured with the field pinned) is a genuinely independent
  atomic-structure × Gph-weighting defect.
- **C9 T_rad pin** — **independent instrument issue** (MC T_rad moment-estimator collapse), NOT a driver
  of the main channels (mc_J drives Gph; T_e roots drive radeq; only s9-12 sit on the pin). But it is
  *coupled in*: T_rad=10470 is the temperature at which the excluded stage-IV Co IV levels are frozen
  in dilute-Boltzmann (coiv_funnel step 2), and those biased pops are exactly what kp_emiss's CDF
  samples. Independent defect that feeds the criminal.

---

## TASK 2 — historical cold cases (serial-crime sweep)

Grades vs the signature classes. CONVICTED / PROBABLE / POSSIBLE / UNRELATED.

| case | recorded symptom | kp_emiss signature? | grade | reason / corpse evidence |
|---|---|---|---|---|
| (a)(d) fluorescence gap "UV 51 vs 23" + Kromer S III | emergent UV 42.9-51.6% vs 23.3-23.8%; 52% S III line | **S6 + S2** (photospheric attractor) | **CONVICTED** | kromer: emergent UV 100% line, S III 52%; DR_BOOST falsifier => S III is the funnel-EXIT (blanket/re-emitter), donors Co III/Fe III (=axis2 today); "ALGORITHM not DATA" (ARTIS algo+Lumina data 20.2% vs native 42.9%) = today's "데이터 무죄, 알고리즘 유죄" same criminal |
| (g) ARTIS macro-atom "3-month wall" (Div2) | UV->UV 99.5% (KPACKET on) vs 28.9% (off) | **S6** (k-packet re-emit) | **CONVICTED** | Div2 IS the criminal: SAME p_kpacket unit-mismatch (energy-weighted sum_rates vs bare kp_deact) flagged then at plasma.c:1886, today plasma.c:2214-2219. Direct earlier sighting, mis-attributed to the re-inject loop rather than the CDF content |
| (ancestor) missing k-packet thermal pool (2026-06-04) | over-redshift, pure radiative down-cascade | — (pre-channel) | **CONVICTED (ancestor)** | macro-atom was PURELY RADIATIVE, no thermalization at all; the FIX (add k-packet) INTRODUCED today's convict (a k-packet that re-emits biased-CDF lines, not continuum). Crime evolution: "no thermalization" -> "counterfeit thermalization" |
| (f) super-thermal S_l | S_l/B 1947 (top III) vs 1.01 (I/II) | S1/S5 architectural cousin | **PROBABLE** | top NLTE stage (III) excited levels have NO bf continuum anchor -> non-thermal pops -> super-thermal S_l. SAME "top-stage excluded from continuum SE" architecture that today excludes stage-IV IGE and biases kp_emiss's CDF. Deterministic-side manifestation; feeds the criminal's populations; distinct code path (TOPSTAGE_IV fix) |
| (i) IR NLTE fluorescence too-red | S_l/B=1.0000; peak 9202 vs 6590; UV over-prod | mixed | **POSSIBLE (with prophecy)** | dominant root = binned-J thermal S_l + inner-BC UV starvation (source-side, DDC15). EXPLICITLY flagged the k-packet risk 2026-06-20: "if too much line energy routes to k-packets at cool T_e it re-thermalizes to IR exactly like now." k-packet was OFF/capped in those tests; the fluorescence over-production sub-symptom is kp_emiss-family |
| (h) inner n_e FUV thermalization | inner n_e -0.27dex; Gamma(Mg I/Si I) 5-1000x low | cs-side twin of the coverage gap | **POSSIBLE** | non-NLTE line forest gets THERMAL source B(T_e) in cmfgen_assemble -> FUV over-thermalized. The cs-side TWIN of kp_emiss (cs OVER-thermalizes; MC UNDER-thermalizes; same "non-NLTE line -> wrong source function" root). Distinct code path (deterministic formal) |
| (b) too-red / binned-J root | SED peak 6595 vs 9200A (+2570A) | mostly not | **POSSIBLE (mostly UNRELATED)** | DOMINANT lever = binned-J continuum field grey collapse (no freq contrast, cs-side, DDC15). k-packet was OFF in these runs. Fluorescence sub-component (~10% of gap) is kp_emiss-family; the +2570A dominant too-red is a separate field defect (frequency-resolved-field fix) |
| (c) MC blue-tilt / T_inner drift | MC continuum ~20% blue-tilt | none | **UNRELATED** | PROVEN observer-frame Doppler (MC correct; falsifier t_exp x1000 -> tilt 1.000). T_inner-controller-drift hypothesis REFUTED. Real bug = CMF formal observer-frame transform missing. Not kp_emiss |
| (e) far-outer hot band death / EPAY | outer T_e runaway 289% | none (physical RE) | **UNRELATED** | far-outer thermal instability (heating~rho vs cooling~rho^2) + forbidden-line Omega cooling gap + shell self-illumination fixed point. Radiative-equilibrium physics. (NB: the fluorescence gap that co-lives in this saga file = case (a), CONVICTED — do not confuse the two) |

---

## TASK 3 — repair traceability (per-crime validation checklist)

Upstream single-physics-mode repair components (mastermind_test/VERDICT.md §Task3):
(i) **stage-IV SE promotion** — put Co IV/Fe IV/Ni IV (and top stages of S/Si etc.) in the SE/NLTE set
    so kp_emiss is built from true SE populations and its emission relaxes toward B(T_e).
(ii) **ff/fb continuum branch restoration** — raise the k-packet continuum branch (p_ff+p_fb,
    plasma.c:2237-2269) to its physical thermal fraction so the pool emits a real ff/fb continuum.
(iii) **B(T_e) clamp where continuum-thick** — clamp the k-packet line source to B(T_e) in the deep,
    optically-thick region.

| crime | cured by | validation gate (pre-registered) |
|---|---|---|
| C1 deep FUV | (i)+(iii) | mc_J(918-1290,s0) -> ~2e-4 (from 1.9e-6); Co IV pile share collapses; but needs T_e converge (see residuals) |
| C2 Fe gradient | (i)+(iii) | f(FeIV,s8) 0.46 -> ~0.02; slope s0->s8 +0.65 -> +5.09 (jtable already restored +4.3 with correct field) |
| C3 deep T_e | (i)+(ii)+(iii) | radeq root climbs 13120 -> toward 18277 (LTHERM/ForkB already self-heat to 14080-14585K on funnel-kill) |
| C5 EUV | (i)+(ii) + **selective** (iii) | mc_J(300-450,s0) recovers; CAVEAT: full S=B (LTHERM) killed non-thermal EUV -1.9dex — clamp must be selective |
| C6 phot FUV / f(FeIV) | (iii) + S top-stage SE | mc_J(918-1290,s8) 5.68e-6 -> ~7e-7; F4 (LUMINA_A2_SIII_FUV_THERM) is the per-ion stopgap |
| C8 valley | deep-field amplitude (via i+ii+hot T_e) | valley auto-fills once deep field brightens; NOT (iii) — CMFGEN valley is scattering-dominated 4.76xB, clamp is wrong physics |
| C11 Co IV pile | (i)+(iii) | pile emission-share 96% -> ~30%; mc/cs 39x -> ~1 (ForkB already achieved 39->1.9) |
| C12 split-field | (i) | mc_J and cs_J converge once kp_emiss is thermally consistent |
| H1/H2 fluorescence/S III | (iii) + S top-stage SE | emergent UV 51% -> ~23%; = C6 one epoch removed |
| H3 ARTIS re-inject | (ii)+(iii) (+KPACKET_ONESHOT for the loop) | UV->UV 99.5% -> physical; ff/fb share up |
| H5 super-thermal S_l | (i)-analog = TOPSTAGE-IV continuum node | S_l/B(top III) 1947 -> O(1) w/ T_e/n_e unchanged (design: TOPSTAGE_IV_CONTINUUM_NODE_DESIGN.md) |

### Crimes that require something ELSE (pre-register these residuals so the repair is not over-claimed)

1. **C4 u amplitude** — killing the funnel color DRAINS u (measured 400 -> 264 under Fork B alone).
   Restoring u=695 needs component **(ii) the real bf/ff continuum reservoir + a genuinely hot gas**,
   NOT a per-ion output patch. Fork B/F4 (per-ion stopgaps on the channel OUTPUT) cannot supply amplitude.
2. **C7 + C10 Co rate** — kp_emiss repair fixes the EUV *field* but NOT the Co III/IV rate residual
   (threshold cliff 33.5 vs 30.65 eV × LTE b_k weighting). Needs a **SE-population-weighted Gph** for
   the promoted ions — a distinct rate-side fix outside (i)/(ii)/(iii). Twin proves it survives the
   correct field.
3. **C8 valley amplitude** — needs deep-field brightening, not a B(T_e) clamp (clamp gives the wrong,
   thermal physics for a scattering-dominated valley).
4. **C9 T_rad pin** — separate T_rad estimator fix (independent instrument). Correct stage-IV SE pops
   remove its *downstream* harm (the biased CDF), but the uniform-pin diagnostic itself is untouched.
5. **T_e convergence** — the N30 (30-iter) Fork-B run plateaued at T_e(s0)=14524K (< 18kK) and u drained
   to 264: 12-30 iterations are insufficient. Residual is CONVERGENCE (NITER), not missing physics —
   but must be demonstrated, not assumed.
6. **Historical (b)/(c)/(e)** — need their OWN repairs (frequency-resolved continuum field; CMF
   observer-frame transform; far-outer forbidden-line Omega + self-illumination). The kp_emiss repair
   will NOT touch them; pre-register no improvement there.

---

## Files (this directory)
- `crime_table.csv` — machine-readable: crime x victim x shells/bands x grade x linking evidence x repair.
- `corpse_signature.py` / `.csv` / `.out` — arm-invariance spot-check (gphground arm A + evlog earliest).
- `CRIMINAL_RECORD.md` — this file.

## Evidence base relied upon (all pre-existing, this campaign)
mastermind_test/VERDICT.md (same-ion discriminator, p_kpacket 0.943), reddening_localization/VERDICT.md
(Co IV pile, taskB), coiv_funnel_trace/VERDICT.md (level144, no thermal exit), axis2_valley_forensics/
VERDICT.md (S III, valley), radeq_ledger_audit/VERDICT.md (balance faithful), trapping_audit/VERDICT.md
(u, tau, forest census), GRADIENT_BUDGET_VERDICT.md (field carries 6.40dex), F0_F1_FUV_GRADIENT_VERDICT.md,
splitfield_audit, co3_closure_trace/final_ledger.csv, crime_reconstruction/part{1,3}. Historical:
memory project_{macroatom_missing_kpacket_thermal_pool, macroatom_artis_diff_breakthrough,
fluor_reprocessing_root, kromer_sulfur_uv_reframe, super_thermal_sl_2reviewer, inner_ne_fuv_thermalization,
toored_rootcause_ladder, toored_ir_fluorescence_rootcause, mc_tinner_bluetilt, farouter_5layer_audit_fixplan}.
