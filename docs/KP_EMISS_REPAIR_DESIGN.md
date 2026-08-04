# kp_emiss / kpacket_cdf — the ONE-PHYSICS upstream repair (DESIGN)

Offline design, 2026-07-20. DESIGN-ONLY: no source edits, no runs, no commit, read-only on
`logs/`. Convicted mechanism established in `mastermind_test/VERDICT.md` +
`criminal_record/CRIMINAL_RECORD.md` (read first). Every mechanism claim carries a file:line;
every number is an event-log measurement or an offline computation in
`validation/cmfgen_toy06_19p48d/analysis/kp_emiss_repair/` (scripts + CSVs shipped alongside).

Composes with `docs/STAGE4_ROUND2_DESIGN.md` (depth-gate+cap Gph, Ni/Ti fixes) and the Fork B
(`LUMINA_LINE_BSRC`) N30 outcome (u 400→264). Reported as a CMFGEN-divergence map, not a gate list.

---

## 0. The single decisive offline result (read this first)

I rebuilt `kp_emiss` at s0 two ways — dilute-Boltzmann (the convicted build) and the stage4 run's
SE/NLTE pops (`kp_emiss_cdf_rebuild.py`, reproducing plasma.c:2095-2135 line-for-line). The
reconstruction **self-validates**: my offline `C_ff`, `C_fb`, `tot` reproduce the event-log exit
shares to 2 sig figs.

| quantity (s0) | offline (this design) | event-log corpse (record) |
|---|--:|--:|
| free-free exit share `C_ff/(C_ff+C_fb+tot)` | **0.0066 %** | 0.007 % (Task 1 path 3) |
| free-bound exit share `C_fb/(…)` | **0.0155 %** | 0.015 % (Task 1 path 4) |
| ff+fb continuum share | **0.0220 %** | 0.02 % (S3 signature) |

Then the two headline numbers that **reframe the whole repair**:

| s0 rebuild | dilute-Boltzmann (convict) | SE pops (component i) | Δ |
|---|--:|--:|--:|
| line-emissivity `tot` (=C_collexc) [erg cm⁻³ s⁻¹] | 0.1842 | 0.1987 | **+8 %** |
| ff+fb continuum share | 0.0220 % | **0.0204 %** | **down, not up** |
| deep attractor = Co IV (ion 27_3) emission share | 91.36 % | **91.12 %** | unchanged |

**Read-off.** Feeding `kp_emiss` the SE pops **does not** raise the continuum share and **does not**
move the attractor off Co IV. This is not a surprise once seen: at s0 the ion densities are the same
in both builds (nlte_solve only redistributes *within* an ion), and CMFGEN itself is Co-IV-dominated
deep (f(CoIV)=0.993, `STAGE4_ROUND2 §0`). **The deep pile is not a wrong-pops artifact — deep Co IV
dominance is physically correct. What is wrong is that the channel re-emits Co IV's *resonant line
forest* instead of a photon from the thick-zone *thermal source function*.** That single fact sets
the entire staging below: component (i) is a prerequisite (and a photosphere fix), but the deep-pile
/ FUV / u cure is the source-function repair (iii), backed by a real continuum channel (ii).

---

## 1. Mechanism recap + the traffic map

### 1.1 What builds `kp_emiss`, and when (per-iteration)
`kp_emiss[dst] += n_lower·C_up·dE` is built inside **`compute_transition_probabilities`**
(`plasma.c:1449`), at the up-transition branch **`plasma.c:2110-2135`**; the per-shell CDF is
normalized at **`plasma.c:2226-2229, 2365-2371`**. It is sampled on the GPU at
**`cuda.cu:3071-3079`** (k-packet re-excite) and again at **`cuda.cu:3998-4008`** (bf-abs with no
mapped level routes to the *same* CDF).

Per-iteration call order in the coevolve loop (`cuda.cu`):
`compute_radiative_equilibrium_te` (5144, builds Gph) → `compute_plasma_state` (5148, ion densities)
→ `nlte_solve_all_gpu` (5159, the III/IV **SE drain**) → **`compute_transition_probabilities`
(5226, builds kp_emiss/CDF)**. So — unlike Gph, which runs *before* the drain and lags one iteration
(`STAGE4_ROUND2 §1.1`) — **kp_emiss is rebuilt AFTER the drain and reads the current-iteration ion
densities.** The defect is not staleness: it is that the level distribution is re-synthesized as
**dilute-Boltzmann** (`plasma.c:2117-2126`: `n_lower = n_ion·wgt·g_lo·exp(−E/kT_rad)/Z`,
`wgt = meta?1:W`, `Z` from `compute_partition_functions` :494-527) instead of the SE `n_k` that
`nlte_solve_all_gpu` just produced. (Legacy loop mirror: build at `cuda.cu:6692` after drain 6640;
also `cuda.cu:6211`, `main.c:597`.)

### 1.2 The line-vs-continuum split at the k-packet exit
On k-packet formation (`cuda.cu:3047`), the exit is drawn (`cuda.cu:3058-3070`):
`p_ff → −2` (free-free continuum), `p_fb → −3` (free-bound continuum), else → sample `kpacket_cdf`
→ re-excite → cascade → **line**. The split is set at **`plasma.c:2267-2269`** (run's active
Kramers branch; FB-MULTI print absent in the stage4 stdout):
```
p_ff = C_ff /(C_ff + C_fb + tot),   p_fb = C_fb /(C_ff + C_fb + tot)
C_ff = 1.426e-27·√T_e·n_e·Σ_{Z≥1} Z²·n_ion                         (plasma.c:2265)
C_fb = Σ_{Z≥1} [2.6e-13·Z²·(T_e/1e4)^−0.75]·n_ion·n_e·kT_e         (plasma.c:2253-2254)
tot  = Σ_lines n_lower·C_up·dE  = C_collexc                        (plasma.c:2132, 2229)
```
All three terms are **erg cm⁻³ s⁻¹ emissivities** — the split is *dimensionally sound* (§2.2).

### 1.3 Traffic map (s0 deep; p_kpacket=0.943, event-log + offline)

| stage | path | file:line | share of deep activations | emits |
|---|---|---|--:|---|
| A. thermalize? | k-packet roll (p_kpacket) | cuda.cu:3047; plasma.c:2214-2219 | **94.3 %** | → stage B |
| A′. else | radiative block-walk (per-ion, manifold-confined) | cuda.cu:3099-3147 | 5.7 % | Co IV line (same-ion residue) |
| B1 | k-packet → free-free `−2` | cuda.cu:3061-3064, 3358-3388 | 94.3 %×0.0066 % | thermal ff continuum |
| B2 | k-packet → free-bound `−3` | cuda.cu:3066-3069, 3389-3444 | 94.3 %×0.0155 % | recomb-edge continuum |
| B3 | else → `kpacket_cdf` re-excite → cascade | cuda.cu:3071-3079 | 94.3 %×**99.978 %** | **Co IV 1490–1650Å forest (91 % of emit-E)** |
| (feed) | bf-abs, no mapped level → same CDF | cuda.cu:3998-4008 | (etype3 feed) | as B3 |
| downstream | Fork B refreq of flagged Fe/Co/Ni IV line → Planck(T_e) | cuda.cu:3221,3254,3345,3458 | catches A′ **and** B3 emits | thermal (color only) |

The mastermind number `same-ion 37 %` deep and `p_kpacket 0.94` are the same story: ~94 % of deep
activations pass through stage B, of which 99.978 % re-emit the globally-CDF-sampled Co IV forest;
only 0.022 % leave as a genuine continuum photon.

---

## 2. Per-component analysis (with the quantified offline results)

### (i) Rebuild `kp_emiss` from SE populations — **prerequisite + photosphere fix, NOT the deep-pile cure**
- **Code change (one branch):** at `plasma.c:2117-2126`, replace the dilute-Boltzmann `n_lower`
  with the SE level population `n_l = atom->level_pop[glo·n_shells+s]` when the lower level's ion is
  in the NLTE/SE set (i.e., promoted by round-2 STAGE4). Keep the dilute-Boltzmann fallback for
  ions that remain non-SE, so the change is null where no SE pop exists.
- **Offline quantification (s0, `band_shares_s0.csv`, `ion_shares_s0.csv`, `continuum_split_s0.csv`):**
  the SE rebuild raises `tot` 0.1842→0.1987 (line cooling *up* 8 %), so the continuum share
  **drops** 0.0220 %→0.0204 %, and the Co IV attractor is **unchanged** (91.36 %→91.12 %). The
  collisional-excitation *input* band-shares redden slightly (opt 3200–7000Å 69.8 %→62.4 %,
  IR>7000Å 9.96 %→18.1 %; the FUV pile bands stay <0.1 % in *both* — the exp(−hν/kT_e) factor
  Boltzmann-suppresses the UV *input*). The emitted UV forest is set by the **destination ion**
  (Co IV, intra-ion bb) — which SE does not move at s0.
- **Why (i) is still required:** (a) it is the single-physics-mode source fix — the pops the CDF
  reads should be the pops the NLTE solve produced (removes the "non-SE stage-IV IGE" bias the record
  convicts); (b) it is load-bearing at the **photosphere**, where the S III attractor lives and where
  the round-2 depth-gate corrects the ionization that sets *which* ion the CDF locks onto. The s0
  no-op is precisely the evidence that the deep pile is a source-function problem, not a pops problem.
- **Verdict:** ship (i), but do not expect it to drain the deep pile or refill FUV on its own. It
  composes with round-2 (same SE pops) and is the clean substrate on which (iii) acts.

### (ii) ff/fb continuum branch — **the 0.02 % is a MODELING GAP, not a rate BUG**
- **Is it the case-14 dimensional bug?** No. Unlike the `p_kpacket` denominator (`plasma.c:2214`,
  which mixes energy-weighted `sum_rates` with the bare `kp_deact` — the mastermind's unit-mismatch),
  the `p_ff/p_fb` denominator adds three *consistently* energy-weighted emissivities (all
  erg cm⁻³ s⁻¹, §1.2). The split is dimensionally clean.
- **What SHOULD the fraction be at s0?** With the loaded pops at T_e=13116 K, n_e=4.89e9, the
  IGE line (collisional) cooling genuinely dominates: `tot/(C_ff+C_fb) ≈ 4500`. That is *correct*
  local-cooling physics — at 10⁴ K a Fe/Co-rich gas cools through its line forest, not through
  ff+fb. **So 0.02 % is approximately the true local cooling partition; it is not a wrong number to
  be "raised."** (Caveat: the run's Kramers `C_fb` under-estimates the real Milne/`frozenin_alpha_rr`
  recombination cooling; even a ×10 correction lands ff+fb at ~0.2 %, still ≪ line cooling — the
  verdict is robust. The FB-MULTI path `plasma.c:2288-2358` already swaps in `frozenin_alpha_rr` when
  `LUMINA_KPKT_FB_MULTI=1`; it was OFF in this corpse.)
- **So where is the crime?** The gap is two-fold and both are *architecture*, not rate inputs:
  1. **No thermalized-continuum FIELD.** CMFGEN's deep u=695 is not a claim that local emission is
     mostly continuum — it is that the deep zone is continuum-thick, so the field *between* the
     (also thermalized) lines has a bf/ff continuum floor → the escaping-from-deep field ≈ B(T_e).
     The k-packet re-emits an *individual resonant line* (τ~1e4, `C11`), which traps and reprocesses
     into the Co IV pile — there is no continuum floor to thermalize the inter-line field, so FUV/EUV
     between the forest lines stays dark (`C1/C5`).
  2. **The fb exit is a caricature.** It emits at one/few dominant edges with a `kT_e`-only tail
     (`FB-COOL-KT`, `plasma.c:2308-2314`) and no per-level Milne SED; it cannot build the true
     recombination-continuum shape even at its 0.015 % weight.
- **Verdict:** (ii) is a **feature-add** (install a genuine, edge-resolved bf continuum *photon*
  channel — FB-MULTI + real `frozenin_alpha_rr` `C_fb` — so the deep field acquires a continuum
  floor), NOT a one-line fraction fix. Necessary for the **u amplitude** (`C4`) and EUV field
  (`C5`); insufficient alone for color (that is iii).
  - PLACEHOLDER ➊ — **CMFGEN deep continuum share (from ETA_DATA when it lands):** the fraction of
    deep (sub-3900Å) emissivity that is bf+ff continuum in CMFGEN toy06 @19.48d at s0. Fill:
    `________`. This calibrates whether FB-MULTI's continuum weight needs an explicit floor, or
    whether (iii) (source-function thermalization) is doing the real work.

### (iii) B(T_e) clamp where continuum-thick — **the physically-correct deep repair, not merely a stopgap**
- **Machinery:** LTHERM already re-emits from a Planck(T_e) draw
  (`d_ltherm_reemit`/`d_sample_planck_frequency(d_ltherm_te)`, `cuda.cu:3451-3453, 2568`). The
  clean single-channel form is a **new k-packet exit type** (a `−4` sibling of `−2/−3`) that, when
  the shell qualifies, draws the re-emission frequency from B(T_e) instead of the `kpacket_cdf`
  line — i.e. sample the *thick-zone source function* directly at the point of re-excitation
  (`cuda.cu:3070`, before the CDF binary search).
- **Why it is *correct*, not a hack, in the thick zone:** where the pops are trustworthy (deep,
  Co IV b_k(144)=0.888 at s0, `STAGE4_ROUND2 §4.2`), S_line→B(T_e) *and* S_cont→B(T_e); a photon
  drawn from B(T_e) is exactly what both should emit. This is why Fork B (Planck(T_e)) already kills
  the funnel color (mc/cs 39→1.90) — it is sampling the right source function. (iii) does the same
  thing **at the source** (the 94 % k-packet traffic) instead of at the 4 output sites.
- **Scope / criterion (must be selective — the C5 caveat):** full S=B (LTHERM everywhere) killed the
  non-thermal EUV −1.9 dex (`crime_table C5`). So gate on a **measured continuum thermalization
  depth**, not on all shells: qualify shell s when the local **continuum optical depth to the surface
  is ≥ a threshold** (τ_cont(s) ≳ 1). Operationally, reuse the same depth proxy round-2 adopts —
  `W(s) > W_thr` with `W_thr≈0.13` (s0–s2), the region where `STAGE4_ROUND2 §0` shows the field is
  continuum-thick and CMFGEN f(IV)>LTE. This keeps the clamp off the dilute outer/photospheric shells
  where the non-thermal EUV/line structure must survive.
- **Why it is a *stopgap only if applied alone*:** clamping to B(T_e) supplies **color** (kills the
  Co IV pile) but not **amplitude** — B(T_e) at a cold deep gas (13.1 kK) is faint; u recovers to 695
  only when the gas also *heats* (radeq root climbs, `C3`) and the continuum reservoir (ii) exists to
  hold the field. (iii) without (ii) reproduces the N30 drain (400→264). Hence the composition below.
  - PLACEHOLDER ➋ — **CMFGEN FUV formation depth:** the velocity/shell at which the emergent 918–1290Å
    flux forms (τ_λ(FUV)≈1 surface) in CMFGEN toy06 @19.48d. Fill: `________`. Sets whether the
    τ_cont gate for (iii) must extend past s2 to cover the FUV forming layer.

### Fork B (BSRC) — subsumed at the deep shell; retire only after the no-Fork-B gate passes
- **Traffic split (measured):** Fork B intercepts the *final emitted line frequency* of flagged
  Fe/Co/Ni IV lines at **4 emission sites** (`cuda.cu:3221, 3254, 3345, 3458`), which sit
  **downstream** of both the block-walk (A′, 5.7 %) and the k-packet cascade (B3, 94.3 %×99.978 %).
  `kp_emiss` gates the **routing** upstream (94.3 % of deep activations enter stage B).
- **Does (iii) subsume Fork B?** For the **94 %** that route through the k-packet exit: yes — if the
  k-packet re-emits B(T_e) directly (type −4), that traffic never reaches a Co IV bb-emission site,
  so Fork B has nothing to catch there. The only residue Fork B still covers is the **~6 % pure
  block-walk** (A′) Co IV emission. So the upstream repair collapses Fork B's job from "the whole
  deep pile" to "a 6 % same-ion residue."
- **Recommendation:** stage the kp_emiss repair with **Fork B still ON** (belt-and-suspenders), then
  run the **A/B toggle**: with (iii) on, turn `LUMINA_LINE_BSRC` OFF and confirm the funnel stays
  dead (G-FUNNEL below). If it does → retire Fork B (one physics mode, per NO-OVERFITTING). If the
  6 % residue re-piles → keep a **depth-scoped** Fork B (same τ_cont gate) as the cheap safety net.
  Do **not** add BSRC mode-3 (sample S_l) — `STAGE4_ROUND2 §4.2` shows it re-opens the funnel.

---

## 3. Composed repair — recommendation & staging

**One physics mode, one gate, three composable knobs (all default OFF = round-1 behavior).**
The single mechanism is: *make the k-packet channel re-emit the thick-zone thermal source instead of
the resonant line forest, from correct SE pops, with a real continuum floor.*

```
LUMINA_KPEMISS_REPAIR=0        # master gate (0 = convict build; unset everywhere = no-op)
  └─ knob (i)  LUMINA_KPEMISS_SE_POPS=1        # n_lower <- SE n_k for promoted ions (plasma.c:2117-2126)
  └─ knob (ii) LUMINA_KPKT_FB_MULTI=1          # real frozenin_alpha_rr C_fb + edge-resolved fb SED (EXISTING gate, plasma.c:2282)
  └─ knob (iii)LUMINA_KPEMISS_BSRC_TAU=0.13    # B(T_e) k-packet exit (type -4) where W(s)>this (0=off); reuses d_ltherm_te machinery
```

**Staging (attribute cleanly — do NOT stack a second thermal-exit A/B):**
1. **S0 — substrate.** Land round-2 first (SE pops correct, no blowup): `STAGE4_ROUND2 §5` env
   (`GPH_NLTE_WMIN=0.13`, `GPH_BK_CAP=1000`, `SIMUL_CAP_TOPION=1`, drop Ti). kp_emiss reads these
   pops at `cuda.cu:5226` (after the drain) automatically.
2. **S1 — (i) alone.** `KPEMISS_SE_POPS=1`. Pre-registered null-ish at s0 (§2i); confirm no
   regression and that the *photospheric* attractor ion tracks the corrected ionization.
3. **S2 — (iii)+(ii) together, Fork B ON.** Add `KPEMISS_BSRC_TAU=0.13` and `KPKT_FB_MULTI=1`.
   This is the amplitude+color repair. Expect deep pile to collapse and u to climb (needs S3 too).
4. **S3 — NITER + Fork-B retirement A/B.** Run to ≥30 iters (the N30 residual is convergence, not
   physics — `CRIMINAL_RECORD residual 5`). Toggle `LUMINA_LINE_BSRC` OFF; funnel must stay dead
   (G-FUNNEL). Retire Fork B iff it does.

**Ordering / shared gates with round-2:** the SE promotion (round-2) MUST precede knob (i) — same
pops. The τ_cont depth gate for (iii) shares round-2's `W_thr≈0.13` depth map (`STAGE4_ROUND2 §0`),
so s0–s2 is the single continuum-thick region for both the Gph weighting and the B(T_e) clamp — one
depth criterion, no second free parameter. (ii) reuses the existing FB-MULTI gate untouched.

---

## 4. Validation card (inherits the criminal record's per-crime checklist)

Yardstick = CMFGEN toy06 @19.48d at Lumina velocities. Pre-registered gates; report as
quantity×location×magnitude×cause (`feedback_report_as_cmfgen_divergence`).

| # | crime | quantity @ location | gate | who cures |
|---|---|---|---|---|
| K1 | C11 Co IV pile | Co IV emit-share, s0 | 91 % → **≤ 30 %**; mc/cs @1526Å → ~1 | iii (+i) |
| K2 | C1 deep FUV | mc_J(918–1290,s0) | 1.9e-6 → **~2e-4** | iii+ii |
| K3 | C5 EUV | mc_J(300–450,s0) | 4.6e-6 → recovers; **must stay non-thermal** (selective iii) | ii (selective iii) |
| K4 | C4 deep u | u_bol(s0) | 400/264 → **climbs toward 695** (needs ii + hot T_e + NITER) | ii (+iii+NITER) |
| K5 | C3 deep T_e | T_e(s0) | 13120 → **past 15–16 kK** toward 18277 | ii+iii (self-heat on funnel-kill) |
| K6 | C2 Fe gradient | f(FeIV) slope s0→s8 | +0.65 → **+5.09** (field un-flattened) | i+iii |
| K7 | C6/H1/H2 phot FUV & S III | S III emit-share, s8; emergent UV frac | S III share **collapses WITHOUT F4**; UV 51 %→~23 % | i (round-2 pops) + iii |
| K8 | ff/fb share | k-packet continuum exit share, s0 | 0.02 % → **rises with KPKT_FB_MULTI** (measure) | ii |
| G-FUNNEL | **subsumption** | mc_J/cs_J @1526Å s0, **Fork B OFF** | **≤ ~2 (funnel stays dead WITHOUT Fork B)** | iii subsumes Fork B |

**Pre-registered residuals (do NOT claim improvement — inherited from `CRIMINAL_RECORD Task 3`):**
- **R-Co (C7+C10):** Co III/IV **rate** residual (threshold cliff 33.5 vs 30.65 eV × LTE b_k
  weighting) survives — needs SE-population-weighted Gph, outside (i)/(ii)/(iii). Twin proves it
  survives the correct field. Pre-register: f(CoIV,s8) still ~5–20× low.
- **R-Trad (C9):** T_rad uniform-pin @10470 K (MC estimator collapse) untouched; correct SE pops only
  remove its *downstream* CDF bias, not the pin.
- **R-tilt (H9):** MC blue-tilt (observer-frame Doppler) untouched — separate CMF frame transform.
- **R-hotband (H10):** far-outer hot-band death (RE instability) untouched — separate physics.
- **R-valley (C8):** deep valley fills via *amplitude* (ii+hot T_e), NOT the B(T_e) clamp (CMFGEN
  valley is scattering-dominated 4.76×B — a clamp would impose wrong thermal physics there).
- **R-NITER:** if u/T_e plateau below target at 12 iters, that is convergence (demonstrate at ≥30),
  not missing physics.

---

## 5. Risks & kill-criteria

| risk | signature | kill / mitigation |
|---|---|---|
| (iii) over-thermalizes EUV | mc_J(300–450,s0) collapses (K3 fails) like full-LTHERM | τ_cont gate too broad → raise W_thr / restrict to s0–s1; PLACEHOLDER ➋ sets the floor |
| (iii) drains u without (ii) | K1 passes but K4 regresses (400→264 redux, the N30 trap) | never ship (iii) without (ii); B(T_e) needs the continuum reservoir + hotter gas to hold u |
| (i) shifts the *photospheric* attractor onto a new ion | K7 S III share replaced by another ion's pile | it is the **channel**, not S III, that is guilty (arm-invariance, `CRIMINAL_RECORD` NEW EVIDENCE); the fix is (iii) at the phot τ_cont, not a per-ion patch |
| FB-MULTI `frozenin_alpha_rr` double-counts DR | ff/fb share overshoots; ionization balance shifts | audit `FROZENIN_DR` OFF vs ON (record: OFF=consistent); K8 measures the share |
| retiring Fork B too early | G-FUNNEL fails (6 % block-walk re-piles) | keep depth-scoped Fork B as safety net; retire only after G-FUNNEL passes with it OFF |
| stacking (iii) with Fork B mode-3/S_l leaks | funnel re-piles at phot | forbidden by design — MC never samples S_l (`STAGE4_ROUND2 §4.2`) |

**Hard kill-criterion for the whole repair:** if S2 (iii+ii, Fork B ON) does **not** collapse the
Co IV emit-share below 30 % at s0 (K1) while holding K3 (EUV non-thermal), the "thick-zone source
function" thesis is wrong — escalate to a transport-side continuum-thermalization audit (is the deep
continuum optical depth actually ≥1 in Lumina? — the τ_cont the gate assumes), because then the
field is not thermalizing even with the right source.

---

## Artifacts (this design, in `validation/cmfgen_toy06_19p48d/analysis/kp_emiss_repair/`)
- `kp_emiss_cdf_rebuild.py` — reproduces plasma.c:2095-2135 + the continuum split at s0; DB vs SE.
- `continuum_split_s0.csv` — tot_DB/tot_SE, C_ff, C_fb(Kramers), continuum shares, state.
- `band_shares_s0.csv` — collisional-excitation input band-shares, DB vs SE.
- `ion_shares_s0.csv` — per-ion (destination-ion = emitted-forest ion) emissivity, DB vs SE.
- Self-validation: offline C_ff/C_fb reproduce the event-log 0.007 %/0.015 % exit shares (§0).
