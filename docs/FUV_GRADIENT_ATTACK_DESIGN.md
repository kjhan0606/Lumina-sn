# FUV/EUV Gradient Attack Design — the transport front

2026-07-19 (Fable). Successor to the #32 gradient-budget verdict and the #33
gradient-transplant PASS. Supersedes the "IGE under-ionization" framing and the
uniform-boost integrated arm.

## Established causal chain (do not re-litigate)

1. CMFGEN toy06 @19.48d has a ~4.6-dex deep→photosphere Fe recombination
   gradient. Lumina is nearly flat (#32: budget table; rates/n_e innocent).
2. Injecting CMFGEN's J_nu(v) into Gph alone restores 4.3 of the missing 4.4 dex
   (#33, job 179575: photosphere f(FeIV) 0.46→0.009, deep 0.79→0.994).
   The field IS the mechanism for Fe. Co retains a ~10x rate-side deficit
   (separate secondary front).
3. **F2 (this doc's trigger)**: in the #33 run the plasma state was
   CMFGEN-correct, yet the *transported* mc_J stayed flat
   (s0→s8: −0.11 dex vs CMFGEN +2.42). The flat field is **transport-intrinsic**,
   not downstream of the wrong ionization. One-cycle feedback recovery is refuted.

## The measured decomposition (F2 absolute values, FUV 918–1290 Å)

| shell | v | Lumina mc_J | CMFGEN | ratio |
|---|---|---|---|---|
| s0 | 4264 | 0.9–1.4e-6 | 2.02e-4 | **−2.3 dex (deficit)** |
| s8 | 10088 | 1.2–2.2e-6 | 7.7e-7 | +0.2–0.4 dex (excess) |

**Axis 1 (dominant): deep FUV amplitude deficit, ~−2.3 dex at s0.**
Candidates, with a-priori sizes:
- (1a) deep T_e deficit 13120 K vs CMFGEN 18900 K → Wien color factor at
  1100 Å ≈ **−1.3 dex** of source color alone. The T_e deficit is itself a
  radeq/backwarming question (under-trapped UV → cooler gas → weaker UV: a
  self-consistent wrong fixed point).
- (1b) trapping/thermalization deficit (~−1 dex residual): deep FUV emissivity
  present but not built up to quasi-thermal J (insufficient line-blanketing
  optical depth, or emission channels missing: bf/ff/fluorescent cascades).

> **2026-07-19 F0b/F0a/F1 VERDICT (see analysis/F0_F1_FUV_GRADIENT_VERDICT.md):
> fork = T_e's FAULT (1a).** Deep J sits on its own W·B(13120 K) ceiling
> (0.3–1.1×); the deficit tracks the Wien color penalty band-by-band and
> vanishes at 1290–2000 Å; F1 closure: 2.12 dex = color 1.29 + dilution 0.35 +
> transport residual ≤0.49 (the a-priori "~−1 dex trapping" above was an
> overestimate — corrected). F0a: deep FUV emit≈abs (well-trapped), deep EUV
> emitted but bf-absorbed en route, photospheric FUV locally created
> (upconversion, s7–s8 activity 45× deep) = Axis 2 confirmed. F3-T promoted;
> the root question is why radeq lands the deep shells 5600 K cold
> (backwarming/EPAY lineage). Event-log caveats: etype 7/8 unlogged
> (bf-reemit invisible), CAP128 saturated, single iteration (11).

**Axis 2 (secondary): photospheric FUV excess, ~+0.4 dex** — the one-way
upconversion family (S III em/abs≈7.5, outer-shell J̄-pump) adding FUV outward.

## Falsifier ladder (offline first; each rung kills or promotes a mechanism)

**F0b — deep thermalization test [offline, decisive fork for 1a vs 1b].**
Compute J/B(T_e_local) per band (FUV + 2–3 flanking bands) per shell s0–s8 for
the B-run and the jtable run. CMFGEN reference: same ratio from the jnu4 J_nu
with published T(v).
- If Lumina deep J ≈ W·B(13120) (as thermalized as its temperature allows):
  the deficit is **T_e's fault** → promote F3-T (temperature-table probe);
  the transport verdict shifts to the radeq/backwarming front.
- If deep J ≪ W·B(13120): transport starves the deep field independent of T_e
  → promote F0a emission/trapping forensics.

**F0a — deep FUV emission & absorption ledger [offline, event-log].**
The a10_kx runs carry EVENT_LOG (lumina_events.bin, CAP128M) and the event-log
reader (commit ac8ef44). For s0–s2 vs s6–s10, band 918–1290 Å (+300–450 Å):
per-process emission tallies (bf/ff/line/macro-atom channel), absorption
tallies, and the net per-shell FUV source function. Deliverables:
- deep FUV emissivity: present at CMFGEN-like strength (→ trapping problem) or
  2-dex weak (→ source problem)?
- photospheric FUV creation: how much of s6–s10 FUV is locally created
  (upconversion) vs transported from below? (Axis-2 quantification.)
- EUV (300–450): does the deep core emit at all? (The photospheric EUV floor
  of #32-D2 needs an origin: never-emitted vs absorbed-en-route.)

**F1 — arithmetic closure [offline, no new data].**
Wien/color accounting: exact band-integrated B(T) ratios 13120 vs 18900 K per
band; dilution-only profile W(r) normalized at s0. Decompose the measured −2.3
dex into color + dilution + residual. (Guards against narrative drift; the
numbers above are a-priori estimates.)

**F3-T — temperature-table probe [GPU run, only if F0b says "T_e's fault"].**
The T-analog of #33: LUMINA_TE_TABLE forcing T_e(v) to CMFGEN's published
profile in the *emission/thermal source terms* (surgical, default-OFF,
counters; design review before implementation — T_e has many consumers, the
probe must pin only the source-term temperature). Pre-registered: deep mc_J
rises ≥1 dex at s0 and the s0→s8 slope steepens ≥1 dex → T_e axis confirmed
causal; then the real fix = why radeq lands 5600 K cold (backwarming/EPAY
lineage). If mc_J barely moves → 1b trapping dominates.

**F3-B — inner-boundary COLOR probe [GPU run, promoted 2026-07-19 after F3-T
refuted the local-T_e model].** LUMINA_TINNER_COLOR recolors only the
DIFFUSE_INNER_BC re-emission (+ boundary-SED injection) frequency draws from
Planck(T_inner=10020 K) to Planck(18760 K); energy/L/controller bit-unchanged.

> **2026-07-19 F3-B VERDICT (179665 tincol-only, wiring live 0.4–5.4M
> recolors/iter): REFUTED as dominant cause.** Deep s0 FUV rose only +0.46 dex
> (5.81e-6→1.66e-5; pre-registered PASS ≥ +1.5), slope −0.06→+0.40 (target
> +2.42), EUV s0 unchanged/down, T_e free response −600 K, f(FeIV) unchanged.
> CORRECTED interpretation (J-budget decomposition): TINCOL's deep J sits at
> 5.4× its local thermal ceiling W·B(T_e) (B-run: 1.19×) — boundary color DOES
> propagate, but the boundary-recycle channel is amplitude-limited (its added
> FUV ~1.3e-5 ≈ 6% of CMFGEN's 2.02e-4). The Wien arithmetic (−2.7 ≈ −2.3 dex)
> ignored channel amplitude — another arithmetic-coincidence (yardstick
> case-12 pattern). Identity decomposition of the s0 divergence (−1.54 dex):
> color(13120→18760 K) 1.25 + dilution(W=0.298 vs CMFGEN's undiluted diffusion
> interior, J/B=0.70) 0.53 − occupancy diff 0.23. F3-T in these terms: with
> T_e pinned hot the occupancy COLLAPSED 1.19→0.23 — the field cannot fill the
> raised thermal ceiling (deep FUV emissivity/trapping does not scale with
> B(T)). Note the pre-registered
> absolute anchors (1.4e-6 / −0.19) were an F2-era different aggregation; on the
> consistent band-mean metric the dex-form gate applies and the verdict is
> robust (+0.46 ≪ +1.5; F3-T on the same metric: +0.54). Decision now rests on
> the 179667 end-member (tincol+T_e-pin): if deep FUV still ≪ CMFGEN there, the
> residual is transport-structural (dilution W~0.3 + sub-ceiling occupancy /
> line-blanketing buildup / emission channels).

**F4 — upconversion gate A/B [GPU run, Axis 2, only after F0a quantifies it].**
If photospheric FUV creation is dominated by an identifiable channel (S III
family), a channel-gate A/B quantifies Axis 2's contribution to the flatness.

> **2026-07-19 F3-B END-MEMBER (179667 tincol+tetab): deep amplitude gate PASS,
> slope half.** FUV s0=7.14e-5 (within 2.8× of CMFGEN, +1.09 dex vs B-run);
> slope +1.23 (target +2.42) — the missing half is the photospheric FUV excess
> (s8 4.2e-6 vs 7.7e-7, +0.74 dex; Axis-2 upconversion persists). u_bol
> OVERFILLED 2.35-2.60× ⟹ the T_e pin is non-conservative forcing: this proves
> deep FUV *responds* to gas temperature + boundary color, not that the
> physical system can supply it. The causal chain to fix remains: bath
> reddening → cold radeq root → FUV collapse (see audits below). Photospheric
> f(FeIV) NOT restored without jtable (s6 0.556 vs 0.069) — photospheric
> over-ionization rides the photospheric FUV excess.

## 2026-07-19 trapping/energy audit (pre-registered) — REDEFINES THE CRIME SCENE

Artifacts: validation/cmfgen_toy06_19p48d/analysis/trapping_audit/. All four
calibration anchors reproduced before extension.
- **Audit U (energy)**: u(s0) Lumina/CMFGEN = 0.576 (−0.24 dex; pre-registered
  0.10-0.15 REFUTED). Mid-shells s2-s8 Lumina holds MORE energy (s4 1.57×).
  → ~1.3 of the −1.54 dex FUV deficit is SPECTRAL redistribution (energy sits
  in redder bands), not missing energy.
- **Audit T (tau)**: reversed — Lumina MORE opaque everywhere (es floor
  1.18-1.56×, Rosseland 5.8 vs 4.1 at s0); FUV tau≈70 outward from s0 →
  J_FUV is slaved to the local gas temperature (thermalized), not escaping.
- **Census C**: transport traverses the full 2,565,342-line forest
  (lumina_atomic.c:392) — truncation ELIMINATED. But 97.9% of transport lines
  draw lower-level pops from super-level Boltzmann(T_e) redistribution
  (population accuracy, demoted to amplifier).
- **New central puzzle**: u(s0) = 1.79 × a·T_e^4(13120K) — the gas sits
  2000-2600K BELOW its own bath (bath-equivalent T ≈ 15-16 kK); CMFGEN is the
  opposite (u = 0.74·a·T^4, gas hotter than bath). Suspects now: (1) radeq
  thermal-coupling defect (why the cold root), (2) one-way blue→red
  down-conversion reddening the bath (fluorescence-saga profile).
  Follow-up offline audits launched: radeq_ledger_audit/, reddening_localization/.

## 2026-07-19 afternoon — CLOSED CAUSAL CHAIN (radeq ledger + reddening localization)

- **Radeq ledger audit** (validation/.../analysis/radeq_ledger_audit/): the
  balance is FAITHFUL — the same simul_r1 formulas fed CMFGEN's J solve to
  T_e=18277 K vs truth 18760 (2.6%). Ladder: zero-pump 13214 (≈ run 13120) →
  thermal 14818 → cs.J 16617 → CMFGEN jtable 18277. Heating at s0 is 100%
  γ-deposition (bf photo-heating 0.05%, taps only EUV<912 Å; ×520 under
  CMFGEN's field); cooling 71% un-pumped lines + 26% fb. s0-s2 are NOT on the
  T_rad pin (s9-s12 are). Flagged, not asserted: split-field symptom (photoion
  consumes mc_J, cooling consumes cs.J). Verdict: radeq is a faithful reporter
  of a starved field — the criminal is upstream.
- **Reddening localization** (validation/.../analysis/reddening_localization/,
  driver spot-checked): the deep MC field is NOT a continuum — it is a
  **Co IV emission-line spectrum**. 51.4% of s0 u in 1290-2000 Å (42% in one
  log-λ bin at ~1508 Å); mc/cs = 39× at 1526 Å (Co IV), while 1700-2100 Å mc_J
  is 4% of cs_J. The deep forest absorbs NUV/blue (2000-4500) + red and
  funnels it into the Co IV 1490-1650 Å complex (84% of deep line emission);
  deep emission-weighted λ=1553 Å, bluer than B(T_e) — thermal fraction <2%.
  Red/NIR field excess = 10020 K boundary seed + unlogged bf, not lines.

**CLOSED CHAIN**: MC line transport funnels deep energy into the Co IV complex
(S_line ≁ B thermalization defect; CMFGEN holds S≈B(18760) with the same Co IV
line data → data innocent, algorithm guilty = the fluorescence-saga serial
criminal) → EUV/FUV starvation + NUV pile → bf heating (EUV-only) + line
pumping starve → radeq lands the zero-pump root 13120 K → FUV (τ≈70)
thermalizes to the cold gas → −1.54 dex deep FUV → Fe recombination gradient
dead. Photospheric FUV excess (upconversion, Axis-2) is the other half of the
slope, independent.

**Driver correction to the localization agent's proposal**: a Co IV→III
ionization-shift A/B is REJECTED as the primary probe — CMFGEN's deep Co IS
fully IV; the ionization is right, the complex's THERMALIZATION is wrong.
Next probes (design): (1) deep-shell line-thermalization gate (force S=B(T_e)
for s0-s2 line interactions; ENERGY-CONSERVING unlike the T_e pin;
pre-register: 1508 Å pile collapses, EUV/FUV refill, radeq root climbs
toward 16-18 kK with NO pin), (2) offline split-field audit (mc_J vs cs.J
consumers), (3) event-log with bf channels (etype 7/8) enabled.

## 2026-07-19 ARREST EXPERIMENT (LINE_THERM manual run, syn08, 12 iters) — FUNNEL CONVICTED (PARTIAL)

Gates: **(i) PASS decisively** — 1290-2000Å u-frac 0.514→0.045, max-bin 0.42@1508→0.012@3648 (funnel erased). **(ii) PARTIAL** — FUV(s0) +0.60 dex (2.31e-5; −0.94 to CMFGEN). **(iii) near-miss** — T_e(s0)=14080 vs gate 14500, but monotonic climb all 12 iters, still +~100K/iter at end (s2=14882); NOT converged (DAMP=0.5). **(iv) FAIL, physics-consistent** — EUV −1.9 dex: blanket S=B forcing also kills legitimate non-thermal EUV line emission (Planck(14 kK) has none <450 Å).

Two decisive confirmations: **u_bol(s0) 400→758** (above CMFGEN 695; bath-equiv ~17.8 kK) — the funnel WAS the deep energy leak; **TEHOLD root-found 11/12 iters** — the radeq freeze is a state-dependent symptom of the sick bath, not an independent bug (demotes the split-field audit's "+3400K co-criminal" to symptom-amplifier; B-run freeze itself remains an uninstrumented inference).

Side-effect discovery: deep Fe f(IV) 0.79→0.294 — the B-run's deep near-IV was funnel-pumped excited-state photoionization (right answer, wrong reason); CMFGEN's deep IV needs the hot gas/field itself.

Remaining to full recovery: (a) T_e convergence (NITER=24-30 extension run), (b) legitimate non-thermal EUV — blanket S=B is too blunt; the production fix is selective thermalization coupling (collisional destruction ε at deep n_e), not full redistribution.

## Secondary front (parallel, cheap): Co rate deficit

With the jtable background (correct field), Co lands at 0.005–0.026 photosphere
(CMFGEN 0.10) and 0.50 deep at s2 (CMFGEN 0.98): the ~10x Co-specific rate
deficit is now measurable in isolation. Offline first: per-shell Γ(Co III)
multiplier that closes Co to CMFGEN under the CMFGEN field; compare against the
spin-gate α-cut (d57bc98) and Co DR (60144ad) magnitudes. Then a single
jtable+spingate A/B run if the arithmetic matches.

## Ordering & discipline

- F0b + F0a + F1 run in parallel offline (Opus delegation), zero GPU hours.
- No GPU run before its offline rung reports. No uniform-knob fixes; every
  intervention is a gated diagnostic probe with pre-registered outcomes.
- Watcher preflight rule (yardstick case 11): any monitor must verify its
  watched file actually grows during the first minutes of the run.
