# MASTERMIND TEST — ONE criminal, not two

Offline, read-only, 2026-07-20. B-run corpse `logs/coevolve_consume_a10_kx_gphall/`
(iter=11, 128M-event CAP). Live `bsrc` untouched. No GPU/sbatch/commit/edit.
Every number below = event-log measurement or file:line. Scripts + CSV in this dir.

## VERDICT: **H1 — ONE mastermind.** The deep Co IV 1500Å pile and the photospheric
## S III FUV excess are the SAME broken channel: the **k-packet global re-excitation
## CDF** (a cross-ion thermal-emissivity sampler). The local peak-emissivity ion is
## the attractor — Co IV deep, S III at the photosphere. There is NO independent
## deep resonator; H2's premise is empirically falsified.

The apparent contradiction between the two prior verdicts is a DENOMINATOR artifact,
resolved exactly below. Both measured the same data; one measured the attractor's
*self-loyalty* (high), the other the *whole population's* loyalty (low).

---

## The discriminator (one number, Task 2)
SAME-ION fraction of consecutive (line-abs etype1 → next line-emit etype2) pairs,
63.9M pairs, ion convention **0-based, spectroscopic = ion_field+1** (verified: line_id
391357 = 1526.170Å, Z=27, ion_field=3; validated `reddening_localization/taskB_top_ions.csv`
maps (27,3)→"Co IV"). Cross-ion line-emit is reachable **only** via k-packet re-excite
(the radiative block-walk is per-ion; `MACROATOM_BF` cross-ion path is OFF —
`d_recomb_enabled=0`, plasma.c:1249-1252). So (1 − same-ion) = k-packet traffic (lower bound).

| shell group | same-ion % | cross-ion % (= k-packet floor) |
|---|---|---|
| overall | **17.2%** | 82.8% |
| s0-2 (deep) | **37.1%** | 62.9% |
| s3-6 (mid) | 15.6% | 84.4% |
| s7-9 (phot) | **9.0%** | 91.0% |

Same-ion is LOW at BOTH depth and photosphere → **H1 (global re-routing dominates)**.
Per-shell gradient is smooth (shell2 42% → shell9 5%), one continuous mechanism.

## Why it *looks* like two (the reconciliation, Task 2 last bullet)
The pattern is a global attractor with a self-looping core:

**Deep s0-2, by donor ion** (`same_ion_results.csv`):
- **Co IV donor → 84.6% self-recycle** ← the funnel_trace's "77%" (their 0.756 block prob).
- Co III 9.9% same → **emits Co IV 91%** of cross; Fe III 2.6%→Co IV 83%; Ni IV 5.8%→Co IV 89%;
  Ni III 1.4%→Co IV 84%; Fe IV 1.3%→Co IV 85%. **Every non-Co-IV donor funnels into Co IV.**

**Phot s7-9, by donor ion:**
- **S III donor → 78.8% self-recycle**.
- Co III (largest donor, 10.5M) 10.5% same → **emits S III 90%**; Fe III 2.5%→S III 82%;
  Ni III 1.6%→S III 80%; Si III 3.4%→S III 87%. **Every non-S-III donor funnels into S III.**

- funnel_trace **77%** = Co-IV-DONOR-conditioned self-recycle (measured here 84.6%). Correct,
  but conditional on *entering as Co IV*.
- axis2 **5.8%** = ALL-DONOR same-ion at s7-9 FUV (measured here 9.0% over all line-emits). Correct.
- They differ because the attractor is a *small share of the donor pool*: deep Co IV = 40% of
  donors → overall deep same-ion 37%; phot S III = 2% of donors → overall phot same-ion 9%.
  **Same mechanism, two conditional slices. Not two mechanisms.**

Top emitted lines confirm the attractors: deep = Co IV 1490-1650Å forest (1526.17Å lid 391357
leads, the funnel line); phot = S III 911/1152/1480/1577Å FUV. All emitted energy in the pile
bands is the attractor ion.

---

## Task 1 — emission-selection path census (B-run gphall)
Resolved config: `LINE_INTERACTION=macroatom`, `KPACKET=1`, `KPACKET_EXIT=1`,
`MACROATOM_EWEIGHT=1`, `IDOWN_BETA=1`, `NEUTRAL_E=1`; `MACROATOM_BF` absent; BSRC/LTHERM/EPS_UV/
EPS_IR all default OFF. (Launcher prints `KPACKET=0` as a `${:-0}` default; the submit env
exported `LUMINA_KPACKET=1`, so the binary's RESOLVED CONFIG shows KPACKET=1 — active.)

| # | path | file:line | class | traffic |
|---|---|---|---|---|
| 1 | macro-atom radiative block-walk (internal up/down/emit) | cuda.cu:3099-3147 | **(a) manifold-confined** | the ≤17% same-ion residue |
| 2 | **k-packet collisional re-excite from per-shell CDF over ALL levels/ions** | cuda.cu:3042-3088, esp. **3071-3079**; CDF built plasma.c:2135, 2226-2229 | **(b) GLOBAL** | **≥83% overall, ≥91% phot; p_kpacket(s0)=0.94** |
| 3 | k-packet → free-free continuum | cuda.cu:3358-3388 (`-2`) | (c) thermal | 0.007% of exits |
| 4 | k-packet → free-bound continuum | cuda.cu:3389-3444 (`-3`) | (c) thermal | 0.015% of exits |
| 5 | bf-abs → (no mapped level) → **same global k-packet CDF** | cuda.cu:3998-4008 | **(b) GLOBAL** | bf-abs feed (etype3, unpaired) |
| 6 | eps_uv / eps_ir Planck(T_rad) thermalize | cuda.cu:3282-3318 | (c) | OFF (d_eps_uv=0) |
| 7 | LTHERM / BSRC per-line thermal re-emit | cuda.cu:3214-3226 etc. | (c) | OFF in B-run |
| 8 | Fe two-level resonance scatter | cuda.cu:3238-3266 | (a) | fe_scatter_mode off |

`J_src=MC_histogram` feeds the **rate/TransProb builder** (plasma.c up-rates), i.e. it sets the
INTERNAL cascade branching — it is not itself the emission sampler. The emission-selection global
sampler is path #2's `kpacket_cdf` (`kp_emiss[dst] += n_lower·C_up·dE`, plasma.c:2132-2135).

**The p_kpacket=0.94-vs-8e-10 inversion (why funnel_trace mis-diagnosed):** funnel_trace computed
`p = ΣC_down/(ΣC_down+ΣA·β) = 1.16/(1.16+1.44e9) = 8.1e-10`, comparing the collisional rate against
the **UNWEIGHTED** radiative rate. The code (plasma.c:2214-2219) computes
`pkv = kp_deact/(sum_rates + kp_deact)` where `sum_rates` is **ENERGY-FLOW-WEIGHTED** (each term
×hν or ×e_low·EV_TO_ERG ≈ 1e-11–1e-10 erg; eweight applied plasma.c:2076-2093) while `kp_deact` is
the **unweighted** rate (plasma.c:2109). The denominator mixes erg/s with s⁻¹ — the radiative side is
~1e11× smaller than funnel_trace assumed, so `pkv → ~0.9`. The run's own log confirms it:
`[KPACKET] mean p_kpacket: shell0=9.43e-01`. So the deep macro-atom thermalizes into the k-packet
channel ~94% of the time — but that channel re-emits a **globally-CDF-sampled UV forest LINE**
99.98% of the time (ff+fb continuum = 0.02%), NOT a smooth continuum. The thermal sink funnel_trace
said was "unreachable" is firing constantly; it just emits the wrong thing.

---

## Task 3 — repair + the N30 deep-drain constraint

**Guilty path:** #2, the k-packet global re-excitation CDF (`kpacket_cdf`/`kp_emiss`,
plasma.c:2132-2135 & cuda.cu:3071-3079). Traffic ≥83% overall (≥91% phot, 63% deep). Root of the
bias: `kp_emiss` is built from **non-SE dilute-Boltzmann populations** of the deep stage-IV IGE ions
(Co IV/Fe IV/Ni IV are excluded from the NLTE set), so the "thermal" collisional emissivity does NOT
integrate to B(T_e) — it over-weights the UV forest. The k-packet is a *global line-emissivity
resampler*, not a thermalizer.

**Correct single-channel repair (one physics mode):** fix what path #2 EMITS so its ensemble is
thermally consistent — any of: (a) promote stage-IV IGE into the SE/NLTE set so `kp_emiss` reflects
true populations (its emission then relaxes toward B(T_e)); (b) raise the k-packet continuum branch
(p_ff+p_fb, plasma.c:2237-2269) to its physical thermal fraction so the pool emits a genuine ff/fb
continuum instead of a line; (c) clamp the k-packet line source to B(T_e) where continuum-thick. All
three repair the ONE channel. **Fork B (`d_line_bsrc`, cuda.cu:388-393 masks Z∈{26,27,28} ion=3) and
the proposed F4 S III gate are per-ion patches on this channel's OUTPUT** — Fork B thermalizes the
stage-IV output (deep), F4 the S III output (phot). Both are validated stopgaps that treat the same
criminal's two crime scenes; neither repairs the channel. Fork B stays useful as a *color* stopgap
(it does kill the funnel: mc/cs 7.74→0.24) — but it cannot supply amplitude (below).

**N30 deep-drain (u_bol(s0): B 400 → n12 327 → n30 264 vs CMFGEN 695) — H1 explains it:**
The funnel was doing DOUBLE DUTY: (bad) mis-coloring the deep field into the UV forest AND
(accidentally) TRAPPING energy — the forest lines are τ~1e4, so photons bounced and held up a
partial u reservoir (though still 400<695). Fork B thermalizes the stage-IV emission to
~2000-3500Å, where line opacity is far lower → photons ESCAPE → u drains (400→264). The
color-fix and the drain are **separable but share one root**, exactly as the driver anticipated.
- **Who holds CMFGEN's u=695:** a genuinely HOT deep gas (T_e≈18760K vs Lumina 13120→14524K,
  ~4200-5600K too cold) radiating a real **bf/ff CONTINUUM reservoir** in the SE-thick sub-3900Å
  interior. Lumina's deep stage-IV IGE, being dilute-Boltzmann (not SE), has **no continuum thermal
  emission** — only the line forest. The funnel was a line-based *counterfeit* of that missing
  continuum reservoir. Remove the counterfeit (Fork B) without installing the real one → u collapses.
  ⇒ **The criminal still starving the deep bath after Fork B = the missing deep continuum
  thermalization (stage-IV IGE excluded from SE) + a T_e ~5 kK too cold — upstream of BOTH the funnel
  color and the u amplitude.** Fork B's source-function repair is NOT refuted; it fixes color, and
  the amplitude channel (deep continuum reservoir / hotter gas) is simply still absent.
- **Photospheric strengthening under N30** (f(FeIV) s8 0.461→0.693, S III channel up): the energy
  Fork B redistributes outward from the deep region is reprocessed by the SAME global attractor one
  band up (S III FUV), driving more over-ionization. Same channel, same criminal — an H1 signature,
  not a second mechanism.

**What each hypothesis predicts for the N30 plateau (T_e(s0)~14.5kK, deep u after Fork B=326 vs 695):**
- **H1 (this verdict):** killing the funnel color necessarily drains u unless the deep *continuum*
  reservoir is installed; the deep bath stays starved because SE-thick stage-IV continuum + hot gas
  are missing. The plateau at 14.5kK and u≈264 is the honest floor of a cold gas with no line trap.
- **H2 (rejected):** would predict Fork B's deep patch is independent of the photosphere and would
  not systematically strengthen the S III channel — contradicted by the measured f(FeIV) rise and by
  the 63% cross-ion / 91%-into-Co IV deep donor structure.

## Files
`same_ion_discriminator.py` → `same_ion_results.csv` (same-ion by shell-group & donor; convention
check; Co IV pile & S III phot donor breakdown). Numbers cross-checked against the run's own
`[KPACKET] mean p_kpacket` log line and `taskB_top_ions.csv`.
