# Transport-Coupled T_e (CMFGEN complete-linearization) — Design

_Created 2026-06-30. Decision: user chose the faithful "transport-coupled (CMFGEN)" path
for the electron-temperature solve, over the ARTIS MC-estimator route and the
2×T_rad-cap approximation. Multi-session build. Production T_e method for the paper
("we use the CMFGEN method, not ARTIS")._

## 1. Diagnosis (established by extensive testing, 2026-06-29/30)

toy06 + radioactive deposition: the deterministic **local** heating=cooling T_e solve
cannot set the right T_e at optically-thick deposition-heated shells.

- Weak cooling (collisional-net lagged; ETLA trial-T_e 2-level): **no-root pin** at
  2×T_rad (bisection) / 3×T_rad (coupled Newton). At a dense hot shell the levels are
  collisionally thermalized (n_up→Boltzmann), so the *net* collisional line cooling → 0.
- Strong cooling (escape form Σβ_esc·A_ul·n_up·dE): breaks the pin (no-root 50→0) but
  **T_e collapses to the floor** — the gross escape over-counts (ignores re-absorption),
  and uses lagged n_up → downward runaway.

**Root (physics):** at a thick deposition shell the deposited energy leaves by **radiative
diffusion (transport)**, not local emission. In SE the dense hot gas is ~LTE
(collisional excitation = de-excitation, line escape = absorption) ⇒ net local line
cooling ≈ 0; ff+recomb are ~200× too small. The local heating=cooling balance is the
WRONG MODEL there — it is a transport problem. CMFGEN solves transport+energy together
(the small J−B that carries the flux divergence IS the cooling); ARTIS transports MC
packets. A local per-shell balance is structurally unable to reproduce this.

## 2. Why the EXISTING transport terms don't already fix it

Infrastructure that already exists (do not rebuild):
- `radeq_line_re` (plasma.c:4054): Option-2 line-transport energy term
  4π∫χ_line(J−S_l)dν, J registered from the CMFGEN formal solve via
  `radeq_set_line_re_source` (4009). Gate LUMINA_RADEQ_LINE_RE.
- `radeq_Hresp` (3886): bf heating response ∂J/∂T_e ≈ Λ*·ε·(B(T_e)−J*), Λ* = formal-solve
  diagonal operator. Couples J's response to the trial T_e.
- bf net = H_photo (∫Jσ_bf) − radeq_recomb_cool (∫B(T_e)σ_bf): continuum transport energy.
- `a4_global_newton` (5512): Stage-4 block-tridiagonal global Newton skeleton
  (LUMINA_A4_GLOBAL), per-shell warm start + off-diagonal staging.

Why they pin/cancel for the deposition case (to be confirmed in Stage 0):
- **Hypothesis A (leading): the deposition is NOT a source in the J transport.** H_gamma
  is added to the gas-energy balance only; the formal solve's emissivity η does not carry
  the deposited energy, so J never develops the (J−B) flux-divergence that would radiate
  it away ⇒ ∫χ(J−B)dν ≈ 0 ⇒ no transport cooling ⇒ pin. In a self-consistent solve the
  deposition must flow: H_gamma → T_e/populations → η(∝B(T_e)) → J transports it out.
- **Hypothesis B:** J is binned/thermalized (J≈B(T_e)) so the line-RE term ≈0 and the bf
  net cancels bin-by-bin; the tiny (J−B) that carries the flux is below the bin/numerical
  resolution.

## 3. Stages (each gated; nothing adopted until verified vs CMFGEN/ARTIS, per user rule)

**Stage 0 — confirm the mechanism (cheap, diagnostic).** Dump, at a thick deposition
shell, the per-bin (J−B(T_e)) from the formal solve and ∫χ_ν(J_ν−B_ν)dν (line + cont).
Confirm Hypothesis A vs B. If A: J carries ~0 of the deposited energy. Tool:
extend LUMINA_RADEQ_DIAG / JDUMP. NO code changes to the solver yet.

**✅ DONE (2026-06-30, toy06 19.48d + deposition, run logs/stage0_toy06_diag).**
Read-only probe `LUMINA_STAGE0_DIAG` (+optional `LUMINA_STAGE0_CSV`) in
`compute_radiative_equilibrium_te` (plasma.c, just after H_gamma): per deposition
shell it reports `4π∫χ(B−J)dν` (line + cont, from the registered formal-solve field
g_lre_J = cs.J), `radiated/Hgamma`, and the χ_abs-weighted Jcont/B(Te). Gated,
default-off, no solver state touched — kept as the **Stage 1/2 verification metric**
(re-run after injecting deposition into η: PASS = radiated/Hgamma goes 0 → ~1).
- **RESULT (robust across all 8 passes — Te=Trad seed, Te=12458 healthy, and the
  Te 12458→3787 escape-cooling collapse):** at every deposition shell
  **radiated/Hgamma ≈ 0** (|·| ≤ 6%, mostly <1%, sign varies) and **Jcont/B(Te) =
  1.0000**. ⇒ the deterministic field J carries essentially NONE of the radioactive
  deposition; it thermalises to the local B(Te). The deposited energy is disposed of
  LOCALLY by the escape/collisional cooling (which drives the unstable Te collapse),
  not by radiative transport — exactly the Section-1 root (local heating=cooling is
  the wrong model; it is a transport problem).
- **A vs B:** the design's stated A-criterion ("J carries ~0") is MET → A. Jcont/B(Te)
  =1.000 is also the B "thermalised" signature, but that χ_abs-weighted ratio is
  partly tautological in thick continuum (J→B(Te) by construction), so it is NOT a
  clean A/B separator — do not over-claim. The two are facets of one thing here, and
  Stage 1 is the same fix either way. **Gate PASSED → proceed to Stage 1.**
- Launch note: needs the `nlte` positional arg / `LUMINA_NLTE=1`, else
  `compute_radiative_equilibrium_te` early-returns at the NULL-NLTE guard
  (plasma.c:4324) and T_e freezes at ratio×T_rad (no RADEQ runs). A new exit
  **RUN FOOTER** (cuda.cu, atexit) now re-echoes env+argv+resolved enable_nlte so a
  result log's tail always proves what actually ran. Driver: scripts/run_stage0_diag.sh.

**Stage 1 — deposition into the transport source.** Add the (thermalised fraction of)
deposition to the emissivity seen by the formal solve, so J carries the deposited energy
outward. Then ∫χ(J−B)dν becomes the real diffusion cooling and the radeq has a root.
Touch: the η assembly feeding cmf_solve_J + the registration in radeq_set_line_re_source.

**⚠️ DONE but FAILED its purpose (2026-06-30).** Built: `cmfgen_set_deposition` +
`LUMINA_CMF_DEP_SOURCE` injects eta_dep=kappa·B(T_e) into S_fixed (kappa grid-exact so
4π∫eta_dep=f·H_gamma per shell). Gated, default-off, no regression. BUT the injection
only reaches **2.5% of the deposited power into the emergent** (lumina_spectrum.csv ΔL
1.93e41 vs 7.79e42 deposited) — the operator-split formal solve traps the rest, so J
does NOT carry the deposition. **ISOLATION (logs/stage1_toy06_s2dep0): the injection
changes the Stage-2 T_e by only 0.12% ⇒ INERT.** The staged plan does not stack here —
Stage 1's output is not load-bearing for Stage 2.

**Stage 2 — couple T_e to that transport energy in the solve.** Use radeq_line_re + bf-net
(transport terms) as the cooling in the T_e solve (NOT the local collisional/escape forms),
with the Λ* response (radeq_Hresp) giving ∂J/∂T_e so the Newton/bisection sees a real
restoring slope. This is the per-shell complete-linearization (T_e + δJ-response).

**⚠️ DONE — cured the collapse but NOT a pass; structurally pinned (2026-06-30,
logs/stage1_toy06_s2).** DEP_SOURCE=1 + RADEQ_LINE_RE=1 + RADEQ_COOL_ESCAPE=0. Cured the
escape-cooling COLLAPSE (T_e[0] 3787→13606, stable) — but the win is ENTIRELY the cooling
switch (isolation above), not the deposition. T_e ends at **0.71× CMFGEN inner / mean
0.55× / outer 0.15×** — NOT in-range. **Root (data): T_e ≈ 1.30 × T_rad UNIFORMLY across
all 50 shells** despite deposition varying 8 orders of magnitude ⇒ the local radeq pins
T_e to the inner-BB-anchored field color (T_rad≈T_inner≈10020), not the deposition-powered
value (CMFGEN T_e/T_rad≈1.84). Secondary real bug: radeq_line_re no-pumping floor
(plasma.c:4148) makes the full iron forest an ε=1 coolant, reinforcing the pin.
**Side-by-side (toy06 19.48d): Lumina is the COLDEST of {CMFGEN, ARTIS, TARDIS, Lumina};
ARTIS≈CMFGEN.** Paper bar = ≥ARTIS ⇒ the local-radeq path will not pass without breaking
the field-color pin (make the field deposition-powered / drop the inner BB, like ARTIS),
or pivoting to the ARTIS MC-estimator T_e. Figure figures/2026-06-30_te_methods_vs_standart.png.

**Stage 3 — global block-tridiagonal (a4_global_newton).** Couple shells (the diffusion is
non-local: dL/dr = deposition). Reuse the Stage-4 skeleton; the off-diagonal blocks are the
inter-shell J coupling (Λ tridiagonal from the formal solve).

**Stage 4 — verify + adopt.** toy06 T_e vs CMFGEN(v4k 19411 / v8k 12616 / v15k 9659) and
ARTIS(26100/15300/10800). Converged, no pin, no collapse, in-range. THEN add the config
selector LUMINA_TE_METHOD = legacy|artis|cmfgen (default cmfgen) and formally insert.

## 4. Notes / guardrails
- The ETLA-in-bisection code (gate LUMINA_RADEQ_LINE_RESPOND) is implemented + built but
  NOT adopted (does not break the pin) — keep gated for the comparison record.
- slurm overrides added: COOL_NLTE_ONLY / LINE_RESPOND / RADEQ_DAMP / COOL_ESCAPE.
- The 2×T_rad cap (NEWTON=0) coincidentally ≈ codes for toy06 — useful as an interim
  reference while building, but it is NOT a real balance (do not present as the method).
- cuda/13.0.2 build. ⚠️ never run two lumina_cuda on one GPU (OOM crash; check pgrep=0).
- See memory project_artis_whitebox_validation_2026-06-29.md and
  project_A4_simultaneous_coupling_design.md (the A4 infrastructure this reuses).

---
## 2026-07-01 (autonomous): fine-ν photoion (CMFGEN method) for the OUTER

**Goal:** the outer (v>22000) stays cold/under-ionized vs CMFGEN (24600 K, ne 3.6e4,
Fe III). Diagnosed lever = hard-UV PHOTOIONIZATION (not deposition, not non-thermal —
both ruled out). The binned-J collapses at the bf edges → soft outer photoion.

**Method = CMFGEN (deterministic fine-ν J reconstruction → Γ=4π∫σ_bf J^fine/(hν)dν).
NOT ARTIS (MC estimators, separate CC-SN path).**

**Built (all gated, A/B-validatable, no regression off):**
1. `fine_correct_R_bf()` in lumina_nlte_gemm.cu — corrects the production photoion GEMM
   (R_bf=K^T·J) over the fine window: R_bf = K_outside·J_bin + K_fine·J_fine (subtract
   coarse in-window via offset GEMM, add fine in-window via TILED K_fine^T·J_fine;
   memory-aware tile, CPU-fallback, Kramers via col_sig0). Gate = producer fine field
   (LUMINA_CMF_FINE_PHOTOION). ⚠️ first wired into coupled_photoion_rate_jnu = DEAD
   (COUPLED_NEWTON=0 prod path); real path is nlte_rates_gpu_compute.
2. `bf_gemm_compute_fine()` in lumina_bf_gemm.cu — fine-ν bf opacity chi_bf_fine[s,i]=
   Σ_l n_l·σ_l(ν_i) with SHARP edges at exact thresholds (tiled GEMM, reuse bf_gemm
   dilute-LTE n_level). Producer (cmfgen_fine_jbar) uses it: chi_abs = interp_chi_abs −
   bf_get_chi(smeared) + chi_bf_fine(sharp). Gate LUMINA_CMF_FINE_BF_OPAC. Setter
   cmfgen_fine_set_bf_atom (cuda.cu after bf init).

**Root found (cmfgen.c:2008): the producer INTERPOLATES the continuum opacity from the
binned state → fine field has no across-edge structure → fine photoion ≈ binned ≈ null.**
Fine photoion alone (FINE_PHOTOION, smooth field): only shell 25 moved (−5%, WRONG way),
outer exactly 0.0 (FROZENIN pins).

**NEGATIVE RESULT (fine bf opacity A/B, fphval vs fphbf): BYTE-IDENTICAL T_e through
iter 3 incl. J[mid,500]. The sharp-edge bf opacity fires ([FINE-BF-OPAC], Σchi_abs
preserved, +0.3-0.5% redistribution) but does NOT change the field.** ⇒ the field is
NOT controlled by the bf continuum. Two candidates under diagnosis (LUMINA_CMF_FINE_BF_DIAG,
dumps chi_es/smeared_bf/fine_bf at the outer): (a) chi_abs SWAMPED by chi_line forest +
chi_es scattering in the UV (→ bf-edge resolution is not the lever); (b) chi_bf_fine ≈
smeared (→ sharpening too weak). iter-2 diag (real pops) pending.

**Open question this raises:** if the field is forest/scattering-controlled (not bf),
the outer photoion may be limited by the hard UV genuinely not reaching the outer (inner
forest absorbs it), not by binning. Next diagnostic if (a): dump fs.J at the outer across
the UV vs B(T_e) — does hard UV reach the outer at all? If not, the lever is elsewhere
(trapped-field heating / the global coupling Stage 3), not the local photoion.

### 2026-07-01 cont: bf-edge NOT the lever; FROZENIN + coupling ARE

**bf-edge resolution refuted (data):** fine bf opacity (LUMINA_CMF_FINE_BF_OPAC) gave
byte-identical T_e. Diag: fine_bf/smeared_bf ≈ 1.00 — the edge sharpening is sub-binned-bin
= measure-zero in ∫σJ/ν dν. bf >> chi_es in the hard UV but it doesn't matter. The fine
photoion machinery is correct & reusable but the field is forest/scattering-controlled and
bf-reshaping can't change it. DO NOT re-attempt bf-edge sharpening.

**OUTER LEVER (data, run frz0 FROZENIN=0 vs fphval FROZENIN=1):** the outer ne RESPONDS to
unfreezing — sh40 7.68e4→1.59e5 (2.07×), sh49 7.09e3→1.53e4 (2.16×), ~half the gap to CMFGEN
(3.62e4). Inner/mid unharmed. FROZENIN=1 (a nebular freeze-out approx) was PINNING the outer
under-ionized; CMFGEN is full-NLTE-eq at 19.48d, no freeze. **BUT T_e unchanged
(5988→5982 vs CMFGEN 24600): ionization rose, temperature did not follow** — operator-split
decouples them; the thin outer T_e is pinned by the field color temp (~6000), no local
heating anchor. ⟹ outer needs (a) FROZENIN=0 + (b) the ionization↔T_e coupling = Stage-3
global coupling (carry the hot inner field out). The local photoion is NOT the lever.

**Next (in flight): test the existing coupled solve (LUMINA_COUPLED_NEWTON=1 + FROZENIN=0,
run cpl1) — does the A4 coupled ioniz-T_e Newton heat the outer? If yes, the lever +
machinery are validated; if unstable/cold, Stage-3 global block-tridiag needs the build.**

### 2026-07-01 far-outer turn-up: ionization-temperature bistability

After CN_DAMP=0.5 fixed the sh32-33 dip (numerical oscillation), the FAR-OUTER TURN-UP
remains: CMFGEN T_e rises outward (sh36 10574 → sh49 24600), Lumina declines (8900→7989).

Diagnosis (no band-aid):
- The T_e cap (coupled-Newton T_hi=3·T_rad, plasma.c:6597; made configurable
  LUMINA_CN_THI_FAC/ABS) is NOT the limiter: raising it to 140000 gave byte-identical
  results (Lumina sits BELOW the cap; the physics gives ~8000).
- The outer T_e is set by the deposition↔cooling balance. Deposition (H_gamma=3.1e-11) is
  the same input for all codes. Lumina balances at ~8000; CMFGEN at 24600.
- ROOT = ionization-temperature BISTABILITY. CMFGEN's outer is highly ionised (Fe III/IV,
  Si IV) → FEW line coolants → low cooling → deposition heats to 24600 (hot branch).
  Lumina's outer is under-ionised → MANY coolants → high cooling → deposition balances at
  8000 (cool branch). Both self-consistent; CMFGEN on hot, Lumina on cool.
- ⟹ the lever is the OUTER IONISATION. Getting it CMFGEN-high (Fe III/IV) drops the cooling
  and moves the balance to the hot branch. That needs hard-UV PHOTOIONISATION reaching the
  thin outer (H_photo=0 in Lumina because the binned J collapses at the bf edges) = the
  documented hard-UV / hot-ionised-outer frontier.
- ⚠️ Wiring trap: LUMINA_OUTER_TE_SEED (seed outer T_e high before the loop) is overwritten
  by compute_radiative_equilibrium_te which recomputes T_e each iter BEFORE coupled_newton.
  A hot-root-existence test must seed AFTER radeq (or disable radeq for the outer).

### 2026-07-02 far-outer (C): ionization is the lever; the field COLOR (T_rad) is the root

(B) CONFIRMED via LUMINA_OUTER_ION_BOOST probe (boost the coupled-Newton Γ for s>=smin,
plasma.c:5420): boosting the outer ionisation drives T_e strongly up (6737 → 72000 at even
5×; SATURATES — any boost >~5× fully ionises → same 72000, way past CMFGEN 24600 because full
ionisation strips the coolants). ⇒ the outer T_e IS ionisation-limited (more ionised → fewer
coolants → less cooling → deposition heats higher). CMFGEN's 24600 is a PARTIAL ionisation
(Fe III/IV), not full. [factor read VERIFIED via a [ION-BOOST] stderr print — the 5×≡1000×
result is genuine saturation, not a parse bug.]

Root of the under-ionisation = **the field COLOR T_rad is too low at the outer.** Lumina
T_rad: sh15 5100 → sh49 3134, DECLINING outward, far below the photospheric T_inner=10020.
The nebular ionisation φ_neb uses W·B(T_rad); with T_rad=3134 the hard UV is Wien-killed →
weak ionisation. The IONISING field should be the dilute PHOTOSPHERIC field W·B(T_phot~10020)
(the diluted hot inner UV that streams out), not the local cold T_rad. = the documented
too-red / frequency-resolved-field problem, now pinned to the OUTER IONISATION.

(C) TEST in flight: LUMINA_COUPLED_JNU_WBFLOOR=10020 floors the coupled photoion integrand to
W·B_ν(T_inner=10020) (the physical Mazzali-Lucy nebular ionising field). If the outer ionises
+ T_e rises toward CMFGEN, the field-colour is the lever and the FAITHFUL fix is getting the
outer field colour right (carry the diluted hard UV out). ⚠️ recurring trap this session:
`sed` adding an `export` silently fails on mid-line patterns (4×) → ALWAYS verify with
`tr '\0' '\n' < /proc/$(pgrep -x lumina_cuda)/environ | grep LUMINA_X` after launch.

### 2026-07-02 far-outer (C) result: WBFLOOR too weak; the deep bistability-calibration frontier

WBFLOOR=10020 (floor coupled photoion field to W·B(T_inner), the dilute photospheric hard UV)
TESTED: it0-3 T_e[49] = 6737,5710,6644,6799 — tracks the no-floor baseline (converges ~8000),
NEGLIGIBLE change. Confirms the memory's "diluted photospheric field too weak (W~0.0024)".
The (B) Γ-boost (×5) sent T_e to 72000; WBFLOOR barely moves it → the dilute-photospheric
photoion is ≪ what's needed. ⇒ CMFGEN's outer Fe III/IV is NOT reachable from the simple
nebular W·B(T_phot) field.

Deeper structure (the real frontier):
- The outer is an ionization-TEMPERATURE BISTABILITY. Cool branch (Lumina: T_e~8000,
  under-ionised, many coolants). Hot branch (CMFGEN: 24600, Fe III/IV partial, few coolants).
  Both self-sustaining (high T_e → low recomb α∝T^-0.5 → high ionisation → low cooling →
  high T_e).
- The (B) boost PROVES a hot branch exists in Lumina (it ran there) — BUT Lumina's hot branch
  overshoots to 72000 (FULL ionisation) whereas CMFGEN stabilises at 24600 (PARTIAL). So even
  reaching the hot branch, Lumina's ionisation runs away too far / its cooling at high
  ionisation is too low.
- ⟹ the faithful fix needs BOTH (a) a strong enough real hard-UV field to reach the hot
  branch (the frequency-resolved-field / too-red problem — the dilute nebular field is too
  weak; CMFGEN's actual RT field carries more hard UV out, or non-thermal contributes), AND
  (b) the ionisation-cooling balance calibrated so the hot branch sits at ~24600 (partial Fe
  III/IV), not 72000 (full). This is a coupled RT + non-thermal + atomic-cooling problem =
  the project's central open frontier, not a single knob.

Probes added (all gated, default-off): LUMINA_OUTER_ION_BOOST (plasma.c:5420, +[ION-BOOST]
verify print), LUMINA_CN_THI_FAC/ABS (plasma.c:6597), LUMINA_OUTER_TE_SEED (cuda.cu, note:
overwritten by radeq), LUMINA_DIP_TRACE, per-iter LUMINA_ION_POP_DUMP_ITER.

### 2026-07-02 far-outer (b) SOLVED: missing dielectronic recombination

The 72000 hot-branch overshoot (vs CMFGEN 24600) = MISSING dielectronic recombination (DR).
LUMINA_FROZENIN_DR (frozenin_alpha_rr +DR term, plasma.c:2335, Burgess form :3356) was OFF in
the config. DR dominates recombination at T~1-5e4 K; without it α is too low at high T → the
ionisation runs away to full stripping → 72000. Test boostdr (5× Γ-boost to REACH the hot
branch + LUMINA_FROZENIN_DR=1): T_e[49] 72037→46023→38212→31765→25792→21854→26983, converging
to ~24-26k ≈ CMFGEN 24600. ⇒ DR is the physics that calibrates the hot branch to CMFGEN's
partial Fe III/IV. Real physics + existing knob = not a band-aid.

far-outer decomposition now:
- (b) hot-branch calibration = DR (SOLVED — with DR the branch sits at CMFGEN's 24600, not 72000).
- (a) REACHING the hot branch = a strong-enough real hard-UV field (the Γ-boost is the test
  proxy; dilute nebular W·B(T_phot) via WBFLOOR is too weak; DR alone on the cool branch does
  NOT lift it). = the frequency-resolved-field / too-red frontier — the remaining piece.
Caveats: the crude Γ-boost propagates inward and over-heats sh25 (boost artefact, absent with
a field-only fix); DR should be ON as real physics but only helps once (a) reaches the hot branch.

### 2026-07-02 far-outer (a) REFRAMED: the mechanism is NON-THERMAL, not the hard-UV field

Decisive OFFLINE check at CMFGEN sh49 (T_e=24600, ne=3.6e4, W=0.0024, Si III edge 33.5 eV):
- Γ_dilute (photoion from W·B(T_phot=10020)) = 1.87e-11 s^-1
- Γ_needed (α·ne·n_IV/n_III for CMFGEN's Fe/Si III→IV) = 2.0e-7 s^-1
- **Γ_needed / Γ_dilute ≈ 1e4** — the diluted photospheric field is FUNDAMENTALLY ~10^4× too
  weak (hard UV at 33.5 eV is Wien-killed in B(10020)). ⇒ (a) is NOT achievable with any
  dilute-photospheric hard-UV field (explains why WBFLOOR was inert). CORRECTS the earlier
  "outer is photoionisation-driven" conclusion.
- Non-thermal (Spencer-Fano) rate from the deposition: Γ_nt ≈ dep·η_ion/(n_ion·W_ion) ≈
  (3e-11·0.03)/(1e4·5.6e-11) ≈ 1.6e-6 s^-1 ≳ Γ_needed. ⇒ **the outer Fe III/IV is maintained
  by NON-THERMAL ionisation, not the radiation field.**

WIRING GAP found: nonthermal_ioniz_rate is applied ONLY to the NLTE LEVEL matrix
(plasma.c:8210, ground→ground_hi), NOT to the ion-STAGE balance (coupled_charge_density_tdep
Γ_j / compute_ion_populations nebular Saha) that OWNS the outer ionisation (per the
control-flow map). So the memory's "non-thermal too weak" was really "non-thermal not wired
into the ion-stage balance". FIX (autonomous): LUMINA_COUPLED_NT=1 adds the per-atom NT rate
g_nt_ioniz_rate[s]/n_atom to each pair's Γ_j in coupled_charge_density_tdep (plasma.c:5386
region; budget-conserving: Σ Γ_nt·n_j = R_nt_total). Combined with LUMINA_FROZENIN_DR=1 (the
(b) hot-branch calibration), test ntdr checks whether NT-ionisation + DR → CMFGEN's 24600.

### 2026-07-02 (a) non-thermal wiring: confirmed active but NO effect — unresolved subtlety

Wired the NT rate into coupled_charge_density_tdep's Γ_j (LUMINA_COUPLED_NT=1). VERIFIED
active via [COUPLED-NT] print: s=49 g_nt_rate=2.767e-2, natom=1.387e4 => Gamma_nt=1.995e-6
s^-1 (matches the offline estimate, ≫ photoion ~1e-11). BUT run ntdr (NT + DR) is
BYTE-IDENTICAL to the no-NT baseline through it0-3 (6737,5762,6729,6637) — the NT does NOT
lift the outer, despite Gamma_nt being added to Γ_j.
PUZZLE: the OUTER_ION_BOOST (Γ_j *= 5, SAME line) DID jump the outer (boostdr), but adding a
much larger absolute Gamma_nt does nothing. Candidate explanations to debug next session:
(1) coupled_charge_density_tdep is a Newton RESIDUAL; the final ion STAGES that set the
cooling come from compute_ion_populations (nebular Saha, control-flow map) which has NO NT —
so changing Γ_j here may move the Newton n_e slightly but not the ion-stage populations/cooling.
(2) the boostdr "jump to 72000" may have been a GLOBAL numerical perturbation (sh25, NOT
boosted, also hit 72000) that DR then damped — i.e. boostdr's ~24600 might be a boost
artefact, not the physical hot branch; the NT (physical, per-shell) correctly stays cool.
⇒ RESOLUTION NEEDED: is CMFGEN's 24600 a real root of Lumina's equations reachable by NT, or
does Lumina genuinely lack the mechanism? Add a print of the resulting n(IV)/n(III) at s=49
with/without NT to localise (ionisation-changes-but-T_e-doesn't vs Γ_j-add-doesn't-reach-solve).
Also wire NT into compute_ion_populations (the ion-stage owner), not just the charge residual.
