# 확증 사다리 (Confirmation Ladder) — known-answer controlled tests, whole module (2026-06-26)

> **Naming (user 2026-06-26):** this is the **확증 사다리 (Confirmation Ladder)** — distinct
> from the old **검증 사다리 (Validation Ladder, V1/V2)**. The validation ladder used empirical
> pokes against gold (no per-step ground truth) and wandered for 2 days. The Confirmation
> Ladder admits a rung ONLY if it has an analytic/known answer → every rung is an unambiguous
> pass/fail → cannot wander. We re-confirm even "already-passed" stages here.


**Mandate (user):** re-validate EVERY stage — including those we *thought* were
confirmed — with **controlled tests against a KNOWN analytic/reference answer**. Weeks OK.

**Principle (why this is NOT the 2-day whack-a-mole):** the wandering came from
*empirical poking with no ground truth* — change something, get an ambiguous result,
chase the next anomaly. A **controlled test has the answer known a priori**, so every
result is an unambiguous pass/fail. Breadth is fine *as long as every test is anchored
to a known answer.* No test enters the suite without its analytic/reference target.

**Order: bottom-up.** Transport first (where the failures concentrated), then the
radiation field, then NLTE populations, then plasma, then the full assembly. A stage
is not "trusted" until its controlled tests pass.

---

## 🔴 CURRENT CRITICAL FINDING (2026-06-27, REFINED 2026-06-28) — the rate-network bug is REAL

**The rate-network bug is REAL — do NOT dismiss as harmless (user, 2026-06-28).** The
in-situ NLTE rate network produces UNPHYSICAL output in the COLD shells (Te~2500-3000K)
across ALL ions: **negative populations** (O II shell-24: 115/202 NEGATIVE) and
**super-thermal S_l**, and — the load-bearing failure — it relaxes the populations to
**near-LTE (b_k≈1) instead of the correct NON-LTE state with fluorescence/scattering**.
Fe II (5088 lines, 55%), Co/Ni/Ti/Mn/Cr II, O I/II, C/Si/S/Ca II (13 ion species).

### What 2026-06-28 falsifiers NARROWED (every run config-verified via RESOLVED CONFIG dump)
These RULE OUT specific causes — they do NOT make the bug go away:
- **JEQB (force bb J=B):** garbage unchanged (19.4% vs 19.5%) → not the bb radiation field.
- **BF_JEQB (force bf continuum J=B, full-LTE):** unchanged → not the bf radiation field.
- **ARTIS iterative row-col equilibration:** byte-identical → NOT numerical conditioning
  (the O II solve is already accurate: cond 7.8e11, SVD resid 9e-6).
- **b_k direct (LEVELPOP_DUMP, full-LTE):** the b_k=1e19 blow-up sits at high-E levels with
  n_k≈1e-7·ground; the POPULATED levels relax to b_k≈1 (near-LTE).

⟹ The bug is **structural** (in the rate formulas / the binned-J system that feeds them),
NOT numerical and NOT a single mis-set radiation field. Two faces of the same defect:
1. genuinely unphysical numbers (negative pops, super-thermal weak-line S_l) — a real
   detailed-balance / assembly defect even where currently sub-dominant; and
2. the systemic failure: the network cannot pump fluorescence, so it returns near-LTE
   populations → **thermal line sources (S_l=B(Te))**.

### Why this IS the spectrum killer (A1, 2026-06-28 — figures/spectrum_vs_gold*_2026-06-28.png)
The emergent spectrum does NOT match gold. Smoothed to gold resolution:
- **freqres (freq-resolved field):** right color envelope but **FEATURELESS** — no 6590
  peak, no 7700 trough (shape-corr 0.66 = broad slope only). A noisy thermal continuum.
- **formal (binned):** features present but wrong color (peak 9212, too-red; corr 0.47).
- Neither reproduces gold's line-formed structure.

A **thermal line (S_l=B(Te)) fills in its own absorption → no net feature.** Gold's
features need lines that **SCATTER (P-Cygni) + BLANKET** = NON-LTE line sources. The
near-LTE populations from the rate-network bug are exactly what prevents this. This is the
N4 capstone confirmed at the spectrum level: **binned-J populations structurally cannot
fluoresce.** Same root as the month-long too-red/fluorescence/A4 problem.

### Why the raw binned-J field is NOT harmless — ARTIS proves it (user, 2026-06-28)
ARTIS deliberately feeds its rate network **smooth fitted radiation values**, NOT a raw
binned-J integration: continuum/bf use a per-cell **dilute blackbody (W, T_R)** fit (or a
smooth precomputed α(T_R) LUT, ratecoeff.cc:700), and bb fluorescence uses **per-line
J_blue** estimators (radfield.cc:704). **If the noisy raw field were harmless, ARTIS would
never have engineered around it.** Their design IS the evidence the defect is real and
load-bearing. Our raw 1000-bin J_nu at cold shells is both noisy (Wien-cutoff collapse) and
NOT the right non-local field → it drives the rate network to garbage + near-LTE.

Note the falsifier nuance: BF_JEQB forced the field to **B(T_e)** = smooth but *local-thermal*
(the LTE limit, which cannot fluoresce) → no change because the populated levels were already
near-thermal. ARTIS's **W·B(T_R)** is smooth but *non-local* (T_R > T_e at outer shells,
diluted by W) → it carries the photospheric color that PUMPS the transitions. So the fix is
NOT "force LTE"; it is "replace the raw binned-J with a smooth NON-LOCAL field."

### Next
1. **Adopt ARTIS's smooth radiation-field representation for the rates** (the real fix):
   continuum/bf ← dilute-BB (W, T_R) fit or α(T_R) LUT; bb fluorescence ← per-line J_blue /
   freq-resolved J̄_l, replacing the raw 1000-bin J_nu integration (plasma.c:7232-7300 etc.).
2. Wire the freq-resolved line-specific J̄_l into the NLTE **population** solve
   (LINERES_CONSUME / LUMINA_NLTE_JBAR_POPS) — currently it reaches only the emergent
   extraction while populations stay binned-J→near-LTE. Goal: non-LTE pumping →
   scattering/fluorescence line sources → recover gold's 6590 peak / 7700 trough.
3. Fix the unphysical negatives / super-thermal weak-line S_l (detailed-balance / assembly),
   not clamp over them.

⚠️ Lesson banked: env/build silent-misconfig cost many no-op runs (BF_JEQB, EQUILIBRATE,
POP_Z not reaching the binary). FIXED by the RESOLVED CONFIG dump (cuda.cu main prints the
actual env the binary received) — verify it on EVERY run before trusting a result.

## Stage T — Observer-frame transport (the obs emergent extractor)
| ID | Controlled case | Known answer | Catches |
|----|-----------------|--------------|---------|
| **T1** | single resonance line, pure scatter (ε=0), homologous flow, BB core | **net EW = 0** (exact photon conservation) + Castor/SEI P-Cygni shape; τ_S→∞ ⇒ blue-edge flux→0 | the 0.47 leak / 1.72 over-emit / featureless bugs, isolated, no plasma |
| T2 | pure continuum (no lines), β→0 | obs-march == static extractor exactly | Doppler/D³ bookkeeping in the no-Doppler limit |
| T3 | grey Thomson-scattering sphere over BB core | J → W·B(T_inner) dilution (analytic W(r)); emergent = diluted BB | the continuum beaming / dilution |
| T4 | two overlapping lines, pure scatter | net EW = 0 (both); known overlap | line-overlap accumulation (if T1 passes but full fails) |

## Stage F — Comoving radiation field (`cmf_solve_J` formal solve)
| ID | Controlled case | Known answer | Catches |
|----|-----------------|--------------|---------|
| F1 | isothermal, pure absorption (no scatter), thick | J → B(T) (thermalization) | the ALI/source closure |
| F2 | pure scattering, no absorption, closed | ∮ conserved (no destruction) | scattering conservation |
| F3 | thin scattering halo + BB core | J = W·B(T_inner), analytic W(r) | dilution transport |
| F4 | CMF advection: single Sobolev line | escape probability β = (1−e^{−τ_S})/τ_S | the a_lam advection term (the Courant-80 fix) |

## Stage N — NLTE populations / line source functions
| ID | Controlled case | Known answer | Status | 비고 (ARTIS) |
|----|-----------------|--------------|--------|--------------|
| **N1** | two-level atom | S_l=(J̄+εB)/(1+ε) | ✅PASS (num==ana exact) | — |
| **N2** | high-collision LTE limit (non-thermal field) | ne→∞ b_i→1, S_l→1 | ✅PASS | — |
| **N3** | detailed balance A_ij n*_j=A_ji n*_i | symmetric | ✅ (≈0.5%) | — |
| **N4** | 3-level UV pump→fluorescence, binned vs freq J̄ | freq pumps b_2>1; binned b_2≈1 | ✅PASS **CAPSTONE** | **ARTIS**: per-line J_blue (Lucy 2002, radfield.cc:704 / macroatom.cc:571) = our jbar_line_det; 256-bin bypassed → confirms our fix |
| **N5** | NLTE matrix conditioning (250-order span) | equilibration recovers x_true where raw LU fails | ⚠️ toy FALSIFIED (raw LU OK to 200 orders; eq-alone hurts) | **ARTIS**: row-col equilibration `nltepop_matrix_normalise` (nltepop.cc:733) f=√(col/row) 10× + PartialPivLU+refine — applied AFTER super-levels |
| **N6** | super-levels (collapse high-E manifold) | K explicit + Boltzmann super-level recovers full low-level pops | ✅PASS standalone (24→7, err<0.5%) + ✅IMPLEMENTED in plasma.c (machinery already complete; added ARTIS cutoff `LUMINA_SUPER_CUTOFF=K`, O II 340→101, total 25620→4651 @K=100). ⏳ in-situ test pending resources | **ARTIS**: `superlevel_boltzmann`/s_renorm = our within_sl_frac (plasma.c:8301, Boltzmann, ARTIS-match). f_to_s only collapsed Fe-group; ARTIS cutoff added for O II et al. |
| **N7** | ionization+excitation coupling | one matrix, simultaneous | ✅ (already one matrix) | **ARTIS**: `solve_nlte_pops_element` bb+ionisation+nt+autoion in ONE matrix (nltepop.cc:1220) |

## Stage P — Plasma (T_e, n_e, ionization)
| ID | Controlled case | Known answer | Catches |
|----|-----------------|--------------|---------|
| **P1** ✅PASS | 2-ion Saha LTE limit (photoion field on) | ne→∞ ratio→1 | ionization rate network → Saha at high ne (LTE limit correct) |
| **P2** ✅PASS | gray radiative equilibrium, J=W·B(Trad) | Te=W^{1/4}·Trad | energy-balance root-finder exact (ratio 1.000-1.004) |

## Stage A — Full assembly
| ID | Controlled case | Known answer | Catches |
|----|-----------------|--------------|---------|
| A1 | the full pipeline on DDC15 | gold emergent (peak 6595, grn/nir 0.58, P-Cygni) | the wiring — only trusted AFTER T/F/N/P pass |

---

## Rules
- **No test without a known answer.** (The anti-wander rule.)
- Bottom-up: don't trust Stage N until F passes, etc.
- Each test: a self-contained synthetic setup (no full pipeline) where possible — `lumina_cmf_selftest.c`.
- Measure converged only; one knob per variation.
- A "previously passed" stage (plasma T_e/n_e via gold) is NOT trusted until its *controlled* (analytic-limit) tests pass — gold-scalar agreement ≠ analytic correctness.

## Status
- **T1 — DONE (170712 scatter / 170713 SEI), single line tauS=100, analytic net EW=0:**
  - **SCATTER: net EW +79.8A (+13%), CORRECT P-Cygni** (blue 0.10 deep, red 1.45 emit). Mild over-emit bug.
  - **SEI: net EW +1080A (+180%), BROKEN** (no absorption, all 3x emit). Jbar_C source wrong → abandon SEI.
  - **Key:** obs-march CAN make P-Cygni (scatter); the +13% per-line over-emit COMPOUNDS over the forest →
    fills troughs → explains the full-run shallow features. SEI was a wrong fix.
  - **+13% characterized (170714/15/16):** CONSTANT across tauS 1→100 (12.5→13.3%) AND DVRES 30→10 (invariant)
    → GEOMETRIC, not source-self-consistency-scaling, not freq-discretization. Remaining: Wd·B point-dilution
    source ≠ true comoving J̄ (code authors flagged W·B non-conserving, line 1054) OR ray-angle discretization.
    NS=40/80/160 ALL byte-identical +13.3% → NOT angle-discretization. g_sob_rphot set correctly (line 1620) → Wd r-dependent OK. +13% = D³-beaming balance OR fixed-backlight scatter approx. Testing beta->0 (static, D=1) to separate Doppler/D³ from static source balance. SEI=double-beaming bug (jc already D⁴-beamed × D³ at line 1084).
- **T1 VERDICT (170712-170764, definitive): single-pass prescribed-source obs-march CANNOT conserve a pure-scatter line.**
  Sources tested: W·B (+13%/−12% beta-full/~0), producer self-consistent J̄_l (−7%/+47%), SEI jc (broken, double-beam).
  ALL net EW ≠ 0; sign flips per source×beta. D³ is CORRECT (physics-agent, Lorentz invariants). producer J̄_l is a
  CONVERGED comoving solve (not under-converged) yet still −7% → the residual is geometry/quadrature mismatch between
  the comoving J̄ solve and the obs-march emergent integral, NOT a source-magnitude bug.
  **⟹ Exact conservation of a scattering line needs EITHER (P-γ) Λ-iterating the obs-march's OWN J̄ to self-consistency,
  OR (P-β) a photon-conserving method = MC (TARDIS/ARTIS standard, conserves by construction). This is the
  deterministic-vs-MC fork, now PROVEN on a known-answer single line.**
  Best deterministic option for gold-match: producer J̄_l (−7% energy, DEEP trough blue 0.374 — gold wants deep dips).
- **F1 (cmf_solve_J thermalization, 170772) PASS:** J/B=0.997-0.998 all shells → thick pure-absorption thermalizes to B. Solver thermalization sound.
- **F3 (cmf_solve_J dilution, 170773) MIXED:** J/(W·B) inner s0=1.48, mid s20=1.02, outer s35=0.76. Mid OK, inner +48%/outer −24%.
  tau_es sweep: taues=1.0 all J/(W·B)>1 (physical scatter diffuse). **taues=0.01 (pure dilution, negligible scatter):
  J/(W·B) declines 1.44(s0)→0.62(s35) — solver does NOT reproduce W·B geometric dilution, ~38% deficit outer.**
  Either (a) outward flux-loss bug or (b) W·B reference idealized. **RESOLVED (170776/7/8): NS sweep — inner overshoot
  1.44→1.19 = discretization (converges). Outer deficit persists at 0.68 (NS-independent). advection OFF (ALAM=0):
  J/(W·B)=1.04 FLAT all outer shells → the outer deficit was 100% CMF EXPANSION REDSHIFT (fixed-freq measure), NOT
  flux loss. ⟹ F3 PASS: cmf_solve_J reproduces W·B dilution AND correctly applies expansion redshift. SOLVER SOUND.
  T1b producer-J̄ −7% is therefore the transport-METHOD limit (T1 verdict), NOT a solver bug.**
- **F1+F3 ⟹ cmf_solve_J (the foundational comoving field solver) is CONFIRMED SOUND** (thermalization + dilution +
  redshift all correct). The plasma/producer J̄ rest on a sound solver.
- **N1 (two-level, standalone) PASS:** rate-solve S_l == analytic (J̄+εB)/(1+ε) exact; ε→0 S_l→J̄ limit ✓. NLTE 2-level machinery sound.
- **N4 (3-level UV-pump fluorescence, standalone) PASS — SAGA CAPSTONE:** FREQ-resolved UV pump (J̄=2.1×B(Te)) → b_2=1.99, S_l(2→1)=3.96×B = FLUORESCENCE. BINNED (UV diluted to B(Te)) → b_2=0.96, S_l=1.85×B = NONE. ratio ≈2.1×.
  **⟹ binned-J populations STRUCTURALLY cannot pump fluorescence (dilutes super-thermal UV to local-thermal). This is the controlled, known-answer confirmation of the month-long too-red/fluorescence root cause.**
  **IMPLICATION for P-β: MC emergent on the CURRENT binned-J plasma will MISS fluorescence regardless of transport quality — the populations MUST go freq-resolved for gold-equivalence. N-stage orthodox-ization is essential, not optional.**
- T4/P — queued; transport-method fork (T1 verdict) → user chose P-β (MC emergent).

## Links
[[project_autonomous_stage2_2026-06-25]], docs/LADDER_V2.md (the failure history this suite addresses),
docs/ORTHODOX_FREQRES_NLTE_DESIGN.md.
