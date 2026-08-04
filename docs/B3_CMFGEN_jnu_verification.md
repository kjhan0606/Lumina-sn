# B3 direction verification — deterministic J_ν photoionization vs scalar-T_rad nebular closure

**Date:** 2026-06-09
**Purpose:** Independent confirmation (codex + a physics-review agent) of the CMFGEN method, to decide whether to abandon the scalar-T_rad Mazzali-Lucy nebular ionization closure and build a deterministic frequency-dependent J_ν photoionization solver with Λ*/VEF coupling into the per-shell Newton. Saved for comparison against forthcoming run results (de-risk smoke 164726 and the B3-1 implementation).

---

## Problem being fixed

Transition zone (τ~1, W≪1, Fe/Co/Ni UV blanketing) of DDC15 0.976d. LUMINA holds carbon ~100% C II; CMFGEN keeps that C/O layer ~95% NEUTRAL (mean charge 0.05→0.51). Root cause: coupled-Newton Γ=α·φ_neb with φ_neb's W·(1−ζ)·φ_LTE(**T_rad**) term, driven by a spuriously hot/noisy binned T_rad (7–19 kK vs gas T_e 4–12 kK). The φ_LTE(T_rad) factor is exponentially sensitive → over-ionizes C by ~10⁴.

**Estimator-toggle A/B (164710 binnedJ vs 164711 first-moment) PROVED it is NOT an estimator bug:** cooler T_rad did not free carbon (100% C II in both arms); first-moment additionally collapsed the far-outer (W→2–22 unphysical, sh48 n_e→1, T_e→956 K). Only force-locking ionization to φ_LTE(T_e) cures it.

---

## Verified CMFGEN method (5 claims, all CONFIRMED)

Primary sources: Hillier & Miller 1998 (ApJ 496, 407); Hillier & Dessart 2012 (MNRAS 424, 252) eqs. 26–27; Dessart & Hillier 2010; Blondin et al. 2013/2014.

1. **No scalar T_rad for ionization.** CMFGEN carries comoving-frame J_ν on a >10⁵-point frequency grid and forms `Γ_i = ∫_{ν0}^∞ (4πJ_ν/hν) σ_i(ν) dν` (the R_ij in HD2012 eq.26 are these integrals). T_rad is ONLY an initialization/diagnostic device, never an ionization state variable. → scalar-T_rad is structurally unfixable for the C I ~1100 Å FUV edge.

2. **VEF closure.** f_ν = K_ν/J_ν from a ray-by-ray formal solution of the (time-independent) relativistic CMF transfer equation, closing the moment equations for J_ν, H_ν. NOT a T_rad/W parameterization. **Precision:** f_ν is computed from a LAGGED formal solution and held fixed within a linearization step (outer VEF iteration) — so VEF is NOT in the Jacobian; only the local Λ* response is.

3. **Gas temperature.** Steady: `4π∫(χ_ν J_ν − η_ν)dν = 0`. Time-dependent SN (HD2012 eq.27): `ρ D(e/ρ)/Dt + P D(1/ρ)/Dt = 4π∫(χJ−η)dν + Ė_decay`, with `D(1/ρ)/Dt = 3/(ρt)` (adiabatic cooling) and e = thermal + excitation + ionization energy. CONFIRMED.

4. **Ionization / n_e.** Time-dependent SE rate equations `ρ D(n_i/ρ)/Dt = Σ(n_j R_ji − n_i R_ij)` (radiative+collisional+dielectronic+charge-exchange+Auger+non-thermal), closed by charge conservation `n_e = Σ Z_ion n_i`, with n_e an explicit Newton unknown — NOT Saha/dilute-Saha. The Lagrangian ∂/∂t term enables ionization freeze-out. **Caveat:** super-levels, not every individual level as an independent unknown.

5. **Single coupled linearization.** Transfer is linearized to `δJ_ν ≈ Λ*_ν[δη_ν − J_ν δχ_ν]` as a function of δn_i, δn_e, δT, substituted into the rate+energy equations. **Precision:** PARTIAL (not complete) linearization — diagonal OR tridiagonal Λ* + Λ-iteration/Ng acceleration; "single global complete-linearization Newton" overstates it. Block-tridiagonal in radius for τ≫1. Operator-split (frozen-J Gauss-Seidel) FAILS because it uses a stale Λ response — exactly the LUMINA pathology, and why the A2 lagged-binned-J_ν attempt (164542) regressed.

---

## VERDICT (codex + physics agent agree)

**Build deterministic J_ν photoionization with Λ*/VEF coupling into the Newton; abandon the scalar-T_rad ionization closure. There is no fully faithful scalar-T_rad rescue.**

**MANDATORY for C/O at τ~1:**
- (a) frequency-resolved J_ν spanning the FUV edges (C I ~1100 Å, O I ~910 Å) — bolometric/SED T_rad is the wrong variable regardless of estimator.
- (b) the **Λ* response δJ_ν(δn_i, δT_e) INSIDE the Newton** — diagonal Λ* is the MINIMUM that converges at τ~1 (dilute, near-coherent scattering). The lagged-J regression is direct evidence this is non-optional.

**Nice-to-have (NOT required for transition zone):** full VEF / tridiagonal Λ* depth coupling — that is the τ≫1 refinement (A4).

**Cheaper alternative assessed:** a blanketed two-band (FUV vs optical) W·B color temperature is defensible ONLY as a bridge/initialization, NOT as the converged ionization state variable (a frozen two-T J reproduces the regression in milder form).

---

## Key code gap identified (the thing B3-1 must implement)

In `lumina_plasma.c`, the existing `LUMINA_COUPLED_JNU_PHOTOION` path computes `gamma_jnu[ip]` via `coupled_photoion_rate_jnu` (plasma.c:4033) reading the **lagged `nlte->J_nu`** (plasma.c:4538-4543). The existing Λ* (`LUMINA_COUPLED_LAMBDA_STAR`, gbin/lstar/blag at 4513-4531) is wired ONLY into the RADEQ bf-heating / radeq_net thermal balance — it does NOT touch the photoionization rate. Hence JNU+LSTAR together (job 164698) still regressed: Λ* never reached the photoion integral.

**B3-1 = wire the diagonal-Λ* response into the J_ν used for the photoionization integral**, so in the transition zone J_ν → blended toward W·B(T_e) by Λ*=1−e^{−τ_bf}, pulling carbon ionization down to the trial T_e instead of running off the lagged hot field.

Existing reusable assets: `coupled_photoion_rate_jnu` (Γ integral done), `lumina_cmfgen.c` ray solver + diagonal lambda_star (thick J/S=1.000 validated, thin-UV diagnosed).

---

## Runs to compare against this verdict

- 164710 — baseline (binnedJ=1, JNU=0, LSTAR=1, 120k smoke): all-RMS 0.298, carbon 100% C II.
- 164711 — first-moment estimator (failed): all-RMS 1.832.
- 164698 — JNU=1 + LSTAR=1, 200k (regressed): all-RMS 0.598.
- **164726 — de-risk smoke: JNU=1 + LSTAR=1 at 120k** (apples-to-apples vs 164710). Expected: re-confirm regression / no carbon relief, since Λ* is not wired to the photoion rate. RESULT: _pending_.
- B3-1 implementation run: _pending_.
