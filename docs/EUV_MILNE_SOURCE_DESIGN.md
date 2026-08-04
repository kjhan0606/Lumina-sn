# Exact per-level Milne fb recombination — design spec (SUPERSEDES the case-B version)

**Date:** 2026-07-21 (rev 2 — exact routine, per user's "no approximation-and-test" directive)
**Status:** DESIGN / awaiting driver confirmation of scope. No code written.
**Principle:** [[feedback-exact-physics-not-approx-test]] — implement the correct routine (detailed-balance-exact, correct by construction for all epochs/compositions), not an approximation validated per-case. The case-B redirect (fbcb) and FB_COOL_KT are re-classified as **diagnostics that confirmed the mechanism**, NOT the fix.

---

## 1. Confirmed diagnosis (keep)

The photospheric Fe III→IV over-ionization is a **super-Planckian free-bound source**: at s8, mc_J/B_ν(T_e) = 693–1180× in 404–520Å while CMFGEN is 0.06–0.08× (sub-Planckian), giving Fe III Gph = 2209× CMFGEN. Root: the fb (recombination continuum) re-emission violates the Kirchhoff/Milne relation with the bf opacity.

## 2. Why the current fb is not detailed-balance-exact (the structural gap)

| | ARTIS (exact) | Lumina (current, approximate) |
|---|---|---|
| resolution | **per level** — loops all ionising levels, each level's edge (`kpkt.cc:150-163`) | **per ion** — one ground edge per ion (`plasma.c:2360-2378`, `find_ioniz_energy(Z,stage-1)`) |
| cooling coeff | exact Milne integral `∫σ_bf(ν)(ν−ν_edge)(2H/c²)ν²e^(−h(ν−ν_edge)/kT_e)dν`, tabulated over T_e (`ratecoeff.cc:82-89, 173-182`) | `α_RR(total)·(hν₀+kT_e)` — Kramers/frozen-in α × energy (`plasma.c:2388-2389`); FB_COOL_KT just swaps the energy factor to kT_e |
| emission spectrum | σ_bf(ν)-weighted Milne draw (`select_continuum_nu`) | σ=const thermal tail `ν=ν_edge−(kT_e/h)ln ξ` (`cuda.cu:3948`) — wrong spectral shape |
| population | NLTE level pop (`kpkt.cc:155-160`, `BFCOOLING_USELEVELPOPNOTIONPOP`) | ion density `n_ion` |

Because emission is per-ion-ground with a Kramers amplitude and a σ=const spectrum, the fb emissivity `j_ν` is **not** `χ_bf(ν)·B_ν·[NLTE departure]`. Detailed balance is broken → S_ν = j_ν/χ_ν is super-Planckian in the Wien EUV. **No wavelength cutoff, energy-factor swap, or thick-cell redirect can restore per-level Milne balance** — only emitting the correct per-level recombination continuum can.

## 3. The exact routine (port ARTIS's per-level bf recombination)

Lumina already has the two ingredients: **per-level CMFGEN σ_bf** (`cmfgen_sigma_bf.bin`, global-level indexed, verified byte-exact by the Gph decomposition) and **NLTE level populations**. The fb channel is rebuilt to use them, per level, exactly:

### C1 — per-level Milne cooling coefficient (amount)
Precompute, per ionising level i and per T_e (a table like ARTIS `bfcooling_coeffs`):
```
Λ_i(T_e) = 4π·(Saha* factor)·∫₀^{νmax−ν0,i} σ_i(ν)·(ν−ν0,i)·(2H/c²)·ν²·exp(−h(ν−ν0,i)/kT_e) d(ν−ν0,i)
```
Charges the electron pool the **photoelectron KE (ν−ν0,i)** only (ARTIS convention). The k-packet fb exit weight is then `C_fb = Σ_i n_i^{NLTE}·n_e·Λ_i(T_e)` — replacing `α_RR·(hν₀+kT_e)`. This feeds `p_fb` and the per-level edge CDF.

### C2 — per-level Milne emission spectrum
When a k-packet exits via fb, pick level i from the per-level cooling-weighted CDF, then draw ν from the **σ_i(ν)-weighted Milne spectrum** `∝ σ_i(ν)·ν²·exp(−h(ν−ν0,i)/kT_e)` (ARTIS `select_continuum_nu`), replacing the σ=const thermal tail. Needs σ_i(ν) on device (extend the fb-edge device arrays from per-ion to per-level, carrying a σ-table handle).

### C3 — NLTE population consistency
Use `n_i^{NLTE}` in C1/C2, the same populations as the bf opacity χ_bf uses in transport (`cuda.cu` continuum opacity). Then j_ν and χ_ν share σ_i and n_i ⇒ **S_ν = j_ν/χ_ν = B_ν(T_e)·[NLTE departure] by construction** — sub-Planckian exactly where the recombined level is sub-thermal, at every epoch/composition, with no tuning.

### C4 — remove the approximations
Delete from the fb path: the case-B/OTS redirect (`FB_OTS*`, the `d_kpr_bteq_draw` fb target), FB_COOL_KT (subsumed — the KE convention is now intrinsic to C1), and the α_RR×energy weight. These become dead gates. (The **line**-channel k-packet thermalization — BSRC/TE_POP — is a separate front, out of scope here; note it is the parallel exactness task for bound-bound.)

## 4. Scope / effort (honest)

This is a real re-architecture of the fb channel (per-ion → per-level), essentially an ARTIS port, not an env gate:
- **New:** per-level Milne cooling table `Λ_i(T_e)` (host precompute over σ_bf, ~like ARTIS ratecoeff.cc); per-level fb edge CDF; device σ_i(ν) access for the emission draw.
- **Changed:** `plasma.c:2360-2432` (fb weight build → per-level Milne), `cuda.cu:3902-3960` (fb emission draw → σ_bf-weighted Milne), population source → NLTE.
- **Removed:** FB_OTS/case-B fb redirect, FB_COOL_KT, α_RR×energy.
- **Cost driver:** the per-level cooling integral × N_levels × N_Te table (precompute, one-time per run) and the per-level device arrays (memory: KPKT_FB_NEDGE → per-level is larger; may need a capped per-shell level list by cooling weight, as ARTIS does with a coolinglist).
- Estimated: a focused multi-hour Opus implementation + one toy06 validation run.

## 5. Validation — ONCE, then it generalizes

Because the routine is detailed-balance-exact, validate **once** on toy06 @19.48d; correctness then holds by construction for other epochs/compositions (the whole point).
- **PASS:** s8 mc_J/B_ν in 404–520Å → CMFGEN's 0.06–0.08×; Fe III Gph → ~1× CMFGEN; f(FeIV) s8 → ~0.022; recombination transition sharpens to CMFGEN's v~7–8k; deep s0 held ~0.98.
- **Sanity:** optical/IR field unchanged (already correct, mc/B≈0.4); energy conservation in the fb draw; per-iteration convergence (the NLTE loop — watch f(FeIV) s8 trace; under-relax if needed).
- Tools: `scripts/euv_planck_check_s8.py` (now argv-driven), `fe4_readout.py`, profile vs `ionfrac_fe_toy06_cmfgen.txt`.

## 6. Gate policy
Implement as the **default-correct** fb routine behind a single master gate `LUMINA_FB_MILNE_EXACT=1` for a clean A/B vs the current champion (OFF = byte-identical). Once validated, it becomes the default and the approximate gates (FB_OTS/FB_COOL_KT/α_RR fb path) are retired. This is not "another gate to tune" — it is one exact routine, gated only for the single validating A/B.

---

**Decision requested:** confirm the scope — implement the exact per-level Milne fb (ARTIS port), accepting it is a multi-hour re-architecture rather than a quick change. This is the correct-by-construction path; the alternative (keep approximating and validating per-case) is the one your directive rules out.

---
## C2 implementation plan (σ_bf-weighted emission) — concrete, for clean completion

**Status:** Increment 1 done (fields added, compile-safe): `kpacket_fb_edge_lev` (lumina.h host opacity), `d_kpkt_fb_edge_lev` + `d_fb_sigma_bf` (lumina_cuda.cu device). Unused fields → tree still compiles.

**Remaining (do + build as ONE unit):**
1. **plasma.c C1 loop** — declare `int e_lev[KPKT_FB_NEDGE];`; on every append/evict set `e_lev` (OFF path = -1; ON per-level path = `l`, the global level index); in the CDF finalization add `int *o_lev = opacity->kpacket_fb_edge_lev + s*KPKT_FB_NEDGE; o_lev[q]=e_lev[q];`.
2. **cuda.cu device array** — mirror `d_kpkt_fb_edge_zs` exactly for `d_kpkt_fb_edge_lev`: NULL-init, cudaMalloc(ne*int), cudaMemcpy upload from `opacity->kpacket_fb_edge_lev`, cudaFree. (grep the 4 `d_kpkt_fb_edge_zs` sites.)
3. **expose sigma_bf** — add `extern "C" const float* bf_gemm_get_d_sigma_bf(void)` in lumina_bf_gemm.cu returning `g_bf_gemm.d_sigma_bf` (col-major, sigma[l] contiguous at `+l*NLTE_N_FREQ_BINS`); at device setup store it into `dev->d_fb_sigma_bf` (only when bf_gemm initialized).
3. **cuda.cu emission (the σ-weighted draw, replaces the σ=const thermal tail at ~cuda.cu:3947-3956):** when `d_fb_sigma_bf && lev>=0`, rejection-sample: propose `nu_c = nu_edge - (kTe/h)ln(ξ)`; map to bin `f=clamp(log(nu_c/NLTE_NU_MIN)/dln)`; read `sig=d_fb_sigma_bf[lev*NLTE_N_FREQ_BINS+f]`, `sig0=d_fb_sigma_bf[lev*NLTE_N_FREQ_BINS+f0]` (f0=edge bin); accept with `min(1, (sig*nu_c*nu_c)/(sig0*nu_edge*nu_edge))`, else redraw (cap ~8 tries → fallback thermal). This suppresses the far-blue recomb tail (σ decreasing) that the σ=const tail over-produces — the blue-tail→bluer-band super-Planckian feeder. Needs the edge's `lev` from `d_kpkt_fb_edge_lev` (thread it to `d_kpr_bteq_draw`/the fb block scope).
4. **gate** — all under `LUMINA_FB_MILNE_EXACT` (C1 already gated); C2 auto-active when the gate is on and sigma handle present. OFF = byte-identical.
5. **build** `lumina_cuda.withMilne` (rebuild), verify compile + `[FB-MILNE]` + a σ-draw counter.

**Note:** C2 is exact-routine-necessary (not gold-plating) regardless of the C1 milne result — the σ(ν) weighting is the physically-correct emission spectrum; completing it makes the fb routine fully detailed-balance-exact.
