# Deterministic Emergent — Bug/Problem Resolution Table (2026-06-25)

Goal: one deterministic integrator producing **P-Cygni profiles + correct color +
fluorescence**, comparable to DDC15 CMFGEN gold (peak 6790Å, grn/nir 0.58).
Current best pieces: plasma ✅ (T_e 0.98, n_e dex 0.18); static-freqres color ✅
(peak 6782); P-Cygni ✅ via scatter source on *binned* field (contrast 2.0-2.4).
The unification has the bugs below.

## Problem inventory (observed today)

| ID | Problem | Evidence | Root-cause hypothesis |
|----|---------|----------|----------------------|
| **P1** | obs Doppler march **too-red** | 169874 scatter peak 11826 vs static-freqres 6782 on the *same fine field* | Doppler obs march (`cmf_obs_march_sob`) reddens the continuum: either D³ weighting, the `S_c=(χ_ab·B+χ_es·J)/χ_c` form, or `nu_cmf` interpolation. **The #1 blocker.** |
| **P2** | red-edge upturn | 169874 flux rises at 11000-12000 | `nu_cmf=q·nu_obs` leaves fine window `[nu_lo,nu_hi]` → `cmf_interp_nu_asc` clamps to edge value → spurious flux |
| **P3** | fluorescence pump **FAIL** | 169838 green Fe II S_l/B≈1.000 with CONSUME=1 | confounded: **P3a** TAUMIN=0.1 may have skipped the UV forest pump (count unknown — FINE_DIAG wasn't plumbed); **P3b** orthodox floor thermalizes pumped upper levels |
| **P4** | Fix-B/Fix-C not built | design only | genuine super-thermal fluorescence S_l would be killed by clamp (Fix-B) + orthodox floor (Fix-C) even if pump works |
| **P5** | cost: obs emergent every iter | ~28 min/iter × 8 | producer+obs emergent rerun each NLTE iteration; only the converged one is needed |
| **P6** | two-window (pump vs emission) | UV forest deposit blew up (169831) | pump needs UV 1000-3000 (line-resolved J̄), emission needs optical 3000-12000; one fine window can't cheaply cover both |
| **P7** | harness plumbing gaps | FINE_DIAG never reached binary (169873 wasted) | harness passes an explicit env allowlist; new gates must be added |

## Resolution plan (ordered by dependency)

| # | Step | Fix / action | Predicted result (PASS gate) | Contingency (if prediction FAILS) |
|---|------|--------------|------------------------------|-----------------------------------|
| **1** | **Isolate P1** ✅ DONE | Ran obs `OBS_CONTONLY=1` (169882) on fine field | **RESULT: CONTONLY-obs peak=6513 ≈ gold 6610 → Doppler CONTINUUM march is COLOR-CORRECT.** P1 is NOT a continuum bug. The too-red (full-scatter 8576) is the LINE path → line blanketing without fluorescence = **P1 ≡ P3**. | — |
| **2** | ~~Fix P1 continuum~~ **NOT NEEDED** | — | continuum march sound (step 1 verdict) | — |
| **3** | **P-Cygni + color milestone** | With P1 fixed, full scatter obs (lines on) on fine field | peak ≈ 6800 (gold 6790) AND P-Cygni contrast: blue >2, CaII-NIR >2 (like standalone 2.0-2.4) → **MILESTONE: P-Cygni + correct color**. Draw figure vs gold | P-Cygni weak (contrast <1.5) → output grid too coarse (raise NObs) or DVRES too large (lower `OBS_DVRES` to 20 km/s) |
| **4** | **Fix P5 cost** | Gate obs emergent to the final converged iter only (pass iter index into `cmfgen_fine_jbar`, or run a one-shot post-convergence emergent call) | run time ≈ baseline (~18 min/iter) + one obs pass (~10 min); total ~2.5 h not 4+ h | If iter index unavailable cheaply → run N_ITER=8 plasma-only first, then a separate single emergent pass reading the converged state |
| **5** | **Diagnose P3 pump** | Short run `FINE_DIAG=1 TAUMIN=0.1 UV-window CONSUME=1 N_ITER=3` (FINE_DIAG now plumbed); read skipped-weak count + lines-in-window | skipped ≪ in-window (pump lines survive) → P3a ruled out → P3b (floor) is the cause → step 6 | skipped ≈ all (TAUMIN killed the forest pump) → P3a confirmed → lower TAUMIN (0.01) or accept the dense forest with a one-time cost; re-test |
| **6** | **Build Fix-C (floor exemption)** | In `cuda.cu` orthodox floor (955-993): exempt levels that are the upper level of an in-window line with `J̄_pump > 1.5·B(T_e)` from the LTE floor | green Fe II S_l/B rises >1 in mid-vel shells (vs control =1); plasma gate held (T_e 0.98, n_e dex 0.18) → **fluorescence pump LIVE** | S_l still ≈1 → pump rate not reaching the matrix (CONSUME wiring) → trace `det_jbar` firing for Fe II upper-level rows via a one-shot dump |
| **7** | **Build Fix-B (b-ceiling)** | Replace `FINE_SL_CLAMP` scalar with per-level `b_j ≤ B(T_phot)/B(T_e)` cap gated on pump provenance (design §Fix-B) | super-thermal green S_l (1<S_l/B<10²) survives to emergent; cold-shell garbage stays clamped; no NaN/blowup | b-ceiling over-clamps (kills green) → T_phot mis-set (MC blue-tilt 168221) → validate T_phot first; or under-relax ω=0.25 |
| **8** | **Fluorescence emergent** | Two-window producer (P6): UV pump 1000-3000 (TAUMIN, coarse PPD) ∪ optical 3000-12000 (fine) → jbar_line_det for both; NLTE source obs emergent | grn/nir 0.41 → toward 0.58; green Fe II emission appears; P-Cygni preserved → **FULL deterministic emergent** | grn/nir flat despite S_l>1 → 2nd root (green Fe II opacity/ionization too low) — not a source-function problem; pivot to opacity/ionization audit |

## Predicted end state
After step 3: **P-Cygni + correct color** figure vs gold (fluorescence green still
missing — the honest near-term milestone). After step 8: **full deterministic
emergent** with fluorescence. Steps 1-4 are the immediate critical path
(unblock + milestone + make iteration affordable); 5-8 are the fluorescence
science. Each step has a falsifiable gate so we never proceed on a wrong assumption.

## Cross-cutting rules
- One change per run; verify config tag + flux scale + producer ran (FINE_DIAG) before trusting a result ([[feedback_verify_before_concluding]]).
- Launch emergent runs via `scripts/run_emergent.sh` (10-knob guardrail, [[feedback_emergent_launch_wrapper]]).
- Plasma gate (T_e≥0.95, n_e dex≤0.3) on every run — any regression = stop.
- Design reference: docs/FLUORESCENCE_DESIGN.md; status: [[project_autonomous_stage2_2026-06-25]].
