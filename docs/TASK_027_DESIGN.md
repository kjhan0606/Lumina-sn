# Task #27 — Radiation-field estimator / T_inner overshoot fix

**Status:** design (2026-06-02)
**Goal:** stop the iron-peak from over-ionizing to III at convergence (loss of UV iron-curtain blanketing → ~33× UV leak). Triangulated root cause = **T_inner self-pin runaway to ~16000 K** (paper DDC15/CMFGEN = 12630 K), which over-heats the converged radiation field so the NLTE solve itself returns III everywhere.

---

## Why we are here (what the negatives eliminated)

Six independent probes all converged on the same upstream cause — each definitively cleared a downstream candidate:

| Probe | Result | Hypothesis killed |
|---|---|---|
| DDC15 stratification A/B (#28) | NULL | "abundance is the cause" |
| Probe A (bf channel) | pins to Saha | "bf over-ionizes" |
| Probe B + write-back A/B (#29, 161967/68) | B/A = 0.999 | "opacity wiring (NLTE→ion_number_density) is the leak" |
| SCE T_e test (#305-1) | per-band flat | "T_e profile is the lever" |
| per-Z NLTE ablations | NULL | "a specific ion's atomic data" |
| CMF transport verdict | Sobolev ≈ CMF | "the transfer method" |

Arrow points at: **T_inner overshoot → over-hot field → radiation-driven over-ionization inside the NLTE solve.**

---

## Confirmed mechanism (from the code)

| Channel | Radiation-field input | State |
|---|---|---|
| **bf photoionization** (`R_bf`) | real binned `J_nu` (GPU fills it → `R_bf = K^T · J_nu`, nlte_gemm.cu:6) | ✅ already binned-J |
| **bb line excitation** (NLTE matrix) | j_blue **skipped on GPU** (cuda.cu:1300-1302) → falls back to `W·B(T_rad)` | ❌ binned-J gap |
| **bulk opacity / phi_neb ionization** | `W·B(T_rad)` 2-param fit (plasma.c:43) | ❌ 2-param fit |
| **T_inner** | self-pin `T_inner·(L_emitted/L_req)^(-0.5)` (plasma.c:64) | ⚠️ runaway source |

**Decisive fact:** bf already uses the true `J_nu`, yet convergence is still III-dominant → the field is *genuinely* hot because T_inner is pinned to ~16000 K. Memory record: with `LUMINA_T_INNER_FIX=12630`, **L_emitted = 64 %** (a 36 % energy deficit). So the self-pin is doing its job — it cranks T_inner up to close that 36 % deficit, and over-ionization is the side effect.

➡️ **Binned-J alone may not be sufficient. The 36 % deficit must be localized first.**

Code anchors:
- `solve_radiation_field` (Lucy W,T_rad estimator + `LUMINA_W_CAP`): `src/lumina_plasma.c:19-58`
- `update_t_inner` (self-pin): `src/lumina_plasma.c:64-76`
- binned `J_nu` alloc/normalize: `src/lumina_plasma.c:3000`, `nlte_normalize_j_nu` 3024
- GPU `J_nu` fill + line-estimator skip: `src/lumina_cuda.cu:1286-1302`
- GPU energy counters `d_n_escaped` / `d_n_reabsorbed`: `src/lumina_cuda.cu:83-84`, reabsorb at 1945
- bf rate `R_bf = K^T · J_nu`: `src/lumina_nlte_gemm.cu:6, 228-283`

---

## Design — 3 phases, diagnostic gate first

### Phase 0 — energy-budget diagnostic (cheap, decisive — DO FIRST)
At `LUMINA_T_INNER_FIX=12630`, print per-iteration energy decomposition:
- **escaped** (outer boundary) vs **reabsorbed-at-photosphere** (`d_n_reabsorbed`, already counted) vs **truncated** (MAX_INTERACTIONS cap-hit, already counted).
- ~20 lines: emit the three fractions + their energy weights.

**Branch decision:**
- **(A) reabsorbed-dominated** → the iron curtain back-scatters energy into the core → deficit is *radiative trapping*. This is an **inner-boundary-condition** problem, not an estimator problem → go to Phase 2.
- **(B) reabsorbed small, still over-ionizes** → deficit is line-channel over-pumping / accounting → go to Phase 1.

This single diagnostic closes the branch that the prior negatives never closed.

### Phase 1 — binned-J for bb line rates (CMFGEN-faithful)
In the NLTE matrix assembly, evaluate the line excitation rate from the **already-resident GPU `J_nu[bin(ν_line)]`** instead of `W·B(T_rad, ν_line)`. No need to accumulate the full j_blue array (137252×30) — just look up the line-frequency bin during the matrix build → cheap, zero extra memory. Makes bb and bf consistent on the same field. (The task #29 write-back machinery becomes useful again once these rates are correct.)

### Phase 2 — T_inner / inner-BC stabilization (core fix if Phase 0 = A)
If back-scatter trapping drives the deficit: the paper (CMFGEN) uses a **diffusive lower boundary** (energy sent inward returns), whereas LUMINA uses a **reabsorbing hard sphere + self-pin**. Options:
- **(2a)** re-inject reabsorbed energy at the inner BC (luminosity-conserving lower boundary) → T_inner no longer has to over-heat to compensate.
- **(2b)** strengthen self-pin damping + weakly anchor T_inner near the paper value.

---

## Success criteria (must move together — they are coupled)
1. T_inner approaches **12630 K**.
2. Line-forming-shell **fIII < 0.5** (II-curtain returns).
3. **UV leak shrinks** (UV<3400 ratio toward ~1).

All three are one physical state; partial movement of only one is not success.

## Validation
- Each phase = one short slurm A/B (login-node runs are forbidden — sbatch only).
- Baseline = keeper config (DDC15-strat, scatter, radeq-off, 200k, N_ITER=10), e.g. 161922 / 161967.
- Re-run `scripts/probe_b_three_way_ionization.py` on the converged dump to read fIII at shells 0/15/29.

## Recommendation
Start with **Phase 0**. ~20 lines + one short job determines whether the real lever is the binned-J line rates (Phase 1) or the inner boundary condition (Phase 2). Jumping straight to Phase 1 risks producing yet another precise negative if the deficit turns out to be back-scatter trapping — exactly the pattern of the last three days. Close the branch with the diagnostic first.
