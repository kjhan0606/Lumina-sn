# Y2 instrument certification (C2 bf-rate dump) + Y3 pre-registration — **PRELIMINARY-on-partial**

**Status: PRELIMINARY-on-partial. Not a certification of record.**
Input = the iter 0–10 completed block of a walltime-killed run
(`logs/coevolve_consume_parity46_killed_partial/`, NOTE.md: runner1 pid 186309,
killed during iter 11). The run's schedule was `LUMINA_PURE_CMFGEN_ITER=12`, so
these are iterations 0–10 of 12 — **one-plus iteration short of the intended end,
and demonstrably not converged** (§1.5). Formal certification is the clean rerun.

Author: Opus. Date: 2026-07-30.
All numeric outputs in this directory: `y2_*.csv`. Scripts: `step1..step8_*.py`,
`y2_common.py`. Every claim below is reproducible by rerunning them in order.

---

## 0. What the two paths actually are (established from code, not assumed)

| | field it integrates | where |
|---|---|---|
| **C2 path** (armed by Y3 = `LUMINA_C2_MATRIX_BF=1`) | `sigma*bfr` on MC-sampled bins, `pref*J_nu` on MC-unsampled bins | `src/lumina_plasma.c:14621-14636` |
| **GEMM path** (this run's actual matrix route) | `R_bf_table[shell*L_phot+idx]` = `K^T · J_nu` | `src/lumina_plasma.c:14601-14604`; `src/lumina_nlte_gemm.cu:217-229, 416-443` |

`pref = 4*pi*sigma/(h*nu)*dnu` (`lumina_plasma.c:14620`) and the GEMM kernel
`K[bb,idx] = sigma*4*pi/(h*nu_bb)*dnu_bb` (`lumina_nlte_gemm.cu:226-227`) are the
**same expression**. The two paths therefore differ *only* in the field.

The field the GEMM stages is `nlte->J_nu` (`lumina_nlte_gemm.cu:416-419`), which
at that moment holds the **C1 dilute-BB refit** written at
`lumina_plasma.c:1049-1067`:

```
c = (int)((long)f * NC / nb)                       # NC=24, nb=1000
if  degen_c[c]        : Jrow[f] = jr[f]*norm/dnu_f     # == the dump's J_raw
elif W_c>0 and TR_c>0 : Jrow[f] = W_c * B_nu(TR_c, nu_ctr)
else                  : Jrow[f] = 0
```

Both `bfr` (normalised at `lumina_plasma.c:1072-1076`) and this `J_C1` are
produced by the **same call** to `nlte_build_perbin_dilute_field`, so pairing
them at a fixed dump `iter` is exact — no lag assumption is involved.

**Scoping fact (load-bearing, and it narrows Y3 considerably).**
`nlte_get_pairs` with `LUMINA_NLTE_STAGE4` unset returns the 16 **base** pairs
(`lumina_plasma.c:7268-7273`). Only the **lower** member of each pair enters the
R_bf loop (`lumina_plasma.c:14564`). Against the slot table printed at
`stdout_partial.log:278-308` the lower ions are:

> Si II, Ca II, Fe II, S II, Co II, Ni II, C II, Mg II, Ti II, Cr II, Al II,
> Sc II, V II (0 levels), Mn II, O I, O II

Their level counts sum to **10340**, matching the banner
`[NLTE-GEMM] init: 16 pairs, 10340 phot levels` (`stdout_partial.log:7285`)
exactly. Consequences:

1. **Si III, S III, Fe III, Fe IV, Co III receive no matrix R_bf at all in this
   configuration.** Their only matrix photoion channel is the top-ion drain
   (`lumina_plasma.c:14788-14934`), gated by `LUMINA_TOPSTAGE_IV`, which is
   **not in this run's env footer** (0 hits in `stdout_partial.log`). Numbers for
   them below are reference-only.
2. The **R1 ionization closure already runs on C2** (`parity_gamma_phot`,
   `lumina_plasma.c:1731-1732`), unconditionally under parity, for *all* ions.
   So the C2/GEMM ratio measured here is **an internal inconsistency between two
   solvers inside the same run**: the ionization closure and the NLTE matrix
   integrate the same edge against different fields. Y3 removes that split.

---

## 1. Instrument hygiene — **PASS on every offline-checkable axis**

### 1.1 Completeness / finiteness / sign
550,000 data rows = 11 iters × 50 shells × 1000 fine bins; every block complete.

| array | non-finite | negative | zeros | range |
|---|---|---|---|---|
| `J_raw` | **0** | **0** | 198,612 | [0, 1.1114e-02] |
| `bfr` | **0** | **0** | 198,612 | [0, 1.1169e+23] |
| `j_nu_count` | — | 0 entries of `-1` (array present everywhere) | — | [0, 15,567,677] |

### 1.2 Structural identity of the C2 preference test
`{j_nu_count == 0} ≡ {bfr <= 0} ≡ {J_raw <= 0}` — **0 mismatches in 550,000 rows**
(`n_cnt0_bfr_pos = 0`, `n_cnt_pos_bfr0 = 0`, `y2_unsampled_fraction.csv`).
So `bfr > 0` at `lumina_plasma.c:14634` fires on exactly the MC-sampled bins;
the documented fallback semantics are exact.

### 1.3 MC-unsampled fraction (`j_nu_count == 0`), by shell × iter

| shell | it0 | it4 | it8 | it10 |
|---|---|---|---|---|
| 0 | 0.266 | 0.247 | 0.188 | **0.199** |
| 8 | 0.332 | 0.264 | 0.268 | **0.273** |
| 20 | 0.416 | 0.376 | 0.360 | **0.354** |
| 30 | 0.425 | 0.390 | 0.374 | **0.367** |
| 45 | 0.462 | 0.424 | 0.372 | **0.357** |
| 49 | 0.471 | 0.439 | 0.388 | **0.367** |

All-shell mean falls monotonically 0.400 (it0) → 0.332 (it10).
Full table: `y2_unsampled_fraction.csv`.

### 1.4 Two independent normalisation certifications

**(a) Cross-dump roundtrip.** `sum_f J_raw[f]*dnu_f` over each coarse bin, versus
the C1 dump's `J_bin` column: 9,097 non-empty bins compared, **0 disagreements on
which bins are zero**, ratio median `1.000000000`, **max |1−r| = 5.6e-07** — i.e.
the `%.6e` print precision of the dumps. The two instruments are mutually
consistent to their own resolution.

**(b) `bfr` reduces to the raw-J integrand.** ARTIS defines
`Gamma_bf density = sum(e*dist/nu)/(V*t*h)` and `J = sum(e*dist)/(4pi*V*t*dnu)`,
so `sigma*bfr / (pref*J_raw)` must equal `nu_mid/nu_eff` with `nu_eff` inside the
bin — bounded by the log-bin geometry to `[exp(-dlog/2), exp(+dlog/2)] =
[0.997354, 1.002653]`. Measured over **351,388 paired bins**:

```
min = 0.997355   median = 0.999998   max = 1.002652   out-of-window = 0 (0.0000%)
```

The measured extremes land on the theoretical bounds to 6 decimal places.
**The C2 estimator is the frequency-weighted twin of the raw MC field, not an
independent physical quantity** — this is the single most important
interpretive fact for Y3, and it is now proven rather than assumed.

**(c) bonus cross-check on the C1 dump.** `LUMINA_C1_SUPERBIN_TEPIN=1` sets
`T_R := T_e(shell)` for pinned coarse bins (`lumina_plasma.c:952-956`), so
`T_R[iter,shell,cbin=14]` *is* T_e. Recovered values at iter 10 —
`T_e[0]=21210, T_e[25]=8428, T_e[49]=13051 K` — match
`stdout_partial.log:33484` (`[CMFGEN] iter 10: T_e[0]=21210K T_e[25]=8428K
T_e[49]=13051K`) **exactly**.

### 1.5 Iteration stability over the final three blocks (8, 9, 10) — **NOT CONVERGED**

`J_raw`, median per-bin relative change iter 9→10:

| shell | 0 | 8 | 20 | 30 | 45 | 49 |
|---|---|---|---|---|---|---|
| median \|ΔJ\|/J | 0.064 | 0.081 | **0.291** | **0.221** | **0.215** | **0.211** |
| integrated ∫J dν | −0.074 | +0.082 | +0.184 | +0.099 | −0.027 | −0.043 |

`bfr` tracks `J_raw` to within 0.5% on every one of these (`y2_bfr_stability.csv`),
which is itself another confirmation of §1.4(b).

**Verdict — §1.** The instrument is clean: no non-finite, no negative, no missing
array, exact fallback semantics, and two independent normalisation identities
closing at 5.6e-07 and 0.27%. What is **not** certified is the *value* of the
field: shells ≥ 20 are still moving 20–29% per iteration when the run was killed.

---

## 2. Path comparison — the pre-registration numbers

Weighting for the per-ion figure is exactly `parity_gamma_phot`'s
(`lumina_plasma.c:1746-1786`): `f_lev = g*exp(-E/kT_e)/Z_part` at the T_e
recovered in §1.4(c), consistent with the parity **B3** partition-function
convention (functions at T_e, undiluted W=1; banner `stdout_partial.log:168`).

### 2.1 HEADLINE — Boltzmann-weighted `Gamma_phot(C2) / Gamma_phot(GEMM)`, dump iter 10

| ion | s0 | s8 | s20 | s30 | s45 | s49 |
|---|---|---|---|---|---|---|
| **Si II** | 0.967 | 0.970 | **1.79** | **2.80** | 1.19 | 1.03 |
| **Fe II** | 1.022 | 0.985 | **1.47** | **3.10** | 1.31 | 1.08 |
| **Mg II** | 1.018 | 1.004 | 1.39 | **1.60** | 1.01 | 0.98 |
| **Co II** | 1.001 | 0.990 | 1.06 | **1.46** | 1.19 | 1.05 |
| Ca II | 1.019 | 1.011 | 1.15 | 1.12 | 1.09 | 1.01 |
| Cr II | 0.995 | 1.005 | 1.07 | 1.09 | 1.04 | 1.03 |
| Mn II | 1.007 | 1.005 | 1.04 | 1.10 | 1.03 | 1.02 |
| **Ni II** | 1.010 | 0.978 | **0.75** | 1.08 | 1.05 | 0.97 |
| Ti II | 1.011 | 0.987 | 0.97 | 1.00 | 1.10 | 1.01 |
| Sc II | 1.003 | 1.002 | 1.04 | 0.95 | 1.01 | 1.04 |
| Al II | 0.991 | 1.007 | 0.87 | 1.13 | 0.97 | 0.94 |
| O I | 1.005 | 0.995 | 0.92 | 1.00 | 1.04 | 1.08 |
| S II † | 1.024 | 0.902 | 0.98 | 0.99 | 0.99 | 1.37 |
| C II † | 0.928 | 0.761 | 1.25 | 1.10 | 1.30 | 1.33 |
| O II † | 0.999 | 1.690 | 0.91 | 1.00 | 1.04 | 1.00 |

† S II / C II / O II carry Γ that is **≥ 89× / ≥ 1876× / ≥ 6.6e6× below**
max(Si II, Fe II) at every shell beyond s0 (`y2_gamma_per_ion.csv`); their
ratios are numerically real but dynamically subdominant.

**Reference-only** (no matrix R_bf in this config — see §0):
Fe III 1.31/2.36/1.42 at s20/s30/s45; Si III 0.94/1.98/1.05;
S III 0.99/1.80/1.47; Co III 1.01/1.85/1.31.

### 2.2 GROUND-level ratio (the channel B4 actually routes; `lumina_plasma.c:14499`)

| ion | s0 | s8 | s20 | s30 | s45 | s49 |
|---|---|---|---|---|---|---|
| Si II | 1.06 | 0.93 | **2.08** | **6.36** | 1.59 | 1.62 |
| Fe II | 1.03 | 1.00 | 1.19 | **6.37** | 1.50 | 1.40 |
| Mn II | — | — | — | **3.46** | — | — |
| Mg II | 1.05 | 0.98 | 1.49 | 2.46 | 1.53 | 1.50 |
| Cr II | 1.04 | 0.94 | 1.34 | 1.65 | 1.28 | 1.48 |
| Co II | 1.00 | 0.97 | 1.00 | 1.56 | 1.68 | 1.11 |

The realized matrix effect must lie **between** §2.2 (ground) and §2.1
(Boltzmann) — the matrix weights per level by the actual NLTE populations × `f_lev`
(`lumina_plasma.c:14686`), which were not dumped.

### 2.3 Mechanism — verified, not asserted

The gap lives in the **T_e-pinned coarse bins**. With `LUMINA_C1_SUPERBIN_TEPIN=1`,
coarse bins 14–23 (λ_hi ≤ 1085 Å) take `T_R := T_e` instead of a colour fit.
Coarse bin 14 spans 728.8–905.6 Å and is `pin` in **all 50 shells** at every iter.

Measured shape distortion inside bin 14 (both fields renormalised to unit bin
energy, so only the *shape* is compared — `step6_mechanism.py`):

| shell | T_e | Wien contrast `exp(-h·Δν/kT_e)` | C1/raw in RED third | C1/raw in BLUE third |
|---|---|---|---|---|
| 0 | 21210 | 0.170 | 1.10 | 0.84 |
| 8 | 11994 | 0.044 | 1.02 | 1.09 |
| 20 | 8659 | 0.013 | 1.19 | **0.24** |
| 30 | 8353 | 0.011 | 1.48 | **0.19** |
| 45 | 11892 | 0.042 | 1.80 | **0.36** |
| 49 | 13051 | 0.056 | 1.33 | **0.54** |

And the edges sit exactly where that hurts (0 = red edge, 1 = blue edge of their
own coarse bin):

```
Si II  758.5 A -> coarse bin 14, position 0.82
Fe II  765.9 A -> coarse bin 14, position 0.78
Cr II  752.1 A -> coarse bin 14, position 0.86
Mg II  824.6 A -> coarse bin 14, position 0.43
Ni II  682.4 A -> coarse bin 15, position 0.29
Co II  725.7 A -> coarse bin 15, position 0.01
```

**A Wien exponential pinned at ~8.4 kK across a 1.24:1 frequency bin buries the
blue end by ~e^-4.5, and the Si II / Fe II / Cr II edges sit at 78–86 % of the
way to that blue end.** That is the whole of the s20–s30 effect, and it explains
both its sign (C2 > GEMM: the refit starves the edge) and its ordering
(Si II ≈ Fe II ≈ Cr II > Mg II > Co II).

### 2.4 Two structural checks on the comparison itself

* **The GEMM's missing threshold cut is a no-op.** The K build applies
  `nu_bin < nu_thresh -> continue` only on the Kramers branch
  (`lumina_nlte_gemm.cu:223`), not the CMFGEN-σ branch, while the CPU consumer
  always applies it (`lumina_plasma.c:14608`). Measured: **zero** CMFGEN σ rows
  have `sigma > 0` below their own threshold, for all of Si II / Fe II / Co II /
  Ni II / S II / Ca II, so `leak_frac = 0.000000` everywhere
  (`y2_decomposition.csv`). The two paths integrate identical bin sets.
* **Closed-form re-derivation.** Steps 1.4(b) + 2.4 imply
  `R_C2 ≈ Σ_sampled pref·J_raw + Σ_unsampled pref·J_C1`, computed **without ever
  touching `bfr`**. Against the direct consumer form: max relative disagreement
  **1.16e-03** over all ion×shell (median 2.2e-05) — exactly the 0.265 % in-bin
  ν-weighting bound. Two independent computations agree.
  **Y3 is exactly: "use the raw MC field wherever the MC sampled it."**

### 2.5 Robustness — the magnitude is drifting, the sign is not

Ratio at s30 by iteration (`y2_ratio_by_iter.csv`):

| ion | it4 | it6 | it8 | it9 | it10 |
|---|---|---|---|---|---|
| Si II | 1.47 | 1.84 | 2.00 | 2.20 | **2.80** |
| Fe II | 1.88 | 2.16 | 2.48 | 2.42 | **3.10** |
| Mg II | 1.27 | 1.35 | 1.41 | 1.54 | **1.60** |
| Co II | 1.99 | 1.51 | 1.33 | 1.39 | **1.46** |

Sign is stable from iter 0 onward for Si II / Fe II / Mg II (never below 1.0 at
s30 in any of the 11 blocks). Magnitude is **monotonically rising** for Si II and
Fe II and has not plateaued. Spread over the last three blocks: 1.40× (Si II s30),
1.28× (Fe II s30); ≤ 1.05× for every ion at s0/s8.

---

## 3. N1 pathology — how much the C1 refit distorts, and what Y3 can and cannot fix

### 3.1 C1 coarse-bin mode census (13,200 iter×shell×bin cells)

| mode | count | share |
|---|---|---|
| fit | 7,672 | 58.1 % |
| **empty** | 4,103 | **31.1 %** |
| **pin** | 1,397 | **10.6 %** |
| degen | 28 | **0.21 %** |

By coarse bin at iter 10 (`y2_c1_mode_census.csv`):
bins 14–15 (583–906 Å) = `pin` in all 50 shells; bins 16–19 (241–583 Å) = mixed
pin/empty; bins **20–23 (100–241 Å) = `empty` in all 50 shells** (⇒ `J_C1 = 0`,
so the GEMM path carries *no* photoionization at all shortward of ~241 Å).

### 3.2 The 250 kK rail fires — and mostly **survives** into the field

`T_R ≥ 0.95 × 250 kK` in **1,669 / 13,200 cells (12.6 %)**. Of those:
`degen` (raw published instead) **28**, `pin` 19, and **`fit` = 1,622 (97 %) —
the railed fit reaches the field.** The rail's location is the surprise: median
`T_R = 250000 K` in coarse bins **0, 1, 3, 4, 5**, i.e. **5314–19986 Å (NIR/optical)**,
not the EUV. `LUMINA_C1_DEGEN_FALLBACK`'s `raw_frac < 1e-3` criterion
(`lumina_plasma.c:1010`) cannot fire there because those bins carry most of the
energy — **so the degeneracy fallback is effectively inert in this configuration
(28 firings in 13,200 cells).**

### 3.3 Band decomposition of the distortion (iter 10, 50 shells pooled)

`D = J_C1/J_raw`; weights = the photoion kernel measure `4π/(hν)·dν`.

| band | frac MC-unsampled | kernel share with **D > 2** | kernel share with D < 0.5 | ∫kernel C1/raw |
|---|---|---|---|---|
| NIR/opt 20000–4000 Å | 0.009 | 0.008 | 0.017 | 0.998 |
| near-UV 4000–2000 Å | 0.000 | 0.005 | 0.002 | 1.000 |
| mid-UV 2000–1200 Å | 0.000 | 0.030 | 0.027 | 1.007 |
| **FUV 1200–912 Å** | 0.002 | **0.176** | 0.044 | 0.943 |
| **EUV 912–500 Å** | **0.324** | **0.324** | 0.022 | 1.010 |
| **deep-EUV 500–100 Å** | **0.961** | **0.332** | 0.013 | **1.351** |

Read: **above the photoion thresholds, a third of the C1 photoion-kernel weight
sits in bins where the refit exceeds the raw MC by more than 2×**, while the
band *integral* stays near 1 (the refit conserves energy per coarse bin and
merely moves it around inside — §2.3). In the deep EUV 96 % of bins are
MC-unsampled and the C1 kernel runs 1.35× hot overall.

### 3.4 The part Y3 **cannot** repair

Every "fill" bin (MC saw nothing, C1 publishes flux) has `bfr == 0` — verified
for all **1,840** of them at iter 10 — so **both paths take `pref·J_C1` there and
Y3 leaves them untouched.** Share of each *ground edge's* photoion kernel that is
pure C1 fill (`y2_fill_share_at_edge.csv`):

| ion | s20 | s30 | s45 | s49 |
|---|---|---|---|---|
| Ni II | 0.20 | **0.55** | 0.52 | **0.76** |
| Co II | 0.04 | **0.47** | 0.38 | **0.50** |
| Fe II | 0.05 | 0.25 | 0.25 | **0.46** |
| Cr II | 0.02 | 0.15 | 0.22 | 0.37 |
| Si II | 0.01 | 0.10 | 0.17 | 0.30 |
| Mg II | 0.02 | 0.10 | 0.16 | 0.25 |
| Ca II | 0.00 | 0.00 | 0.02 | 0.02 |

**Between 25 % and 76 % of the outer-shell IGE ground photoion integral is
C1-fabricated flux in bins the transport never sampled, and Y3 does not touch
any of it.** That is the honest ceiling on the repair.

---

## 4. Y3 pre-registration draft

Gate under test: `LUMINA_C2_MATRIX_BF=1` (`lumina_plasma.c:14541-14550`).
Arms A (off, = this run) / B (on), everything else identical.

### 4.1 Registrable — direction and size

| # | prediction | direction | size (iter-10 basis) | basis |
|---|---|---|---|---|
| P1 | Si II Γ_phot at s20–s35 rises | **UP** | ×1.6–3.5 pop-wtd; ×2–6.5 ground | §2.1, §2.2, §2.5 |
| P2 | Fe II Γ_phot at s20–s35 rises | **UP** | ×1.4–3.5 pop-wtd; ×1.2–6.5 ground | §2.1, §2.2, §2.5 |
| P3 | Mg II, Co II at s30 rise | **UP** | ×1.4–1.7 | §2.1 |
| P4 | Ni II at s20 falls | **DOWN** | ×0.75–0.98 | §2.1, §2.5 |
| P5 | **inner shells s0, s8 unchanged** | **NULL** | \|ratio−1\| ≤ 0.035 for every ion with Γ > 1 s⁻¹ | §2.1 |
| P6 | Ca II, Cr II, Mn II, Ti II, Sc II, O I: ≤ 15 % anywhere | **NULL-ish** | ×0.92–1.15 | §2.1 |
| P7 | Si III / S III / Fe III / Fe IV / Co III matrix R_bf: **exactly zero change** | **NULL (exact)** | 0 | §0 scoping |
| P8 | R1 `parity_gamma_phot` output identical between arms | **NULL (exact)** | 0 | `lumina_plasma.c:1731` already C2 |

P5, P7, P8 are the wiring falsifiers; P1–P4 are the physics claims.

### 4.2 Characterization only — record, do not score

| # | item | number |
|---|---|---|
| C1 | ratio magnitude is still drifting at kill time (Si II s30: 2.00→2.20→2.80 over it8/9/10) | spread 1.40× |
| C2 | C1-fill share of the outer ground edges — the part Y3 leaves on C1 | 0.25–0.76 at s45/s49 |
| C3 | 250 kK rail surviving into the field, in **NIR** coarse bins 0/1/3/4/5 | 1,622 / 13,200 cells |
| C4 | `LUMINA_C1_DEGEN_FALLBACK` effectively inert under TEPIN | 28 / 13,200 firings |
| C5 | coarse bins 20–23 (100–241 Å) `empty` in all 50 shells ⇒ GEMM carries zero photoion there | 100 % |
| C6 | MC-unsampled fine-bin fraction still falling at kill | 0.400 → 0.332 |
| C7 | result is **conditional on `LUMINA_C1_SUPERBIN_TEPIN=1`** — the mechanism (§2.3) is the T_e pin. Changing that gate changes every number in §2 | — |

### 4.3 Hard-stop candidates

| # | trip condition | why it is a stop |
|---|---|---|
| HS1 | any s ≤ 8 per-ion Γ changes by > 5 % | violates P5; gate is reaching past the matrix branch |
| HS2 | any Si III / S III / Fe III / Fe IV / Co III matrix R_bf changes | violates P7; scope leak (would mean the top-ion drain or stage-4 silently armed) |
| HS3 | R1 ionization closure output differs between arms | violates P8; double-routing of the same estimator |
| HS4 | any new non-finite or negative R_bf / b_k in the ledger | instrument or solver corruption |
| HS5 | b_k or n_e moves at s0–s8 beyond that arm's own iteration-to-iteration noise | same as HS1, seen from the solution side |

### 4.4 What must be re-run before any of this is certified

1. Clean rerun to completion (`PURE_CMFGEN_ITER=12`) with the same three dumps —
   §1.5 and C1 show the iter-10 magnitudes are not the converged ones.
2. A **GEMM-side dump of `R_bf_table`**. Everything in §2 reconstructs the GEMM
   integral in float64 from the C1 `(W, T_R, mode)` columns; production runs it
   in FP32 with `CUBLAS_COMPUTE_32F_FAST_TF32` (`lumina_nlte_gemm.cu:442`).
   TF32's ~10-bit mantissa is ~1e-3 relative — negligible against ratios of
   1.5–6× — but this is **inference, not measurement**, and it is the one
   load-bearing quantity in this report that was never observed directly.
3. A **level-population dump**, so §2.1/§2.2 can be replaced by the actual
   population-weighted matrix effect instead of bracketed by them.

---

## 5. PRELIMINARY limits — explicit

* Source is a **walltime-killed partial**: iters 0–10 of 12; iter 11 discarded.
  Byte-identity certification is impossible against it (NOTE.md).
* **Not converged.** Median per-bin `J_raw` change iter 9→10 is 21–29 % in shells
  ≥ 20; the headline Si II / Fe II s30 ratios are still climbing monotonically.
* Both dumps carry `%.6e` precision. The roundtrip (5.6e-07) and the `bfr`
  identity (≤ 0.27 %) close far inside that, so **print precision is not the
  limiting factor — iteration non-convergence is.**
* The GEMM arm is reconstructed, not observed (§4.4 item 2).
* Level populations were not dumped; per-ion weighting uses the Boltzmann/T_e
  form the code's own `parity_gamma_phot` uses (§2), which brackets rather than
  equals the matrix's weighting.
* All of §2 is **conditional on `LUMINA_C1_SUPERBIN_TEPIN=1`** (C7).
