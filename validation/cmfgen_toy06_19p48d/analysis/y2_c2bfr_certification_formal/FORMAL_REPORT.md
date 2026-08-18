# Y2 instrument certification (C2 bf-rate dump) + Y3 pre-registration — **FORMAL**

**Status: FORMAL certification of record.**
Input = `logs/coevolve_consume_parity46/`, the clean complete 12-iteration run
(`LUMINA_PURE_CMFGEN_ITER=12`, iters **0–11**). `lumina_c2_bfr_dump.csv` =
600,000 data rows = 12 × 50 × 1000, every block complete;
`lumina_c1_bins.csv` = 14,400 = 12 × 50 × 24. Judgment outputs byte-identical to
parity44, so the field is the adopted **K6 baseline field**.

Author: Opus. Date: 2026-07-30.
Scripts: `y2_common.py`, `step1..step8_*.py` (adapted from the prelim — see
Appendix A), plus `step9..step12` (new). All numeric outputs: `y2_*.csv`,
console transcripts `step*.out`. Every number below is reproducible by running
`step1 … step12` in order from this directory.

The PRELIMINARY report (`../y2_c2bfr_certification/PRELIM_REPORT.md`, iters 0–10
of a walltime-killed run) is **superseded** by this document but retained as
provenance. Nothing in that directory was modified.

---

## 0. Determinism: the clean run reproduces the partial exactly

| test | result | source |
|---|---|---|
| `lumina_c2_bfr_dump.csv` line-by-line over the 550,001 common lines | **1 line differs**, in the 7th printed digit of `J_raw` only: `6,33,442,…,5.090371e-07,…` vs `…5.090372e-07,…` (2e-7 relative; `%.6e` last-digit tie) | `step11.out` |
| Boltzmann ratio + `Gamma_C2`, all 12 ions × 6 shells × iters 0–10 | max abs Δratio = **0.000e+00**, max rel ΔΓ = **0.000e+00** | `step11.out` |
| `∫J dν` at iter 10, all 50 shells | max rel diff = **0.000e+00** | `step11.out` |
| `[CMFGEN] iter 10: T_e[0]=21210K T_e[25]=8428K T_e[49]=13051K` | identical in both logs | `stdout.log:33485` |

The single non-identical median in `step11.out`
(`max |med_rel_d_9_10 diff| = 1.559e-02`, shell 13) is **not** a determinism
defect: `step1`'s stability mask is "positive in all three iterations of the
window", which is (8,9,10) in the prelim and (9,10,11) here, so the two medians
are taken over different bin sets (`n_common_pos` 664 vs 679 at shell 13).

---

## 1. Instrument hygiene on all 600,000 rows — **PASS, no regression**

### 1.1 Finiteness / sign / zeros (`y2_finiteness.csv`, `step1.out`)

| array | n | non-finite | negative | zeros | range |
|---|---|---|---|---|---|
| `J_raw` | 600,000 | **0** | **0** | 215,159 | [0, 1.178824e-02] |
| `bfr` | 600,000 | **0** | **0** | 215,159 | [0, 1.184445e+23] |
| `j_nu_count` | 600,000 | — | 0 entries of `-1` (array present in every block) | — | [0, 15,567,677] |

Grid self-check: `max |nu_mid_file/nu_mid_hdr − 1| = 4.987e-07` against
`NU_MIN=1.5e14`, `DLOG=5.298317367e-03` (`src/lumina.h:491-493`).

### 1.2 Structural identity `{j_nu_count == 0} ≡ {bfr <= 0} ≡ {J_raw <= 0}`

```
n_cnt0_bfr_pos  = 0        (cnt==0 but bfr>0)
n_cnt_pos_bfr0  = 0        (cnt >0 but bfr<=0)
```
**0 mismatches in 600,000 rows** (`y2_unsampled_fraction.csv` summed over all
600 iter×shell blocks). The zero counts of `J_raw` and `bfr` are equal
(215,159 = 215,159), and 600,000 − 215,159 = 384,841 = the paired-bin count
step2 finds independently. So `bfr > 0` at `src/lumina_plasma.c:14634` fires on
exactly the MC-sampled bins; the documented fallback semantics are exact.

### 1.3 Cross-dump roundtrip vs the C1 `J_bin` column

`sum_f J_raw[f]*dnu_f` per coarse bin vs `lumina_c1_bins.csv:J_bin`:

```
compared bins        : 9,947 / 14,400
disagree-on-zero     : 0
ratio median         : 1.000000001
ratio min / max      : 0.999999472 / 1.000000562
max |1 − r|          : 5.619e-07          (prelim: 5.6e-07)
```
This is the `%.6e` print precision of the dumps. The two instruments are
mutually consistent to their own resolution.

### 1.4 The `bfr/(pref·J_raw) = nu_mid/nu_eff` identity

Log-bin geometry bounds the ratio to `[exp(−dlog/2), exp(+dlog/2)] =
[0.997354, 1.002653]`. Measured over **384,841 paired bins** (`step2.out`,
`y2_bfr_identity.csv`):

```
min = 0.997355   p1 = 0.998454   median = 0.999997   p99 = 1.001624   max = 1.002652
out-of-window = 0   (0.0000 %)
shells with any out-of-window bin: 0 / 50
```
The measured extremes land on the theoretical bounds to 6 decimals. **The C2
estimator is the frequency-weighted twin of the raw MC field, not an independent
physical quantity** — re-proven on the complete run.

### 1.5 T_e crosscheck vs `stdout.log`

`LUMINA_C1_SUPERBIN_TEPIN=1` sets `T_R := T_e(shell)` for pinned coarse bins
(`src/lumina_plasma.c:952-956`), so `TR[iter, shell, cbin=14]` *is* T_e.
Compared against every `[CMFGEN] iter N: T_e[0]=…K T_e[25]=…K T_e[49]=…K` line
(`step9.out`, `y2_te_crosscheck.csv`):

```
rows compared                       : 36   (iters 0..11 x shells 0,25,49)
rows where C1 bin 14 is actually pin : 35
max |diff| over pinned rows          : 0.46 K   (stdout prints integer K -> +-0.5 K is exact)
mismatches (>= 1 K) among pinned     : 0
non-pinned rows                      : 1  -> (iter 1, shell 49), C1 mode = 'empty'
```
The one non-recoverable row is a genuine early-iteration property, not a defect:
in 16 of 600 iter×shell cells the MC put **no** energy into 728.8–905.6 Å, so
the C1 bin is `empty` and the pin never happens
(cells: iter 0 s37–40; iter 1 s40–49; iter 2 s40; iter 4 s41).
**At the certifying iteration 11 all 50 shells are pinned** (min `T_R` =
8341.57 K), so the T_e used for every headline number below is fully recovered.
Iter 11: `T_e[0]=21203 K, T_e[25]=8501 K, T_e[49]=13066 K` recovered as
21202.68 / 8500.72 / 13066.37 vs `stdout.log:36005`.

### 1.6 MC-unsampled fraction, by shell × iter (`y2_unsampled_fraction.csv`)

| shell | it0 | it4 | it8 | it10 | **it11** |
|---|---|---|---|---|---|
| 0 | 0.266 | 0.247 | 0.188 | 0.199 | **0.193** |
| 8 | 0.332 | 0.264 | 0.268 | 0.273 | **0.277** |
| 20 | 0.416 | 0.376 | 0.360 | 0.354 | **0.350** |
| 30 | 0.425 | 0.390 | 0.374 | 0.367 | **0.370** |
| 45 | 0.462 | 0.424 | 0.372 | 0.357 | **0.362** |
| 49 | 0.471 | 0.439 | 0.388 | 0.367 | **0.373** |

All-shell mean: 0.3998 (it0) → 0.3316 (it10) → **0.3309 (it11)**. The monotone
fall has flattened (−0.0007 on the last step vs −0.0056 on the previous).

### 1.7 Structural checks on the comparison itself (unchanged)

* **GEMM's missing threshold cut is a no-op.** `leak_frac = 0.000000` at every
  ion × shell for Si II / Fe II / Co II / Ni II / S II / Ca II; 0 CMFGEN-σ rows
  have `sigma > 0` below their own threshold (`y2_decomposition.csv`).
* **Closed-form re-derivation** (`R_C2 ≈ Σ_sampled pref·J_raw + Σ_unsampled
  pref·J_C1`, never touching `bfr`) vs the direct consumer form:
  max relative disagreement **1.142e-03**, median **3.126e-05** — inside the
  0.265 % in-bin ν-weighting bound (`step8.out`; prelim 1.16e-03 / 2.2e-05).
* **FILL bins survive Y3 unchanged.** 1,834 of 50,000 at iter 11; every one has
  `bfr == 0`, so both paths take `pref·J_C1` (`step6.out`).

**Verdict §1: no hygiene regression on any axis.** Every prelim identity closes
at the same or better precision on 9 % more data.

---

## 2. Convergence — the key new information. **NOT CONVERGED.**

### 2.1 Median per-bin `|ΔJ_raw|/J`, it9→10 **and** it10→11 (`y2_Jraw_stability.csv`)

| shell | med it9→10 | med it10→11 | smaller? | wmean it9→10 | wmean it10→11 | ∫J it9→10 | ∫J it10→11 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0642 | **0.0462** | yes | 0.0776 | 0.0696 | −0.0740 | −0.0184 |
| 8 | 0.0812 | **0.1029** | **no** | 0.0910 | 0.0847 | +0.0820 | −0.0755 |
| 20 | 0.2916 | **0.3113** | **no** | 0.3783 | 0.3911 | +0.1842 | +0.1289 |
| 30 | 0.2214 | **0.2055** | yes | 0.2611 | 0.2354 | +0.0986 | +0.1105 |
| 45 | 0.2152 | **0.1926** | yes | 0.1997 | 0.1777 | −0.0271 | +0.0133 |
| 49 | 0.2121 | **0.1913** | yes | 0.1935 | 0.1683 | −0.0433 | −0.0018 |

All-shell median of the per-shell medians: **0.2267 (it9→10) → 0.1938 (it10→11)**.

**Plainly: the it10→11 step is smaller than it9→10 in 4 of the 6 report shells
and in the all-shell median, but it is not smaller at s8 (0.081→0.103) or s20
(0.292→0.311), and shells ≥ 20 are still moving 19–31 % per iteration at the
scheduled end of the run.** The field is *slowly* contracting, not converged.
`bfr` tracks `J_raw` to within 0.0065 in the median on every one of the 50
shells (`step1.out`), a further confirmation of §1.4.

### 2.2 The prelim's open question — Si II s30 had no plateau. Answer: **still none.**

Boltzmann-weighted `Γ(C2)/Γ(GEMM)` at s30 by iteration (`y2_ratio_by_iter.csv`):

| ion | it4 | it6 | it8 | it9 | it10 | **it11** | it10→11 |
|---|---|---|---|---|---|---|---|
| **Si II** | 1.47 | 1.84 | 2.00 | 2.20 | 2.80 | **3.38** | ×1.207 |
| **Fe II** | 1.88 | 2.16 | 2.48 | 2.42 | 3.10 | **3.50** | ×1.129 |
| **Mg II** | 1.27 | 1.35 | 1.41 | 1.54 | 1.60 | **1.68** | ×1.050 |
| **Co II** | 1.99 | 1.51 | 1.33 | 1.39 | 1.46 | **1.57** | ×1.075 |

Si II rose by +0.60 on it9→10 and +0.58 on it10→11 — the *absolute* step has not
shrunk at all. Fe II likewise (+0.68, +0.40). **No plateau; the monotone climb
that the prelim flagged continues through the last scheduled iteration.**
The same at s20 is worse: Si II 1.79 → **3.59** (×2.00 in one iteration),
Ca II 1.15 → 1.25, Mg II 1.39 → 1.63.

Last-3-iteration spread (`step7.out`): Si II s20 **2.004×**, Si II s30 1.538×,
Fe II s30 1.448×, Fe II s20 1.290×; ≤ 1.03× for every ion at s0/s8.

---

## 3. Headline tables at iter 11

### 3.1 Boltzmann-weighted `Γ_phot(C2) / Γ_phot(GEMM)` (`y2_gamma_per_ion.csv`)

Weighting is exactly `parity_gamma_phot`'s (`src/lumina_plasma.c:1746-1786`):
`f_lev = g·exp(−E/kT_e)/Z_part` at the T_e of §1.5. Confirmed independently by
`step8` (`y2_headline_summary.csv`) to 1.1e-03.
`X` = elemental mass fraction in that shell (`data/…/abundances.csv`) — an ion
with `X = 0` has no atoms there, so its ratio is a rate-coefficient diagnostic
that cannot move `b_k`, `n_e` or the spectrum.

| ion | s0 | s8 | s20 | s30 | s45 | s49 | X(s0) | X(s20) |
|---|---|---|---|---|---|---|---|---|
| **Si II** | 0.971 | 0.964 | **3.591** | **3.379** | 1.256 | 1.048 | 0 | **0.550** |
| **Fe II** | 1.021 | 0.976 | **1.770** | **3.504** | 1.350 | 1.090 | 0.098 | **0** |
| **Mg II** | 1.020 | 1.002 | **1.627** | **1.680** | 1.025 | 0.988 | 0 | 0 |
| **Co II** | 1.006 | 0.985 | 1.140 | **1.573** | 1.196 | 1.073 | 0.794 | **0** |
| Ca II | 1.020 | 1.011 | **1.250** | 1.109 | 1.102 | 0.990 | 0 | **0.100** |
| Cr II | 0.996 | 1.004 | 1.147 | 1.106 | 1.039 | 1.019 | 0 | 0 |
| Mn II | 1.005 | 1.004 | 1.122 | 1.119 | 1.026 | 1.019 | 0 | 0 |
| **Ni II** | 0.983 | 0.978 | **0.747** | 1.095 | 1.013 | 1.036 | 0.108 | **0** |
| Ti II | 1.015 | 0.984 | 0.896 | 0.974 | 1.030 | 1.018 | 0 | 0 |
| Sc II | 1.003 | 1.002 | 1.012 | 0.958 | 0.995 | 1.042 | 0 | 0 |
| Al II | 0.992 | 1.005 | 0.767 | 1.094 | 0.978 | 0.934 | 0 | 0 |
| O I | 1.005 | 0.994 | 0.944 | 0.998 | 1.005 | 1.054 | 0 | 0 |
| S II | 1.035 | 0.895 | 0.855 | 0.968 | 1.195 | 1.055 | 0 | **0.350** |
| C II | 0.940 | 0.763 | 1.754 | 1.113 | 1.007 | 1.209 | 0 | 0 |
| O II | 0.816 | 0.983 | 0.740 | 0.985 | 1.097 | 0.995 | 0 | 0 |

**Reference-only** (no matrix R_bf in this configuration — §5):
Fe III 1.376/2.520/1.407 at s20/s30/s45; Si III 1.063/2.141/1.081;
S III 1.092/1.670/1.432; Co III 0.996/1.816/1.439; Fe IV 0.796/1.244/1.000 at
s0/s8/s20 (zero beyond).

**Model composition (`step10.out`, `data/…/abundances.csv`) — load-bearing:**

```
Z=28 Ni : shells 0..11 only        Z=20 Ca : shells 4..49
Z=27 Co : shells 0..11 only        Z=16 S  : shells 4..49
Z=26 Fe : shells 0..11 only        Z=14 Si : shells 4..49
Z= 8 O  : ZERO in all 50 shells    Z= 6 C  : ZERO in all 50 shells
Mg(12) Al(13) Sc(21) Ti(22) V(23) Cr(24) Mn(25): absent from abundances.csv entirely
```
Only **Si II, Ca II, S II** carry atoms anywhere in s20–49; only
**Fe II, Co II, Ni II** carry atoms at s0–s11.

### 3.2 Ground-level ratio (the channel B4 routes; `src/lumina_plasma.c:14499`)

| ion | s0 | s8 | s20 | s30 | s45 | s49 |
|---|---|---|---|---|---|---|
| Si II | 1.084 | 0.890 | **3.364** | **6.671** | 1.710 | 1.632 |
| Fe II | 1.013 | 0.994 | 1.187 | **6.411** | 1.666 | 1.241 |
| Mn II | 1.111 | 0.949 | **2.404** | **3.912** | 1.752 | 1.548 |
| Mg II | 1.058 | 0.961 | 1.725 | 2.563 | 1.516 | 1.346 |
| Cr II | 1.030 | 0.906 | **2.197** | 1.515 | 1.512 | 1.472 |
| Co II | 0.987 | 0.966 | 1.001 | **1.843** | 1.466 | 1.352 |
| Ca II | 1.019 | 0.998 | 0.772 | 0.878 | 1.042 | 1.110 |
| Ni II | 0.666 | 0.987 | 0.688 | 1.006 | 1.584 | 1.000 |
| Ti II | 1.180 | 1.036 | 0.430 | 1.183 | 0.777 | 1.482 |
| Sc II | 0.980 | 0.997 | 1.043 | 1.030 | 0.997 | 0.957 |
| O I | 0.989 | 0.991 | 0.978 | 1.039 | 1.029 | 1.052 |
| S II | 0.903 | 0.235 | 0.922 | — | — | — |

### 3.3 The realized (population-weighted) ratio — **and the prelim's bracket claim is false**

The clean run dumped `lumina_levelpop_resolve_{raw,ema}.csv` (writer
`src/lumina_cuda.cu:855-895`; gate `LUMINA_NLTE_FINAL_RESOLVE`, block
`src/lumina_cuda.cu:8592-8686`). Schema confirmed against the writer:
`shell,Z,ion,level_num,E_eV,g,n_k,n_ground,b_k,has_sigma,n_sig_pos`, a **single
post-iter-11 final-resolve snapshot with no `iter` column**; `level_num` is
`atom->level_num[gl]`, which joins 1:1 to `levels.csv:level_number`
(157 Si II / 2698 Fe II / 2747 Co II / 1000 Ni II — exact match).
`Γ_pop = Σ_l n_k[l]·R_bf[l] / Σ_l n_k[l]` replaces the Boltzmann proxy.

Si II over the registered band s20–35 (`step12.out`):

| weighting | min | max |
|---|---|---|
| ground | 2.742 | 6.853 |
| Boltzmann (the §3.1 proxy) | 2.192 | 3.591 |
| **pop, resolve_raw (realized)** | **1.621** | **3.492** |
| pop, resolve_ema | 1.738 | 3.748 |

PRELIM_REPORT §2.2 asserted "the realized matrix effect must lie **between**
ground and Boltzmann". **Falsified**: the realized value lies *below both* in
**13 of the 16 band shells** (inside only at s33, s34, s35). At s30 the realized
ratio is 3.209 (raw) / 3.518 (ema) vs Boltzmann 3.379 and ground 6.671.

Populations exist only where the element does, so `ratio_pop_*` is available for
**Si II, Ca II, S II** at s20–49 and for **Fe II, Co II, Ni II** at s0–s8 only;
it is NaN elsewhere by construction (`step10.out`).

---

## 4. Y3 pre-registration — mechanical re-derivation (P1–P8)

Domain, weighting and bracket edges are read verbatim from PRELIM_REPORT §4.1 /
`runner2_spool/hold/56_parity52_c2matrixbf.sh`. Verdicts from
`y2_prereg_verdicts.csv` + `y2_amended_brackets.csv`.

| # | quantity | clean iter-11 number | registered | verdict |
|---|---|---|---|---|
| **P1** | Si II Γ, **s20–35**, pop-wtd (Boltzmann proxy) | min **2.192** (s22), max **3.591** (s20); all 16 shells > 1 (UP ✓) | ×1.6–3.5 | **OUTSIDE-RANGE** — max exceeds by **+0.091 (+2.6 %)** |
| P1′ | Si II, s20–35, **realized pop-wtd** (resolve_raw) | min **1.621**, max **3.492** | ×1.6–3.5 | WITHIN-RANGE (but resolve_ema max 3.748 → OUTSIDE by +0.248) |
| **P1″** | Si II, s20–35, ground | min **2.742** (s21), max **6.853** (s29) | ×2–6.5 | **OUTSIDE-RANGE** — max exceeds by **+0.353 (+5.4 %)** |
| **P2** | Fe II Γ, **s20–35**, pop-wtd (Boltzmann) | min **1.770** (s20), max **3.504** (s30); all UP | ×1.4–3.5 | **OUTSIDE-RANGE** — max exceeds by **+0.0044 (+0.13 %)** |
| **P2′** | Fe II, s20–35, ground | min **1.187** (s20), max **7.005** (s29) | ×1.2–6.5 | **OUTSIDE-RANGE** — min below by **0.013**, max above by **+0.505 (+7.8 %)** |
| | *(Fe X_elem = 0 across the whole s20–35 band; no realized pop-wtd value exists)* | | | |
| **P3** | Mg II s30 | **1.680** | ×1.4–1.7 | **WITHIN-RANGE** (Mg absent from the model — diagnostic only) |
| **P3** | Co II s30 | **1.573** | ×1.4–1.7 | **WITHIN-RANGE** (Co X=0 at s30 — diagnostic only) |
| **P4** | Ni II s20 | **0.7468** (DOWN ✓) | ×0.75–0.98 | **OUTSIDE-RANGE** — below by **0.0032 (0.43 %)** (Ni X=0 at s20) |
| **P5** | s0/s8 `|ratio−1| ≤ 0.035` for every ion with Γ > 1 s⁻¹ | **5 of 28** ion×shell exceed 0.035. Worst: **O II s0 0.184**, **S II s8 0.105**, C II s0 0.060, Si II s8 0.0358, S II s0 0.0353. dev range 0.0017–0.1838 | ≤0.035 | **OUTSIDE-RANGE**. Restricted to ions that actually have atoms (X>0): **2 of 9** exceed — **S II s8 (0.1047)** and **Si II s8 (0.0358)**; the other **seven** rows (Fe II s0/s8, Co II s0/s8, Ni II s0/s8, Ca II s8) are all ≤ **0.0238** |
| **P6** | Ca II, Cr II, Mn II, Ti II, Sc II, O I ≤ 15 % anywhere | 6 report shells: **0.896–1.250**; all 50 shells: 0.684–1.345. Offenders: **Ca II s20 1.250**, Cr II s20 1.147, Mn II s20 1.122, **Ti II s20 0.896** | ×0.92–1.15 | **OUTSIDE-RANGE**. Restricted to ions with atoms (= Ca II only): **0.990–1.250**, still OUTSIDE |
| **P7** | Si III / S III / Fe III / Fe IV / Co III matrix R_bf: exactly 0 change | **N-A** for the A/B (evaluated inside parity52). Scoping facts **re-verified on the clean log**: `TOPSTAGE_IV` **0 hits**, `STAGE4` **0 hits**, `C2_MATRIX_BF` **0 hits** in `stdout.log`; banner `stdout.log:7286` = `[NLTE-GEMM] init: 16 pairs, 10340 phot levels`; the 16 base-pair **lower** ions' slot-table level counts sum to **10340** exactly | 0 | **SCOPING CONFIRMED — unchanged** |
| **P8** | R1 `parity_gamma_phot` output identical between arms | **N-A** — no computation. Code fact: `c2mx` (the `LUMINA_C2_MATRIX_BF` flag) appears **only** at `src/lumina_plasma.c:14541,14542,14544,14545,14601,14621,14625,14630` — entirely inside the matrix R_bf loop. `parity_gamma_phot` (`src/lumina_plasma.c:1725-1793`) never reads it and takes `bfrow` unconditionally under parity (`src/lumina_plasma.c:1731-1732`), so its output is arm-independent by construction | 0 | **CODE-CONFIRMED** |

### 4.1 Amended P1–P4 brackets (`y2_amended_brackets.csv`)

Rule fixed before the numbers were read (`step12_amendment.py` docstring):
`lo = floor_1dp(min at it11)`, `hi = ceil_1dp(max at it11 × the **measured**
it10→it11 drift at the shell carrying that max)`. The drift multiplier is
measured, not chosen, because the field is still moving (§2).

| # | registered | it11 observed (min@shell – max@shell) | drift at max, it10→11 | **amended bracket** |
|---|---|---|---|---|
| P1 Si II s20–35 pop-wtd | ×1.6–3.5 | 2.192@s22 – 3.591@s20 | ×2.004 | **×2.1–7.2** |
| P1 Si II s20–35 ground | ×2.0–6.5 | 2.742@s21 – 6.853@s29 | ×1.210 | **×2.6–8.3** |
| P2 Fe II s20–35 pop-wtd | ×1.4–3.5 | 1.770@s20 – 3.504@s30 | ×1.132 | **×1.7–4.0** |
| P2 Fe II s20–35 ground | ×1.2–6.5 | 1.187@s20 – 7.005@s29 | ×1.290 | **×1.1–9.1** |
| P3 Mg II s30 | ×1.4–1.7 | 1.680 | ×1.048 | ×1.6–1.8 (registered bracket holds at it11) |
| P3 Co II s30 | ×1.4–1.7 | 1.573 | ×1.079 | ×1.5–1.7 (registered bracket holds at it11) |
| P4 Ni II s20 | ×0.75–0.98 | 0.7468 | ×0.990 | **×0.7–0.8** |

The Si II s20 drift of ×2.004 in a single iteration is the dominant source of
bracket width; it is real (the prelim's own it10 value at that shell was 1.792)
and it is why a one-iteration bracket cannot be trusted.

Also amend, though outside the P1–P4 "size bracket" rule:
* **P5** → the registered `≤ 0.035` is exceeded by **S II s8 = 0.1047** even
  after restricting to ions with atoms; register **≤ 0.11**, or restrict the
  null to `X_elem > 0` **and** `Γ > 100 s⁻¹` (which excludes S II s8,
  Γ = 4.62 s⁻¹) and register **≤ 0.036** — the observed max under that
  restriction is Si II s8 = **0.0358**, which still just breaks the old 0.035.
* **P6** → Ca II s20 = 1.250; register **×0.92–1.30** (or drop the five
  zero-abundance ions from the item, leaving Ca II 0.990–1.250).

### 4.2 Characterization items refreshed (record, do not score)

| # | item | prelim (it10) | **formal (it11)** |
|---|---|---|---|
| C1 | ratio still drifting | Si II s30 2.00→2.20→2.80 | Si II s30 …2.80→**3.38**, still no plateau; s20 1.79→**3.59** |
| C2 | C1-fill share of outer ground edges (Y3 leaves it on C1) | 0.25–0.76 at s45/s49 | Ni II 0.605/0.836, Fe II 0.439/0.622, Co II 0.264/0.601, Si II 0.154/0.370 at s45/s49 |
| C3 | 250 kK rail surviving into the field, NIR bins 0/1/3/4/5 | 1,622 / 13,200 | **1,856 / 14,400** (rail fires 1,934; degen 51, pin 27) |
| C4 | `LUMINA_C1_DEGEN_FALLBACK` effectively inert | 28 / 13,200 | **51 / 14,400** (0.35 %) |
| C5 | coarse bins 20–23 (100–241 Å) `empty` in all 50 shells | 100 % | **100 %** (bins 16–19 mixed pin/empty: 26/30/45/49 empty) |
| C6 | MC-unsampled fine-bin fraction | 0.400 → 0.332 | 0.3998 → **0.3309**, fall has flattened |
| C7 | conditional on `LUMINA_C1_SUPERBIN_TEPIN=1` | — | unchanged; gate confirmed at `stdout.log:76` |
| C8 | mode census | fit 58.1 / empty 31.1 / pin 10.6 / degen 0.21 % | fit **57.98** / empty **30.92** / pin **10.74** / degen **0.35** % |
| C9 | **NEW** — realized pop-weighting is *below* the ground↔Boltzmann bracket | claim asserted | **falsified**, 13/16 band shells (§3.3) |
| C10 | **NEW** — model composition restricts which P-items can move the solution | not noted | Fe/Co/Ni exist only s0–11; Mg/Cr/Mn/Ti/Sc/Al/V and O/C do not exist at all (§3.1) |

### 4.3 Mechanism (§2.3 of the prelim) — reconfirmed at iter 11

The gap still lives in the T_e-pinned coarse bins. Bin 14 (728.8–905.6 Å),
`pin` in all 50 shells at iter 11, shape distortion with both fields
renormalised to unit bin energy (`step6.out`):

| shell | T_e | Wien contrast `exp(−hΔν/kT_e)` | C1/raw RED third | C1/raw BLUE third |
|---|---|---|---|---|
| 0 | 21203 | 0.170 | 1.180 | 0.817 |
| 8 | 12008 | 0.044 | 0.997 | 1.132 |
| 20 | 8799 | 0.014 | 1.328 | **0.189** |
| 30 | 8381 | 0.011 | 1.526 | **0.153** |
| 45 | 11897 | 0.042 | 1.623 | **0.346** |
| 49 | 13066 | 0.056 | 1.312 | **0.585** |

Edge placement inside its own coarse bin (0 = red, 1 = blue) is unchanged:
Si II 758.5 Å → bin 14, 0.82; Fe II 765.9 → 0.78; Cr II 752.1 → 0.86;
Mg II 824.6 → 0.43; Ni II 682.4 → bin 15, 0.29; Co II 725.7 → bin 15, 0.01.

---

## 5. Scoping (unchanged, re-verified on the clean log)

`nlte_get_pairs` with `LUMINA_NLTE_STAGE4` unset returns the 16 **base** pairs
(`src/lumina_plasma.c:7268-7273`); only the **lower** member enters the R_bf
loop (`src/lumina_plasma.c:14564`). Against the slot table at
`stdout.log:278-308` the lower ions and their level counts are

```
Si II 157  Ca II 77   Fe II 2698  S II 324   Co II 2747  Ni II 1000
C II  338  Mg II 80   Ti II 600   Cr II 600  Al II 80    Sc II 500
V II    0  Mn II 600  O I   199   O II  340         sum = 10340
```
matching `stdout.log:7286` `[NLTE-GEMM] init: 16 pairs, 10340 phot levels`
exactly. `LUMINA_TOPSTAGE_IV` and `LUMINA_NLTE_STAGE4`: **0 hits** in the clean
`stdout.log`, so Si III / S III / Fe III / Fe IV / Co III receive **no matrix
R_bf at all**; their numbers above are reference-only.

---

## 6. Prelim-vs-formal comparison

### 6.1 Hygiene (no regression anywhere)

| axis | prelim (550,000 rows, it0–10) | **formal (600,000 rows, it0–11)** | change |
|---|---|---|---|
| `J_raw` non-finite / negative | 0 / 0 | **0 / 0** | none |
| `bfr` non-finite / negative | 0 / 0 | **0 / 0** | none |
| `J_raw` zeros | 198,612 | 215,159 | scales with row count |
| `j_nu_count` missing (`-1`) | 0 | **0** | none |
| structural identity mismatches | 0 + 0 | **0 + 0** | none |
| roundtrip max \|1−r\| | 5.6e-07 | **5.619e-07** | none |
| `bfr` identity out-of-window | 0 / 351,388 | **0 / 384,841** | none |
| `bfr` identity min / max | 0.997355 / 1.002652 | **0.997355 / 1.002652** | none |
| closed-form vs direct, max rel | 1.16e-03 | **1.142e-03** | slightly better |
| `leak_frac` (threshold cut) | 0.000000 | **0.000000** | none |
| T_e crosscheck | 3 values at it10, exact | **35 pinned rows over 12 iters, max 0.46 K** | strengthened |

**No hygiene regression. No structural-identity mismatch.**

### 6.2 Headline numbers (`y2_prelim_vs_formal_headline.csv`)

Boltzmann `Γ(C2)/Γ(GEMM)` — prelim it10 → formal it11 (Δ):

| ion | s0 | s8 | s20 | s30 | s45 | s49 |
|---|---|---|---|---|---|---|
| Si II | 0.967→0.971 (+0.004) | 0.970→0.964 (−0.006) | 1.792→**3.591 (+1.799)** | 2.801→**3.379 (+0.578)** | 1.193→1.256 (+0.063) | 1.028→1.048 |
| Fe II | 1.022→1.021 | 0.985→0.976 (−0.009) | 1.472→**1.770 (+0.298)** | 3.096→**3.504 (+0.408)** | 1.312→1.350 | 1.078→1.090 |
| Mg II | 1.018→1.020 | 1.004→1.002 | 1.388→**1.627 (+0.239)** | 1.604→1.680 (+0.076) | 1.012→1.025 | 0.981→0.988 |
| Co II | 1.001→1.006 | 0.990→0.985 | 1.064→1.140 (+0.076) | 1.458→**1.573 (+0.115)** | 1.191→1.196 | 1.045→1.073 |
| Ca II | 1.019→1.020 | 1.011→1.011 | 1.148→**1.250 (+0.101)** | 1.119→1.109 | 1.093→1.102 | 1.014→0.990 |
| Ni II | 1.010→0.983 (−0.027) | 0.978→0.978 | 0.754→0.747 | 1.080→1.095 | 1.046→1.013 | 0.973→1.036 |
| Ti II | 1.011→1.015 | 0.987→0.984 | 0.967→0.896 (−0.071) | 0.998→0.974 | 1.103→1.030 (−0.073) | 1.010→1.018 |
| S II | 1.024→1.035 | 0.902→0.895 | 0.984→0.855 (−0.130) | 0.990→0.968 | 0.985→**1.195 (+0.211)** | 1.369→1.055 (−0.313) |
| O II | 0.999→**0.816 (−0.183)** | 1.690→**0.983 (−0.707)** | 0.908→0.740 (−0.167) | 1.001→0.985 | 1.037→1.097 | 1.001→0.995 |
| Fe III | 1.005→0.990 | 1.036→1.019 | 1.308→1.376 | 2.361→2.520 | 1.423→1.407 | 1.265→1.227 |
| Si III | 0.928→0.940 | 1.018→1.016 | 0.937→1.063 (+0.126) | 1.976→2.141 | 1.053→1.081 | 0.873→0.881 |

**Qualitative changes flagged:**
* **13 of 120** ion×shell cells cross 1.0 (sign flip of the effect direction),
  ranked by how far the it11 value now sits from 1.0:
  Fe IV s0 (0.204), S II s45 (0.195), S III s20 (0.092), Si III s20 (0.063),
  Ni II s49 (0.036), Ni II s0 (0.017), O II s8 (0.017), O II s30 (0.015),
  Ca II s49 (0.010), Fe III s0 (0.010), Sc II s45 (0.005), O II s49 (0.005),
  Co III s20 (0.005). **None of them is a P1–P4 subject.** The two large ones
  (Fe IV s0, S II s45) are on ions with `X_elem = 0` at that shell; the two
  mid-sized ones (S III s20, Si III s20) are reference-only ions that receive no
  matrix R_bf at all (§5). The remaining nine are ≤ 3.6 % from 1.0.
* **Range exits** on the registered brackets: P1 (Si II, both weightings),
  P2 (Fe II, both), P4 (Ni II s20), P5 (5/28), P6 (Ca II s20, Ti II s20).
  Every P1–P2 exit is on the **high** side except the Fe II ground minimum
  (1.187 vs the registered 1.2 floor, a 1.1 % undershoot at s20 only), and all
  are in the **registered direction** (UP) — the direction claim is
  strengthened, the size claim is broken.
* Ground table (§3.2 vs prelim §2.2): Si II s20 2.084→**3.364**,
  Cr II s20 1.340→**2.197**, Mn II s20 1.979→**2.404**, Ni II s30 0.763→1.006,
  Ti II s49 0.228→1.482. Si II/Fe II s30 essentially frozen (6.364→6.671,
  6.371→6.411).
* §3.3 falsifies the prelim's ground↔Boltzmann bracketing claim outright.

---

## Bottom line

**PRE-REGISTRATION MUST BE AMENDED: P1, P2, P4 (size brackets); additionally
P5 and P6.**

New brackets, from `y2_amended_brackets.csv`:

* **P1** Si II Γ, s20–35, UP: pop-wtd **×2.1–7.2** (was ×1.6–3.5);
  ground **×2.6–8.3** (was ×2–6.5).
* **P2** Fe II Γ, s20–35, UP: pop-wtd **×1.7–4.0** (was ×1.4–3.5);
  ground **×1.1–9.1** (was ×1.2–6.5).
* **P4** Ni II Γ, s20, DOWN: **×0.7–0.8** (was ×0.75–0.98).
* **P5** inner-shell null: **≤ 0.11** as registered, *or* **≤ 0.036** after
  restricting the item to ions with `X_elem > 0` **and** `Γ > 100 s⁻¹`
  (observed max then = 0.0358, Si II s8).
* **P6** NULL-ish six: **×0.92–1.30** (was ×0.92–1.15), driven by Ca II s20.
* **P3** stands as registered (Mg II 1.680, Co II 1.573, both inside ×1.4–1.7).
* **P7** stands: scoping re-verified on the clean log, unchanged.
* **P8** stands: `c2mx` is confined to `src/lumina_plasma.c:14541-14630`; the R1
  closure is arm-independent by construction.

The instrument **PASSES** certification — no hygiene regression, zero
structural-identity mismatches, every normalisation identity closing at ≤
5.62e-07 (roundtrip) and ≤ 0.265 % (ν-weighting). What fails is the *size*
pre-registration, because the field was still drifting monotonically upward at
the scheduled end of the run (Si II s30 2.80 → 3.38, s20 1.79 → 3.59 on the last
iteration alone). **Direction (UP for Si II/Fe II/Mg II/Co II, DOWN for Ni II
s20) survived every iteration and is the only part of P1–P4 that should carry
scoring weight.**

Two further facts the driver's seat should weigh before re-queuing 56:
1. Of the eight P1–P6 subjects, only **Si II** (P1) and **Ca II** (P6) have any
   atoms in the shells where they are registered. **Fe II (P2), Mg II and Co II
   (P3), Ni II (P4), and Cr/Mn/Ti/Sc II (P6) have `X_elem = 0` in their
   registered domain**, so those items can only ever move a rate diagnostic, not
   `b_k`, `n_e` or the spectrum — they cannot be falsified by HS5 or by any
   solution-side observable.
2. The realized population-weighted Si II effect (1.62–3.49 over s20–35) is
   **smaller than both** the ground and Boltzmann proxies over 13 of 16 shells,
   so the pre-registered brackets were built on a proxy that overstates the
   effect.

---

## Appendix A — script adaptations vs the prelim copies

Every line changed relative to
`../y2_c2bfr_certification/{y2_common,step1..step8}*.py`. Science untouched:
only paths, iteration labels, log names, and one CSV column-naming scheme.

**`y2_common.py`**
1. Docstring: "PRELIMINARY-on-partial … iter 0-10 … killed_partial" → "FORMAL …
   clean complete 12-iteration parity46 … iters 0-11, 600,000 rows, final
   iteration index 11".
2. `_CERT` → `…/y2_c2bfr_certification_formal`.
3. `RUN` default → `$ROOT/logs/coevolve_consume_parity46` (was
   `…_killed_partial`); `Y2_RUN` env override kept.
4. `OUT` default → the formal dir; `Y2_OUT` override kept.
5. Added `STDOUT_LOG = $Y2_STDOUT or $RUN/stdout.log` (was `stdout_partial.log`,
   referenced only in comments before).
6. Added `IT_FINAL = 11`.
   *`load_dumps()`'s `assert len(c2) == ni*ns*NB` passes unmodified: 600,000 =
   12 × 50 × 1000.*

**`step1_hygiene.py`**
7. Docstring → FORMAL; note that the 3-iteration window is now (9,10,11).
8. Added `IA, IB, IC = iters[-3], iters[-2], iters[-1]` and renamed the
   stability columns from the hardcoded `med_rel_d_8_9 / med_rel_d_9_10 /
   wmean_* / Jint_i8 / Jint_i9 / Jint_i10 / Jint_rel_d_9_10` to f-string names
   `med_rel_d_9_10 / med_rel_d_10_11 / … / Jint_i9 / Jint_i10 / Jint_i11 /
   Jint_rel_d_9_10 / Jint_rel_d_10_11`. Same dict-literal → `dict(...)` replaced
   by a `{}` literal so the keys can be f-strings. **Both** deltas are now
   emitted (the prelim wrote both but printed only one).
9. Added three summary prints: the per-shell converging? boolean, the all-shell
   medians of both deltas, and `max |median ΔJ_raw − median Δbfr|` over 50
   shells.
10. `bfr` stability: `bfr_sum_i9/i10` → f-string `bfr_sum_i10/i11`; print label
    "iter 9->10" → "iter 10->11".

**`step2_bfr_identity.py`**
11. Docstring → FORMAL. 12. Print label "per-shell (iter 10)" → "(iter 11 = last
    block)". (`R[-1]` is unchanged and now means iter 11.)

**`step3_path_ratio.py`**
13. Docstring → FORMAL. 14. `IT = -1` comment "dump iter 10 = the last complete
    block" → "dump iter 11 = the last block of the clean 12-iter run".
15. Comment `stdout_partial.log:278-308` → `stdout.log:278-308`.
16. Print header "dump iter 10" → "dump iter 11".

**`step4_n1_pathology.py`**
17. Docstring → FORMAL. 18–20. Three print headers "iter 10" → "iter 11".

**`step5_gamma_ion.py`**
21. Docstring → FORMAL; the T_e provenance line now cites
    `stdout.log:36005 '[CMFGEN] iter 11: T_e[0]=21203K T_e[25]=8501K
    T_e[49]=13066K'` and points at step9 for the mechanical check.
22–23. Two print headers "dump iter 10" → "dump iter 11".

**`step6_mechanism.py`**
24. Docstring → FORMAL. 25. Print "total fill bins at iter 10" → "iter 11".

**`step7_robustness.py`**
26. Docstring → FORMAL. 27. Hardcoded print "last 3 iters (8,9,10)" → f-string
    `({ni-3},{ni-2},{ni-1})`. 28. "GROUND edge kernel (iter 10)" → f-string
    `(iter {ni-1})`. (`last = rb[rb["iter"] >= ni-3]` and `if it == ni-1` were
    already generic.)

**`step8_crosscheck.py`**
29. Docstring → FORMAL. 30. Print header "dump iter 10" → "dump iter 11".

**New scripts (no prelim counterpart)**
* `step9_te_scoping.py` — mechanises §1.5 (T_e vs every `[CMFGEN] iter` line in
  `stdout.log`, with the `pin`/`empty` mode annotation) and the P7 scoping facts
  (`TOPSTAGE_IV`/`STAGE4` hit counts, the GEMM banner, the 16 lower-ion level
  sum). Outputs `y2_te_crosscheck.csv`, `y2_scoping.csv`.
* `step10_preregistration.py` — the P1–P6 bracket verdicts over the *registered*
  domain (all shells 20–35, not just s20/s30), the model-composition mask, and
  the realized population-weighted ratio. `Y2_IT=-2` re-runs it at iter 10 for
  the drift basis. Outputs `y2_prereg_allshells{,_it-2}.csv`,
  `y2_prereg_verdicts{,_it-2}.csv`.
* `step11_prelim_vs_formal.py` — determinism (line-level dump diff + derived
  quantities over the shared iters) and the §6 side-by-side tables. Reads the
  prelim directory read-only. Outputs `y2_prelim_vs_formal_headline.csv`.
* `step12_amendment.py` — the amended brackets of §4.1 and the test of the
  prelim's ground↔Boltzmann bracketing claim. Outputs
  `y2_amended_brackets.csv`.

No step script errored after adaptation; no fix beyond the list above was
needed.

## Appendix B — reproduce

```bash
cd validation/cmfgen_toy06_19p48d/analysis/y2_c2bfr_certification_formal
for s in 1 2 3 4 5 6 7 8 9; do python3 step${s}_*.py > step${s}.out 2>&1; done
python3 step10_preregistration.py            > step10.out 2>&1
Y2_IT=-2 python3 step10_preregistration.py   > step10_it10.out 2>&1
python3 step11_prelim_vs_formal.py           > step11.out 2>&1
python3 step12_amendment.py                  > step12.out 2>&1
```
(step1 must run first: it writes the `_cache_*.npy` arrays every later step
loads. Total wall time ≈ 45 s.)
