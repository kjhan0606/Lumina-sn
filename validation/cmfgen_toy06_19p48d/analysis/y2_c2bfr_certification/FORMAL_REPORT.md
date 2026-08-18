# Y2 정식 인증 (Opus, 2026-07-30, clean parity46 덤프)

**Y2 instrument certification of the C2 bf-rate estimator (`bfr`) — FORMAL, on the
clean 12-iteration parity46 run.**

Supersedes `PRELIM_REPORT.md` (PRELIMINARY-on-partial, iters 0–10 of a
walltime-killed run). Judgement vocabulary is **CONFIRMED / REFUTED /
UNVERIFIABLE** only; adoption/rejection is the driver's seat.

| | |
|---|---|
| input | `logs/coevolve_consume_parity46/lumina_c2_bfr_dump.csv` (30.6 MB), `lumina_c1_bins.csv`, `lumina_ion_pops.csv`, `lumina_levelpop.csv` |
| block | 600,000 rows = **12 iters (0–11) × 50 shells × 1000 fine bins**, every block complete |
| schedule | `LUMINA_PURE_CMFGEN_ITER=12`, `argv … 12 spectrum nlte`, rc 0 — **ran to the intended end** |
| arm | `LUMINA_C2_MATRIX_BF` **absent** from the 119-var RUN FOOTER ⇒ this is arm A (GEMM), as designed |
| baseline | physics byte-identical to parity44 (`VERDICT_NOTE.md`) |
| outputs | `clean/y2_*.csv`, `clean/step{1..10}.out`; PRELIM's CSVs untouched |

---

## 0. Provenance and determinism (new, offline)

`y2_common.py` now takes `Y2_RUN` / `Y2_OUT` from the environment, **defaulting to
the PRELIM paths**, so both reports reproduce from one script set.

**Determinism of the estimator across two independent executions.** The
killed-partial run and this clean run share config; their overlapping blocks
(iters 0–10) can therefore be diffed directly:

| file, iters 0–10 | result |
|---|---|
| `lumina_c1_bins.csv` (13,200 rows) | **byte-identical** (md5 `271fa1ee…`) |
| `lumina_c2_bfr_dump.csv` (550,000 rows × 3 numeric cols) | **1 differing line**: `it6,s33,bin442` `J_raw 5.090372e-07 → 5.090371e-07` (2e-7 rel, last printed digit) |

The instrument reproduces to the print precision across executions. This is the
same last-digit-flip class already attributed in `VERDICT_NOTE.md` and is not a
new defect.

**Scoping gates re-verified on this run's own log** (`stdout.log`):
`[NLTE-GEMM] init: 16 pairs, 10340 phot levels (10120 active: 9964 CMFGEN +
156 Kramers)` at line 7286; `TOPSTAGE` **0 hits**; `LUMINA_NLTE_STAGE4` absent;
`LUMINA_C1_SUPERBIN_TEPIN=1`, `LUMINA_C1_DEGEN_FALLBACK=1` present.

---

## 1. Hygiene battery — **PASS on every offline-checkable axis** (unchanged verdict)

### 1.1 Finiteness / sign / completeness

| array | non-finite | negative | zeros | range |
|---|---|---|---|---|
| `J_raw` | **0** | **0** | 215,159 | [0, 1.1788e-02] |
| `bfr` | **0** | **0** | 215,159 | [0, 1.1844e+23] |
| `j_nu_count` | — | **0** entries of `-1` | — | [0, 15,567,677] |

Writer grid self-check: `max |nu_mid_file/nu_mid_hdr − 1| = 4.99e-07`.

### 1.2 Structural identity `{cnt==0} ≡ {bfr≤0} ≡ {J_raw≤0}`

`n_cnt0_bfr_pos = 0`, `n_cnt_pos_bfr0 = 0` — **0 mismatches in 600,000 rows**.
The `bfr > 0` test at `lumina_plasma.c:14634` fires on exactly the MC-sampled
bins; the documented fallback semantics are exact. **CONFIRMED.**

### 1.3 Cross-dump roundtrip

`Σ_f J_raw·dν_f` per coarse bin vs the C1 dump's `J_bin`: 9,947 non-empty bins,
**0 disagreements on which bins are zero**, ratio median `1.000000001`,
**max |1−r| = 5.619e-07** — the `%.6e` print precision. **CONFIRMED.**

### 1.4 `bfr` normalisation, independent certification

`σ·bfr / (pref·J_raw) = ν_mid/ν_eff` must lie inside the log-bin geometric window
`[exp(−dlog/2), exp(+dlog/2)] = [0.997354, 1.002653]`. Over **384,841 paired bins**:

```
min = 0.997355   p1 = 0.998454   median = 0.999997   p99 = 1.001624   max = 1.002652
outside window = 0 bins (0.0000%)      shells with any excursion = 0 / 50
```

Extremes land on the theoretical bounds to 6 decimals. **CONFIRMED.**

### 1.5 T_e recovery cross-check (bonus)

`T_R[final, shell, cbin=14]` under TEPIN *is* `T_e`. Recovered
`s0 = 21203`, `s25 = 8501`, `s49 = 13066 K` vs `stdout.log:36005`
`[CMFGEN] iter 11: T_e[0]=21203K T_e[25]=8501K T_e[49]=13066K` — **exact at all
three**. (PRELIM checked iter 10 and also matched.)

### 1.6 Convergence — **still NOT converged at the scheduled end**

Median per-bin `|ΔJ_raw|/J_raw`, **iter 10 → 11** (the last available pair):

| shell | 0 | 8 | 20 | 30 | 45 | 49 |
|---|---|---|---|---|---|---|
| median \|ΔJ\|/J | 0.046 | 0.103 | **0.311** | **0.206** | **0.193** | **0.191** |
| integrated ∫J dν | −0.018 | −0.076 | +0.129 | +0.111 | +0.013 | −0.002 |

`bfr` tracks `J_raw`: over all 50 shells the two median relative changes differ by
at most **0.0065** — another confirmation of §1.4.

MC-unsampled fine-bin fraction (all-shell mean) `0.3998 (it0) → 0.3309 (it11)`,
still falling. **Running the full 12 iterations did not buy convergence**: shells
≥ 20 move 19–31 % on the final step, no better than the 21–29 % PRELIM saw on its
final step.

---

## 2. NEW — composition scoping. Most of the PRELIM headline table is on empty ions.

The PRELIM tabulated 15 lower ions × 6 shells without asking whether each ion has
any population in that shell. **toy06 is a stratified model.** From
`data/…/abundances.csv` and `lumina_ion_pops.csv` (byte-identical to parity44):

| element | shells with X > 0 |
|---|---|
| Ni (28), Co (27), Fe (26) | **0 – 11** (IGE core) |
| Ca (20), S (16), Si (14) | **4 – 49** (IME) |
| everything else | **none** |

Of the 16 matrix-consumed lower ions:

* **6 exist anywhere**: Si II, S II, Ca II (s4–s49); Fe II, Co II, Ni II (s0–s11).
* **10 have `n_ion = 0` in all 50 shells**: C II, Mg II, Ti II, Cr II, Al II,
  Sc II, V II, Mn II, O I, O II. `lumina_levelpop.csv` independently confirms
  `n_k = 0` on every level of every one of them, in every shell.

Consequence for the PRELIM headline (`PRELIM §2.1`): the Fe II / Mg II / Co II /
Ni II / Cr II / Mn II / Ti II / Sc II / Al II / O I / C II / O II entries at
**s20 / s30 / s45 / s49 are ratios computed on ions with zero population**. They
are arithmetically real but carry **zero ionization flux**, so Y3 cannot move
anything through them. The full map is `clean/y2_composition_scope.csv`.

---

## 3. Path ratio at convergence — measured three ways

### 3.1 The realized weighting is now measurable (PRELIM §4.4 item 3 closed)

The clean run carries `LUMINA_LEVELPOP_DUMP=1`. From the consumer
(`lumina_plasma.c:14686-14689`), `ACM(ground_hi, sl) += R_bf · f_lev` with
`f_lev = FRAC_OF` = within-SL population fraction, so the ion's total
ionization flux is **exactly** `Σ_lev n_lev · R_bf(lev)` — computable from the
dump's `n_k`. Level-table alignment audit (`E_eV`, `g`, `has_sigma`, levels.csv vs
levelpop.csv): **0 mismatches**. PRELIM's ground/Boltzmann bracket is therefore
replaced by a measurement, not merely re-run.

The realized value falls **outside** the PRELIM's [ground, Boltzmann] bracket in
**12 of 18** checkable ion×shell cells, so the bracket was not conservative.

### 3.2 HEADLINE — realized population-weighted `Γ_phot(C2)/Γ_phot(GEMM)`, final iter, **real ions only**

| ion | s4 | s8 | s12 | s16 | s20 | s25 | s30 | s35 | s40 | s45 | s49 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Si II** | 0.987 | 0.954 | 2.187 | 2.785 | **3.106** | 2.753 | **3.924** | **4.063** | 2.050 | 1.765 | 1.589 |
| Ca II | 1.003 | 1.012 | 1.102 | 1.188 | 1.256 | 1.167 | 1.110 | 1.146 | 1.163 | 1.136 | 0.993 |
| S II | 0.970 | 0.942 | 1.535 | 0.946 | 0.963 | 1.044 | 1.001 | 0.979 | 1.005 | 1.268 | 1.001 |
| Fe II | 0.971 | 0.969 | — | — | — | — | — | — | — | — | — |
| Co II | 0.987 | 0.979 | — | — | — | — | — | — | — | — | — |
| Ni II | 0.979 | 0.971 | — | — | — | — | — | — | — | — | — |

(`—` = ion absent. s0 row: Fe II 1.021, Co II 1.006, Ni II 0.984; Si/S/Ca absent.)

Per-ion spread over every shell the ion exists in:

| ion | shells | min | median | max | max \|r−1\| |
|---|---|---|---|---|---|
| Si II | 4–49 | 0.954 | 2.266 | 4.222 | **322 %** |
| S II | 4–49 | 0.348 | 0.999 | 1.535 | 65 % |
| Ca II | 4–49 | 0.977 | 1.138 | 1.256 | **26 %** |
| Fe II | 0–11 | 0.942 | 0.988 | 1.128 | 13 % |
| Co II | 0–11 | 0.973 | 0.992 | 1.070 | 7 % |
| Ni II | 0–11 | 0.963 | 0.981 | 1.753 | 75 % |

**Shell-total photoionization flux ratio** (all real ions, population-weighted):
s0 0.988 · s8 0.943 · s12 1.535 · s20 0.964 · s30 1.017 · s40 1.328 · s45 1.299 ·
s49 1.057. The shell total is dominated by S II wherever S II is dominant, so the
Si II effect is far larger per-ion than per-shell.

### 3.3 Boltzmann weighting (PRELIM's method, re-run verbatim) — the delta

| ion, shell | PRELIM it10 | FORMAL it11 | realized pop-wtd it11 | status |
|---|---|---|---|---|
| **Si II s30** | 2.80 | **3.38** | **3.92** | real, still rising |
| **Si II s20** | 1.79 | **3.59** | **3.11** | real, still rising |
| Si II s45 / s49 | 1.19 / 1.03 | 1.26 / 1.05 | 1.77 / 1.59 | real |
| Si II s0 / s8 | 0.967 / 0.970 | 0.971 / **0.964** | — / **0.954** | s8 breaks the 3.5 % NULL |
| Ca II s20 / s30 | 1.15 / 1.12 | 1.25 / 1.11 | 1.26 / 1.11 | real |
| S II s8 / s20 | 0.902 / 0.98 | 0.895 / 0.855 | 0.942 / 0.963 | real; s8 breaks NULL |
| Fe II s0 / s8 | 1.022 / 0.985 | 1.021 / 0.976 | 1.021 / 0.969 | real |
| Co II s0 / s8 | 1.001 / 0.990 | 1.006 / 0.985 | 1.006 / 0.979 | real |
| Ni II s0 / s8 | 1.010 / 0.978 | 0.983 / 0.978 | 0.984 / 0.971 | real |
| *Fe II s30* | *3.10* | *3.50* | *n/a* | **empty ion** |
| *Mg II s30* | *1.60* | *1.68* | *n/a* | **empty ion (everywhere)** |
| *Co II s30* | *1.46* | *1.57* | *n/a* | **empty ion** |
| *Ni II s20* | *0.75* | *0.747* | *n/a* | **empty ion** |

Iteration trend of the Boltzmann ratio at s30 (`clean/y2_ratio_by_iter.csv`):

| ion | it8 | it9 | it10 | **it11** |
|---|---|---|---|---|
| Si II | 2.00 | 2.20 | 2.80 | **3.38** |
| *Fe II* | *2.48* | *2.42* | *3.10* | ***3.50*** |
| *Mg II* | *1.41* | *1.54* | *1.60* | ***1.68*** |
| *Co II* | *1.33* | *1.39* | *1.46* | ***1.57*** |

**Si II rose monotonically through the final iteration and never plateaued.** The
extra iteration the clean run bought did not stabilise the magnitude; it raised it.

### 3.4 Mechanism — CONFIRMED for Si II, and a *second, distinct* channel found

Bin-14 shape distortion reproduces (both fields renormalised to unit bin energy,
so only shape is compared):

| shell | T_e | Wien contrast | C1/raw RED third | C1/raw BLUE third |
|---|---|---|---|---|
| 0 | 21203 | 0.170 | 1.18 | 0.82 |
| 8 | 12008 | 0.044 | 1.00 | 1.13 |
| 20 | 8799 | 0.014 | 1.33 | **0.19** |
| 30 | 8381 | 0.011 | 1.53 | **0.15** |
| 45 | 11897 | 0.042 | 1.62 | **0.35** |
| 49 | 13066 | 0.056 | 1.31 | **0.59** |

Edge placement inside the ion's own coarse bin is unchanged (Si II 758.5 Å →
coarse bin 14, position **0.82** toward the blue edge).

**New: per-coarse-bin attribution of the population-weighted (C2 − GEMM)
difference** (not in the PRELIM battery; added because Ca II's edge lies outside
the pinned bins):

| ion | shell | share of the total difference from **pinned bins 14+15** | dominant bin |
|---|---|---|---|
| Si II | 20 / 30 / 45 | **+1.011 / +0.994 / +0.989** | c14 (731–903 Å) |
| **Ca II** | 20 / 30 / 45 | **−0.022 / −0.022 / −0.011** | **c12 (1134–1409 Å), a `fit` bin** |
| S II | 20 / 30 / 45 | +1.50 / −4.66 / +1.01 | cancelling, no clean attribution |

* For **Si II** the T_e-pin mechanism accounts for **99–101 %** of the gap.
  The PRELIM's mechanism claim is CONFIRMED, and now *exclusively* so.
* For **Ca II** it accounts for essentially **none** of it. Ca II's photoion
  integral is dominated by its metastable-level edges at 1218 / 1417 Å, which sit
  in coarse bin 12 — `fit` in all 50 shells (`T_R_med = 29584 K`,
  `W_med = 2.3e-4`). Ca II's ≤26 % deviation is a **colour-fit** error in the
  mid-UV, a channel distinct from TEPIN and not named in the PRELIM.

### 3.5 Structural checks on the comparison itself — both hold

* **GEMM threshold-cut no-op**: `leak_frac = 0.000000` for all of Si II / Fe II /
  Co II / Ni II / S II / Ca II at every shell; **0** CMFGEN σ rows have `σ > 0`
  below their own threshold. The two paths integrate identical bin sets. **CONFIRMED.**
* **Closed-form re-derivation** (`Σ_sampled pref·J_raw + Σ_unsampled pref·J_C1`,
  computed without touching `bfr`) vs the direct consumer form: max relative
  disagreement **1.142e-03**, median **3.126e-05** — inside the 0.265 % in-bin
  ν-weighting bound. **CONFIRMED.** *Y3 is exactly "use the raw MC field wherever
  the MC sampled it."*

---

## 4. GEMM arm — still a **reconstruction**, not a measurement

`R_bf_table` is **not dumped in parity46 either** (0 hits for `R_bf_table` /
`RBF_TABLE` / `R_BF_DUMP` in `stdout.log`). Every GEMM number in this report is
rebuilt in float64 from the C1 `(W, T_R, mode)` columns via
`assemble_JC1` + `K = σ·4π/(hν)·dν`, i.e. the same reconstruction the PRELIM used.
Production runs the same integral in FP32 with `CUBLAS_COMPUTE_32F_FAST_TF32`
(`lumina_nlte_gemm.cu:442`); TF32's ~10-bit mantissa is ~1e-3 relative, negligible
against ratios of 1.5–4×, **but this remains inference**.

**Verdict on the GEMM arm: UNVERIFIABLE** (unchanged from PRELIM). The C2 arm is
measured; the GEMM arm is reconstructed.

---

## 5. N1 pathology — re-run, one number moved materially

| quantity | PRELIM (it10) | FORMAL (it11) |
|---|---|---|
| mode census fit / empty / pin / degen | 58.1 / 31.1 / 10.6 / 0.21 % | 57.98 / 30.92 / 10.74 / **0.35** % |
| 250 kK rail cells | 1,669 / 13,200 (12.6 %) | 1,934 / 14,400 (**13.4 %**) |
| …of those, `fit` (rail reaches the field) | 1,622 (97 %) | **1,856 (96 %)** |
| `DEGEN_FALLBACK` firings | 28 / 13,200 | **51 / 14,400** (still inert) |
| FUV 1200–912 Å, kernel share with D > 2 | 0.176 | 0.181 |
| EUV 912–500 Å, kernel share with D > 2 | 0.324 | 0.341 |
| **deep-EUV 500–100 Å, kernel share D > 2** | **0.332** | **0.159** |
| **deep-EUV ∫kernel C1/raw** | **1.351** | **1.001** |
| FILL bins (MC empty, C1 publishes), all with `bfr==0` | 1,840, True | 1,834, **True** |
| coarse bins 20–23 (100–241 Å) `empty` in all 50 shells | yes | **yes** |

The deep-EUV overshoot the PRELIM flagged (C1 running 1.35× hot) **is not present
at iter 11** (1.001). That characterisation number was an artifact of the
truncation point, not a stable property. Everything else moved ≤ 1 pp.

**The part Y3 cannot repair** — MC-unsampled share of each ground edge's photoion
kernel, final iter, *real ions only*:

| ion | s20 | s30 | s45 | s49 |
|---|---|---|---|---|
| Si II | 0.004 | 0.101 | 0.154 | **0.370** |
| Ca II | 0.000 | 0.003 | 0.008 | 0.011 |
| S II | **0.899** | — | — | — |

(PRELIM's 0.25–0.76 headline was carried by Ni II / Co II / Fe II, which do not
exist in those shells.) For the ions that exist, the untouchable fraction at the
outer edge is **≤ 37 % (Si II s49)** rather than 76 %.

---

## 6. Verdicts on the three PRELIM conclusions

| # | PRELIM claim | verdict |
|---|---|---|
| ① | `bfr` is the ν-weighted twin of the raw MC field integral, not an independent physical quantity | **CONFIRMED** — 384,841 paired bins, 0 outside the geometric window, extremes on the theoretical bounds to 6 decimals; closed-form re-derivation agrees to 1.1e-03 |
| ②a | path ratio **NULL in the deep interior** (≤ 3.5 %) | **s0: CONFIRMED** (max \|r−1\| = 2.1 %, Fe II). **s8: REFUTED** — Si II **4.64 %**, S II **5.85 %** (pop-wtd; 3.6 % / 10.5 % Boltzmann), both with Γ ≫ 1 s⁻¹ |
| ②b | path ratio **UP in the outer shells** | **CONFIRMED, but the carrier is Si II alone** (×1.6–4.2 over s12–s49), plus Ca II ×1.10–1.26. Every other outer-shell UP entry in the PRELIM table is an empty ion |
| ②c | mechanism = `C1_SUPERBIN_TEPIN` bin-14 blue-edge starvation | **CONFIRMED for Si II** and now exclusively so (pinned bins carry 99–101 % of the gap). **Does not cover Ca II** — its gap is 105 % from `fit` coarse bin 12 (mid-UV colour fit), a distinct channel |
| ③ | scope = the 16 lower ions only; Si III / S III / Fe III / Fe IV / Co III get no matrix R_bf | **CONFIRMED** — 16-pair / 10340-level banner, `TOPSTAGE` 0 hits, no `STAGE4`. **Narrowed further**: of the 16, only 6 carry any population, and only 3 (Si II, S II, Ca II) exist outside s11 |
| — | GEMM arm | **UNVERIFIABLE** — `R_bf_table` still not dumped; reconstruction only |

---

## 7. Effect on the Y3 pre-registration (P1–P8)

| # | registered prediction | status against the clean converged-end measurement |
|---|---|---|
| **P1** | Si II Γ at s20–s35 **UP ×1.6–3.5** pop-wtd | **band is too narrow.** Realized pop-wtd over s20–s35: min **1.882**, median **3.111**, max **4.063**; **44 % of the window sits above 3.5**. Direction UP is CONFIRMED; the upper bound is exceeded and the value was **still rising at the last iteration** (s30: 2.80 → 3.38 Boltzmann, 3.92 pop-wtd) |
| **P2** | Fe II at s20–s35 UP ×1.4–3.5 | **VACUOUS** — Fe exists only in shells 0–11 (`n_ion = 0`, `n_k = 0` beyond s11). Unmeasurable as written |
| **P3** | Mg II, Co II at s30 UP ×1.4–1.7 | **VACUOUS** — Mg absent from the model entirely; Co absent beyond s11 |
| **P4** | Ni II at s20 DOWN ×0.75–0.98 | **VACUOUS** — Ni absent beyond s11 |
| **P5** | s0, s8 NULL, \|ratio−1\| ≤ 0.035 for every ion with Γ > 1 s⁻¹ | **s0 CONFIRMED** (Fe II 1.021, Co II 1.006, Ni II 0.984). **s8 REFUTED** — Si II 0.954 (4.6 %, Γ = 2.2e2) and S II 0.942 (5.9 %, Γ = 1.0e1). As a hard-stop (HS1, > 5 %) S II at s8 would trip on arrival |
| **P6** | Ca II, Cr II, Mn II, Ti II, Sc II, O I ≤ 15 % anywhere | **REFUTED on the one real member.** Ca II reaches **1.256 (25.6 %)** at s20. The other five are absent from the model |
| **P7** | Si III / S III / Fe III / Fe IV / Co III matrix R_bf: exactly zero change | **CONFIRMED** (scoping, §0) |
| **P8** | R1 `parity_gamma_phot` identical between arms | **UNVERIFIABLE offline** — code-level only (`lumina_plasma.c:1731-1732` already integrates C2 unconditionally under parity); needs the B-arm run |

**One-line answer to the final question.** P1's *direction* survives, but its
*band* does not: the registered ×1.6–3.5 excludes 44 % of the s20–s35 window
(realized max ×4.06, and unconverged/rising), so the band needs widening; and
P2/P3/P4/P6 rest on ions with zero population in this model, so four of the eight
pre-registered predictions are unmeasurable as written.

### 7.1 Facts a revised registration would need (recorded, not prescribed)

* Only **Si II, S II, Ca II** (s4–s49) and **Fe II, Co II, Ni II** (s0–s11) can
  carry a Y3 signal in toy06.
* Si II s20–s35 realized band, at the last iteration and still rising:
  **1.88 – 4.06** (median 3.11).
* S II is the flux-dominant ion in most IME shells but its ratio is ~1 there
  (median 0.999); the shell-total flux ratio is consequently much milder than
  Si II's per-ion ratio (s20 0.964, s30 1.017, s45 1.299).
* s8 is **not** a NULL shell: Si II −4.6 %, S II −5.9 %.
* Ca II carries a **separate, non-TEPIN** ≤26 % channel (coarse bin 12 colour fit).

---

## 8. Limits of this certification — explicit

1. **Not converged.** The run completed its 12 scheduled iterations, but shells
   ≥ 20 still move 19–31 % per iteration in `J_raw`, and Si II's ratio rose
   monotonically through the final step. The headline magnitudes are a lower
   bound on a still-climbing quantity, not a converged value.
2. **GEMM arm reconstructed, not observed** (§4). `R_bf_table` was never dumped.
3. Level populations are the **final-state** dump; they are paired with the
   final-iteration field. No per-iteration population history exists, so §3.3's
   iteration trend stays on the Boltzmann weighting.
4. Both dumps carry `%.6e` precision; the roundtrip (5.6e-07) and the `bfr`
   identity (≤ 0.27 %) close far inside it. **Print precision is not the limiting
   factor — iteration non-convergence is.**
5. Everything in §3 remains **conditional on `LUMINA_C1_SUPERBIN_TEPIN=1`**
   (PRELIM C7). §3.4 now shows the Si II effect *is* that gate, quantitatively.
6. All composition statements are specific to
   `data/tardis_reference_toy06_19p48d_sivcaiv`. A model with the full IGE/IME
   mix would re-activate P2–P4, on numbers not measured here.
7. P8 cannot be settled without the B arm.

---

## 9. Reproduction

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/y2_c2bfr_certification
export Y2_RUN=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity46
export Y2_OUT=$PWD/clean
mkdir -p "$Y2_OUT"
for s in 1 2 3 4 5 6 7 8 9 10; do python step${s}_*.py > "$Y2_OUT/step${s}.out" 2>&1; done
```

Steps 1→4 must run in order (they write the `_cache_*.npy` arrays the rest read).
Unsetting `Y2_RUN` / `Y2_OUT` reproduces `PRELIM_REPORT.md` verbatim from the
killed-partial dump.

Determinism check quoted in §0:

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs
head -550001 coevolve_consume_parity46/lumina_c2_bfr_dump.csv | md5sum
md5sum coevolve_consume_parity46_killed_partial/lumina_c2_bfr_dump.csv
diff <(head -550001 coevolve_consume_parity46/lumina_c2_bfr_dump.csv) \
     coevolve_consume_parity46_killed_partial/lumina_c2_bfr_dump.csv
```

New in this report vs PRELIM: `step9_popweighted.py` (realized population
weighting), `step10_composition_scope.py` (composition scoping + P5 check), and
the `Y2_RUN` / `Y2_OUT` overrides in `y2_common.py`.
New CSVs: `clean/y2_gamma_popweighted.csv`, `clean/y2_composition_scope.csv`,
`clean/y2_popratio_allshell.csv`, plus clean re-runs of all 12 PRELIM CSVs.
