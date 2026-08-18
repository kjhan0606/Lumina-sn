# Trapping / energy-density audit — Lumina deep shell vs CMFGEN toy06 @19.48d

Offline analysis, 2026-07-19. Read-only on logs/ and /gpfs. No source edits, no commit.
Question: is the deep-shell FUV deficit (band-mean mc_J −1.54 dex at s0) caused by too-little
optical trapping (deep radiation energy density ~8-10× low because photons escape too easily),
or by spectral redistribution (energy present, wrong colors)?

## Pre-registered predictions (from the task)
- **U**: ratio u_Lumina/u_CMFGEN(s0) ≈ 0.10-0.15, rising to 0.5-1 at s8-s10. If ratio(s0)≈1, trapping REFUTED.
- **T**: CMFGEN tau(s0→out) several-to-10× Lumina's.
- **C**: the ARTIS super-cutoff (K=100) removes lines from the transport forest.

## Verdict summary
| Prediction | Result | Verdict |
|---|---|---|
| U: ratio(s0) ≈ 0.10-0.15 | ratio(s0) = **0.576** (−0.24 dex); rises **above 1** at s2-s8 (peak 1.57 @ s4) | **REFUTED** |
| T: CMFGEN tau several-to-10× Lumina | Lumina tau ≥ CMFGEN at every shell (es ratio 1.18; Rosseland ratio 1.43 @ s0) | **REFUTED (reversed)** |
| C: lumping truncates transport forest | Full 2,565,342-line forest loaded & traversed; lumping only remaps NLTE super-levels | **REFUTED** |

**Trapping hypothesis is REFUTED.** The deep FUV deficit is spectral redistribution / color, not an
energy-density shortfall from insufficient optical depth. Lumina is, if anything, *more* optically
thick than CMFGEN in the deep shell.

---

## CALIBRATION (yardstick discipline — all four anchors reproduced before extending)
| Anchor | Target | Reproduced | Convention |
|---|---|---|---|
| Lumina FUV band-mean mc_J, s0 | 5.81e-6 | **5.809e-6** | arithmetic mean over 64 bins in 918-1290Å |
| Lumina FUV band-mean mc_J, s8 | 6.66e-6 | **6.656e-6** | arithmetic mean |
| CMFGEN band-geo J, s0 (v=4264) | 2.02e-4 | **2.023e-4** | geometric mean (extract_jnu.py:99) |
| CMFGEN band-geo J, s8 (v=10088) | 7.73e-7 | **7.729e-7** | geometric mean |

Units: EDDFACTOR stores J_ν directly in **erg cm⁻² s⁻¹ Hz⁻¹ sr⁻¹** (CGS; comp_j_blank.f, per
extract_jnu.py:2-8). Sanity: innermost-depth ∫J dν = 2.008e3 vs a·T⁴ = 2.125e3 (ratio 0.945,
nearly thermalized) → units confirmed. Lumina mc_J is the same convention (the −1.54 dex FUV
framing is only meaningful if it is; the FUV band anchor 5.81e-6 vs 2.02e-4 IS that cross-calibration).
Energy density u = (4π/c)∫J dν → erg cm⁻³.

Shell↔velocity map: Lumina shell mid-v = 4264 + 728·i (geometry.csv, 728 km/s spacing), so
s0=4264 … s8=10088 … s10=11544. NOTE: extract_jnu.py's internal s0..s8 labels are mis-indexed
(it skips shells); its TARGET_V velocities are correct and were used directly.

---

## AUDIT U — energy density profile u(s) = (4π/c)∫J_ν dν  [erg cm⁻³]
CMFGEN: EDDFACTOR full range ν = 3.5e12 … 1.0e18 Hz, integrated per depth then log-interpolated in
velocity. Lumina: mc_J over its 1000-bin grid (ν 1.5e14 … 3.0e16 Hz), trapezoid in ν.
Overlap caveat: **99.9% of CMFGEN's u lies inside the Lumina ν-overlap** (u_below 1.5e14 ≈ 0.36 erg/cm³
at s0, u_above 3e16 ≈ 1e-26) — the restriction is immaterial; full-range and overlap agree to <0.1%.

| shell | v | u_CMFGEN | u_Lumina(Brun) | ratio | dex | tetab ratio | tincol ratio |
|---|---|---|---|---|---|---|---|
| s0 | 4264 | 694.8 | 400.2 | **0.576** | −0.24 | 1.104 | 0.662 |
| s1 | 4992 | 522.6 | 417.3 | 0.799 | −0.10 | 1.548 | 0.790 |
| s2 | 5720 | 394.4 | 392.5 | 0.995 | −0.00 | 1.628 | 0.932 |
| s3 | 6448 | 288.1 | 358.9 | 1.246 | +0.10 | 1.682 | 1.134 |
| s4 | 7176 | 196.5 | 308.6 | **1.570** | +0.20 | 1.790 | 1.432 |
| s5 | 7904 | 134.5 | 192.1 | 1.428 | +0.15 | 1.329 | 1.482 |
| s6 | 8632 | 94.7 | 118.3 | 1.250 | +0.10 | 1.152 | 1.401 |
| s7 | 9360 | 65.4 | 80.6 | 1.232 | +0.09 | 1.028 | 1.196 |
| s8 | 10088 | 45.2 | 53.2 | 1.178 | +0.07 | 0.904 | 1.114 |
| s9 | 10816 | 36.8 | 35.1 | 0.955 | −0.02 | 0.746 | 0.898 |
| s10| 11544 | 40.3 | 19.4 | 0.481 | −0.32 | 0.365 | 0.457 |

**Prediction U REFUTED.** The bolometric deep energy density at s0 is only **1.74× low (−0.24 dex)**,
not the predicted 8-10× (0.10-0.15). The ratio does not rise monotonically outward — it *crosses
above 1* in the mid shells (Lumina holds MORE energy than CMFGEN at s2-s8), then falls at the outer
edge. This is the signature of **spectral redistribution**, not trapping: the FUV band is −1.54 dex
but bolometric is −0.24 dex, so **~1.30 dex of the FUV deficit is energy sitting at the wrong (redder)
frequencies**, ≤0.24 dex is any true energy shortfall. (Consistent with the campaign color-decomposition:
color 1.25 + dilution 0.53 − occupancy 0.23 = 1.55.) The color probes behave as color probes:
tetab (T_e pin) *raises* u past CMFGEN (s0 ratio 1.10); tincol partially (0.66).

---

## AUDIT T — optical depth / trapping profile (outward optical depth from shell to surface)
CMFGEN from MEANOPAC (Tau accumulates surface→inward, so Tau at v = outward depth from v). Lumina es
from n_e·σ_T·dr (plasma_state + geometry); Lumina line from the run's own level pops
(lumina_levelpop.csv) via the exact Sobolev formula (lumina_plasma.c:10950, SOBOLEV_COEFF=2.6540281e-2).
CMFGEN inner boundary v=1025 km/s, total Tau(Ross)=20.5, Tau(es)=5.26 (extends far below Lumina's
s0=3900 boundary — see note). n_e agrees between codes (ratio ~1.0 at every shell).

| shell | v | CMFGEN TauRoss | CMFGEN Tau_es | Lumina tau_es | Lumina tau_Ross | Lumina tau_FUV | es ratio L/C | Ross ratio L/C |
|---|---|---|---|---|---|---|---|---|
| s0 | 4264 | 4.085 | 1.523 | 1.800 | **5.83** | **69.8** | 1.18 | 1.43 |
| s1 | 4992 | 2.892 | 1.154 | 1.439 | 4.98 | 59.6 | 1.25 | 1.72 |
| s2 | 5720 | 2.041 | 0.874 | 1.144 | 4.35 | 51.2 | 1.31 | 2.13 |
| s3 | 6448 | 1.414 | 0.656 | 0.911 | 3.88 | 44.0 | 1.39 | 2.74 |
| s4 | 7176 | 0.969 | 0.497 | 0.728 | 3.48 | 37.5 | 1.46 | 3.59 |
| s5 | 7904 | 0.677 | 0.386 | 0.582 | 3.09 | 30.6 | 1.51 | 4.57 |
| s6 | 8632 | 0.475 | 0.297 | 0.464 | 2.00 | 17.7 | 1.56 | 4.21 |

es floor (shared, unambiguous): Lumina **≥** CMFGEN everywhere (ratio 1.18-1.56). Rosseland-mean total
(T=13120K): Lumina 5.83 vs CMFGEN 4.09 at s0 — Lumina MORE opaque. Line blanketing enhances the
Rosseland mean over the es floor by 2.4× (Lumina) vs 2.7× (CMFGEN) — comparable. Lower-level pop hit
rate 93.9%; the missing 6.1% (line-list levels beyond the levelpop dump, e.g. Cr/Ti/Mn >1000) would
only ADD opacity, so Lumina's true tau is a lower bound → refutation holds a fortiori.

**Prediction T REFUTED (reversed).** CMFGEN is NOT several-to-10× more opaque than Lumina; Lumina is
1.2-1.4× MORE optically thick at s0 by every measure. **The FUV band in Lumina is tau ≈ 70 outward
from s0** — FUV photons are heavily trapped/reprocessed, the opposite of "escape too easily." With
tau_FUV≈70 the deep FUV field is locked to the local line source function S_l ≈ B_ν(T_e=13120K),
which on the Wien tail (ν~2.8e15) is intrinsically faint → this is precisely the color mechanism.

---

## CENSUS C — is the line forest truncated in TRANSPORT?
**No.** The transport line list is loaded whole: 2,565,342 lines (lumina_atomic.c:392-393, reads
line_list.csv column "nu"). tau_sobolev is computed for **every** line — nlte_update_tau_sobolev
loops `for line in 0..n_lines` (lumina_plasma.c:10905) and assigns each a tau from full-level NLTE
populations (10940-10953); un-mapped lines keep their nebular tau (10907), none are dropped.

The ARTIS super-cutoff (lumina_atomic.c:713-734) only rewrites `atom->level_super[l] = min(level_num,K)`
— the super-level index consumed by the NLTE SE solve dimensionality. It touches neither the line list
nor n_lines. The B-run had it active (stdout.log:18 SUPER_LEVELS=1, :84 CUTOFF=100, :122 "21009 levels
lumped"). Effect on the forest: **zero lines removed**; effect on physics: **97.9% of transport lines
(122,345/124,799 = 98.0% in the FUV band) draw their lower-level population from a Boltzmann(T_e)
redistribution within a super-level rather than an explicit NLTE solve.**

Top lumped-line ions: Co III 678k/679k (100%), Co II 592k (100%), Fe II 529k (100%), Fe III 135k (99%),
Cr III, Mn III, Ni III/II/I, Ti II. → a **population-accuracy** concern (and it is T_e-driven, tying
straight into the color finding), NOT a truncated-opacity concern. This KILLS the truncation suspect;
audit weight shifts to how those lumped lower-level populations (and the local T_e that sets both the
Boltzmann fractions and S_l) are computed.

---

## Suspect ranking update
1. **(NEW #1) Deep-shell color temperature / spectral source function.** The FUV deficit is spectral,
   not energetic: bolometric u only −0.24 dex while FUV −1.54 dex; FUV is tau≈70 thick so J_FUV≈B(T_e).
   The binding variable is the deep color temp (T_e/T_rad ~13120/10470 vs CMFGEN ~18760). Directly
   confirms the campaign F3-T finding (DIFFUSE_INNER_BC color thermostat). **Now the top suspect.**
2. **(RAISED) Super-level Boltzmann redistribution of FUV lower-level populations at low T_e.** 98% of
   FUV lines get lower-level n_l from within_sl_frac·n_SL at T_e=13120K; since S_l is built from the
   same pops (lumina_plasma.c:10958-10962), a too-cool/wrong redistribution reddens the FUV source
   function. Not a truncation — an accuracy issue coupled to suspect #1.
3. **(DEMOTED — eliminated) Optical trapping / deep energy density too low.** Refuted: u(s0) 1.74× low
   not 8-10×; tau(s0) Lumina ≥ CMFGEN by every measure. Photons are NOT escaping the deep shells too
   easily.
4. **(DEMOTED — eliminated) Transport line-forest truncation.** Refuted: full 2.565M-line forest traversed.

## Note on inner boundary (context, not a scored prediction)
CMFGEN has real material from v=1025 to 3900 (tau 20.5→4.1 Rosseland) that Lumina replaces with the
DIFFUSE_INNER_BC at s0. This changes what enters at s0's base (a 10020K blackbody surface vs CMFGEN's
18-23kK interior) but does NOT reduce the optical depth *above* s0 — Lumina's own tau(s0→out) is larger
than CMFGEN's. So the boundary is a *color/source* effect at the base, reinforcing suspect #1, not a
trapping (energy-density) effect.

## Artifacts (this directory)
- audit_U_energy_density.csv — u per shell, 3 Lumina runs + CMFGEN + overlap breakdown
- audit_T_optical_depth.csv — tau (Ross/es/FUV) both codes + n_e + ratios
- census_C_line_forest.csv — line counts, lumped fraction, top-10 ions
- u_cmfgen.csv, tau_es.csv, tau_lumina_line.csv — intermediate per-code tables
- audit_u_cmfgen.py, audit_u_lumina.py, audit_t_es.py, audit_t_expop.py — reproducible scripts

## Source files relied upon (paths + lines)
- /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/extract_jnu.py:2-8,99 (EDDFACTOR reader + band-geo mean)
- /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR, EDDFACTOR_INFO, RVTJ, MEANOPAC
- src/lumina_atomic.c:392-393 (line_list load), :713-734 (super-cutoff = level_super remap only)
- src/lumina_plasma.c:10905 (loop over ALL n_lines), :10940-10953 (tau per line), :10958-10970 (S_l)
- src/lumina.h:29 (SOBOLEV_COEFF), :272,360-364 (super-level fields)
- logs/coevolve_consume_a10_kx_gphall/{lumina_coevolve_field,lumina_plasma_state,lumina_levelpop}.csv, stdout.log:18,84,122
- data/tardis_reference_toy06_19p48d/geometry.csv; data/.../line_list.csv (2.565M lines)
