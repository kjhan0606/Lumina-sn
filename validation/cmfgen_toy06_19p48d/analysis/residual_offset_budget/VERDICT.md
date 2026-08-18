# kpr4 residual T_e warm-offset budget + f(FeIV) Wien-shadow verdict

Offline reconstruction 2026-07-20. Read-only on `logs/`, no rerun/edit/GPU.
Run: `logs/coevolve_consume_a10_kx_kpr4/` (integrated repair: chi-weighted B3 SRC=2 + PUMPF
alpha-blend a=1 + DBFB Wien + stage4-r2 + Fork B). Benchmark CMFGEN toy06 @19.48d.
Every claim = number or file:line. Grids: field CSV / sigma_bf / CMFGEN jtable all share the
1000-bin log-nu grid (numin 1.5e14, numax 3e16) — verified identical, so all field swaps are bin-exact.

---
## HEADLINE (two independent verdicts)

1. **Wien-shadow hypothesis for f(FeIV) s8 = REFUTED as the driver of the 45x f-ratio.**
   Swapping kpr4's field for CMFGEN's changes Gamma(FeIII→IV) by only **0.6x** (s8: 34.9→21.4),
   and predicted f(FeIV) stays **0.998** (measured 0.980). The 45x f-ratio (r_pop 689 vs 0.023)
   needs a ~3e4x swing in Gamma/alpha; the field supplies 0.6x. **What actually holds f(IV) up
   is an NLTE level-population FLOOR** (n_k = 1.342729e-3, hit by 1400/1500 Fe III levels at s8,
   b_k up to 3e8) that over-populates near-threshold Fe III excited levels, which then photoionize
   off the OPTICAL field regardless of the EUV.

2. **The T_e warm offset is UNDER-COOLING-driven, and the EUV excess IS guilty — but via bf
   photo-HEATING, not ionization.** kpr4's radiation bath is *cooler* than CMFGEN (u 0.56–0.77x),
   yet the mc field is EUV-excessive by 1e6x, and that excess pours into the bf photo-heating
   integral (H_photo(mc) = 236% of H_dep at s8 vs 72% for CMFGEN's field). Removing it cools
   −557…−656 K. Plus the 25.8% cs_J super-thermal pump fallback (depth-growing warm arm).

**Both verdicts share ONE root: the population floor → trace-Fe III over-ionized → big Fe IV
reservoir → (a) IV→III recombination emits the EUV excess → bf over-heating; (b) III coolant
burned out → under-cooling; and independently sustains f(FeIV) via the floored excited levels.**

---
## (a) f(FeIV) s8 — the double-lock, with Gamma decomposition

**Depth gate held:** W(s8)=0.0389 << 0.13 (LUMINA_STAGE4_GPH_WTHR=0.13, stdout:110), so stage4
SE-weighted Gph is OFF at s8 (fires only s0/s1/s2, W>0.13). Candidate-3 innocent at s8 by construction.

**Gamma(FeIII→IV) with kpr4's OWN NLTE pops (gamma_fieldswap.py):** field swap is nearly inert.

| shell | Te | f_act(IV) | G(kpr4 mcJ) | G(CMFGEN J) | G_cmf/G_kpr4 | f_pred under CMFGEN field |
|---|---|---|---|---|---|---|
| s0 | 20381 | 0.810 | 948 | 569 | 0.60 | 0.999 |
| s4 | 15356 | 0.973 | 15.9 | 10.9 | 0.68 | 0.993 |
| s8 | 12181 | 0.980 | 34.9 | 21.4 | **0.61** | **0.998** |

**Why the field is inert — band decomposition of G_nlte at s8** (bands overlap, do not sum):

| field | EUV(≤405A) | 300–450A | FUV(912–2000A) | opt+(>2000A) | total |
|---|---|---|---|---|---|
| kpr4 mcJ | 21.0 | 23.2 | 0.10 | 5.81 | 34.9 |
| CMFGEN J | 1.7e-6 | 4.0e-5 | 1.65 | **19.7** | 21.4 |

Under CMFGEN's field, **essentially all** of Gamma comes from `opt+` (>2000A) = 19.7 — i.e.
photoionization of **near-threshold excited Fe III levels** by the optical field, NOT ground-state
EUV. Those levels are the floored ones (E≈28.7 eV, threshold at optical λ). So killing the 1e6x
EUV excess barely dents Gamma.

**The 2×2 double-lock (gamma_te_sensitivity.py, self-consistent Boltzmann pops):**

| pops \ field | kpr4 mcJ | CMFGEN J |
|---|---|---|
| **kpr4 floored** (actual) | 0.999 | **0.999** ← floor lock |
| **Boltzmann** (no floor) | **0.999** ← field(EUV) lock | **0.0013** ✓ (CMFGEN meas 0.022) |

Only when BOTH the floor is removed AND the field is CMFGEN's does f collapse to 0.02. Either lock
alone keeps f≈0.999. This is why the single-variable field swap (task's test) fails: the floor is a
second, independent lock.

**The floor is the proximate cause (levelpop query):** at s8, 1400/1500 Fe III levels pinned at
n_k=1.342729e-3 (LTE≈1e-12 → b_k=2e7–3e8 near threshold). At **s0**, where Fe III is *not* a trace
(f(FeIII)=0.19), **0/1500** levels hit the floor and max near-threshold b_k=119 (physical). The floor
only bites for trace ions at the photosphere — tracking the defect's outward growth exactly.

→ **Verdict: the offset is NOT the last lever for f(FeIV). Fixing T_e alone leaves f(IV)≈0.98
(floor lock). The floor must be fixed.** The EUV excess is a genuine co-conspirator (field lock in
the thermal limit) but is downstream of the same over-ionization.

---
## (b) T_e offset budget (per candidate × shell × ΔT; sign: −=cools toward CMFGEN)

Offsets: s0 +1621, s2 +3097, s4 +1699, s6 +1522, s8 +1798 K.
Ledger `residual_ledger_kpr4.py`; slopes dr/dT from same. (NOTE: an inherited sign bug in the
first run flipped the Gph term; corrected below and in the re-run.)

| term (fix = cool by) | s0 | s2 | s4 | s6 | s8 | mechanism |
|---|---|---|---|---|---|---|
| **bf-heat EUV excess** (Gph mcJ→CMFGEN) | **−557** | **−777** | −5 | −269 | **−656** | mc EUV 1e6x → bf photo-heat; H_photo(mc)=79–236% Hdep vs 33–72% for CMFGEN |
| **cs-fallback pump** (26% cs→mc) | −88…−246 | −68…−633 | −3…−350 | −33…−699 | **−56…−1683** | 25.8% of pump lines use super-thermal cs_J (∫J_cs ~100x mc at depth); grows outward |
| coolant burnout (pops→CMFGEN III) | −2 | −126 | −146 | −38 | −137 | restore burned Fe/Co/Ni III line coolant (small: Si/S/Co carry cooling) |
| C1 DBFB Wien→Planck | −42 | −33 | −95 | −76 | −38 | Wien defect 3.3–5.2% of C_fb — confirms kpr2 verdict (not the disease) |
| C_ff, C_ad | ≈0 | ≈0 | ≈0 | ≈0 | ≈0 | C_ff ≤0.4% Hdep; C_ad ~1e-8 |
| _pump mcJ→CMFGEN (info)_ | +307 | +475 | +252 | +83 | +122 | mc pump already cools MORE than CMFGEN's would → NOT a warm culprit |

**Closure (offset_budget_summary.csv):** 32–52% (EUV-line cs bound) to 35–140% (all-line cs bound).
Like kpr2, the frozen-pop per-term swap **under-closes** (it cannot capture the coupled re-solve; the
independent radeq_ledger_audit established the full CMFGEN-field root = 18277 K vs 18760, 2.6%, i.e.
the field is ~100% of the bias). The table's job is to **rank**: bf-heat-EUV-excess (robust, dominant
inner) ≈ cs-fallback (depth-growing, uncertain magnitude) > coolant > C1 ≈ 0(C_ff,C_ad).

**Corroboration that cs-fallback is large at depth:** the reconstructed root (pure-mc pump) sits
BELOW committed T_e by 1366 K at s4 and 1373 K at s6 — i.e. the actual code is *warmer* than a
pure-mc-pump balance, exactly the sign/magnitude of adding the 26% cs super-thermal pump heating.

**Candidate-2 (u/energy residual) resolved:** bath u(s0)=534 vs 695 (0.77x) yet warm because the mc
spectrum is WRONG-SHAPED — deficient in optical/IR (→ cool bath, cooler T_urad by ~1100 K) but
excessive in EUV (→ bf over-heat). The heating term that is "over per unit bath" is **bf photo-
heating from the mc EUV excess** — the task's guessed self-consistent warm loop, CONFIRMED. u values
reproduced as u = 4π/c·∫J dν (u_gph.py inline): s0 533.7/694.1, s8 25.2/45.1.

**Candidate-3 (deep Gph SE-weighting, W>0.13 = s0/s1/s2):** INNOCENT. At s0 GPH-ALLLEVEL banner
G_all/G_gnd=40.7 (stdout:257), but kpr4 s0 f(FeIV)=0.810 is *below* CMFGEN's 0.982 — deep Gph is
UNDER-ionizing, not over. As the task anticipated.

---
## (c) Fix spec (env-gated, minimal) — TWO fixes, one dominant

**FIX-1 (dominant, unifying): the trace-ion NLTE population floor.**
`n_k = 1.342729e-3` pins 1400/1500 Fe III levels at s8 (b_k→3e8). This over-populates near-threshold
excited levels → sustains photoionization (f(IV) floor-lock) AND feeds IV→III recomb EUV → bf
over-heat. The floor should scale to the LTE population (or a much smaller relative floor), not sit at
a fixed absolute value that yields b_k≫1 for trace ions. Env-gate a floor policy, e.g.
`LUMINA_NLTE_POPFLOOR ∈ {abs(current), rel_lte, bk_cap}` — capping departure at b_k≤(say)10–100 for
excited levels, or flooring at n_k^LTE·b_max. This is the single lever that addresses f(FeIV) (removes
the floor lock), the EUV-emission source (removes bf over-heat), and the coolant burnout together.
*Locate:* the NLTE solve population floor / clamp in `src/lumina_plasma.c` (levelpop write path) — the
identical n_k across 1400 levels is a hard clamp, not a solved value.

**FIX-2 (secondary, depth-growing): the cs-fallback pump field.**
25.8% of `simul_line_term` pump evaluations fall back to cs_J on zero-count mc bins
(PUMPF `blended=21503332 cs_fallback=7473805`, stdout:3626+). cs_J is ~100x super-thermal at depth
(∫J_cs s8 = 5.8e12 vs mc 6.0e10), injecting pump-heating (up to −1683 K if removed at s8). Route the
fallback to a THERMAL/jtable value instead of the super-thermal cs_J:
`LUMINA_PUMP_FALLBACK ∈ {cs (current), wien, jtable, floor}` — for empty mc bins use B_ν(T_e) or the
CMFGEN jtable J, not cs_J. Minimal, and it removes a term the current field cannot self-correct.

Do NOT build DB_FB=2 (C1 Wien→Planck moves the root <95 K everywhere; not the disease — confirms kpr2).

---
## (d) Pre-registered kpr5 predictions

Baseline to beat (kpr4): T_e s0/s2/s4/s6/s8 = 20381/19448/15356/13451/12181; offsets +1621…+1798;
f(FeIV) s8 = 0.980; field FUV/slope/u near-CMFGEN.

- **FIX-1 (floor) alone:** Fe III near-threshold b_k drops to O(1–100); f(FeIV) s8 **≤ 0.25** (target;
  DB predicts →CMFGEN 0.02 once the floored opt+ channel is gone AND the EUV-emission source falls
  with the Fe IV reservoir). III coolant restored + EUV bf-heat source removed ⇒ T_e cools ~0.4–0.9 kK
  inner, more outward. Predicted T_e s8 → 10.8–11.5 kK (offset ≤ +1.1 kK).
- **FIX-1 + FIX-2 (floor + pump fallback):** removes the depth-growing cs warm arm on top ⇒ predict
  **T_e all shells within ±1 kK of CMFGEN**; f(FeIV) s8 **≤ 0.25** (likely ≤0.10); field gains held
  (FUV s0 ~1.7e-4, slope ~+2.4 dex, u(s0)~530 — untouched, both fixes act on pops/pump not the FUV
  transport).
- **Falsifiers:** (i) if FIX-1 leaves f(FeIV) s8 > 0.5, the floor is not the lock (re-open field-lock
  / recombination). (ii) if FIX-2 alone (no floor fix) is tried, predict f(FeIV) s8 stays ≈0.98
  (floor lock) and only −56…−700 K of cooling at depth — pump fix cannot close the inner offset.
  (iii) DB_FB=2 alone: <95 K everywhere (C1 falsification).

---
## Artifacts (this dir)
- `VERDICT.md` — this file
- `gamma_fieldswap.py` — Gamma(FeIII/CoIII→IV) 3-field swap + band decomposition (kpr4 NLTE pops)
- `gamma_te_sensitivity.py` — 2×2 field×Te double-lock, Boltzmann pops, recomb-deficit check
- `residual_ledger_kpr4.py` — simul_r1 term reconstruction + root + field-swap dTe (sign-fixed)
- `coolant_burnout_kpr4.py` — Lambda_line kpr4-pops vs CMFGEN-pops (coolant restore dTe)
- `offset_budget_summary.csv`, `ledger_shells.csv` — machine-readable budgets

## Source / data
- `src/lumina_plasma.c`: simul_line_term (pump Jb), simul_r1 balance, DBFB C_fb Wien, GPH α-blend,
  GPH_JTABLE loaders (line refs per te_bias_budget/VERDICT.md §Source)
- `logs/coevolve_consume_a10_kx_kpr4/`: lumina_plasma_state.csv, lumina_coevolve_field.csv (cs_J,mc_J),
  lumina_ion_pops.csv, lumina_levelpop.csv (n_k,b_k), stdout.log (:110 WTHR, :257 GPH-ALLLEVEL,
  :3626 PUMPF fallback counts)
- `data/tardis_reference_toy06_19p48d/`: cmfgen_sigma_bf.bin, levels.csv, line_list.csv,
  ionization_energies.csv, deposition_cmfgen.csv; `data/cmfgen_jtable_toy06_19p48d.bin`(+.json);
  `data/standart_data1/toy06/ionfrac_{fe,co,ni}_toy06_cmfgen.txt`; `data/cmfgen_te_table_toy06_19p48d.csv`
- prior: `../te_bias_budget/VERDICT.md` (kpr2), `../radeq_ledger_audit/VERDICT.md` (CMFGEN-field=18277K)
