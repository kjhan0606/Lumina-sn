# kpr2 T_e warm-bias budget — Phase 1a

Offline reconstruction, 2026-07-20. Read-only on logs/ and data/. No source edits, no rerun, no job touched.
Run: `logs/coevolve_consume_a10_kx_kpr2/` (the DB-fb repaired state: RADEQ_TE=1 RADEQ_SIMUL=1 RADEQ_DB_FB=1,
GPH_ALLLEVEL=1 GPH_SIGMA_CMFGEN=1, NLTE_STAGE4=1, MC_COEVOLVE consume, PHOTOION_ALPHA=1.0, FB_RATE=1).
CMFGEN toy06 @19.48d benchmark. Every claim = number or file:line.

---
## Measured warm bias (lumina_plasma_state.csv vs data/cmfgen_te_table_toy06_19p48d.csv)

| shell | v (km/s) | kpr2 T_e | CMFGEN T_e | bias |
|---|---|---|---|---|
| s0 | 4264 | 20545 | 18760 | **+1785** |
| s2 | 5720 | 20362 | 16351 | +4011 |
| s4 | 7176 | 17447 | 13657 | **+3790** |
| s6 | 8632 | 17384 | 11929 | +5455 |
| s8 | 10088 | 16784 | 10383 | **+6402** |

The kpr2 T_e profile is nearly **flat** (16.8–20.5 kK) while CMFGEN cools steeply outward (18.8→10.4 kK). The bias
therefore grows with depth-out — the photosphere gas **cannot cool**. NOTE (correction to the tasking): kpr2 s2 = 20362
(not 23334; 23334 was the *pre-DBFB* kpr runaway value, `kpr_runaway_trace/TRACE_LEDGER.txt`). DBFB already killed the
runaway (s4 65273→17447).

## Headline
**kpr2 is a GENUINE converged root** — `[SIMUL] done: pins hi=0 lo=0 of 50 shells` on **every** iteration
(stdout.log:254,3620,…,28692), still cooling ~150 K/iter at iter 11 (T_e[0] 21741→20545 over iters 6–11,
stdout.log:19735–28771). So the gphall "+3400 K lag/HOLD lever" (radeq_ledger_audit) **does NOT apply here** — the warm
bias is a **real balance-term / field error**, not a convergence artifact.

**The bias is FIELD-driven, not a DBFB defect.** This is the same verdict class the radeq_ledger_audit reached for gphall
(same balance formulas fed CMFGEN's own J reproduce CMFGEN's deep T_e to 2.6%). Candidate 1 (the DBFB Wien partner) is
**negligible**; the bias is carried by the two field consumers that disagree with CMFGEN, both growing toward the
photosphere and both compounded by over-ionization.

---
## Candidate verdicts (each reconstructed with kpr2's own field + committed state at s0/s4/s8)

### C1 — DBFB Wien-limit partner (no stimulated/induced term): **REJECTED (≈ 0)**
The DBFB bf cooling uses `B_ν^Wien = (2hν³/c²)e^{−hν/kT}` (plasma.c:5099-5108, self-check net(J=B)/H=2.5e-16 at
stdout.log:247-250). Reconstructing the SAME bf integral with the full Planck partner `B_ν^Planck = (2hν³/c²)/(e^{hν/kT}−1)`
(the induced/stimulated term evaluated at LTE) over the real σ_bf grid + level populations:

| shell | C_fb(Wien) | C_fb(Planck) | Wien defect | as % of C_fb | root shift |
|---|---|---|---|---|---|
| s0 | 2.33e-3 | 2.37e-3 | 3.97e-5 | **1.70 %** | **−6…−18 K** |
| s4 | 3.41e-3 | 3.48e-3 | 6.40e-5 | 1.88 % | −27 K |
| s8 | 3.01e-3 | 3.02e-3 | 1.40e-5 | **0.47 %** | −6 K |

**Reason it is ~0:** the σ_bf-weighted bf emission of IGE ions is **EUV-dominated** (recombination edges Fe/Co/Ni III→IV
at χ = 30–55 eV ⇒ hν/kT ≈ 15–30 at these T_e), where Wien ≡ Planck to 1e-7. The tasking premise "hν/kT ~ 2–10" does NOT
hold for the actual emission — that regime would need χ_l ≲ 3 kT ⇒ near-continuum excited levels whose Boltzmann weight
carries the same `e^{−χ/kT}` suppression as the ground edge. The Wien approximation is **correct here**.
**→ Do NOT build LUMINA_RADEQ_DB_FB=2; it moves the root <30 K and is not the disease.**

### C2 — split/line-ledger field (cs_J pump vs mc_J heating): **CONFIRMED, the dominant lever**
The split (splitfield_audit) is real: the line pump `simul_line_term` reads deterministic **cs_J** (plasma.c:5054-5074,
Jb=`nlte_get_J_at_nu`), while Gph photoion + bf-heating `Hex` read **mc_J** (α=1.0, plasma.c:5748-5768). Λ_line at kpr2
pops/T_e, swapping only the pump field (coolant_burnout.py):

| shell | Λ(cs_J) | Λ(mc_J) | Λ(CMFGEN-J) | Λ(thermal B) | cs_J warm arm vs CMFGEN-J |
|---|---|---|---|---|---|
| s0 | +5.30e-4 | +2.37e-3 | +1.44e-3 | +2.84e-4 | Λ(cs)−Λ(CMFj) = **−9.09e-4** |
| s4 | −1.43e-4 (**heating**) | +7.55e-4 | +5.98e-4 | +5.69e-5 | **−7.41e-4** |
| s8 | −1.08e-3 (**heating**) | +5.71e-4 | +5.59e-4 | +8.4e-6 | **−1.64e-3** |

The super-thermal **cs_J pump flips Λ_line to net HEATING at s4/s8** (positive Λ = cooling). Against CMFGEN's own field the
pump would COOL (+6e-4). This under-cool/pump-heat warm arm grows −9e-4→−1.6e-3 with depth. (mc_J would over-cool vs
CMFGEN; neither run field matches CMFGEN — see fix.)

### C3 — Λ_ETLA coolant burnout at the over-ionized photosphere: **CONFIRMED, entangled with C2**
kpr2 is massively over-ionized vs CMFGEN, growing outward (lumina_ion_pops.csv vs standart CMFGEN ionfrac):

| shell | Fe f(IV) kpr2 | Fe f(IV) CMFGEN | Co f(IV) kpr2/CMFGEN | Ni f(IV) kpr2/CMFGEN |
|---|---|---|---|---|
| s0 | 0.983 | 0.982 | 0.999 / 0.993 | ~ / 0.978 |
| s4 | 0.977 | **0.714** | 0.984 / 0.522 | / 0.312 |
| s8 | **0.983** | **0.022** | / 0.099 | / 0.026 |

At s8 the entire Fe/Ni III coolant is stripped to IV (f(FeIV) 0.983 vs CMFGEN 0.022). This burns the III resonance/forbidden
coolant that CMFGEN uses to radiate the deposition away → the gas cannot reach the deposition rate at a low T → warm.
The over-ionization is **caused by** the bright EUV mc_J Gph (all-level, G_all/G_gnd=40.7 at s0, stdout.log:253) + the
lagged photoion — i.e. C3 is the ionization-side face of the same wrong field as C2. At s0, where ionization already
matches CMFGEN, the coolant-burnout ΔΛ is ~0 (−5e-6); the bias there is smallest (+1785 K), consistent.

### C4 — C_ff, C_ad, adiabatic, deposition: **negligible / unchanged**
C_ff = 5.7e-6 (s0) → 2.1e-7 (s8), ≤0.4 % of H_dep; C_ad ≤ 4e-8 (~0); H_dep injected verbatim from CMFGEN
(data/…/deposition_cmfgen.csv, stdout.log:224) so it is identical between the codes and cannot be the difference.

---
## Budget table (calibrated field-swap root shifts)

`C_fb(Wien)` absolute is uncertain by ~2× (the emit_bf pop-weight is stage-split: NLTE for III combs, Boltzmann@T_e for IV
combs, ladder-coupled; calibrated_budget.py absorbs it into a per-shell κ≈0.2–0.6 that pins the baseline root to the
committed T_e, so the field-swap shifts below are the trustworthy quantity — the ~2× C_fb uncertainty cancels in the
relative solve).

Each column = the T_e-root shift from swapping ONE consumer's field/pops to the CMFGEN-consistent value, holding the
others at their kpr2 value (calibrated_budget.py; κ pins root_base ≡ committed T_e).

| shell | bias | **pump arm** cs_J→CMFGEN | coolant/ioniz (pop→CMFGEN) | bf-heat arm mc_J→CMFGEN | C1 Wien→Planck | total field→CMFGEN | closed |
|---|---|---|---|---|---|---|---|
| s0 | +1785 | **−664** (37%) | ~+33 (≈0) | +83 | −15 | −548 (root 19997) | 31% |
| s4 | +3789 | **−1563** (41%) | −928 | +198 | −21 | −2293 (root 15153) | 61% |
| s8 | +6402 | **−3863** (60%) | −1380 | −89 | −5 | −5332 (root 11452) | 83% |

(coolant/ioniz column = total − pump − bf-heat, isolating the CMFGEN-ionization pop swap; it is negative = restoring the
burned-out III coolant cools, growing −0→−1380 with depth.) **The super-thermal cs_J line-pump is the single dominant
lever and it grows monotonically with depth (37→60 % of the bias), exactly tracking the flat-vs-cooling divergence.**

**Closure caveat (honest):** the κ-calibration anchors C_fb to the *warm* committed root, and the bf ionization ladder is
frozen at kpr2 pops, so this field-swap **under-closes** (31 % at s0 where the field is already near-CMFGEN; 83 % at s8).
The independent, un-calibrated cross-check is authoritative for the closure: radeq_ledger_audit solved the *same* balance
formulas fed CMFGEN's J from scratch → **18277 K vs truth 18760 K (2.6 %)**. So the field is ~100 % of the bias to within
the ~480 K formula residual; the table's role is to **rank the arms** (pump ≫ coolant/ioniz ≫ bf-heat ≫ C1≈0), which it
does robustly.

---
## Fix spec

**Do NOT touch the DBFB Wien partner (C1 rejected).** The disease is the two field consumers disagreeing with each other
AND with CMFGEN — the coevolve split-field. Env-gated options (analysis-only; implementer chooses):

1. **Line-pump field gate (NEW) — the dominant lever.** Route `simul_line_term`'s `Jb` through the same field the Gph
   loop uses instead of hard-wiring `nlte_get_J_at_nu`=cs_J (plasma.c:5054-5074 vs the Gph α-blend 5748-5768). Gate
   `LUMINA_RADEQ_PUMP_FIELD ∈ {cs, mc, blend, jtable}`. *Risk:* mc_J over-cools (−: root undershoots CMFGEN); cs_J is the
   current warm state; a per-band blend needs justification (NO-OVERFITTING).
2. **Gph/Hex field gate — already exists:** `LUMINA_GPH_JTABLE=<cmfgen J bin>` (#33 GRADIENT-TRANSPLANT, plasma.c:5346-5415)
   feeds CMFGEN's EUV J into the photoion+bf integral, curing the over-ionization/coolant-burnout arm. Test it with the
   pump gate above set to `jtable` to unify BOTH consumers on one field.
3. **Root cause (campaign F3):** neither cs_J nor mc_J matches CMFGEN because the deep MC emission is reddened / EUV-shaped
   (the DIFFUSE_INNER_BC / Co IV fluorescence color machinery). The orthodox end-state is a *single* self-consistent J
   (per CMFGEN), which requires fixing the emission color — the standing campaign target.

## Pre-registered predictions for kpr3
- If **both** field consumers are unified onto the CMFGEN-consistent J (options 1+2 → `jtable`), the budget predicts the
  root lands at CMFGEN: **T_e(s0) 18–19 kK, s4 13–15 kK, s8 10–11 kK**, and Fe f(IV) drops to CMFGEN (s4≈0.71, s8≈0.02).
- If **only** the pump is unified (option 1, Gph left on mc_J): over-ionization persists ⇒ residual warm at the
  photosphere (s8 stays ≳13 kK); pump arm alone closes s0 but not s8.
- Switching DBFB to full-Planck (DB_FB=2) alone: **no measurable change** (<30 K at every shell) — falsification test for C1.

## Artifacts (this directory)
- `VERDICT.md` — this file
- `recon_terms.py` — per-shell simul_r1 term reconstruction (H_photo, C_fb Wien/Planck, Λ_line) from σ_bf + NLTE n_k
- `coolant_burnout.py` — Λ_line field-swap + CMFGEN-ionization-pop recompute (C2/C3)
- `calibrated_budget.py` — κ-calibrated field-swap coupled-root shifts (the budget table)
- `budget_shells.csv` — machine-readable budget

## Source / data relied upon
- `src/lumina_plasma.c`: 5054-5074 (`simul_line_term`, Jb=cs_J), 5076-5142 (`simul_r1` balance),
  5087-5109 (DBFB C_fb Wien), 5739-5746/5797-5803/5859-5865 (emit_bf build), 5748-5768/5804-5824/5867-5891 (Gph/Hex mc_J
  α-blend), 5463-5516 (DBFB self-check), 5346-5461 (GPH_JTABLE / TE_TABLE loaders)
- `logs/coevolve_consume_a10_kx_kpr2/`: lumina_plasma_state.csv, lumina_coevolve_field.csv (cs_J,mc_J), lumina_ion_pops.csv,
  lumina_levelpop.csv (NLTE n_k), stdout.log (:254 pins=0, :247-253 DBFB/GPH banners, :19735-28771 iter T_e)
- `data/tardis_reference_toy06_19p48d/`: cmfgen_sigma_bf.bin (25620 lev × 1000 bins), levels.csv, line_list.csv (2.565 M),
  ionization_energies.csv, deposition_cmfgen.csv; `data/cmfgen_jtable_toy06_19p48d.bin`, `data/cmfgen_te_table_toy06_19p48d.csv`,
  `data/standart_data1/toy06/ionfrac_{fe,co,ni}_toy06_cmfgen.txt`
- prior audits: `../radeq_ledger_audit/VERDICT.md` (CMFGEN-field→18277 K), `../splitfield_audit/VERDICT.md` (7–77× band split),
  `../kpr_runaway_trace/TRACE_LEDGER.txt` (pre-DBFB runaway root cause)
