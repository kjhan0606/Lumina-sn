# Frequency-Resolved Radiation Field Transport — Design

**Date**: 2026-06-25
**Root cause (confirmed, 4× gold-independent)**: the binned-J continuum field (NB~1000)
has no frequency contrast, so at cold outer shells it collapses to grey-thermal
(J/B≈2.1 vs the correct line-resolved 4.4). The deep ~4400K photosphere color is
NOT carried outward → cold thermal line re-emission → emergent too-red (9200 vs
gold 6630Å). Opacity (Sobolev τ hand-validated ratio 1.000), plasma (T_e 0.98,
n_e dex 0.18), ionization (Saha 100% II correct), input densities (gold 0.999) are
ALL correct. The bug is purely in the **field representation / transport**.

**Confirmed engine**: `cmfgen_fine_jbar` (src/lumina_cmfgen.c:1373) already solves the
frequency-coupled field `fs.J` on a vdop-resolved mesh via the validated `cmf_solve_J`
kernel, and it demonstrably carries the correct color (run 169643: cold shell24
fine J̄/B=4.40 vs binned 2.13). It currently extracts only per-line J̄_l and discards
the rest of fs.J.

---

## Where the grey field leaks into color-setting outputs (architecture map)

| Path | reads | file:line | status |
|---|---|---|---|
| bb-rate R_lu | jbar_line_det (freq-resolved) else binned | plasma.c:1513–1522, 6954–6992 | ✅ wired (LINERES_JBAR+CONSUME), window-limited |
| **bf-rate Γ (photoion)** | **binned nlte->J_nu ONLY** | plasma.c:4730 | ❌ grey leak |
| FI emergent line source | line_source_S (NLTE S_l) | plasma.c:8910 | ✅ correct if pops correct |
| **FI emergent continuum** | **W·B(T_rad)** dilute-LTE | plasma.c:8894,8951 | ❌ not field-resolved |
| CMF comoving spectrum | cs->J (binned) | lumina_cmfgen.c:787 | ❌ grey (secondary output) |

S_l is computed from NLTE populations (plasma.c:8268). Populations are set by bb-rate
(✅) + bf-rate (❌ grey) + collisions. So S_l inherits grey color via the bf-rate and
via the window limit on the bb-rate.

---

## Goal

Make a frequency-resolved field the **field-of-record** for the color-setting outputs:
1. NLTE populations (bb + bf rates) → correct S_l.
2. The emergent spectrum (line source + continuum).

so the warm photosphere color is transported outward and too-red is fixed.

---

## Staged plan

### Stage 1 — PROOF (minimal wiring, direct emergent) — RECOMMENDED FIRST
Prove the frequency-resolved field yields the correct emergent color, with NO
population/plasma rewiring (plasma already matches gold).

- **1a**: Extend `cmfgen_fine_jbar` window to the full optical–NIR (3000–12000Å first;
  1000–25000Å later) via existing `LUMINA_CMF_FINE_LAMLO/LAMHI`. Persist the full
  per-shell fine field `fs.J` (new output buffer `opac->jfine` + the fine ν grid),
  not just per-line J̄_l.
- **1b**: Add a **frequency-resolved formal integral** to the observer that integrates
  the fine field's source (S_fine = (chi_abs·B + chi_es·J_fine + η_line)/chi_tot) along
  observer rays on the fine grid → `lumina_spectrum_freqres.csv`. This is a new emergent
  path parallel to the FI; reuses the fine ν grid and the existing impact-parameter
  geometry.
- **Gate**: does the freq-resolved emergent peak move from 9200→~6600Å (gold)? If yes,
  root cause + fix direction CONFIRMED at the emergent level. Cheap, decisive.

### Stage 2 — SELF-CONSISTENT populations
Wire the fine field into the rate solve so S_l and ionization are color-correct.

- **2a**: bf-rate — replace binned `nlte->J_nu` at plasma.c:4730 with the fine field
  interpolated to the bf cross-section grid (integrate 4π J_fine σ_bf/hν dν on the fine
  mesh). Gate `LUMINA_BF_RATE_FREQRES`.
- **2b**: bb-rate — already uses jbar_line_det; just extend producer window to full
  spectrum so ALL lines (not just the pump window) get the freq-resolved J̄.
- **2c**: iterate producer(fine field) → rates → populations → S_l → opacity →
  producer to convergence (lagged-S_l scheme already validated, gate 5d).
- **Gate**: S_l/B and the populations shift toward color-correct; T_e/n_e hold (must
  not regress the gold-matched plasma).

### Stage 3 — PRODUCTION
- Cost optimization: GPU port of the fine `cmf_solve_J` (currently CPU/OpenMP), or
  adaptive frequency grid (fine at line cores, coarse in continuum windows = opacity
  sampling), or reduced ppd/vdop where the continuum dominates.
- Make the freq-resolved field the default emergent + rate field; deprecate grey-J
  for color-setting consumers (keep binned-J only where its speed is needed and color
  is irrelevant).

---

## Cost (the main risk)

Fine grid = uniform log-ν at ppd points per Doppler width.
- vdop=10 km/s, ppd=12, 1000–25000Å: NF≈1.16M freq × 49 shells ≈ 57M cells ≈ 4 GB,
  cmf_solve_J ~minutes/call × ~8 outer iters.
- Mitigations for Stage 1: restrict 3000–12000Å (NF≈500k), ppd=6 (½), vdop=20 (½) →
  ~125k freq ≈ manageable on one node. Continuum color does NOT need 12 ppd; only line
  cores do, and slightly under-resolved cores are fine for the FIELD color (windows
  dominate transport).

---

## Open decisions (need input)

1. **Stage 1 scope**: direct freq-resolved emergent (proof, recommended) vs jump
   straight to Stage 2 self-consistent populations?
2. **Resolution for Stage 1**: 3000–12000Å @ ppd=6 vdop=20 (fast proof) vs full
   1000–25000Å @ ppd=12 (definitive but heavy)?
3. **CPU vs GPU**: accept CPU/OpenMP for Stages 1–2 (slower but no port), defer GPU to
   Stage 3?

## Decision (recommended)
Stage 1 first, 3000–12000Å @ ppd=6 vdop=20, CPU/OpenMP. One run proves the color fix
at the emergent; then commit to Stage 2 self-consistency.

---

## STAGE 1 RESULT — PASSED (2026-06-25, run 169651)

Implemented `cmfgen_fine_emergent` (src/lumina_cmfgen.c): static formal integral on the
fine grid, source S = S_fixed + (χ_es/χ_tot)·J_fine, reuses binned ray grid. Gate
`LUMINA_CMF_FINE_EMERGENT=1`, called in `cmfgen_fine_jbar` after `cmf_solve_J`. Writes
`lumina_spectrum_freqres.csv`. Harness: `FINE_EMERGENT` passthrough.

Run 169651 (LINERES_JBAR=1, 3000–12000Å, vdop=20 km/s, ppd=6, 124681 fine freqs):

| | GOLD | freq-resolved (NEW) | binned formal |
|---|---|---|---|
| SED peak | 6630Å | **6785Å** ✅ | 9200Å |
| centroid | 7850Å | 7441Å | 8673Å |
| green frac | 21% | 14% | 7% |
| NIR frac | 27% | 32% | 45% |

**The frequency-resolved field moves the peak 9200→6785Å ≈ gold, confirming binned-J
grey collapse as the root cause and frequency-resolved transport as the fix.**

Residual (Stage 2/3): slight blue excess (9% vs gold 2%), mild NIR excess — attributable
to the static no-Doppler approximation, incomplete convergence, or the 3000Å window edge.

→ **Stage 2 GO**: wire the fine field into bf-rate (plasma.c:4730) + extend bb-rate window,
make S_l and the production emergent color-correct, gate on T_e/n_e holding gold.

---

## STAGE 2 — self-consistent populations (stabilization-first)

**Tie-off finding (169651)**: the closed loop producer→jbar→R_lu→pops→S_l→η→producer
is UNSTABLE: shell0 Jbar/B = 0.58, 0.847(stable iter2-4), then 454(iter5), 659(iter7).
Cause: cold-shell super-thermal S_l (documented NLTE ill-conditioning artifact, rates are
DB-correct) is amplified by the producer's line-emissivity deposit (η = χ_line·S_l) and
fed back through the bb-rate. So Stage-2 is NOT mere wiring — feedback stabilization comes
first.

### Stage 2-pre — STABILIZE the S_l feedback (IMPLEMENTED 2026-06-25)
`LUMINA_CMF_FINE_SL_CLAMP=C` (cmfgen_fine_jbar): clamp the lagged S_l used in the line
emissivity deposit to `S_l ≤ C·B(T_e)`. Physical color-bearing S_l at cold shells is only
~few×B (fine J/B≈4.4 at shell24), so C=50 preserves the physical super-thermal field while
cutting the 454–659× numerical artifact. Diagnostic: prints `max S_l/B` and `clamped/total`
each producer call (gate FINE_DIAG). Harness: `FINE_SL_CLAMP`.

**Test 169688** (N_ITER=8, FINE_SL_CLAMP=50): gate = Jbar/B stays stable past iter 4 (no
454/659 blow-up) AND the freqres color holds (peak ~6800). If max_slb shows huge pre-clamp
values + run stays stable, both the diagnosis (super-thermal S_l is the trigger) and the
fix (clamp) are confirmed in one run.

### Stage 2a/2b/2c (after stabilization holds)
- 2a: bf-rate (plasma.c:4730) binned J → fine field.
- 2b: extend bb-rate producer window to full spectrum.
- 2c: iterate to convergence; gate T_e/n_e hold gold AND grn/NIR → 1.62 (currently 0.89,
  the lagged-S_l residual the self-consistent loop should close).

### STAGE 2 RESULT (2026-06-25 autonomous) — stabilization works, but 2c is BLOCKED
- **2-pre stabilization (LUMINA_CMF_FINE_SL_CLAMP)**: clamp dial reduces the closed-loop
  blow-up monotonically — shell0 Jbar/B at iter5: 454 (no clamp) → 7.1 (clamp50) → 2.2
  (clamp10). Plasma T_e/n_e do NOT regress through the oscillation (T_e 0.984, n_e dex
  0.16) — the instability is confined to the diagnostic fine-field path; production plasma
  is robust.
- **KEY BOUNDARY**: the emergent color tracks how much NLTE S_l enters: thermal S_l (=B)
  → peak 6785/grn-NIR 0.89; clamp10 (Jbar/B 2.2) → 8339/0.72; clamp50 (7.1) → 9630/0.58.
  **More cold-shell NLTE S_l = REDDER**, because the cold-shell NLTE S_l is super-thermal
  garbage (max S_l/B ~1e9, documented month-long conditioning artifact) and at cold shells
  B peaks in the NIR, so the garbage line emission is NIR-heavy → reddens.
- **CONCLUSION**: the BEST emergent color (6785 ≈ gold) comes from a THERMAL line source
  (S_l=B) + the frequency-resolved continuum field — i.e. Stage-1. Stage-2c (self-consistent
  NLTE line source) makes the color WORSE, not better, until the cold-shell NLTE S_l is
  fixed to be physical-warm. **grn/NIR → 1.62 is gated by the NLTE cold-shell conditioning
  problem (hard, user decision), NOT by the field representation.**
- **Production config**: `FINE_SL_CLAMP=1.0` forces a THERMAL line source (S_l≤B) → no
  super-thermal injection → no oscillation → stable convergence at peak ~6785 (the
  frequency-resolved continuum field carries the color). This is the achievable Stage-1/2
  deliverable. (Verified by dial: clamp=1.5 already stable but reddened to 8524 because it
  admits 1.5×B NIR-heavy cold-shell emission; clamp=10→2.2/8339; clamp=50→7/9630;
  no-clamp→454/blow-up. Run 169726 confirmed clamp=1.5 removes the oscillation, Jbar/B
  iter5-7 = 0.97/0.92/1.04. A converged clamp=1.0 run to reconfirm 6785 is the only
  follow-up.)
- **STAGE-2 BOUNDARY (user decision)**: grn/NIR → gold 1.62 (currently 0.89 with thermal
  lines) requires the cold-shell NLTE line source to be physical-warm, not the super-thermal
  garbage (max S_l/B ~1e9) it currently produces. That is the documented month-long NLTE
  conditioning problem — NOT the field representation. The dominant too-red (peak +2570Å)
  is fixed by the frequency-resolved field; the residual color is gated by NLTE.
