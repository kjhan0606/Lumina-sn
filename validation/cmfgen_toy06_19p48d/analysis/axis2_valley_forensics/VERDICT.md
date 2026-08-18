# Axis-2 forensics — photospheric FUV excess (Case A2) & unfilled valley (Case V)

Offline, read-only, 2026-07-19. No GPU, no sbatch, no kill, no source edit, no commit.
LIVE run `logs/coevolve_consume_a10_kx_bsrc/` **untouched**. Corpses used:
`coevolve_consume_a10_kx_{bsrc.n12 (Fork B 12-iter), gphall (B-run), ltherm (LTHERM)}`.
Methods reuse the validated packet-chaining kernel (`crime_reconstruction/part3_redist_kernel.py`)
and event format (`scripts/read_events.py`). Scripts + CSVs in this directory.

**Coverage (every run):** single iter=11, CAP128M hit (128,000,000 events).
etype 8 (bf-reemit) UNLOGGED → bf recomb continuum invisible; etype 7 (e-scatter) IS
logged (bsrc.n12/ltherm) and is <0.5% of FUV creation. Packet energies are relative;
calibrated dex comes from the `mc_J`/`cs_J` field CSVs and CMFGEN jnu4 `EDDFACTOR`.

**Ion labels are 0-based → spectroscopic = ion_field + 1** (confirmed: prior validated
`reddening_localization/taskB_top_ions.csv` has "Fe V" at ion=4, "Co IV" at ion=3;
`Fe` max ion field = 4 = Fe V). An earlier draft here mislabeled every ion one stage
low; **all labels below are corrected** (the top FUV creator is **S III**, not S II —
which *vindicates* the pre-registered "S III family" suspicion).

---

## CASE A2 — the channel that MANUFACTURES the photospheric FUV excess

### Divergence, quantified
`mc_J`(918–1290Å) at s8: **bsrc.n12 5.68e-6, gphall 6.66e-6 vs CMFGEN 7.73e-7**
(+0.87 / +0.94 dex). CMFGEN's s8 FUV is near-pure continuum/thermal; Lumina's is a
manufactured **line pseudo-continuum**. Downstream: photospheric `f(FeIV)` s8 = **0.461**
(gphall) / 0.516 (bsrc) vs CMFGEN **0.022** (`lumina_ion_pops.csv`).

### A2.1 — Emission ledger into the FUV (918–1290Å) at s7–9  (energy-weighted)
Creation is **100.0% line-emit**; kpkt-ff / kpkt-fb / boundary ≈ 0 (ff+fb < 1e-5 of
creation). e-scatter merely redistributes (0.5% of creation). Top channels (`a2_emission_ledger.csv`):

| channel (emitting ion) | bsrc.n12 s7–9 | gphall s7–9 | **s8 only (removal basis)** |
|---|---|---|---|
| **S III line** | **84.2%** | **85.0%** | **88.0 / 88.5%** |
| Si III line | 9.0% | 9.2% | 7.1 / 7.2% |
| Co III line | 5.0% | 3.7% | 3.5 / 2.9% |
| Fe III line | 1.6% | 1.7% | 1.2 / 1.1% |

**Feeder band 1290–2000Å (s7–9):** S III **65.0 / 60.0%**, Co III **27.7 / 23.4%**,
Si III 4.1 / 4.3%, Fe III 2.5 / 2.8% (gphall also Co IV 8.0%). The feeder that up-pumps
the FUV is the same S III + Co III pair.

### A2.2 — Redistribution kernel at s7–9 (who feeds the FUV exits)
Packet-chaining, 2.38M / 2.58M FUV-exit emissions paired (`a2_out.txt`):

| entry band of the absorption feeding each FUV photon | bsrc.n12 | gphall |
|---|---|---|
| 2100–4500 (red) | 19.7% | 18.6% |
| 1290–1490 (red) | 16.2% | 15.1% |
| 1490–1650 (red) | 8.2% | 10.1% |
| 1650–2100 (red) | 7.6% | 8.9% |
| 4500–20000 (red) | 5.6% | 4.5% |
| 918–1290 (in-band) | 14.5% | 16.1% |
| 450–918 (blue) | 28.2% | 26.7% |
| **UP-conversion (redder→FUV)** | **57.3%** | **57.2%** |
| DOWN-conversion (bluer→FUV) | 28.2% | 26.7% |

**Mechanism is inter-ion, not line scattering** (`a2_pairing_check.py`): 99.9% of FUV
emits are immediately preceded by a line-abs on the same packet, but the emitter equals
the absorber only **5.8%** of the time. When **S III** emits an FUV photon, the packet
was just absorbed by **Co III (73.0%)** or **Fe III (20.6%)** — across *all* bands
(28.7% from blue 450–918, 19.3% from red 2100–4500, …). So Lumina's macro-atom
deactivation **re-samples from the global line emissivity**, and at the photosphere that
sampling **concentrates ~88% of every FUV photon into S III lines**, regardless of which
ion/band donated the energy. This is the *same pathology* as the deep Co IV 1500Å pile
(`reddening_localization/VERDICT.md`), one band up: an emissivity-sampled pseudo-continuum.
**Old suspicion updated:** it is not an "S III em/abs≈7.5 self-pump" — S III is the
*emitter/funnel exit*; the energy donors are Co III/Fe III. Confirmed as S III family.

### A2.3 — Arithmetic removal → does it land on CMFGEN?
Scaling `mc_J`(s8) by (1 − creation share) of the removed channel(s):

| | bsrc.n12 | gphall |
|---|---|---|
| measured s8 FUV | 5.68e-6 (+0.87) | 6.66e-6 (+0.94) |
| **− top-1 (S III, 88%)** | **6.81e-7 (−0.05 dex)** | **7.66e-7 (−0.00 dex)** |
| − top-3 (S III+Si III+Co III) | 7.50e-8 (−1.01) | 9.43e-8 (−0.91) |

**Removing S III alone lands Lumina's s8 FUV on CMFGEN's 7.73e-7 to ±0.05 dex** — a
bullseye. Removing top-3 overshoots ~1 dex below. **S III is a single, cleanly-separable
channel; it is the entire photospheric FUV excess.**

### A2.4 — F4 channel-gate A/B (diagnostic, NO runs here)
**Gate:** `LUMINA_A2_SIII_FUV_THERM` — at shells ≥6, force the **S III (Z=16, ion=2)**
line emission with λ_emit < 2000Å to thermalize: draw its deactivation from the local
thermal pool / redward lines instead of re-emitting into the FUV (equivalently, clamp the
S III FUV line source function to B(T_e)). Strictly scoped by (Z=16, ion=2, band, shell≥6)
so s0–5 and every other ion are byte-identical → deep side is a hard control.
**Pre-registered gates (A=B-run, B=gate on):**
1. `mc_J`(s8, 918–1290) **5.68e-6 → ~7e-7** (predicted 6.8e-7; must fall ≥0.8 dex).
2. s0→s8 FUV slope **steepens toward CMFGEN +2.42** (outer half un-flattens).
3. `f(FeIV)`(s8) **0.46 → toward 0.022** (this is the causal test: the S III FUV
   pseudo-continuum is the over-ionization driver, or it is not).
4. **Deep control:** s0 FUV, the Co IV 1490–1650 pile, and the valley move < noise.
**Falsifier:** if s8 FUV does *not* drop when S III FUV is thermalized (Si III/Co III
refill the band), the channel is not cleanly separable and the arithmetic in A2.3 is
wrong. (Prediction: it will drop — A2.3 already removed it arithmetically to the target.)

---

## CASE V — the valley (1650–2100Å) that Fork B left empty at s0–2

### V.0 — Field state (`v_valley_field.csv`, `v_supp.py`)
Band-avg valley `mc_J` / `cs_J`, and vs **actual** CMFGEN jnu4 J (`cmfgen_band_J.csv`,
correct v-grid: v(s_n)=4264+728n; CMFGEN valley = 8.12e-4 / 6.04e-4 / 4.43e-4 at s0/s1/s2
= **4.76 / 3.54 / 2.59 × B(13120)**):

| run | s0 mc/cs | s0 mc/CMFGEN | s0 cs/CMFGEN | s0 cs/B13 | s0 mc/B(Te) |
|---|---|---|---|---|---|
| bsrc.n12 (Fork B) | 0.048 | **0.028 (−1.55dex)** | 0.585 | 2.78 | 0.074 |
| gphall (B-run) | 0.059 | 0.074 (−1.13) | **1.26–5.19** | **6.0–13.5** | 0.35 |
| ltherm (LTHERM) | 0.630 | 0.196 (−0.71) | 0.31 | 1.48 | 0.62 |

### V.1 — Who absorbs, and where the energy goes now (`v_valley_absorbers.csv`, `v_valley_kernel.csv`)
**Valley absorbers are 100% NLTE-IGE, doubly-ionized:** Co III **51–62%**, Fe III **26–47%**,
Ni III **0–19%** (bsrc: Co III 62.4 / Fe III 34.7; gphall: Co III 51 / Fe III 26 / Ni III 19).
Identical set in all three runs. Kernel row = where valley-absorbed energy exits:

| exit band | gphall (B-run) | bsrc.n12 (Fork B) | ltherm |
|---|---|---|---|
| 1490–1650 (Co IV pile) | **74.9%** | 14.2% | 4.4% |
| 1290–1490 | 14.1% | 17.6% | 4.5% |
| in-band 1650–2100 | 7.7% | 14.5% | 14.5% |
| 2100–4500 (redward) | 1.2% | **34.6%** | **49.5%** |
| 4500–20000 (redward) | 0.1% | 13.7% | 21.9% |
| **exit ion** | **Co IV 81.3%** | Co III 30% (+64% untab. lines) | thermalized (id<0) |

- **B-run:** valley energy is **UP-pumped 89% into the 1490–1650 Co IV pile** — the funnel.
- **Fork B:** the funnel is **dead** (pile mc/cs 7.74→0.24 at s0, `v_supp.py`), but the
  valley absorbers (Co III/Fe III) **still ship the energy out — now 48% REDWARD** to
  2100–4500+. Re-emission is 100% via line-emit (etype 2), i.e. resonant scattering, not
  thermal destruction. The valley stays empty because the absorbers were never
  thermalized; Fork B only redirected the exit from blue-pile to red.
- **LTHERM:** same 71% redward exit, but re-emission is thermalized (id<0), and the valley
  fills to **`mc_J`/B(T_e) ≈ 0.62** (near-thermal).

### V.2 — Why LTHERM filled it and Fork B did not; is `cs_J`=10.5×B CMFGEN-consistent?
**Confirmed hypothesis.** The valley absorbers are NLTE-IGE lines (Co III/Fe III/Ni III)
whose source functions **Fork B leaves resonant** — they scatter valley energy out
(→red). LTHERM thermalized *those same lines* (S_l→B, all-line s0–2), so they emit into
the valley → `mc_J`/B(T_e) 0.045→0.62. Per-line-thermalization "cost": LTHERM's valley
absorption *rises* (E 2.33→3.70) as thermal emission feeds more valley photons to
re-absorb, and 71% still leaks redward — thermalization adds a floor, it does not stop the
scatter-out.

**But `cs_J`=10.5×B is NOT CMFGEN-consistent — it is the yardstick that is wrong.** The
"10.5×B" is the **B-run (gphall) deterministic solve** (`cs`/B13 = 6.0–13.5). The **actual
CMFGEN jnu4 valley is only 2.6–4.8×B** — so the B-run `cs_J` **overshoots real CMFGEN by
2–3×** (`cs`/CMFGEN = 1.26–5.19). This is the campaign's known super-thermal-S_l / binned-J
inflation of the deterministic solve. Fork B's own `cs_J` dropped to 2.5–2.8×B, which is
actually **closer to real CMFGEN** (`cs`/CMFGEN 0.58–0.96) — yet its `mc_J` stayed empty.
So the deterministic solve *also* disagrees with CMFGEN (too high in B-run), and the MC
field is a further −1.3 to −1.55 dex below **actual** CMFGEN.

### V.3 — Verdict
**Not (a), partly (c), mostly (b′):**
- **NOT (a) same-family Fork-B-scope defect.** Broadening Fork B to thermalize the valley
  lines *is* LTHERM, and it only reaches thermal B — still **−0.71 dex below CMFGEN's
  super-thermal 4.76×B**, and it is the **wrong physics**: CMFGEN's valley is
  scattering-dominated (4.76×B ≫ 1), not thermal. Thermalizing to B would trade one
  disagreement for another.
- **Partly (c) yardstick/estimator.** The `cs_J`=10.5×B target is itself ~2–3× too high
  vs real CMFGEN (deterministic super-thermal S_l). Measured against the *correct* CMFGEN
  yardstick, "mc/cs=0.05" becomes mc/CMFGEN=0.028 — still deeply empty, so this does not
  rescue the field, but the specific "10.5×" number is inflated.
- **Mostly (b′) legitimate NLTE scattering of a too-faint field.** Lumina gets the valley
  *physics* qualitatively right — it is Co III/Fe III/Ni III resonant scattering, exactly
  as in CMFGEN's scattering-dominated valley. The outcome disagrees because Lumina's lines
  scatter a deep radiation field that is itself **~1.5 dex too faint** — the **same deep
  FUV/NUV color deficit as Axis-1** (`F0_F1_FUV_GRADIENT_VERDICT.md`). CMFGEN's valley is
  bright because its 18900 K deep gas radiates a bright continuum; Lumina's is empty
  because its cold 13–15 kK gas does not. **The valley emptiness is a downstream symptom of
  the deep color/field root, not an independent valley defect** — and it shares that root
  with Case A2's inverse (the photosphere manufactures FUV that the deep core never had).

---

## Files (copies only — no commit)
`a2_fuv_excess.py`→`a2_emission_ledger.csv`, `a2_out.txt`; `a2_pairing_check.py`;
`cmfgen_band_J.py`→`cmfgen_band_J.csv`; `v_valley.py`→`v_valley_{field,absorbers,kernel}.csv`,
`v_out.txt`; `v_supp.py`. No `src/` or existing-script edits.
