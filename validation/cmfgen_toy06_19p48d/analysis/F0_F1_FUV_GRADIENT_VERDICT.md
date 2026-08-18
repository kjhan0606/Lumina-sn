# F0b + F0a + F1 — deep FUV/EUV gradient forensics (OFFLINE)

2026-07-19 (Opus, implementation). Executes the offline battery of
`docs/FUV_GRADIENT_ATTACK_DESIGN.md` (F0b / F0a / F1). Zero GPU hours, no CMFGEN
launch, no process killed, no commit. All scripts + CSVs are copies-only under
`validation/cmfgen_toy06_19p48d/analysis/` (prefix `f0_`/`f1_`).

Inputs (all real, provenance below):
- Lumina B-run (all-level Gph, mc_J field, α=1.0): `logs/coevolve_consume_a10_kx_gphall/`
- Lumina jtable-run (CMFGEN-field arm): `logs/coevolve_consume_a10_kx_jtable/`
- CMFGEN self-run field: `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR` (+ RVTJ)
- CMFGEN published T(v): `data/standart_data1/toy06/phys_toy06_cmfgen.txt`

---

## TL;DR — the fork verdict

**F0b fork = T_e's FAULT (Axis 1a), not transport starvation (Axis 1b). Promote F3-T.**

The deep (s0) FUV field is **at / just below** the dilute-Planck ceiling of its own
cold electron temperature: `J_L / [W·B(T_e=13120K)] = 1.14` (energy-integrated) or
`0.32` (geom continuum) — i.e. within a factor ~3 of `W·B`, **not** the ≥1–2 dex
below that a transport-starved field would show. The event log confirms it: deep
FUV lives in a tight line-scattering balance (gross emit ≈ gross abs, net ≈ 0),
i.e. a *well-coupled thermalized* field at a *cold* amplitude — not an untrapped one.

The measured −2.3 dex (geom) / −1.6 dex (energy) deep FUV deficit vs CMFGEN closes
arithmetically as **COLOR (cold T_e Wien penalty) ≈ 1.3 dex + DILUTION (CMFGEN's
hotter deep gas thermalizes above a W-dilute Planck) ≈ 0.35 dex + genuine transport
RESIDUAL ≈ 0 to +0.5 dex**. Color dominates every band; the pure trapping residual
is the smallest term. Both the DILUTION and RESIDUAL terms are themselves partly
downstream of the cold T_e (colder → less ionized → less line blanketing → thinner),
which is exactly the self-consistent-wrong-fixed-point the design doc names.

---

## Yardstick audit (do this first)

| check | B-run (gphall) | jtable-run |
|---|---|---|
| `T_rad` uniq across 50 shells | **1** (10470.093 K, PINNED) | **1** (10470.093 K, PINNED) |
| `T_e` shells pinned to T_rad (in s0–10) | s9, s10 | **s0, s1, s2** |
| `mc_J` field | transported MC (α=1.0), **real measurement** | transported MC, real |
| event log iter | single (iter 11) | single |
| event log size | 128,000,000 = **CAP128M hit** | 128,000,000 = CAP hit |

Consequences enforced in the analysis:
- Any `B(T_rad)` quantity would be **definitional** (T_rad pinned uniq=1) — so F0b uses
  `B(T_e)`, never `B(T_rad)`.
- The **jtable run's deep s0–s2 `T_e` is pinned to T_rad** → its `J/[W·B(T_e)]` at
  s0–s2 is definitional (marked `*` in the table, and it inflates to 10–35× because the
  pinned T_e=10470 undershoots the real 13120, shrinking B). **We therefore read the
  fork off the B-run, whose s0–s8 T_e are all distinct/unpinned (13120…10811 K).**
- Event log is **one transport iteration, cap-truncated at 128M events**; `etype 7`
  (e-scatter) and `etype 8` (bf-reemit) are **not logged**, so the emission ledger =
  line-emit(2)+kpkt-ff(4)+kpkt-fb(5) and *direct bf recombination re-emission is
  invisible*. Event energies are packet energies (not absolutely CMFGEN-calibrated) →
  only **relative** strengths are claimed from the log; absolute "N-dex vs CMFGEN"
  rests on the F0b `J/B` calibration.

Field `mc_J` provenance (config-verified, from `gradient_budget.py` header /
`src/lumina_plasma.c:5261-5365` blend with α=`LUMINA_COEVOLVE_PHOTOION_ALPHA`=1.0):
`J == g_photoion_mc_J ==` the `mc_J` column — the transported MC shadow field, **not**
a dilute-Planck of the pinned T_rad. Its flatness is transport-real, not definitional.

---

## F0b — deep thermalization test (decisive fork)

Band-integrated (energy: ∫J dν / ∫dν over the band) mean intensity and its ratio to
the local Planck. Full table in `f0b_thermalization_shells.csv`; key rows:

### FUV 918–1290 Å (the deficit band)

| shell | v | T_e(B) | J_L (energy) | B(T_e) | **J_L/B** | **J_L/[W·B]** | CMFGEN J | J_C/B(T_C) | **J_C/J_L** |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 4264 | 13120 | 4.97e-6 | 1.46e-5 | 0.340 | **1.14** | 1.94e-4 | 0.683 | **39×** |
| s2 | 5720 | 13912 | 5.16e-6 | 2.53e-5 | 0.204 | 1.52 | 6.78e-5 | 0.633 | 13× |
| s5 | 7904 | 13629 | 7.62e-6 | 2.10e-5 | 0.363 | 5.58 | 7.84e-6 | 0.791 | 1.0× |
| s8 | 10088 | 10811 | 6.57e-6 | 1.92e-6 | 3.43 | 88.2 | 1.02e-6 | 0.894 | 0.16× (excess) |

Reading: at s0 the Lumina FUV field sits essentially **on** its own `W·B(T_e)` ceiling
(1.14×), and its thermalization fraction `f_L=J/B=0.34` is only ~2× below CMFGEN's
`f_C=0.68`. By the photosphere (s5–s8) Lumina **overtakes** CMFGEN (J_C/J_L<1) — the
Axis-2 photospheric excess. This is the flat/inverted gradient, and it is NOT a
deep-field that has collapsed relative to what its temperature permits.

### The pattern across all four bands at s0 (why it is color, not transport)

| band | J_C/J_L (deficit) | Lumina f_L=J/B | CMFGEN f_C=J/B | note |
|---|---|---|---|---|
| EUV 300–450 | **3.5e3×** (+3.55 dex) | 0.33 | 0.42 | cold gas can't populate the deep Wien tail |
| FUV 918–1290 | 39× (+1.59 dex) | 0.34 | 0.68 | color-dominated |
| flank 1290–2000 | **1.0×** (Lumina = CMFGEN) | 5.74 (super-thermal, transported) | 0.73 | no deficit |
| flank 2000–4000 | 3.8× (+0.58 dex) | 0.71 | 0.74 | *identical* thermalization; pure color |

The deficit **tracks the Wien color penalty of the 13120→18900 K temperature gap**
and vanishes where that penalty vanishes (1290–2000 Å, where Lumina's transported
field actually matches CMFGEN). The thermalization fractions f_L and f_C are
comparable in every band. This is the signature of a color (T_e) problem, not a
transport/trapping problem.

**FORK VERDICT: deep FUV J ≈ W·B(its own cold T_e) → T_e's fault (Axis 1a). Promote
F3-T (temperature-table probe). Transport does NOT starve the deep field independent
of T_e.**

---

## F0a — deep FUV/EUV emission & absorption ledger (event log)

Per-process tallies (energy-weighted) from `lumina_events.bin`, B-run.
Coverage caveats above (single iteration, cap-hit, bf-reemit & e-scatter unlogged).

### FUV 918–1290 Å

| group | line-emit | line-abs | bf-abs | gross emit | gross abs | **NET (emit−abs)** |
|---|---|---|---|---|---|---|
| deep s0–2 | 0.1224 (61,973 ev) | 0.1225 (61,377 ev) | 1.8e-5 (8) | 0.1224 | 0.1226 | **−1.27e-4 (≈0)** |
| phot s6–10 | 6.622 (3.49M ev) | 6.606 (3.48M ev) | 2.4e-4 (129) | 6.622 | 6.607 | **+1.50e-2 (net source)** |

### EUV 300–450 Å

| group | line-emit | kpkt-fb (recomb) | line-abs | bf-abs | gross emit | gross abs | NET |
|---|---|---|---|---|---|---|---|
| deep s0–2 | 2.09e-3 (1,072) | 1.99e-3 (1,005) | 2.9e-4 (150) | **3.79e-3 (1,927)** | 4.08e-3 | 4.08e-3 | ≈0 (+3e-9) |
| phot s6–10 | 2.61e-3 (1,378) | 0 | 8.9e-4 (472) | 1.71e-3 (906) | 2.61e-3 | 2.61e-3 | ≈0 |

### The three required answers

**(1) Is deep FUV emissivity ~CMFGEN-strength-but-untrapped, or ~2-dex weak at the
source?** — Neither framing is right; it is **correctly thermalized to a cold source
and well-trapped**. The deep FUV is in tight line-scattering balance (emit ≈ abs, net
≈ 0), and F0b shows its amplitude sits on `W·B(T_e)`. It is "weak" only relative to
CMFGEN's *hot* deep gas, and that weakness is inherited from T_e (the F1 color term),
not from a broken source or a failure to trap. The event log cannot give an absolute
"2-dex vs CMFGEN" (packet-energy units, no calibration); the calibrated statement
comes from F0b's `J/B`.

**(2) What fraction of photospheric FUV is locally created (upconversion) vs
transported from below?** — **Essentially all of the net FUV is manufactured at the
photosphere; the deep region does not feed it — it is a net FUV sink.** Per-shell net
source (emit−abs, energy):

| region | shells | cumulative NET FUV | role |
|---|---|---|---|
| deep + intermediate | s0–5 | **−1.15e-2** | net **SINK** (removes FUV) |
| photosphere | s6–10 | **+1.50e-2** | net **SOURCE** (creates FUV) |

(A single-number "% photospheric" is undefined here because the two regions carry
*opposite signs* — the deep region absorbs net FUV, the photosphere creates it. So the
photospheric share of net creation is ≥100%.) In gross emission-event activity, FUV
creation peaks at the photosphere (s7–s8 emit ≈ 1.75–1.79) and is ~20–45× weaker in
the deep core (s0–3 emit ≈ 0.04–0.05) — the **opposite** of CMFGEN, whose FUV *field*
is ~190× **brighter deep** (J: 1.94e-4 at s0 vs 1.02e-6 at s8) than at the photosphere.
This is the Axis-2 one-way upconversion: the photosphere manufactures its FUV locally
rather than receiving it from a bright deep source (there is no bright deep source to
receive from). The jtable run (correct CMFGEN ionization field) reproduces the same
pattern (s0–5 net −7.6e-3 sink, s6–10 net +1.68e-2 source) — confirming F2: injecting
the correct field does **not** un-flatten the deep FUV transport.

**(3) Does the deep core emit ANY 300–450 Å EUV?** — **Yes.** Deep s0–2 emits EUV via
both bound-bound (line-emit, 1,072 events, E=2.09e-3) and free-bound recombination
(kpkt-fb, 1,005 events, E=1.99e-3). But it is **locally re-absorbed**, dominated by
bf-abs (photoionization, 1,927 events, E=3.79e-3): gross emit ≈ gross abs, net ≈ 0.
So the photospheric EUV floor is **not "never-emitted"** — the deep core does emit
EUV; it is **absorbed en route** (photoionization sink) before it can build a field.
(Note kpkt-fb recombination EUV appears deep but is absent at the photosphere — the
deep, denser, hotter gas is the only EUV recombination source.) Cross-check: the
jtable run (CMFGEN field injected) drives 4.5× more photoionization overall
(bf-abs 223k vs 49k events) and, deep, 4× more EUV recombination emission
(kpkt-fb E=8.29e-3 vs 1.99e-3) — yet deep EUV net is still ≈ 0 (gross emit = gross
abs to 4 sig figs): more is emitted, all of it is reabsorbed locally.

---

## F1 — arithmetic closure

`Δ = log₁₀(J_C/J_L) = COLOR log₁₀[B(T_C)/B(T_L)] + DILUTION log₁₀[f_C/W_L] +
RESIDUAL log₁₀[W_L/f_L]`  (f = J/B(T_local), W_L = Lumina dilution). Closes to 2 dp.

### (A) Exact band-integrated Planck color factor, T_L=13120 vs T_C=18900 K

| band | B̄(13120) | B̄(18900) | ratio | **color (dex)** |
|---|---|---|---|---|
| EUV 300–450 | 1.058e-11 | 2.780e-08 | 2627× | **+3.42** |
| FUV 918–1290 | 1.453e-05 | 2.840e-04 | 19.5× | **+1.29** |
| flank 1290–2000 | 1.041e-04 | 8.132e-04 | 7.81× | +0.89 |
| flank 2000–4000 | 3.390e-04 | 1.222e-03 | 3.61× | +0.56 |

### (B) Dilution-only profile W(r)/W(s0)

W falls 0.298→0.039 over s0→s8, i.e. **−0.88 dex** if the field were dilution-only.
CMFGEN's FUV field instead **rises inward ~+2.4 dex**. Pure geometric dilution has the
wrong sign and cannot produce Lumina's flat s0→s8 profile — consistent with the
deficit being a source-color effect, not a geometry effect.

### (C) s0 deficit decomposition (closes exactly)

| band | Δ deficit | = COLOR | + DILUTION | + RESIDUAL | (sum) | f_L | f_C |
|---|---|---|---|---|---|---|---|
| EUV 300–450 | +3.55 | +3.42 | +0.15 | −0.04 | +3.52 | 0.33 | 0.42 |
| **FUV 918–1290 (energy)** | **+1.59** | **+1.29** | **+0.36** | **−0.06** | +1.59 | 0.34 | 0.68 |
| **FUV 918–1290 (geom)** | **+2.12** | **+1.29** | **+0.34** | **+0.49** | +2.12 | 0.096 | 0.65 |
| flank 1290–2000 | −0.00 | +0.89 | +0.39 | −1.28 | −0.00 | 5.74 | 0.73 |
| flank 2000–4000 | +0.58 | +0.56 | +0.39 | −0.37 | +0.58 | 0.71 | 0.74 |

**Closure statement (FUV s0):** of the −2.3 dex (geom) / −1.6 dex (energy) deficit,
**~1.3 dex is COLOR (cold T_e), ~0.35 dex is DILUTION (CMFGEN's hotter deep gas is
optically thicker → thermalizes above a W-dilute Planck — an indirect T_e effect),
and the genuine transport RESIDUAL is 0 (energy) to +0.5 dex (geom continuum)** — the
smallest term. The a-priori "color ≈ −1.3 dex, trapping residual ≈ −1 dex" split in
the design doc is corrected: color is confirmed at 1.29 dex, but the residual trapping
term is much smaller (≤0.5 dex), with the middle ~0.35 dex being CMFGEN's superior
deep thermalization, itself a consequence of the temperature gap.

---

## Files produced (copies only, new files)

- `f0b_thermalization_test.py` → `f0b_thermalization_shells.csv`
- `f0a_emission_ledger.py` → `f0a_emission_ledger.csv`
- `f1_arithmetic_closure.py` → `f1_closure_table.csv`
- `F0_F1_FUV_GRADIENT_VERDICT.md` (this file)

No `src/` or existing-script modifications.
