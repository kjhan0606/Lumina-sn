# Reddening localization & emission forensics — Lumina deep bath vs CMFGEN toy06 @19.48d

Offline analysis, 2026-07-19. Read-only on `logs/` and `/gpfs`. No source edits, no commit.
Data: CMFGEN `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4` (EDDFACTOR J_ν) vs Lumina B-run
`logs/coevolve_consume_a10_kx_gphall/{lumina_coevolve_field.csv (mc_J), lumina_events.bin}`.

**Question.** The deep FUV(918–1290Å) deficit is spectral (bolometric u only −0.24 dex, FUV −1.54 dex →
~1.3 dex is energy at the wrong colors). *Where* does Lumina's deep energy actually sit, and *which
emission channel* puts it there?

---

## Headline
1. **The deep bath is not smoothly "reddened" — it is FUNNELED into a narrow ~1500Å pile.** At s0, Lumina
   holds **42% of its total energy density in one log-λ bin at 1508Å** (CMFGEN: 9%; a smooth broad UV
   plateau). Everything blueward of 1290Å is starved (−1.6 to −18 dex) and the 2000–4500Å near-UV is *also*
   a deficit valley (−0.2 to −1.0 dex). A genuine red/NIR excess (×5–10) sits beyond 4500Å.
2. **The single most guilty emission channel is Co IV.** Co IV line emission = **83.0% of ALL deep (s0-2)
   emission energy** and **84.0% of the 1290–2000Å pile**; the pile itself is **96.3% of all s0-2 emission.**
   So **Co IV NUV emission ≈ 80.9% of every erg the deep line forest emits.** It is a dense Co IV line
   complex at 1490–1650Å (strongest 1526.17Å), re-emitting into the ~1500Å pile seen in the field.
3. **The red/NIR field excess is NOT line emission** — the line forest is a *net sink* redward of 2000Å.
   It is consistent with the cool 10020K DIFFUSE_INNER_BC seed (+ unlogged bf recombination) streaming
   through, i.e. the campaign's established inner-BC color thermostat — a *separate* mechanism from #2.

---

## Divergence table (quantity × shell × band × CMFGEN vs Lumina × ratio × mechanism)

### TASK A — band-resolved energy density  u_band = (4π/c)∫J_ν dν  [erg cm⁻³]
Anchors reproduced before extending: CMFGEN FUV geo-mean s0 = **2.0231e-4** (t 2.023e-4), s8 = **7.7286e-7**
(t 7.729e-7); Lumina FUV arith-mean s0 = **5.809e-6**. Bolometric ratios s0/s2/s4 = **0.577 / 0.996 / 1.573**
(match trapping-audit 0.576/0.995/1.570). Full table: `taskA_band_table.csv`.

**s0 (v=4264):** total u CMFGEN 694.1, Lumina 400.2, ratio 0.577.

| band (Å) | u_CMFGEN | u_Lumina | ratio | dex | f_CMF | f_Lum | role |
|---|---|---|---|---|---|---|---|
| 300–450 EUV | 2.05e-2 | 4.60e-6 | 0.000 | −3.65 | .000 | .000 | deep deficit |
| 450–918 xuv | 20.71 | 0.299 | 0.014 | −1.84 | .030 | .001 | deep deficit |
| **918–1290 FUV** | **83.60** | **1.922** | **0.023** | **−1.64** | **.120** | **.005** | **deep deficit (headline)** |
| **1290–2000 NUV** | 218.4 | **205.5** | 0.941 | −0.03 | .315 | **.514** | **Lumina PILE (½ of its u)** |
| 2000–3000 UV | 204.8 | 21.82 | 0.107 | −0.97 | .295 | .055 | deficit VALLEY |
| 3000–4500 blue | 114.3 | 69.08 | 0.604 | −0.22 | .165 | .173 | mild deficit |
| 4500–7000 opt | 47.47 | 65.72 | 1.384 | +0.14 | .068 | .164 | **excess** |
| 7000–10000 red | 3.458 | 19.91 | 5.758 | +0.76 | .005 | .050 | **excess** |
| 10000–19933 NIR | 1.371 | 13.65 | 9.954 | +1.00 | .002 | .034 | **excess** |

s2 and s4 follow the same shape (NUV pile f_Lum 0.64 / 0.58; FUV −1.17 / −0.60 dex; red/NIR ×4–10). At s4
the bolometric crosses above 1 (1.573) yet the FUV is still −0.60 dex and the pile still dominates —
i.e. the deficit is *spectral at every shell*, not an energy shortfall.

**Crossing wavelengths (ratio=1, smoothed SED, s0):** the SED ratio crosses 1 at **≈1490Å and ≈1650Å**
(bracketing the Co IV pile) and at **≈4900Å** (the persistent blue-deficit → red-excess crossover).
Below 4900Å Lumina is in deficit *except* the isolated 1490–1650Å Co IV spike; above 4900Å it is in excess.
Per-shell crossings: `taskA_crossing.csv`, `taskA_sed_shape.py` output.

**Field diagnostic (raw mc_J vs cs_J, s0, 1430–1650Å):** mc_J exceeds the deterministic cs_J by 5–39× inside
the pile and is 10–50× *below* cs_J at 1700–2100Å. The pile is a property of the **Monte-Carlo** field, not
the deterministic solve — the code's own `[COEVOLVE-COLOR]` "MC_bluer / MC_UV_stronger" probe already
flags this mc-vs-deterministic color split (`src/lumina_cuda.cu:5377-5384`).

### TASK B — emission/absorption event ledger (s0-2)  `taskB_band_ledger.csv`

Coverage (audited, restated): n = **128,000,000 = CAP128M (SATURATED)**; single iteration (iter=11);
etype hist line-abs 63.88M ≈ line-emit 63.92M (resonant scattering), bf-abs 49k, kpkt-ff 4160, kpkt-fb 9330,
escape 135k. **etype 7 (e-scatter) & 8 (bf-reemit) UNLOGGED → the bf recombination continuum (CMFGEN's
thermal source) is invisible; this ledger is the LINE-FOREST flow only.** Energies are packet units
(relative shares only).

| band (Å) | emitE | absE | net | flow |
|---|---|---|---|---|
| 450–918 xuv | 0.224 | 0.225 | −0.001 | ~balanced |
| 918–1290 FUV | 0.122 | 0.123 | −0.0001 | slight sink |
| **1290–2000 NUV** | **16.22** | **15.98** | **+0.241** | **SOURCE (Co IV pile)** |
| 2000–3000 UV | 0.232 | 0.366 | −0.134 | **net sink** |
| 3000–4500 blue | 0.024 | 0.091 | −0.067 | **net sink** |
| 4500–7000 opt | 0.011 | 0.033 | −0.022 | net sink |
| 7000–19933 red/NIR | 0.001 | 0.016 | −0.015 | net sink |

**Net spectral flow:** the deep line forest ABSORBS from the 2000–4500Å near-UV/blue and from the red, and
RE-EMITS (net) into the 1290–2000Å Co IV pile. The net band-level flow is 2000–4500 → 1290–2000, i.e.
*blueward* concentration, **not** a broad blue→red cascade. The forest neither feeds the FUV (918–1290, a
slight sink) nor the red/NIR (a sink) — it is a **resonant funnel to the Co IV 1500Å complex** that robs
both sides.

**Top emitters into the pile (1290–2000, s0-2):** Co IV 84.0%, Co III 8.0%, Ni IV 4.8%, Fe III 1.5%.
**Top FUV/xuv absorbers (450–1290, s0-2):** Co III 37.6%, Fe III 33.9%, Fe IV 8.7%, Co IV 8.7%.
→ Co III+Fe III soak up the FUV; Co IV dumps it back at ~1500Å. `taskB_top_ions.csv`.

### Up-conversion across 1290Å (blueward emission), same footing  `taskB_upconversion.csv`
| group | n_emit(<1290) | E_emit | E_emit/shell | net(emit−abs) | verdict |
|---|---|---|---|---|---|
| s0-2 (3 sh) | 177,856 | 0.351 | 0.117 | **−0.001** | **net FUV SINK** |
| s7-8 (2 sh) | 4,485,422 | 8.511 | 4.255 | **+0.012** | **net FUV SOURCE** |

Per-shell blueward-emission ratio **s7-8 / s0-2 = 36.4×** (by event count 37.8×) — corroborates the known
photosphere-dominated upconversion (task cited ~45×; same-footing value 36–38×). **The deep shells do not
up-convert; they are a slight net FUV sink.** All blue is manufactured at the photosphere.

### CMFGEN-side color contrast (s0-2)
Lumina s0-2 emission-weighted mean λ = **1553Å**; B(T_e=13120) mean = 3876Å; B(T_col=18760) mean = 2770Å.
The deep line emission is **neither thermal nor "redder than thermal" — it is a narrow non-thermal Co IV UV
spike, BLUER in mean than either blackbody.** Emission-energy fractions vs thermal:

| band | Lumina emit | B(13120) | B(18760) |
|---|---|---|---|
| <1290 | 0.021 | 0.028 | 0.145 |
| 1290–2000 pile | **0.963** | 0.162 | 0.290 |
| 2000–4500 | 0.015 | 0.545 | 0.439 |
| >4500 | 0.001 | 0.265 | 0.125 |

Only **0.1%** of deep emission energy lies redward of the B(13120) mean (thermal → ~50%). **<2% of the deep
line emission is consistent with a thermal B(13120) color.** The red/NIR *field* excess (Task A) therefore
does **not** come from the logged line channels (a net sink there) — it is the cool inner-BC seed + unlogged
bf, a separate mechanism.

---

## Reconciliation with the "one-way down-conversion" suspicion
Partially confirmed, precisely bounded by the data:
- **Confirmed (weak form):** the FUV(918–1290) is not refilled by the deep forest; its would-be energy sits
  ~200–500Å redward in the 1290–2000 Co IV pile → a *small* redward displacement of the FUV.
- **Refuted (strong form):** there is **no broad blue-absorbed→red-emitted cascade** at the deep shells. The
  net line-forest flow is a *blueward* concentration into 1500Å; red/NIR is a net sink. The big red/NIR
  *field* excess is inner-BC/transport, not line down-conversion.
- **Mechanism = source-function / thermalization, not line data.** CMFGEN carries the same Co IV complex yet
  its deep field is a smooth ≈B(18760) continuum. Lumina's MC field piles at the Co IV lines because the
  local line source function is built from low-T_e (13120K) pops with Co IV dominant, and the MC transport
  resonantly recycles energy in that complex instead of thermalizing it (mc_J ≠ cs_J).

---

## Most guilty channel & the test (design only — no runs)
**Guilty channel: Co IV line emission, 1490–1650Å complex — 80.9% of all deep (s0-2) emission energy.**
Co IV is the dominant Co ion at s0-2 (n(Co IV)=9.58e8 vs n(Co III)=3.11e8 at s0; `lumina_ion_pops.csv`),
so both the population and the emission point at Co IV.

**Pre-registered A/B (offline design):**
1. **Co IV/Co III balance probe (primary).** Toggle the Co IV→III recombination/DR channel (the existing
   `LUMINA_ALPHA_SPINGATE` / Co-DR gate; ties to the campaign's "Co rate ~10× deficit"). Predict: raising Co
   recombination shifts Co IV→Co III at s0-2 *toward the CMFGEN ratio*, collapses the 1290–2000 pile share
   (from 96% toward CMFGEN's ~30%), and refills the 918–1290 FUV and 2000–4500 valley. Metric: pile
   emission-share + FUV(918–1290) mc_J vs CMFGEN 2.023e-4.
2. **Thermalization probe (secondary).** The pile lives in mc_J, not cs_J. Force deep-shell MC thermalization
   (more line interactions / redistribution at high τ) and re-measure mc_J vs cs_J at 1500 vs 1800Å. If the
   Co IV pile relaxes toward the broad cs_J/CMFGEN continuum, the funnel is an MC source-function artifact,
   not an ionization error.
3. **Inner-BC color probe (already the campaign F3-B).** Separates the red/NIR field excess (cool 10020K
   seed) from the Co IV pile; the two are additive contributors to the FUV deficit and must be tested apart.

---

## Artifacts (this directory)
- `taskA_band_localization.py` → `taskA_band_table.csv` (per shell×band u, ratio, fraction),
  `taskA_overlay_spectrum.csv` (per-λ J_Lumina, J_CMFGEN, ratio for plotting), `taskA_crossing.csv`.
- `taskA_sed_shape.py` (coarse-log SED both codes + smoothed crossings).
- `taskB_event_forensics.py` → `taskB_band_ledger.csv`, `taskB_top_ions.csv`, `taskB_upconversion.csv`.

## Sources relied upon (paths + lines)
- Event schema: `scripts/read_events.py:4-25` (LUMEVT01 header 32B, 20-byte EventRec; LUMLIN01 lines
  {f32 lam, u16 Z, u16 ion}); etype map `:24-25` (7=e-scatter, 8=bf-reemit both unlogged in this run).
- Field writer: `src/lumina_cuda.cu:5387-5402` (cs_J = cs.J deterministic, mc_J = nlte_Jmc Monte-Carlo);
  `[COEVOLVE-COLOR]` mc-vs-det probe `:5377-5384`.
- CMFGEN EDDFACTOR reader: `validation/.../analysis/extract_jnu.py:17-35` (validated); anchors `:99`.
- Ion balance: `logs/coevolve_consume_a10_kx_gphall/lumina_ion_pops.csv` (Co stage-3 = Co IV dominant).
- Prior context: `validation/.../trapping_audit/VERDICT.md` (bolometric u, τ, forest census),
  `validation/.../f0a_emission_ledger.py` (deep=sink/phot=source precedent).
