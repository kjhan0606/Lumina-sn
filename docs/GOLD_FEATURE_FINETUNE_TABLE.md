# Gold Feature Comparison + Fine-Tuning Items (2026-06-26)

DDC15 0.976d gold vs current best model (**scatter-source obs, 170032**: peak ~6400,
grn/nir 0.51, P-Cygni present). All flux **normalized to each spectrum's own peak**.
Goal: drive every gold feature (position + magnitude) to match.

## A. Gold spectral features = the quantitative comparison targets

### Peaks (emission shoulders) — position / gold value / model value@λ / carrier
| # | λ (Å) | gold | model | carrier | verdict |
|---|-------|------|-------|---------|---------|
| P1 | **6590** | **0.99** | 0.77 | continuum color peak (~4400K) + Si II flank | **green peak weak** |
| P2 | 5645 | 0.73 | 0.82 | S II / Fe II emission shoulder | model slightly high |
| P3 | 5910 | 0.51 | 0.61 | Si II 5972 emission | model high |
| P4 | 8650 | 0.65 | 0.86 | Ca II NIR triplet emission | **model NIR too high** |
| P5 | 8875 | 0.65 | — | Ca II NIR red component | check |
| P6 | 9910 | 0.47 | — | continuum/Fe | check |

### Dips (absorption troughs) — position / gold depth(1−f) / model / carrier
| # | λ (Å) | gold f | model f | gold depth | model depth | carrier | verdict |
|---|-------|--------|---------|-----------|-------------|---------|---------|
| D1 | 3375-3900 | 0.00-0.01 | **0.59** | 0.99 | 0.41 | Ca II H&K + UV blanket | **far too shallow** |
| D2 | 4910 | 0.03 | **0.60** | 0.97 | 0.40 | Fe II forest | **far too shallow** |
| D3 | 5175 | 0.02 | **0.76** | 0.98 | 0.24 | Fe II / Mg II / S II | **far too shallow** |
| D4 | 5990 | 0.51 | 0.66 | 0.49 | 0.34 | **Si II 6355** (signature) | too shallow |
| D5 | 7760 | 0.09 | **0.54** | 0.91 | 0.46 | O I 7774 | **far too shallow** |
| D6 | 9790 | 0.47 | 0.55 | 0.53 | 0.45 | Ca II NIR trough | close-ish |

### Plateau
| region | gold height | carrier |
|--------|-------------|---------|
| 3000-4090Å | ~0.00 (deep) | UV line-blanketing black-out (Fe-peak iron curtain) |

## B. Diagnosis from the comparison (the dominant pattern)

**Every absorption dip is far too shallow** (gold 0.01-0.09, model 0.54-0.76); the NIR
emission (8650) is too high; the green peak (6590) is weak. This is the **scatter-source
trade-off**: pure-scatter line source `S_l=W·B` re-emits the backlight back into the
trough → fills it. Gold has genuine deep absorption. So the continuum COLOR is now right
(scatter fixed the reddening) but the line FEATURES are washed out.

⟹ The master lever is the **line absorption/scattering balance** (ε): pure scatter (ε=0,
current) = right color, no troughs; full thermal (ε=1) = deep troughs, reddened (the old
8544). The optimum is intermediate, **per-line** (resonance lines like Ca II / Si II / O I
scatter but with a destruction fraction that deepens the trough; Fe forest blankets).

## C. Fine-tuning items (detailed, each → knob → target feature)

| # | Item | Knob | Mechanism | Target feature(s) |
|---|------|------|-----------|-------------------|
| **F1** | **Trough depth (DOMINANT)** | `g_sob_eps` (LUMINA_CMF_OBS_EPS): line source `S_l=(1−ε)·W·B + ε·B`. Raise ε from 0 → adds local-thermal (absorbing) fraction that does NOT refill the trough | deeper troughs without global reddening | **D1,D2,D3,D5** (Ca H&K, Fe II, O I) |
| **F2** | **Si II 6355 trough** (signature) | `OBS_DVRES` (resolution) + ε; Si II ionization (Si II→III at green shells?) | sharpen/deepen the 5990 P-Cygni | **D4** |
| **F3** | **green peak 6590 strength** | continuum color fine (the static gives 0.99 here; obs 0.77) — check obs continuum S_c vs static; + grn/nir | raise P1 to ~0.99, move peak 6408→6590 | **P1**, grn/nir 0.51→0.58 |
| **F4** | **NIR emission 8650 too high** | coupled to F3 (cooler-blue peak ⇒ relatively high NIR); Ca II NIR source | lower P4 0.86→0.65 | **P4** |
| **F5** | **fluorescence (color-safe re-add)** | NLTE source ONLY for pumped super-thermal upper levels (b_j>1), scatter for the rest | adds the real 4200-4660 Fe II emission without the thermal-reddening of full NLTE source | minor green/blue-green, P2/P3 |
| **F6** | **red-edge / NIR window** | P2 artifact: `nu_cmf` leaves window → clamp | clean >10000Å (full-range 2000-12000; H Lyα 1215 cold-shell clamp if going <2000) | NIR shape, D6 |
| **F7** | **DVRES / NObs** (resolution) | output grid + sub-step resolution | sharpen all P-Cygni edges | all dips |

### Priority order (by leverage on gold-match)
1. **F1 (ε trough depth)** — the dominant residual; one scalar A/B (ε = 0, 0.05, 0.1, 0.2) to find the depth-vs-color optimum. Most dips fixed here.
2. **F3+F4 (green peak / NIR)** — the grn/nir 0.51→0.58 residual; check obs continuum S_c vs static (why obs P1=0.77 vs static 0.99 at 6590).
3. **F2 (Si II 6355)** — the signature feature; ionization + resolution.
4. **F5 (color-safe fluorescence)** — refinement once F1-F4 land.
5. **F6/F7** — clean-up (window edge, resolution).

### Per-feature PASS gate (gold-match tolerance)
- Peaks: position within ±100Å, value within ±0.05 (normalized).
- Dips: depth within ±0.10; the deep ones (D1,D2,D3,D5) must reach <0.15.
- grn/nir: 0.58 ± 0.03; peak 6590 ± 100Å.
- One knob per A/B run; falsify each against this table.

## Links
[[project_autonomous_stage2_2026-06-25]] (milestone: scatter-obs P-Cygni+color),
docs/ORTHODOX_FREQRES_NLTE_DESIGN.md, figures/2026-06-26_obs_scatter_vs_nlte.png.
