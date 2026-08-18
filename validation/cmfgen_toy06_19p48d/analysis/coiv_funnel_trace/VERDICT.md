# Co IV funnel trace — WHY Lumina's deep MC field piles at 1490-1650Å

Offline, read-only, 2026-07-19. B-run `logs/coevolve_consume_a10_kx_gphall/`.
Target number: deep shell **s0** (shell 0: T_e=13119.87K, n_e=4.426e9, W=0.298,
T_rad=10470K pinned), mc_J/cs_J = 39× at 1526.17Å; Co IV = 84% of pile emission.
The 1526.17Å line = **line_id 391357**, Co IV (Z=27, ion=3), lower level 50 →
upper level 144 (global macro level **22564**), f_lu=0.275, A_ul=7.886e8.

---

## VERDICT — DEFECT FOUND (architectural inconsistency, not a per-line formula bug)

**The MC macro-atom re-emits the deep-shell Co IV forest as a PURE resonant-
scattering/fluorescence source (effective destruction probability ε_eff ≈ 8×10⁻¹⁰),
funneling 95% of every Co IV activation back into the 1490-1650Å complex, while BOTH
CMFGEN and Lumina's own deterministic `cs` solve THERMALIZE that same forest to a
smooth B(T_e)-like source. The 39× mc/cs pile is exactly this inconsistency.**

The macro-atom's per-line arithmetic is *correct* (reproduced exactly below). The
defect is that **no thermalization channel is reachable in the deep MC**, so the MC
line source function ≠ the cs line source function for non-NLTE ions.

### Magnitudes (reconstructed, s0)
| quantity | value | source |
|---|---|---|
| P(Co IV lev144 → internal-down via 1526) | **0.756** | `level144_exit_channels.csv` |
| P(Co IV lev144 → emit 1526) | **0.081** | ratio 0.756/0.081 = 9.3 = e_low/hν = 75.5/8.1 |
| P(emission → 1490-1650 pile), per activation | **95.5%** | `trace_block.py` |
| P(emergent emission → pile), full cascade | **95.2%** (83.7% is 1526.17 alone) | `trace_cascade.py` |
| p_kpacket (collisional/thermal exit prob) | **8.1×10⁻¹⁰** | `trace_block.py`, Step 3 |
| ΣC_down vs ΣA_ul·β (level 144) | 1.16 s⁻¹ vs **1.44×10⁹ s⁻¹** | collisions 9 orders down |
| Co IV share of a-priori line opacity, pile band | **2.1%** (yet 84% of emission) | `trace_step4.py` |

---

## The causal chain (each link cited)

### 1. Line-interaction mode = MACROATOM, ARTIS single-exit (stdout config)
`LUMINA_LINE_INTERACTION=macroatom`, `LUMINA_KPACKET_EXIT=1`,
`LUMINA_DYNAMIC_TRANSPROB=1`, `LUMINA_MACROATOM_EWEIGHT=1`,
`LUMINA_MACROATOM_NEUTRAL_E=1`, `LUMINA_MACROATOM_IDOWN_BETA=1`, `LUMINA_KPACKET=1`.
The macro-atom IS a true ARTIS-style internal-jump atom (up/down/emit), NOT a
bare "de-excite-from-upper-level" resonator — `d_macro_atom_interaction`
(`src/lumina_cuda.cu:2906-3071`) follows `d_transition_probabilities` blocks with
internal jumps. So the resonance is NOT structural in the engine; it is produced
by the atomic structure + the rate weighting, below.

### 2. Co IV is NOT in the NLTE/SE set — only ions I,II per element
stdout NLTE list: every element gets ion=1 and ion=2; **ion=3 (Co IV, Fe IV, Ni IV)
is excluded.** Co IV level populations are therefore nebular dilute-Boltzmann at the
**pinned T_rad=10470K / W=0.298** (`compute_tau_sobolev`, `src/lumina_plasma.c:960-999`),
not SE-solved as in CMFGEN.

### 3. The macro-atom block of Co IV level 144 is a resonator BY THE EWEIGHT
Block = 110 transitions (55 emit + 55 internal-down; **count_up = 0**). Rates
(`src/lumina_plasma.c:1817-2038`, replicated):
- emission i→j: `A_ul·β · (h·ν_ij)`   (eweight `:2023`)
- internal-down i→j: `A_ul·β · (e_lower)`, `e_lower = E_dest + ΣIP(<ion)` (neutral-ground, eweight `:2024-2034`)

For Co IV, `ΣIP = IP(CoI)+IP(CoII)+IP(CoIII) = 58.45 eV`. So for the SAME 1526 line
(same A_ul·β), internal-down/emission = e_low/hν = (17.03+58.45)/8.12 = **9.3×**.
→ 75.6% internal-down vs 8.1% emit — reproduced to the digit. The eweight drives the
packet to cascade *within* Co IV rather than emit at entry, and every downward step
lands in the Co IV manifold whose lines are UV/EUV (atomic fact: Co IV is a high ion).
Result: 95% of all emission from level 144 is the 1490-1650 complex; the full
downward cascade emits **83.7% at 1526.17Å**, mean 1534.5Å.

*(Neutral-ground eweight is the ARTIS convention and is not itself the bug — ion-ground
would emit near entry, also UV. Neither moves energy out of Co IV UV.)*

### 4. NO thermalization exit is reachable in the deep MC — the actual defect
Collisions are physically negligible: p_kpacket = ΣC_down/(ΣC_down+Σradiative) =
**8.1×10⁻¹⁰** (`src/lumina_plasma.c:2158-2164`). This is *correct physics* at
n_e=4.4e9. But it means the k-packet is essentially never formed — and the k-packet
is the **only gate** to the continuum thermalization channels (free-free `-2`,
free-bound `-3`, `src/lumina_cuda.cu:2944-2985`): those fire only AFTER a k-packet
forms, i.e. with prob ≈ 8e-10.

The other possible thermal exit — the macro-atom **bound-free recombination cascade**
(Co IV ↔ Co III/Co V continuum, the coupling CMFGEN uses to bleed Co IV energy into
a smooth recombination continuum) — is **DISABLED**: `LUMINA_MACROATOM_BF` is absent
from the 93-var config, so `build_recomb_topology` returns at its gate
(`src/lumina_plasma.c:1245-1252`), `recomb_block_refs` stays NULL, and the recomb
branch in `d_macro_atom_interaction` (`src/lumina_cuda.cu:3016-3030`) is never taken.

⇒ Every erg absorbed into deep-shell Co IV can leave ONLY as a Co IV UV line photon.
That is the funnel: absorption-into-Co IV × (95% resonant UV return) with the thermal
sinks both unreachable.

### 5. The deterministic cs solve DOES thermalize the same lines → 39× split
The SE line source `S_l = (2hν³/c²)/(g_u n_l/(g_l n_u) − 1)` is written **only for
NLTE lines** — the update `continue`s when either level is non-NLTE
(`src/lumina_cuda.cu:1467-1469`). Co IV lines are non-NLTE ⇒ `line_source_S` is never
written for them ⇒ the cs assemble uses its thermal treatment (code's own comment
`:1476-1478`: "cmfgen_assemble's B(T_e) fallback fired for 100% of the forest …
maximal local thermalization"; with `LUMINA_CMFGEN_LINE_EPS_PHYS=1`, non-tabulated
lines fall back "to the legacy fully-thermal treatment", `src/lumina_plasma.c:4502-4507`).
So `cs_J` sees Co IV as a smooth ~B(T_e)/continuum-scattering source, while `mc_J`
sees it as ε_eff≈8e-10 resonant recycling. **Same lines, two inconsistent source
functions → mc_J/cs_J = 39× at 1526, mc_J = 4% of cs_J at 1700-2100Å.** This matches
CMFGEN truth (deep field smooth ≈ B(18760)) on the cs side; the MC side is the
divergent stage.

### 6. Control — Fe III does NOT pile (why it is Co-IV-specific)
Fe III (Z=26, ion=2) is a rich NLTE ion. Its a-priori opacity dominates the FUV
(51%) and is 23% of the pile band, yet it is only **1.5% of pile emission** — because
its dense low-level ladder lets UV-absorbed energy cascade step-by-step to lower
levels and out (and its S_l IS SE-coupled, step 5). Co IV's manifold decays into the
metastable level-50 (E=17 eV) trap with only an EUV gap to ground → energy stalls in
the 17-25 eV manifold and re-emits the 1490-1650 complex. `trace_step4.py`.

---

## What to fix / next concrete target
Reconcile the MC macro-atom emissivity with the cs line source for **non-NLTE ions**.
Three orthodox options (design only — no runs here):
1. **Promote stage-IV iron-peak (Co IV, Fe IV, Ni IV) into the NLTE/SE set** so `S_l`
   is written (`lumina_cuda.cu:1495-1503`) and the SAME source drives both paths.
2. **Enable the macro-atom bound-free thermal exit** (`LUMINA_MACROATOM_BF`, the
   Co IV↔Co III/Co V recomb cascade, `lumina_plasma.c:2090-2123`) so absorbed UV can
   leave as recombination continuum — the CMFGEN mechanism — instead of Co IV lines.
3. **Make deep-shell MC line interactions sample the cs thermal source** (the
   `radeq_line_eps_phys` ε / B(T_e) the cs solve already uses), i.e. give the macro-
   atom the thermalization the continuum-thick region physically has.

Immediate falsifier for the primary claim: with option 2 on, the pile share should
collapse toward CMFGEN's smooth continuum and mc_J→cs_J at 1526 vs 1800Å.

## Artifacts (this dir)
- `trace_block.py` → `level144_exit_channels.csv` — exact block reconstruction (Step 2/3).
- `trace_cascade.py` — full Co IV downward cascade, emergent line distribution.
- `trace_step4.py` — tau-by-ion (entry side) + Fe III control (Step 4).
