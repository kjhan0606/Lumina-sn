# DDC15 0.976d — Gold feature reproduction tracking table

Reference: CMFGEN gold `data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat`.
Current best emergent: **radiative macro-atom** (THEN_MC + EWEIGHT energy-weighted +
neutral-ground, k-packet OFF). peak 7050, corr 0.70, red(6-8k) 0.36, NIR(8-12k) 0.56.
Plot: `figures/C7_ddc15_features_2026-06-29.png` (5 feature bands annotated).

Status legend: ✅ reproduced · △ partial (shifted/weak) · ❌ absent/wrong.

| # | Feature (gold) | Gold | Ours (radiative macro-atom) | 재현 | Diagnosis / lever |
|---|---|---|---|---|---|
| ① | **4475 peak** | clear peak @4475, max 1.1e-4 | peak @4475 but **~10× too weak** (1.1e-5) | △→❌ | carrier present at right λ but far too weak. Likely Si II 4481 / Fe II blue-window emission under-produced. Needs more blue fluorescence into this window. |
| ② | **5600 peak** | peak @5645, max 2.5e-4 | peak **@5480** (blue-shifted ~165Å), 1.9e-4 | △ | peak present but **blue-shifted + slightly weak**. Wrong line carrier dominating the window, or velocity/profile offset. |
| ③ | **6500-7200 declining plateau** | broad peak @6590 **declining** to 7200 | **sharp peak @7055** (4.5e-4, overshoot) | ❌ | the gentle 6590→7200 decline is replaced by a red-shifted spike at 7055. Energy piled red of 6600 → the headline color miss. |
| ④ | **7600 dip** | deep trough ~3e-5 @7700 | shallower dip (~6e-5) | △ | dip present but **~2× too shallow** — not enough absorption/blanketing across 7600. |
| ⑤ | **8800-10200 plateau** | flat plateau ~2.2e-4 | **big peak @9290** (3.8e-4) then drop | ❌ | flat NIR plateau replaced by an over-produced 9290 peak (Ca II NIR / Fe-group NIR over-emission). NIR excess (0.56 vs 0.49). |

## Overall assessment
The radiative macro-atom reproduces the **gross structure** (two humps + the 7700 trough,
corr 0.70) — a real breakthrough vs thermal-freqres (featureless) and k-packet macro-atom
(too-red 9240). But the **detailed features fail in a consistent pattern**:

- **Energy shifted red + concentrated into peaks** where gold has *declining/flat plateaus*
  (③ 7055-spike vs 6590-decline; ⑤ 9290-peak vs 8800-10200-plateau).
- **Blue/optical under-produced** (① 4475 10× weak; ② 5600 blue-shifted).
- **Blanketing too weak** (④ 7600 dip too shallow).

Root pattern: emission is **too red and too peaky** — the macro-atom redistributes into a
few strong red/NIR line complexes instead of gold's smoother, bluer plateau distribution.
This is the residual after the k-packet over-thermalization was removed (which fixed the
gross color 9240→7050). Remaining levers to test (1 at a time):
1. **cascade reference** (neutral vs ion ground) — controls red-shift of redistribution (③⑤). [TESTING: ddc15iong]
2. **NIR line over-emission** (⑤, 9290) — Ca II NIR / Fe-group; check populations/branching.
3. **blue window pumping** (①②) — UV→4475/5600 fluorescence under-produced.
4. **blanketing depth** (④ 7600) — absorption opacity.

Gates: T_e must stay ~0.98×CMFGEN (no regression); corr should rise toward 1 as features improve.

## Attempt log
- **2026-06-29 BF=1 (radiative recomb cascade, k-packet off)**: NO effect. peak 7050, corr 0.70,
  red 0.37, NIR 0.55 — essentially identical to nokp baseline (7050/0.70/0.36/0.56). T_e[0]=4259 (no
  regression). Topology built fine (21596 entries, 34 source levels). **Root cause (verified, not a
  bug)**: at the upper-ion ground (the recomb source the bf-activation lands on), radiative excitation
  B_lu·J(UV) ≈ 2e4 /s dwarfs radiative recomb n_e·α ≈ 3e-4 /s by ~7e7 → recomb_prob ~1.5e-8 →
  the macro-atom re-absorbs UV (goes up) ~1e8× more often than it recombines. **Radiative recomb is
  physically negligible here.** BF=2 (RADRECOMB continuum) would share the same rate → also negligible
  (not worth running).
- **Implication**: ARTIS's plateau-filling recomb is the **THERMAL (k-packet collisional) recomb**
  (COLRECOMB from the electron pool), NOT radiative. To fill ③⑤ we need the k-packet thermal pool
  with a COLRECOMB channel routing thermal energy → lower-ion excited levels → cascade. Tension: the
  earlier k-packet (COLL-EXC only) over-thermalized → red (9240). The fix is likely the FULL k-packet
  with COLRECOMB competing against COLL-EXC (ARTIS-balanced) so thermal energy recombines+cascades
  (plateau) instead of re-exciting cool lines (red). Requires re-engaging k-packet carefully.
