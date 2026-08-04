# Stage-IV iron-peak NLTE repair — decision document (Fork A vs Fork B)

Offline design + sizing, 2026-07-19. No source edits, no runs, no commit. All numbers
measured from `data/tardis_reference_toy06_19p48d/` (25,620 levels / 2,565,342 lines) and
the completed B-run `logs/coevolve_consume_a10_kx_gphall/stdout.log` (A100-80GB). Do not
read/modify the live run `logs/coevolve_consume_a10_kx_mabf/`.

## The defect being repaired (from `coiv_funnel_trace/VERDICT.md`)
Deep-shell (v<7000, stage-IV-dominated) Co IV / Fe IV / Ni IV are **outside the NLTE/SE
set** (`NLTE_TARGET_Z/ION` covers ion_number 1,2 = II,III only, plus the O I/II/III
overlap; `lumina_plasma.c:3934-3941`). Three consequences:
1. `S_l` is never written for their lines — the S_l writer `continue`s when either level
   is non-NLTE (`lumina_cuda.cu:1466-1503`, gate at 1467-1469) — so the deterministic `cs`
   solve falls back to a **B(T_e) thermal** source (`radeq_line_eps_phys` returns −1 →
   "legacy fully-thermal", `lumina_plasma.c:4502-4507`), while the MC macro-atom recycles
   the same lines **resonantly** (ε_eff≈8e-10). Two inconsistent source functions →
   **mc_J/cs_J = 39× at 1526Å**.
2. Their level pops are nebular dilute-Boltzmann at the pinned T_rad=10470K, not SE.
3. The macro-atom's only thermal exits — the k-packet continuum (prob ≈8e-10 at
   n_e=4.4e9) and the bf recomb cascade (`LUMINA_MACROATOM_BF`, off by default; under live
   A/B test) — are both unreachable, so every erg into deep Co IV leaves only as a Co IV
   UV line (95% resonant return to the 1490–1650Å pile).

CMFGEN thermalizes that same forest to a smooth ≈B(18760K) deep field. The `cs` side
already matches CMFGEN; **the MC side is the divergent stage.**

---

## Fork A — promote stage-IV iron-peak into the NLTE/SE set

### A.1 Level & line counts (measured)
`ion_number` is 0-based (0=I). Stage-IV iron-peak ions each carry **200 levels**; the
ARTIS super-cutoff K=100 (`LUMINA_SUPER_CUTOFF=100`, active in the B-run) lumps
level_num≥100 into one super-level (`lumina_atomic.c:722-736`), so each adds
**min(200,101)=101 explicit NLTE super-levels**.

| ion | Z,ion# | full levels | lines | NLTE super-levels (K=100) | III/IV pair N=super(III)+super(IV) |
|---|---|---|---|---|---|
| Fe IV | 26,3 | 200 | 4,336 | 101 | 202 |
| Co IV | 27,3 | 200 | 4,041 | 101 | 202 |
| Ni IV | 28,3 | 200 | 4,199 | 101 | 202 |
| (opt) Cr IV | 24,3 | 200 | 4,609 | 101 | 202 |
| (opt) Ti IV | 22,3 | 126 | 1,000 | 101 | 202 |

The III partner already contributes 101 super-levels (Fe III 1500→101, Co III 3917→101,
Ni III 1000→101), so every new III/IV pair matrix is **N=202** — identical to every
existing II/III pair. Full sizing in `stage4_sizing/forkA_ion_sizing.csv`.

### A.2 Memory & solve cost (measured against the B-run)
The GPU SE solve dimensions off **super-levels** (`nlte_ion_super_offset`,
`lumina_cuda.cu:791-796`), not full levels — the batched cuBLAS buffer is
`n_shells·max_N²·8 B`. B-run baseline (verbatim from stdout): `n_nlte_levels_total=21038`,
`n_super_total=2828`, `Lines mapped to NLTE: 2410046/2565342`, NLTE pop memory **8.8 MB**,
R_bf GEMM table **41 MB (TF32, 16 pairs, 10340 phot levels × 50 shells)**. n_shells=50,
GPU = A100 **79.3 GB**.

| scenario | slots | pairs | full levels | super total | max_N | rate-matrix VRAM | pop+within VRAM | Δ pop VRAM | +NLTE lines |
|---|---|---|---|---|---|---|---|---|---|
| CURRENT | 31 | 16 | 21,038 | 2,828 | 202 | 16.3 MB | 16.8 MB | — | — |
| +Fe/Co/Ni IV | 34 | 19 | 21,638 (+2.85%) | 3,131 | **202** | **16.3 MB** | 17.3 MB | **+0.48 MB** | +12,576 |
| +Fe/Co/Ni/Cr/Ti IV | 36 | 21 | 21,964 (+4.40%) | 3,333 | **202** | **16.3 MB** | 17.6 MB | +0.74 MB | +18,185 |

(`stage4_sizing/forkA_aggregate_sizing.csv`.) **VRAM is a non-issue**: max_N is unchanged
(the Co II/III pair already sets N=202), so the batched-solve buffer does not grow;
population/within-SL arrays grow ~0.5 MB; R_bf grows by ~3×101 phot levels (≈+1 MB). On an
80 GB GPU that already fits comfortably (the B-run's 41 MB R_bf + 8.8 MB pops are <0.1% of
VRAM; the transport-side arrays — tau_sobolev 615 MB, transition_probabilities 1.85 GB on
disk — are the real VRAM consumers and are **untouched** by Fork A). **Headroom: >70 GB.**

The real cost is **compute, not memory**: +3 pairs (16→19, +19%) each add one full
2.565M-line assembly-kernel scan and one 50×202³ batched LU, ×5 CE iterations. Estimated
**+15–20% NLTE-solve wall time** per outer iteration. Modest and bounded.

### A.3 Ladder-closure blocker (measured — decides the pattern)
The O I/II/III precedent (slots 28-30, shared slot 29, `lumina_plasma.c:3928-3933`)
**does extend** structurally to Fe/Co/Ni III/IV: append IV as a new slot, add a
`(III,IV)` pair that shares the III slot with the existing `(II,III)` pair; the outer CE
loop converges the triple sequentially, exactly as O does. The `(III,IV)` pair-solve needs
only: IP(III→IV) — **present** for all three (`ionization_energies.csv`: 26,2=30.65;
27,2=33.50; 28,2=35.19 eV) — and σ_bf of the III levels (already loaded, since III is an
NLTE ion). **IV as top ion does NOT need stage V.** This matters because:

- **Fe V exists** (200 levels, 4,558 lines) — Fe could optionally go II/III/IV/V.
- **Co V and Ni V do NOT exist** in the dataset (Co max = IV, Ni max = IV; no levels, and
  no IP beyond 27,3 / 28,3). So Co/Ni **must** close the ladder at IV as the top ion. This
  is fine — the pair solve treats IV as `hi` and never photoionizes it further — but it is
  a hard constraint: **you cannot make IV a `lo` ion for these two.**

### A.4 Wiring inventory (touch points, file:line)
| # | concern | location | change |
|---|---|---|---|
| 1 | NLTE target tables | `lumina_plasma.c:3934-3941` `NLTE_TARGET_Z/ION[]` | append 3 (or 5) slots (Z=26/27/28 ion=3) |
| 2 | array dimensioning | `lumina.h:331-332` `NLTE_MAX_IONS 31→34`, `NLTE_PAIR_COUNT 16→19` (fixed-size arrays `nlte_Z[NLTE_MAX_IONS]` etc. `lumina.h:341-368`) | bump both defines |
| 3 | pair list (×2 literals) | `lumina_cuda.cu:707-714` and `lumina_plasma.c:11080` `pairs[][2]` + `names[]` | append IV slots (Fe IV=31, Co IV=32, Ni IV=33) and add pairs `{5,31},{9,32},{11,33}` (Fe/Co/Ni III→IV) sharing the existing III slots 5/9/11 — the O-triple pattern |
| 4 | **max_N precompute assumes hi=lo+1** | `lumina_cuda.cu:4667` `pair_lo_init[]` | **must switch to the explicit `pairs[][2]` table** — the #281 comment already warns this (overlap pair lo≠2·p); an appended IV slot is not lo+1 of III, so the naive `hi=lo+1` computes the wrong N |
| 5 | top-ion closure | `lumina_plasma.c:9860-9880` (`hi_ground_global_nlte`, `nu_edge=IP(lo)`) | works as-is once IV slot exists; verify IV has no bf-out demand |
| 6 | S_l writer | `lumina_cuda.cu:1466-1503` | **no code change** — once both IV levels are NLTE-mapped, `nlte_lo/nlte_up ≥ 0` and S_l is written automatically |
| 7 | Gph / photoion consumers | `lumina_plasma.c:4839,5037+` (`radeq_simul_all` Gph loop, all-level `g_gph_alllevel`) | III→IV photoion now feeds the SE balance; verify Gph sums include the new pair's lo=III levels (already does — III is unchanged) |
| 8 | writeback | `lumina_plasma.c:1017-1065` `nlte_writeback_ion_stage` | **overlap pairs are SKIPPED** (1026-1034) — so the III/IV (and existing O) tau writeback is not applied; flag as a known limitation, not a regression |
| 9 | macro-atom level mapping | `line2macro_level_upper`, `macro_block_references` (loaded from `macro_atom_data.csv`) | **no change** — Co IV already activates the macro-atom; promotion changes only the *populations* feeding dynamic transprob, not the topology |

### A.5 The critical caveat — Fork A does not, by itself, break the MC funnel
The MC macro-atom transport consumes **transition probabilities** (the eweight cascade,
`lumina_cuda.cu:2906-3071`), not `line_source_S`. `line_source_S` drives the **cs** solve
and the formal spectrum; it reaches the macro-atom only indirectly, via dynamic-transprob
rebuild from populations (and only under `LUMINA_NLTE_JBAR_POPS`, `lumina_cuda.cu:6444-6510`).
Promoting Co IV to NLTE therefore fixes: (i) populations → SE, (ii) cs↔SE consistency,
(iii) S_l is written. But the funnel geometry (VERDICT step 3: internal-down/emit =
e_low/hν = 9.3×, set by Co IV *atomic energies*, not populations) and the k-packet thermal
exit (≈8e-10 at n_e=4.4e9, a physical fact) are **unchanged by promotion**. The macro-atom
will still cascade resonantly inside the (now-NLTE) Co IV manifold unless a thermal exit is
added — i.e. Fork A is the *fidelity endpoint for the cs/populations*, but needs a
companion (`LUMINA_MACROATOM_BF` recomb cascade — VERDICT option 2 — or Fork B's thermal
re-emit) to break the MC-side 39× pile. **This is the decisive interaction below.**

---

## Fork B — source-function consistency for non-NLTE MC lines

### B.1 The mechanism already exists; only the gate is wrong
`d_ltherm_reemit` (`lumina_cuda.cu:2774-2777`) **already re-emits from Planck(T_e[shell])**
(`d_sample_planck_frequency(d_ltherm_te[shell_id], rng)`). For a non-NLTE line the cs
source *is* the B(T_e) fallback (S_l never written), so **sampling Planck(T_e) in the MC
exactly reproduces the cs treatment of exactly those lines.** No new source array is
needed. (`line_source_S` is a host opacity array not uploaded to the transport kernel, and
for non-NLTE lines it is 0 anyway — Planck(T_e) is both the correct target and the cheap
one.) Note: sample **T_e**, not T_rad — the eps_uv/bf path samples Planck(T_rad=10470K),
but the deep field truth is B(T_e≈13120K locally / 18760K CMFGEN); T_e is the cs's own
choice and the conservative match.

### B.2 Exact decision point & the per-line flag
The macro-atom is activated at `lumina_cuda.cu:3205-3218` inside `d_line_scatter_event`.
The device **already knows** the activating line's species: `d_line_atomic_number[*next_line_id]`,
`d_line_ion_number[*next_line_id]` (used at :3132 for `is_fe_scatter`, :3158-3159 for
`act_Z/act_ion`). So Fork B can gate exactly like `is_fe_scatter` does. Two options:

- **(cheap, self-contained)** a per-line `uint8` bitmask `d_line_is_nonnlte[n_lines]`
  uploaded once (2.5 MB), set from `nlte->nlte_line_map[line] < 0` at init. This is the
  ground-truth "non-NLTE" predicate — a bb line's two levels share an ion, so
  "line has a non-NLTE level" ⟺ "line's ion ∉ NLTE set" ⟺ `nlte_line_map[line] < 0`.
- (or) an in-kernel `(Z,ion)` membership test against the small NLTE-ion list.

Decision: **before** `d_macro_atom_interaction` at :3205 (mirroring the eps_uv/eps_ir early
returns at :3167-3203), branch on `d_line_is_nonnlte[*next_line_id]` → call the existing
`d_ltherm_reemit(...)` (Planck(T_e), all shells) and return. This is a ~10-line insert plus
one bitmask upload. **Per-LINE, all-shells, physics-scoped** — the opposite of LTHERM's
per-SHELL/all-line blunt gate (`d_ltherm_on && shell_id ≤ d_ltherm_smax`, :3111 etc.).

### B.3 Energy conservation
`d_ltherm_reemit` redraws only the comoving frequency from Planck(T_e) and reindexes
`next_line_id`; the packet **energy is untouched** (the isotropic-scatter Doppler
bookkeeping at :3099-3107 runs before the branch, identical to every other exit). Energy is
conserved bin-to-bin exactly as the resonant/macro-atom exits are — this is a frequency
*redistribution*, matching the cs which also conserves the line's absorbed power into the
B(T_e) emissivity. No packet is created or destroyed.

### B.4 Fraction of the forest affected (measured)
`stage4_sizing/forkB_line_census.csv`:

| set | lines | % of 2.565M forest |
|---|---|---|
| NLTE-mapped (Fe II/III etc.) — **untouched** | 2,410,046 | 93.9% |
| **non-NLTE (max Fork-B scope)** | **155,296** | **6.1%** |
| — stage-IV Fe-peak funnel Fe/Co/Ni IV (min scope) | 12,576 | 0.490% |
| — +Cr/Ti IV | 18,185 | 0.709% |
| top non-NLTE by count: Ni I 52,578 · Si I 21,791 · S I 19,813 · C I 10,204 · Fe I 6,984 · Cr IV 4,609 · Fe V 4,558 · Fe IV 4,336 · Ni IV 4,199 · Co IV 4,041 | | |

The funnel is 0.49% of the forest yet 84% of deep-pile emission — so the **minimal scope**
(Fe/Co/Ni IV only, 12,576 lines) already targets the defect. The neutral ions (Ni I, Si I,
S I, C I) dominate the non-NLTE count but are cool-outer, not deep-funnel; scoping Fork B
to the stage-IV Fe-peak (or the full 6.1%) is a knob, not a rewrite.

### B.5 Side-effect profile vs the blunt LTHERM probe
LTHERM (`LUMINA_LINE_THERM`, per-shell s0..smax, **all** lines) had two damage modes; Fork
B avoids the first by construction and shrinks the second:

- **EUV kill — AVOIDED.** LTHERM thermalized the *NLTE* EUV emitters too (Fe III's S_l IS
  SE-coupled and matches CMFGEN — VERDICT step 6), replacing their EUV escape with the
  faint B(T_e) Wien tail. Fork B is per-line scoped to `nlte_line_map<0`, so it **never
  touches Fe II/III** — the SE EUV physics that the trapping audit found already correct
  (Lumina tau ≥ CMFGEN) is preserved. This is the main win.
- **deep f(IV) crater — MINIMIZED.** Both probes alter only MC *transport* (re-emission),
  not the plasma ionization solve, so neither directly moves f(IV). But LTHERM's all-line
  thermalization distorts the deep radiation field feeding photoionization far more
  broadly than Fork B's funnel-scoped redistribution — and Fork B pushes the non-NLTE line
  field toward the **same** B(T_e) the cs/photoion machinery already assumes, introducing
  no *new* inconsistency. Expected f(IV) perturbation: smaller and same-signed as the cs.

Both bullets are **expected/argued** and are exactly what the falsifier below tests.

---

## Recommendation (ranked, staged, with falsifier gates)

**Do Fork B first as the gate; Fork A is the fidelity endpoint — and they interact.**

Framed as CMFGEN divergence: the target is `mc_J → cs_J` in the deep FUV (the cs side
already reproduces CMFGEN's smooth ≈B(T_e) deep field; only the MC diverges, 39× at 1526Å).

### Stage 1 — Fork B (minimal scope: Fe/Co/Ni IV), the falsifier
- **Why first:** ~10-line insert + one 2.5 MB bitmask, reversible, zero VRAM/solve cost,
  reuses the existing Planck(T_e) sampler. It is the *direct* MC-side fix and a clean test
  of the VERDICT's central claim without touching the SE solver.
- **Falsifier gate (pre-registered):** with Fork B on, in deep shell s0, `mc_J/cs_J` at
  1526.17Å must collapse from **39× toward ~1**, and the 1490–1650Å pile share of Co IV
  emission must fall from 84% toward the CMFGEN smooth-continuum level; bolometric u(s0)
  should stay within its measured −0.24 dex (Fork B redistributes color, must not destroy
  energy). If mc_J/cs_J does **not** move, the funnel is not the source-function
  inconsistency and the VERDICT is wrong — stop and re-open.
- **Guard:** compare against the live `LUMINA_MACROATOM_BF` A/B — Fork B and bf-recomb are
  alternative thermal exits (VERDICT options 3 vs 2); run them as separate arms, not
  stacked, to attribute the pile collapse.

### Stage 2 — Fork A (Fe/Co/Ni IV triples), the fidelity endpoint
- **Why second:** it is the physically complete fix (SE populations + cs consistency +
  S_l written), cheap in VRAM (+0.5 MB, max_N unchanged, >70 GB headroom) but +15–20%
  solve and 9 wiring touch points including the `pair_lo_init` hi=lo+1 hazard (A.4 #4).
- **Interaction to respect:** Fork A moves Co/Fe/Ni IV lines *out* of Fork B's
  `nlte_line_map<0` scope. Once promoted, those lines are NLTE and Fork B no longer
  thermalizes them — but the MC macro-atom still lacks a thermal exit (A.5). So **Stage 2
  must ship with a thermal exit for the now-NLTE stage-IV lines**: either keep
  `LUMINA_MACROATOM_BF` (the CMFGEN recomb-cascade mechanism) on, or retain a residual
  Fork-B thermalization keyed on "NLTE line but deep & continuum-thick." Do **not** assume
  promotion alone closes the 39×.
- **Falsifier gate:** deep Co IV level populations must match the self-run CMFGEN SE within
  the campaign's n_e 1% / J_nu benchmark tolerance; deep ionization f(Co IV) must not
  crater; and the emergent FUV gradient must move toward the DDC15/CMFGEN truth (the
  campaign's F3-T color metric). If populations improve but mc_J still piles, confirms A.5
  — the exit channel, not the SE set, is the operative fix.

### Ranking rationale
1. **Fork B (min scope)** — highest information per unit risk; isolates the mechanism.
2. **Fork A + thermal exit** — the durable, CMFGEN-faithful endpoint.
3. Fork A alone — **not recommended** as a standalone fix: it corrects populations and the
   cs, but A.5 shows it does not break the MC funnel by itself.

## Artifacts
- `validation/cmfgen_toy06_19p48d/analysis/stage4_sizing/forkA_ion_sizing.csv`
- `validation/cmfgen_toy06_19p48d/analysis/stage4_sizing/forkA_aggregate_sizing.csv`
- `validation/cmfgen_toy06_19p48d/analysis/stage4_sizing/forkB_line_census.csv`
