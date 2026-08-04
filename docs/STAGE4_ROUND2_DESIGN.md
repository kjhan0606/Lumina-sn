# Stage-IV iron-peak NLTE promotion — ROUND 2 design (fix the round-1 blowup)

Offline design + sizing, 2026-07-19. No source edits, no runs, no commit. DESIGN-ONLY.
Round-1 lineage: `docs/STAGE4_NLTE_REPAIR_DESIGN.md` (Fork A/B decision). Round-1 run:
`logs/coevolve_consume_a10_kx_stage4/` (gate `LUMINA_NLTE_STAGE4`). Failure ledger:
`validation/cmfgen_toy06_19p48d/analysis/crime_reconstruction/` Part 2. Sizing artifacts
(this round): `validation/cmfgen_toy06_19p48d/analysis/stage4_round2/`.

Round 2's job is **NOT** the MC funnel — that is now owned by Fork B (`LUMINA_LINE_BSRC`,
validated tonight, mc/cs 39→1.90). Round 2's job is **correct stage-IV ionization and
populations WITHOUT the blowup, coexisting with Fork B.**

---

## 0. The ground truth (yardstick pinned first, per feedback_cmfgen_is_the_standard)

The Lumina grid is uniform in velocity, `v_inner[0]=3900`, `v_outer[49]=40300` km/s
(`stdout.log:106`), so `v_mid(s)=3900+(s+0.5)·728`. Interpolating the published
CMFGEN toy06 @19.48d ionfrac (`data/standart_data1/toy06/ionfrac_{fe,co,ni}_toy06_cmfgen.txt`)
onto the **Lumina** shell velocities (`stage4_round2/fiv_depth_crossover.csv`):

| shell | v (km/s) | W | CMFGEN f(IV) Fe / Co / Ni |
|---|---|---|---|
| s0 (deep) | 4264 | 0.298 | **0.982 / 0.993 / 0.978** |
| s8 (phot) | 10088 | 0.039 | **0.022 / 0.099 / 0.026** |

So the task's "deep f(IV)→0.99" gate **is** CMFGEN-correct at the Lumina inner boundary
(v~4264; it is only V-dominated far deeper than the model, v~1025). The photosphere
recombines to III. The **entire physical ask** is: lift the deep (s0-s2) f(IV) from the
LTE-underionized baseline toward ~0.98, while leaving the photosphere at its
already-nearly-correct value.

The LTE-all-level baseline is the completed `gphall` run (`g_gph_alllevel` on, no NLTE
weighting). Co f(IV) depth profile, CMFGEN vs LTE-baseline vs round-1 blowup:

| s | v | W | CMFGEN | LTE(gphall) | stage4(R1) | verdict |
|---|---|---|---|---|---|---|
| 0 | 4264 | 0.298 | 0.993 | 0.753 | 0.908 | **drain needed** |
| 1 | 4992 | 0.188 | 0.995 | 0.884 | 0.901 | **drain needed** |
| 2 | 5720 | 0.134 | 0.983 | 0.934 | 0.909 | **drain needed** |
| 3 | 6448 | 0.102 | 0.908 | 0.955 | 0.928 | LTE already OK |
| 4 | 7176 | 0.080 | 0.522 | 0.941 | 0.972 | LTE already over |
| 8 | 10088 | 0.039 | 0.099 | **0.083** | **0.992** | LTE already OK; R1 **blown up** |

**Decisive fact:** the LTE (gphall) Gph already reproduces CMFGEN at the photosphere
(Co s8 0.083 vs 0.099). The NLTE weighting is what **breaks** it (0.083→0.992). The
enhancement is warranted **only at s0–s2 (W≳0.13)** — where CMFGEN f(IV)>0.98 and LTE
under-ionizes — and is **actively harmful everywhere W<0.10.** This single observation
drives the round-2 recommendation.

---

## 1. Gph weighting staging

### 1.1 Mechanism recap (file:line)
The offending code is the SE-weighted all-level Gph for **drained III combs**, gated at
`src/lumina_plasma.c:5490-5498` (`want_nlte_w`): under `LUMINA_NLTE_STAGE4`, a III comb
(`ion_pop_stage==2`) whose IV is now an NLTE ion is population-weighted. The weight is the
**actual population fraction** `pop_l = n_l/n_ion_nlte` (`:5537-5538`, summed at
`:5506-5508`), fed into the all-level photoion integral `G += pop_l·w` (`:5573`,
`w = 4π σ_l J /(hν) dν`, `:5571`). Result committed to `sh.Gph[p]` (`:5685`), which the
ionization ladder consumes as `r = G/(n_e·α)` (`:4966`).

The comb `b_k` that `pop_l` carries is **raw and lagged**: within one outer iteration the
Gph build runs in `compute_radiative_equilibrium_te → radeq_simul_all`
(`plasma.c:6450→6460`) at `cuda.cu:5144`, **BEFORE** the III/IV drain `nlte_solve_all_gpu`
at `cuda.cu:~5159`. So Gph always consumes the previous iteration's un-drained pops.

### 1.2 Why the comb is super-thermal — and why it is NOT a lag artifact (measured)
`stage4_round2/gph_bk_distribution.csv` — sigma-bearing III-comb `b_k` in the **converged
final-iteration** dump:

| comb | s0 (deep) median / p90 / max | s8 (phot) median / p90 / max |
|---|---|---|
| Fe III | 57.8 / 548 / 5.3e3 | **1.8e7 / 3.0e8 / 3.2e9** |
| Co III | 192 / 1.4e3 / 7.7e3 | **2.4e9 / 2.7e10 / 1.7e11** |
| Ni III | **1.2e4 / 1.3e5 / 8.8e5** | 1.4e7 / 2.4e8 / 1.8e9 |

The s8 comb is **not** a slow-relaxation transient: this is the fully-drained SE solution,
and `gphall` (no promotion) already carries Fe III s8 median 1.8e5 (`part2_feiii_comb.csv`).
The photospheric super-thermal comb is the **standing dilute-field NLTE departure**
(`plasma.c:9838-9848`: `J_bar=W·B(T_R)≠B(T_e)` drives `b_k` away from 1; at s8 W=0.039,
T_R=10470 hot → extreme pumping). It is a pre-existing feature (cf. `super_thermal_sl`
campaign), **independent of promotion**; promotion's only sin is making Gph **consume** it.

Two corollaries that **falsify two of the four staging options outright**:
- **(a) convergence-gate is INSUFFICIENT** — the s8 comb is converged-super-thermal
  (final dump = 2.4e9), so "weight only after residual<eps / after N iters" still fires
  on 2.4e9.
- **(c) drain-first sequencing is INSUFFICIENT** — the drained (SE) s8 comb **is** 2.4e9.
  Reordering `nlte_solve_all_gpu` before `compute_plasma_state`, or a second Gph pass after
  the drain, does not lower it. (It is also a hazardous global reorder: `compute_plasma_state`
  supplies the ion densities `nlte_solve_all_gpu` reads.) Drain-first would only remove the
  1-iteration lag; the lag is not the disease.

### 1.3 The b_k cap (option b) — necessary, sized, but NOT sufficient alone
`stage4_round2/gph_cap_sizing.csv` reproduces the `plasma.c:5523-5627` all-level integral
(NLTE-weighted vs Boltzmann) under a per-level cap `b_l→min(b_l,C)`, with `J=W·B(T_R)`
(the run's dilute field, `plasma.c:9846`) and a hot `B(T_e)` bracket. Ratio = Gph_nlte/Gph_LTE:

| comb / shell | uncapped | C=1e4 | C=5e3 | C=1e3 | C=500 | C=100 |
|---|---|---|---|---|---|---|
| Co III s0 (hot T_e field) | **26.7** | 26.7 | 26.7 | 26.1 | 24.3 | 13.0 |
| Co III s8 | **6.6e6** | 378 | 272 | 117 | 72 | 19 |
| Fe III s8 | 5.3e5 | — | 123 | 48 | 40 | 18 |
| Ni III s0 (hot) | **5.4e3** | 1224 | 704 | 183 | 98 | 23 |
| Ni III s8 | 1.3e6 | 724 | 507 | 168 | 96 | 23 |

Two findings:
1. **s0 anchor is field-dependent but ~CMFGEN-implied:** the hot deep field (T_e=13115K)
   gives Co III s0 = **26.7×** — matching the ADDENDUM's "22× deficit" target
   (`plasma.c:5479-5488`) within field-color uncertainty. The cap is a **no-op at s0**
   down to C≈1000 (s0 comb max is 7657 < C), so it preserves the physical deep drain.
2. **The cap collapses the s8 catastrophe by 4–5 orders (6.6e6→O(100)) but cannot reach
   the physical s8 value (~1×).** Even C=5000 leaves Co III s8 at 272× — because at s8 the
   comb median (2.4e9) ≫ C, so *nearly every* sigma-bearing level is pinned at C and the
   weighting degenerates to "uniform-at-C", which still over-weights the low-threshold
   excited levels vs LTE. The task's own gates are internally inconsistent here: "Gph(Co III)
   s8 within 5–50× of LTE" permits 50×, but "Co f(IV) s8 in 0.05–0.15" (CMFGEN 0.099)
   requires the s8 enhancement to be **~1–2×** (LTE is already right). The binding gate is
   f(IV); a cap that lands s8 at 50× **fails** the f(IV) gate.

**Conclusion:** the cap is *necessary* (it is the only bound on the pathological
super-thermal comb, and Ni III is super-thermal even at s0 — 1.2e4 — so a cap is required
just to keep the deep solve sane), but *not sufficient*. It cannot, by construction, give
the correct **depth dependence** (large deep, ~1 at the photosphere).

### 1.4 RECOMMENDATION — depth-gate the weighting, cap within the gate (option "e")
The physically-correct enhancement is depth-dependent and the depth map is **known**
(§0): the drain is warranted **only where the shell is deep/continuum-thick and CMFGEN
f(IV)>LTE** — s0–s2, W≳0.13. Everywhere W<0.10 the LTE Gph already meets or exceeds CMFGEN.
So:

> **Gate the `want_nlte_w` NLTE weighting on a depth/dilution threshold** (enable only
> where `W(s) > W_thr`, W_thr≈0.13, i.e. s0–s2), **AND cap `b_l→min(b_l, C_cap)` with
> C_cap≈1000 inside the gated shells.**

- New env, both defaulting to the round-1 behavior when unset: `LUMINA_GPH_NLTE_WMIN`
  (depth gate; 0 ⇒ all shells = round-1) and `LUMINA_GPH_BK_CAP` (per-level cap; 0/∞ ⇒
  round-1). Both read once, cached, at the top of the `want_nlte_w` block (`plasma.c:5490`),
  and applied as: skip the NLTE-weight branch when `W[s] ≤ WMIN` (fall through to the
  existing Boltzmann path `:5581`, which reproduces CMFGEN at those shells); clamp
  `pop_l` construction to `min(b_l,C_cap)·n_l^LTE` (n_l^LTE = n_k/b_k available per level).
- **Why the depth gate is the load-bearing piece and the cap the safety net:** the gate
  gives the right *shape* (deep-only), which the cap alone cannot; the cap bounds the
  residual within s0–s2 (chiefly the pathological Ni III comb) and removes sensitivity to
  the exact WMIN choice near the transition.
- **Physical trigger note:** W is a clean, monotone, already-available depth proxy. A local
  Rosseland/continuum optical-depth threshold (continuum-thick) is more physical if a
  per-shell τ_cont is exposed; W_thr≈0.13 is the operational equivalent from
  `fiv_depth_crossover.csv`.

**Pre-registered prediction (Co III, the task's named target):**
- s0–s2: enhancement ~27× (physical, cap no-op) → Co f(IV) 0.75/0.88/0.93 → **~0.98–0.99**
  (CMFGEN 0.99/0.99/0.98). Cannot overshoot (f(IV) saturates at 1).
- s8: gate OFF → LTE → Co f(IV) → **~0.083** (CMFGEN 0.099); Gph(Co III) s8 enhancement
  **~1×**, NOT 1885×.
- **Yardstick correction (flag for the run card):** the task's "Gph(Co III) s8 within 5–50×"
  gate is mis-specified against CMFGEN; s8 should be ~1× (LTE). Recommend re-registering
  that gate as "s8 enhancement ≤3× (LTE-like)".

---

## 2. Ni closure (runaway to V)

### 2.1 Root cause (measured)
Ni ion_pop carries a bare stage V (`ion_number 4`) — dataset max Ni level ion = IV
(`levels.csv`: Ni has ion 0/1/2/3 only), but `ionization_energies.csv` has `28,3`=54.92 eV
so `radeq_simul`'s `simul_ladder` (`plasma.c:4951-4988`) builds a V rung with **no levels,
no σ_bf, no NLTE, no proper closure.** Round-1 Ni s0: III=6.68, IV=298, **V=1.74e8**
(`lumina_ion_pops.csv`), f(IV)=1.7e-6.

The ladder is a product chain `y[j+1]=y[j]·r_j`, `r=(Gph+γ_nt)/(n_e·α)` (`:4965-4969`). The
runaway is **caused by the round-1 weighting**, not by the bare V rung per se:
- The Ni III comb is super-thermal **even at s0** (median b_k=1.2e4, `gph_bk_distribution.csv`)
  — 60× worse than Fe/Co — so its NLTE weighting over-drives Gph(Ni III→IV) ~1.5e4× at
  *every* depth (`gph_cap_sizing.csv`, Ni III s0 hot=5.4e3×). That pushes the whole Ni
  reservoir up the ladder; the bare IV→V rung (no recombination floor because α(V→IV) is
  weak/absent for the level-less V) then takes it the rest of the way to V.
- **Proof it is weighting-induced:** in `gphall` (no NLTE weighting) Ni s0 f(V)=3.7e-4 —
  no runaway. Co, whose III comb is only 192× super-thermal, holds at IV in round-1
  (f(V)=0.003) despite the **same** bare-V rung. Ni is singled out purely by its pathological
  comb.

### 2.2 RECOMMENDATION — the §1 fix is primary; add a hard IV top-ion clamp as safety
1. **Primary:** the depth-gate + cap of §1.4 removes the Ni III over-drive (caps the 1.2e4
   comb at 1000, gates it to s0–s2) → the ladder self-limits at IV as it did in `gphall`.
   This is the single-mechanism cure and should be tried first (no new Ni-specific code).
2. **Safety net (recommended to ship together):** for the ions whose dataset top real ion is
   IV (Co, Ni — no V levels; verified from `levels.csv`), **truncate the `simul_ladder` at IV**:
   force `r(IV→V)=0` for those elements (a "top-ion" Saha closure — treat IV as the highest
   populated stage, no bf drain past it). This mirrors the NLTE-side top-stage detection
   `hi_is_topstage` (`plasma.c:9627-9632`, "no NLTE ion is (same Z, ion_hi+1)"). Concretely:
   in the `simul_ladder` element loop, when `atom` has no levels for stage `j+1` of this Z
   (`level_offset[ip+1]==level_offset[ip+2]`, i.e. a level-less rung), set that step's
   `y[j+1]=0`. Gate `LUMINA_SIMUL_CAP_TOPION` (default off = round-1). Cost: ~5 lines,
   zero VRAM.
3. **Rejected: import Ni V/VI from CMFGEN.** Verified NOT available in this dataset —
   `levels.csv` has no Ni ion≥4, and the tardis reference tree stops at Ni IV (200 levels).
   Adding V would be a data-pipeline change (levels+σ_bf+lines+recomb) far larger than the
   defect warrants. Fe **does** have a real V (200 levels) and needs no clamp — its ladder
   is physical through V.
4. **Rejected: freeze Ni IV/V to nebular** — introduces a hand-tuned ratio (violates
   NO-OVERFITTING); the top-ion clamp is parameter-free.

**Pre-registered prediction:** Ni f(IV) s0 → **~0.98** (CMFGEN 0.978), s8 → ~0.026
(CMFGEN 0.026); f(V) < 0.05 at all shells.

---

## 3. Ti singular matrices (info=199)

### 3.1 Diagnosis (measured — it is a data-topology defect, not a plasma/numeric issue)
`stderr.log`: `[NLTE-FALLBACK] GPU pair (Z=22, ions 2/3, N=202) shell=0..49 … info=199 →
Boltzmann@T_rad`, **identical at all 50 shells and every iteration**. A shell- and
iteration-independent singularity at a *fixed* pivot (199 of 202) is the signature of a
structural **zero row** — a level with no rate in or out — not conditioning.

Confirmed: Ti IV (`22,3`, 126 levels — `stdout.log:201`) has **4 radiatively-isolated
levels (level_number 19, 23, 36, 83) that appear in ZERO bb lines** (`line_list.csv` scan,
122/126 levels connected). An isolated level with no bb line and (if) no σ_bf has an
all-zero rate-matrix row → singular getrf. The existing collisional floor
`LUMINA_NLTE_COLL_FLOOR` (`plasma.c:9647-9655`) floors `C_up` at `ε·A_ul` — but A_ul is
undefined for a level with **no line**, so it cannot rescue these; and
`LUMINA_NLTE_FLOOR_REG` (`:9600-9604`) only tracks bb-connectivity, it does not pin
unconnected levels.

### 3.2 RECOMMENDATION
1. **Primary (zero-risk): drop Ti from the stage-IV promotion set.** Ti IV is flagged
   optional in round-1 (`STAGE4_NLTE_REPAIR_DESIGN.md` A.1) and is a minor coolant. Remove
   the three Ti slots (`20,21,22`) from `NLTE_TARGET_{Z,ION}4` (`plasma.c:4005-4012`), or
   for the tau side keep `LUMINA_NLTE_SKIP_Z=22` (`cuda.cu:1493-1500`). This is the
   recommended round-2 default — it removes the only fallback-spamming pair and de-risks the
   run.
2. **General fix (if Ti is wanted later): a zero-row guard.** Before the batched getrf,
   detect rows with `|diag|+Σ|offdiag| < tiny` (an isolated level) and pin them to Boltzmann
   closure (diagonal=1, RHS = LTE population), i.e. extend `LUMINA_NLTE_FLOOR_REG` to cover
   *bb-unconnected* levels, not just weakly-connected ones. This makes the matrix
   non-singular and sends the 4 dead levels to their (unpopulated) Boltzmann value —
   physically harmless. Gate `LUMINA_NLTE_ZEROROW_REG` (default off).

---

## 4. Interaction with Fork B (BSRC)

### 4.1 The two mechanisms and their consumers (file:line)
- **Fork B (`LUMINA_LINE_BSRC`)** builds a static per-line mask over **Fe/Co/Ni IV species**
  (`cuda.cu:4766-4774`, `cuda_bsrc_build_mask`) and, in transport, re-emits a flagged line's
  photon from **Planck(T_e[shell])** (`d_bsrc_reemit`→`d_ltherm_reemit`,
  `cuda.cu:2870-2889`; sampler `d_sample_planck_frequency(d_ltherm_te=T_e)`, `:2568,:2774`).
  Mode 1 = full thermal, mode 2 = redshift-only guard (`:2878-2885`). Consumer = **MC
  transport** (frequency redistribution; energy untouched).
- **Fork A** promotion makes those same lines NLTE-mapped, so the S_l writer fires
  (`cuda.cu:1539-1568`: `S_l=(2hν³/c²)/(g_u n_l/g_l n_u −1)`). Consumer = **cs formal solve
  + jbar** (`line_source_S` is a host array, **never uploaded to the transport kernel** —
  round-1 doc B.1). The MC does **not** read S_l.

So with both gates on there is **no double-count and no conflict**: cs consumes the SE S_l,
MC consumes Planck(T_e) via BSRC. The mask is by *species*, not by `nlte_line_map`, so it
keeps thermalizing the promoted lines automatically.

### 4.2 Which source should the MC sample? — keep Planck(T_e); do NOT add a mode-3 (S_l)
The question is whether to add "BSRC mode 3: sample S_l-consistent frequency" for the
now-NLTE stage-IV lines. **Recommendation: no — MC keeps sampling Planck(T_e) (mode 1).**

Argument from CMFGEN-fidelity, using round-1's own pops (`part2_bk_coiv.csv`):
- **Deep, S_l→B(T_e) already** — Co IV b_k(144)=**0.888** at s0 (near unity). For a line
  with b_lo≈b_up, `S_l/B = (e^{hν/kTe}−1)/((b_lo/b_up)e^{hν/kTe}−1) → 1`. So where the pops
  are trustworthy (deep), **mode 1 and a hypothetical mode 3 coincide** — B(T_e) is both
  the cs target and the CMFGEN-thermalized deep-field truth (VERDICT: CMFGEN smooths this
  forest to ≈B). Mode 3 buys nothing deep.
- **Photosphere, S_l is an artifact** — Co IV b_k(144)=**208**, b_k(50)=**461** at s8
  (super-thermal, the same dilute-field departure of §1.2). A mode-3 sampler would inject
  that super-thermal S_l straight back into the MC = **re-open the funnel that Fork B was
  built to kill** (mc/cs 39→1.90). Mode 3 is not merely unnecessary, it is
  **counter-productive** exactly where BSRC earns its keep.
- **Robustness:** Planck(T_e) is parameter-free, always finite, and self-consistent with the
  cs *where the cs is right*. Sampling S_l couples MC transport to the least-trustworthy
  (near-inversion) pops; note the codebase already had to bolt a near-inversion guard onto
  the *cs* S_l (`LUMINA_DETFLUOR_SL_CEIL`, default 10× B(T_e), `cuda.cu:4280-4287`) —
  evidence that raw S_l is unsafe to sample.

### 4.3 Should BSRC scope shrink as promotion matures? — No; keep it, optionally depth-scope
Promotion does **not** add an MC thermal exit (round-1 A.5: the macro-atom funnel geometry
and the ~8e-10 k-packet exit are set by Co IV *atomic energies*, unchanged by SE pops). So
**BSRC must remain the MC thermal exit for the promoted stage-IV lines** — its scope must
**not** shrink to exclude them. Two refinements:
- Keep the species mask (Fe/Co/Ni IV) as-is; the coexistence is orthogonal by consumer.
- *Optional* optimization: depth-scope BSRC to the continuum-thick shells where the funnel
  lives (same W≳0.13 region as §1), since the thermalization target B(T_e) is exact there
  and the dilute outer shells re-emit little into the 1490–1650Å pile. This is a tuning
  knob, not required for round 2.

**Consistency closure:** mc_J→cs_J holds *by construction* where pops are correct — deep,
cs uses S_l≈B(T_e) (b_k~0.89) and MC uses B(T_e) (BSRC); both →B. Where pops are wrong
(photosphere), B(T_e) is the safer common target and the cs S_l should be clamped by the
existing near-inversion guard, not sampled.

---

## 5. Validation card (pre-registered)

**Env (single driver's seat; a10_kx background per MEMORY campaign):**
```
LUMINA_NLTE_STAGE4=1
LUMINA_LINE_BSRC=1               LUMINA_LINE_BSRC_MODE=1     # keep Planck(T_e); no mode 3
LUMINA_GPH_NLTE_WMIN=0.13        # NEW: depth-gate the III-comb NLTE weighting (§1.4)
LUMINA_GPH_BK_CAP=1000           # NEW: per-level b_k cap inside the gate (§1.4)
LUMINA_SIMUL_CAP_TOPION=1        # NEW: clamp Co/Ni ladders at IV (level-less V) (§2.2)
LUMINA_NLTE_SKIP_Z=22            # drop Ti from promotion (§3.2) — or remove Ti slots
# a10_kx gates (unchanged, src is ours/uncommitted): GPH_JTABLE / TE_TABLE / TINNER_COLOR,
# g_gph_alllevel, GPH_SIGMA_CMFGEN, EVENT_LOG=1 (CAP128M per feedback_event_log_default)
```
Run as a single arm on `slurm` h200/h100 (full-NLTE needs 80GB; a40 excluded).
Do NOT stack against a second thermal-exit A/B (attribute cleanly).

**Pre-registered gates** (yardstick = CMFGEN at Lumina velocities, §0):

| # | quantity | gate | source of truth |
|---|---|---|---|
| G1 | deep Fe f(IV) s0 | ≥ 0.95 | CMFGEN 0.982 |
| G2 | **Fe f(IV) s8 (no blowup)** | **≤ 0.10** | CMFGEN 0.022; R1 was 0.978 |
| G3 | Co f(IV) s0 | ≥ 0.90 | CMFGEN 0.993 |
| G4 | **Co f(IV) s8** | **0.05 – 0.15** | CMFGEN 0.099; R1 was 0.992 |
| G5 | **Ni f(IV) s0 (no crater)** | ≥ 0.90 | CMFGEN 0.978; R1 was 1.7e-6 |
| G6 | Ni f(IV) s8 | 0.01 – 0.10 | CMFGEN 0.026 |
| G7 | Ni f(V) all shells | < 0.05 | R1 was 0.9999 |
| G8 | **Gph(Co III) enhancement s8** | **≤ 3× LTE** (RE-CALIBRATED from task's "5–50×", §1.3) | LTE already ≈CMFGEN |
| G9 | Gph(Co III) enhancement s0 | 5 – 50× LTE | ~27× physical (`gph_cap_sizing`) |
| G10 | Ti pair fallback spam | 0 `info=199` lines | §3 |
| G11 | **funnel stays dead (Fork B)** | mc_J/cs_J @1526Å s0 ≤ ~2 (≈1.90) | tonight's BSRC validation |
| G12 | energy conservation | u(s0) within measured −0.24 dex | BSRC redistributes color |

**Falsifiers / attribution:**
- If **G2/G4 fail** (photosphere still ionized) with the depth gate on → WMIN too low or the
  W proxy is wrong for the transition; inspect `fiv_depth_crossover.csv` region s3–s7.
- If **G5 fails** (Ni still craters) with cap+clamp on → the Ni III comb over-drive survives
  the cap; tighten C_cap or confirm `SIMUL_CAP_TOPION` fired (check the `simul_ladder` V rung).
- If **G1/G3/G5 pass but G11 regresses** (funnel re-piles) → a mode-3/S_l path leaked into
  transport, or BSRC scope shrank off the promoted lines (§4.3). Confirms round-1 A.5: SE
  pops are not the MC exit.
- If **G8 lands at ~1× but Co f(IV) s0 undershoots** → the deep enhancement is field-limited
  (hot-vs-dilute anchor, §1.3); the deep J color (EUV/FUV-starved per campaign) is the lever,
  not the weighting.

Report per `feedback_report_as_cmfgen_divergence` (quantity × location × magnitude × cause),
not a gate list.

---

## Artifacts (this round)
- `validation/cmfgen_toy06_19p48d/analysis/stage4_round2/gph_cap_sizing.py` + `gph_cap_sizing.csv`
  — Gph_nlte/Gph_LTE vs b_k cap, per comb, s0/s8, dilute+hot fields (§1.3).
- `.../stage4_round2/gph_bk_distribution.csv` — sigma-bearing III-comb b_k distributions (§1.2).
- `.../stage4_round2/fiv_depth_crossover.py` + `fiv_depth_crossover.csv` — CMFGEN vs LTE vs R1
  f(IV) depth profile + W, the depth-gate calibration (§0, §1.4).
- Failure ledger consumed: `crime_reconstruction/part2_{ionfrac,feiii_comb,bk_coiv}.csv`.
