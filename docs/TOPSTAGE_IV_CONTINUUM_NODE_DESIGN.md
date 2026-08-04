# Top-Ion (III) Super-Thermal Fix — Option A: IV Ground-Only Continuum Node

**Status:** design (pre-build). Triple-verified root cause (2026-06-16). Prior
synthetic-IV attempt FAILED — this doc encodes the codex-identified failure
modes so the rebuild avoids them.

---

## 1. Confirmed root cause (agent acef3606 + codex 019ecde3 + dump/code triple-check)

The highest tracked NLTE ion of every element is stage III (`ion=2`). It is the
`hi` of its `(II,III)` pair but is **never a `lo`** (there is no `(III,IV)`
pair). In `nlte_assemble_rate_matrix` the photoionization/recombination loop runs
`for (lev = 0; lev < n_lo_levels; lev++)` — **only the lower ion's levels** —
so the III **excited** levels receive **no R_bf (photoionization) and no R_rec
(Milne recombination) edge**. Their rate-matrix rows carry only bound-bound terms
within the III manifold + the conservation row → **rank-deficient block**.

Direct evidence (per-level departure dump `LUMINA_LEVELPOP_DUMP=1`, job 166450):
- Low levels (E<5eV) thermal: `b_k=1.0`. Super-thermal is **exclusively** the
  near-ionization-threshold high levels.
- **Smoking gun:** O III excited `n_k` is **bit-identical across all 343 levels
  incl. ground** (shell 10: all `= 71528.06`, CV=0.0000), constant across
  `g=1..459, E=0..55eV` → pure uniform, NOT `g·exp(-E/kT)` of any temperature.
- Non-top ions (O II CV=18, Si II CV=12) keep real structure → fine.
- Mechanism of the uniform value: identity-mode (`within_sl_frac=1`), the cuBLAS
  LU **"succeeds" (info=0)** on the rank-deficient block and returns a uniform
  particular solution; the `inv_ceil` Boltzmann-ceiling gate (`cuda.cu:547`)
  misses it because uniform ⇒ excited/ground = 1 < 1e4 ⇒ no fallback. O (Z=8)
  did NOT appear in `[NLTE-FALLBACK]` (only Z=16,6 did) → O III came from the
  "success" path.
- `b_up/b_lo` of the super-thermal optical lines == measured `S_l/B`
  (O II 7.9e4, O III 1.8e3, C III 2.4e3) → the flat departures ARE the spectrum
  defect → MC over-interacts ~47× → too-blue / featureless spectrum.

σ_bf coverage REFUTED as cause (falsifier 166454): E>20eV levels have
`has_sigma=1.00`, median 665–832 positive σ_bf bins. The defect is the missing
**(III,IV) pair**, not missing data.

The prior `LUMINA_TOPSTAGE_THERMALIZE` anchor (force Boltzmann@T_e on top-ion
levels) was a **clamp** version of this fix: it imposed `b≈1` and collapsed
C/S III, but pushed population into O II (whack-a-mole) and is not
detailed-balanced. **Discard it.** Option A restores the physics.

---

## 2. The fix (faithful, detailed-balance)

For each top ion III, add the next ion IV as a **ground-only continuum node** and
a `(III,IV)` rate pair. The existing Milne machinery at `plasma.c:6830-6973`
(photoionization R_bf out of each III level + spontaneous-recombination
`R_rec = n_star_ratio·I_rec`, `I_rec = ∫(4πσ/hν)(2hν³/c² + J)e^{-hν/kTe}dν`,
including the committed spontaneous term) then runs with `ion_idx_lo = III`,
`ion_idx_hi = IV-ground`. At `J=B` the Planck identity pins each III level to its
Saha-Boltzmann reference vs the IV ground → `b_up/b_lo → 1` → thermal `S_l`.
No-op in the hot interior (reduces to Saha). The IV node needs **only its
ground** — IV excited levels are unnecessary and would just move the ceiling up
one stage.

Why it is not a drain/clamp: recombination IN balances photoionization OUT by
construction (Milne detailed balance). The thermalization is **radiative**, which
is correct for the cold/thin outer ejecta — codex REFUTED the collisional-
ionization/3-body alternative (exponentially tiny at T_e~2000K, would need an
artificial boost = clamp in disguise).

---

## 3. codex's 5 implementation pitfalls (the prior attempt died on #1 and #5)

1. **NO 1e30 `n_star_ratio` cap.** The O III→O IV edge has χ~55eV → `n_star_ratio
   ~1e82`, which the `1e30` cap (`plasma.c:6946`) flattens → all high levels
   share the same artificial ratio → detailed balance destroyed. **Fix:** remove
   the cap on the top-ion path; compute the Saha/Milne ratio in **log space**.
   `1e82` is far below `DBL_MAX (~1.8e308)` so the true value is representable —
   a plain (un-capped) `double` is fine here; only add log-space rescaling if a
   future edge overflows. **Action:** gate the cap so it does NOT apply when
   `ion_idx_hi` is a synthetic IV node (or raise/remove it and verify no overflow
   via the rate-balance dump).
2. **Isolate to the top ion.** Do not touch the existing `(II,III)` machinery.
   Non-top ions are currently CORRECT (CV~12-18) and must stay byte-identical.
3. **Keep the IV node out of charge/n_e/spectrum bookkeeping** unless the
   ionization model explicitly carries stage IV. The node is a *rate reservoir*
   for the III↔continuum balance, not a new charge carrier. (See OPEN Q1.)
4. **The solved III shape must survive postprocessing.** The per-ion conservation
   rescale (ratio-preserving) is fine. But verify the new pair does not trip a
   merge/rescale that flattens it.
5. **`overlap-restore` must not overwrite the solved III block.** The C1 restore
   at `cuda.cu:705` (`saved_lo` memcpy) puts the shared `lo`-ion block back after
   the solve. For the O triplet the `(II,III)` and `(I,II)` pairs already overlap
   on O II. Adding `(III,IV)` makes III a `lo` in the new pair and a `hi` in the
   old `(II,III)` pair → **ordering hazard**: the `(II,III)` solve writes O III,
   then the `(III,IV)` solve must run AFTER and its restore must protect O III.
   The prior attempt's "overlap-restore discarded the solve" bug lived here.

---

## 4. Exact code changes (15 top ions: Si,Ca,Fe,S,Co,Ni,C,Mg,Ti,Cr,Al,Sc,V,Mn,O)

Adding one IV ground-node + one `(III,IV)` pair per element = **+15 ion slots,
+15 pairs**. Counts: `NLTE_MAX_IONS 31→46`, `NLTE_PAIR_COUNT 16→31`.

| Location | Current | Change |
|---|---|---|
| `src/lumina.h:258` | `NLTE_MAX_IONS 31` | `46` |
| `src/lumina.h:259` | `NLTE_PAIR_COUNT 16` | `31` |
| `src/lumina_plasma.c:2843` `NLTE_TARGET_Z[]` | 31 entries | append IV Z for each element (15 entries) |
| `src/lumina_plasma.c:2847` `NLTE_TARGET_ION[]` | 31 entries | append `3` ×15 |
| `src/lumina_cuda.cu:373` `pairs[][2]` | 16 pairs | append 15 `(III_slot, IV_slot)` pairs |
| `src/lumina_cuda.cu:377` `names[]` | 16 | append 15 names |
| `src/lumina_plasma.c:7704` `pairs[][2]` | 16 | mirror append |
| `src/lumina_nlte_gemm.cu:68` `NLTE_PAIR_LO[]` | 16 | append 15 III-slot lo-indices |
| `src/lumina_cuda.cu:~2980` `pair_lo_init` | — | mirror |

**Slot map (proposed):** keep slots 0-30 as-is; append IV grounds 31-45 in the
element order of the existing pairs, and pairs 16-30 = `(top_III_slot, IV_slot)`.
e.g. Si III is slot 1 → new pair `(1, 31)`; O III is slot 30 → new pair `(30, 45)`.

**IV ground level data:** Si(Z14) and Al(Z13) have real `ion3` levels in the h5;
**O(Z08), C(Z06), S(Z16), and most others have NO ion3** → the level loader must
synthesize a 1-level IV ground (g from CMFGEN ground term, E=0) OR the bf uses the
**Kramers fallback** (already in `plasma.c:6856-6859`). Verify `find_ioniz_energy`
returns χ(III→IV) for all 15.

---

## 5. Design questions — RESOLVED (codex 019ecdf3 + physics agent a92f21a8, 2026-06-16)

- **Q1 — n_IV reservoir / gauge. RESOLVED: FAITHFUL.** `R_rec = n_star_ratio·I_rec`
  is **independent of the IV ground's absolute population** (`n_star_ratio =
  n_e·deBroglie·(g_lev/2g_ion)·exp(χ_lev/kTe)` — no n_IV). The IV population only
  multiplies the recombination coefficient in the SE rows; at J=B detailed balance
  gives `n_III_level/n_IV_ground = n_star_ratio` → the within-III SHAPE (hence S_l)
  is gauge-free w.r.t. IV normalization. Physics agent VERIFIED the post-solve
  rescale `scale_hi = n_hi_total/sum_hi` (`cuda.cu:670-686`, fallback 613-624) is a
  **single uniform scale → preserves all within-III ratios** → the IV node only
  re-shapes within III, never re-ionizes (III block conserved to upstream Saha
  `n_III` via conservation rows `plasma.c:7314-7331`). **The arbitrary IV
  normalization cancels in S_l.**
  - **Sub-resolution (supersedes the doc's earlier OPEN-Q1):** the Saha solve
    **DOES** produce stage-IV density. Ion ladder length = `n_ioniz+1` per element
    (`atomic.c:756`; O → O I…O IX). `compute_electron_density` (`plasma.c:797-803`)
    already sums IV,V… into n_e. ⇒ **use the REAL `ion_number_density[ip_IV]` as the
    (III,IV) pair-total**, FLOORED to a small positive value (codex: a zero/​tiny
    conservation RHS forces the continuum variable to 0 → reintroduces singularity).
    Reserve a floating pseudo-total only for elements whose ladder genuinely lacks a
    IV slot. **NEW pitfall #6:** the IV node must stay **read-only** w.r.t. the
    charge ladder — do NOT write back into `ion_number_density[ip_IV]` (IV is
    already in n_e; a writeback double-counts on the next iteration).
- **Q2 — n_star cap. RESOLVED: gate OFF for synthetic (III,IV) pairs only**
  (per-pair flag), keep for existing `(II,III)`. For the IV pairs use
  log/common-factor scaling: factor `A_common(T_e,n_e,χ_ground)` out of
  `n_star_i = A_common·g_i·exp(-E_i/kT)/(2g_c)` and absorb it into the continuum
  unknown's normalization; the III shape needs only the relative `g_i·exp(-E_i/kT)`.
  (Plain uncapped `double` holds 1e82 < DBL_MAX, but log-assembly is the robust
  form if any edge overflows.) Re-verify `(II,III)` byte-identical with
  `LUMINA_NLTE_RATE_DUMP=1`.
- **Q3 — cold-shell limit. RESOLVED: let the (III,IV) solve RUN even in cold
  shells** — the committed spontaneous `2hν³/c²` term keeps `R_rec>0` as J→0, so
  the block is non-singular and need not be skipped. This is the honest fix
  (avoids choosing an analytic distribution at all). IF a skip is still needed
  (`n_III_total ≤ floor`), write normalized **Boltzmann@T_e** (codex) — better than
  the current uniform/T_rad fallback for the thick-line carriers that drive the
  spectrum — but **gate it** and check in the falsifier that cold-shell S_l does
  not overshoot the other way (too RED); thin-line-only metastable levels lean
  W·B(T_rad), so Boltzmann@T_e is a caveated default, not exact. Also confirm the
  `inv_ceil` gate (`cuda.cu:547`) no longer trips for III (non-singular now).
- **Q4 — ground-only IV. RESOLVED: acceptable, caveated.** Direct Milne
  recombination into each III level (per-level `n_star_ratio`, `plasma.c:6944-6962`)
  is the first-order cure (kills the flat CV=0 → `b_k→O(1)`). LOST (second-order):
  recombination cascades through IV **excited** levels + dielectronic recombination
  pattern (DR currently routes to III **ground only** via `dr_lookup`,
  `plasma.c:7066-7085`) → residual fine-structure bias in the III departure pattern.
  Acceptable for the spectrum goal; **verify `ΣR_rec ≈ ΣR_bf` per III level**
  (detailed balance) in the rate-balance dump before declaring the pattern correct.
- **Q5 — counts/fixed-size. RESOLVED: audit required.** 46 ions / 31 pairs: scan for
  literal `16`/`31`/old `NLTE_MAX_IONS`; **32-bit ion masks** (46 > uint32 bit
  capacity — needs uint64 if any bitmask-over-ions exists); constant-memory tables;
  mirrored host/device arrays updated in one TU but not another; cuBLAS batch
  sizing; GEMM `phot_offset`/`L_phot_total`/per-level bf-table lengths (the IV
  ground adds +1 to top-pair dimension). Add **static asserts** on all mirrored
  arrays + **runtime startup validation** (pair indices in range; each synthetic IV
  = exactly 1 level; each top-III lower level has a bf edge or Kramers fallback;
  `phot_offset+L_phot ≤ L_phot_total`; no duplicate final owner per block).
- **Block-ownership rule (codex Q4, generalizes pitfall #5):** order each element's
  pairs monotonically by stage; the **later `lo` solve owns** the final stored
  population of any ion that is `hi` in an earlier pair and `lo` in a later one. For
  O: `(O I,O II) → (O II,O III) → (O III,O IV)`; generalize `saved_lo`/restore so it
  does NOT restore stale O III after the `(O III,O IV)` solve. State final block
  ownership explicitly in code.

**Phasing (Q5-scope):** minimal first falsifier = **Si III + Fe III** (real IV
level data) → prove thermalization (S_l/B→O(1), T_e/n_e held) cheaply, then extend
to O/C/S (Kramers IV — primary UV science target) and the rest.

---

## 6. Verification plan (falsifier)

1. Build with the IV nodes; run no-anchor champion config + `LUMINA_LEVELPOP_DUMP`.
2. **PASS criteria:**
   - Top-ion III `n_k` CV: 0.0000 → structured (Boltzmann-like, b_k→O(1)).
   - `S_l/B` of thick (τ>30) III optical/UV lines: 1e3-1e5 → O(1).
   - **T_e/n_e remain within 0.5% of gold** (proves thermalize, not drain).
   - O II does NOT regrow (the anchor's whack-a-mole) — CV stays structured.
3. If PASS → `LUMINA_CMFGEN_THEN_MC=1` spectrum vs DDC15 gold + 6/10 baseline
   165138; expect features to form (no 47× over-interaction).
4. Re-verify the design with codex + physics agent BEFORE build (per user's
   conservative path 2026-06-16).

---

## 7. Prior-failure ledger (do not repeat)

- Synthetic O IV node (earlier session): defeated by (i) `1e30` cap flattening the
  high-χ Saha (pitfall #1), (ii) the `(II,III)` overlap-restore discarding the
  solve (pitfall #5) — codex 7621/7677/7698. Both are now explicitly guarded.
- `LUMINA_TOPSTAGE_THERMALIZE` Boltzmann anchor: clamp, not detailed-balance;
  collapsed C/S III but pushed population to O II. Discarded.
