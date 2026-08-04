# Photospheric EUV repair — two-pronged design (the last defect)

**Status:** DESIGN-ONLY decision document. No source edits, no runs. Every mechanism claim
is `file:line`. Benchmark = CMFGEN toy06 @19.48d at Lumina velocities.

**Scope forensics (required reading, already done):**
`validation/cmfgen_toy06_19p48d/analysis/photospheric_euv_source/VERDICT.md` and
`.../bistability_test/VERDICT.md`. Run under audit = `logs/coevolve_consume_a10_kx_kpr5/`
(env footer in `stdout.log`: `LUMINA_BF_OPACITY=1`, `LUMINA_KPKT_FB_MULTI=1`,
`LUMINA_KPEMISS_BSRC_TAU=0.13`, `BSRC_SRC=2`, `COOLGUARD=1`, `RADEQ_DB_FB=1`; no GPH_JTABLE).

---

## 1. Mechanism recap — with the transport detailed-balance (DB) framing

### 1.1 The two locally-generated photospheric EUV channels (established)
The photospheric ionizing EUV field is ~1e6× CMFGEN and is **100% locally regenerated**
(deep-leak = 0%, `photospheric_euv_source/VERDICT.md §1`). It over-ionizes Fe III:
`f(FeIV,s8)=0.982` vs CMFGEN `0.022`; `Gph(FeIII,s8)=27.37/s` vs CMFGEN `3.5e-5/s`
(≈ 39000× the `~7e-4/s` critical for `f(FeIV)=0.022`; `bistability_test/VERDICT.md §1`).
EUV creation splits into (energy-weighted, `euv_source.py Part 1`):

| channel | share of EUV creation | dominant driver | Gph it feeds | prong |
|---|---|---|---|---|
| **line-emit, cross-ion (S III) k-packet CDF** | ~59.5% line-emit, of which **82.9% cross-ion** ⇒ **~49% of total EUV** | 450–912 Å excited Fe III edges (95% line, S III 81%) | **excited** Gph ≈ 20.82/s (76% of 27.37) | **A** |
| **fb-recombination continuum** (kpkt-fb, type -3) | **40.5%** | 300–450 Å **ground** Fe III edge 404 Å (98.5% kpkt-fb) | **ground** Gph ≈ 6.56/s (24%) | **B** |
| legitimate same-ion radiative cascade | ~17.1% of lines ≈ 10% | — | — | untouched |

(Gph ground/excited split from `bistability_test/VERDICT.md §1`. The "excited" levels there
bind >13.6 eV, i.e. they photoionize **in** the EUV — not near-threshold optical; the optical
warm-loop hypothesis is refuted, `bistability §1`.)

### 1.2 Why the field builds up: a broken transport detailed balance
CMFGEN's EUV declines **1.42e6** from s0→s8 (τ_eff≈14, optically THICK and faint at the
photosphere). Ours declines only **~53×** — under-absorbed (`photospheric_euv_source/VERDICT.md §5`).

This is the **transport analog of the radeq DBFB fix**. The radeq ledger's `[DBFB]`
(`lumina_plasma.c:4973–5092`) makes bf **cooling** `C_fb(T)` the exact `emit_nu`/Wien partner of
bf **heating** `H_photo`, so the plasma-side net `nion·(Hex − C_fb)` cancels bin-by-bin when
`J=B_nu^Wien(T)`. The **radiation-transport** side has the mirror pair:

- **fb EMISSION** (the kpkt-fb channel, type -3) — energy leaves the electron pool as a
  recombination-continuum photon. Its per-edge weight is
  `w = n_e · n_ion · α · (hν0 + kTe)` (`lumina_plasma.c:2337–2338`), where `n_ion` is the
  **recombining (upper) ion** — Fe **IV**, which is over-abundant — and `ν0` is the **ground**
  ionization threshold of the product ion (`find_ioniz_energy(Z, stage−1)`,
  `lumina_plasma.c:2327–2329`) = the 404 Å Fe III **ground** edge.
- **bf ABSORPTION** (the matching opacity) — `chi_bf = Σ_l n_level · σ_bf`
  (`lumina_plasma.c:3784–3786`), where `n_level` for the 404 Å edge is the Fe III **ground**
  population — which the runaway has **depleted** (`f(FeIII,s8)=0.018`).

In LTE/Milne detailed balance these cancel (`S_bf → B_nu`, no net field). Out of balance the
ratio of emission to re-absorption at the Fe III edge is set by
`n(FeIV)/n(FeIII) = r34(s8) = 591.7` (`bistability_test/validation_roundtrip.csv`), versus
CMFGEN's Saha value `≈0.022/0.97 ≈ 0.023` — an emission overshoot of order **~2.6e4×**, the same
order as the EUV-decline gap (`1.42e6/53 ≈ 2.7e4`) and the Gph excess (`3.9e4`). Confirming this
is over-emission, `stdout.log` shows `p_fb → 0.99` at iter 0 (`[FB-MULTI] p_fb s0: 9.928e-01`):
**≈99% of thermalized k-packets exit as a ground-edge recombination photon**, and those photons
free-stream because the depleted Fe III ground provides negligible re-absorption opacity.

**One sentence:** the fb recombination continuum is emitted at the actual (Saha-super)
recombination rate `∝ n(FeIV)`, but the matching bf photoabsorption `∝ n(FeIII_ground)` is
self-consistently collapsed, so the EUV free-streams instead of being re-absorbed on the spot —
CMFGEN closes this loop with case-B/on-the-spot re-absorption of ground-edge recombination
photons; the LUMINA transport does not.

---

## 2. Prong A — extend the B3 thermal k-packet exit to the photosphere

### 2.1 Current wiring
The k-packet macro-atom exit ladder (`lumina_cuda.cu:3278–3341`) fires, in order,
`-2` ff (`:3297`), `-3` fb (`:3302`), then the `[KPR B3]` `-4` `B(Te)` exit (`:3314–3320`). The
`-4` exit intercepts **only** the resonant CDF (line) re-excitation that would otherwise follow
(`:3323–3339`); it does **not** touch the `-2`/`-3` continuum branches (they `return` first). The
`-4` exit is gated by the per-shell mask `d_kpr_qualify[shell]` (`:3315`), built host-side as
`q = (plasma.W[s] > kpr_bsrc_tau)` with `kpr_bsrc_tau=0.13` (`lumina_cuda.cu:5639`). With the
photosphere at `W(s6–9)=0.054–0.034 < 0.13` (plasma_state.csv), s6–9 are **disqualified** and run
the CDF → the S III cross-ion attractor (`photospheric_euv_source/VERDICT.md §3`; iter-1 log
`bteq_exits=0 cdf_exits=59M`). COOLGUARD does not rescue this: it zeroes `q` only where
`f(Fe stage≥V)>0.5` (`lumina_cuda.cu:5626–5640`), and the photosphere is Fe **IV**-dominant, so
COOLGUARD never fires there.

### 2.2 Options and the recommended variant
- **(A1) lower/remove the W threshold** so photospheric k-packets also take the `-4` exit.
- (A2) a separate photospheric exit — same effect, more code; rejected as redundant.

**Frequency source at the photosphere — B(Te) [SRC=1] vs chi-weighted forest [SRC=2].**
The deep tier uses `SRC=2` = draw `ν` from the `chi_line(ν)·B_nu(Te)` forest
(`d_kpr_chi_sample`, `lumina_cuda.cu:3091–3104`; host CDF `:5658–5692`) to preserve the emergent
thermal continuum's line-blanketing **color** (the FUV gains). At the photosphere that choice is
**risky**: `SRC=2` re-seeds `next_line_id` into the forest, and the forest there contains the S III
resonance lines — the very attractor we are trying to kill. Quantitatively `SRC=2` is nearly
harmless in the EUV anyway (at `Te=12208 K`, `x=hν/kTe≈29` at 404 Å ⇒ `B_nu` Wien-suppressed
`~2.5e-13`, and there are no strong lines at 404 Å to lift `chi_line·B_nu` there), so `SRC=2_phot`
≈ `SRC=1_phot` **in the EUV**; the only difference is the residual line-forest re-seeding risk.

> **Recommended A variant: pure Planck `B(Te)` (SRC=1) for the new photospheric tier; keep SRC=2
> for the deep tier (W>0.13, unchanged).** Rationale: at the photosphere the goal is to remove
> non-thermal EUV, not to preserve continuum color; `B(Te)` is Wien-dead in the EUV
> (`B(404 Å,12 kK) ~ 2.5e-13` of peak) and exits as a clean thermal r-packet with **no** line-forest
> re-excitation (directly answering the forensic's SRC=2 concern and the original C5
> "protect legitimate non-thermal EUV" worry — which the forensic proved misplaced *here*, since
> the photospheric non-thermal EUV IS the S III cross-ion pathology).

**EUV removed by A:** the `-4` exit replaces the k-packet **CDF** exit (the cross-ion S III path;
91% of photospheric k-packet exits are line-CDF, `euv_source.py Part 4`). It removes the cross-ion
line-EUV = `0.829 × 0.595 ≈ **49% of total EUV creation**` and the **excited** Gph channel
(~20.82/s, 76% of Gph). It does **not** touch (i) the ~10% same-ion genuine radiative cascade
(that is a deterministic downbranch, not a k-packet exit), nor (ii) the fb continuum (Prong B).

**Caveat (pre-registered, not a defect of A):** `Te(s8)=12208 K` is +1.9 kK over CMFGEN 10383 K
(`T_rad` pinned 10470 all shells). `B(Te)` therefore leaves a *residual* thermal EUV — but at
+1.9 kK the Wien tail at 404 Å is still `~1e-13`, negligible. A is EUV-effective; it does not by
itself fix the +1.9 kK, which is a downstream consequence of the field, expected to relax once
A+B cut the field (bistability §4: "Fix = the FIELD").

**A alone is insufficient** (as the forensic states): after A, the ground fb Gph (6.56/s) alone is
still `9400×` critical ⇒ `f(FeIV)` stays ≈0.98. A must be composed with B.

---

## 3. Prong B — the fb / bf detailed-balance fix (the core, higher leverage)

### 3.1 THE load-bearing determination: is bf ABSORPTION opacity present in transport? **YES.**
The mission's central question — do the emitted recombination photons ever see bf photoabsorption
opacity in transport — resolves **affirmatively** from source:

- `chi_bf` is looked up **every transport step** at the packet's comoving frequency and added to
  the continuum opacity: `chi_bf_val = d_bf_get_chi(...); chi_continuum = chi_e + chi_bf_val`
  (`lumina_cuda.cu:4122–4128`), and a continuum interaction branches to a **bf macro-atom
  absorption** event with probability `1 − chi_e/chi_continuum` (`lumina_cuda.cu:4263–4304`).
  The same lookup guards the virtual-packet and `d_trace_packet` opacity
  (`:3846–3919`, `:4254–4255`).
- The grid spans `[NLTE_NU_MIN, NLTE_NU_MAX] = [1.5e14, 3.0e16] Hz = [20000, 100 Å]`
  (`lumina.h:329–330`; `bf_opacity_init`, `lumina_plasma.c:3433–3435`). **404 Å = 7.42e15 Hz and
  the whole 300–912 Å band (3.29e15–9.99e15 Hz) are in-grid** — `d_bf_get_chi` does **not** return
  zero out-of-bounds there (`lumina_cuda.cu:2684`).
- It is **enabled in this run**: `stdout.log` line 80 `LUMINA_BF_OPACITY=1`, line 167
  `BF+FF opacity: ENABLED`, `:5226–5231`.

**Therefore the defect is NOT missing bf opacity, and the mission's primary framing ("add chi_bf to
transport") is already satisfied. The defect is the DB asymmetry of §1.2:** `chi_bf(404 Å) =
n(FeIII_ground)·σ_bf` is population-weighted (`lumina_plasma.c:3713–3786`, `n_level·σ`) and the
runaway has collapsed `n(FeIII_ground)`; meanwhile the kpkt-fb channel **emits** at the actual
`∝ n(FeIV)` recombination rate. There is also **no ordering bug**: after the `-3` emission the loop
returns to the top and recomputes `chi_bf` at the new frequency (`:4117–4128`), so the photon *is*
subject to the (depleted) edge opacity — it simply free-streams because the opacity is ~590× too
small relative to the emission. `eta_bf` (the BF-MILNE emissivity) is **host-only** — there is no
`d_eta_bf` and no `bf_get_eta` in `lumina_cuda.cu` (grep: 0 hits) — so the **only** transport
bf-emission channel is kpkt-fb (type -3); no second emission source is hiding.

> **"Apply physical `σ_bf·n_lower`" is a no-op here:** that is exactly what `chi_bf` already is,
> and `n_lower` (Fe III ground) is genuinely depleted. Raising it artificially (e.g. to LTE) would
> be an unphysical patch (violates `feedback_no_unphysical_patches`). The physical fix is on the
> **emission** side: enforce case-B / on-the-spot re-absorption of the ground-edge recomb photon.

### 3.2 The fix — transport-side OTS (case-B) for EUV ground-edge fb emission
Every FB-MULTI edge is a **ground-state** recombination edge (`ν0 = find_ioniz_energy(Z, stage−1)`,
`lumina_plasma.c:2327–2329`), and each carries a `Z*100+stage` code (`kpacket_fb_edge_zstage`,
`:2341`) available at the emission site as `zsel` (`lumina_cuda.cu:3660`). In an optically-thick
zone, a ground-edge recombination photon is re-absorbed locally (case B / on-the-spot) and never
joins the ionizing field — this is standard nebular physics and is already the code's own
convention for the *emissivity* via `LUMINA_CMF_OTS` (`lumina_plasma.c:3547–3559`, which **excludes
ground-edge recombination emission from `eta_bf`**). The transport `-3` channel does not honor it.

> **Recommended B fix — redirect EUV ground-edge fb to `B(Te)` in thick shells (the transport DBFB
> partner of A).** At the `-3` emission site (`lumina_cuda.cu:3641–3698`), when
> (i) the OTS gate is on, (ii) the shell is in the thick tier (same mask as A, §4), and
> (iii) the drawn edge is in the EUV (`ν0 > ν(912 Å)=3.29e15 Hz`), **replace** the raw
> `nu_edge`-based emission (`:3666–3675`) with the `-4` `B(Te)` Planck draw (`:3711–3718`). Energy
> is conserved (the packet keeps `pkt_energy`); only the **color** is thermalized (Wien-dead in the
> EUV). This is the MC-faithful realization of "the ground-edge recombination photon is re-absorbed
> and its energy re-emerges thermally through the cascade": it removes the 40.5% fb-continuum EUV
> and the **ground** Gph channel (6.56/s) at the 404 Å edge.

**Energy / detailed-balance argument.** The redirect does not change the plasma energy ledger (that
is `[DBFB]`, `simul_r1`, host-side); it changes only the radiation-field color. In the thick limit
the emergent recombination continuum *is* thermal (`S_bf→B_nu`), so emitting at `B(Te)` is the
correct optically-thick source function — precisely the closure A applies to the line channel. The
gate is restricted to the EUV (`ν0 > ν_912`) so **FUV ground edges (>912 Å) are left alone**,
protecting the deep FUV gains; and to the thick tier so the genuinely **thin far-outer nebula**,
where recombination EUV legitimately escapes, keeps the raw fb edge.

**Alternative framings considered and rejected:**
- *Route the fb energy back into the k-packet pool* (collisional re-excite → line/ff): more
  ARTIS-literal, but re-feeds the line cascade (the S III attractor) and, with A active, would be
  re-thermalized anyway — circular and riskier. Rejected.
- *Force an immediate bf scatter with the physical `σ_bf`* (bypassing the depleted global
  `chi_bf`): most DB-literal but requires a per-edge `σ_bf(ν0)` at the emission site and an extra
  RNG-consuming interaction; the `B(Te)` redirect achieves the same emergent result (no
  free-streaming EUV) far more simply. Hold as a fallback if the redirect under-absorbs.

### 3.3 Distinguishing over-emission vs under-absorption (numbers)
Both are true and are two faces of one non-equilibrium ionization: emission `∝ n(FeIV)` is
Saha-super-abundant (`p_fb≈0.99`), absorption `∝ n(FeIII)` is Saha-sub-abundant; their ratio is
`r34=592` at s8 vs Saha ≈0.023. The B(Te) redirect attacks the **emission** side (the actionable,
physical lever — the absorption side is correct physics fed by a wrong population, which A+B fix
by letting Fe III recover, see §5.1). Emit-rate ≫ absorb-rate at 404 Å ⇒ the imbalance is real
over-emission of an EUV photon that should have been case-B suppressed.

---

## 4. Composed gate spec (env-gated, default OFF, byte-identical when off, separately stageable)

All gates live under the existing `LUMINA_KPEMISS_REPAIR=1` master (`lumina_cuda.cu:5086`). Default
OFF ⇒ the new branches are never entered ⇒ the RNG stream and every result are **byte-identical**
to kpr5 (mirrors the proven `d_kpr_bsrc_on=0 ⇒ CDF byte-identical` pattern, `:3307–3322`).

### Prong A — photospheric B3 tier
| env var | default | effect |
|---|---|---|
| `LUMINA_KPEMISS_BSRC_PHOT` | `0` (off) | when 1, the qualify build (`lumina_cuda.cu:5639`) also sets `q=1` for shells with `W > BSRC_WFLOOR` (the phot tier), in addition to the deep `W>BSRC_TAU` tier |
| `LUMINA_KPEMISS_BSRC_WFLOOR` | `0.02` | phot-tier lower bound; captures the EUV-thick photosphere (s≈3–10; `W(s10)=0.029`) and **excludes** the thin far-outer nebula (`W<0.02`) so legitimate escaping nebular EUV keeps the CDF |
| `LUMINA_KPEMISS_BSRC_PHOT_SRC` | `1` (Planck) | frequency source for the **phot** tier only; deep tier keeps `BSRC_SRC` (=2). Implemented as a per-shell src array `d_kpr_bsrc_src_sh[256]` (deep=2, phot=1), or equivalently a second qualify mask `d_kpr_qualify_phot` routed to the SRC=1 path at `:3707–3718` |

Mechanics: the `-4` exit at `:3314–3320` already fires wherever `d_kpr_qualify[shell]=1`; extending
the mask to the phot tier is a host-only change (`:5618–5644`). Deep shells (W>0.13) are untouched.

### Prong B — EUV ground-edge fb OTS
| env var | default | effect |
|---|---|---|
| `LUMINA_KPEMISS_FB_OTS` | `0` (off) | when 1, at the `-3` fb emission site (`lumina_cuda.cu:3647–3675`), if the shell is in the thick tier (reuse the A mask) **and** the drawn `nu_edge > FB_OTS_NUMIN`, emit via the `B(Te)` Planck path (`:3711–3718`) instead of the raw edge |
| `LUMINA_KPEMISS_FB_OTS_NUMIN` | `3.29e15` (912 Å) | EUV cutoff: only ground edges bluer than 912 Å are OTS-thermalized; FUV/optical edges keep the raw fb continuum (protects deep FUV) |

Mechanics: a single `if` before the `nu_edge` emission (`:3666`). Reuses `d_kpacket_te_g`, the
Planck sampler, and the thick-tier mask — no new tables. Composes with COOLGUARD (`:3648–3649`,
which already skips FB-MULTI in Fe-V-burned deep shells) — the two gates are ANDed on the `-3`
branch.

### Staging (each prong independently testable)
1. **A-only** (`BSRC_PHOT=1`, `FB_OTS=0`): isolates the line channel. Expect EUV(450–912) ↓, excited
   Gph ↓, but `f(FeIV)` **not** yet at target (fb ground channel remains).
2. **B-only** (`BSRC_PHOT=0`, `FB_OTS=1`): isolates the fb channel. Expect EUV(300–450) ↓, ground
   Gph ↓; `f(FeIV)` still high (S III line channel remains).
3. **A+B composed**: the success configuration (both channels cut; the runaway reverses, §5.1).

---

## 5. Validation card — pre-registered gates (do NOT move)

### 5.1 PRIMARY (the composed A+B target)
| quantity | kpr5 now | gate | CMFGEN truth |
|---|---|---|---|
| `f(FeIV, s8)` | 0.982 | **≤ 0.25** (expect toward 0.022) | 0.022 |
| `Gph(FeIII, s8)` | 27.37/s | **within 10× of 3.5e-5** (≤ ~3.5e-4/s) | 3.5e-5/s |
| `EUV(300–450, s8)` outward decline | ~53× | **steep, CMFGEN-like** (≫ 1e3×; toward 1.42e6) | 1.42e6 (s0→s8) |
| `T_e(s8)` | 12208 K | **toward 10.4 kK** (≤ 11.4 kK) | 10383 K |

**Nonlinear expectation (why A+B, not A or B, crosses the threshold):** A+B remove ~90% of local
EUV creation (49% line + 40.5% fb). With deep-leak=0%, the photospheric field is ~100% locally
regenerated, so cutting the sources drops the field, Fe recombines toward III, `n(FeIII_ground)`
repopulates, `chi_bf(404 Å)` rises, EUV re-absorption grows, the field falls further — a **virtuous
cycle reversing the runaway** toward the CMFGEN-like III-rich/faint-EUV state. This is the physical
realization of `bistability_test/VERDICT.md §4` ("Fix = the FIELD") and probe #1 (`GPH_JTABLE`
transplant), achieved by fixing the transport sources rather than transplanting the field. Each
prong alone leaves Gph at 9400× (fb-only) or ~30000× (line-only) critical — below the recombination
threshold — hence "neither alone suffices" (`photospheric_euv_source/VERDICT.md §6`).

### 5.2 RETAINED — deep gains must HOLD (from `sbatch_kpr5.sh` pre-registered gates)
Both prongs act only on **thick-tier photospheric/mid** shells and (B) only on EUV edges; the deep
`-4` exit (W>0.13, SRC=2) and FUV (>912 Å) fb are untouched.
| gate | threshold |
|---|---|
| FUV(918–1290, s0) | ≥ 1.5e-4 |
| FUV gradient slope | ≥ +2.0 |
| u_bol(s0) | ≥ 450 |
| funnel dead: mc/cs @1450–1650 | ≤ 3× |
| residuals unchanged | Co twin rate deficit (~10×), MC blue-tilt, T_rad pin — **no claim of change** |

### 5.3 WIRING (check FIRST on any null)
- `[KPR] ... BSRC_PHOT=1 WFLOOR=0.02 PHOT_SRC=1` printed once at init.
- Per-iter `bteq_exits` must become **large at the photosphere** (currently
  `bteq_exits=0 cdf_exits=59M` at iter 1 — the WARNING is the disease); after A, phot shells report
  `-4` exits.
- `[FB-OTS]` per-iter counter of EUV ground-edge redirects > 0 after iter 0.
- Byte-identical check: with both new gates unset, a 1-iter run must reproduce kpr5 bit-for-bit.

---

## 6. Risks, kill-criteria, and test order

### 6.1 Risks / kill-criteria
- **(A) over-thermalizing the thin outer nebula** → kills legitimate escaping nebular EUV. Guard =
  `BSRC_WFLOOR` (only W>0.02). *Kill A* if the emergent spectrum's genuine nebular EUV/optical
  recombination features collapse.
- **(A) SRC choice** — if SRC=1_phot proves to redden the emergent continuum vs SRC=2_phot beyond
  the EUV, fall back to SRC=2_phot (the EUV effect is ~identical; §2.2). Not a kill, a knob.
- **(B) B(Te) redirect under-absorbs** (if 12.2 kK B(Te) still leaves too much 404 Å because Te is
  +1.9 kK high) → escalate to the §3.2 fallback (immediate `σ_bf` re-scatter). *Kill the redirect
  variant* only if EUV(300–450) fails to steepen while `[FB-OTS]` demonstrably fires.
- **(B) touching FUV edges** → guarded by `FB_OTS_NUMIN=ν(912 Å)`; if FUV(s0) drops, the cutoff is
  wrong. *Kill* if deep FUV gate (5.2) breaks.
- **Composed over-correction** — if `f(FeIV,s8)` overshoots to CMFGEN-cold *and* the emergent
  spectrum degrades elsewhere, the tier is too aggressive; tighten `WFLOOR`/`NUMIN`. This is the
  good failure mode (we can dial back).
- **Falsifier (from bistability §4):** if A+B fire (counters confirm) but `f(FeIV,s8)` stays >0.5,
  the field is not the whole lock — re-open the ionization balance (α/DR) rather than transport.

### 6.2 Which prong first — recommendation
**Test order: A → B → A+B.**
- **A first (lower risk, larger raw Gph share):** it is a **host-only mask extension** of the
  already-validated B3 machinery (`:5639`) + an SRC selector — near-zero implementation risk — and
  it removes the **larger** Gph fraction (excited, 76%). Running A-only immediately tests half the
  mechanism and de-risks the field response before writing new transport code.
- **B second (the core / higher-leverage novel fix):** B is the load-bearing detailed-balance
  repair for the **hardest** over-ionization (the 404 Å ground edge, where the field is ~400× CMFGEN
  and worst) and the one that proves the transport-DB thesis. It requires the new `-3` OTS branch.
- **A+B composed** is the only configuration expected to pass 5.1 (each alone stays above the
  recombination threshold). Ship A and B as independent gates so the composed run is just both flags
  on, and each can be A/B-tested in isolation for attribution.

---

## Appendix — file:line index for every mechanism claim
- k-packet exit ladder (-2 ff / -3 fb / -4 B(Te)); B3 gate on `d_kpr_qualify`:
  `lumina_cuda.cu:3278–3341` (fb `:3302`, B3 `:3314–3320`, CDF `:3323–3339`).
- `-3` fb emission (FB-MULTI edge draw + thermal tail): `lumina_cuda.cu:3641–3698` (emit `:3666–3675`; `zsel` `:3660`).
- `-4` B(Te) exit; SRC=1 Planck `:3710–3718`, SRC=2 chi-forest `:3707–3709` (`d_kpr_chi_sample :3091–3104`).
- Transport bf absorption: lookup `chi_bf` `:4122–4128`; bf event `:4263–4304`; trace `:3846–3919`; `d_bf_get_chi` `:2681–2694`.
- bf grid bounds `[1.5e14,3.0e16]`: `lumina.h:329–330`; `bf_opacity_init` `lumina_plasma.c:3433–3435`.
- `chi_bf` fill (`n_level·σ`, population-weighted; ground edge): `lumina_plasma.c:3713–3786` (ground-edge `:3743–3744`).
- FB-MULTI edge CDF build (weight `∝ n(recombining ion)`, ground threshold, `zstage`): `lumina_plasma.c:2306–2385` (w `:2337–2338`, ν0 `:2327–2329`, zs `:2341`, `p_fb=C_fb_real/denom` `:2381`).
- Radeq `[DBFB]` partner (the analog): `lumina_plasma.c:4973–5092`.
- B3/COOLGUARD host masks + SRC=2 CDF: `lumina_cuda.cu:5618–5693` (qualify `:5639`, COOLGUARD `:5626–5640`).
- `LUMINA_CMF_OTS` (case-B for `eta_bf`, host-only precedent): `lumina_plasma.c:3547–3559`.
- BF enabled in kpr5: `stdout.log` L80 `LUMINA_BF_OPACITY=1`, L167 `ENABLED`; gate `lumina_cuda.cu:4910–4912`, `:5226`.
- W/Te profile: `logs/coevolve_consume_a10_kx_kpr5/lumina_plasma_state.csv` (s8: W=0.0389, Te=12208; s2: W=0.134).
