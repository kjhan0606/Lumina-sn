# OPUS5 — Wave 3 element-wide NLTE: independent code audit

Auditor: Opus 5 (independent). Source-only. No runs, no edits, no git.
Date: 2026-07-31.

Scope read: `src/lumina_element_wide.c` (1323 L), its integration points in
`src/lumina_plasma.c` and `src/lumina.h`, the binding spec
`docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md`, the ARTIS reference
`../artis-ref/nltepop.cc` (+ `ratecoeff.cc`, `input.cc`, `atomic.h`), and the
atomic-data manifests that the gated path consumes.

**Deliberately NOT read** (independence): `docs/CODEX_WAVE3_*`, `docs/FABLE_*`,
`docs/WAVE3_ELEMENTWIDE_SPEC.md`. No A/B/C conclusion was consulted.

Convention note used throughout: Lumina `nlte_ion`/`level_ion` is 0-based
(1 = II, 2 = III, 3 = IV). `EW(A,n,i,j)` is column-major, `i` = row = target,
`j` = col = source — the same orientation as ARTIS's row-major
`matrix[row*dim + col]`.

---

## 0. Architecture as built (needed to read the rest)

The module does **not** re-implement the rate arithmetic (mostly). It calls the
production producer twice per element:

`src/lumina_element_wide.c:1160-1172`
```
for(int pair=0;pair<2;pair++) {
    int lo=slots[pair],hi=slots[pair+1];
    ew_capture_begin(...,lo,Np,N,base=stage_start[pair],skip_lo_bb=(pair==1),plane);
    nlte_assemble_rate_matrix(nlte,atom,plasma,opacity,lo,hi,shell,...,tmpA,tmpb,Np,...);
    ew_capture_end(...);
}
```
`tmpA/tmpb` are discarded; the rates are harvested by hooks placed inside the
producer (`src/lumina_plasma.c:15089-15095, 15220-15221, 15275-15276,
15367-15368, 15456-15457, 15559, 16054, 16131`) and written into seven dense
`N×N` channel planes at offset `wide_base = stage_start[pair]`.

Pair 0 = (II,III) contributes II bb + III bb + II→III bf; pair 1 = (III,IV)
contributes IV bb + III→IV bf, with III bb/coll suppressed by `skip_lo_bb`
(`src/lumina_element_wide.c:226-228`). **Verified: no bb/coll double count of
the shared III ion, and no bf double count.** The producer's bb line loop covers
both ions of a pair (`src/lumina_plasma.c:14736-14737`), and the Fe III Zhang /
generic `col_data` passes loop `ion = lo..hi`
(`src/lumina_plasma.c:15302, 15388`), so III collisions are harvested in pair 0
and correctly suppressed in pair 1.

The bound-free block is the exception: under capture the producer's entire
per-level bf arithmetic is bypassed (`src/lumina_plasma.c:15551-15562`) and
replaced by a second, independent implementation inside the new module
(`src/lumina_element_wide.c:245-379`). This is the one place where the spec's
§9 "call the common rate producer, do not make two copies of the arithmetic" is
not honoured — and it is where most of the content findings live.

---

## Axis 1 — Physics content of every matrix channel

### 1.1 `rad_bb` / `coll_bb` — **SOUND**

`src/lumina_plasma.c:15083-15096`
```
ACM(i_up,i_lo) += total_up*f_lo;   ACM(i_lo,i_lo) -= total_up*f_lo;
ACM(i_lo,i_up) += total_down*f_up; ACM(i_up,i_up) -= total_down*f_up;
nlte_ew_capture_transition(NLTE_EW_RAD_BB,  i_up, i_lo, R_absorb*f_lo);
nlte_ew_capture_transition(NLTE_EW_RAD_BB,  i_lo, i_up, (R_stim+R_spont)*f_up);
nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_up, i_lo, C_up*f_lo);
nlte_ew_capture_transition(NLTE_EW_COLL_BB, i_lo, i_up, C_down*f_up);
```
Splitting the production `total_up/total_down` into radiative and collisional
planes is exact; the sum of the two planes reproduces the production entries
bit-for-bit. Source-side within-SL fraction on both directions matches ARTIS
`s_renorm[level]` on the *emitting/absorbing* level
(`../artis-ref/nltepop.cc:512-524, 540-559`). Row/column orientation matches
ARTIS. Units s⁻¹. Intra-SL transitions (`i_lo==i_up`) are dropped at
`src/lumina_element_wide.c:225`, equivalent to the exact cancellation the spec
§3.3.3 demands.

Note (behaviour change, declared): under capture the producer disables the
per-line collisional floor (`src/lumina_plasma.c:15018`) and the entire
METACOLL metastable floor (`src/lumina_plasma.c:15171`). That is spec-compliant
("no new Ω floor, zero clamp firings") but means the EW lane's `coll_bb` is
weaker than the pair baseline's for drainless metastables. Report it as a
channel difference, not a defect.

### 1.2 `rad_bf` / `coll_bf` — **DEFECT (several; see D2/D3/D4)**

`src/lumina_element_wide.c:369-376` places
```
A_rad_bf[U,L] += p*rad_ion*f_l ;  A_rad_bf[L,L] -= …
A_rad_bf[L,U] += p*rad_rec*f_t ;  A_rad_bf[U,U] -= …
A_coll_bf[…] likewise
```
Term-by-term against `../artis-ref/nltepop.cc:587-615`:

| term | Lumina EW | ARTIS | verdict |
|---|---|---|---|
| placement/sign | row=target, col=source, diagonal debit on source | identical | SOUND |
| lower-side projection | `lower_fraction` (`f_l`) | `s_renorm[level]` | SOUND |
| upper-side projection | `f_upper` (l.299-300) | `s_renorm[upper]` | SOUND — and a genuine *fix*: legacy applied **no** weight to recombination (`src/lumina_plasma.c:15712-15715`) |
| target probability | applied once, outside (`p*…`, l.369-376) | `FOURPI * get_phixsprobability(...) * ∫` (`ratecoeff.cc:437-440`) | SOUND |
| Γ (photoion) | `Σ_bins 4πσ/(hν)·J·Δν`, or `σ·Γ_est` (l.321-333) | `corrphotoioncoeff` = Γ − stimulated-recomb correction | **differs by construction** (see note) |
| α (recomb) | `n* · Σ 4πσ/(hν)(2hν³/c²+J)e^{−hν/kT}Δν` (l.334-352) | `rad_recombination_ratecoeff` (spontaneous only when stim is folded into Γ) | net physics equivalent; per-channel entries are NOT comparable to ARTIS at 0.03 dex |
| n* (Saha) | `n_e (h²/2πm kT)^{3/2} (g_l/2g_u) e^{χ_t/kT}` (l.343-351) | `SAHACONST·g_l/g_u·T^{-3/2}` | SOUND, level-resolved (better than legacy, which used the upper-ion **ground** g: `src/lumina_plasma.c:15665-15676`) |
| C_ion | `n_e·1.55e13/√T·g_bf·σ_edge·e^{−u}/u` (l.354-362) | `col_ionisation_ratecoeff` (macroatom.cc) | SOUND, same Seaton/Mihalas form and the same 0.1/0.2/0.3 Gaunt |
| C_rec | `C_ion · n*` (l.361) | `col_recombination_ratecoeff` | SOUND (exact DB inverse) |

Stimulated-recombination bookkeeping note: ARTIS subtracts stimulated
recombination from the ionization coefficient; Lumina adds it to the
recombination coefficient (the `+J` inside the Milne integral). The **net**
matrix is the same, but the §4.2 acceptance table compares
`|log10(Lumina/ARTIS)|` *per channel entry* at median ≤0.03 dex. Those entries
are not the same quantity. This must be reconciled in the comparator or the
§4.2 gate is meaningless. Flagged as **DEFECT[medium] D5b** below (grouped with
the ARTIS-parity deviations).

### 1.3 `nt_bf` — **DEFECT[low-medium] (D9)**

`src/lumina_element_wide.c:381-416` routes the non-thermal ionization rate
through the `ma_rr` photoionization CSR. ARTIS
(`../artis-ref/nltepop.cc:634-651`) routes NT ionization from **every** lower
level into the **upper ion ground state** (`upper_groundstate_index`), with an
**ion-specific** `Y_nt`. The comment at `src/lumina_plasma.c:16129-16131`
("ARTIS traverses every lower level and its continuum target") misstates the
reference. Lumina's `R_nt_per_particle = R_nt_total/n_total_atoms`
(`src/lumina_plasma.c:16123-16126`) is a legacy per-atom average, not an
ion-resolved `Y_nt`; the EW lane now applies that same per-atom rate to
III→IV as well as II→III, a channel the 16-pair lane never had.

Additional: levels with no CSR entry get **no** NT ionization and no counter
(`src/lumina_element_wide.c:398-399`); and the NT target is not validated for
`Z`/`ion+1` (contrast the bf hook's check at l.283-286), so a malformed route
that happens to land in range silently deposits NT flux into a wrong SL.

### 1.4 `autoion_DR` — **DEFECT[medium] (part of D2)**

`src/lumina_plasma.c:16038-16057`
```
const DRCoefficient *coef = dr_lookup(Z_pair, ion_hi_stage);
double R_dr = alpha_dr * n_e;
if (R_dr > 0.0 && ground_hi < N) {
    ACM(0, ground_hi) += R_dr; ACM(ground_hi, ground_hi) -= R_dr;
    nlte_ew_capture_transition(NLTE_EW_AUTOION_DR, 0, ground_hi, R_dr);
}
```
Indexing is correct (pair-local 0 = lower-ion ground SL, `ground_hi` = upper-ion
ground SL; the capture adds `wide_base`). But:

1. For pair 1 this is `dr_lookup(26,3)` = the **Fe IV → Fe III** DR entry
   (`src/lumina_plasma.c:7716`). The legacy 16-pair layout has **no (III,IV)
   pair**, so this IV→III drain is *new to the EW lane*.
2. It is one-way. ARTIS pairs autoionisation (upward) with the collisional
   capture (downward) as a DB couple (`../artis-ref/nltepop.cc:654-706`);
   Lumina adds only the downward half. The manifest labels the plane
   `autoion,inactive:no_autoion_data_producer` while simultaneously reporting
   `DR,active` from the same plane.
3. Ground-to-ground routing only (no cascade), unchanged from legacy.
4. Pre-existing and out of scope, but worth one line: if the CMFGEN `σ_bf`
   grid carries low-T DR resonances, the Milne α and this DR α double count.

### 1.5 `nt_bb` — inactive, correctly declared (no producer exists).

### 1.6 Channels present in the pair baseline but **absent** from the EW lane

All are reachable only after the EW early return at
`src/lumina_plasma.c:16144-16148`, or are `!ew_capture`-gated:

| channel | gate | declared? |
|---|---|---|
| charge exchange | `src/lumina_plasma.c:16062` | yes, manifest `charge_exchange,inactive` |
| time-dependent ionization (backward-Euler) | early return | not in manifest |
| TOPSTAGE_IV Saha-IV reservoir | `src/lumina_plasma.c:15832` | yes |
| METACOLL metastable Ω floor | `src/lumina_plasma.c:15171` | not in manifest |
| per-line collisional floor | `src/lumina_plasma.c:15018` | not in manifest |
| `LUMINA_DR_FLOOR_CMS` empirical floor | `src/lumina_plasma.c:16049` | not in manifest |
| Kramers σ₀ν⁻³ fallback continuum | `src/lumina_element_wide.c:257-259` | **no — silent (D3)** |

**Axis 1 verdict: DEFECT[high]** — driven by D2/D3/D4 below.

---

## Axis 2 — Conservation row & normalization: **SOUND, with one caveat (D11)**

`src/lumina_element_wide.c:1176-1178`
```
double n_elem = ew_element_density(atom,plasma,Z,shell);
for(int j=0;j<N;j++) EW(Anorm,N,0,j)=1.0;
b[0]=n_elem;
```
Exactly one row is overwritten, exactly one RHS entry is non-zero, and there is
no charge row. This matches `../artis-ref/nltepop.cc:1250-1262` verbatim.

I traced every legacy pin/rescale/anchor that the spec §2.1 requires to be
bypassed and confirmed each is unreachable in the gated path:

* per-ion conservation / `LUMINA_NLTE_ION_LOCK`, floor-reg Boltzmann anchors,
  negative-pop repair, `b_k` cap, per-ion rescale — all live **after**
  `src/lumina_plasma.c:16144-16148` (the `if (ew_capture) { free(bb_connected);
  return; }` early return).
* `pair_shares_slot` per-ion pin — the EW element pairs `{4,5}` (Fe) and
  `{7,8}` (S) share no slot with any other pair
  (`src/lumina_plasma.c:14018-14026`), so `pair_shares_slot==0` for them anyway;
  and `nlte_solve_ion_shell` is not called at all for a committed `(Z,s)`
  (`src/lumina_plasma.c:17099-17103`).
* `saved_lo` save/restore — the restore is skipped for committed `(Z,s)`
  (`src/lumina_plasma.c:17117-17125`).
* TOPSTAGE_IV reservoir and RHS source — `!ew_capture` at
  `src/lumina_plasma.c:15832`; the only `b[]` write before the early return is
  that block's `b[sl] -= n_lte_hl*I_rec_hl` (`:15954`), so **the EW RHS is
  provably zero except `b[0]`**.

Caveat **D11 [low]**: `n_elem` is the *total* element density
(`src/lumina_element_wide.c:522-535`, X·ρ/(A·amu)), while the matrix spans only
II–IV. Any Fe I / Fe V mass is forced into the II–IV window. The boundary gate
(`:1214-1235`) demands `boundary_max==0.0` **exactly** and `tau_boundary==0.0`
exactly — far stricter than §1.3.2's 1e-8 — so this is fail-closed, but it also
means `pass` (and therefore *commit*) is effectively unreachable in any model
carrying a non-zero Fe I or Fe V population. In practice the pilot will emit
`EW_VALID_P_ELEM_SCOPE_FAIL` and the §4.3 `p_elem` numbers are produced with the
full element density in II–IV. Since the conservation row only fixes the overall
scale (the shape comes from the N−1 SE rows), *ion fractions* are unaffected;
absolute populations are inflated. Benign for the declared metric, but the
diagnostics should say so.

---

## Axis 3 — Level indexing & super-level projection: **SOUND, one silent truncation**

Indexer (`src/lumina_element_wide.c:1123-1126`, `:229-238`):
`stage_start[q+1] = stage_start[q] + (SL count of slot q)`, `wide = wide_base +
pair_local`. Traced both pairs by hand:

* pair 0, `wide_base=0`: II SLs → `[0,nII)`, III SLs → `[nII,nII+nIII)`. ✅
* pair 1, `wide_base=nII`: III SLs → `[nII,nII+nIII)`, IV SLs →
  `[nII+nIII,N)`. ✅ The shared III block lands on the *same* wide indices from
  both pairs, which is exactly what makes `skip_lo_bb` sufficient and
  III→IV connect to real IV **columns** rather than an RHS reservoir.

Projection: `upper_sl = fl_to_super[ng] - pair_super_start`
(`:294`), bounds-checked against `N_pair` (`:295`); `f_upper` read from
`within_sl_frac[ng*n_shells+shell]` (`:299-300`). `within_sl_frac` is refreshed
once at the top of `nlte_solve_all` (`src/lumina_plasma.c:16986`) *before* the
EW run (`:17023`), so EW and the pair lane see the same fractions. Restoration
`n_FL = x[SL]·f` (`src/lumina_element_wide.c:1290-1294`) is the same formula the
pair lane uses. ✅

**Silent truncation (D3)**: `src/lumina_element_wide.c:255-259`
```
if (!a->cmfgen_loaded || !a->cmfgen_has_sigma ||
    !a->cmfgen_sigma_bf || !a->cmfgen_has_sigma[lower_global])
    return;                      /* no counter, no coverage entry */
```
and the mirror in the coverage counter `:486-491`. A level with no CMFGEN σ row
gets **zero** ionization and **zero** recombination in the EW lane, is never
counted in `continua_expected`, and the diagnostics therefore report 100%
continuum coverage. The legacy path gave exactly those levels a Kramers
continuum (`src/lumina_plasma.c:15606-15616`,
`sigma = sigma_0·(ν_thr/ν)³`). Spec §3.2 forbids this explicitly ("구현되지
않은 활성 항을 0으로 조용히 대체하면 실패").

Magnitude, from the shipped data
(`data/tardis_reference_toy06_19p48d/ma_radrecomb_target_manifest.csv`):

| ion | levels | with bf σ | dropped in EW |
|---|---:|---:|---:|
| Fe II | 2698 | 2576 | **122** |
| Fe III | 1500 | 1500 | 0 |

The loss is **Fe II only**. It removes ionization channels from the bottom of
the ladder and none from the middle — an ion-asymmetric bias toward more Fe II.

Also note: `LUMINA_SUPER_CUTOFF` is counted (`:1128-1139`) and declared as an
input projection, not gated. Fine as declared.

---

## Axis 4 — Numerics: **SOUND core, DEFECT in the equilibration contract and in diagnostic honesty**

**Correct and honest:**
* `ew_lu_factor` (`:624-658`) — partial pivoting, rank counted against
  `tol = ε·N·max(1,‖A‖_max)`, returns −1 on rank<N so a rank-deficient matrix
  can never be solved. `pivot_growth = max|U|/max|A_0|` with `A_0` captured
  before factorization (`:626`). ✅
* `ew_lu_solve` (`:660-675`) — LAPACK-style sequential `ipiv` application;
  correct given how the swaps were recorded. ✅
* Iterative refinement (`:929-944`) — residual accumulated in `long double`
  against the equilibrated system, ≤10 passes, history stored. ✅
* `ew_lu_rcond_1` (`:775-796`) — this is an **exact** ‖A⁻¹‖₁ by N solves, not an
  estimator, despite the dump key `rcond_1_solver_estimate`. Honest (better than
  labelled).
* `ew_singular_extrema` (`:705-773`) — real Golub–Reinsch SVD; κ₂ is a true
  spectral condition number. ✅
* Un-scaling `x[i] *= cs[i]` (`:945`) is the correct inverse of the column
  scaling; `b[i] *= f` (`:873`) the correct partner of the row scaling. ✅
* `ew_scaled_residual` (`:1056-1072`) is evaluated on **`Araw`** (no conservation
  row) over rows 1..N−1 with denominator `max(inflow, outflow, |x_i|/t_ref)` —
  i.e. the physical SE residual, per §5.2.D. ✅
* Permutation check (`:1074-1108`) — stage-block reverse + xorshift shuffle
  seeded by the atomic checksum (deterministic), `Ap[i][j]=A[perm[i]][perm[j]]`,
  `bp[i]=b[perm[i]]`, un-permute `xu[perm[i]]=xp[i]`: algebraically exact. ✅

**Failure semantics — fail-closed, verified:**
`x` is `calloc`'d (`:1149`); a failed solve leaves `conservation=|0−n|/n=1 >
1e-12` so `pass=0`; commit is guarded by `if(pass && commit_requested)`
(`:1286`); the return value is `pass ? 1 : -1` and the caller only skips the
pair lane when `ew_status==1` (`src/lumina_plasma.c:17099-17103`). **No state is
committed on failure.** ✅

**D5 [medium] — the equilibration is not the algorithm the spec binds.**
`src/lumina_element_wide.c:855-895` does a row-norm pass (`f=1/√‖row‖₂`) then a
column-norm pass (`f=1/√‖col‖₂`). The spec §3.6.2 and
`../artis-ref/nltepop.cc:733-757` specify **one** factor per index,
`f = √(col_norm/row_norm)`, applied as row×f and col×(1/f). Both are legitimate
scalings and the solution is unaffected, but the reported κ₂, the `|f−1|≤1e-3`
stopping test, the pre-registered κ₂ ≤ 1e12 PASS threshold, and the §4.2
"equilibrated matrix vs ARTIS" comparison are all measuring a different matrix
than the spec declares.

**D6 [medium] — the channel column-sum gate is structurally vacuous.**
`src/lumina_element_wide.c:236-238` writes `+r` at `[i,j]` and `−r` at `[j,j]`
in the same two statements; every rejection path (`:218-228`) drops both.
Therefore `max_j|Σ_i A_c[i,j]| / max|A_c|` is **roundoff by construction** for
every plane, and the gate at `:1257`/`:1259` (`max_colsum<=1e-12`) cannot fail
for *any* assembly error — wrong target row, wrong channel, wrong sign, double
count. §4.1 lists this as the primary assembly-correctness gate. It is currently
a tautology and should be labelled as such (or replaced by an independent
recomputation of the column sums from the provenance dump).

**D12 [low] — diagnostic labelling.**
* `raw_singular_value_max/min`, `raw_kappa_2` (`:1198-1201`, key names at
  `:1278`) are computed on **`Anorm`** — the matrix *after* the conservation row
  was inserted — not on the raw channel sum.
* `raw_rank` (`:798-825`, gate `raw_rank==N-1` at `:1256`) is the rank of rows
  1..N−1 of `Araw` under a fixed 1e-12 relative tolerance after one global
  scaling. That is a defensible (and conservative) test, but it is not "the rank
  of the raw matrix", and a fixed relative tolerance on a rate matrix spanning
  ~30 decades is arbitrary.
* `identity` dump's `g_or_SL_partition` (`:1018`) is Σg over members, while the
  solve uses the Boltzmann-weighted `within_sl_frac`.

**D13 [low]** — `solve_piv` is `malloc`'d (`:1152`) and only filled after a
successful `ew_lu_factor` (`:927`), yet `pivot_%d` is printed whenever
`solve_attempted` (`:1278`). An early failure dumps uninitialized memory.

**D15 [low]** — in `ew_singular_extrema` the QR loop
(`:754-758`) uses `l` after `for(l=k;l>=0;l--)`; if no break fired, `l==-1` and
`w[l]` at `:758` reads out of bounds. Safe only because `rv1[0]` is structurally
0 (`:715, 723`), so the first test always breaks at `l==0`. Fragile, not
currently reachable.

---

## Axis 5 — Gate discipline (OFF = true no-op): **SOUND**

Adversarial pass over every OFF path:

* `ew_parse_gate` (`:41-49`): unset and explicit `0` take the *same* branch and
  return before any banner, allocation, counter, dump-dir env read, or RNG.
  `enabled=0`, `requested=0`. ✅
* `nlte_element_wide_layout_enabled()` → 0, so `nlte_init`
  (`src/lumina_plasma.c:13973-13982, 14087-14090`) uses the byte-stable base
  tables and leaves `super_mode` under `LUMINA_SUPER_LEVELS` alone. ✅
* `nlte_get_pairs` (`src/lumina_plasma.c:8118`) → legacy branch. ✅
* `src/lumina_atomic.c:988-1000`: the `ma_radrecomb_target.bin` load is only
  forced when EW is enabled (or the two pre-existing bf gates). OFF ⇒ no extra
  file I/O. ✅
* `src/lumina_cuda.cu:997-1003`: OFF ⇒ the GPU solver is untouched. ✅
* Every `!ew_capture` guard in the producer (`15018, 15171, 15832, 16049,
  16062, 16144`) evaluates `ew_capture = nlte_ew_capture_active() = 0`, so the
  legacy arithmetic and call order are preserved. ✅
* `int *bb_connected = (floor_reg_mode && !ew_capture) ? calloc : NULL`
  (`src/lumina_plasma.c:15195-15196`) — unchanged when OFF. ✅
* `nlte_element_wide_matches()` at `src/lumina_plasma.c:10031` returns 0, so the
  `!matches` term is 1 and the STAGE4 `b_k` cap keeps its legacy behaviour. ✅
* All capture hooks return on `!ew_cap.active` (`:216, 249, 382`). ✅

Residual OFF cost: the hooks are *called* in the hot bb loop (6 calls per line
per shell). Their arguments (`R_absorb*f_lo`, …) are side-effect-free, so output
is byte-identical; only CPU time is affected.

**D14 [low, latent]** — `ew_parse_gate` memsets and fills a file-static
(`:41-49`) and is reachable from inside `#pragma omp parallel for`
(`src/lumina_plasma.c:17094-17103`, and `:10031`). In practice it is always
pre-parsed serially at init (`:8118`, `:13973`) and at `:17017`, so no race
occurs today. One `pthread_once`/`__atomic` would remove the latent hazard.

**But: ON is NOT a no-op outside the pilot (D1, D8).** See below. The OFF
invariant holds; the *ON off-target* invariant of spec §6.2.5 does not.

---

## Axis 6 — The key physics question: top-stage-asymmetric defects

### D1 [HIGH] — arming the gate deletes Fe IV line opacity model-wide

Chain, all verified in source:

1. `nlte_element_wide_layout_enabled()` ⇒ `nlte_init` uses `NLTE_TARGET_Z_EW`
   (33 slots), which **inserts Fe IV (slot 6) and S IV (slot 9)** into the NLTE
   ion table — `src/lumina_plasma.c:7610-7618, 13973-13982`.
2. `nlte_line_map` is built purely by `(Z,ion)` membership in that table —
   `src/lumina_plasma.c:14100-14113`. Fe IV lines now map to slot 6.
3. `nlte_get_pairs` under EW returns only the **16 base pairs**
   (`src/lumina_plasma.c:8118-8135`) — slots 6 and 9 appear in **no** pair, so
   the pair lane never solves them.
4. `nlte->nlte_level_populations` is `calloc`'d (`src/lumina_plasma.c:14175`).
   The **only** writer for slot 6/9 is the EW commit
   (`src/lumina_element_wide.c:1286-1296`), which fires only for the single
   pilot shell and only when every gate passes.
5. `nlte_update_tau_sobolev` is called unconditionally at the end of
   `nlte_solve_all` (`src/lumina_plasma.c:17212`) and, for every line with
   `nlte_line_map[line] >= 0`, overwrites the Sobolev optical depth:
   `src/lumina_plasma.c:16894-16909`
   ```
   double n_lower = nlte->nlte_level_populations[nlte_lo*n_shells+s];   /* = 0 */
   double tau_nlte = SOBOLEV_COEFF*f_lu*lam_cm*t*n_lower*stim_corr;     /* = 0 */
   if (!(tau_nlte > 1e-100)) tau_nlte = 1e-100;
   if (!skip_tau) opacity->tau_sobolev[line*n_shells+s] = tau_nlte;
   ```
   and leaves `line_source_S = 0` (consumers then substitute B(T_e)).
6. The shipped line list has **4336 Fe IV lines**
   (`data/tardis_reference_toy06_19p48d/line_list.csv`, Z=26 ion=3; S IV has 0).

**Consequence:** with `LUMINA_NLTE_ELEMENT_WIDE=1`, *even in COMMIT=0 shadow
mode*, all 4336 Fe IV lines get τ = 1e-100 in **every** shell. Fe IV blanketing
is removed from the entire model. With COMMIT=1 the single pilot shell is
repaired (if it passes); every other shell is not.

This is a mechanical, certain, IV-only regression that no gate in the module can
see, and it violates spec §6.2.2 ("COMMIT=0 keeps the pair-wise authority
result") and §6.2.5 ("off-target elements/shells keep legacy arithmetic"). It is
by itself sufficient to produce "II/III unchanged-or-better, IV catastrophically
worse" in any downstream state/opacity/spectrum comparison.

Minimal fix direction (not applied): either exclude un-solved slots from
`nlte_line_map`, or seed slot 6/9 populations from the nebular/Saha ion density,
or skip the τ override for ions with zero total NLTE population.

### D2 [HIGH] — the III→IV balance is radiatively starved *by construction*

Three independent source facts:

1. **Frequency ceiling.** `NLTE_NU_MAX = 3.0e16 Hz` = 124 eV
   (`src/lumina.h:505-506`). The EW Γ integral runs over
   `[ν_threshold, ν_max]` (`src/lumina_element_wide.c:313-333`). For Fe II
   (χ = 16.2 eV) that is 16→124 eV; for Fe III (χ = 54.8 eV) it is 55→124 eV.
   The top-stage integral is truncated far harder, and it sits in exactly the
   EUV band the campaign has repeatedly measured as starved.
2. **No collisional support at the top.** `u = χ/kT_e ≈ 63` at T_e ~ 1e4 K
   ⇒ `exp(−u) ~ 4e-28` in `coll_ion` (`src/lumina_element_wide.c:355-360`). The
   III→IV collisional channel is numerically dead.
3. **Recombination out of IV is not starved.** The Milne integrand keeps the
   spontaneous `2hν³/c²` term (`:336-338`), whose `n*·e^{−hν/kT}` product is
   O(1) at the edge regardless of J. On top of that the EW lane adds a **new,
   one-way IV→III dielectronic drain** (§1.4 above,
   `src/lumina_plasma.c:16038-16057` with the `{26,3}` table entry at `:7716`),
   which the 16-pair lane never had because there is no (III,IV) pair.

Net: in the EW matrix, `n_IV/n_III` is set by (≈0 radiative + ≈0 collisional)
÷ (finite spontaneous recombination + DR). It should collapse toward zero
wherever J(>55 eV) ≈ 0, while II↔III retains a genuine radiative balance at
16–30 eV where J is real. In the pair lane, Fe IV never came from this
balance at all — it came from the external ionization solver / TOPSTAGE_IV
reservoir (`src/lumina_plasma.c:15818-15960`), which the EW lane deliberately
switches off.

This is a *reasoned prediction from the source and the grid constants*, not a
measurement. But it is exactly the signature the audit asks about: a solve that
can improve II/III while annihilating IV, with no gate able to detect it (the
boundary gate looks at stages I and V, not at whether IV was starved).

Recommended falsifier before any further Wave-3 verdict: dump, for the pilot
cell, Γ(III→IV), α(IV→III), C_ion(III→IV), C_3b, R_DR(IV→III) and the resulting
`n_IV/n_III`, and compare against the same five numbers from the pair lane's
TOPSTAGE_IV block and from CMFGEN. That is a pure offline read of the existing
`matrix_raw`/`provenance` dumps.

### D3 [MEDIUM-HIGH] — Fe II-only continuum deletion (details in Axis 3)

122 Fe II levels lose their continuum entirely; Fe III loses none. Direction:
less II→III ionization ⇒ more Fe II, less Fe III **and** less Fe IV. This does
not by itself produce "II/III better, IV worse", but it is a second uncounted
lever that moves the ladder as a whole and is invisible to the coverage gates.

### D4 [MEDIUM] — the bf radiation field is a different field from the baseline's

`src/lumina_element_wide.c:324-333` consumes `nlte->bf_rate_estimator` whenever
the array exists. The legacy producer consumes it only under
`(artis_parity_enabled() || LUMINA_C2_MATRIX_BF) && !LUMINA_NLTE_BF_JEQB`
(`src/lumina_plasma.c:15628-15642`). So in a non-parity run the pair baseline
integrates `pref·J` while the EW candidate uses the MC estimator `σ·Γ_est` —
an uncontrolled A/B. The `LUMINA_NLTE_BF_JEQB` falsifier is also silently
inoperative in the EW lane. Given that the EUV bins are exactly where the MC
estimator is sparsest, this compounds D2 at the top of the ladder.

### D5b [MEDIUM, latent] — level-resolved bf targets are not ARTIS-equivalent

ARTIS rescales the phixs table onto the **target-specific** edge:
`nu_edge = get_phixs_threshold(level, phixstargetindex)/H`
(`../artis-ref/atomic.h:545-552`, `../artis-ref/input.cc:822`,
`../artis-ref/ratecoeff.cc:418-420`), so an excited target gets the full edge
cross-section starting at its own higher threshold. Lumina EW keeps the level's
own σ(ν) curve (edge at the ground-target threshold) and merely truncates the
integral at the higher threshold (`src/lumina_element_wide.c:303-320`) —
an O((ν_ground/ν_target)³) underestimate for excited targets.

**Currently latent**: the shipped map is single-route,
`1route exc=0.000 eV … -> upper ground gidx`, p = 1
(`data/tardis_reference_toy06_19p48d/ma_radrecomb_target_manifest.csv`), i.e.
every route is to the upper-ion **ground** with `E_upper = 0`, so
`threshold = χ − E_lower` reduces exactly to the legacy formula. Two
consequences worth stating plainly:

* the D-5 fix actually delivered here is "III→IV connects to a real IV
  **column** instead of an RHS reservoir" — which is real and is the important
  part — **not** "level-resolved bf targets"; §11's checkbox "bf upper-target
  coverage 100%" is satisfied trivially by a ground-only map;
* the moment a genuine multi-target v2 map is supplied, the σ handling diverges
  from ARTIS in a direction that suppresses excited-target ionization.

### D8 [MEDIUM] — ON silently changes the whole-model baseline

`src/lumina_plasma.c:13973-13979` (33 slots ⇒ different
`n_nlte_levels_total`, `n_super_total`, all derived array shapes) and
`:14087-14090`
```
nlte->super_mode = ((env_on || element_wide) &&
                    nlte->n_super_total < nlte->n_nlte_levels_total) ? 1 : 0;
```
— arming the pilot **forces super-level mode on for every element and every
shell**, regardless of `LUMINA_SUPER_LEVELS`. So `p_pair` measured inside an ON
run is not the production pair-wise baseline, and §4.3's
`improvement = 1 − D(p_elem)/D(p_pair)` is only meaningful if the baseline run
is configured with `LUMINA_SUPER_LEVELS=1` and the same 33-slot layout —
which it cannot be, because the layout is coupled to the EW gate.

### D7 [MEDIUM] — the manifest reports unmeasured constants as measurements

`src/lumina_element_wide.c:1282`: `candidate_pair_owner_calls,0`,
`save_restore_calls,0`, `per_ion_pin_calls,0`, `topstage_IV_calls,0`,
`conservation_rows,1`, `charge_rows,0`,
`hot_cold_seed,frozen_rate_not_applicable` are **string literals in the format
string**, not counters. I independently verified the first four are *true* for
the EW lane (Axis 2), so the artifact is not lying — but §5.1.6 asks for "모든
guard/fallback counter" and §10 makes "ON인데 pair/save-restore/topstage counter
발화" a FAIL trigger, and there is no such counter to fire. Under
`docs/VERDICT_PROTOCOL.md` these lines must not be quoted as measurements.

The `hot_cold_seed` claim is additionally unverified: bb radiative rates can
depend on lagged populations under `LUMINA_NLTE_JBAR_POPS` modes 2/3 and MALI
(`src/lumina_plasma.c:15125-15126`), so the "frozen rate ⇒ seed-independent"
assertion is config-dependent, and §5.2.D requires the hot/cold seed run when it
is not.

### D10 [LOW] — `LUMINA_NLTE_OPACITY_IONSTAGE` is IV-blind and not in the guard list

`src/lumina_plasma.c:2624-2671` rescales `ion_number_density` for the II/III
pair to the **old II+III** nebular total, ignoring slots 6/9. If armed alongside
EW, it undoes any II/III↔IV transfer the pilot produced. Neither this knob nor
`LUMINA_TOPSTAGE_IV` appears in `ew_guard_config_count`
(`src/lumina_element_wide.c:973-987`), and `guard_config_count` is dumped but
never gates the verdict (`:1117`, `:1278`).

---

## Axis 7 — Other material findings

* **Memory safety**: all allocation failures route to `cleanup_fail` with
  complete frees (`:1297-1304`); `ew_singular_extrema`, `ew_lu_rcond_1`,
  `ew_components`, `ew_permutation_check` all free on every path. `EW()` accesses
  are bounds-guarded at the capture boundary (`:218-234`). No leaks or overruns
  found other than D13 (uninitialized read for dump) and D15 (unreachable
  `w[-1]`).
* **Races**: `nlte_assemble_rate_matrix` contains no `omp parallel` region
  (only four `omp critical` dump blocks: `src/lumina_plasma.c:14940, 15112,
  15786, 15978`), and `nlte_element_wide_run` is invoked from a **serial** shell
  loop (`src/lumina_plasma.c:17022-17033`). The file-static `ew_cap` is
  therefore safe. Only D14 (first-parse of `ew_gate`) is a latent hazard.
* **Units**: every captured rate is s⁻¹ per source-state particle
  (`R_absorb=B_luJ̄`; `Γ`; `α·n_e` folded into `n*`; `C_ion∝n_e`;
  `C_3b = C_ion·n*` ∝ n_e²; `R_DR = α_DR n_e`; NT s⁻¹). `x` is cm⁻³, `b[0]` is
  cm⁻³. Dimensionally consistent; the provenance dump's `s^-1` is correct.
* **Determinism**: the shuffle seed is the FNV atomic checksum (`:1211`); all
  dumps use `%.17g`. Repeat-run byte identity (§6.3) is achievable.
* **Artifact contract §5.1**: all eight artifacts (identity, matrix_raw,
  matrix_normalized, matrix_equilibrated, solution, diagnostics, provenance,
  manifest) are emitted (`:1270-1283`). ✅ One undeclared env var,
  `LUMINA_NLTE_ELEMENT_WIDE_DUMP_DIR` (`:113`), is not in the §6.1 gate table.
* **Verdict honesty**: `EW_VALID_P_ELEM_SCOPE_FAIL` requires both
  `topology_gate_pass` and `numerical_gate_pass` (`:1265-1268`), and never
  commits. `p_elem_valid` (`:1245-1246`) is correctly narrower than `pass`. ✅

---

## Defect list (severity ranked)

| # | Sev | Title | Primary file:line |
|---|---|---|---|
| D1 | **HIGH** | Gate ON zeroes Fe IV Sobolev τ (4336 lines) in every shell, even in COMMIT=0 shadow | `src/lumina_plasma.c:7610-7618`, `:14100-14113`, `:14175`, `:16894-16909`, `:17212`; `src/lumina_element_wide.c:1286-1296` |
| D2 | **HIGH** | III→IV is radiatively + collisionally starved by construction while IV→III keeps spontaneous Milne **plus** a new one-way DR drain | `src/lumina_element_wide.c:313-362`; `src/lumina.h:505-506`; `src/lumina_plasma.c:16038-16057`, `:7716` |
| D3 | MED-HIGH | Silent, uncounted deletion of the Kramers continuum for levels without a CMFGEN σ row (122 Fe II levels, 0 Fe III) | `src/lumina_element_wide.c:255-259`, `:486-491` vs `src/lumina_plasma.c:15606-15616` |
| D4 | MED | EW consumes `bf_rate_estimator` unconditionally; the pair baseline does so only under parity ⇒ uncontrolled A/B, `BF_JEQB` falsifier inert | `src/lumina_element_wide.c:324-333` vs `src/lumina_plasma.c:15628-15642` |
| D5 | MED | Equilibration is row-then-column norm scaling, not the ARTIS/spec `f=√(col/row)` algorithm ⇒ κ₂ gate and §4.2 comparison measure a different matrix | `src/lumina_element_wide.c:855-895` vs `../artis-ref/nltepop.cc:733-757`, spec §3.6.2 |
| D5b | MED (latent) | Excited-target bf uses the ground-edge σ curve truncated at the higher threshold; ARTIS rescales the phixs table onto the target edge | `src/lumina_element_wide.c:303-320` vs `../artis-ref/atomic.h:545-552`, `ratecoeff.cc:418-420` |
| D6 | MED | Channel column-sum gate is a tautology — cannot detect any assembly error | `src/lumina_element_wide.c:236-238`, `:964-971`, `:1257` |
| D7 | MED | Manifest prints unmeasured literals (`*_calls,0`, `hot_cold_seed`) as if measured; no pair/save-restore/topstage counters exist | `src/lumina_element_wide.c:1282` |
| D8 | MED | ON forces `super_mode` and a 33-slot layout for **all** elements/shells ⇒ the in-run pair baseline is not the production baseline | `src/lumina_plasma.c:13973-13979`, `:14087-14090` |
| D9 | LOW-MED | NT ionization routed through the bf CSR (ARTIS routes to upper ground with ion-specific Y_nt); comment misstates ARTIS; unvalidated targets; silent drop for route-less levels | `src/lumina_element_wide.c:241-244`, `:381-416`; `src/lumina_plasma.c:16117-16137` vs `../artis-ref/nltepop.cc:634-651` |
| D10 | LOW | `LUMINA_NLTE_OPACITY_IONSTAGE` writeback is IV-blind and absent from the guard list; `guard_config_count` never gates | `src/lumina_plasma.c:2624-2671`; `src/lumina_element_wide.c:973-987`, `:1117` |
| D11 | LOW | Conservation target is the total element density (I and V included); boundary gate demands exactly 0 ⇒ commit effectively unreachable, and this is not stated | `src/lumina_element_wide.c:522-535`, `:1176-1178`, `:1214-1235` |
| D12 | LOW | `raw_kappa_2`/`raw_singular_*` computed on `Anorm`; `raw_rank` is rows 1..N−1 with an arbitrary 1e-12 tolerance; `g_or_SL_partition` is Σg not the SL partition function | `src/lumina_element_wide.c:798-825`, `:1018`, `:1198-1201` |
| D13 | LOW | Uninitialized `solve_piv` can be dumped when the solve fails early | `src/lumina_element_wide.c:1152`, `:927`, `:1278` |
| D14 | LOW | Latent first-parse race on the `ew_gate` file-static (reachable from an OpenMP region) | `src/lumina_element_wide.c:41-49`; `src/lumina_plasma.c:17094-17103` |
| D15 | LOW | `w[-1]` reachable in the SVD QR sweep if `rv1[0]≠0`; safe only structurally | `src/lumina_element_wide.c:754-758` |

---

## Top-stage asymmetry candidates (ranked)

1. **D1 — Fe IV opacity annihilation.** Mechanical, certain, model-wide,
   affects *only* the IV stage, fires even in shadow mode, invisible to every
   gate in the module. If any Wave-3 evidence shows "IV worsened", this is the
   first thing to rule out, and it can be ruled out offline by diffing
   `opacity->tau_sobolev` for Z=26 ion=3 between an OFF run and an ON run.
2. **D2 — starved III→IV up-rate vs unstarved IV→III down-rate.** Γ truncated
   to 55–124 eV in an EUV-poor field, `exp(−63)` collisional ionization, plus a
   newly-introduced one-way Fe IV→III dielectronic drain with no inverse and no
   V→IV source. Predicts `n_IV → 0` from the solve itself while II↔III stays
   healthy. This is the content defect the question is looking for.
3. **D4 — the bf field the EW lane reads is not the one the baseline reads**,
   and the discrepancy is largest exactly in the EUV bins that set III→IV.
   Compounds D2 and makes the pair-vs-element comparison uncontrolled.
4. **D3 — Fe II-only continuum deletion.** Moves the whole ladder toward II;
   contributes to "II/III looks different" while further starving the feed to
   IV. Not IV-specific, but uncounted and ion-asymmetric.
5. **D9 — NT ionization newly applied to III→IV** at the same per-atom rate as
   II→III. This one pushes IV *up*, so it partially masks D2 and makes the
   direction of any measured IV change harder to attribute.
6. **`f_upper` weighting of recombination** (`src/lumina_element_wide.c:372,
   376`). Physically correct and an improvement over legacy
   (`src/lumina_plasma.c:15712-15715` applied no weight), but it systematically
   *reduces* IV→III recombination relative to the baseline — another
   uncontrolled top-stage term that must be held fixed when attributing a
   IV change.
7. **D11 — conservation absorbs any I/V mass into II–IV.** Scale only; ion
   fractions are unaffected. Weakest candidate, listed for completeness.

---

## Overall assessment

The structural core is genuinely good. The indexer, the sign/placement
convention, the level→SL projection (including a real correction to the legacy
unweighted recombination), the single conservation row, the LU/SVD/rcond/
refinement stack, the fail-closed commit, and the OFF invariant are all sound
and match ARTIS where they claim to. The D-5 topology fix is real: III→IV now
terminates on an actual IV **column** instead of an external Saha reservoir, and
every legacy pin/anchor/cap/floor is provably out of the gated path via the
early return at `src/lumina_plasma.c:16144-16148`.

What is not sound is the content around that core. Arming the gate rewrites the
global NLTE ion table, which (D1) silently destroys Fe IV line opacity in every
shell and (D8) forces super-level mode on the whole model, so neither the ON
shadow run nor its "pair-wise baseline" is the object the acceptance protocol
assumes. Inside the matrix, the top of the ladder is fed by a photoionization
integral that is truncated to an EUV band the campaign already knows to be
starved, gets no collisional support at χ = 54.8 eV, and is drained by a
newly-introduced one-way dielectronic channel (D2) — a configuration that
predicts IV collapse independently of anything II/III does. Three of the gates
that were supposed to catch this cannot: the channel column-sum test is a
tautology (D6), the continuum-coverage counters skip exactly the levels that
were silently dropped (D3), and four of the "zero legacy calls" manifest entries
are hard-coded literals (D7).

Verdict for the pilot as a whole: **PASS-WITH-SCOPE on structure,
FAIL-TOPOLOGY on the ON-mode invariant (D1), and FAIL on content honesty
(D3/D6/D7).** First-failing line: `src/lumina_plasma.c:16894` (τ overwritten
from a zero population for an ion that no solver owns). No Wave-3 "improvement"
or "map recovery" number should be quoted until D1 and D8 are removed from the
ON configuration and D2 is measured directly from the existing `matrix_raw` /
`provenance` dumps.
