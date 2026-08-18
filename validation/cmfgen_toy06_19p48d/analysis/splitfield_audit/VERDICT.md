# SPLIT-FIELD audit — does the coevolve-consume thermal ledger and ionization ledger live in different universes?

Offline source + run-state analysis, 2026-07-19. Read-only on logs/ and /gpfs. No source edits, no commit, no job touched.
B-run: `logs/coevolve_consume_a10_kx_gphall` (RADEQ_TE=1 RADEQ_SIMUL=1 VR_STD=1 DAMP=0.5 FB_RATE=1 GPH_ALLLEVEL=1,
MC_COEVOLVE **consume**, PHOTOION_MC=1 **ALPHA=1.0**, JBAR_POPS=3, CMF_LINERES_JBAR=2; **no GPH_JTABLE, no TE_TABLE**).
CMFGEN toy06 @19.48d benchmark. Follows up the flag raised in `../radeq_ledger_audit/VERDICT.md`.

---
## Headline
**The split-field is REAL and confirmed on both sides of the wire.** In consume mode there are two separately-maintained
binned fields, and different consumers read different ones:

| # | consumer | field it reads | source anchor |
|---|---|---|---|
| a | **Gph photoionization rate** (III→IV etc.) | **mc_J** (α=1.0 ⇒ pure MC shadow) | `lumina_plasma.c:5489-5501` (ground) / `5434-5442`,`5385-5393` (all-level); blend `J=α·mc_J+(1−α)·cs_J`, α=`LUMINA_COEVOLVE_PHOTOION_ALPHA`=1.0 |
| b | **bf photo-heating** `Hex` | **mc_J** (same integrand) | same Gph loop, `Hx` accumulator `5513`/`5454`/`5405` |
| c | **line cooling + line PUMP** `Λ_line` | **cs_J** (`nlte->J_nu`, NO blend) | `simul_line_term` `4867-4887`; `sh.l_BluJ/l_ABulJ` set from `Jb=nlte_get_J_at_nu` at `5562`,`5572-5573` |
| d | **fb / C_fb recomb cooling** | **no J** (rate-based `α·(χ+kT)`) | `simul_r1` `4900-4909` |
| e | **NLTE level-pop line J̄** | **per-line Sobolev field** (`jbar_line_det`→`jbar_line`→`cs_J` fallback) | `9563-9576`,`9591-9598`; NOT either binned field |
| f | **T_rad / W fit** | **MC estimators** (`nu_bar_estimator`/`j_estimator`) | `solve_radiation_field` `109-127` |

Field identities verified on the CUDA side (identification only): `cmfgen_write_jnu(&cs,&nlte)` (`lumina_cuda.cu:4798`)
writes the deterministic CMFGEN field `cs.J` into `nlte.J_nu` ⇒ **`nlte->J_nu` = cs_J**. The MC shadow is a **separate**
buffer `nlte_Jmc` ("kept OUT of the state", `4669-4671`); the MC normalize swaps it in and back
(`saved=nlte.J_nu; nlte.J_nu=nlte_Jmc; …; nlte.J_nu=saved;` `5250-5253`) so **`nlte->J_nu` is never overwritten by MC**;
`nlte_Jmc` is copied to `photoion_mc_J` and registered via `plasma_set_photoion_mc_field` (`5259-5265`) ⇒ **`g_photoion_mc_J`
= mc_J**. The CSV dumps `cs.J` vs `nlte_Jmc` (`5387-5400`). **Provenance note:** mc_J is produced by MC-transporting a field
sampled from cs.J (inject CDF built from `nlte.J_nu`, `5055-5063`,`5950-5958`) — so mc_J is a *reddened/scattered* image of
cs.J, which is exactly why the two disagree band-by-band (transport depletes the resonance forest, piles up in emission bumps).

So: **the ionization ledger (Gph) runs on mc_J; the thermal ledger's dominant term (line cool/pump) runs on cs_J; bf-heating
(0.05% of H_dep) is the only mc_J term in the thermal ledger.** The two fields disagree by **7–77× band-by-band**. Confirmed.

---
## FORK (bottom line)
**Split is REAL and band-large, but it is NOT the dominant driver of the deep-T_e deficit — and, counter-intuitively, the
current wiring is the *warmer* of the two pump assignments.** Materiality is asymmetric:

- **Thermal root, direct effect: SMALL-to-MODERATE.** The dominant non-deposition term (line cool/pump) reads **cs_J — the
  super-thermal field** (the "correct" one for pumping). bf-heating (the only mc_J term in the ledger) is 0.05% of H_dep, so
  which field it reads is immaterial to the root. Fully-coupled counterfactual (`radeq_ledger_s0.csv` root ladder): pump on
  **cs_J → 16617 K**, pump on **mc_J → 15460 K** — the split's marginal effect on the *pump root* is **~1150 K (0.03 dex)**,
  and the current assignment (cs_J) is the hotter one.
- **Ionization rate, direct effect: ORDER-OF-MAGNITUDE.** Gph reads mc_J, which in the EUV ionizing bands is **5–50× below
  cs_J** (838 Å: mc/cs=0.021; 404 Å Fe/Co III threshold: 0.135). Photoion is therefore EUV-starved vs what cs_J would give →
  keeps IGE under-ionized (in III, the run's known symptom vs benchmark IV).
- **The real +3400 K deficit is a LAG+HOLD artifact of the thermal ledger's OWN (correctly-assigned) cs_J consumer**, not the
  split — see §3.

The deep 13120→18760 K deficit decomposes (per `../radeq_ledger_audit`) as: **+3400 K** un-pumped/HOLD lever (thermal-ledger
timing, §3) + **+1660 K** global bath reddening (cs_J→CMFGEN, affects BOTH fields) + **+480 K** residual formula error. The
split-field's *own* independent contribution is the ~1150 K pump-field spread and the EUV ionization starvation — real, but
sub-dominant to lag/HOLD and to the reddening that starves cs_J and mc_J alike.

---
## 1. Consumer census (from source) — which field each consumer reads
Established above and in the table. Key mechanics:

- **(a) Gph photoion, α=1.0 ⇒ pure mc_J.** Blend `5495-5501`: `J = α·mc_J + (1−α)·cs_J`. Footer confirms
  `LUMINA_COEVOLVE_PHOTOION_ALPHA=1.0` ⇒ `J = mc_J` exactly. All-level path fired (`[GPH-ALLLEVEL] s0 Fe III G_all/G_gnd=40.7`,
  `stdout:216`); its two sites (`5385-5393`,`5434-5442`) apply the *same* α-blend, so all-level Gph is also pure mc_J.
  `g_gph_jtable` is NULL (no GPH_JTABLE, no `[GRADIENT-TRANSPLANT]` in stdout), so no jtable override — mc_J it is.
- **(b) bf photo-heating `Hex`** is accumulated inside the *same* Gph frequency loop from the *same* blended J
  (`Hx += … (hν − χ)`, `5513`/`5454`/`5405`) ⇒ **mc_J** at α=1.0.
- **(c) line cool + pump `Λ_line`.** `simul_line_term` (`4867-4887`) has **no no-pump guard**: `Rul=A·Bul·Jb·β`,
  `Rlu=Blu·Jb·β`, net `dE·(n_lo q_lu n_e − n_up q_ul n_e)` — signed, heating allowed. `Jb` is baked into `sh.l_BluJ/l_ABulJ`
  at `5572-5573` from `nlte_get_J_at_nu(nlte,s,nu_l)` = **cs_J** at the exact line-frequency bin. **No mc_J blend anywhere on
  this path.**
- **(d) C_fb, C_ff, C_ad** are rate/analytic (`4898-4909`) — read no radiation field.
- **(e) NLTE level-pop line pump** (`build_nlte…`, JBAR_POPS=3 + CMF_LINERES_JBAR=2) reads a **third** field: the per-line
  Sobolev `jbar_line_det` (fine-grid deterministic) overriding `jbar_line` (MC-realized), falling back to `cs_J`
  (`9563-9598`). This is neither binned field; it sets level departures `b_k` (⇒ emergent spectrum, and the ion pops that
  feed `simul_r1`), but is not itself a term in the thermal root.
- **(f) T_rad/W** come from the **MC transport moments** `nu_bar_estimator/j_estimator` (`109-127`) — MC lineage. T_rad
  (pinned 10470 K here) sets the initial T_e guess and the `W·B(T_R)` detailed-balance reference (`9491-9501`).

## 2. Quantify the split at s0–s2 (consumer × field-it-reads × band × mc/cs)
Band-integrated energy density `u = (4π/c)∫J dν` from `lumina_coevolve_field.csv` (`splitfield_bands.py`, table in
`consumer_table.csv`). cs_J = thermal-ledger field, mc_J = ionization-ledger field.

**s0 (T_e=13120):**

| band | dominant consumer (field) | u_cs | u_mc | **mc/cs** |
|---|---|---|---|---|
| EUV <912 | Gph(mc)+bf-heat(mc) | 4.81e-1 | 2.92e-1 | **0.61** |
| FUV 912-1290 | Gph excited(mc) / pump(cs) | 7.00 | 1.93 | **0.28** |
| res-pump 1490-1650 | line PUMP (cs) | 17.7 | 136.5 | **7.73** |
| **res-pump 1700-2100** (dominant heaters) | **line PUMP (cs)** | **158.6** | **6.25** | **0.039** |
| NUV 2100-3000 | line cool/pump (cs) | 144.0 | 20.6 | **0.143** |
| opt 3000-7000 | line cool (cs) | 84.2 | 135.5 | **1.61** |
| IR >7000 | far-IR forbidden cool (cs) | 25.8 | 33.8 | **1.31** |

**Per-bin probes (s0, exact `nlte_get_J_at_nu` lookup):**

| λ (Å) | note | cs_J | mc_J | mc/cs | cs/mc |
|---|---|---|---|---|---|
| 1526 | flag's 39× | 1.65e-4 | 6.43e-3 | **39.0** | 0.03 |
| 1857 | top Fe/Co II pump line | 3.92e-3 | 5.09e-5 | 0.013 | **77.1** |
| 2498 | NUV super-thermal | 4.96e-4 | 5.93e-5 | 0.12 | 8.4 |
| 839 | EUV ionizing | 1.53e-6 | 3.29e-8 | 0.021 | **46.6** |
| 404 | Fe/Co III bf threshold | 9.59e-12 | 1.30e-12 | 0.135 | 7.4 |
| 6000 | optical (agree) | 5.01e-4 | 5.04e-4 | 1.01 | 1.0 |

s1, s2 in `consumer_table.csv` — same structure, **sharper with depth**: the res-pump 1700-2100 mc/cs falls to **0.013 (s1)**
and **0.009 (s2)**; NUV to 0.066 / 0.054. The two ledgers diverge *more* the deeper you go.

**Reading of the table.** The **1526 Å 39×** in the flag is a *localized mc emission bump* sitting in the 1490-1650 band; it
is **NOT** where the pump lives. The **dominant heating lines are all 1760-1970 Å** (`radeq_diag.py` top-15: 1760, 1780, 1789,
1857, 1895, 1914, 1928-1970 Å — Fe II / Co II / Ni II resonance forest), i.e. the **1700-2100 band where cs_J is 26× ABOVE
mc_J**. So the pump consumer (cs_J) sees the super-thermal forest; had it read mc_J it would see a sub-thermal (0.039×) field
and not pump. Symmetrically, the Gph consumer (mc_J) sees the EUV 5–50× starved vs cs_J.

## 3. The "un-pumped" anomaly — settled with numbers
The ledger found `Λ_line = +1.064e-3` (net **cooling**, 71% of H_dep) at the committed root, *despite* cs_J being super-thermal
in the NUV/resonance bands. Mechanism, ruled in/out:

- **Does the pump read cs_J?** YES (§1c). **Wrong bins?** NO — `nlte_get_J_at_nu` is the exact log-bin lookup (`9289-9296`),
  same one the diag uses. **Structurally absent?** NO — `simul_line_term` allows heating and `simul_r1` does `C += Σ
  simul_line_term` (signed).
- **What the *assigned* field would do:** reconstructing `Λ_line(T_e=13120)` with the **FINAL dumped cs_J** gives
  **−1.074e-3 (net HEATING)** — 948,869 heating lines, **79% from β<0.1 thick lines**, all in the 1760-1970 Å forest with
  cs/B(T_e)=8–23× (`radeq_diag.py`). Thermal-pump check `Λ_line(Jb=B(T_e)) = +3.2e-6 ≈ 0` confirms the formula obeys detailed
  balance. So the field assigned to the pump (cs_J) **is** super-thermal and **would** flip the term to strong heating.
- **Yet the run committed +1.064e-3 (cooling).** These two facts are only simultaneously true if the run **did not consume the
  final super-thermal cs_J** in the pump. The committed root (13120 K) matches the **zero-pump** coupled root (13214 K),
  **not** the cs_J-pump root (16617 K) — `radeq_ledger_s0.csv` ladder. Combined with the field growing **20×** over iters
  while T_e[0] is **digit-identical (13119.874754) for iters 2–11** (`stdout:242,…,29938`) — the fingerprint of a `pin_lo`
  HOLD (`5639-5640`) — the mechanism is: the T_e solve consumed the **early, ~20× weaker (sub-thermal) cs_J** at the iteration
  it committed (iter 2), then **froze** (cold-branch HOLD) while cs_J matured to super-thermal.

**Verdict on the anomaly:** it is **NOT** a wrong-field / wrong-bin / missing-pump defect. It is a **LAGGED-FIELD + COLD-BRANCH-
HOLD timing artifact of the thermal ledger's own, correctly-assigned cs_J consumer.** The +3400 K "un-pumped lever" is a
*convergence* pathology, distinct from — though co-resident with — the mc_J/cs_J split. (Caveat carried from the radeq audit:
the exact per-iteration field at s0 is not dumped, so "the run consumed the early weak cs_J" is inferred from the root match +
digit-identical HOLD, not read directly; a one-line runtime probe of `Λ_line`/held-flag at s0 would nail it.)

## 4. Verdict + unification options (no code)
**Is the split real and material?** REAL — confirmed on both sides of the wire, 7–77× band-by-band, sharpening with depth.
MATERIAL as an **architectural inconsistency** (the ionization and thermal ledgers genuinely consume fields that disagree by
order-of-magnitude), and **order-of-magnitude for the photoion RATE** (EUV mc_J 5–50× below cs_J → IGE-in-III). But **NOT the
dominant term in the deep-T_e deficit**: the deficit is led by the lag/HOLD (+3400 K, §3) and the global bath reddening that
starves *both* fields (+1660 K); the split's *own* first-order effect on the thermal root is the ~1150 K pump-field spread,
and the current wiring already puts the pump on the hotter (cs_J) field.

**CMFGEN-divergence framing.** This split has **no analog in CMFGEN**, which converges a *single* J_ν per iteration that drives
statistical equilibrium (ionization AND level pops that set line cooling) **and** radiative equilibrium simultaneously. CMFGEN
never maintains two fields; LUMINA's cs_J-for-thermal / mc_J-for-ionization is a pure coevolve artifact. So the orthodoxy target
is *one field, all rates*.

Options (each with its risk):

1. **All consumers on mc_J** (fully MC-transported, ARTIS-orthodox realized field). *Pro:* single self-consistent field that
   carries trapping/scattering; removes the split; matches ARTIS's "estimator IS the field." *Con:* mc_J is **shot-noisy**
   (already needs the OCC zero-count guard, `4966-4968`); it is **more reddened/EUV-starved** than cs_J, so the pump root drops
   to **15460 K** (colder than cs_J's 16617) and photoion stays EUV-starved. Trades staleness for noise **and gives up the
   super-thermal pump.** Net: likely *colder*, not warmer.

2. **All consumers on cs_J** (deterministic CMFGEN field — the campaign's stated standard). *Pro:* noise-free, orthodox to the
   pure-CMFGEN benchmark; pump keeps its super-thermal cs_J (**16617 K** achievable); photoion gets the EUV field **5–50×
   higher** → pushes IGE toward IV (the campaign goal). Puts both ledgers in **one universe**. *Con:* cs_J is **lagged** (the
   exact staleness that, with the HOLD, caused the un-pumped freeze) and **does not carry MC reddening**, so photoion re-inherits
   the **too-blue over-ionization** the P1/mc_J rewiring was introduced to cure (`5490-5501` rationale). Unifying on cs_J only
   helps **if the lag/HOLD is fixed too** (else you unify two ledgers onto a field the solve still freezes on).

3. **Hybrid — one blended field per band, both consumers, α chosen physically; plus break the HOLD.** The real disease is (i)
   two separately-maintained fields and (ii) the pin_lo freeze. Fix (i): make **one** `J(s,b) = α·mc_J + (1−α)·cs_J` that
   **both** Gph *and* the line pump read (route the pump's `Jb` through the same blend `simul_line_term` currently bypasses),
   so the ionization and thermal ledgers can never diverge — α set by band physics (MC-reddened in scattering-dominated
   continuum where cs_J is spuriously blue; deterministic cs_J in the optically-thick resonance forest where MC is noisy and
   under-tallied). Fix (ii): let `radeq_simul_all` **re-evaluate** the root when the field has grown materially rather than
   `pin_lo`-HOLDing the cold branch, so the pump can cash the matured field. *Risk:* α is another knob (campaign NO-OVERFITTING
   rule) — it must be justified per-band by the MC-vs-deterministic noise/reddening tradeoff, not tuned to a target T_e; and
   re-opening the HOLD risks the 140 kK strip-attractor the HOLD was added to suppress (`5528` note), so it needs the LOWEST-
   root bracket kept.

**Recommendation (analysis-only):** the split is worth removing for *orthodoxy* (one field, per CMFGEN), but on its own it will
not close the deep-T_e gap — **the lag/HOLD (§3) is the larger lever and should be settled first** (a runtime `Λ_line`/held-flag
probe at s0). Unifying the field without fixing the HOLD merely puts both ledgers on a field the solve still freezes on.

## Artifacts (this directory)
- `VERDICT.md` — this file
- `consumer_table.csv` — shell × band × consumer(field) × u_cs vs u_mc × mc/cs ratio (s0,s1,s2)
- `splitfield_bands.py` — band integrator + per-bin probes (read-only; regenerates the table)

## Source / data relied upon
- `src/lumina_plasma.c`: `4867-4887` (`simul_line_term`, no no-pump guard, reads cs_J), `4889-4931` (`simul_r1` balance),
  `4900-4909` (C_fb rate-based), `5385-5393`/`5434-5442`/`5489-5501` (Gph α-blend ⇒ mc_J at α=1), `5405`/`5454`/`5513` (Hex),
  `5562`,`5572-5573` (line Jb=cs_J), `5626-5667` (LOWEST-root + pin_lo HOLD), `109-127` (`solve_radiation_field` T_rad/W from MC
  moments), `9289-9296` (`nlte_get_J_at_nu`), `9563-9598` (NLTE per-line jbar), `4938-4968` (`plasma_set_photoion_mc_field`, OCC)
- `src/lumina_cuda.cu` (identification only — file under concurrent edit; cited by function+anchor): `cmfgen_write_jnu`
  writing cs.J→`nlte.J_nu` (~4798), `nlte_Jmc` shadow alloc/keep-out-of-state (~4669-4671), MC-normalize swap keeping
  `nlte.J_nu` intact (~5250-5253), `photoion_mc_J` copy + `plasma_set_photoion_mc_field` register (~5259-5265), field CSV dump
  cs.J vs nlte_Jmc (~5387-5400), inject CDF from `nlte.J_nu` (~5055-5063,5950-5958)
- `logs/coevolve_consume_a10_kx_gphall/`: `lumina_coevolve_field.csv` (cs_J,mc_J per shell,bin), `lumina_plasma_state.csv`
  (committed T_e/n_e/W/T_rad), `stdout.log` (:9,23,66,89 env; :216 GPH-ALLLEVEL; :217+ SIMUL pins; :31086-31182 RUN FOOTER)
- `../radeq_ledger_audit/`: `radeq_ledger_s0.csv` (term ledger + root ladder), `radeq_diag.py` (exact-bin cs_J pump = −1.074e-3),
  `radeq_ledger.py` (H_photo mc/cs/jtable, band u), `VERDICT.md` (the flag)
