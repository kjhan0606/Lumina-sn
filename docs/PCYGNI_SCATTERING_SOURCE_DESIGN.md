# P-Cygni scattering line-source (ETLA/SEI) — finalized design

**Status:** design FINALIZED 2026-06-18 (implementation deferred to a dedicated session).
Triple-verified: my draft + codex (019ed80e) + physics agent (ac6efe0e), strongly convergent.

## Problem (verified, not assumed)

The deterministic observer-frame formal integral (`compute_formal_integral_spectrum`,
`lumina_plasma.c:8348`) does **not** reproduce P-Cygni profiles. Measured: Ca II NIR 8542
blue/red flux ratio = 0.65–0.68 vs gold CMFGEN 0.25–0.31; invariant to every population
fix (b_k cap, collision fix, LTE zone all give ~0.65). The machinery is fine
(`I += S_l*(1-e^-tau)*e^-tau_acc`, inner-boundary `B(T_inner)` on core rays). The defect is
the **line source is thermal**: Ca II 8542 has `S_l/B(T_e)=1.00` and `J_line/S_l=1.00` at
**all** shells incl. the line-forming `tau~1` shells (sh13–15, T_e~2880 K). A line emitting
`S_l=B(T_e)` ≈ the continuum brightness refills its own absorption → no trough.

**Physics:** Ca II 8542 (A_ul~1e6) at n_e~1e8 has critical density A/q~3e13 ≫ n_e, so it is
radiation-dominated and **should scatter** (S_l = J̄ = diluted nonlocal photospheric field
W·B(T_phot), *fainter* than the photosphere → absorption trough). Instead the in-line J̄ is
itself ≈B(T_e) (over-thermalized to the LOCAL Planck), so the rate solve gives thermal
populations and S_l=B(T_e). This is a pre-existing line-source-closure issue, **orthogonal
to and unaffected by** the just-committed population fixes (van Regemorter trap etc.).

## Faithful design — ETLA Sobolev source (= Sobolev-with-continuum / SEI)

Per line, per shell, replace the thermal line source with the equivalent-two-level-atom
Sobolev scattering source:

```
S_l = ( eps*B(T_e) + (1-eps)*beta*J_inc ) / ( eps + (1-eps)*beta )
```

- `beta` = Sobolev escape probability `(1-e^-tau)/tau` (already computed; large-tau branch
  `tau>700 => 1/tau` is correct).
- `eps`  = destruction probability `C_ul/(C_ul + A_ul)` (collision rates now correct after
  the van Regemorter fix). For Ca II 8542: eps ~ n_e/n_crit ~ 3e-6 ≪ 1 ⇒ deeply scattering.
  (More exact: `eps = C_ul*(1-e^-hnu/kTe)/(A_ul + C_ul*(1-e^-hnu/kTe))`; numerically same here.)
- `J_inc` = **continuum-only** incident mean intensity at the line frequency, from the
  deterministic continuum field (chi_es+chi_bf+chi_ff, **excluding chi_line**). NOT the
  line's own binned J̄ (which is the contaminated ≈B(T_e) field — feeding that back
  reproduces the bad answer and re-introduces the self-coupling runaway).

Limits: eps→0 ⇒ S_l→J_inc (scattering, P-Cygni); eps→1 ⇒ S_l→B(T_e) (thermal). One-shot
closed form (NOT a feedback iteration), so it does **not** crash like the documented frozen-
S_l-into-J/T_e runaway — the S_l↔J̄_line loop is broken by using a FIXED continuum J_inc.

### Decoupling (stability + honesty)

Use this scattering S_l in the FORMAL INTEGRAL (post-convergence spectrum synthesis) ONLY.
Keep the plasma solve (T_e, n_e, ionization — validated vs gold) on its thermal closure; do
**not** feed the scattering S_l back into T_e/J. This is the standard SEI posture (Lamers,
Cerruti-Sola & Perinotto 1987; Mazzali & Lucy 1993) and is honest for line-profile
MORPHOLOGY. Conditions (codex+agent): opacity from converged populations; emissivity =
chi_l*S_ETLA; J_inc excludes line self-emission; not fed back into Te/J.

Known inconsistency: the plasma cooled with Ca II radiating as B(T_e); the spectrum now says
it scatters (removes less energy). Acceptable iff the re-sourced lines are minor coolants.
**Quantify** with the existing probe `4π∫(χ_line·J − η_line)dν` (`lumina_cmfgen.c:647`):
compute net line cooling both ways for the strong lines, per shell. <~few % of total cooling
⇒ decoupling honest. 10s of % ⇒ MALI/A4 justified.

## THE GATE — Phase 0 falsifier (run BEFORE writing ETLA code)

**Both reviewers: the design's load-bearing assumption (J_inc ≪ B(T_e)) is UNVERIFIED and
has a failure precedent** (the inner-FUV "J over-thermalized to local cold B within <1 shell"
root). If the continuum J_inc at 8542 is itself ≈B(T_e), ETLA buys nothing.

**Falsifier:** at the **continuum bin nearest 8542 Å**, shells 13–15 (T_e~2880 K), read J/B
and compare to the geometric dilution W (cmfgen_validate prints τ_r, W, B, S, J, J/B, J/S at
`lumina_cmfgen.c:611-645`; or read `lumina_cmfgen_jnu.csv` from LUMINA_CMFGEN_JDUMP=1 — 8542
is a low-line-opacity red window so binned J ≈ continuum J_inc).

- **PASS (build ETLA):** τ_r(8542) ≲ 0.3 AND `J/B ≈ W` (≈0.1–0.25, clearly <0.5, tracking
  dilution, not 1.0) ⇒ J_inc = diluted photospheric ≪ B(T_e) ⇒ S_l→J_inc gives a real
  trough. Predicted blue/red → ~gold 0.25–0.31.
- **FAIL (fix continuum FIRST):** `J/B ≈ 1` at sh13–15 despite τ_r<1 ⇒ continuum transfer is
  over-thermalized (diagonal/tri-ALI recirculating local B in the thin red continuum) ⇒ no
  line-source fix can make a trough. Repair the continuum scattering transport (anchor the
  escaping photospheric field / extend tri-ALI to τ≪1, r→1) before ETLA.

One number decides the whole design: **J/B at the 8542 continuum bin, sh13–15.**

### Phase-0 RESULT (job 167211, 2026-06-18): **FAIL — continuum is over-thermalized.**

Measured J/B(T_e) at TRUE continuum windows (chi_line/chi_es < 0.3, 7000–7300 Å) at the
Ca II line-forming shells: sh13 = 0.88–1.14, sh14 = 0.62–0.93, sh15 = 0.57–0.82. The
diluted-photospheric target (W·B(T_phot)/B(T_e), Wien at NIR with T_phot=4434≫T_e≈2880) is
≈1.3–1.8. So the continuum J is ≈B(T_e) or below — NOT the bright photospheric field. The
photospheric continuum is NOT propagating to the line-forming shells at diluted strength; it
is over-thermalized/suppressed to the local Planck. (Caveat noted: the first naive read hit
pitfall-2 — the 8542 bin itself is line-DOMINATED, chi_line/chi_es~300; the result above uses
genuine continuum windows.)

**WHY the continuum is over-thermalized — CORRECTED, verified diagnosis (167211 opacity
decomposition at 7100 Å).** Two earlier hypotheses were tested against the dump and BOTH
REFUTED:
  - "diagonal/tri-ALI floor (Λ*≈1 stall)": REFUTED — Λ* = 0.002-0.006 (tiny, thin medium), and
    J/S_fixed = 4-45 (the scattering solve IS pulling J well above the thermal seed; not stalled).
  - "separate continuum-transport bug": REFUTED — true absorptive opacity is negligible
    (chi_abs/chi_es = ε ~ 0.01→0.000), so bf/ff thermalization is NOT the cause.
**ACTUAL mechanism = LINE-BLANKETING thermalization.** At the inner-mid shells (sh0-9) the
"continuum" at 7100 Å is dominated by the iron / expansion-opacity LINE FOREST: chi_line is
7-22× chi_es (sh0: 2.1e-13 vs 3.0e-14; sh5: 8.2e-14 vs 3.7e-15). That forest emits THERMALLY
(S_l = B(T_e), same root as the Ca II finding). A thick + thermal forest is a THERMALIZING
SCREEN: it absorbs the photospheric field and re-emits B(T_e), so the field that reaches the
Ca II line-forming shells (sh13-15) has already lost its photospheric memory and sits at
≈B(T_e_inner). The continuum is NOT separately broken — it is thermalized BY the thermal line
forest, the SAME thermal-line-source root acting through blanketing.

**Consequence — design CORRECTED:** a DECOUPLED one-shot ETLA on Ca II alone CANNOT work: the
field Ca II would scatter is itself thermalized by the forest. The fix must make the WHOLE
forest scatter/fluoresce (Fe II etc. are physically resonance/scattering lines that should
redistribute the photospheric field in frequency, NOT destroy it thermally), and because the
field and all line sources are coupled through blanketing, this is the **self-consistent
MALI/A4 problem**, NOT the lighter decoupled SEI path. The Phase-0 gate "FAIL" stands; the
re-ordered first task is the self-consistent scattering/fluorescence line transfer for the
forest (= the long-standing "fluorescence missing" item). The decoupled ETLA-in-formal-integral
(Phase 1 below) is demoted to a possible LATER cross-check, not the first step.

### Codex AGREES with the corrected diagnosis (019ed80e, 2026-06-18) — A/B/C confirmed, +refinements:
- **Caveat (important):** not every forest line is PURE coherent scattering. Many Fe/IGE lines
  FLUORESCE via radiative branching (redistribute photons in frequency, not coherent). Decide
  per-line by the rates, NOT species labels: thermal-destruction = collisional-deexcitation /
  total-deactivation; scattering/fluorescence = radiative-deactivation / total. At SN Ia near
  max, n_e~1e8, most permitted Fe/IGE lines (A~1e7-1e9) are NOT collisionally thermalized =>
  should scatter/fluoresce. Some low-lying/forbidden can be near-thermal => use ε per line.
  ⇒ the source closure should be a branching/macro-atom view, not just two-level ETLA.
- **MINIMAL FAITHFUL PATH (codex):** do NOT jump to full A4-in-Newton. (1) keep validated plasma
  FIXED; (2) global line-forest MALI/Sobolev source iteration on the dominant-opacity lines
  (S_l = ETLA/branching, Λ*≈1-β); weak-opacity lines stay thermal initially (they don't control
  the field); (3) iterate J_nu ↔ forest S_l to convergence; (4) feed back into NLTE pops/T_e
  only LATER (A4 next tier) if the fixed-plasma forest-MALI still can't match or if cooling
  shifts materially. So: **global line-forest MALI on fixed plasma FIRST.**

### NEXT FALSIFIER (codex; run BEFORE the MALI build, to confirm the forest is the thermalizer):
Matrix-free intervention on the existing J solve: for the lines/shells where chi_line >> chi_es,
override the forest line source from B(T_e) to a scattering value (S_forest = J_old, or
diagnostically S_forest = W·B(T_phot)); recompute the deterministic J_nu.
**PASS:** J/B(T_e) at Ca shells 13-15 (7000-7300 Å) jumps from ~0.6-1.0 toward the diluted-
photospheric ~1.6, AND the Ca II ETLA S_l drops below B(T_e) ⇒ thermal forest confirmed as the
upstream thermalizer ⇒ build the forest MALI.
**FAIL:** if J/B does NOT rise, the lost photospheric memory is elsewhere — inner-boundary
dilution, expansion-opacity treatment, frequency redistribution, or continuum ray geometry —
diagnose that instead.

### FALSIFIER RAN (jobs 167224 coupled; 167228/167229 frozen-attempt) — RESULT: REQUIREMENT ESCALATED.

The existing transfer-only forest-scattering knob `LUMINA_CMFGEN_LINE_EPS_UV=0.03` (forest 97%
scattering in the J transport) was used. **Result: T_e CRASHES to the 1000 K floor over 23/49
shells** (sh0-24 all at 1000 K; outer ~1376 K), in the coupled run AND with `RADEQ_TE=0`. Env
flags do NOT freeze the plasma (the coupled-Newton / line-RE path still drives T_e). The clean
directional J read ("does forest scatter raise J toward W·B(T_phot)?") was therefore BLOCKED by
the crash — J came out LOWER (collapsed plasma), uninterpretable.

**But the crash IS the decisive finding:** turning the iron forest from thermal to scattering
removes a DOMINANT GAS-HEATING term and the gas cools to the floor. The thermal forest
(line-RE chi_line·(J−B), with J>B because the forest sees the hot photospheric UV) is a major
gas HEATER — it absorbs photospheric UV and thermalizes it into gas heat (ε=1). Pure scatter
(ε→0) removes all that heating → catastrophic cooling. This is codex's caveat-B regime: the
re-sourced lines carry a DOMINANT share of the energy budget, not a minor one.

**CONSEQUENCE — design requirement ESCALATED beyond the "light forest-MALI on fixed plasma":**
1. A morphology-only spectrum (decoupled SEI) now needs a TRUE plasma-FREEZE code module — hold
   T_e/n_e/populations/opacities at the validated values and re-solve ONLY J with forest
   scattering, then synthesize. Env flags cannot do this; it is a code path.
2. A SELF-CONSISTENT solution needs the FULL A4 (coupled Newton with the line source as unknown,
   solved together with the gas energy balance), because the forest scattering↔heating coupling
   is load-bearing. The lighter "forest MALI on fixed plasma" iteration cannot hold the plasma
   fixed in reality.
3. The per-line ε is now CENTRAL to the energy budget, not just the line profile: ε=1 (current)
   over-thermalizes ALL absorbed UV into gas heat; ε=0 under-heats and crashes. The faithful ε
   (small but nonzero) + the fluorescence energy redistribution (where the absorbed UV re-emerges
   in frequency) IS the A4 problem. The validated T_e may currently be propped up by the forest's
   ε=1 over-thermalization heating — a caveat to re-examine when ε is corrected.

So the dedicated session starts NOT with a light forest-MALI but with: (a) a true plasma-freeze
morphology module (fast, isolates whether forest-scatter restores the photospheric J/profiles),
and/or (b) the full A4 self-consistent line-source + energy solve. The forest-thermal-is-a-major-
heater finding is the gate that chose A4-class over decoupled-SEI.

## Phased plan

- **Phase 0 — GATE:** the J/B-vs-W falsifier above. Branch on PASS/FAIL.
- **Phase 1 (if PASS):** compute continuum-only J_inc(nu_line, shell) (exclude chi_line);
  evaluate ETLA S_l per line/shell; use it in the formal integral only. Validate: Ca II /
  Si II / S II blue/red ratios → gold (~0.25–0.31); super-thermal stays killed; T_e
  untouched.
- **Phase 2 — consistency:** cooling delta (thermal vs scattering source) via the :647 probe.
  <few % ⇒ ship decoupled. Else → Phase 3.
- **Phase 3 (only if needed):** self-consistent MALI on the line field
  `S_l[1-(1-eps)Λ*] = eps*B + (1-eps)(Λ-Λ*)S_l^old`, Λ*_diag = 1-beta (extends the existing
  tri-ALI). Or A4 in-Newton source. Required only if (i) re-sourced lines are major coolants,
  (ii) line overlap/blanketing matters (Fe curtain; Ca II triplet mutual scattering), or
  (iii) absolute flux/color (not just shape) must be trustworthy.

## Pitfalls (physics agent)

1. β→0 thick limit: S_l→B regardless of J_inc (correct, can't run away). Intermediate
   β~eps: denominator delicate (the `+1e-300` guard at cmfgen.c:192 is right).
2. Continuum-only J_inc: pull from chi_es+chi_bf+chi_ff at the comoving line frequency where
   the line forms; never from a bin polluted by the line's own opacity.
3. Expanding-flow anisotropy: the β-source is angle-averaged/isotropic; the blue/red P-Cygni
   asymmetry comes from the **p-z ray geometry + velocity field** (already in the formal
   integral), NOT from S_l. So expect the TROUGH (blue absorption) to be the clean test; the
   emission peak from an isotropic S_l is less perfect.
4. Overlapping lines (Ca II 8498/8542/8662 triplet; Si II 6355 doublet; Fe forest): isolated-
   line ETLA drops one line's emergent field as a neighbor's incident field. OK for first
   profiles; a known SEI limitation → Phase 3.
5. ε accuracy: the thermal/scattering split hinges on ε=C_ul/(C_ul+A_ul); relies on the
   now-correct collision rates.

## References
Castor 1970 (Sobolev escape); Castor-Abbott-Klein 1975; Olson-Auer-Buchler 1986 (ALI/Λ*);
Hummer & Rybicki 1985 (Sobolev+continuum); Lamers-Cerruti-Sola-Perinotto 1987 (SEI);
Mazzali & Lucy 1993 (SN Ia Sobolev/SEI P-Cygni).

## Code anchors
- Formal-integral line source: `lumina_plasma.c:8478` (`S = opacity->line_source_S[...]`).
- ETLA net-heating weight already present: `lumina_cmfgen.c:192` (`el*be/(el+be-el*be)`).
- J/B-vs-W diagnostic (Phase-0 gate): `cmfgen_validate`, `lumina_cmfgen.c:611-645`.
- Line cooling probe (Phase-2): `lumina_cmfgen.c:647`.
- Sobolev tau / beta: `lumina_plasma.c` tau_sobolev; tri-ALI: LAMBDA_TRI=1.
