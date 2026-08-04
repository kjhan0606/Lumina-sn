# Deterministic Emergent — Ladder V2 (rewritten 2026-06-26)

Rewritten after 3 failed observer-frame line-transport attempts. The V1 ladder
(DETERMINISTIC_EMERGENT_BUGTABLE.md) chased fluorescence as the green fix; that was
**falsified** — gold's green is a continuum-color feature, not lines (see SOLID #6).
This V2 reflects what is actually solid, what the obs-extractor saga taught, and the
re-scoped remaining problem.

GOAL: emergent spectrum matching DDC15 gold (0.976d) — peak ~6595Å, grn/nir 0.58,
sharp P-Cygni (Si II 6355, Ca II H&K/NIR, O I 7774), **energy-conserving**.

---

## A. SOLID — verified, do NOT re-litigate

| # | Result | Evidence |
|---|--------|----------|
| S1 | **Plasma** T_e/T_gold 0.98, n_e dex 0.18 | champion config (LSTAR/LINE_RE/FROZENIN/JNU_PHOTOION + orthodox NLTE closure) |
| S2 | **Continuum color** = static `cmfgen_fine_emergent` peak **6502 ≈ gold 6595**, CONSERVING (L=3.90e39) | freq-resolved fine field cures binned-J grey collapse |
| S3 | **Advection optical-transport bug FIXED** (`ADV_SPLIT`): Courant β·ds~80 → e^−80 killed optical field → J/B 1e-4; implicit-upwind fix → **0.89** | 169993; physics-agent + ALAM=0 falsifier |
| S4 | **Fluorescence mechanism** activates (super-thermal optical lines 8→54) but is **ORTHOGONAL to the green** | gold green = pseudo-continuum, not lines |
| S5 | **Beaming D⁴ is real, not a bug**: CONTONLY-obs +38% over static at β_outer=0.23 | physics-agent Lorentz derivation; CONTONLY-obs peak 6242 bluer than static (beaming blueshifts) |
| S6 | **Gold's green 5600-7300 = smooth pseudo-continuum** (roughness 0.4%/step), peak 6595=~4400K color — NO red Fe II/Co II line forest to pump | physics-agent direct measurement of gold |

**Implication of S2+S6:** the static extractor ALREADY has the right conserving color and
the (narrow, comoving) line dips. The whole remaining problem is the **observer-frame
Doppler P-Cygni broadening**, done **conservingly**.

---

## B. The OBS-extractor problem — 3 attempts, all failed, + the methodology lesson

| Attempt | Energy (full/static) | Features | Color | Verdict |
|---------|----------------------|----------|-------|---------|
| Sobolev-jump (W·B / J̄_l source) | **0.47** (−53% leak) | present | reddened 8544 | non-conserving |
| FAITHFUL (transport static chi_tot+Sbin) | 1.19 ✓ | **featureless** (roughness 0.011) | 6408 ✓ | no P-Cygni |
| 2-pass SEI (true τ_S + beamed Jbar_C) | **1.72 ✗ (lines +24% over-emit)** | broad-only | **too-blue 4354** | CONVERGED: non-conserving + too-blue |

**Root difficulty:** an observer-frame line transport that is simultaneously
(a) energy-conserving and (b) reproduces gold's SHARP P-Cygni. Each attempt got one, not both.

**⚠️ METHODOLOGY LESSON (the "나왔다 안나왔다" confusion):** I repeatedly measured/reported
**mid-run (non-converged) obs** files, which are overwritten each iteration. The SAME run
170654 read peak 6408 then 4353 at different iters — **with the plasma FIXED (CONSUME=0,
T_e constant)**. Two problems: (1) reporting mid-run snapshots = inconsistent, unreliable
(my error — fixed: measure ONLY `done` runs, once); (2) the obs spectrum CHANGING across
iters at fixed plasma signals the **obs extractor itself is iteration-unstable** — a real
flag that may underlie all 3 "attempts." **Diagnose the obs iter-stability BEFORE trusting
any obs comparison.**

---

## C. Remaining problem — re-scoped, with candidate paths

The remaining gap is narrow and well-defined: **a conserving observer-frame emergent that
reproduces gold's sharp P-Cygni**, given the already-correct conserving comoving field (S2).

| Path | Idea | Pro | Con / risk |
|------|------|-----|-----------|
| **P-α: fix obs iter-stability first** | Why does obs change at fixed plasma (6408↔4353)? Producer field or SEI Pass-1 Jbar_C not converged. | Prerequisite — all 3 attempts may be unreliable until stable | must isolate (run N_ITER sweep, compare converged obs) |
| **P-β: MC emergent (THEN_MC) on converged orthodox plasma** | Monte-Carlo macro-atom NATURALLY does conserving P-Cygni (TARDIS/ARTIS). Feed it the now-good plasma. | MC is the natural P-Cygni+conservation tool; deterministic obs-march is fighting it | MC blue-tilt bug (168221, T_inner controller) must be checked; but that's a known separate fix |
| **P-γ: Λ-iterated SEI** | Iterate Pass-2 S_l = obs J̄ to self-consistency (physics-agent fallback) | conserving by construction | expensive; doesn't fix the sharp-vs-broad if NObs/DVRES limited |
| **P-δ: static + validated Doppler** | static (conserving, color, narrow dips) + a conservation-checked velocity convolution per resonance | cheapest if asymmetry handled | symmetric convolution kills P-Cygni asymmetry (physics-agent: invalid for shapes) |

**Per-stage gate (any path):** converged (not mid-run!) AND energy conserved (full/static-beamed
≈1.0-1.4) AND roughness ≈ 0.03 (gold, sharp) AND peak 6595±100 AND grn/nir 0.58±0.03 AND
deep dips (Ca H&K, O I < 0.15).

---

## D. Next step (decision)

**FIRST: P-α — diagnose the obs iteration-instability.** The flip-flop (6408↔4353 at fixed
plasma) means every obs comparison so far is on shifting sand. One clean experiment: a single
converged run, dump the obs at each iteration, see if it stabilizes; if not, find why the
producer field / SEI Pass-1 changes at fixed plasma. **No new obs-extractor variants until the
existing one is stable and measured converged.**

**THEN reconsider P-β (MC emergent)** as the likely-correct tool: the deterministic obs-march
has cost 3 failed attempts on what MC does natively. The orthodox plasma (S1-S6) is the hard-won
deliverable; the emergent transport may be better served by THEN_MC than by hand-built obs-march.

## Cross-cutting rules
- **Measure ONLY converged (`done`) runs, once.** No mid-run snapshots (the V1→V2 lesson).
- One knob per A/B; verify config reached the binary (SOB_EPS A/B confirmed the env path works).
- Plasma gate on every run (T_e 0.98, n_e dex 0.18).
- North Star: orthodox/correct first, optimize later [[feedback_orthodox_first_optimize_later]].

## Links
[[project_autonomous_stage2_2026-06-25]], docs/ORTHODOX_FREQRES_NLTE_DESIGN.md,
docs/GOLD_FEATURE_FINETUNE_TABLE.md, docs/DETERMINISTIC_EMERGENT_BUGTABLE.md (V1, superseded).
