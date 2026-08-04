# Orthodox Frequency-Resolved NLTE Design (2026-06-26)

**Decision (user, 2026-06-26):**

> **North Star (user, 2026-06-26 재확인):** prove orthodox = gold-equivalent FIRST (cost no object — full-range, full convergence, NO approximation), THEN optimize. Correctness is the deliverable; the ~40min/solve full-range cost is acceptable in this proof phase. A cheap narrow-window *mechanism gate* before the expensive full run is valid ordering, not a cost shortcut. [[feedback_orthodox_first_optimize_later]]
 stop chasing cheap window/pump shortcuts. Build the
**orthodox** method — the full-wavelength frequency-resolved radiation field driving
the NLTE level populations self-consistently (CMFGEN's core) — then make *that* fast
by computation (tensor cores + GPU), **not** by approximating the physics away.

This document is the build spec. It folds in every diagnostic verdict from the
fluorescence campaign so we build the right thing once.

---

## 1. Why this is necessary (diagnostic verdicts that closed every cheap path)

The emergent has a **green deficit / blue excess** (user band decomposition, 169874):
blue 3000-5000 model/gold **6.5×**, green 5600-7300 **0.57×**, 7700 dip too shallow,
NIR matches. This is missing **UV→optical fluorescence**. The campaign falsified, in order:

| Cheap path tried | Verdict | Evidence |
|---|---|---|
| forest-overlap (more UV pump lines) | DEAD | 169884/911/914: 8983→16441→30547 lines, green S_l **byte-identical** (8 lines, max 1.45) |
| orthodox LTE floor caps green | NOT the cap | 5170 sub-thermal (0.46-0.67) = resolved, not floored (`lumina_cuda.cu:982`) |
| producer line-scattering ε<1 | NO lift | 169955 ε=0.1: green-shell UV J̄/B **identical** to thermal (1.000) |
| ALI under-convergence confound | FALSIFIED | sh24 J/B=9 (moved off warm start) while sh11-15=1.000 → solver not frozen |
| blue 3000-4500 pump window | REFUTED | 169972 jmap: blue **not** super-thermal (J/B≈0) at green shells |

**Root cause (triple-verified, `lumina_cmfgen.c:1736`):** the producer emits the
in-window forest with **thermal** source `S_l→B(T_e)`, so the line-resolved J̄ at the
green-forming shells thermalises to B(T_e) (measured J̄/B=1.001, 2300-2600Å, sh11-15).

**Why even the correct pump field can't be harvested cheaply (jmap 169972):**
the super-thermal pump field *does* exist but only in the **far-UV 2000-2500Å
inter-line continuum** (green sh13: J/B=133 @2050Å, 1.85 @2450Å, <1 beyond 2550Å).
The strong resonance lines (2382Å, τ_S=1.2e5) **thermalise their own cores** → their
line-averaged J̄≈B → the consumer's line-J̄ pump cannot see the inter-line super-thermal.

**The physics the cheap path structurally cannot do (physics review):** real Fe II
fluorescence is the **branching** of an absorbed UV photon into optical decay channels
(z⁶P°→a⁶S = 5170Å), which happens **even when J̄=B**, *provided the optical line emits
from the solved multi-level population instead of coherently scattering its own binned-J*.
Our optical lines are locked to binned-J coherent scatter (5170 J̄/B≈W) → the branched
green photon is re-absorbed → no net green. **Only a fully coupled multi-level solve with
all lines frequency-resolved produces fluorescence.**

**New flag (jmap 169972):** the *fine producer's* optical field **collapses** at outer
shells (sh13 J/B≈1e-4 vs sh0 0.54). The wide-window `cmf_solve_J` is not transporting the
optical continuum outward (binned gives ~W·B). This is a prerequisite bug — fixed in Phase 0.

---

## 2. The orthodox architecture (target)

Solve, self-consistently to convergence:

```
  J_ν(full grid, all λ)  ──►  J̄_l = ∫φ_l J_ν dν   (every bb line)
                          ──►  R_bf = ∫σ_ν J_ν dν   (every bf edge)
        ▲                              │
        │                              ▼
   source functions  ◄──  NLTE rate matrix  ──►  level populations n_i
   S_l = (from n_u,n_l)                          (branching is automatic:
   S_c = (χ_abs B + χ_es J)/χ                     n_i set by ALL in/out rates)
```

Two non-negotiable differences from the current (cheap) code:

1. **Every transition uses the frequency-resolved J̄** — no binned-J anywhere in the
   rate matrix, no window gate. (Currently `LUMINA_CMF_LINERES_CONSUME` is in-window
   only; out-of-window lines fall to `nlte_get_J_at_nu` binned, `plasma.c:7051`.)
2. **Optical line emission uses the solved population** (S_l from n_u/n_l), not coherent
   scatter of binned-J. This is what lets the UV-pumped population radiate green = the
   branching. The emergent then reads these S_l.

Self-consistent loop = Λ/ALI iteration of (J ⇄ n), warm-started from the thermal champion.

---

## 3. Computational structure & where tensor cores apply

Cost profile at full-wavelength fine resolution (target NF ~ several×10⁵ over 1000-12000Å):

| Kernel | Math structure | Engine | Status |
|---|---|---|---|
| **Formal solve** `cmf_solve_J` (J_ν transport) | per-ray recurrence `I=I·e^{-Δτ}+S(1-e^{-Δτ})`, ALI scatter iter | **GPU CUDA cores** (per ray×freq parallel). **NOT tensor core** (sequential recurrence). **Dominant new cost.** | currently CPU OpenMP → **port to GPU** |
| **bf rates** `R_bf=K^T·J_ν` | dense (level×freq)·(freq×shell) GEMM | **TF32 tensor core** | ✅ exists `lumina_nlte_gemm.cu`, `lumina_bf_gemm.cu` (`CUBLAS_COMPUTE_32F_FAST_TF32`) — extend to full grid |
| **bb line rates** `J̄_l=∫φJ` | sparse (profile local ±4 v_dop) | per-line gather, **GPU-parallel** | gather (dense GEMM = 99.9% zeros, wasteful). keep sparse |
| **NLTE matrix solve** | dense LU per shell | **FP64** cuSOLVER (cond 1e15-1e18 → FP16/TF32 LU unsafe) | exists, keep FP64 |

**The lever is GPU-porting the formal solve, not tensor cores per se** — the bf GEMM is
already TF32; the transport recurrence (the new bottleneck) needs raw GPU throughput.

**TF32 precision caveat (jmap-informed):** the field spans J/B = 1e-2…1e+2 across far-UV→optical.
TF32 = FP32 range, ~3-digit mantissa; GEMM accumulates in FP32 (OK), but inputs lose the 4th
digit. Mitigate by feeding the bf GEMM **scaled** quantities (e.g. J/B or per-shell-normalised
J) so the mantissa covers the dynamic range; **validate with the existing tie-back test**
(`Int χ_line dν` ratio=1.0000, `lumina_cmfgen.c:1768`).

---

## 4. Design inputs from the jmap (169972)

1. **Far-UV 2000-2500Å is the pump band** and carries J/B up to ~160 at green shells —
   the full grid **must resolve 2000-2500Å at line-Doppler resolution** (this is where
   the fluorescence energy enters). Do not coarsen the far-UV.
2. **Optical transport bug (Phase 0 prerequisite):** the fine `cmf_solve_J` collapses the
   optical field at outer shells (sh13 J/B 1e-4). Must be fixed/validated against the
   binned field (~W·B) before the coupled loop can mean anything — else the optical line
   sources are starved regardless of pumping.
3. **Dynamic range 1e-2…1e+2** drives the TF32 scaling decision above.

---

## 5. Staged implementation plan (each stage has a falsifiable gate)

**Phase 0 — fix optical transport in the fine producer.** Diagnose why `cmf_solve_J` over
a wide window collapses optical J at outer shells (candidate: inner-BC `B(T_inner)` not
leaking through the optical line forest; or `chi_abs≈0` optical + scattering-only with
under-resolved Λ). *Gate:* fine optical J̄/B at sh13 matches the binned ~W·B (0.3-0.6),
not 1e-4. Cheap, read-only A/B (jmap before/after).

**Phase 1 — full-range producer.** Extend the producer grid to 1000-12000Å (or 2000-12000),
all transitions deposited, far-UV at full Doppler resolution. *Gate:* tie-back ratio=1.0000
on all shells; far-UV super-thermal reproduced (J/B>1 at 2000-2500, sh11-15); optical ~W·B
(Phase 0 held). Cost measured here → decides GPU-port urgency.

**Phase 2 — ungate the consumer (the actual fix).** All bb lines (up AND down rates) use
fine J̄_l; optical line source = solved n_u/n_l (not binned scatter). *Gate (the one that
finally passes):* Fe II **5170 S_l/B > 1** in green shells sh11-15 (was 0.46-0.67), traced
to far-UV pump + branching. Plasma gate held (T_e 0.98, n_e dex 0.18). If unstable →
under-relax ω=0.25, staged from thermal champion.

**Phase 3 — self-consistent convergence.** Iterate J⇄n to convergence (ALI). *Gate:*
S_l, T_e, n_e converged (no oscillation/explosion — the loop that blew up before; warm
start + under-relax are the stabilisers).

**Phase 4 — emergent + the multi-band falsifier.** Doppler obs emergent from the converged
fine field. *Gate (user band decomposition):* blue 3000-5000 6.5→~1×, green 5600-7300
0.57→~1×, **moving together** (energy conservation), 7700 dip deepens, NIR holds. Draw vs gold.

**Optimization (interleave or after correctness):** GPU-port the formal solve; extend the
TF32 bf GEMM to full grid with scaled inputs; GPU-parallelize the bb gather. Correctness
(Phases 0-4 on a tractable grid) **before** the speed work.

---

## 6. Code map (reuse vs new)

- **Reuse:** `cmf_solve_J` (formal kernel, validated), `cmfgen_fine_jbar` (producer,
  extend window), consumer mode-2 (`plasma.c:7022`, ungate), `lumina_nlte_gemm.cu` /
  `lumina_bf_gemm.cu` (TF32 bf, extend), NLTE matrix assembly+LU (`plasma.c:6896`, FP64).
- **New:** Phase-0 optical-transport fix; full-range grid management (memory for NF~5×10⁵×NS);
  GPU port of the formal solve recurrence; TF32 scaling layer for the field; convergence
  control (ALI accel + under-relax) for the coupled J⇄n loop.
- **Keep FP64:** NLTE LU (conditioning); the orthodox NLTE closure (LTE_FLOOR/COLL_FIX/etc.)
  stays — it is independent of the field-resolution fix ([[project_basics_decompose_plan]]).

---

## 7. Risks & open questions

- **Conditioning is orthogonal and still present** (O II 250-order Saha-Boltzmann span,
  cond 1e15-1e18). The frequency-resolved field does **not** fix it; it must coexist with
  the orthodox closure. Do not conflate (this was an excuse, called out 2026-06-26).
- **Cost:** full grid × GPU formal solve × coupled iterations. Phase 1 measures it; the
  GPU port is the mitigation. Worst case → adaptive mesh as a *later* speed optimization
  (not a physics approximation), only after full-grid correctness is proven.
- **Convergence stability:** the coupled J⇄n loop exploded in past attempts. Warm-start
  from the thermal champion + under-relaxation ω=0.25 + the b-ceiling safety net.
- **TF32 vs dynamic range:** validate per-stage with tie-back; fall back to FP32 (no tensor
  core) for the bf GEMM if far-UV accuracy degrades.

## Links
[[project_autonomous_stage2_2026-06-25]] (campaign log + all diagnostic verdicts),
[[project_toored_rootcause_ladder]] (freq-resolved field Stage-1, the color half already fixed),
[[project_basics_decompose_plan]] (NLTE conditioning, orthogonal), docs/FLUORESCENCE_DESIGN.md,
docs/DETERMINISTIC_EMERGENT_BUGTABLE.md (Steps 5d/6/6b). Physics agent a07a5d0fcee5a7db1.
