# Fluorescence Design — Paths A & B with a config switch (2026-06-28)

## Why (root cause, ARTIS-verified)

Lumina's emergent spectrum is too-red because **the trusted deterministic emergent uses a
coherent two-level line source** (`lumina_plasma.c:8314-8347` → `S_l=(2hν³/c²)/(g_u n_l/(g_l n_u)−1)`,
applied per-line in the Sobolev formal integral `lumina_cmfgen.c:940,1021-1052`). In that scheme
the **emission frequency of a line ≡ its absorption frequency**, so UV energy absorbed in a UV
line can never re-emerge in an optical line. ARTIS has **no two-level source**: every re-emission
is a macro-atom draw over all down-transitions (`macroatom.cc:196-219`, emit at independently
sampled `ε_trans/H`), plus a **k-packet thermal pool** (ff/fb/collisional re-emission, `kpkt.cc`)
that dumps thermalized UV energy back across the optical. The binned radiation field is a
**secondary** issue (ARTIS ships binned-J and still fluoresces).

Two faithful ways to add fluorescence; we build both and let the user pick.

```
LUMINA_EMERGENT_MODE = thermal | detfluor | macroatom     (env, default thermal = current safe baseline)
   thermal   : current two-level / FINE_SL_CLAMP path (no fluorescence; baseline)
   detfluor  : Path B — deterministic fluorescent populations + un-clamped NLTE S_l
   macroatom : Path A — MC macro-atom with a real k-packet thermal pool (ARTIS-faithful)
```

The two physics paths are **independent** and share only the (now-correct, super-level-fixed)
populations + the deterministic plasma. Neither touches T_e (verified 0.98×CMFGEN, no regression).

---

## Path A — MC macro-atom + real k-packet thermal pool (ARTIS-faithful)

The macro-atom state machine already fluoresces correctly (EWEIGHT + neutral-ground energy
reference, `lumina_plasma.c:1457-1640` + emit-at-different-line `lumina_cuda.cu:2471-2476`).
The **missing heart** is a true k-packet (thermal pool). Currently `LUMINA_KPACKET`
(`lumina_cuda.cu:2287-2304`) only re-seeds the macro-atom entry level from the collisional
distribution — no ff, no fb, no Planck(T_e) sink. ARTIS's `do_kpkt` (`kpkt.cc:392-545`) is the
engine that converts UV→heat→optical.

### A-1. k-packet cooling channels (build to match `kpkt.cc`)
On a k-packet, sample a cooling process from a per-shell CDF built from:
- **free-free**  `C_ff = 1.426e-27·√T_e·Z²·n_ion·n_e` → emit r-packet at `ν = −kT_e/h·ln(ξ)` (ARTIS `kpkt.cc:63-77,440-449`).
- **free-bound** `C_fb = bfcoolingcoeff·n_ion·n_e` → emit at the bf edge / `select_continuum_nu` (ARTIS `kpkt.cc:150-178,466-474`).
- **collisional excitation** `C_exc = n_lower·Υ·ε_trans` → **activate a macro-atom** on the upper level (ARTIS `kpkt.cc:96-100,500-526`). ← thermal-pool→fluorescence channel.
- (optional) **collisional ionization** → MA on upper ion.
In optically-thick cells, thermalize directly to a Planck(T_e)-sampled r-packet (ARTIS `do_kpkt_blackbody`, `kpkt.cc:355-371`).

### A-2. Wire k-packet as a first-class deactivation channel
Add `MA→k-packet` (collisional de-excitation) into the macro-atom branching CDF as a real
type change (TYPE_KPKT), not the current in-cascade pre-roll. Energy stays indivisible.

### A-3. Emergent
`LUMINA_EMERGENT_MODE=macroatom` ⇒ run THEN_MC (existing frozen-plasma pass) with EWEIGHT=1 +
neutral-ground + the new k-packet pool ON; the escaping packets are the spectrum. Use a tractable
`MAX_INT` (~1000) and `N_PKT` (~1e5). **One** validation run vs gold; no per-flag re-runs.

Code loci: new `kpacket_cool.{c}` (or in `lumina_plasma.c`) for the cooling CDF; GPU k-packet
handler in `lumina_cuda.cu` near 2287; reuse `bf` tables for fb.

---

## Path B — Deterministic fluorescent source (keep CMFGEN strengths)

Insight: in NLTE, the line emissivity `η_l = n_u A_ul hν_ul` already carries fluorescence **through
the populations** — a UV-pumped upper level that decays via an optical line emits that optical line.
The deterministic formal integral DOES add every line's `η_l`. So the deterministic path fluoresces
**iff the upper levels are correctly UV-pumped and S_l is not clamped to thermal**. Two requirements,
both now reachable because the super-level population bug is fixed:

### B-1. UV-contrast field driving the populations
The population solve must see the UV line field with contrast, not the grey binned-J. The
line-resolved deterministic estimator already exists: `jbar_line_det` (producer `cmfgen_fine_jbar`
in `lumina_cmfgen.c`; consumer gate `LUMINA_CMF_LINERES_JBAR` / `LINERES_CONSUME`, used at
`lumina_plasma.c:1531-1535` and the population J̄ path). `detfluor` mode turns the consumer ON so
the bb excitation rates that set `n_u` carry UV contrast → optical upper levels get pumped.

### B-2. Un-clamp the line source
`detfluor` mode sets the formal/fine emergent to use the **NLTE two-level S_l from the (correct)
populations** (`opacity->line_source_S`, `lumina_plasma.c:8320`) — NOT `FINE_SL_CLAMP=1.0`
(thermal) and NOT the scattering blend. With pumped populations, `S_l` of optical lines rises
above B(T_e) → the formal integral emits the fluoresced optical flux.

### B-3. Stability guard (why this was clamped before)
NLTE S_l was clamped because cold-shell ill-conditioned populations produced super-thermal S_l
garbage — **that was the super-level bug, now fixed** (J=B ⇒ b_k=1.000 verified). Keep a *physical*
guard only: `S_l` from populations with the existing finite/positivity checks; an optional
`LUMINA_DETFLUOR_SL_CEIL` (default large, e.g. 50×B) as a falsifier-only sanity cap, NOT a thermal
clamp. If correct populations still yield localized garbage, that is a real remaining population
bug to diagnose (not to clamp over).

Code loci: emergent-mode dispatch in `lumina_cuda.cu` (the pure-CMFGEN block ~3540 + the emergent
writers); the consumer gate already exists; the formal source selection in `lumina_cmfgen.c:164,178`.

---

## Config dispatch (single switch + sub-knobs)

```
LUMINA_EMERGENT_MODE   thermal(default) | detfluor | macroatom
# Path A sub-knobs
LUMINA_KPACKET=1                 (auto-on in macroatom mode)
LUMINA_MACROATOM_EWEIGHT=1       (auto-on in macroatom mode)
LUMINA_MACROATOM_NEUTRAL_E=1     (auto-on; the Lucy/ARTIS reference)
LUMINA_KPACKET_FF / _FB / _COLLEXC = 1   (cooling channels; all on by default in macroatom)
LUMINA_MAX_INTERACTIONS=1000  N_PKT=1e5
# Path B sub-knobs
LUMINA_CMF_LINERES_JBAR=1        (auto-on in detfluor mode; UV-contrast pops)
LUMINA_DETFLUOR_SL_CEIL=50       (sanity cap only; NOT a thermal clamp)
```

`thermal` reproduces today's headline exactly (regression-safe). Modes are mutually exclusive;
the dispatch lives in one place so neither driver can desync.

---

## Validation plan (NO per-flag re-runs)

DDC15 0.976d, champion plasma config (`JNU_LSTAR0 LSTAR1 LINE_RE1 ratio1.0 PI1 FZ1`),
SUPER_LEVELS=1 + super-level fix + ARTIS LTE criterion. Run **exactly three** emergents (one each):
| mode | expectation vs gold (peak 6600, NIR 0.49) |
|---|---|
| thermal | baseline too-red (peak ~9200, NIR ~0.69) — control |
| detfluor | optical lines pumped → peak blueward, NIR↓ toward gold; T_e unchanged |
| macroatom | k-packet redistributes UV→optical → peak→6600, NIR→0.49 |

Gate: T_e median/CMFGEN stays 0.95–1.05 (no thermal regression) in all modes. Compare
peak/red(6-8k)/NIR(8-12k)/corr + overlay plot. Whichever wins becomes the recommended mode;
both remain available by config.

## Build/verify discipline
- Implement A and B fully, then run the three-mode comparison ONCE. No toggle-by-toggle full runs.
- Physics first: each k-packet channel checked against `kpkt.cc` before wiring; B's un-clamp
  justified by the verified b_k=1.000 (no clamp needed on correct pops).

---
## Path A implementation spec (ARTIS kpkt.cc formulas, verified 2026-06-28)

k-packet cooling channels (ARTIS `kpkt.cc`), each competes by rate; sample one:
- **FREEFREE** (kpkt.cc:65,441): `C_ff = 1.426e-27·√T_e·Σ_ions(charge²·n_ion)·n_e`; emit r-packet at
  `ν_cmf = −k_B·T_e/h·ln(ξ)`, ξ∈(0,1]. EXIT macro-atom as continuum r-packet.
- **FREEBOUND** (kpkt.cc:457-467): `C_fb = bfcoolingcoeff·n_ion·n_e`; emit at `select_continuum_nu`
  (bf edge dist). Needs the bf tables. [increment 2 — fb after ff.]
- **COLLEXC** (kpkt.cc:483-520): the existing channel — re-excite macro-atom from kp_emiss CDF.
- **COLLION** (kpkt.cc:527): re-activate on upper ion. [optional, last.]
- thick cells: `do_kpkt_blackbody` → Planck(T_e) r-packet.

Implementation order (increment, "계속 업데이트"):
1. **ff** (dominant continuum sink): plasma.c — add `p_kpacket_ff[s] = C_ff/(C_ff + C_collexc)`
   (C_collexc = Σ kp_emiss). cuda.cu — when k-packet forms, roll ξ<p_ff ⇒ out_type=−2 (ff cont.),
   ν_cmf=−kB·Te/h·ln(ξ'), EXIT; else coll-exc (current). Caller (cuda.cu:2471): type==−2 ⇒
   *nu = ν_cmf·inv_doppler, emit continuum r-packet (no line). Needs d_T_e on GPU + out_kpkt_nu param.
2. **fb** via select_continuum_nu using existing bf tables.
3. (optional) coll-ion.
Validation: macroatom-mode DDC15 run once; expect NIR redshift broken (k→continuum sink) → peak→6600.
