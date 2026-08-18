# BAKEFIX — repairing the two σ-bake bugs the certification named

**2026-07-29.** Offline python only. `src/` untouched, no GPU, no LUMINA run.
Target: `scripts/expand_atomic_data_cmfgen.py`. Judge: `certify_rate_machine.py`,
**unmodified** — the yardstick was not touched, only re-run against a new `.bin`.

Inputs and outputs:

| | path |
|---|---|
| before (shipped) | `data/atomic/cmfgen_sigma_bf_superlev_ionfix_ddc15strat_sivcaiv.bin` |
| after (this fix) | `data/atomic/cmfgen_sigma_bf_sivcaiv_bakefix.bin` |
| after, ref dir | `data/tardis_reference_cmfgen_sivcaiv_bakefix/` (new; nothing overwritten) |
| gate output, after | `bakefix/{rates_certification.csv,…,run_log.txt}` |
| gate output, rejected variant | `bakefix_drop_hydrogenic_REJECTED/…` (see §4 — FAILS, kept as evidence) |

Rebake command (same env as the shipped bake — `CMFGEN_SUPER_LEVELS=1` is
required, the shipped reference is a super-level build):

```
CMFGEN_SUPER_LEVELS=1 CMFGEN_OUT_SUFFIX=_sivcaiv_bakefix \
    python3 scripts/expand_atomic_data_cmfgen.py
```

Provenance, verified not assumed: re-running `bake_sigma_bf_grid` from the
committed script into a scratch path reproduces the certified binary bit for
bit — `sha256 = 8ed3f16907de321374837b18033ec5e3e60496c2bf3f341272a27b6e15227e4e`.
Coverage `has_cmfgen=1` is **26 087 / 26 592 = 98.1 %**, identical to the shipped
bake, and per-ion identical where the certification can see it (Co III 1000/1000,
Fe III 1500/1500, S II 315/322, S III 256/256).

---

## 1. Verdict — pre-registered gate

Registered before the rebake: **Fe III s6 `D/A` 1.020 → ≤1.005**, **S II
`Bpt/A|g` 1.047 → ≤1.01**, **no other ion worse by more than ±0.5 %**, **level
count and line count unchanged**.

`D/A` = shipped-binary Γ over CMFGEN's-own-σ Γ, i.e. the machine-only statistic
of VERDICT §1 (it divides out the shell-dependent −0.4…−12 % that the control
carries for all four ions).

### D/A — before → after, all nine gated shells

| ion | s0 | s1 | s2 | s3 | s4 | s5 | **s6** | s7 | s8 |
|---|---|---|---|---|---|---|---|---|---|
| Co III before | 1.150 | 1.156 | 1.161 | 1.173 | 1.181 | 1.186 | **1.184** | 1.189 | 1.155 |
| Co III **after** | 1.151 | 1.157 | 1.163 | 1.175 | 1.183 | 1.187 | **1.186** | 1.191 | 1.158 |
| Fe III before | 1.018 | 1.021 | 1.023 | 1.023 | 1.022 | 1.022 | **1.022** | 1.022 | 1.032 |
| Fe III **after** | 1.000 | 1.000 | 1.001 | 1.001 | 1.001 | 1.001 | **1.001** | 1.001 | 1.000 |
| S II before | 1.012 | 1.018 | 1.025 | 1.034 | 1.041 | 1.047 | **1.046** | 1.041 | 1.071 |
| S II **after** | 0.998 | 0.998 | 0.997 | 0.998 | 0.999 | 0.999 | **1.000** | 1.000 | 1.007 |
| S III before | 1.001 | 1.001 | 1.001 | 1.001 | 1.000 | 1.000 | **0.999** | 0.999 | 0.998 |
| S III **after** | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 0.999 | **0.998** | 0.999 | 1.000 |

| pre-registered item | target | measured | |
|---|---|---|---|
| Fe III s6 `D/A` | ≤ 1.005 | **1.001** | PASS |
| S II s6 (worst shell s5) | ≤ 1.01 | **1.000** (s5 0.999, worst 1.007 @ s8) | PASS |
| Co III, any shell | within ±0.5 % | **+0.07…+0.29 %** | PASS |
| S III, any shell | within ±0.5 % | **−0.13…+0.17 %** | PASS |
| levels / lines unchanged | exact | **26 592 / 2 584 132**, `levels.csv` and `macro_atom_references.csv` byte-identical to the shipped reference | PASS |

Gate itself (`D/PRRR ∈ [0.5,2.0]` everywhere and `[0.8,1.25]` at s6–s8): **all
four ions PASS**. `D/PRRR` at s6 moves Fe III 0.994 → 0.973, S II 1.004 → 0.960,
S III 0.969 → 0.968, Co III 1.152 → 1.154 — i.e. every ion now lands essentially
**on top of the control row `A/PRRR`** (0.972 / 0.960 / 0.970 / 0.973). The
residual that remains is the control's own, not the machine's.

**Control that the change is confined to the σ table:** the layers that never
read the `.bin` — `G_prrr`, `G_A`, `G_Agrid`, `G_Bpt`, `G_Bavg`, `G_C` — are
**bit-identical** between the two runs (max |rel diff| = 0.000e+00 across all
4 ions × 14 depths). Only column `D` moved.

---

## 2. Fix 1 — CMFGEN fit type 7 evaluated exactly

`sub_phot_gen.f:505-512` (read from `/gpfs/kjhan/cmfgen_src/cur_cmf/newsubs/`):

```fortran
RU=(EDGE+PD(ID)%CROSS_A(LMIN+3))/FREQ_VEC(I)
IF(RU .LE. 1.0_LDP)THEN
  PHOT(I)=PHOT(I) + CONV_FAC*CROSS_A(LMIN)*( CROSS_A(LMIN+1) +
1         (1-CROSS_A(LMIN+1))*RU )*( RU**CROSS_A(LMIN+2) )
END IF
```

The edge sits at `ν_th + A3`, and σ is **identically zero below it**. The baker
opened a Kramers edge at `ν_th` instead. Now implemented verbatim.

Measured on Fe III, from the certification's own per-fit-type decomposition
(share of Γ, normalised to Γ_Bpt):

| Fe III type 7 (399 levels) | exact | before | **after** |
|---|---|---|---|
| s0 | 0.00000 | 0.01733 | **0.00000** |
| s6 | 0.00000 | 0.02017 | **0.00000** |

That is the whole of the old `C/Bpt = 1.017…1.029` for Fe III. Scope of type 7
across the tree: **332 entries, in Fe I (159), Fe III (156), Ti IV (17)**; all
carry exactly 4 parameters and all have `A3 > 0` (min 0.0599, median 8.28, max
12.28, in units of 1e15 Hz), so none of them was a no-op.

## 3. Fix 2 — bin-**averaged** σ instead of a bin-centre point sample

σ is now `(1/Δν_b) ∫_{bin b} σ(ν) dν`, which is the quantity the 1000-bin
weight `Γ = Σ_b σ_b · J̄_b · 4π/(hν_c,b) · Δν_b` actually wants. Quadrature
nodes = bin edges ∪ the level's own structure nodes ∪ 5 log subdivisions per
bin, trapezoid, accumulated into the owning bin — the same construction the
certification uses for its `Bav` layer. The threshold bin is now partially
filled from `ν_th` rather than being all-or-nothing about the bin centre.

Effect is largest exactly where VERDICT §2.2 predicted — the ion whose OP data
are **not** pre-smoothed:

| ion | phot data | table Δlnν vs bin | Γ_D after/before, s0 → s6 → s8 |
|---|---|---|---|
| Fe III | `FE/III/19apr23` (smoothed 3000 km/s) | 1.05× | 0.982 → 0.979 → 0.969 |
| S III | `SUL/III/…` (smoothed 3000 km/s) | 1.04× | 0.999 → 0.999 → 1.002 |
| **S II** | `SUL/II/19apr23` (**unsmoothed**) | **2.63×** | **0.986 → 0.957 → 0.940** |
| Co III | `COB/III/19apr23` (fits, smooth) | — | 1.001 → 1.002 → 1.003 |

Independent offline check on the S II tabulated levels alone, Γ-weight-summed:
point-sample / bin-average = **0.946**, i.e. the old point sampling was biasing
Γ(S II) high by ~5.7 %, sign and size as VERDICT §2.2 said.

### Known-answer test (run before any rebake)

New baker vs the certification's **independent** exact evaluator
(`Sigma` + `bin_integral_exact`, written from `sub_phot_gen.f` without reference
to the baker), per level, over Fe III / S II / S III / Co III / Fe I / Ti IV:

| fit type | levels compared | max relative deviation |
|---|---|---|
| 1 (Seaton) | 658 | **0.000e+00** |
| 20/21/22 (tabulated) | 632 | **0.000e+00** (Ti IV: see below) |
| 7 (mod. Seaton) | 332 | one bin per level only |

The type-7 and the three Ti IV tabulated entries deviate in **exactly one bin
each — the bin containing the discontinuous edge** — and the deviation is in the
reference's favour to fix, not mine: `bin_integral_exact` trapezoids *across* the
step (spurious half-triangle), while the baker starts the integral exactly at the
step. Γ-weight-relative size ≤ 1.2e-3, located at λ = 188–449 Å. Reported here
rather than silently absorbed.

---

## 4. The `params[0]` misread is real, is **1200× larger than reported**, and is NOT fixed here

VERDICT §3.1 scoped this as "In S II 9 levels take that path … negligible here,
arbitrary in general". Measured across the tree the baker actually bakes:

| fit type | entries in tree | baked levels |
|---|---|---|
| 2 (hydrogenic n,l) | 3771 | 5322 |
| 3 (hydrogenic, gaunt) | 385 | 272 |
| 8 (hydrogenic, shifted edge) | 4593 | 5254 |
| 9 (Verner) | 12 | 28 |
| | | **10 876 of 26 592 levels (41 %)** |

concentrated in precisely the campaign's ions: **Co III 3344, Co II 2700,
Fe II 1973, Ni I 655, Sc II 486, Si I 345, S II 278**.

**The obvious repair — stop fabricating, leave them to the C loader's Kramers
fallback — was built, baked and certified, and it FAILS the gate:**

| ion | s0 | s3 | s6 | s8 | verdict |
|---|---|---|---|---|---|
| Co III, types 2/3/8/9 dropped | 0.116 | 0.113 | **0.111** | 0.099 | **FAIL** |

Cause, from the certification's own diagnostic: only **311/1000** Co III model
levels keep σ, and **88.5 % of the Co III population at s6 then sits on a σ==0
row** (was 0.000). The C fallback is a single *per-ion* σ₀
(`lumina_plasma.c:6117-6122`; `get_bf_sigma0(27,2)` = 2.0 Mb), so it cannot stand
in for 1441 individually-thresholded hydrogenic levels. Evidence kept in
`bakefix_drop_hydrogenic_REJECTED/`.

Neither stand-in is defensible, and **which one is less wrong depends on the
ion**. Threshold-σ audit against CMFGEN's *own* type-8 value, evaluated from
`HYD_L_DATA` per `sub_phot_gen.f:308-362` (ratio to CMFGEN truth; 1.00 = right):

| ion | type-8 levels | legacy `params[0]` | C Kramers fallback |
|---|---|---|---|
| Co III | 1441 | **0.85** | 0.34 |
| Co II | 555 | 0.41 | **1.03** |
| Fe II | 522 | 0.41 | **0.53** |
| S III | 125 | **0.75** | 0.31 |
| Sc I | 1400 | 0.09 | **0.12** |
| Si I | 172 | 0.09 | **0.12** |
| Al I | 144 | 0.08 | **0.09** |
| Ni II | 21 | **0.84** | 1.52 |

So the shipped bake **keeps the legacy stand-in, now labelled** (the baker prints
a `[bakefix] WARNING` with the count and the worst ions on every run), and the
change delivered here is single-variable: type 7 + bin average.

**The real fix is exact evaluation of types 2/3/8, and it is now within reach.**
`HYD_L_DATA` and `GBF_N_DATA` were located at
`/gpfs/kjhan/cmfgen_runs/toy06_19.48d/` and the units were verified against an
analytic known answer: `10^BF_L_CROSS(n=1,l=0,u=1) × 1e-10 = 6.3034e-18 cm²` vs
the analytic H ground-state 6.30e-18 cm². Remaining unknowns for a full
implementation: `NEF` (type 2), `ALPHA_BF` + `GBF_N_DATA` (type 3), and CMFGEN's
`ZION` convention. **Deliberately not attempted here**: the certification harness
has no exact evaluator for these types either (VERDICT caveat 2), so it would
ship untestable — the same objection that makes the current stand-in unacceptable.

---

## 5. Not changed, still open

1. **VERDICT §3.2 — tabulated `left=0` / `right=const`.** CMFGEN returns
   `CROSS_A[1]` below the first table node and extrapolates `CROSS_A[N]·(u_N/u)³`
   above the last; the baker returns 0 and a constant. Left alone so this bake is
   a single-variable change. Measured cost, unchanged: −0.24 % (S II s0),
   −0.11 % (S II s6). Its constant-tail limb is essentially unreachable (tables
   run to u ≈ 27–32, i.e. ~15 Å, outside the 100 Å grid edge).
2. **`C2` provenance is now expected to be nonzero.** The certification's `C2`
   layer is a *verbatim copy of the old baker*, so
   `Σ|D−C2|/Σ|D|` no longer reads ~1e-14; it now reads Fe III 1.47e-1,
   S II 5.04e-1, S III 6.8e-3, Co III 3.9e-3. **This is the size of the fix in
   σ-magnitude, not an error** — Γ moved by ≤5 % (§1) because the σ that changed
   most sits where J is small (Fe III type-7 edges at 150–250 Å) or averages out
   (S II resonances). If `C2` is ever wanted as a provenance check again,
   `bake_lumina` in `certify_rate_machine.py` has to be re-synced — that is a
   change to the yardstick and was not made here.
3. **Co III `D/A` ≈ 1.19 is untouched and is not a bug.** It is atomic-data
   vintage (LUMINA bakes `COB/III/19apr23`, toy06 ran `18oct00`), per VERDICT
   §3.3. The fix moved it by +0.2 %, as expected for smooth single-parameter
   Seaton fits.
4. **The s8 column remains the least trustworthy**, for the reason VERDICT §3.4
   gives (`*PRRR`/`POP*` written from different states over 34 depths). Unchanged
   by this work.
5. **Scope.** One epoch, one model, four certified ions. The 10 876 stand-in
   levels of §4 are certified at *zero* ions — Fe III/S III carry none, S II
   carries 9 (share ≤ 1.5e-4 of Γ), and Co III's are all above the toy06 model's
   1000-level ceiling.
6. **Type 1 entries carrying only 1 parameter (198 in the tree) are still not
   baked**, by both the old and the new code, and the certification's evaluator
   also declines them. Unexamined.
