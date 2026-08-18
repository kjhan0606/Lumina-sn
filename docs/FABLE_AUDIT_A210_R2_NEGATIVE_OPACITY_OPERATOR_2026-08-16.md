# Fable audit: A2-10 R2 negative-opacity operator

- Date: 2026-08-16 KST
- Model requested: `fable`
- Canonical model: `claude-fable-5`
- Mode: read-only architecture audit
- Query: `docs/QUERY_FABLE_A210_R2_NEGATIVE_OPACITY_OPERATOR_AUDIT_2026-08-16.md`
- Verdict: **APPROVE**
- Architecture: **A — CMFGEN-equivalent non-overlap Sobolev line operator**

## Fable verdict

> **VERDICT: APPROVE** — root cause confirmed, architecture **A**.
>
> **Q1 — Root cause: CONFIRMED, with one precision.** The inversion is physical
> (CMFGEN's own finite FeIIIOUT inverts the same 1100/1296 levels at 41/90
> depths) and the failure is the operator, not the material. CMFGEN under
> `ALLOW_OL=F` consumes each line through the escape-probability operator, where
> mild negative tau is well-defined — beta = `(1-exp(-tau))/tau` stays positive
> and finite on `(-0.5, 0)`, a bounded per-line amplification — and never sums
> signed profiles into a shared bin. Lumina deposits signed Gaussians into
> `fs.chi_line` (`lumina_cmfgen.c:5453-5459`) and then demands `chi_tot > 0`
> (`:5548`), a contract the sealed deck's material legitimately violates. The
> precision: the deck's `NEG_OPAC_OPT=SRCE_CHK` block
> (`cmfgen_sub.f:2230-2244`) is the CMF/blanketing branch; the operative branch
> for this Sobolev deck is `CHECK_LINE_OPAC && SOBOLEV`
> (`cmfgen_sub.f:3551-3581`, `tau < -0.5` gives `CHIL := 1.0`,
> `CHIL_MAT := 1/N`) — which `line_net_sobolev_material`
> (`line_net_rate.c:119-125`) already mirrors exactly. So the `tau < -0.5`
> policy is not the gap; the gap is solely the `-0.5 <= tau < 0` range being
> summed instead of consumed per-line. Lumina's `SRCE_CHK` enum name is a
> misnomer (naming only).
>
> **Q2 — Architecture A.** Parity-lane fine grid becomes continuum-only
> (positive by construction; keep the fail-closed positivity check there);
> per-(line, shell) `Jbar/Znet` from a CMFGEN-equivalent Sobolev operator
> implemented from `sobjbar_sim.f` — `EXPONX` with its exact branch structure
> (`abs(x) < 1e-3` series, `x >= 40` asymptote), `BETAC` continuum coupling,
> `THICK/DIF` boundaries. In homology `SIGMA = 0`, so the angle-dependent
> `GAM` reduces exactly to the isotropic `GAMH`. **B rejected**: operator
> selection by data sign is a disguised clamp at operator level, and the same
> line would switch operators across shells. **C rejected** for this lane:
> exceeds the CMFGEN footprint before parity closes, making differences
> unattributable.
>
> **Q3 — Typed policy view PERMITTED; the gate must NOT stop at
> `tau < -0.5`.** The clamp discriminant is "can the exact solution violate the
> guard?" — the exact benchmark solution (CMFGEN's own finite output) applies
> this replacement, so it is part of the benchmark's definition, not a repair.
> Conditions: raw tau/populations immutable and persisted; raw vs effective
> typed separately (both already true); `srce_chk` census published against a
> pre-registered count (4 here) with mismatch failing closed; enum-scoped to
> the parity lane only.
>
> **Q4 — Minimal proof obligations before another A100x2** (all offline):
> (1) host micro-parity fixture of the new operator versus direct
> CMFGEN-formula evaluation, including witness line 2164811 shell 0 and several
> mild-negative cells, values pre-registered; (2) negative control — injected
> defect must make the gate FAIL closed; (3) R1 lane BIT_EXACT versus sealed k24
> after the refactor; (4) pre-registered R2 expectation: census identical
> (4,246,581 raw negatives preserved, `srce_chk=4`), safety counters zero, run
> passes the previous fail point with all `Jbar` finite, generation counters
> unchanged. Then one adjudication run.
>
> **Q5 — Required changes (correctness):** remove line deposition from the
> shared extinction in the parity lane; implement the per-line Sobolev operator
> per Q2; keep raw values immutable with fail-closed census accounting; pass Q4
> before submission; re-verify R1 bit-exactness and the generation barrier.
> **Optional diagnostics:** bin-187 contributor-identity dump (moot after the
> swap), LU_NEG-style negative-line ledger, enum rename to the actual CMFGEN
> branch name.

## Codex disposition

Accepted. The next implementation will change the parity-lane line operator,
not populations or signed tau. No physical-value repair will be introduced.
