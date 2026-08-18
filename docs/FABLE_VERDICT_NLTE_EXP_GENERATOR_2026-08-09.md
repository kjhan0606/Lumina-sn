# Fable 중요 판정 — NLTE `exp(z)` / `exp(tQ)` (2026-08-09)

- model: `claude-fable-5`
- 호출: Claude Code CLI 단발 `--max-turns 1`
- 범위: 저장소 탐색·코드 생성 없이 Si I–II 202×202 rate-generator 구조와
  관측 수치만 전달
- 사유: 사용자가 직접 요청한 `exp(**)` 양수성 대안의 핵심 물리·수치 판정

## 질의 요약

pre-lock `Q`는 열 관례의 연속시간 rate generator
(`Q_ij>=0, i!=j`, 열합 0, `Q n=0`)를 의도한다. 생산 코드는 Si I과 Si II의
총량을 각각 고정하려고 두 SE 행을 두 개의 stage-total 식으로 교체한다. post-lock
해의 음수는 double과 80자리에서 일치하고, SE 행의 signed cancellation ratio는
중앙 `7–8e15`, 최대 `~2e18`이다. 후보인 `n=exp(z)`와 `exp(tQ)`/uniformization이
해결책인지, 그리고 pre-lock 대각 직접 복원이 어떤 판별력을 갖는지 물었다.

## Fable 응답 원문

> **Verdict**
>
> **1) n = exp(z): no — it hides the defect.** The exp-transform enforces positivity by construction, so it is legitimate only when the true solution is positive and negatives are a floating-point artifact. Your 80-digit solves reproduce the same signs and magnitudes, which proves the exact solution of the *locked* system lies outside the positive cone. In that regime a Newton iteration on z either stalls, drives some z_i → −∞, or converges to a stationary point that does not satisfy the equations — silently converting an infeasibility into an unreported residual. This is a modeling/assembly problem; a positivity-forcing parametrization is a mask, and under this project's clamp/floor rule it is disallowed for the same reason.
>
> **2) exp(tQ) guarantees exactly one conservation, not two.** For an exact column-convention generator (off-diagonals ≥ 0, column sums zero), exp(tQ) is column-stochastic: nonnegative entries, and it preserves only the single invariant 1ᵀn (total over all 202 levels). Uniformization P = I + Q/Λ, Λ ≥ max|q_ii|, computes it as a sum of nonnegative terms, so positivity holds even numerically. But stage totals are *not* invariants when Q contains bound-free inter-stage rates — ionization/recombination moves mass between Si I and Si II by design. The stationary partition between stages is an *output* of Q (unique up to overall scale, given irreducibility). Prescribing both stage totals independently adds one constraint beyond what the flow can honor; it is feasible only if the prescribed ratio happens to equal Q's own stationary partition. Your production scheme deletes two SE rows and imposes the two locks — the solution then violates the deleted rows whenever the locks conflict with the generator, and the exact solution generically leaves the positive cone. The 140 kK case is the smoking gun: the negative ground level (−4.6e-22) exceeds the prescribed Si I total (1.16e-23) by ~40×. For a minority stage nine decades below Si II, the locked distribution is dictated entirely by inter-stage coupling residuals — a classic lock/SE incompatibility signature, not roundoff.
>
> **3) Role of exp(tQ)/uniformization: diagnostic, not production.** As production it answers a different question — it relaxes to Q's own stationary state and cannot honor the two locks. As a diagnostic it is excellent: evolve a positive n₀ to steady state (202×202 is cheap), obtain a guaranteed-nonnegative unconstrained SE solution, and compare its Si I/Si II partition against the prescribed totals. A material mismatch is a direct, quantitative measurement of lock/SE incompatibility.
>
> **4) Clean discriminator from the pre-lock experiment.** Precondition first: if raw RHS ≠ 0 or any off-diagonal is negative, stop — that is an assembly error in its own right. Otherwise, with diagonals rebuilt as compensated (Neumaier) negated off-diagonal sums and the same two locks reapplied:
> - **Negatives vanish** (float64 and 80-digit): the cause was diagonal-consistency error — with cancellation ratios of 1e16–1e18, float64 diagonals can carry O(1) relative error on the residual scale. Assembly defect; fix accumulation.
> - **Negatives persist at 80 digits**: the generator is exact by construction, so the infeasibility is structural — the prescribed stage totals are inconsistent with the level SE. The indictment moves upstream to whatever produced the totals (the ionization solver) versus the rates inside Q.
> Complement this with the uniformization steady state of the repaired Q: its stage ratio vs the prescribed ratio quantifies the incompatibility in one number.
>
> **5) Actions.** Do: complete the pre-lock capture; verify generator axioms; run the repaired-diagonal locked solve *and* the unconstrained uniformization steady state; compare partitions. If locks are inconsistent, repair upstream: keep a single normalization lock (total Si, plus charge conservation as appropriate) and let the stage partition emerge from SE, or reconcile the ionization-balance prescription with Q's actual rates. No-go: exp(z) parametrization as a cure; clamping/flooring negatives; Cholesky+jitter (correctly rejected — the operator is genuinely nonsymmetric); adopting exp(tQ) as production while retaining two independent stage locks; any rescaling that makes negatives disappear without identifying which of the two causes above produced them.

## Codex 판정

동의한다. `exp(tQ)`는 폐기할 아이디어가 아니라 **수정과 독립된 진단 잣대**로
채택한다. 다만 생산 해법으로 승격하지 않는다. job `251515`의 repaired pre-lock
`Q`에서 generator 공리를 먼저 확인하고, (1) 두 lock 재적용 해와 (2) 단일 총량만
정규화한 unconstrained 정상상태의 Si I/Si II 비를 함께 산출한다.
