#!/usr/bin/env python3
"""Golden physics test: Sobolev escape probability β(τ).

Reference: canonical β(τ) = (1 − exp(−τ)) / τ, with β(0)=1.
LUMINA implementation: src/lumina_plasma.c:729-733 (3-branch split).

If this test fails after a C-side change, either:
  (a) the C change is intentional → update the Python replica below to match,
      AND verify the invariants still hold;
  (b) the C change is a regression → revert it.

Exit 0 = pass, 1 = fail. Run from anywhere.
"""
from __future__ import annotations
import math
import sys


# MUST mirror src/lumina_plasma.c:729-733
def beta_lumina(tau: float) -> float:
    if tau < 1e-6:
        return 1.0 - 0.5 * tau              # Taylor
    if tau > 500.0:
        return 1.0 / tau                    # asymptotic
    return (1.0 - math.exp(-tau)) / tau     # exact


def beta_canonical(tau: float) -> float:
    if tau == 0.0:
        return 1.0
    if tau > 700.0:                         # exp(-700) underflows; use asymptotic
        return 1.0 / tau
    return (1.0 - math.exp(-tau)) / tau


failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [ok ] {name}" + ((" — " + detail) if detail else ""))
    else:
        msg = f"[FAIL] {name}" + ((" — " + detail) if detail else "")
        failures.append(msg)
        print(f"  {msg}", file=sys.stderr)


def main() -> int:
    print("=== golden test: beta_sobolev (Sobolev escape probability) ===")

    # 1. Limits
    check("β(0+) = 1", abs(beta_lumina(1e-30) - 1.0) < 1e-10)
    check("β(very large) → 0", beta_lumina(1e30) < 1e-20)

    # 2. Known values vs canonical (anywhere both are well-defined)
    for tau in [0.0, 1e-9, 1e-3, 0.1, 1.0, 5.0, 10.0, 100.0]:
        got = beta_lumina(tau)
        ref = beta_canonical(tau)
        rel = abs(got - ref) / max(abs(ref), 1e-30)
        check(f"β(τ={tau:>8g}) matches canonical", rel < 1e-5,
              f"got {got:.8e} ref {ref:.8e} rel_err {rel:.2e}")

    # 3. Continuity at branch boundaries
    for tau in [1e-6 * 0.999, 1e-6 * 1.001, 500.0 * 0.9999, 500.0 * 1.0001]:
        got = beta_lumina(tau)
        ref = beta_canonical(tau)
        rel = abs(got - ref) / max(abs(ref), 1e-30)
        check(f"continuity at τ={tau:.6e}", rel < 1e-5,
              f"got {got:.8e} ref {ref:.8e} rel {rel:.2e}")

    # 4. Monotone non-increasing across full sweep (strict equality allowed at
    # float64 precision floor τ < ~1e-15 where 1−τ/2 underflows to 1.0)
    prev = 2.0
    monotone_ok = True
    last_bad = None
    for log_tau in range(-30, 31):
        tau = 10.0 ** log_tau
        cur = beta_lumina(tau)
        if cur > prev:
            monotone_ok = False
            last_bad = (tau, cur, prev)
        prev = cur
    check("β monotone non-increasing across τ ∈ [1e-30, 1e30]", monotone_ok,
          "" if monotone_ok else f"violation at τ={last_bad[0]:.3e}, β={last_bad[1]:.3e} > prev {last_bad[2]:.3e}")

    # 5. β ∈ (0, 1] for all τ ≥ 0
    in_range = True
    last_oor = None
    for log_tau in range(-30, 31):
        tau = 10.0 ** log_tau
        b = beta_lumina(tau)
        if not (0.0 < b <= 1.0):
            in_range = False
            last_oor = (tau, b)
    check("β ∈ (0, 1] for τ ≥ 0", in_range,
          "" if in_range else f"out-of-range at τ={last_oor[0]:.3e} β={last_oor[1]}")

    # 6. No NaN/Inf at extremes
    for tau in [0.0, 1e-300, 1e300, float("inf")]:
        try:
            b = beta_lumina(tau)
            finite = math.isfinite(b) and not math.isnan(b)
            check(f"finite β(τ={tau})", finite, f"got {b}")
        except Exception as e:
            check(f"finite β(τ={tau})", False, f"exception {type(e).__name__}: {e}")

    # 7. Asymptotic-branch accuracy: β ≈ 1/τ for τ > 500
    for tau in [500.001, 1000.0, 1e6, 1e12]:
        got = beta_lumina(tau)
        expect = 1.0 / tau
        rel = abs(got - expect) / expect
        check(f"asymptotic 1/τ at τ={tau:.0e}", rel < 1e-10,
              f"got {got:.6e} expect {expect:.6e}")

    # 8. Taylor-branch accuracy: β ≈ 1 − τ/2 for τ < 1e-6
    for tau in [1e-30, 1e-20, 1e-10, 1e-7]:
        got = beta_lumina(tau)
        expect = 1.0 - 0.5 * tau
        rel = abs(got - expect) / expect
        check(f"Taylor 1−τ/2 at τ={tau:.0e}", rel < 1e-12,
              f"got {got:.16e}")

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        return 1
    print("=== all checks passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
