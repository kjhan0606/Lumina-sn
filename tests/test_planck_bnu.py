#!/usr/bin/env python3
"""Golden physics test: Planck spectral radiance B_ν(T, ν).

Reference: B_ν = (2hν³/c²) / (exp(hν/kT) − 1)  [erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹]
LUMINA implementation: src/lumina_plasma.c:735-740.

Invariants checked here:
  - Stefan–Boltzmann:  ∫ B_ν dν = (σ/π) T⁴
  - Wien displacement: ν_peak ≈ 2.821 k T / h
  - Rayleigh–Jeans:    B_ν → (2ν²/c²) k T  for hν ≪ kT
  - Wien tail:         B_ν → (2hν³/c²) exp(−hν/kT)  for hν ≫ kT
  - Overflow guard:    x>500 returns 0 (no NaN/Inf)
"""
from __future__ import annotations
import math
import sys

# CGS constants (must match src/lumina.h)
H_PLANCK     = 6.62606957e-27
K_BOLTZMANN  = 1.3806488e-16
C_LIGHT      = 2.99792458e10
SIGMA_SB     = 5.670400e-5            # Stefan-Boltzmann


# MUST mirror src/lumina_plasma.c:735-740
def planck_lumina(T: float, nu: float) -> float:
    x = H_PLANCK * nu / (K_BOLTZMANN * T)
    if x > 500.0:
        return 0.0
    return (2.0 * H_PLANCK * nu * nu * nu / (C_LIGHT * C_LIGHT)) / (math.exp(x) - 1.0)


failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "[ok ]" if cond else "[FAIL]"
    line = f"  {tag} {name}" + ((" — " + detail) if detail else "")
    if cond:
        print(line)
    else:
        failures.append(line)
        print(line, file=sys.stderr)


def integrate_bnu(T: float, n_decades: int = 6, points_per_dec: int = 200) -> float:
    """Trapezoid integral in log ν from peak/10⁵ to peak·10⁵ — covers full curve."""
    nu_peak = 2.821 * K_BOLTZMANN * T / H_PLANCK
    log_lo = math.log10(nu_peak) - n_decades
    log_hi = math.log10(nu_peak) + n_decades
    N = (log_hi - log_lo) * points_per_dec
    N = int(N) + 1
    dlog = (log_hi - log_lo) / N
    s = 0.0
    nu_prev = 10.0 ** log_lo
    b_prev = planck_lumina(T, nu_prev)
    for i in range(1, N + 1):
        nu = 10.0 ** (log_lo + i * dlog)
        b = planck_lumina(T, nu)
        s += 0.5 * (b_prev + b) * (nu - nu_prev)
        nu_prev, b_prev = nu, b
    return s


def main() -> int:
    print("=== golden test: planck_bnu (Planck spectral radiance) ===")

    # 1. Stefan-Boltzmann: ∫B_ν dν = (σ/π) T^4
    for T in [5000.0, 10000.0, 50000.0]:
        I = integrate_bnu(T)
        expect = SIGMA_SB * T ** 4 / math.pi
        rel = abs(I - expect) / expect
        check(f"Stefan-Boltzmann ∫B_ν dν at T={T:.0f}K", rel < 1e-3,
              f"got {I:.6e} expect {expect:.6e} rel {rel:.2e}")

    # 2. Wien displacement: B_ν peaks at hν/kT ≈ 2.821 (numerical root of 3(1−e^−x) = x)
    for T in [5000.0, 10000.0]:
        nu_peak_expect = 2.821439 * K_BOLTZMANN * T / H_PLANCK
        # find argmax over log-spaced grid centered on prediction
        best_nu, best_b = 0.0, 0.0
        log_c = math.log10(nu_peak_expect)
        for i in range(-1000, 1001):
            nu = 10.0 ** (log_c + i * 0.001)
            b = planck_lumina(T, nu)
            if b > best_b:
                best_nu, best_b = nu, b
        rel = abs(best_nu - nu_peak_expect) / nu_peak_expect
        check(f"Wien peak ν at T={T:.0f}K", rel < 0.01,
              f"got {best_nu:.6e} expect {nu_peak_expect:.6e} rel {rel:.2e}")

    # 3. Rayleigh-Jeans limit: hν ≪ kT  ⇒  B_ν → 2ν²kT/c²
    for T in [5000.0, 10000.0]:
        nu = 1e-4 * K_BOLTZMANN * T / H_PLANCK   # x = hν/kT = 1e-4
        got = planck_lumina(T, nu)
        rj = 2.0 * nu * nu * K_BOLTZMANN * T / (C_LIGHT * C_LIGHT)
        rel = abs(got - rj) / rj
        check(f"Rayleigh-Jeans at x=1e-4, T={T:.0f}K", rel < 1e-3,
              f"got {got:.4e} RJ {rj:.4e} rel {rel:.2e}")

    # 4. Wien tail: hν ≫ kT  ⇒  B_ν → (2hν³/c²) exp(−hν/kT)
    for T in [5000.0]:
        for x in [10.0, 30.0, 100.0]:
            nu = x * K_BOLTZMANN * T / H_PLANCK
            got = planck_lumina(T, nu)
            wien = 2.0 * H_PLANCK * nu ** 3 / C_LIGHT ** 2 * math.exp(-x)
            rel = abs(got - wien) / wien
            # at x=10 the (e^x − 1) ≈ e^x to 5e-5 accuracy; tighten with x
            tol = 1e-3 if x < 20 else 1e-12
            check(f"Wien tail at x={x}, T={T:.0f}K", rel < tol,
                  f"got {got:.4e} Wien {wien:.4e} rel {rel:.2e}")

    # 5. Overflow guard: x > 500 returns 0 exactly (no exp() overflow)
    T = 5000.0
    nu = 1000.0 * K_BOLTZMANN * T / H_PLANCK
    got = planck_lumina(T, nu)
    check("overflow guard: x>500 returns 0", got == 0.0, f"got {got}")

    # 6. Non-negative and finite over a wide grid
    finite_ok = True
    last_bad = None
    for T in [1000.0, 1e4, 1e5]:
        for log_nu in range(8, 22):           # 1e8 to 1e22 Hz
            nu = 10.0 ** log_nu
            b = planck_lumina(T, nu)
            if not (math.isfinite(b) and b >= 0.0):
                finite_ok = False
                last_bad = (T, nu, b)
    check("B_ν finite and ≥ 0 across [1e8, 1e22] Hz", finite_ok,
          "" if finite_ok else f"violation at T={last_bad[0]} ν={last_bad[1]:.2e} B={last_bad[2]}")

    print()
    if failures:
        print(f"=== {len(failures)} CHECK(S) FAILED ===")
        return 1
    print("=== all checks passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
