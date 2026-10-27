#!/usr/bin/env python3
"""Reconstruct the allowed-component f sum for Si II FL124->FL125."""

from math import exp, log

from scipy.integrate import quad
from scipy.special import eval_genlaguerre, gammaln


HARTREE_CM = 219474.63136320


def radial_wavefunction(n, ell, radius, z_eff):
    if not 0 <= ell < n:
        raise ValueError("ell must satisfy 0 <= ell < n")
    rho = 2.0 * z_eff * radius / n
    log_norm = (
        1.5 * log(2.0 * z_eff / n)
        + 0.5 * (gammaln(n - ell) - log(2.0 * n) - gammaln(n + ell + 1))
    )
    if radius == 0.0:
        return exp(log_norm) if ell == 0 else 0.0
    return exp(log_norm - rho / 2.0 + ell * log(rho)) * eval_genlaguerre(
        n - ell - 1, 2 * ell + 1, rho
    )


def dipole_radial_integral(ni, li, nf, lf, z_eff):
    integrand = lambda r: (
        radial_wavefunction(ni, li, r, z_eff)
        * radial_wavefunction(nf, lf, r, z_eff)
        * r**3
    )
    return quad(integrand, 0.0, float("inf"), epsabs=1e-11, epsrel=2e-12, limit=1000)[0]


def oscillator_strength(ni, li, nf, lf, delta_energy_hartree, z_eff):
    if abs(li - lf) != 1:
        return 0.0
    radial = dipole_radial_integral(ni, li, nf, lf, z_eff)
    angular = max(li, lf) / (2 * li + 1)
    return (2.0 / 3.0) * delta_energy_hartree * angular * radial**2


def main():
    # Values independently read from the documented Si II atomic-data row/model.
    ni, nf = 11, 12
    lower_ells = range(7, 11)  # z superlevel: ell=7..10
    upper_ells = set(range(12))  # w superlevel: ell=0..11
    z_eff = 2.0
    delta_energy_hartree = 579.409773 / HARTREE_CM
    source_f = 2.975

    lower_g = sum(2 * (2 * ell + 1) for ell in lower_ells)
    upper_g = sum(2 * (2 * ell + 1) for ell in upper_ells)
    assert (lower_g, upper_g) == (144, 288)

    # Sanity check: the same angular/radial convention reproduces H 1s->2p.
    h_1s_2p = oscillator_strength(1, 0, 2, 1, 3.0 / 8.0, 1.0)
    assert abs(h_1s_2p - 0.416196717980) < 2e-11

    rows = []
    for li in lower_ells:
        components = []
        for lf in (li - 1, li + 1):
            if lf in upper_ells:
                f_value = oscillator_strength(
                    ni, li, nf, lf, delta_energy_hartree, z_eff
                )
                components.append((lf, f_value))
        rows.append((2 * (2 * li + 1), li, components))

    grouped_f = sum(
        weight * sum(value for _, value in components)
        for weight, _, components in rows
    ) / lower_g
    relative_difference = grouped_f / source_f - 1.0

    print(f"H 1s->2p sanity f={h_1s_2p:.12f}")
    print(f"Si II grouped statistical weights: lower={lower_g}, upper={upper_g}")
    for weight, li, components in rows:
        detail = " ".join(f"l'={lf}:f={value:.12g}" for lf, value in components)
        print(f"l={li} weight={weight} {detail}")
    print(f"allowed-component grouped f={grouped_f:.12f}")
    print(f"raw OSC f={source_f:.6f}")
    print(f"relative difference={100.0 * relative_difference:.8f}%")


if __name__ == "__main__":
    main()
