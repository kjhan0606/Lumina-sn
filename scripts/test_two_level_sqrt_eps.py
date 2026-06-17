#!/usr/bin/env python3
"""Textbook two-level-atom √ε validation (machinery check).

Standard semi-infinite isothermal slab, monochromatic two-level atom with
isotropic scattering:  S = (1-eps)*J + eps*B,  J = Lambda[S].
The classic result (Milne; Mihalas "Stellar Atmospheres"; Hubeny & Mihalas
2014) is the THERMALIZATION / √ε law:
    S(tau=0)/B = sqrt(eps)        (surface source function)
    S -> B for tau >> 1/eps       (thermalization depth)
A correctly-coupled scattering+thermal solver MUST reproduce S(0)/B = √ε and
can NEVER produce a super-thermal surface S/B > 1. This is the sharpest
geometry-independent falsifier for the super-thermal S_l defect.

Method: build the Lambda operator by formal short-characteristics solutions
(Gauss-mu quadrature, log tau grid), then solve (I - (1-eps)Lambda) S = eps*B
directly. No iteration error.

Refs: Avrett & Hummer 1965 MNRAS 130,295; Lambert+2015 arXiv:1509.01158.
"""
import numpy as np


def build_lambda(tau):
    """Lambda matrix: J = Lambda @ S, semi-infinite slab, isotropic scattering.
    tau ascending from 0 (surface). Uses Gauss quadrature in mu in (0,1] and
    linear short-characteristics for the formal solution of dI/ds = -(I - S)."""
    n = len(tau)
    # Gauss-Legendre nodes on (0,1] for the angle integral 0.5*int_{-1}^{1} ... dmu
    xg, wg = np.polynomial.legendre.leggauss(8)
    mu = 0.5 * (xg + 1.0)          # (0,1)
    wmu = 0.5 * wg                 # weights, sum = 1 over (0,1)
    Lam = np.zeros((n, n))
    for m, w in zip(mu, wmu):
        dtau = np.diff(tau) / m    # optical path along this ray
        # --- inward ray (mu>0 going to larger tau), I(0)=0 (no incident) ---
        # Build contribution of each S_k to I at each depth via linear SC.
        # We accumulate the local mean-intensity weights directly.
        # Outgoing (toward surface) and incoming rays both included by symmetry
        # using the standard linear-SC psi coefficients.
        e = np.exp(-dtau)
        # linear SC weights (Olson & Kunasz): for interval k between i-1,i
        u = np.where(dtau > 1e-8, 1.0 - e, dtau - 0.5 * dtau**2)
        # w0 (upwind point i-1), w1 (downwind point i)
        with np.errstate(divide='ignore', invalid='ignore'):
            w1 = np.where(dtau > 1e-4, u / dtau - e, dtau * (0.5 - dtau / 3.0))
            w0 = np.where(dtau > 1e-4, 1.0 - u / dtau, dtau * (0.5 - dtau / 6.0))
        # Inward intensity I_in[i] from S, with I_in[0]=0
        Iin = np.zeros((n, n))     # Iin[i,k] = dI at depth i from S_k
        for i in range(1, n):
            Iin[i] = e[i - 1] * Iin[i - 1]
            Iin[i, i - 1] += w0[i - 1]
            Iin[i, i] += w1[i - 1]
        # Outward intensity I_out from deep boundary: I_out(deep)=B(diffusion);
        # for the LAMBDA (scattering) part we take diffusion S at the base,
        # but for a clean semi-infinite test we set I_out at base = S_base and
        # integrate toward the surface. Approximate base as thermalized.
        Iout = np.zeros((n, n))
        Iout[n - 1, n - 1] = 1.0   # base: I_out ~ S_base (Lambda diagonal seed)
        for i in range(n - 2, -1, -1):
            Iout[i] = e[i] * Iout[i + 1]
            Iout[i, i + 1] += w0[i]
            Iout[i, i] += w1[i]
        Lam += w * 0.5 * (Iin + Iout)
    return Lam


def sqrt_eps_test(eps, tau_max=1e8, n=400):
    tau = np.concatenate([[0.0], np.logspace(-4, np.log10(tau_max), n - 1)])
    Lam = build_lambda(tau)
    B = np.ones(len(tau))
    A = np.eye(len(tau)) - (1.0 - eps) * Lam
    S = np.linalg.solve(A, eps * B)
    return tau, S


if __name__ == "__main__":
    print(f"{'eps':>8} {'S(0)/B numeric':>16} {'sqrt(eps)':>12} {'ratio':>8} {'max S/B':>9}")
    for eps in [1e-1, 1e-2, 1e-4, 1e-6]:
        tau, S = sqrt_eps_test(eps)
        s0 = S[0]
        print(f"{eps:>8.0e} {s0:>16.4e} {np.sqrt(eps):>12.4e} "
              f"{s0/np.sqrt(eps):>8.3f} {S.max():>9.4f}")
    print("\nPASS criterion: S(0)/B ~= sqrt(eps) (ratio ~1), and max S/B <= 1 "
          "(NEVER super-thermal). A solver that yields S/B >> 1 in a thick "
          "scattering line has a defect (this is the LUMINA super-thermal S_l "
          "signature, reproduced here as a unit test of the principle).")
