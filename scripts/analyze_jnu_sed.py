#!/usr/bin/env python3
"""Diagnose whether a hot fitted T_rad is a real blue field or a log-fit artifact.

For chosen shells: compare the actual binned J_nu SED to the fitted dilute Planck
W*B_nu(T_rad), and quantify where the field's energy actually sits (peak nu, mean
nu) vs the fitted color temperature. Also report energy fraction above the C I
bound-free edge (11.26 eV, nu=2.72e15 Hz), which is what drives carbon photoion.
"""
import sys
import numpy as np
import pandas as pd

H = 6.62607015e-27   # erg s
K = 1.380649e-16     # erg/K
C = 2.99792458e10    # cm/s
CI_EDGE_NU = 11.26 * 1.602176634e-12 / H  # C I ionization edge in Hz

def bnu(T, nu):
    x = H * nu / (K * T)
    x = np.clip(x, 1e-300, 700)
    return 2 * H * nu**3 / C**2 / np.expm1(x)

path = sys.argv[1] if len(sys.argv) > 1 else "lumina_jnu_sed.csv"
shells = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [13, 20, 24, 25]
df = pd.read_csv(path)

print(f"C I edge nu = {CI_EDGE_NU:.3e} Hz  ({1e8*C/CI_EDGE_NU:.0f} A)")
print(f"{'sh':>3}{'T_fit':>8}{'W_fit':>10}{'nu_peakJ':>11}{'Tpeak~':>8}"
      f"{'nu_mean':>11}{'Tmean~':>8}{'f_E>CIedge':>11}{'Jblue/Jfit_blue':>16}")
for sh in shells:
    d = df[df.shell == sh].sort_values("bin")
    if d.empty:
        continue
    T = d.T_rad.iloc[0]; W = d.W.iloc[0]
    nu = (0.5 * (d.nu_lo + d.nu_hi)).values
    dnu = (d.nu_hi - d.nu_lo).values
    J = d.J_nu.values
    E = J * dnu                       # energy per bin
    Etot = E.sum()
    if Etot <= 0:
        continue
    nu_peak = nu[np.argmax(J)]        # SED peak (where J_nu max)
    nu_mean = (nu * E).sum() / Etot   # energy-weighted mean freq
    # Wien peak T ~ nu_peak/2.82*h/k ;  mean-freq T ~ nu_mean*h/(3.83 k)
    Tpeak = H * nu_peak / (2.821 * K)
    Tmean = H * nu_mean / (3.832 * K)
    blue = nu > CI_EDGE_NU
    fE_blue = E[blue].sum() / Etot
    # how much the FIT overpredicts/underpredicts blue energy
    model = W * bnu(T, nu)
    Jfit_blue = (model[blue] * dnu[blue]).sum()
    Jblue = E[blue].sum()
    ratio = Jblue / Jfit_blue if Jfit_blue > 0 else np.inf
    print(f"{sh:3d}{T:8.0f}{W:10.2e}{nu_peak:11.3e}{Tpeak:8.0f}"
          f"{nu_mean:11.3e}{Tmean:8.0f}{fE_blue:11.4f}{ratio:16.3e}")
