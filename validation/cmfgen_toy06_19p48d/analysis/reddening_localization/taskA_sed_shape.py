#!/usr/bin/env python3
"""Inspect SED shape: nu*u_nu (energy per dex) on a coarse log grid, both codes,
and a smoothed ratio for clean crossing wavelengths. Reads the overlay CSV made
by taskA_band_localization.py (J on the shared Lumina grid)."""
import numpy as np, csv, os
OUT = os.path.dirname(os.path.abspath(__file__))
C_A = 2.99792458e18; C_CM = 2.99792458e10; FOURPI_OVER_C = 4*np.pi/C_CM

d = {}
with open(f"{OUT}/taskA_overlay_spectrum.csv") as f:
    r = csv.reader(f); next(r)
    for row in r:
        s = row[0]; lam = float(row[2]); jl = float(row[3]); jc = float(row[4])
        d.setdefault(s, []).append((lam, jl, jc))

# coarse log-lambda bins for SED display + smoothed ratio
edges = np.logspace(np.log10(100), np.log10(19933), 41)
for s in ['s0', 's2', 's4']:
    a = np.array(d[s]); lam = a[:, 0]; jl = a[:, 1]; jc = a[:, 2]
    nu = C_A / lam
    print(f"\n### {s}: nu*u_nu per log-lambda bin (fraction of total), and smoothed ratio")
    print(f"  {'lam_mid':>8} {'f_cmf':>7} {'f_lum':>7} {'ratio_smooth':>12}")
    # integrate u in each coarse bin
    uc_tot = 0.0; ul_tot = 0.0
    ucb = []; ulb = []; lmid = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (lam >= lo) & (lam < hi)
        if m.sum() < 2:
            ucb.append(0.0); ulb.append(0.0); lmid.append(np.sqrt(lo*hi)); continue
        nb = C_A / lam[m]; o = np.argsort(nb)
        uc = FOURPI_OVER_C * np.trapz(jc[m][o], nb[o])
        ul = FOURPI_OVER_C * np.trapz(jl[m][o], nb[o])
        ucb.append(uc); ulb.append(ul); lmid.append(np.sqrt(lo*hi))
    ucb = np.array(ucb); ulb = np.array(ulb); lmid = np.array(lmid)
    uc_tot = ucb.sum(); ul_tot = ulb.sum()
    ratio = np.where(ucb > 0, ulb / ucb, np.nan)
    for i in range(len(lmid)):
        mark = ''
        if i > 0 and np.isfinite(ratio[i]) and np.isfinite(ratio[i-1]):
            if (ratio[i]-1)*(ratio[i-1]-1) < 0:
                mark = '  <-- ratio=1 crossing'
        print(f"  {lmid[i]:>8.0f} {ucb[i]/uc_tot:>7.3f} {ulb[i]/ul_tot:>7.3f} "
              f"{ratio[i]:>12.3f}{mark}")
    # cumulative-u crossing (blue->red): where does cumulative L/C first exceed... n/a since <1
    cum_c = np.cumsum(ucb); cum_l = np.cumsum(ulb)
    cumr = cum_l / cum_c
    print(f"  cumulative L/C from blue: {cumr[0]:.2f} .. {cumr[-1]:.3f} (bolometric)")
