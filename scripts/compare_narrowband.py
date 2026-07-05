#!/usr/bin/env python3
"""Narrow-band spectrum comparison vs CMFGEN toy06 19.48d (user rule 2026-07-05:
wide bands are directional only; accurate judgment = narrow bands + feature windows).

Usage: python3 scripts/compare_narrowband.py <spectrum.csv> [label] [--mc]
  spectrum.csv: lumina_spectrum_formal.csv (or lumina_spectrum.csv for THEN_MC=1 runs)
Outputs: per-100A log10(L/C) profile stats, worst windows, feature-window table,
         and a PNG figure figures/narrowband_<label>.png (lambda vs F_lambda, normalized).
"""
import sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BIN_A = 100.0
LAM_LO, LAM_HI = 2500.0, 12000.0
FEATURES = [
    ('Ca II H&K', 3550, 4000),
    ('Si II 4130', 3950, 4150),
    ('Fe/Mg blue', 4200, 4700),
    ('S II W',     5150, 5650),
    ('Si II 5972', 5700, 6000),
    ('Si II 6355', 5900, 6400),
    ('Ca II 8542', 7900, 8700),
]

def load_lumina(p):
    d = np.genfromtxt(p, delimiter=',', names=True)
    wl = d[d.dtype.names[0]]; f = d[d.dtype.names[1]]
    m = (wl > 0) & np.isfinite(f) & (f >= 0)
    return wl[m], f[m]

def load_cmfgen(epoch_col=26):
    dat = []
    for L in open('data/standart_data1/toy06/spectra_toy06_cmfgen.txt'):
        if L.startswith('#') or not L.strip():
            continue
        p = [float(x) for x in L.split()]
        dat.append((p[0], p[1 + epoch_col]))
    d = np.array(dat)
    return d[:, 0], d[:, 1]

def rebin(wl, f, edges):
    out = np.zeros(len(edges) - 1)
    for i in range(len(out)):
        m = (wl >= edges[i]) & (wl < edges[i + 1])
        out[i] = np.trapezoid(f[m], wl[m]) if m.sum() > 1 else 0.0
    return out

def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else 'lumina'
    wl_l, f_l = load_lumina(path)
    wl_c, f_c = load_cmfgen()
    edges = np.arange(LAM_LO, LAM_HI + BIN_A, BIN_A)
    mid = 0.5 * (edges[:-1] + edges[1:])
    L = rebin(wl_l, f_l, edges)
    C = rebin(wl_c, f_c, edges)
    # normalize each to unit integral over the common range (shape comparison)
    L /= L.sum() if L.sum() > 0 else 1.0
    C /= C.sum() if C.sum() > 0 else 1.0
    ok = (L > 0) & (C > 0)
    logr = np.full(len(mid), np.nan)
    logr[ok] = np.log10(L[ok] / C[ok])
    med = np.nanmedian(np.abs(logr))
    rms = np.sqrt(np.nanmean(logr[ok] ** 2))
    corr = np.corrcoef(L[ok], C[ok])[0, 1]
    print(f"[narrow {BIN_A:.0f}A] {label}: bins={ok.sum()}  median|log r|={med:.3f} dex  "
          f"RMS={rms:.3f} dex  narrow-corr={corr:.3f}")
    worst = np.argsort(-np.abs(np.nan_to_num(logr)))[:6]
    print("  worst windows: " + ' '.join(
        f"{mid[i]:.0f}A({logr[i]:+.2f})" for i in sorted(worst)))
    print("  feature windows (log10 L/C of window-integrated flux):")
    mlr = (wl_l >= LAM_LO) & (wl_l < LAM_HI)
    mcr = (wl_c >= LAM_LO) & (wl_c < LAM_HI)
    Lnorm = max(np.trapezoid(f_l[mlr], wl_l[mlr]), 1e-300)
    Cnorm = max(np.trapezoid(f_c[mcr], wl_c[mcr]), 1e-300)
    for nm, lo, hi in FEATURES:
        ml = (wl_l >= lo) & (wl_l < hi); mc = (wl_c >= lo) & (wl_c < hi)
        Lw = np.trapezoid(f_l[ml], wl_l[ml]) / Lnorm
        Cw = np.trapezoid(f_c[mc], wl_c[mc]) / Cnorm
        r = np.log10(Lw / Cw) if Lw > 0 and Cw > 0 else float('nan')
        print(f"    {nm:12s} [{lo}-{hi}]: {r:+.3f} dex")
    # figure
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1]})
    a1.plot(mid, C, 'k-', lw=1.5, label='CMFGEN')
    a1.plot(mid, L, 'r-', lw=1.2, label=label)
    a1.set_ylabel('normalized F (per 100A bin)'); a1.legend()
    a2.axhline(0, color='k', lw=0.5)
    a2.plot(mid, logr, 'b.-', ms=3)
    a2.set_ylabel('log10(L/C)'); a2.set_xlabel('wavelength [A]')
    a2.set_ylim(-1.5, 1.5)
    fig.tight_layout()
    out = f"figures/narrowband_{label}.png"
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")

if __name__ == '__main__':
    main()
