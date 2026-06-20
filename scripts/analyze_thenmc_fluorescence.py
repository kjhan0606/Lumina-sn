#!/usr/bin/env python3
"""THEN_MC macro-atom fluorescence assessment vs DDC15 gold.

Given a run dir, compares the MC macro-atom emergent spectrum (lumina_spectrum.csv,
overwritten by the THEN_MC pass) to gold: energy fractions UV/optical/NIR, the
4475A (Fe II/III fluorescence) + 6590A peaks, flux-weighted centroid, peak, and
shape-correlation. Also greps the stdout for the line-interaction verdict.

Usage: analyze_thenmc_fluorescence.py <run_dir> [<run_dir2> ...]
"""
import sys, os, subprocess
import numpy as np
trapz = getattr(np, 'trapezoid', getattr(np, 'trapz'))
R = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
gw, gf = np.loadtxt(R + '/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat', unpack=True)


def load(p):
    w, f = np.loadtxt(p, delimiter=',', skiprows=1, unpack=True)
    o = np.argsort(w); return w[o], f[o]


def bands(w, f):
    tot = trapz(f[(w >= 2500) & (w < 15000)], w[(w >= 2500) & (w < 15000)])
    out = {}
    for lab, lo, hi in [('UV2500-3500', 2500, 3500), ('opt5000-7000', 5000, 7000),
                        ('NIR9000-15000', 9000, 15000)]:
        m = (w >= lo) & (w < hi); out[lab] = trapz(f[m], w[m]) / tot if m.sum() > 1 else 0
    return out


def stats(w, f):
    m = (w >= 3500) & (w < 9000); W, F = w[m], f[m]
    cen = trapz(W * F, W) / trapz(F, W)
    Fs = np.convolve(F, np.ones(11) / 11, 'same'); pk = W[np.argmax(Fs)]
    g = np.arange(3500, 9000, 5.); fi = np.interp(g, W, F); fi /= fi.mean()
    gi = np.interp(g, gw, gf); gi /= gi.mean()
    corr = np.corrcoef(gi, fi)[0, 1]
    # 4475 fluorescence window vs gold
    def win(ww, ff, lo, hi):
        return np.mean((ff / (trapz(ff[(ww>=3500)&(ww<9000)], ww[(ww>=3500)&(ww<9000)])/5500.))[(ww>=lo)&(ww<hi)])
    return cen, pk, corr, win(W, F, 4200, 4700), win(W, F, 6300, 6900)


# gold reference
gc, gp, _, g4475, g6590 = stats(gw, gf)
print(f"{'GOLD':<22} centroid={gc:.0f} peak={gp:.0f} | 4475win={g4475:.2f} 6590win={g6590:.2f} | "
      f"opt={bands(gw,gf)['opt5000-7000']*100:.0f}% NIR={bands(gw,gf)['NIR9000-15000']*100:.0f}%")
print('-' * 100)
for d in sys.argv[1:]:
    sp = os.path.join(d, 'lumina_spectrum.csv')
    if not os.path.exists(sp):
        print(f"{os.path.basename(d):<22} (no MC spectrum)"); continue
    w, f = load(sp); b = bands(w, f); c, p, corr, w4, w6 = stats(w, f)
    # interaction verdict from stdout
    log = os.path.join(d, 'stdout.log'); inter = '?'
    if os.path.exists(log):
        try:
            t = open(log).read()
            inter = 'NONZERO' if ('no macro-atom interactions' not in t.lower()) else 'ZERO'
        except Exception: pass
    print(f"{os.path.basename(d)[-14:]:<22} centroid={c:.0f} peak={p:.0f} corr={corr:.2f} | "
          f"4475={w4:.2f}(g{g4475:.2f}) 6590={w6:.2f}(g{g6590:.2f}) | "
          f"opt={b['opt5000-7000']*100:.0f}% NIR={b['NIR9000-15000']*100:.0f}% | MAinter={inter}")
