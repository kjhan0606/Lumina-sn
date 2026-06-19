#!/usr/bin/env python3
"""J_inc frozen-morph spectra vs gold + thermal baseline.

Overlays DDC15 0.976d CMFGEN gold, the thermal-forest baseline (167177), and the
continuum-incident J_inc frozen-morph arms (eps = 0, 0.03, 0.1). Marks the
Ca II / Si II / S II / Ca II NIR P-Cygni diagnostic bands.

Usage: plot_jinc_spectra.py THERMAL.csv "eps0:DIR0" "eps0.03:DIR1" ...
  each extra arg is "label:run_dir" (uses run_dir/lumina_spectrum_formal.csv).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

trapz = getattr(np, 'trapezoid', getattr(np, 'trapz'))
ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'

gw, gf = np.loadtxt(ROOT + '/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat',
                    unpack=True)


def load(path):
    w, f = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True)
    o = np.argsort(w)
    return w[o], f[o]


def opt_norm(w, f):
    m = (w >= 3500) & (w < 9000)
    return f / (trapz(f[m], w[m]) / (9000.0 - 3500.0))


thermal = sys.argv[1] if len(sys.argv) > 1 else (
    ROOT + '/logs/ddc15_pc_phase3_jnul1_radls1_linere1_ratio1.0_pi1_fz1_167177'
           '/lumina_spectrum_formal.csv')
arms = []
for a in sys.argv[2:]:
    label, d = a.split(':', 1)
    arms.append((label, d.rstrip('/') + '/lumina_spectrum_formal.csv'))

gn = opt_norm(gw, gf)
tw, tf = load(thermal); tn = opt_norm(tw, tf)

FEAT = [('Ca II H&K', 3700, 3950), ('S II', 5250, 5500),
        ('Si II 6355', 6050, 6300), ('Ca II NIR', 8200, 8550)]
COL = ['#D97757', '#4EC9B0', '#FFC107']

fig, ax = plt.subplots(2, 1, figsize=(13, 9))
for axi, (xlo, xhi, ttl) in zip(
        ax, [(3500, 9000, 'Optical (3500-9000A norm) — J_inc P-Cygni test'),
             (2500, 12000, 'UV-optical-NIR')]):
    axi.plot(gw, gn, color='k', lw=2.2, label='CMFGEN gold', zorder=6)
    axi.plot(tw, tn, color='#3898EC', lw=1.1, alpha=0.8,
             label='thermal forest S_l=B(Te) (baseline)')
    for i, (label, path) in enumerate(arms):
        try:
            w, f = load(path); n = opt_norm(w, f)
            axi.plot(w, n, color=COL[i % len(COL)], lw=1.4, alpha=0.9,
                     label=f'J_inc {label}')
        except Exception as e:
            print('skip', label, e)
    for name, lo, hi in FEAT:
        axi.axvspan(lo, hi, color='gray', alpha=0.08)
    axi.set_xlim(xlo, xhi); axi.set_ylim(0, None)
    axi.set_title(ttl); axi.legend(fontsize=8); axi.set_ylabel('normalized flux')
ax[-1].set_xlabel('wavelength (Å)')
plt.tight_layout()
out = ROOT + '/figures/2026-06-19_jinc_pcygni_spectra.png'
plt.savefig(out, dpi=110)
print('figure ->', out)
