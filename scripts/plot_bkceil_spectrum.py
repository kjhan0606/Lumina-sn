#!/usr/bin/env python3
"""LUMINA b_k-ceiling sweep formal spectra vs DDC15 0.976d CMFGEN gold."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

trapz = getattr(np, 'trapezoid', getattr(np, 'trapz'))
ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
gw, gf = np.loadtxt(ROOT + '/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat', unpack=True)
runs = [('166871', 'CEIL=0 (no cap, too-blue 4770A)', '#3898EC'),
        ('166872', 'CEIL=10 (b_k fix, 8035A)', '#D97757'),
        ('166873', 'CEIL=100 (b_k fix, 8045A)', '#4EC9B0')]


def opt_norm(w, f):
    m = (w >= 3500) & (w < 9000)
    return f / (trapz(f[m], w[m]) / (9000.0 - 3500.0))  # band-mean -> 1


fig, ax = plt.subplots(2, 1, figsize=(11, 9))
gn = opt_norm(gw, gf)
for axi, (lo, hi, ttl) in zip(ax, [(2500, 10500, 'Optical'), (2500, 25000, 'Full UV-optical-NIR')]):
    axi.plot(gw, gn, color='k', lw=2.0, label='CMFGEN gold (color 7424A)', zorder=5)
    for j, lab, c in runs:
        fw, ff = np.loadtxt(f'{ROOT}/logs/ddc15_pc_phase3_jnul1_radls1_linere1_ratio1.0_pi1_fz1_{j}/lumina_spectrum_formal.csv',
                            delimiter=',', skiprows=1, unpack=True)
        o = np.argsort(fw)
        fw, ff = fw[o], ff[o]
        axi.plot(fw, opt_norm(fw, ff), color=c, lw=1.2, alpha=0.85, label=lab)
    axi.set_xlim(lo, hi)
    axi.set_xlabel('wavelength (A)')
    axi.set_ylabel('normalized F_lambda')
    axi.set_title(f'DDC15 0.976d  {ttl}  (normalized to optical 3500-9000A)')
    axi.legend(fontsize=9)
    axi.grid(alpha=0.25)
    axi.axvline(7424, color='k', ls=':', alpha=0.4)
    axi.set_ylim(0, 3.0)
plt.tight_layout()
out = ROOT + '/figures/2026-06-18_ddc15_bkceil_spectrum_vs_cmfgen.png'
plt.savefig(out, dpi=110)
print('saved', out)
