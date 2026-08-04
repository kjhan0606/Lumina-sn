#!/usr/bin/env python3
"""Final faithful-fix spectrum (COLL_FIX+ION_LOCK+FALLBACK_TE+LTE_NCRIT=1e8, job
167177) vs DDC15 0.976d CMFGEN gold. Formal (observer-frame, P-Cygni) + comoving."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

trapz = getattr(np, 'trapezoid', getattr(np, 'trapz'))
ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
RUN = ROOT + '/logs/ddc15_pc_phase3_jnul1_radls1_linere1_ratio1.0_pi1_fz1_167177'

gw, gf = np.loadtxt(ROOT + '/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat',
                    unpack=True)


def load(path):
    w, f = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True)
    o = np.argsort(w)
    return w[o], f[o]


def opt_norm(w, f):
    m = (w >= 3500) & (w < 9000)
    return f / (trapz(f[m], w[m]) / (9000.0 - 3500.0))   # optical band-mean -> 1


fw, ff = load(RUN + '/lumina_spectrum_formal.csv')
cw, cf = load(RUN + '/lumina_spectrum.csv')
gn = opt_norm(gw, gf)
fn = opt_norm(fw, ff)
cn = opt_norm(cw, cf)

fig, ax = plt.subplots(2, 1, figsize=(12, 9))
for axi, (lo, hi, ttl) in zip(ax, [(2500, 10000, 'Optical (3500-9000A normalized)'),
                                    (2500, 16000, 'Full UV-optical-NIR')]):
    axi.plot(gw, gn, color='k', lw=2.0, label='CMFGEN gold (DDC15 0.976d)', zorder=5)
    axi.plot(fw, fn, color='#D97757', lw=1.3, alpha=0.9,
             label='LUMINA formal (observer-frame, P-Cygni)')
    axi.plot(cw, cn, color='#3898EC', lw=1.3, alpha=0.85,
             label='LUMINA comoving (continuum-blended)')
    axi.set_xlim(lo, hi)
    axi.set_ylim(0, 2.6)
    axi.set_xlabel('wavelength (A)')
    axi.set_ylabel('normalized F_lambda')
    axi.set_title(ttl)
    axi.legend(fontsize=9, loc='upper right')
    axi.grid(alpha=0.25)
    for b in (3500, 5000, 9000):
        axi.axvline(b, color='gray', ls=':', alpha=0.3)

fig.suptitle('Faithful fix: COLL_FIX + ION_LOCK + FALLBACK_TE + LTE_NCRIT=1e8 (no b_k cap)\n'
             'formal color 8024A (gold 7424), optRMS 0.351, UV 0.94x gold',
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = ROOT + '/figures/2026-06-18_ddc15_faithful_fix_spectrum_vs_cmfgen.png'
plt.savefig(out, dpi=120)
print('saved', out)
