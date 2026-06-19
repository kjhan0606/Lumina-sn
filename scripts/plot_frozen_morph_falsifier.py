#!/usr/bin/env python3
"""Frozen-plasma morphology falsifier (P-Cygni gate).

Overlays the DDC15 0.976d CMFGEN gold against:
  (1) thermal-forest formal spectrum   (baseline, no scattering: S_l=B(Te))
  (2) frozen-morph formal spectrum      (forest scatters, eps=0; plasma FROZEN)

Decision (one-sided): if (2) develops P-Cygni troughs at Ca II / Si II / S II
that (1) lacks, the thermal forest IS the morphology blocker -> build A4.
If (2) is still featureless like (1), the lost photospheric memory is elsewhere
(geometry / inner-BC dilution / freq redistribution) -> abort the A4 plan.

Usage: plot_frozen_morph_falsifier.py <thermal_formal.csv> <frozen_formal.csv>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

trapz = getattr(np, 'trapezoid', getattr(np, 'trapz'))
ROOT = '/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'

THERMAL = sys.argv[1] if len(sys.argv) > 1 else (
    ROOT + '/logs/ddc15_pc_phase3_jnul1_radls1_linere1_ratio1.0_pi1_fz1_167177'
           '/lumina_spectrum_formal.csv')
FROZEN = sys.argv[2] if len(sys.argv) > 2 else (
    ROOT + '/logs/ddc15_pc_phase3_jnul0_radls0_linere0_ratio0.9_pi1_fz0_167373'
           '/lumina_spectrum_formal.csv')

gw, gf = np.loadtxt(ROOT + '/data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat',
                    unpack=True)


def load(path):
    w, f = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True)
    o = np.argsort(w)
    return w[o], f[o]


def opt_norm(w, f):
    m = (w >= 3500) & (w < 9000)
    return f / (trapz(f[m], w[m]) / (9000.0 - 3500.0))


def trough_depth(w, f, c0, c1, lo, hi):
    """1 - min(flux in [lo,hi]) / mean(flux in continuum windows c0,c1)."""
    cont = np.concatenate([f[(w >= c0[0]) & (w < c0[1])],
                           f[(w >= c1[0]) & (w < c1[1])]])
    if cont.size == 0:
        return np.nan
    cmean = np.median(cont)
    m = (w >= lo) & (w < hi)
    if not m.any() or cmean <= 0:
        return np.nan
    return 1.0 - f[m].min() / cmean


gn = opt_norm(gw, gf)
tw, tf = load(THERMAL); tn = opt_norm(tw, tf)
fw, ff = load(FROZEN);  fn = opt_norm(fw, ff)

# P-Cygni diagnostic features: (name, trough lo, hi, cont-blue window, cont-red window)
FEAT = [
    ('Ca II H&K 3945',   3700, 3950, (3550, 3650), (4000, 4150)),
    ('Si II 6355',       6050, 6300, (5900, 6000), (6400, 6550)),
    ('S II "W" 5400',    5250, 5500, (5100, 5200), (5550, 5700)),
    ('Ca II NIR 8500',   8200, 8550, (7900, 8100), (8700, 8900)),
]

print(f"{'feature':<20} {'gold':>8} {'thermal':>8} {'frozen':>8}   verdict")
print('-' * 64)
any_pass = False
for name, lo, hi, c0, c1 in FEAT:
    dg = trough_depth(gw, gn, c0, c1, lo, hi)
    dt = trough_depth(tw, tn, c0, c1, lo, hi)
    df = trough_depth(fw, fn, c0, c1, lo, hi)
    # PASS if frozen deepens the trough materially toward gold vs thermal
    gain = df - dt
    v = 'TROUGH+' if gain > 0.05 else ('flat' if abs(gain) <= 0.05 else 'shallower')
    if gain > 0.05:
        any_pass = True
    print(f"{name:<20} {dg:>8.3f} {dt:>8.3f} {df:>8.3f}   {v} (Δ={gain:+.3f})")
print('-' * 64)
print("VERDICT:", "FROZEN FOREST-SCATTER PRODUCES TROUGHS -> thermal forest is "
      "the blocker -> build A4" if any_pass else
      "frozen scatter still featureless -> morphology root is elsewhere "
      "(geometry/dilution) -> abort A4 plan")

fig, ax = plt.subplots(2, 1, figsize=(13, 9))
for axi, (xlo, xhi, ttl) in zip(
        ax, [(3500, 9000, 'Optical (3500-9000A norm) — P-Cygni gate'),
             (2500, 12000, 'UV-optical-NIR')]):
    axi.plot(gw, gn, color='k', lw=2.0, label='CMFGEN gold', zorder=5)
    axi.plot(tw, tn, color='#3898EC', lw=1.2, alpha=0.85,
             label='thermal forest (S_l=B(Te), baseline)')
    axi.plot(fw, fn, color='#D97757', lw=1.4, alpha=0.9,
             label='frozen-morph: forest scatter ε=0 (plasma frozen)')
    for name, lo, hi, c0, c1 in FEAT:
        axi.axvspan(lo, hi, color='gray', alpha=0.08)
    axi.set_xlim(xlo, xhi); axi.set_ylim(0, None)
    axi.set_title(ttl); axi.legend(fontsize=9); axi.set_ylabel('normalized flux')
ax[-1].set_xlabel('wavelength (Å)')
plt.tight_layout()
out = ROOT + '/figures/2026-06-18_frozen_morph_falsifier.png'
plt.savefig(out, dpi=110)
print('figure ->', out)
