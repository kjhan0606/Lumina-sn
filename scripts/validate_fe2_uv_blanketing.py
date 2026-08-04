#!/usr/bin/env python3
"""Task #41 — Fe II UV blanketing validation vs Mazzali+2014.

Compares LUMINA's UV (1700-3500 Å) prediction at the (c) UV-filtered MAP point
against the HST/STIS stitched B-max observation of SN 2011fe.

Mazzali+2014 (MNRAS 439, 1959) reports for SN 2011fe at B-max:
  * Fe II/III "iron forest" produces deep absorption trough shortward of ~3000 Å
  * Pseudo-continuum peaks at 3500-4000 Å, drops by factor ~3-7 toward 2500 Å
  * UV(1700-3000) / optical(4000-6000) integrated flux ratio ~ 0.1-0.2 in obs
  * Co II/III contributes additional blanketing at 2500-2700 Å

Outputs:
  figures/fe2_uv_blanketing_validation.png  — 1700-7000 Å overlay
  printed table of diagnostic metrics
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
HST_CSV  = ROOT / 'data' / 'sn2011fe' / 'hst_uv' / 'sn2011fe_hst_bmax_stitched.csv'
UV_DIR   = ROOT.parent / 'Lumina-ML' / 'results_uv_methods_compare'


def integrate_band(w, f, lo, hi):
    m = (w >= lo) & (w <= hi)
    if m.sum() < 2:
        return np.nan
    return np.trapz(f[m], w[m])


def normalize_at(w, f, lam_lo=4000, lam_hi=7000):
    m = (w >= lam_lo) & (w <= lam_hi)
    return f / f[m].max() if m.any() else f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='c_map',
                    help='Tag for input files (lumina_<tag>_wave.npy / _flux.npy)')
    args = ap.parse_args()
    tag = args.tag

    lw_path = UV_DIR / f'lumina_{tag}_wave.npy'
    lf_path = UV_DIR / f'lumina_{tag}_flux.npy'
    out_png = ROOT / 'figures' / f'fe2_uv_blanketing_validation_{tag}.png'

    # --- Load ---
    lw = np.load(lw_path)
    lf = np.load(lf_path)
    hst = np.genfromtxt(str(HST_CSV), delimiter=',', names=True)
    hw = hst['wavelength_angstrom']
    hf = hst['flux_erg_s_cm2_angstrom']

    # --- Common comparison grid (1700-9500 A, log-spaced 4000 bins) ---
    grid = np.geomspace(1700.0, 9500.0, 4000)
    li = np.interp(grid, lw, lf, left=0.0, right=0.0)
    hi = np.interp(grid, hw, hf, left=0.0, right=0.0)

    # Peak-normalize each in the optical (4000-7000 A) for shape comparison.
    li_n = normalize_at(grid, li)
    hi_n = normalize_at(grid, hi)

    # --- Diagnostic metrics ---
    bands = {
        'UV_far'   : (1700, 2500),
        'UV_iron'  : (2500, 3000),
        'UV_break' : (2800, 3000),  # right at Fe II edge
        'UV_blue'  : (3000, 3500),
        'Si_II_peak': (4000, 4500),
        'optical' : (4000, 6000),
        'red'     : (5500, 7000),
    }
    print("\n=== Fe II UV blanketing — band-integrated flux ratios ===")
    print(f"{'band':<12}  {'range[A]':<13}  {'LUMINA(norm)':>13}  {'HST(norm)':>13}  {'ratio':>9}")
    for name, (lo, hi_w) in bands.items():
        Lv = integrate_band(grid, li_n, lo, hi_w)
        Hv = integrate_band(grid, hi_n, lo, hi_w)
        r  = Lv / Hv if Hv > 0 else np.nan
        print(f"  {name:<12} {lo:>5}-{hi_w:<5}  {Lv:>13.3e}  {Hv:>13.3e}  {r:>9.3f}")

    uv_opt_lumina = (integrate_band(grid, li_n, 1700, 3000) /
                     integrate_band(grid, li_n, 4000, 6000))
    uv_opt_hst    = (integrate_band(grid, hi_n, 1700, 3000) /
                     integrate_band(grid, hi_n, 4000, 6000))
    print(f"\n  UV(1700-3000) / optical(4000-6000) flux ratio:")
    print(f"    LUMINA : {uv_opt_lumina:.3f}")
    print(f"    HST obs: {uv_opt_hst:.3f}")
    print(f"    ratio  : {uv_opt_lumina / uv_opt_hst:.3f}  "
          f"(1.0 = perfect blanketing match)")

    # 2900 A pseudo-continuum break: F(2700) / F(3300)
    def at(w, x):
        return np.interp(w, grid, x)
    break_lumina = at(2700, li_n) / at(3300, li_n)
    break_hst    = at(2700, hi_n) / at(3300, hi_n)
    print(f"\n  2900 A pseudo-continuum break  F(2700)/F(3300):")
    print(f"    LUMINA : {break_lumina:.3f}")
    print(f"    HST obs: {break_hst:.3f}")
    print(f"    Mazzali+14 reports a deep break with ratio ~0.2-0.4 at B-max.")

    # --- Plot ---
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={'height_ratios': [2, 1]})

    ax = axes[0]
    ax.plot(grid, hi_n, color='black', lw=1.4, label='SN 2011fe HST (B-max)', alpha=0.85)
    ax.plot(grid, li_n, color='C3',   lw=1.4, label=f'LUMINA ({tag}), NLTE', alpha=0.9)
    ax.axvspan(1700, 3000, color='C0', alpha=0.07, label='Fe II UV blanketing zone')
    ax.axvline(2900, color='gray', ls='--', lw=0.7, alpha=0.5)
    ax.set_xlim(1700, 7500)
    ax.set_ylim(0, 1.2 * max(hi_n.max(), li_n.max()))
    ax.set_xlabel(r'Wavelength [Å]')
    ax.set_ylabel('Flux (peak-normalized in 4000-7000 Å)')
    ax.set_title('SN 2011fe B-max — Fe II UV blanketing validation')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # Zoomed UV
    ax = axes[1]
    ax.plot(grid, hi_n, color='black', lw=1.5, label='HST')
    ax.plot(grid, li_n, color='C3', lw=1.5, label='LUMINA')
    ax.axvline(2400, color='C2', ls=':', alpha=0.5)
    ax.axvline(2600, color='C2', ls=':', alpha=0.5)
    ax.axvline(2900, color='gray', ls='--', alpha=0.5)
    ax.text(2500, 0.05, 'Fe II 2400-2600\nresonance trough',
            ha='center', fontsize=8, alpha=0.7)
    ax.set_xlim(1700, 3500)
    ax.set_ylim(0, max(hi_n[(grid > 1700) & (grid < 3500)].max(),
                       li_n[(grid > 1700) & (grid < 3500)].max()) * 1.2)
    ax.set_xlabel(r'Wavelength [Å]')
    ax.set_ylabel('Flux (norm.)')
    ax.set_title(f'UV detail — UV/opt ratio: LUMINA={uv_opt_lumina:.3f}  HST={uv_opt_hst:.3f}')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"\nSaved: {out_png}")


if __name__ == '__main__':
    main()
