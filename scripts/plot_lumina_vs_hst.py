#!/usr/bin/env python3
"""LUMINA formal-integral spectrum vs HST 2011fe B-max comparison plot.

Usage: plot_lumina_vs_hst.py <lumina_csv> [<lumina_csv2> ...] [--out figures/X.png]
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"

BANDS = [
    ('UVblnk[1700-3000]',  (1700, 3000)),
    ('CaIIK[3000-3300]',   (3000, 3300)),
    ('UVtgt[3300-3700]',   (3300, 3700)),
    ('fluor[3700-4400]',   (3700, 4400)),
    ('green[4400-5500]',   (4400, 5500)),
    ('red[5500-7000]',     (5500, 7000)),
]


def band_lumin(wl, fl):
    out = {}
    for name, (lo, hi) in BANDS:
        m = (wl >= lo) & (wl <= hi)
        out[name] = np.trapezoid(fl[m], wl[m]) if m.sum() >= 2 else np.nan
    out['opt[3800-7000]'] = np.trapezoid(
        fl[(wl >= 3800) & (wl <= 7000)], wl[(wl >= 3800) & (wl <= 7000)])
    return out


def normalize_by_band(wl, fl, ref_wl, ref_fl, lo=4400, hi=5500):
    """Scale spectrum so flux integral over [lo, hi] matches reference."""
    m = (wl >= lo) & (wl <= hi)
    rm = (ref_wl >= lo) & (ref_wl <= hi)
    F = np.trapezoid(fl[m], wl[m])
    R = np.trapezoid(ref_fl[rm], ref_wl[rm])
    return fl * (R / F), R / F


def smooth(wl, fl, sigma_A=10.0):
    from scipy.ndimage import gaussian_filter1d
    dlam = np.median(np.diff(wl))
    return gaussian_filter1d(fl, sigma_A / dlam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LUMINA CSV files")
    ap.add_argument("--out", default="figures/lumina_vs_hst_2011fe.png")
    ap.add_argument("--smooth", type=float, default=10.0, help="Gaussian sigma in Å")
    ap.add_argument("--norm-band", default="green",
                    choices=["green", "red", "fluor", "none"])
    args = ap.parse_args()

    # HST B-max obs
    hst = np.loadtxt(HST, delimiter=",", skiprows=1)
    hwl, hfl = hst[:, 0], hst[:, 1]
    hst_bands = band_lumin(hwl, hfl)

    # Layout: 2x2 — top row UV (zoom 1700-4400), bottom row optical (3800-7000)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    ax_uv, ax_opt = axes

    # HST
    h_smooth = smooth(hwl, hfl, args.smooth) if args.smooth > 0 else hfl
    ax_uv.plot(hwl, h_smooth, "k-", lw=1.4, label="HST 2011fe B-max", zorder=10)
    ax_opt.plot(hwl, h_smooth, "k-", lw=1.4, label="HST 2011fe B-max", zorder=10)

    rows = [("HST 2011fe", hst_bands)]

    colors = ["#3898EC", "#D97757", "#4EC9B0", "#FFC107", "#B07EDF", "#7CB342"]

    norm_lo = {"green": 4400, "red": 5500, "fluor": 3700, "none": None}[args.norm_band]
    norm_hi = {"green": 5500, "red": 7000, "fluor": 4400, "none": None}[args.norm_band]

    for i, path in enumerate(args.inputs):
        p = Path(path)
        if not p.exists():
            print(f"Missing: {p}")
            continue
        d = np.loadtxt(p, delimiter=",", skiprows=1)
        wl, fl = d[:, 0], d[:, 1]
        if norm_lo is not None:
            fl_scaled, scale = normalize_by_band(
                wl, fl, hwl, hfl, lo=norm_lo, hi=norm_hi)
        else:
            fl_scaled = fl
            scale = 1.0
        fl_smooth = smooth(wl, fl_scaled, args.smooth) if args.smooth > 0 else fl_scaled
        label = p.parent.name if p.parent.name.startswith("exp_") else p.stem
        c = colors[i % len(colors)]
        ax_uv.plot(wl, fl_smooth, "-", lw=1.0, color=c, label=label, alpha=0.85)
        ax_opt.plot(wl, fl_smooth, "-", lw=1.0, color=c, label=label, alpha=0.85)
        rows.append((label, band_lumin(wl, fl_scaled)))

    ax_uv.set_xlim(1700, 4400)
    ax_uv.set_ylabel(r"F$_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax_uv.set_title("UV/blue (1700-4400 Å)")
    ax_uv.axvspan(3300, 3700, alpha=0.08, color="orange", label="UVtgt")
    ax_uv.axvspan(3700, 4400, alpha=0.08, color="green", label="fluor")
    ax_uv.legend(loc="upper left", fontsize=8, ncol=2)
    ax_uv.grid(True, alpha=0.3)

    ax_opt.set_xlim(3800, 7500)
    ax_opt.set_xlabel(r"Wavelength [Å]")
    ax_opt.set_ylabel(r"F$_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
    ax_opt.set_title("Optical (3800-7500 Å)")
    ax_opt.legend(loc="upper right", fontsize=8)
    ax_opt.grid(True, alpha=0.3)

    norm_msg = f", normalized to HST in {norm_lo}-{norm_hi} Å" if norm_lo else ""
    fig.suptitle(f"LUMINA vs HST 2011fe B-max (smooth σ={args.smooth} Å{norm_msg})")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")

    # Print comparison table
    print(f"\n{'Run':32s} | {'UVbl':>9s} {'CaK':>9s} {'UVtgt':>9s} {'fluor':>9s} {'green':>9s} {'red':>9s} | UVtgt/opt")
    print("-" * 110)
    for name, b in rows:
        opt = b['opt[3800-7000]']
        ratios = [b[k] / opt * 1000 for k, _ in BANDS]
        u_ratio = ratios[2]  # UVtgt
        print(f"{name:32s} | "
              f"{ratios[0]:8.1f}‰ {ratios[1]:8.1f}‰ {ratios[2]:8.1f}‰ "
              f"{ratios[3]:8.1f}‰ {ratios[4]:8.1f}‰ {ratios[5]:8.1f}‰ | "
              f"{u_ratio:6.1f}‰")


if __name__ == "__main__":
    main()
