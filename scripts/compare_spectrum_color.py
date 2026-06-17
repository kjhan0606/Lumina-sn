#!/usr/bin/env python3
"""Compare a LUMINA emergent spectrum CSV against the DDC15 0.976d CMFGEN gold.

Reports scale-independent color metrics (flux-weighted mean wavelength,
too-blue<4500 fraction, peak wavelength) + optical shape RMS.

Usage: compare_spectrum_color.py spec1.csv [label1] [spec2.csv label2 ...]
"""
import sys
import numpy as np

GOLD = "data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"


def load_csv(path):
    try:
        w, f = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    except Exception:
        w, f = np.loadtxt(path, unpack=True)
    return w, f


def metrics(w, f, lo=3000, hi=10000):
    m = (w >= lo) & (w <= hi) & np.isfinite(f) & (f > 0)
    w, f = w[m], f[m]
    order = np.argsort(w)
    w, f = w[order], f[order]
    fn = f / np.trapezoid(f, w)
    color = np.trapezoid(fn * w, w)
    blue = w < 4500
    tb = np.trapezoid(fn[blue], w[blue]) if blue.any() else 0.0
    peak = w[np.argmax(fn)]
    return w, fn, color, tb, peak


def main():
    gw, gf = np.loadtxt(GOLD, unpack=True)
    gwl, gn, gcol, gtb, gpk = metrics(gw, gf)
    print(f"{'label':<22} {'color(A)':>9} {'tooblue':>8} {'peak(A)':>8} {'optRMS':>8}")
    print(f"{'GOLD':<22} {gcol:>9.0f} {gtb:>8.3f} {gpk:>8.0f} {'-':>8}")
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        path = args[i]
        label = args[i + 1] if i + 1 < len(args) and not args[i + 1].endswith((".csv", ".dat")) else path.split("/")[-1]
        step = 2 if label != path.split("/")[-1] else 1
        w, f = load_csv(path)
        wl, fn, col, tb, pk = metrics(w, f)
        gi = np.interp(wl, gwl, gn)
        opt = (wl >= 4000) & (wl <= 9000)
        rms = np.sqrt(np.mean(((fn[opt] - gi[opt]) / gi[opt].max()) ** 2))
        print(f"{label[:22]:<22} {col:>9.0f} {tb:>8.3f} {pk:>8.0f} {rms:>8.3f}")
        i += step


if __name__ == "__main__":
    main()
