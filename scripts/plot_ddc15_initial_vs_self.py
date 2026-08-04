#!/usr/bin/env python3
"""Overlay LUMINA's 0.976d initial-epoch output against DDC15's OWN emergent
spectrum at the same epoch (DDC15_spec_2500_25500_interp5_000.976d.dat). This is
the most controlled self-test: same model, same epoch, same structure (taken
as-is). Divergence here is internal to LUMINA (transport/atomic), not input.

Usage:
  plot_ddc15_initial_vs_self.py SCATTER_SPEC [MACROATOM_SPEC]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DDC15 = Path("data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat")
BANDS = {"blue": (3500, 4500), "green": (4500, 5500),
         "red": (5500, 7000), "NIR": (7000, 9000)}
NORM = (3500, 9000)


def load(p):
    a = np.loadtxt(p)
    if a.shape[1] > 2:  # LUMINA csv has header -> use genfromtxt fallback
        pass
    return a[:, 0], a[:, 1]


def load_csv(p):
    a = np.genfromtxt(p, delimiter=",", names=True)
    return a["wavelength_angstrom"], a["flux"]


def normband(w, f, ref_w, ref_f):
    m = (w >= NORM[0]) & (w <= NORM[1])
    rm = (ref_w >= NORM[0]) & (ref_w <= NORM[1])
    return f / np.trapz(f[m], w[m]) * np.trapz(ref_f[rm], ref_w[rm])


def band_l1(w, f, rw, rf):
    out = {}
    fi = np.interp(rw, w, f)
    nm0 = (rw >= NORM[0]) & (rw <= NORM[1])
    fi = fi / np.trapz(fi[nm0], rw[nm0])
    rfn = rf / np.trapz(rf[nm0], rw[nm0])
    for nm, (a, b) in BANDS.items():
        m = (rw >= a) & (rw <= b)
        r = np.trapz(fi[m], rw[m]) / np.trapz(rfn[m], rw[m])
        out[nm] = r
    return out


def main():
    dw, df = load(DDC15)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dw, df / np.trapz(df[(dw >= NORM[0]) & (dw <= NORM[1])],
            dw[(dw >= NORM[0]) & (dw <= NORM[1])]),
            "k-", lw=1.6, label="DDC15 self 0.976d", zorder=5)

    colors = {"scatter": "C0", "macroatom": "C3"}
    for arg, mode in zip(sys.argv[1:3], ["scatter", "macroatom"]):
        w, f = load_csv(arg)
        m = (w >= NORM[0]) & (w <= NORM[1])
        fn = f / np.trapz(f[m], w[m])
        ax.plot(w, fn, color=colors[mode], lw=1.0, alpha=0.85, label=f"LUMINA {mode}")
        bl = band_l1(w, f, dw, df)
        l1 = sum(abs(v - 1) for v in bl.values())
        print(f"[{mode}] band ratios (LUMINA/DDC15):  "
              + "  ".join(f"{k}={v:.3f}" for k, v in bl.items())
              + f"   L1={l1:.3f}")

    ax.set_xlim(2500, 10000)
    ax.set_xlabel("wavelength (Å)")
    ax.set_ylabel("normalized flux")
    ax.set_title("DDC15 0.976d initial-epoch self-test: LUMINA vs DDC15 own spectrum")
    ax.legend()
    for a, b in BANDS.values():
        ax.axvspan(a, b, alpha=0.04, color="gray")
    out = Path("figures/2026-06-03_ddc15_initial_0p976d_vs_self.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
