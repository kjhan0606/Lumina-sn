#!/usr/bin/env python3
"""Stitch HST/STIS +0.4d B-max observations into a single 1663-10249 A spectrum.

Inputs (from convert_hst_uv_2011fe.py output):
  G230LB (1663-3068 A) — NUV
  G430L  (2889-5697 A) — blue/optical
  G750L  (5257-10249 A) — red/NIR

Outputs:
  data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv
  data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.png  (sanity check)

Crossover handling: in overlap regions, average the two gratings weighted
by 1/error^2.  Reset error to combined inverse-variance.

Also overlay the existing Snifs-based Bmax CSV used by the optical
likelihood, to verify they agree in the optical.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

UV_DIR = Path(__file__).resolve().parent.parent / "data" / "sn2011fe" / "hst_uv"
OBS_DIR = Path(__file__).resolve().parent.parent / "data" / "sn2011fe"
SNIFS_BMAX = OBS_DIR / "sn2011fe_observed_Bmax.csv"


def load_csv(path):
    arr = np.genfromtxt(path, delimiter=',', skip_header=1)
    return arr[:, 0], arr[:, 1], arr[:, 2]  # wave, flux, err


def coadd(files):
    """Inverse-variance coadd across multiple exposures (already on same grid)."""
    waves, fluxes, errs = [], [], []
    for f in files:
        w, fl, er = load_csv(f)
        waves.append(w); fluxes.append(fl); errs.append(er)
    # Same grating exposures on identical wavelength grid
    w0 = waves[0]
    for w in waves[1:]:
        assert len(w) == len(w0) and np.allclose(w, w0, rtol=1e-4), \
            "wavelength mismatch in coadd"
    fluxes = np.array(fluxes); errs = np.array(errs)
    # Replace zero/negative err with median
    med = np.median(errs[errs > 0])
    errs[errs <= 0] = med
    w_inv = 1.0 / errs**2
    flux_combined = np.sum(fluxes * w_inv, axis=0) / np.sum(w_inv, axis=0)
    err_combined = 1.0 / np.sqrt(np.sum(w_inv, axis=0))
    return w0, flux_combined, err_combined


def stitch(wA, fA, eA, wB, fB, eB):
    """Stitch two segments wA<=wB.  In overlap, inverse-variance average on
    union grid; outside, keep the native grid."""
    if wA[-1] < wB[0]:
        # No overlap
        w = np.concatenate([wA, wB])
        f = np.concatenate([fA, fB])
        e = np.concatenate([eA, eB])
        return w, f, e
    # Overlap region: [wB[0], wA[-1]]
    overlap_lo = max(wA[0], wB[0])
    overlap_hi = min(wA[-1], wB[-1])

    # Pre-overlap
    pre_mask = wA < overlap_lo
    post_mask = wB > overlap_hi
    w_pre, f_pre, e_pre = wA[pre_mask], fA[pre_mask], eA[pre_mask]
    w_post, f_post, e_post = wB[post_mask], fB[post_mask], eB[post_mask]

    # Build union overlap grid (use whichever is denser)
    dA = np.median(np.diff(wA))
    dB = np.median(np.diff(wB))
    step = min(dA, dB)
    w_ov = np.arange(overlap_lo, overlap_hi + step, step)
    fA_ov = np.interp(w_ov, wA, fA)
    eA_ov = np.interp(w_ov, wA, eA)
    fB_ov = np.interp(w_ov, wB, fB)
    eB_ov = np.interp(w_ov, wB, eB)
    eA_ov = np.where(eA_ov > 0, eA_ov, np.median(eA[eA > 0]))
    eB_ov = np.where(eB_ov > 0, eB_ov, np.median(eB[eB > 0]))
    wA_inv = 1.0 / eA_ov**2
    wB_inv = 1.0 / eB_ov**2
    f_ov = (fA_ov * wA_inv + fB_ov * wB_inv) / (wA_inv + wB_inv)
    e_ov = 1.0 / np.sqrt(wA_inv + wB_inv)

    w = np.concatenate([w_pre, w_ov, w_post])
    f = np.concatenate([f_pre, f_ov, f_post])
    e = np.concatenate([e_pre, e_ov, e_post])
    return w, f, e


def main():
    nuv = sorted(UV_DIR.glob("CCD_G230LB_mjd55814*.csv"))
    blue = sorted(UV_DIR.glob("CCD_G430L_mjd55814*.csv"))
    red = sorted(UV_DIR.glob("CCD_G750L_mjd55814*.csv"))
    print(f"NUV  G230LB exposures: {[p.name for p in nuv]}")
    print(f"BLUE G430L  exposures: {[p.name for p in blue]}")
    print(f"RED  G750L  exposures: {[p.name for p in red]}")

    print("\nCoadding within each grating...")
    wN, fN, eN = coadd(nuv)
    wB, fB, eB = coadd(blue)
    wR, fR, eR = coadd(red)
    print(f"  NUV : {len(wN)} pts, {wN[0]:.1f}-{wN[-1]:.1f} A")
    print(f"  BLUE: {len(wB)} pts, {wB[0]:.1f}-{wB[-1]:.1f} A")
    print(f"  RED : {len(wR)} pts, {wR[0]:.1f}-{wR[-1]:.1f} A")

    print("\nStitching NUV+BLUE...")
    w1, f1, e1 = stitch(wN, fN, eN, wB, fB, eB)
    print("Stitching (NUV+BLUE)+RED...")
    w, f, e = stitch(w1, f1, e1, wR, fR, eR)
    print(f"  Final: {len(w)} pts, {w[0]:.1f}-{w[-1]:.1f} A")

    # Save
    out = UV_DIR / "sn2011fe_hst_bmax_stitched.csv"
    arr = np.column_stack([w, f, e])
    np.savetxt(out, arr, delimiter=',',
               header="wavelength_angstrom,flux_erg_s_cm2_angstrom,error",
               comments='', fmt='%.6e')
    print(f"\nWrote {out}")

    # Sanity plot vs Snifs B-max
    snifs = np.genfromtxt(SNIFS_BMAX, delimiter=',', names=True)
    w_s = snifs['wavelength_angstrom']
    f_s = snifs['flux_erg_s_cm2_angstrom']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ax = axes[0]
    ax.plot(w, f, 'k-', lw=0.6, alpha=0.8, label='HST/STIS +0.4d (stitched)')
    ax.plot(w_s, f_s, 'r-', lw=0.8, alpha=0.7, label='Snifs B-max (current obs)')
    ax.set_xlim(1500, 10500)
    ax.set_xlabel('Wavelength (A)')
    ax.set_ylabel('Flux (erg/s/cm^2/A)')
    ax.set_title('SN 2011fe near B-max: HST/STIS vs Snifs')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(1e-15, ax.get_ylim()[1])

    ax = axes[1]
    overlap = (w >= 3300) & (w <= 9700)
    ax.plot(w[overlap], f[overlap], 'k-', lw=0.6, alpha=0.8,
            label='HST/STIS +0.4d')
    ax.plot(w_s, f_s, 'r-', lw=0.8, alpha=0.7,
            label='Snifs B-max')
    ax.set_xlim(3300, 9700)
    ax.set_xlabel('Wavelength (A)')
    ax.set_ylabel('Flux (erg/s/cm^2/A)')
    ax.set_title('Optical overlap (cross-check)')
    ax.legend()

    plt.tight_layout()
    fig.savefig(UV_DIR / "sn2011fe_hst_bmax_stitched.png", dpi=140)
    plt.close(fig)
    print(f"Wrote sanity plot.")

    # Also save UV-only file
    uv_only = (w >= 1600) & (w <= 3300)
    arr_uv = np.column_stack([w[uv_only], f[uv_only], e[uv_only]])
    out_uv = UV_DIR / "sn2011fe_hst_bmax_uvonly.csv"
    np.savetxt(out_uv, arr_uv, delimiter=',',
               header="wavelength_angstrom,flux_erg_s_cm2_angstrom,error",
               comments='', fmt='%.6e')
    print(f"Wrote UV-only file: {out_uv}")


if __name__ == '__main__':
    main()
