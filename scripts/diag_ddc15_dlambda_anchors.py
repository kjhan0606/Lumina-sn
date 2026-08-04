#!/usr/bin/env python3
"""Decisive frame-vs-resolution diagnostic (defect 6): measure the LUMINA->DDC15
wavelength offset Delta-lambda at several anchor wavelengths.

If Delta-lambda/lambda = const  -> multiplicative -> expansion-Doppler / frame bug.
If Delta-lambda      = const  -> additive       -> resolution/binning-grid artifact.

Read-only; uses existing cap-fix spectra (macroatom 162997, scatter 162754)."""
import numpy as np

REF = "data/ddc15_hydro/DDC15_spec_2500_25500_interp5_000.976d.dat"
RUNS = {
    "macroatom_162997": "logs/paperDDC15init_ddc15init_macroatom_cro1_cfe1_kp0_162997/lumina_spectrum.csv",
    "scatter_162754":   "logs/paperDDC15init_ddc15init_scatter_cro1_cfe1_162754/lumina_spectrum.csv",
}
ANCHORS = [4500.0, 6595.0, 7500.0, 9000.0]
HALFWIN = 450.0       # +/- window around each anchor (A)
MAXLAG  = 700.0       # search +/- this lag (A)
GRID    = 1.0         # resample step (A)

def load_ref():
    d = np.loadtxt(REF)
    return d[:, 0], d[:, 1]

def load_lum(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d["wavelength_angstrom"], d["flux"]

def highpass(y, w=31):
    """subtract a boxcar-smoothed continuum to isolate features/slope structure"""
    k = np.ones(w) / w
    sm = np.convolve(y, k, mode="same")
    return y - sm

def measure_shift(lref, fref, llum, flum, anchor):
    lo, hi = anchor - HALFWIN, anchor + HALFWIN
    grid = np.arange(lo, hi + GRID, GRID)
    r = np.interp(grid, lref, fref)
    l = np.interp(grid, llum, flum)
    # normalize each window
    r = highpass(r); l = highpass(l)
    r = (r - r.mean()) / (r.std() + 1e-30)
    l = (l - l.mean()) / (l.std() + 1e-30)
    lags = np.arange(-MAXLAG, MAXLAG + GRID, GRID)
    best_lag, best_c = 0.0, -2.0
    for lag in lags:
        nshift = int(round(lag / GRID))
        # shift LUMINA by +lag: lum(grid) compared to ref(grid - lag)
        ls = np.roll(l, -nshift)
        # valid (non-wrapped) region
        if nshift >= 0:
            a, b = ls[:len(ls)-nshift], r[:len(r)-nshift]
        else:
            a, b = ls[-nshift:], r[-nshift:]
        if len(a) < 50:
            continue
        c = float(np.dot(a, b) / len(a))
        if c > best_c:
            best_c, best_lag = c, lag
    # best_lag>0 means LUMINA must shift blue (toward shorter lambda) to match DDC15
    # i.e. LUMINA is REDSHIFTED by best_lag
    return best_lag, best_c

lref, fref = load_ref()
for name, path in RUNS.items():
    llum, flum = load_lum(path)
    print(f"\n=== {name} : Delta-lambda(LUMINA redshift) vs DDC15 ===")
    print(f"{'anchor':>8} {'Dlam(A)':>9} {'Dlam/lam':>10} {'v(km/s)':>9} {'corr':>6}")
    dl_over_l = []
    for a in ANCHORS:
        dl, c = measure_shift(lref, fref, llum, flum, a)
        frac = dl / a
        v = 299792.458 * frac
        dl_over_l.append(frac)
        print(f"{a:8.0f} {dl:9.1f} {frac:10.4f} {v:9.0f} {c:6.2f}")
    dl_over_l = np.array(dl_over_l)
    print(f"  Dlam/lam: mean={dl_over_l.mean():.4f} std={dl_over_l.std():.4f} "
          f"spread={dl_over_l.max()-dl_over_l.min():.4f}")
    print(f"  -> {'CONSTANT Dlam/lam (multiplicative=FRAME/Doppler)' if dl_over_l.std()<0.010 else 'NON-constant Dlam/lam'}")
