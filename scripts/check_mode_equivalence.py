#!/usr/bin/env python3
"""Mode-equivalence anti-pattern detector — catches silent-noop bugs.

Premise: two runs with the SAME inputs but DIFFERENT line-interaction modes
(scatter vs downbranch vs macroatom) MUST produce different output spectra.
If they don't, the alternative mode is silently no-op'ing — exactly the
fingerprint of the #305-3 L2M schema bug (159145 ↔ 159395 byte-identical).

This is a NECESSARY (not sufficient) check:
  * Catches: silent no-op (mode flag has zero effect)
  * Doesn't catch: wrong-physics mode that still emits a different spectrum

Usage:
    check_mode_equivalence.py <run_dir_A> <run_dir_B>
                              [--spectrum lumina_spectrum_formal.csv]
                              [--tol 0.005]
                              [--bands]

Default σ_MC tolerance 0.005 (5×10⁻³) — anything below is byte-identical
relative to MC noise. Pass `--bands` to report per-band ratios too.

Exit codes:
    0 = spectra differ (≥ tol) — mode flag is doing something
    1 = spectra agree (< tol) — SUSPECTED silent no-op
    2 = files missing / unreadable
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd


def load_spectrum(run_dir: str, fname: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = os.path.join(run_dir, fname)
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p)
    lam_col = [c for c in df.columns if "wavelength" in c.lower()][0]
    flux_col = [c for c in df.columns if "flux" in c.lower()][0]
    return df[lam_col].to_numpy(), df[flux_col].to_numpy()


def per_band(lam: np.ndarray, fa: np.ndarray, fb: np.ndarray) -> dict:
    bands = {
        "UV [1500,3000]":  (1500, 3000),
        "UV [3000,4000]":  (3000, 4000),
        "blue [4000,5500]": (4000, 5500),
        "red [5500,7000]":  (5500, 7000),
        "NIR I [7000,9000]": (7000, 9000),
        "NIR II [9000,10200]": (9000, 10200),
    }
    out = {}
    for name, (lo, hi) in bands.items():
        m = (lam >= lo) & (lam < hi)
        if not m.any():
            continue
        a = float(fa[m].sum())
        b = float(fb[m].sum())
        ratio = a / b if b > 0 else float("nan")
        out[name] = (a, b, ratio)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--spectrum", default="lumina_spectrum_formal.csv",
                    help="spectrum filename (default: lumina_spectrum_formal.csv)")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="relative RMS tolerance below which runs are 'identical' (default 0.005)")
    ap.add_argument("--bands", action="store_true",
                    help="report per-band integral ratios")
    args = ap.parse_args()

    sa = load_spectrum(args.run_a, args.spectrum)
    sb = load_spectrum(args.run_b, args.spectrum)
    if sa is None:
        print(f"[FAIL] missing {args.spectrum} in {args.run_a}", file=sys.stderr)
        return 2
    if sb is None:
        # try MC spectrum as fallback
        if args.spectrum == "lumina_spectrum_formal.csv":
            print(f"  [INFO] formal spectrum missing in {args.run_b}, trying lumina_spectrum.csv")
            sa = load_spectrum(args.run_a, "lumina_spectrum.csv")
            sb = load_spectrum(args.run_b, "lumina_spectrum.csv")
            args.spectrum = "lumina_spectrum.csv"
            args.tol = max(args.tol, 0.01)   # MC noise floor higher
        if sb is None:
            print(f"[FAIL] missing {args.spectrum} in {args.run_b}", file=sys.stderr)
            return 2

    lam_a, fa = sa
    lam_b, fb = sb
    if lam_a.shape != lam_b.shape or not np.allclose(lam_a, lam_b):
        print(f"[FAIL] wavelength grids differ between {args.run_a} and {args.run_b}",
              file=sys.stderr)
        return 2

    # byte-identical check first (the hardest no-op fingerprint)
    byte_identical = bool(np.array_equal(fa, fb))
    # relative L2 norm of difference, normalised by mean flux
    diff = fa - fb
    mean = 0.5 * (np.abs(fa).mean() + np.abs(fb).mean())
    if mean == 0:
        print(f"[FAIL] both spectra are zero — degenerate case")
        return 2
    rel_rms = float(np.sqrt((diff ** 2).mean()) / mean)
    max_abs_dev = float(np.max(np.abs(diff)))
    max_rel = max_abs_dev / mean

    print(f"=== mode-equivalence check ===")
    print(f"  run A   : {args.run_a}")
    print(f"  run B   : {args.run_b}")
    print(f"  spectrum: {args.spectrum} ({lam_a.size} bins, λ ∈ [{lam_a.min():.0f}, {lam_a.max():.0f}] Å)")
    print(f"  byte-identical : {byte_identical}")
    print(f"  rel RMS diff   : {rel_rms:.6f}  (tolerance {args.tol:.4f})")
    print(f"  max |Δ| / mean : {max_rel:.6f}")

    if args.bands:
        print(f"\n  per-band integrals (A / B):")
        for name, (a, b, r) in per_band(lam_a, fa, fb).items():
            print(f"    {name:24s}  A={a:.3e}  B={b:.3e}  A/B={r:.4f}")

    if byte_identical:
        print(f"\n=== FAIL: spectra are byte-identical — SILENT NO-OP suspected ===")
        return 1
    if rel_rms < args.tol:
        print(f"\n=== FAIL: rel RMS {rel_rms:.6f} < tol {args.tol:.4f} — mode flag has no measurable effect ===")
        return 1

    print(f"\n=== PASS: spectra differ above MC noise floor ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
