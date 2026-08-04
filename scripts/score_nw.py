#!/usr/bin/env python3
"""Noise-weighted baseline-normalized RMS for LUMINA spectrum vs observation.

For HST, uses the per-pixel `error` column in the input CSV.
For Snifs (no error column), estimates per-pixel σ empirically via the
Stoehr+ 2008 DER_SNR algorithm applied to the baseline-normalized
observation in a sliding window.

Two output metrics:
  rms_bn            — back-compat unweighted RMS_bn (matches inline SLURM)
  chi_bn (NEW)      — sqrt( <((mod_bn - obs_bn)/σ_bn)^2> ); ≈1 at noise level

Usage as a module:
    from score_nw import rms_bn_nw
    r, chi = rms_bn_nw(mod_lam, mod_flu, obs_lam, obs_flu,
                       obs_err=None, fwhm=20000., wl_lo=3300., wl_hi=8000.)

Usage as a script:
    python score_nw.py <lumina_spectrum.csv> <snifs.csv> [<hst_stitched.csv> --hst-tag <tag>]
"""
from __future__ import annotations
import sys, glob, argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

C_KMS = 2.998e5


def band_int(lam, flu, lo, hi):
    sel = (lam >= lo) & (lam <= hi)
    return float(np.trapezoid(flu[sel], lam[sel]))


def gaussian_continuum(lam, flu, fwhm_kms):
    """Match the inline scripts: per-pixel Gauss kernel via scipy."""
    dl = float(np.median(np.diff(lam)))
    mid = 0.5 * (lam[0] + lam[-1])
    sig_pix = (fwhm_kms / C_KMS) * mid / 2.355 / dl
    return gaussian_filter1d(flu, sig_pix, mode="nearest")


def der_snr_local(flu, win=11):
    """Robust per-pixel noise estimator (Stoehr+ 2008 DER_SNR), localized.

    σ_local = 1.482602 / sqrt(6) * MAD( 2 f_i - f_{i-2} - f_{i+2} )
    evaluated in a sliding window of `win` pixels. The 2-pixel diff cancels
    a locally linear continuum, so the residual is dominated by photon noise
    plus any line-shape residual narrower than the diff stride (~6 Å Snifs).

    Returns per-pixel σ on the same grid as `flu`.
    """
    n = len(flu)
    out = np.zeros(n)
    if n < 5:
        return out + np.nanstd(flu)
    d = np.zeros(n)
    d[2:-2] = 2.0 * flu[2:-2] - flu[:-4] - flu[4:]
    # global fallback estimate for window-edge handling
    g_sigma = 0.6052697 * np.median(np.abs(d[2:-2]))
    half = win // 2
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        seg = d[a:b]
        if len(seg) < 3:
            out[i] = g_sigma
        else:
            out[i] = 0.6052697 * np.median(np.abs(seg))
    out = np.where(out > 0, out, g_sigma)
    return out


def rms_bn_nw(mod_lam, mod_flu, obs_lam, obs_flu, gO,
              obs_err=None, fwhm=20000.0, wl_lo=3300.0, wl_hi=8000.0,
              noise_win=11, clip_chi=10.0):
    """Baseline-normalized RMS plus its noise-weighted χ counterpart.

    Returns (rms_bn, chi_bn).

    `gO` is the green-band integral of the observation (4500-5800 Å); model
    is rescaled by gO/g_model so that the reference band matches before
    baseline division.

    `obs_err` (optional) is per-pixel σ on `obs_lam`. If None, estimated
    via `der_snr_local`. The obs error is propagated through the baseline
    division: σ_bn(λ) = σ_obs(λ) / baseline_obs(λ).

    `clip_chi` caps any single-pixel χ to avoid one bad pixel dominating
    the mean (set to None to disable).
    """
    # Renormalize model to obs green-band integral
    g_mod = band_int(mod_lam, mod_flu, 4500, 5800)
    if g_mod <= 0:
        return float("nan"), float("nan")
    mod_flu = mod_flu * (gO / g_mod)

    # Restrict to the analysis band
    selO = (obs_lam >= wl_lo) & (obs_lam <= wl_hi)
    ol, of = obs_lam[selO], obs_flu[selO]
    oe = obs_err[selO] if obs_err is not None else None
    if len(ol) < 10:
        return float("nan"), float("nan")

    sO = gaussian_continuum(ol, of, fwhm)

    # Model baseline on its own grid, then interp onto obs grid
    selM = (mod_lam >= wl_lo - 100) & (mod_lam <= wl_hi + 100)
    ml, mf = mod_lam[selM], mod_flu[selM]
    if len(ml) < 10:
        return float("nan"), float("nan")
    sM = gaussian_continuum(ml, mf, fwhm)

    common = (ol >= ml[0]) & (ol <= ml[-1])
    if common.sum() < 10:
        return float("nan"), float("nan")
    obs_bn = of[common] / sO[common]
    mod_bn = np.interp(ol[common], ml, mf / sM)
    resid = obs_bn - mod_bn

    rms_bn = float(np.sqrt(np.mean(resid ** 2)))

    # Noise on the baseline-normalized obs
    if oe is not None:
        sigma_bn = oe[common] / sO[common]
    else:
        sigma_bn = der_snr_local(obs_bn, win=noise_win)

    # Guard against zero/negative σ (rare)
    sigma_bn = np.where(sigma_bn > 0, sigma_bn, np.nanmedian(sigma_bn[sigma_bn > 0]))
    chi = resid / sigma_bn
    if clip_chi is not None:
        chi = np.clip(chi, -clip_chi, clip_chi)
    chi_bn = float(np.sqrt(np.mean(chi ** 2)))

    return rms_bn, chi_bn


def stitch_hst(hst_dir, hst_tag):
    """Stitch G230LB+G430L+G750L into a single (lam, flu, err) tuple."""
    parts = []
    for grat, w_lo, w_hi in [("G230LB", 1700, 2900),
                             ("G430L",  2900, 5266),
                             ("G750L",  5266, 10000)]:
        files = sorted(glob.glob(f"{hst_dir}/CCD_{grat}_*{hst_tag}*sx1.csv"))
        if not files:
            continue
        dfs = [pd.read_csv(f) for f in files]
        base = dfs[0].copy()
        if len(dfs) > 1:
            flux_stack = np.column_stack([
                np.interp(base["wavelength_angstrom"].values,
                          d["wavelength_angstrom"].values,
                          d["flux_erg_s_cm2_angstrom"].values) for d in dfs])
            err_stack = np.column_stack([
                np.interp(base["wavelength_angstrom"].values,
                          d["wavelength_angstrom"].values,
                          d["error"].values) for d in dfs])
            base["flux_erg_s_cm2_angstrom"] = np.nanmean(flux_stack, axis=1)
            # Combined error: 1/sqrt(N) of stacked per-pixel σ
            base["error"] = np.sqrt(np.nanmean(err_stack ** 2, axis=1)) / np.sqrt(len(dfs))
        sel = (base["wavelength_angstrom"] >= w_lo) & (base["wavelength_angstrom"] <= w_hi)
        parts.append(base.loc[sel, ["wavelength_angstrom", "flux_erg_s_cm2_angstrom", "error"]])
    if not parts:
        return None
    hst = pd.concat(parts, ignore_index=True).sort_values("wavelength_angstrom").reset_index(drop=True)
    hst = hst[hst["flux_erg_s_cm2_angstrom"] > 0]
    return (hst["wavelength_angstrom"].values,
            hst["flux_erg_s_cm2_angstrom"].values,
            hst["error"].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_csv", help="LUMINA lumina_spectrum.csv")
    ap.add_argument("snifs_csv", help="Snifs reference CSV")
    ap.add_argument("--hst-dir",
                    default="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/sn2011fe/hst_uv",
                    help="HST UV directory")
    ap.add_argument("--hst-tag", default=None,
                    help="HST phase tag, e.g. phase+03.7 (skip HST scoring if absent)")
    ap.add_argument("--fwhm", type=float, default=20000.0,
                    help="Baseline FWHM in km/s (default 20000)")
    ap.add_argument("--noise-win", type=int, default=11,
                    help="DER_SNR window size in pixels (Snifs only)")
    args = ap.parse_args()

    m = pd.read_csv(args.model_csv)
    mlam = m["wavelength_angstrom"].values
    mflu = m["flux"].values

    # --- Snifs (no error column → empirical noise) ---
    o = pd.read_csv(args.snifs_csv, comment="#")
    olam = o["wavelength_angstrom"].values
    oflu = o["flux_erg_s_cm2_angstrom"].values
    gO = band_int(olam, oflu, 4500, 5800)
    r, chi = rms_bn_nw(mlam, mflu, olam, oflu, gO,
                       obs_err=None, fwhm=args.fwhm,
                       wl_lo=3300., wl_hi=8000.,
                       noise_win=args.noise_win)
    print(f"  Snifs [3300,8000]   rms_bn = {r:.4f}   chi_bn = {chi:.3f}  (DER_SNR)")

    if args.hst_tag is None:
        return
    hst = stitch_hst(args.hst_dir, args.hst_tag)
    if hst is None:
        print(f"  ! no HST file for {args.hst_tag}")
        return
    hl, hf, he = hst
    gH = band_int(hl, hf, 4500, 5800)
    if gH <= 0:
        print(f"  ! HST green-band integral non-positive")
        return
    for lo, hi, lbl in [(3000., 8000., "[3000,8000]"),
                        (1700., 3000., "[1700,3000]")]:
        r, chi = rms_bn_nw(mlam, mflu, hl, hf, gH,
                           obs_err=he, fwhm=args.fwhm,
                           wl_lo=lo, wl_hi=hi,
                           noise_win=args.noise_win)
        print(f"  HST   {lbl}  rms_bn = {r:.4f}   chi_bn = {chi:.3f}  (per-pixel error)")


if __name__ == "__main__":
    main()
