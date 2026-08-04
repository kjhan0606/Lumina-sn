#!/usr/bin/env python3
"""#220 Per-trough zoom-in diagnostic on Q1b r03 champion (RMS_bn 0.2375 floor).

For each diagnostic feature of SN Ia near B-max, measure trough
(position v, depth, FWHM) in HST stitched obs, TARDIS reference,
and LUMINA Q1b r03 champion. Identify which features are responsible
for the line-shape correlation floor.

Outputs:
- logs/trough_zoom_metrics.csv  : per-feature numeric table
- figures/trough_zooms_q1br03.png : 5x2 grid of zoom panels
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST    = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TARDIS = ROOT/"data/sn2011fe/tardis_spectrum.csv"
LUMINA = ROOT/"logs/ddc15Q1b_156281_ddc15Q1b_Q1b_r03/lumina_spectrum_formal.csv"
OUT_CSV = ROOT/"logs/trough_zoom_metrics.csv"
OUT_PNG = ROOT/"figures/trough_zooms_q1br03.png"

C_KMS = 299792.458
GREEN_LO, GREEN_HI = 4500, 5800  # normalization band

# 2011fe near B-max trough positions are ~9000-15000 km/s blueshift.
# Per-feature window targets the trough region (NOT the lab line center) so argmin
# captures the actual P-Cygni absorption rather than wandering into neighbours.
# (name, lambda_lab Å, window_lo, window_hi, primary species)
FEATURES = [
    ("Ca II H&K",   3945.0, 3700, 3900, "Ca II"),
    ("Si II 4130",  4130.0, 3990, 4130, "Si II"),
    ("Mg II 4481",  4481.0, 4280, 4430, "Mg II"),
    ("Fe II m42",   4924.0, 4700, 4900, "Fe II 4924 (m42 blue)"),
    ("Fe II 5018",  5018.0, 4830, 4990, "Fe II 5018"),
    ("S II W blue", 5454.0, 5200, 5400, "S II 5454"),
    ("S II W red",  5640.0, 5400, 5620, "S II 5640"),
    ("Si II 5972",  5972.0, 5750, 5950, "Si II"),
    ("Si II 6355",  6355.0, 6050, 6300, "Si II"),
    ("O I 7773",    7773.0, 7400, 7750, "O I"),
    ("Ca II IR",    8542.0, 8000, 8420, "Ca II IR triplet"),
]

def load(path, want_err=False):
    d = pd.read_csv(path)
    lam = d.iloc[:, 0].values.astype(float)
    flu = d.iloc[:, 1].values.astype(float)
    return lam, flu

def band_int(lam, flu, lo, hi):
    m = (lam >= lo) & (lam <= hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum() >= 2 else np.nan

def trough_metrics(lam, flu, lab, lo, hi):
    """Return (v_kms, depth, fwhm_kms, lam_min, f_min, f_cont)."""
    m = (lam >= lo) & (lam <= hi)
    if m.sum() < 5:
        return (np.nan,)*6
    sl, sf = lam[m], flu[m]
    imin = int(np.argmin(sf))
    lam_min = float(sl[imin])
    f_min   = float(sf[imin])
    # Local pseudo-continuum: max of medians at the two endpoints (10% width).
    n = len(sl)
    edge = max(3, n//20)
    f_left  = float(np.median(sf[:edge]))
    f_right = float(np.median(sf[-edge:]))
    f_cont  = max(f_left, f_right)
    depth   = 1.0 - f_min/f_cont if f_cont > 0 else np.nan
    v_kms   = (lab - lam_min)/lab * C_KMS  # positive = blueshift
    # FWHM: find where flux crosses (f_min+f_cont)/2 on either side of imin.
    f_half = 0.5*(f_min + f_cont)
    # left
    j = imin
    while j > 0 and sf[j] < f_half:
        j -= 1
    lam_left = sl[j] if j >= 0 else sl[0]
    # right
    k = imin
    while k < n-1 and sf[k] < f_half:
        k += 1
    lam_right = sl[k] if k < n else sl[-1]
    fwhm_aa  = lam_right - lam_left
    fwhm_kms = fwhm_aa / lab * C_KMS
    return v_kms, depth, fwhm_kms, lam_min, f_min, f_cont

def main():
    hlam, hflu = load(HST)
    tlam, tflu = load(TARDIS)
    llam, lflu = load(LUMINA)

    # Normalize each to HST green-band integral.
    gH = band_int(hlam, hflu, GREEN_LO, GREEN_HI)
    gT = band_int(tlam, tflu, GREEN_LO, GREEN_HI)
    gL = band_int(llam, lflu, GREEN_LO, GREEN_HI)
    tflu_s = tflu * (gH/gT)
    lflu_s = lflu * (gH/gL)

    # Resample TARDIS, LUMINA onto HST grid for residual integration.
    def regrid(src_lam, src_flu, dst_lam):
        return np.interp(dst_lam, src_lam, src_flu, left=np.nan, right=np.nan)
    tflu_h = regrid(tlam, tflu_s, hlam)
    lflu_h = regrid(llam, lflu_s, hlam)

    rows = []
    for name, lab, lo, hi, sp in FEATURES:
        H = trough_metrics(hlam, hflu,   lab, lo, hi)
        T = trough_metrics(tlam, tflu_s, lab, lo, hi)
        L = trough_metrics(llam, lflu_s, lab, lo, hi)
        # In-window L1 residual (LUMINA-HST) normalized to HST mean flux.
        m = (hlam >= lo) & (hlam <= hi) & np.isfinite(lflu_h) & np.isfinite(tflu_h)
        if m.sum() >= 5:
            hmean = float(np.nanmean(hflu[m]))
            res_LH = float(np.nanmean(np.abs(lflu_h[m] - hflu[m]))) / hmean if hmean > 0 else np.nan
            res_TH = float(np.nanmean(np.abs(tflu_h[m] - hflu[m]))) / hmean if hmean > 0 else np.nan
        else:
            res_LH = res_TH = np.nan
        rows.append({
            "feature":  name,
            "species":  sp,
            "lab_AA":   lab,
            "H_v":      H[0], "H_d": H[1], "H_w": H[2],
            "T_v":      T[0], "T_d": T[1], "T_w": T[2],
            "L_v":      L[0], "L_d": L[1], "L_w": L[2],
            "dv_LH":    L[0]-H[0],
            "dd_LH":    L[1]-H[1],
            "dw_LH":    L[2]-H[2],
            "dv_TH":    T[0]-H[0],
            "dd_TH":    T[1]-H[1],
            "dw_TH":    T[2]-H[2],
            "res_LH":   res_LH,
            "res_TH":   res_TH,
        })
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.4f")

    # Console table.
    print("="*120)
    print(f"#220 Per-trough zoom-in diagnostic on Q1b r03 (job 156281, RMS_bn=0.2375)")
    print(f"Normalized to HST green-band [{GREEN_LO},{GREEN_HI}]Å integral.")
    print("="*120)
    hdr = (f"{'feature':<13}{'species':<28}{'lab':>6} | "
           f"{'H_v':>7}{'H_d':>6}{'H_w':>7} | "
           f"{'T_v':>7}{'T_d':>6}{'T_w':>7} | "
           f"{'L_v':>7}{'L_d':>6}{'L_w':>7} | "
           f"{'dv_LH':>7}{'dd_LH':>7}{'dw_LH':>7} | "
           f"{'res_LH':>7}{'res_TH':>7}")
    print(hdr)
    print("-"*len(hdr))
    for r in rows:
        print(f"{r['feature']:<13}{r['species']:<28}{r['lab_AA']:>6.0f} | "
              f"{r['H_v']:>7.0f}{r['H_d']:>6.2f}{r['H_w']:>7.0f} | "
              f"{r['T_v']:>7.0f}{r['T_d']:>6.2f}{r['T_w']:>7.0f} | "
              f"{r['L_v']:>7.0f}{r['L_d']:>6.2f}{r['L_w']:>7.0f} | "
              f"{r['dv_LH']:>+7.0f}{r['dd_LH']:>+7.2f}{r['dw_LH']:>+7.0f} | "
              f"{r['res_LH']:>7.3f}{r['res_TH']:>7.3f}")
    print()
    print("Rank by LUMINA-HST residual (mean |ΔF|/<F_HST>):")
    rank = sorted(rows, key=lambda r: -r['res_LH'] if np.isfinite(r['res_LH']) else 0)
    for r in rank:
        print(f"  {r['feature']:<13} res_LH={r['res_LH']:.3f}   res_TH={r['res_TH']:.3f}   "
              f"Δd={r['dd_LH']:+.2f}  Δv={r['dv_LH']:+.0f}  Δw={r['dw_LH']:+.0f}")
    print()
    print("v in km/s (+ = blueshift), d = trough depth (0-1), w = FWHM in km/s")
    print(f"Wrote: {OUT_CSV}")

    # Plot grid.
    n = len(FEATURES)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(15, 3.0*nrow))
    axs = np.atleast_2d(axs).ravel()
    for i, (name, lab, lo, hi, sp) in enumerate(FEATURES):
        ax = axs[i]
        for cur_lam, cur_flu, color, label in [
            (hlam, hflu,   "k", "HST"),
            (tlam, tflu_s, "tab:blue",   "TARDIS"),
            (llam, lflu_s, "tab:red",    "LUMINA Q1b r03"),
        ]:
            m = (cur_lam >= lo) & (cur_lam <= hi)
            ax.plot(cur_lam[m], cur_flu[m], color=color, lw=1.1, label=label, alpha=0.85)
        ax.axvline(lab, color="0.5", ls=":", lw=0.7)
        ax.set_title(f"{name}  ({sp})  λ_lab={lab:.0f}Å", fontsize=9)
        ax.set_xlim(lo, hi)
        ax.tick_params(axis='both', labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="best")
    for j in range(n, len(axs)):
        axs[j].axis("off")
    fig.suptitle("Q1b r03 champion (job 156281) vs HST stitched vs TARDIS — per-trough zoom",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"Wrote: {OUT_PNG}")

if __name__ == "__main__":
    main()
