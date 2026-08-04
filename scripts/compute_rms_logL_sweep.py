#!/usr/bin/env python3
"""Compute baseline-norm RMS (PRIMARY) + raw MARE (SECONDARY) on the 10
log_L sweep cells (0.25–2.0× L). Per project_rms_metric_ceiling.md, raw
MARE is intrinsically capped at ~40% (TARDIS = 0.397), so baseline-norm
RMS — RMS of (F/F_cont)_LUMINA − (F/F_cont)_HST after Gauss FWHM=40k km/s
continuum removal — is the right line-shape metric. Target ≤ 0.20.

LUMINA→HST scaling via green-band integral [4500, 5800] Å (shape match,
same convention as scripts/champ_vs_hst_baseline_norm.py).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
C_KMS = 299792.458
FWHM_KMS = 40000.0

SWEEPS = [
    # (label, scale, dir)
    ("d060", 0.25, "logL_deep_155379_d060"),
    ("d050", 0.32, "logL_deep_155379_d050"),
    ("d040", 0.40, "logL_deep_155379_d040"),
    ("d030", 0.50, "logL_deep_155379_d030"),
    ("d020", 0.63, "logL_deep_155379_d020"),
    ("m030", 0.50, "logL_sweep_155368_m030"),  # duplicate scale, different binary path
    ("m015", 0.71, "logL_sweep_155368_m015"),
    ("base", 1.00, "logL_sweep_155368_base"),
    ("p015", 1.41, "logL_sweep_155368_p015"),
    ("p030", 2.00, "logL_sweep_155374_p030"),
]

BANDS = [
    ("UVbl", 1700, 2900),
    ("UVtg", 2900, 3700),
    ("CaK",  3700, 3950),
    ("fluo", 3950, 4500),
    ("grn",  4500, 5800),
    ("red",  5800, 7000),
]
LAM_LO, LAM_HI = 1700, 9000
NORM_LO, NORM_HI = 4500, 5800

def load_csv(p):
    df = pd.read_csv(p)
    lam = df.iloc[:, 0].values
    col = "flux" if "flux" in df.columns else df.columns[1]
    return lam, df[col].values

def band_int(lam, flu, lo, hi):
    m = (lam >= lo) & (lam <= hi)
    if m.sum() < 2:
        return np.nan
    return np.trapezoid(flu[m], lam[m])

def gauss_cont(lam, flu, fwhm_kms=FWHM_KMS):
    """Per-pixel Gaussian-weighted mean (FWHM = (fwhm_kms/c)·λ)."""
    cont = np.zeros_like(flu, dtype=float)
    beta = fwhm_kms / C_KMS
    for i in range(len(lam)):
        sigma = beta * lam[i] / 2.3548
        win = 4.0 * sigma
        sel = (lam >= lam[i]-win) & (lam <= lam[i]+win)
        if sel.sum() < 2:
            cont[i] = flu[i]; continue
        w = np.exp(-0.5 * ((lam[sel] - lam[i]) / sigma)**2)
        cont[i] = np.sum(w * flu[sel]) / np.sum(w)
    return cont

print(f"Loading HST: {HST.name}")
hlam_raw, hflu_raw = load_csv(HST)
m_valid = (hlam_raw >= LAM_LO) & (hlam_raw <= LAM_HI) & np.isfinite(hflu_raw) & (hflu_raw > 0)
hlam = hlam_raw[m_valid]
hflu = hflu_raw[m_valid]
print(f"  HST valid bins (I_HST > 0, finite, in [{LAM_LO},{LAM_HI}]): {len(hlam)}")

print("Computing HST Gauss FWHM=40k continuum (once)...")
hcont = gauss_cont(hlam, hflu)
hnorm = hflu / hcont
# Band masks for normalized RMS report (use 3000-8000 Å as primary; full as secondary)
RMS_BN_LO, RMS_BN_HI = 3000, 8000
bn_band = (hlam >= RMS_BN_LO) & (hlam <= RMS_BN_HI)
print()

# Header
print(f"{'label':<6} {'scale':>6} {'log_L':>7} {'norm':>10} "
      f"{'RMS_bn':>8} {'MARE':>7} {'N':>6}  "
      + "  ".join(f"{b[0]:>5}" for b in BANDS))

rows = []
for label, scale, subdir in SWEEPS:
    spec_path = ROOT / "logs" / subdir / "lumina_spectrum_formal.csv"
    if not spec_path.exists():
        print(f"{label:<6} MISSING")
        continue
    llam, lflu = load_csv(spec_path)

    li_grn = band_int(llam, lflu, NORM_LO, NORM_HI)
    hi_grn = band_int(hlam, hflu, NORM_LO, NORM_HI)
    if not np.isfinite(li_grn) or li_grn <= 0:
        print(f"{label:<6} bad green-band integral, skip")
        continue
    norm = hi_grn / li_grn

    # Scale LUMINA (raw) to HST and interpolate onto HST grid for MARE
    l_on_h = np.interp(hlam, llam, lflu, left=np.nan, right=np.nan) * norm

    # Raw MARE (secondary)
    rel = np.abs(l_on_h - hflu) / hflu
    finite = np.isfinite(rel)
    mare = float(np.mean(rel[finite]))
    n_valid = int(finite.sum())

    band_mare = []
    for _, lo, hi in BANDS:
        mb = (hlam >= lo) & (hlam <= hi) & finite
        band_mare.append(float(np.mean(rel[mb])) if mb.any() else np.nan)

    # Baseline-norm RMS (primary) — Gauss FWHM=40k on the scaled LUMINA
    l_scaled = lflu * norm
    lcont = gauss_cont(llam, l_scaled)
    lnorm = l_scaled / lcont
    lnorm_on_h = np.interp(hlam, llam, lnorm, left=np.nan, right=np.nan)
    resid = hnorm - lnorm_on_h
    bn_mask = bn_band & np.isfinite(resid)
    rms_bn = float(np.sqrt(np.mean(resid[bn_mask]**2))) if bn_mask.any() else np.nan
    bn_full = np.isfinite(resid)
    rms_bn_full = float(np.sqrt(np.mean(resid[bn_full]**2))) if bn_full.any() else np.nan

    log_off = np.log10(scale)
    print(f"{label:<6} {scale:>6.2f} {log_off:>+7.3f} {norm:>10.3e} "
          f"{rms_bn:>8.4f} {mare:>7.4f} {n_valid:>6d}  "
          + "  ".join(f"{x:>5.2f}" for x in band_mare))
    rows.append(dict(label=label, scale=scale, log_L_offset=log_off,
                     norm=norm, RMS_bn=rms_bn, RMS_bn_full=rms_bn_full,
                     MARE=mare, N_valid=n_valid,
                     **{b[0]: m for b, m in zip(BANDS, band_mare)}))

if rows:
    df_out = pd.DataFrame(rows).sort_values("RMS_bn").reset_index(drop=True)
    out = ROOT / "logs" / "logL_sweep_rms.csv"
    df_out.to_csv(out, index=False)
    print()
    print(f"Wrote {out}")
    best = df_out.iloc[0]
    print(f"\nBest by baseline-norm RMS [{RMS_BN_LO},{RMS_BN_HI}]Å:")
    print(f"  {best['label']} scale={best['scale']:.2f} (log_L={best['log_L_offset']:+.3f})  "
          f"RMS_bn={best['RMS_bn']:.4f}  MARE={best['MARE']:.4f}")
    print(f"\nTARDIS reference (same metric, baseline-norm 3000-8000 Å) ≈ 0.164")
    print(f"Target ≤ 0.20")
