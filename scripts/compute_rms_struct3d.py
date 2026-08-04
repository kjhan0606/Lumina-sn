#!/usr/bin/env python3
"""Compute baseline-norm RMS (PRIMARY) + raw MARE (SECONDARY) on 3D
structural sweep cells (struct3d_<jobid>_*). Same metric defs as
scripts/compute_rms_logL_sweep.py — baseline-norm = RMS of (F/F_cont)
on [3000,8000] Å after Gauss FWHM=40k km/s smoothing on both LUMINA and
HST; raw MARE = mean |I − I_HST|/I_HST. Target baseline-norm ≤ 0.20.
"""
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
C_KMS = 299792.458
FWHM_KMS = 40000.0

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
    df = pd.read_csv(p); lam = df.iloc[:, 0].values
    col = "flux" if "flux" in df.columns else df.columns[1]
    return lam, df[col].values

def band_int(lam, flu, lo, hi):
    m = (lam >= lo) & (lam <= hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum() >= 2 else np.nan

def gauss_cont(lam, flu, fwhm_kms=FWHM_KMS):
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

ap = argparse.ArgumentParser()
ap.add_argument("--jobid", default="155420")
ap.add_argument("--prefix", default="struct3d",
                help="Dir prefix (e.g. 'struct3d' or 'struct3d_phys')")
args = ap.parse_args()

# Parse cell directory names: <prefix>_<jobid>_r<sign><X>p<Y>_n<X>p<Y>_v<NNNNN>
pat = re.compile(rf"{re.escape(args.prefix)}_\d+_r([mp]?)(\d+)p(\d+)_n(\d+)p(\d+)_v(\d+)")

cells = []
for d in sorted((ROOT / "logs").glob(f"{args.prefix}_{args.jobid}_*")):
    if not d.is_dir(): continue
    m = pat.match(d.name)
    if not m:
        continue
    sgn, rint, rfrac, nint, nfrac, vi = m.groups()
    ro = float(f"{rint}.{rfrac}") * (-1 if sgn == "m" else 1)
    de = float(f"{nint}.{nfrac}")
    vi_kms = int(vi)
    spec = d / "lumina_spectrum_formal.csv"
    if not spec.exists():
        continue
    cells.append((d, ro, de, vi_kms, spec))

if not cells:
    print(f"No landed cells found for {args.prefix}_{args.jobid}_*", file=sys.stderr)
    sys.exit(1)

print(f"Loading HST...")
hlam_raw, hflu_raw = load_csv(HST)
m_valid = ((hlam_raw >= LAM_LO) & (hlam_raw <= LAM_HI)
           & np.isfinite(hflu_raw) & (hflu_raw > 0))
hlam = hlam_raw[m_valid]; hflu = hflu_raw[m_valid]
hi_grn = band_int(hlam, hflu, NORM_LO, NORM_HI)
print(f"  HST valid bins: {len(hlam)}")
print(f"Computing HST Gauss FWHM=40k continuum (once)...")
hcont = gauss_cont(hlam, hflu)
hnorm = hflu / hcont
RMS_BN_LO, RMS_BN_HI = 3000, 8000
bn_band = (hlam >= RMS_BN_LO) & (hlam <= RMS_BN_HI)
print(f"\n{len(cells)} cells landed:\n")

# Header
print(f"{'ro':>5} {'de':>4} {'v_in':>6}  {'RMS_bn':>8} {'MARE':>7}  "
      + "  ".join(f"{b[0]:>5}" for b in BANDS))

rows = []
for d, ro, de, vi_kms, spec in cells:
    llam, lflu = load_csv(spec)
    li_grn = band_int(llam, lflu, NORM_LO, NORM_HI)
    if not np.isfinite(li_grn) or li_grn <= 0:
        print(f"{ro:>+5.2f} {de:>4.1f} {vi_kms:>6d}  bad green-band")
        continue
    norm = hi_grn / li_grn
    l_on_h = np.interp(hlam, llam, lflu, left=np.nan, right=np.nan) * norm

    # Raw MARE (secondary)
    rel = np.abs(l_on_h - hflu) / hflu
    finite = np.isfinite(rel)
    mare = float(np.mean(rel[finite]))
    band_mare = []
    for _, lo, hi in BANDS:
        mb = (hlam >= lo) & (hlam <= hi) & finite
        band_mare.append(float(np.mean(rel[mb])) if mb.any() else np.nan)

    # Baseline-norm RMS (primary)
    l_scaled = lflu * norm
    lcont = gauss_cont(llam, l_scaled)
    lnorm = l_scaled / lcont
    lnorm_on_h = np.interp(hlam, llam, lnorm, left=np.nan, right=np.nan)
    resid = hnorm - lnorm_on_h
    bn_mask = bn_band & np.isfinite(resid)
    rms_bn = float(np.sqrt(np.mean(resid[bn_mask]**2))) if bn_mask.any() else np.nan
    bn_full = np.isfinite(resid)
    rms_bn_full = float(np.sqrt(np.mean(resid[bn_full]**2))) if bn_full.any() else np.nan

    print(f"{ro:>+5.2f} {de:>4.1f} {vi_kms:>6d}  {rms_bn:>8.4f} {mare:>7.4f}  "
          + "  ".join(f"{x:>5.2f}" for x in band_mare))
    rows.append(dict(rho_offset=ro, density_exp=de, v_inner_kms=vi_kms,
                     RMS_bn=rms_bn, RMS_bn_full=rms_bn_full, MARE=mare,
                     **{b[0]: m for b, m in zip(BANDS, band_mare)}))

if rows:
    df = pd.DataFrame(rows).sort_values("RMS_bn").reset_index(drop=True)
    out = ROOT / "logs" / f"{args.prefix}_{args.jobid}_rms.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    b = df.iloc[0]
    print(f"\nBest by baseline-norm RMS [{RMS_BN_LO},{RMS_BN_HI}]Å:")
    print(f"  rho_off={b['rho_offset']:+.2f}  exp={b['density_exp']:.1f}  "
          f"v_in={int(b['v_inner_kms'])}  RMS_bn={b['RMS_bn']:.4f}  MARE={b['MARE']:.4f}")
    print(f"\nTARDIS reference (same metric) ≈ 0.164;  target ≤ 0.20")
