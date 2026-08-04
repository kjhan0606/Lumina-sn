#!/usr/bin/env python3
"""한 run의 raw 6-band / 5-band weighted / baseline-norm RMS를 계산.

Usage: python score_one_run.py <run_name> [<run_name> ...]
"""
import sys, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
C_KMS = 299792.458
FWHM_KMS = 40000.0

BANDS = [("UVbl",1700,2900),("UVtg",2900,3700),("CaK",3700,3950),
         ("fluo",3950,4500),("grn",4500,5800),("red",5800,7000)]
WEIGHTS = dict(UVbl=0.0, UVtg=0.5, CaK=0.7, fluo=1.0, grn=1.0, red=1.0)

def gauss(lam, flu, fwhm_kms=FWHM_KMS):
    cont = np.zeros_like(flu); beta = fwhm_kms / C_KMS
    for i in range(len(lam)):
        sigma = beta * lam[i] / 2.3548
        win = 4.0 * sigma
        sel = (lam >= lam[i]-win) & (lam <= lam[i]+win)
        if sel.sum() < 2: cont[i] = flu[i]; continue
        w = np.exp(-0.5*((lam[sel]-lam[i])/sigma)**2)
        cont[i] = np.sum(w*flu[sel])/np.sum(w)
    return cont

def band_int(lam, flu, lo, hi):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m])

# Load HST
df_h = pd.read_csv(HST)
hlam = df_h.iloc[:,0].values; hflu = df_h.iloc[:,1].values
m = (hlam >= 1700) & (hlam <= 9000) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]
hcont = gauss(hlam, hflu); hnorm = hflu / hcont
mask_bn = (hlam >= 3800) & (hlam <= 8000)

print(f"{'run':50s} {'rms6':>8s} {'rms5w':>8s} {'rms_bn':>8s}  | {'UVbl':>7s} {'UVtg':>7s} {'CaK':>7s} {'fluo':>7s} {'grn':>7s} {'red':>7s}")
print("-"*150)

for run in sys.argv[1:]:
    p = ROOT / f"logs/{run}/lumina_spectrum_formal.csv"
    if not p.exists():
        print(f"{run:50s}  MISSING")
        continue
    df = pd.read_csv(p)
    lam = df.iloc[:,0].values
    col = "flux" if "flux" in df.columns else df.columns[1]
    flu = df[col].values
    flu = np.where(np.isfinite(flu)&(flu>=0), flu, 0.0)
    norm = band_int(hlam, hflu, 4500, 5800) / band_int(lam, flu, 4500, 5800)
    flu = flu * norm

    # 6-band log10 RMS
    log_r = []
    for n, lo, hi in BANDS:
        num = band_int(lam, flu, lo, hi); den = band_int(hlam, hflu, lo, hi)
        if num>0 and den>0: log_r.append(np.log10(num/den))
        else: log_r.append(np.nan)
    LR = np.array(log_r)
    rms6 = float(np.sqrt(np.nanmean(LR**2)))
    W = np.array([WEIGHTS[b[0]] for b in BANDS])
    LR_clip = np.where(np.isnan(LR), 0.0, LR)
    rms5w = float(np.sqrt((W*LR_clip**2).sum() / W.sum()))

    # baseline-norm RMS [3800,8000] — single HST baseline as truth
    hcont_on_l = np.interp(lam, hlam, hcont)
    lnorm = flu / hcont_on_l
    lnorm_h = np.interp(hlam[mask_bn], lam, lnorm)
    rms_bn = float(np.sqrt(np.mean((hnorm[mask_bn]-lnorm_h)**2)))

    print(f"{run:50s} {rms6:8.4f} {rms5w:8.4f} {rms_bn:8.4f}  | "
          + " ".join(f"{x:+7.4f}" for x in LR))
