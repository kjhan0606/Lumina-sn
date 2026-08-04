#!/usr/bin/env python3
"""All-time lowest RMS run vs HST B-max — full spectral comparison.

Lowest raw 6-band log RMS: sigbf_W2x015_W3x065wide_h100_152629 (rms6=0.0183).
- σ_bf ON, W2=0.15, W3=0.65 wide
- v_inner=12000 (default, not v11500k)
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf")
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path

ROOT  = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST   = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
CHAMP = ROOT / "logs/sigbf_W2x015_W3x065wide_h100_152629/lumina_spectrum_formal.csv"
OUT   = ROOT / "figures/champion_152629_vs_hst.png"
C_KMS = 299792.458
FWHM_KMS = 40000.0

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

def load(p):
    df = pd.read_csv(p); lam = df.iloc[:,0].values
    col = "flux" if "flux" in df.columns else df.columns[1]
    return lam, df[col].values

def band_int(lam, flu, lo, hi):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m])

print("Loading HST...")
hlam, hflu = load(HST)
m = (hlam >= 1700) & (hlam <= 9000) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]

print(f"Loading champion: {CHAMP.parent.name}")
llam, lflu = load(CHAMP)
# Normalize to HST [4500, 5800]
norm = band_int(hlam, hflu, 4500, 5800) / band_int(llam, lflu, 4500, 5800)
lflu = lflu * norm
print(f"  norm = {norm:.4e}")

hcont = gauss(hlam, hflu); hnorm = hflu / hcont
lcont = gauss(llam, lflu); lnorm = lflu / lcont

# Per-band log RMS
BANDS = [("UVbl",1700,2900),("UVtg",2900,3700),("CaK",3700,3950),
         ("fluo",3950,4500),("grn",4500,5800),("red",5800,7000)]
log_r = []
for n, lo, hi in BANDS:
    num = band_int(llam, lflu, lo, hi)
    den = band_int(hlam, hflu, lo, hi)
    log_r.append(np.log10(num/den) if (num>0 and den>0) else np.nan)
rms6 = np.sqrt(np.nanmean(np.array(log_r)**2))
print(f"\nrms6 = {rms6:.4f}")
for (n, lo, hi), lr in zip(BANDS, log_r):
    print(f"  {n:5s} [{lo:5d},{hi:5d}]: log ratio = {lr:+.4f}  ({100*(10**lr - 1):+.2f}%)")

# Baseline-norm RMS [3800, 8000]
mask_bn = (hlam >= 3800) & (hlam <= 8000)
lnorm_h = np.interp(hlam[mask_bn], llam, lnorm)
rms_bn = np.sqrt(np.mean((hnorm[mask_bn] - lnorm_h)**2))
print(f"\nrms_bn [3800,8000] = {rms_bn:.4f}")

# ---- Figure ----
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 2, height_ratios=[1.4, 1.0, 1.0, 0.9],
                     hspace=0.35, wspace=0.18)

# (a) Full raw flux
ax = fig.add_subplot(gs[0, :])
ax.plot(hlam, hflu*1e14, "k-", lw=1.0, alpha=0.85, label="HST B-max")
ax.plot(llam, lflu*1e14, color="tab:red", lw=0.9, alpha=0.85,
        label=f"LUMINA champion 152629 (sigbf σ_bf, scaled to [4500,5800])")
ax.plot(hlam, hcont*1e14, color="tab:blue", lw=1.6, alpha=0.7,
        label=f"HST baseline (Gauss FWHM=40k)")
ax.plot(llam, lcont*1e14, color="tab:orange", lw=1.6, alpha=0.7,
        label=f"LUMINA baseline")
ax.set_xlim(1700, 9000)
ax.set_ylim(0, max((hflu*1e14).max(), (lflu*1e14).max()) * 1.15)
ax.set_xlabel("λ (Å)"); ax.set_ylabel(r"Flux (10$^{-14}$ erg/s/cm²/Å)")
ax.set_title(f"(a) All-time lowest-RMS LUMINA vs HST SN 2011fe B-max  "
             f"— rms6={rms6:.4f}, rms_bn={rms_bn:.4f}",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# (b)-(g) zoom panels
zooms = [
    ("(b) UVbl [1700,2900]: Fe forest blanketing",   1700, 2900),
    ("(c) UVtg/CaK [2900,3950]",                      2900, 3950),
    ("(d) fluo [3950,4500]: Fe complex",              3950, 4500),
    ("(e) grn [4500,5800]: norm anchor + Mg/Fe",      4500, 5800),
    ("(f) red [5800,7000]: Si II 6355",               5800, 7000),
    ("(g) Ca II IR triplet [7800,9000]",              7800, 9000),
]
for k, (title, lo, hi) in enumerate(zooms):
    row = 1 + k // 2
    col = k % 2
    ax = fig.add_subplot(gs[row, col])
    sel_h = (hlam>=lo)&(hlam<=hi)
    sel_l = (llam>=lo)&(llam<=hi)
    ax.plot(hlam[sel_h], hflu[sel_h]*1e14, "k-", lw=1.4, alpha=0.9, label="HST")
    ax.plot(llam[sel_l], lflu[sel_l]*1e14, color="tab:red", lw=1.2, alpha=0.85,
            label="LUMINA 152629")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("λ (Å)"); ax.set_ylabel(r"Flux (10$^{-14}$)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("LUMINA-SN champion sigbf_W2x015_W3x065wide_152629  vs  HST SN 2011fe B-max  "
             "— raw 6-band log RMS = 0.0183 (all-time best)",
             fontsize=12, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT, dpi=140); plt.close()
print(f"\nWrote {OUT}  ({OUT.stat().st_size//1024} KB)")
