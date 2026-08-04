#!/usr/bin/env python3
"""(A) FWHM=40k Gaussian baseline 을 HST + LUMINA champion에 적용 → normalized 비교.

LUMINA: PROD_L19_W2x010_W3x065wide_152761 (formal integral spectrum)
- σ_bf-on production floor, RMS≈0.0446 baseline
- Champion of optical-band 6-band log RMS metric
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
CHAMP = ROOT / "logs/PROD_L19_W2x010_W3x065wide_152761/lumina_spectrum_formal.csv"
OUT   = ROOT / "figures/champ_vs_hst_baseline_norm.png"
C_KMS = 299792.458

FWHM_KMS = 40000.0


def gauss(lam, flu, fwhm_kms=FWHM_KMS):
    """Gaussian-weighted mean with FWHM = (fwhm_kms/c)·λ at each pixel."""
    cont = np.zeros_like(flu); beta = fwhm_kms / C_KMS
    for i in range(len(lam)):
        sigma = beta * lam[i] / 2.3548
        win = 4.0 * sigma
        sel = (lam >= lam[i]-win) & (lam <= lam[i]+win)
        if sel.sum() < 2: cont[i] = flu[i]; continue
        w = np.exp(-0.5 * ((lam[sel] - lam[i]) / sigma)**2)
        cont[i] = np.sum(w * flu[sel]) / np.sum(w)
    return cont


def load(p):
    df = pd.read_csv(p); lam = df.iloc[:,0].values
    col = "flux" if "flux" in df.columns else df.columns[1]
    return lam, df[col].values


print("Loading HST...")
hlam, hflu = load(HST)
m = (hlam >= 1700) & (hlam <= 9000) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]

print(f"Loading LUMINA champion: {CHAMP.name}")
llam, lflu = load(CHAMP)
# Normalize champion to HST 4500-5800 band (same convention as 6-band RMS)
band_h = np.trapezoid(hflu[(hlam>=4500)&(hlam<=5800)], hlam[(hlam>=4500)&(hlam<=5800)])
band_l = np.trapezoid(lflu[(llam>=4500)&(llam<=5800)], llam[(llam>=4500)&(llam<=5800)])
lflu *= band_h / band_l

print(f"(A) Gaussian baseline FWHM = {FWHM_KMS/1e3:.0f}k km/s on both...")
hcont = gauss(hlam, hflu)
lcont = gauss(llam, lflu)
hnorm = hflu / hcont
lnorm = lflu / lcont

# Interpolate LUMINA onto HST grid for residual
lnorm_on_hst = np.interp(hlam, llam, lnorm)
resid = hnorm - lnorm_on_hst
rms_full = float(np.sqrt(np.mean(resid**2)))
band = (hlam >= 3000) & (hlam <= 8000)
rms_30_80 = float(np.sqrt(np.mean(resid[band]**2)))

print(f"  RMS(normalized resid) full   = {rms_full:.4f}")
print(f"  RMS(normalized resid) 3000-8000 Å = {rms_30_80:.4f}")

# ---- Plot ----
fig = plt.figure(figsize=(16, 14))
gs  = fig.add_gridspec(4, 1, height_ratios=[1.4, 1.4, 1.2, 0.7], hspace=0.30)

# (a) raw flux + baselines
ax = fig.add_subplot(gs[0])
ax.plot(hlam, hflu*1e14, "k-", lw=0.7, alpha=0.7, label="HST B-max")
ax.plot(llam, lflu*1e14, color="tab:red", lw=0.7, alpha=0.7,
        label="LUMINA champ 152761 (scaled to HST 4500–5800)")
ax.plot(hlam, hcont*1e14, color="tab:blue",  lw=2.0, alpha=0.95,
        label=f"HST baseline (Gauss FWHM={FWHM_KMS/1e3:.0f}k)")
ax.plot(llam, lcont*1e14, color="tab:orange", lw=2.0, alpha=0.95,
        label=f"LUMINA baseline (Gauss FWHM={FWHM_KMS/1e3:.0f}k)")
ax.set_xlim(1700, 9000)
ax.set_ylim(0, max((hflu*1e14).max(), (lflu*1e14).max()) * 1.15)
ax.set_xlabel("λ (Å)"); ax.set_ylabel(r"Flux (10$^{-14}$ erg/s/cm²/Å)")
ax.set_title("(a) Raw flux + (A) Gaussian baseline on both",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# (b) normalized: f / baseline
ax2 = fig.add_subplot(gs[1])
ax2.plot(hlam, hnorm, "k-", lw=0.9, alpha=0.85, label="HST / baseline")
ax2.plot(llam, lnorm, color="tab:red", lw=0.9, alpha=0.85,
         label="LUMINA / baseline (champ 152761)")
ax2.axhline(1.0, color="gray", ls=":", lw=0.7)
ax2.set_xlim(1700, 9000)
ax2.set_ylim(0.0, 1.6)
ax2.set_xlabel("λ (Å)"); ax2.set_ylabel("flux / baseline")
ax2.set_title(f"(b) Normalized — baseline 제거 후 line shape 비교 "
              f"(RMS resid 3000–8000 Å = {rms_30_80:.4f})",
              fontsize=11, fontweight="bold")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(alpha=0.3)

# (c) zoom 3000-7000 (key fitting band) normalized
ax3 = fig.add_subplot(gs[2])
sel = (hlam >= 3000) & (hlam <= 7000)
sel_l = (llam >= 3000) & (llam <= 7000)
ax3.plot(hlam[sel], hnorm[sel], "k-", lw=1.2, alpha=0.9, label="HST norm")
ax3.plot(llam[sel_l], lnorm[sel_l], color="tab:red", lw=1.2, alpha=0.9,
         label="LUMINA norm")
ax3.axhline(1.0, color="gray", ls=":", lw=0.7)
ax3.set_xlim(3000, 7000)
ax3.set_ylim(0.0, 1.5)
ax3.set_xlabel("λ (Å)"); ax3.set_ylabel("flux / baseline")
ax3.set_title("(c) Zoom — Optical fitting band [3000, 7000] Å normalized",
              fontsize=11, fontweight="bold")
ax3.legend(loc="upper right", fontsize=9)
ax3.grid(alpha=0.3)

# (d) residual (HST - LUMINA), normalized
ax4 = fig.add_subplot(gs[3])
ax4.plot(hlam, resid, color="tab:purple", lw=0.9, alpha=0.85,
         label=f"residual (HST − LUMINA), RMS = {rms_30_80:.4f} on [3000,8000]")
ax4.axhline(0, color="black", lw=0.5)
ax4.fill_between(hlam, -0.1, 0.1, color="gray", alpha=0.15,
                 label="±0.1 reference band")
ax4.set_xlim(1700, 9000)
ax4.set_ylim(-0.6, 0.6)
ax4.set_xlabel("λ (Å)"); ax4.set_ylabel("Δ(norm)")
ax4.set_title("(d) Normalized residual — HST − LUMINA (champ 152761)",
              fontsize=11, fontweight="bold")
ax4.legend(loc="upper right", fontsize=9)
ax4.grid(alpha=0.3)

fig.suptitle(f"LUMINA champion 152761 vs HST SN 2011fe B-max — "
             f"(A) Gaussian baseline FWHM = (40k/c)·λ both sides",
             fontsize=12, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT, dpi=140); plt.close()
print(f"\nWrote {OUT}  ({OUT.stat().st_size//1024} KB)")
