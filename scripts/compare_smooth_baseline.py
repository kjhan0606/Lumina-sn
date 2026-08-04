#!/usr/bin/env python3
"""Smooth baseline continuum 후보 비교 — wide Gaussian + polynomial.

물리: photosphere baseline은 Planck-like로 천천히 변해야 함 → wiggle 없는 함수.

  (A)  Gauss FWHM = (40k/c)·λ   (이전 5b)
  (B)  Gauss FWHM = (60k/c)·λ
  (C)  Gauss FWHM = (80k/c)·λ
  (D)  Gauss FWHM = (120k/c)·λ
  (P)  log(flu) ~ Σ a_k log(λ)^k 다항식 fit (deg=5)
       — outlier reject 2 회 (>2σ residual 제외)
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf")
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
OUT  = ROOT / "figures/pcont_smooth_baseline.png"
C_KMS = 299792.458


def gauss(lam, flu, fwhm_kms):
    cont = np.zeros_like(flu); beta = fwhm_kms / C_KMS
    for i in range(len(lam)):
        sigma = beta * lam[i] / 2.3548
        win = 4.0 * sigma
        sel = (lam >= lam[i]-win) & (lam <= lam[i]+win)
        if sel.sum() < 2: cont[i] = flu[i]; continue
        w = np.exp(-0.5 * ((lam[sel] - lam[i]) / sigma)**2)
        cont[i] = np.sum(w * flu[sel]) / np.sum(w)
    return cont


def poly_iter(lam, flu, deg=5, n_iter=2, sigma_clip=2.0):
    """log(flu) ~ poly(log(λ)) with iterative outlier rejection (above-fit clipping for absorption)."""
    x = np.log(lam); y = np.log(np.maximum(flu, flu.max()*1e-6))
    keep = np.ones_like(y, dtype=bool)
    for _ in range(n_iter+1):
        c = np.polyfit(x[keep], y[keep], deg)
        ymod = np.polyval(c, x)
        res = y - ymod
        std = np.std(res[keep])
        # absorption pulls flu DOWN → res < 0; we want to clip those outliers
        keep = res > -sigma_clip * std
    return np.exp(np.polyval(c, x)), keep


def load(p):
    df = pd.read_csv(p); lam = df.iloc[:,0].values
    col = "flux" if "flux" in df.columns else df.columns[1]
    return lam, df[col].values


hlam, hflu = load(HST)
m = (hlam >= 1700) & (hlam <= 9000) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]

print("(A) Gauss FWHM=40k ...")
gA = gauss(hlam, hflu, 40000.0)
print("(B) Gauss FWHM=60k ...")
gB = gauss(hlam, hflu, 60000.0)
print("(C) Gauss FWHM=80k ...")
gC = gauss(hlam, hflu, 80000.0)
print("(D) Gauss FWHM=120k ...")
gD = gauss(hlam, hflu, 120000.0)
print("(P) Polynomial deg=5 in log-log, asymmetric clip (-2σ) ...")
pP, kept = poly_iter(hlam, hflu, deg=5, n_iter=3, sigma_clip=2.0)

# ---- Plot ----
fig = plt.figure(figsize=(16, 12))
gs  = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.0, 1.0],
                       hspace=0.32, wspace=0.20)

# (a) full overlay
ax = fig.add_subplot(gs[0, :])
ax.plot(hlam, hflu*1e14, "k-", lw=0.6, alpha=0.5, label="HST B-max")
ax.plot(hlam, gA*1e14, color="tab:blue",   lw=1.3, alpha=0.85, label="(A) Gauss FWHM=40k (이전 5b)")
ax.plot(hlam, gB*1e14, color="tab:purple", lw=1.5, alpha=0.95, label="(B) Gauss FWHM=60k")
ax.plot(hlam, gC*1e14, color="tab:red",    lw=1.7, alpha=0.95, label="(C) Gauss FWHM=80k")
ax.plot(hlam, gD*1e14, color="tab:orange", lw=1.9, alpha=0.95, label="(D) Gauss FWHM=120k (거의 trend만)")
ax.plot(hlam, pP*1e14, color="tab:green",  lw=2.0, alpha=0.95, ls="--",
        label="(P) deg-5 polynomial in log(λ), asymm clip (-2σ)")
ax.set_xlim(1700, 9000)
ax.set_ylim(0, (hflu*1e14).max() * 1.15)
ax.set_xlabel("λ (Å)"); ax.set_ylabel(r"Flux (10$^{-14}$ erg/s/cm²/Å)")
ax.set_title("HST SN 2011fe B-max — smooth baseline 후보 (wide Gaussian + polynomial)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# (b) Si II 6355
zooms = [
    ("(b) Si II 6355 (P-Cygni)",     5500, 7000),
    ("(c) Ca II H&K (강한 흡수)",     3500, 4200),
    ("(d) UV iron forest",           2200, 3200),
    ("(e) Ca II IR triplet",         7800, 9000),
]
for k, (title, lo, hi) in enumerate(zooms):
    axz = fig.add_subplot(gs[1 + k//2, k%2])
    sel = (hlam >= lo) & (hlam <= hi)
    axz.plot(hlam[sel], hflu[sel]*1e14, "k-", lw=1.0, alpha=0.85)
    axz.plot(hlam[sel], gA[sel]*1e14, color="tab:blue",   lw=1.3)
    axz.plot(hlam[sel], gB[sel]*1e14, color="tab:purple", lw=1.5)
    axz.plot(hlam[sel], gC[sel]*1e14, color="tab:red",    lw=1.7)
    axz.plot(hlam[sel], gD[sel]*1e14, color="tab:orange", lw=1.9)
    axz.plot(hlam[sel], pP[sel]*1e14, color="tab:green",  lw=2.0, ls="--")
    axz.set_xlim(lo, hi)
    axz.set_xlabel("λ (Å)"); axz.set_ylabel(r"Flux (10$^{-14}$)")
    axz.set_title(title, fontsize=10, fontweight="bold")
    axz.grid(alpha=0.3)

fig.suptitle("Smooth baseline continuum 후보 — physical photosphere는 Planck-like 매끈해야 함",
             fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.savefig(OUT, dpi=140); plt.close()
print(f"\nWrote {OUT}  ({OUT.stat().st_size//1024} KB)")
