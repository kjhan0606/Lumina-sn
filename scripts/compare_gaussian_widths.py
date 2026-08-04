#!/usr/bin/env python3
"""(2) narrow Gaussian + (3) wide median → (5) wide Gaussian (대칭+매끈) 접합점 탐색.

세 너비의 Gaussian smoothing + 비교용 wide median:
 (2)  FWHM = (v_2/c)·λ,  v_2 = 20k km/s   — 좁음 (≈ ±10k Doppler half)
 (5a) FWHM = (v_a/c)·λ,  v_a = 30k km/s   — 중간 (P-Cygni 흡수 폭)
 (5b) FWHM = (v_b/c)·λ,  v_b = 40k km/s   — 넓음 (전체 P-Cygni baseline 포괄)
 (3)  median, full window = (30k/c)·λ      — 동일 넓이 비교 reference
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
OUT  = ROOT / "figures/pcont_gaussian_width_scan.png"
C_KMS = 299792.458


def gauss_smooth(lam, flu, fwhm_kms):
    """Gaussian-weighted mean with FWHM = (fwhm_kms/c)·λ at each pixel."""
    cont = np.zeros_like(flu)
    beta = fwhm_kms / C_KMS
    for i in range(len(lam)):
        sigma = beta * lam[i] / 2.3548
        win   = 4.0 * sigma
        sel = (lam >= lam[i]-win) & (lam <= lam[i]+win)
        if sel.sum() < 2: cont[i] = flu[i]; continue
        w = np.exp(-0.5 * ((lam[sel] - lam[i]) / sigma)**2)
        cont[i] = np.sum(w * flu[sel]) / np.sum(w)
    return cont


def wide_median(lam, flu, fwhm_kms=30000.0):
    cont = np.zeros_like(flu); beta = fwhm_kms / C_KMS
    for i in range(len(lam)):
        win = beta * lam[i]
        sel = (lam >= lam[i]-win/2) & (lam <= lam[i]+win/2)
        if sel.sum() < 3: cont[i] = flu[i]; continue
        cont[i] = np.median(flu[sel])
    return cont


def load(p):
    df = pd.read_csv(p); lam = df.iloc[:,0].values
    col = "flux" if "flux" in df.columns else df.columns[1]
    return lam, df[col].values


hlam, hflu = load(HST)
m = (hlam >= 1700) & (hlam <= 9000) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]

print("(2)  Gauss FWHM=20k km/s ...")
g20 = gauss_smooth(hlam, hflu, 20000.0)
print("(5a) Gauss FWHM=30k km/s ...")
g30 = gauss_smooth(hlam, hflu, 30000.0)
print("(5b) Gauss FWHM=40k km/s ...")
g50 = gauss_smooth(hlam, hflu, 40000.0)
print("(3)  Median, win=30k ...")
m30 = wide_median(hlam, hflu, 30000.0)

# ---- Plot ----
fig = plt.figure(figsize=(16, 12))
gs  = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.0, 1.0],
                       hspace=0.32, wspace=0.20)

# (a) overall overlay
ax = fig.add_subplot(gs[0, :])
ax.plot(hlam, hflu*1e14, "k-", lw=0.7, alpha=0.55, label="HST B-max")
ax.plot(hlam, m30*1e14,  color="tab:green",  ls="-",  lw=1.2, alpha=0.85,
        label="(3) wide median, v=30k  (jittery)")
ax.plot(hlam, g20*1e14,  color="tab:blue",   ls="-",  lw=1.5, alpha=0.95,
        label="(2) Gauss FWHM = (20k/c)·λ  (≈ trough only)")
ax.plot(hlam, g30*1e14,  color="tab:purple", ls="-",  lw=1.8, alpha=0.95,
        label="(5a) Gauss FWHM = (30k/c)·λ  (대칭+매끈, 추천)")
ax.plot(hlam, g50*1e14,  color="tab:red",    ls="--", lw=1.6, alpha=0.9,
        label="(5b) Gauss FWHM = (40k/c)·λ  (전체 P-Cygni 포괄)")
ax.set_xlim(1700, 9000)
ax.set_ylim(0, (hflu*1e14).max() * 1.15)
ax.set_xlabel("λ (Å)"); ax.set_ylabel(r"Flux (10$^{-14}$ erg/s/cm²/Å)")
ax.set_title("HST SN 2011fe B-max — Gaussian smoothing 너비 scan + wide median 비교",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

# zoom panels
zooms = [
    ("(b) Si II 6355 (P-Cygni emission+absorption)", 5500, 7000),
    ("(c) Ca II H&K (강한 흡수)",                      3500, 4200),
    ("(d) UV iron forest (거의 모두 흡수)",            2200, 3200),
    ("(e) Ca II IR triplet (다중 trough)",            7800, 9000),
]
for k, (title, lo, hi) in enumerate(zooms):
    axz = fig.add_subplot(gs[1 + k//2, k%2])
    sel = (hlam >= lo) & (hlam <= hi)
    axz.plot(hlam[sel], hflu[sel]*1e14, "k-", lw=1.0, alpha=0.85, label="HST")
    axz.plot(hlam[sel], m30[sel]*1e14,  color="tab:green",  ls="-",  lw=1.0, alpha=0.85)
    axz.plot(hlam[sel], g20[sel]*1e14,  color="tab:blue",   ls="-",  lw=1.5)
    axz.plot(hlam[sel], g30[sel]*1e14,  color="tab:purple", ls="-",  lw=1.9)
    axz.plot(hlam[sel], g50[sel]*1e14,  color="tab:red",    ls="--", lw=1.5)
    axz.set_xlim(lo, hi)
    axz.set_xlabel("λ (Å)"); axz.set_ylabel(r"Flux (10$^{-14}$)")
    axz.set_title(title, fontsize=10, fontweight="bold")
    axz.grid(alpha=0.3)

fig.suptitle("(2) ↔ (3) 접합: Wide-window Gaussian-weighted mean — "
             "median의 jitter 제거 + 대칭 P-Cygni 평균 회복",
             fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.965])
plt.savefig(OUT, dpi=140); plt.close()
print(f"\nWrote {OUT}  ({OUT.stat().st_size//1024} KB)")
