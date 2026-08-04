#!/usr/bin/env python3
"""파장-비례 (Doppler) smoothing 시연.

window(λ) = scale × (v_exp/c) × λ
- v_exp = 10⁴ km/s, scale = 2
- 3000 Å → 200 Å,  6000 Å → 400 Å,  9000 Å → 600 Å

비교: 고정 win=500 Å vs 파장 의존 win
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/home/kjhan/.fonts/NotoSansCJKkr-Regular.otf")
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path
from scipy.signal import savgol_filter

ROOT  = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST   = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
CHAMP = ROOT / "logs/PROD_L19_W2x010_W3x065wide_152761/lumina_spectrum_formal.csv"
OUT   = ROOT / "figures/pcont_velocity_smoothing.png"
C_KMS = 299792.458
V_SMOOTH    = 10000.0
PCONT_SCALE = 2.0


def pcont_fixed(lam, flu, fwhm=500.0):
    """Fixed-FWHM Gaussian smoothing."""
    cont  = np.zeros_like(flu)
    sigma = fwhm / 2.3548
    win   = 4.0 * sigma
    for i in range(len(lam)):
        sel = (lam >= lam[i]-win) & (lam <= lam[i]+win)
        if sel.sum() < 2: cont[i] = flu[i]; continue
        w = np.exp(-0.5 * ((lam[sel] - lam[i]) / sigma)**2)
        cont[i] = np.sum(w * flu[sel]) / np.sum(w)
    return cont


def pcont_velocity(lam, flu, v=V_SMOOTH, scale=PCONT_SCALE):
    """Wavelength-dependent Gaussian smoothing.  FWHM(λ)=scale·(v/c)·λ."""
    cont = np.zeros_like(flu)
    beta = v / C_KMS
    for i in range(len(lam)):
        sigma = scale * beta * lam[i] / 2.3548
        win   = 4.0 * sigma
        sel = (lam >= lam[i]-win) & (lam <= lam[i]+win)
        if sel.sum() < 2: cont[i] = flu[i]; continue
        w = np.exp(-0.5 * ((lam[sel] - lam[i]) / sigma)**2)
        cont[i] = np.sum(w * flu[sel]) / np.sum(w)
    return cont


def load(p):
    df = pd.read_csv(p)
    lam = df.iloc[:,0].values
    col = "flux" if "flux" in df.columns else df.columns[1]
    return lam, df[col].values


hlam, hflu = load(HST)
m = (hlam >= 1700) & (hlam <= 9000) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]

llam, lflu = load(CHAMP)
band_h = np.trapezoid(hflu[(hlam>=4500)&(hlam<=5800)], hlam[(hlam>=4500)&(hlam<=5800)])
band_l = np.trapezoid(lflu[(llam>=4500)&(llam<=5800)], llam[(llam>=4500)&(llam<=5800)])
lflu *= band_h / band_l

print("Computing pseudo-continua (fixed 500 Å vs Doppler-scaled)...")
hcont_fix = pcont_fixed(hlam, hflu, 500.0)
hcont_vel = pcont_velocity(hlam, hflu)
lcont_fix = pcont_fixed(llam, lflu, 500.0)
lcont_vel = pcont_velocity(llam, lflu)

fig = plt.figure(figsize=(15, 11))
gs  = fig.add_gridspec(3, 1, height_ratios=[1, 2.4, 2.4], hspace=0.32)

# (a) window function vs λ
ax0 = fig.add_subplot(gs[0])
lam_grid = np.linspace(2000, 9000, 400)
beta = V_SMOOTH / C_KMS
win_vel = PCONT_SCALE * beta * lam_grid
ax0.plot(lam_grid, win_vel, "C2-", lw=2.0,
         label=f"window = {PCONT_SCALE:.0f}·(v/c)·λ,  v={V_SMOOTH/1e3:.0f}k km/s")
ax0.axhline(500.0, color="gray", ls="--", lw=1.0,
            label="고정 FWHM=500 Å (이전)")
for lx, w in [(3000, 200), (6000, 400), (9000, 600)]:
    ax0.scatter(lx, w, c="tab:red", zorder=5, s=40)
    ax0.annotate(f"({lx}, {w})", (lx, w), xytext=(8, 4),
                 textcoords="offset points", fontsize=9, color="tab:red")
ax0.set_xlim(2000, 9000); ax0.set_ylim(0, 700)
ax0.set_xlabel("λ (Å)"); ax0.set_ylabel("smoothing window (Å)")
ax0.set_title("(a) 파장-비례 smoothing window  vs  고정 FWHM=500 Å",
              fontsize=11, fontweight="bold")
ax0.grid(alpha=0.3); ax0.legend(loc="upper left", fontsize=10)

# (b) HST overlay
ax1 = fig.add_subplot(gs[1])
ax1.plot(hlam, hflu*1e14, "k-", lw=0.7, alpha=0.7, label="HST B-max")
ax1.plot(hlam, hcont_fix*1e14, color="gray", ls="--", lw=1.3, alpha=0.85,
         label="pseudo-cont (고정 FWHM=500 Å)")
ax1.plot(hlam, hcont_vel*1e14, color="tab:green", ls="-", lw=1.6, alpha=0.95,
         label=f"pseudo-cont (Doppler v={V_SMOOTH/1e3:.0f}k, scale={PCONT_SCALE:.0f})")
ax1.set_xlim(2000, 9000)
ax1.set_ylim(0, (hflu*1e14).max() * 1.15)
ax1.set_xlabel("λ (Å)"); ax1.set_ylabel(r"Flux (10$^{-14}$ erg/s/cm²/Å)")
ax1.set_title("(b) HST SN 2011fe B-max — 두 smoothing 비교", fontsize=11, fontweight="bold")
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.3)

# (c) LUMINA overlay
ax2 = fig.add_subplot(gs[2])
ax2.plot(llam, lflu*1e14, "k-", lw=0.6, alpha=0.55, label="LUMINA champ 152761 (formal)")
ax2.plot(llam, lcont_fix*1e14, color="gray", ls="--", lw=1.3, alpha=0.85,
         label="LUMINA pseudo-cont (고정 FWHM=500 Å)")
ax2.plot(llam, lcont_vel*1e14, color="tab:blue", ls="-", lw=1.6, alpha=0.95,
         label=f"LUMINA pseudo-cont (Doppler v={V_SMOOTH/1e3:.0f}k, scale={PCONT_SCALE:.0f})")
# overlay HST envelope as faint dotted reference
ax2.plot(hlam, hcont_vel*1e14, color="tab:green", ls=":", lw=1.2, alpha=0.7,
         label="HST Doppler env (참조)")
ax2.set_xlim(2000, 9000)
ax2.set_ylim(0, max((lflu*1e14).max(), (hflu*1e14).max()) * 1.15)
ax2.set_xlabel("λ (Å)"); ax2.set_ylabel(r"Flux (10$^{-14}$ erg/s/cm²/Å)")
ax2.set_title("(c) LUMINA champion — 두 smoothing 비교  (champ scaled to HST 4500–5800 Å)",
              fontsize=11, fontweight="bold")
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.3)

fig.suptitle("Pseudo-continuum: 파장 의존 (Doppler-scaled) smoothing 시연 — "
             f"win(λ) = {PCONT_SCALE:.0f}·(v_exp/c)·λ,  v_exp = {V_SMOOTH/1e3:.0f},000 km/s",
             fontsize=12, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT, dpi=140); plt.close()
print(f"Wrote {OUT}  ({OUT.stat().st_size//1024} KB)")
