#!/usr/bin/env python3
"""3-metric champions overlay: HST + 5 LUMINA runs.

  vinner_v11500k_153938              — w5 (5-band weighted) #1
  outerFe10x_W11p0_W31p05_154072     — w5 #2
  sigbf_W2x015_W3x065wide_152629     — raw 6-band #1
  exp_strat7_defL_dyntp8_151197      — baseline-norm #1
  PROD_L19_W2x010_W3x065wide_152761  — current champion (sigbf middle)

각 run은 HST [4500,5800] band integral로 정규화. baseline은 Gauss FWHM=(40k/c)·λ.
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
OUT  = ROOT / "figures/w5_winners_overlay.png"
C_KMS = 299792.458
FWHM_KMS = 40000.0

RUNS = [
    ("vinner_v11500k_153938",                "tab:purple", "w5 #1: vinner_v11500k_153938"),
    ("outerFe10x_W11p0_W31p05_154072",       "tab:cyan",   "w5 #2: outerFe10x_W11p0_W31p05_154072"),
    ("sigbf_W2x015_W3x065wide_h100_152629",  "tab:red",    "raw #1: sigbf_W2x015_W3x065wide_152629"),
    ("exp_strat7_defL_dyntp8_151197",        "tab:green",  "bn #1: exp_strat7_defL_dyntp8_151197"),
    ("PROD_L19_W2x010_W3x065wide_152761",    "tab:orange", "champ: PROD_L19_W2x010_W3x065wide_152761"),
]

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
hcont = gauss(hlam, hflu); hnorm = hflu / hcont

# Load all runs, scale to HST 4500-5800
data = []
for run, color, lbl in RUNS:
    p = ROOT / f"logs/{run}/lumina_spectrum_formal.csv"
    lam, flu = load(p)
    flu = np.where(np.isfinite(flu)&(flu>=0), flu, 0.0)
    scale = band_int(hlam, hflu, 4500, 5800) / band_int(lam, flu, 4500, 5800)
    flu = flu * scale
    cont = gauss(lam, flu)
    norm = flu / cont
    data.append((run, color, lbl, lam, flu, cont, norm))
    print(f"  loaded {run[:50]:50s}  scale={scale:.3e}")

# ---- Plot ----
fig = plt.figure(figsize=(18, 16))
gs = fig.add_gridspec(5, 1, height_ratios=[1.4, 1.4, 1.0, 1.0, 1.0], hspace=0.32)

# (a) raw flux full
ax = fig.add_subplot(gs[0])
ax.plot(hlam, hflu*1e14, "k-", lw=1.0, alpha=0.85, label="HST B-max", zorder=1)
for run, color, lbl, lam, flu, cont, norm in data:
    ax.plot(lam, flu*1e14, color=color, lw=0.8, alpha=0.75, label=lbl, zorder=2)
ax.set_xlim(1700, 9000)
ax.set_ylim(0, max((hflu*1e14).max(),
                   max((d[4]*1e14).max() for d in data)) * 1.1)
ax.set_xlabel("λ (Å)"); ax.set_ylabel(r"Flux (10$^{-14}$ erg/s/cm²/Å)")
ax.set_title("(a) Raw flux — 5 LUMINA winners vs HST (scaled to [4500,5800])",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, ncol=1)
ax.grid(alpha=0.3)

# (b) normalized flux
ax = fig.add_subplot(gs[1])
ax.plot(hlam, hnorm, "k-", lw=1.0, alpha=0.85, label="HST / baseline", zorder=1)
for run, color, lbl, lam, flu, cont, norm in data:
    ax.plot(lam, norm, color=color, lw=0.8, alpha=0.75, label=lbl, zorder=2)
ax.axhline(1.0, color="gray", ls=":", lw=0.7)
ax.set_xlim(1700, 9000); ax.set_ylim(0, 1.6)
ax.set_xlabel("λ (Å)"); ax.set_ylabel("flux / baseline (FWHM=40k)")
ax.set_title("(b) Baseline-normalized — line shape comparison",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

# (c-e) zoom panels: UV, CaK/UVtg, Si II
zooms = [
    ("(c) UV [2000,3500]: UVbl/UVtg",      2000, 3500),
    ("(d) [3500,5000]: CaK + fluo",        3500, 5000),
    ("(e) [5000,7500]: grn + Si II 6355",  5000, 7500),
]
for k, (title, lo, hi) in enumerate(zooms):
    ax = fig.add_subplot(gs[2+k])
    sel_h = (hlam>=lo)&(hlam<=hi)
    ax.plot(hlam[sel_h], hflu[sel_h]*1e14, "k-", lw=1.4, alpha=0.9, label="HST")
    for run, color, lbl, lam, flu, cont, norm in data:
        sel = (lam>=lo)&(lam<=hi)
        # Short label only — color shows which
        short = run.split("_")[0]
        ax.plot(lam[sel], flu[sel]*1e14, color=color, lw=1.0, alpha=0.8)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("λ (Å)"); ax.set_ylabel(r"Flux (10$^{-14}$)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

fig.suptitle("3-metric Champions Overlay — Pareto front of dual+weighted metrics",
             fontsize=12, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT, dpi=140); plt.close()
print(f"\nWrote {OUT}  ({OUT.stat().st_size//1024} KB)")

# Print per-band log RMS for each run for diagnosis
print("\n=== Per-band log10(F_lumina/F_HST) ===")
print(f"{'run':52s} {'UVbl':>8s} {'UVtg':>8s} {'CaK':>8s} {'fluo':>8s} {'grn':>8s} {'red':>8s}")
BANDS = [("UVbl",1700,2900),("UVtg",2900,3700),("CaK",3700,3950),
         ("fluo",3950,4500),("grn",4500,5800),("red",5800,7000)]
for run, color, lbl, lam, flu, cont, norm in data:
    parts = []
    for n, lo, hi in BANDS:
        num = band_int(lam, flu, lo, hi)
        den = band_int(hlam, hflu, lo, hi)
        if num>0 and den>0:
            parts.append(np.log10(num/den))
        else:
            parts.append(np.nan)
    print(f"{run[:50]:52s} " + " ".join(f"{x:+8.4f}" for x in parts))
