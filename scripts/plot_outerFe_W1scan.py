#!/usr/bin/env python3
"""Plot outerFe W1 scan vs champion + HST."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
CHAMP = ROOT / "logs/PROD_L19_W2x010_W3x065wide_152761/lumina_spectrum_formal.csv"

RUNS = [
    ("champion 152761",          CHAMP, "black", 1.5),
    ("outerFe W1=1.0 (154062)",  ROOT/"logs/outerFe_W11p0_154062/lumina_spectrum_formal.csv", "tab:blue", 1.0),
    ("outerFe W1=1.3 (154063)",  ROOT/"logs/outerFe_W11p3_154063/lumina_spectrum_formal.csv", "tab:green", 1.0),
    ("outerFe W1=1.7 (154061)",  ROOT/"logs/outerFe_W11p7_154061/lumina_spectrum_formal.csv", "tab:red", 1.0),
]

h = pd.read_csv(HST)
hlam = h.iloc[:,0].values; hflu = h.iloc[:,1].values
m = (hlam>=1700)&(hlam<=8500)&np.isfinite(hflu)&(hflu>0)
hlam, hflu = hlam[m], hflu[m]


def band_int(l,f,lo,hi):
    m=(l>=lo)&(l<=hi); return np.trapezoid(f[m],l[m])


fig, axes = plt.subplots(3, 1, figsize=(14, 11))
bands = [(1700,3200,"UV  (Fe II 2382/2600 + Mg II 2796 + UVbl band)"),
         (3200,5800,"Blue/Optical (Ca II + Mg II + Fe II 4924/5018/5169 + Fe III)"),
         (5800,8500,"Red/NIR (Si II 6355 + O I + Ca IR triplet)")]

for ax, (lo, hi, title) in zip(axes, bands):
    mh = (hlam>=lo)&(hlam<=hi)
    ax.plot(hlam[mh], hflu[mh]*1e14, "k-", lw=1.0, alpha=0.85, label="HST")
    for name, p, color, lw in RUNS:
        if not p.exists(): continue
        df = pd.read_csv(p)
        lam = df.iloc[:,0].values
        flu = df["flux"].values if "flux" in df.columns else df.iloc[:,1].values
        norm = band_int(hlam, hflu, 4500, 5800)/band_int(lam, flu, 4500, 5800)
        flu = flu*norm
        ml = (lam>=lo)&(lam<=hi)
        ax.plot(lam[ml], flu[ml]*1e14, color=color, lw=lw, alpha=0.85, label=name)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("λ (Å)"); ax.set_ylabel(r"Flux (10$^{-14}$)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
out = ROOT/"figures/outerFe_W1scan.png"
plt.savefig(out, dpi=140); plt.close()
print(f"Wrote {out}")
