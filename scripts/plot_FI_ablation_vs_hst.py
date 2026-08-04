#!/usr/bin/env python3
"""FI ablation (FI_base / FI_cont / FI_both) vs HST + S2_f015 vs TARDIS.

Verifies whether the formal-integral continuum opacity addition
preserves line shape (no Planck artifacts) and that the >6000Å
baseline drops to TARDIS-class levels.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
HST  = f"{ROOT}/data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TAR  = f"{ROOT}/data/sn2011fe/tardis_spectrum.csv"
FIBASE = f"{ROOT}/logs/ddc15FI_156433_ddc15FI_FI_base/lumina_spectrum_formal.csv"
FICONT = f"{ROOT}/logs/ddc15FI_156433_ddc15FI_FI_cont/lumina_spectrum_formal.csv"
FIBOTH = f"{ROOT}/logs/ddc15FI_156433_ddc15FI_FI_both/lumina_spectrum_formal.csv"
OUT  = f"{ROOT}/figures/FI_ablation_vs_hst.png"
C_KMS = 2.998e5

def band_int(lam, flu, lo, hi):
    sel = (lam>=lo) & (lam<=hi)
    return np.trapezoid(flu[sel], lam[sel])

def baseline_norm(lam, flu, fwhm=20000.0, lo=3000.0, hi=8000.0):
    sel = (lam>=lo) & (lam<=hi)
    sl, sf = lam[sel], flu[sel]
    dlam = np.median(np.diff(sl)); mid = 0.5*(lo+hi)
    sigma = (fwhm/C_KMS)*mid/2.355/dlam
    base = gaussian_filter1d(sf, sigma, mode='nearest')
    return sl, sf/base

def load_norm(path, gH, hlam, hflu):
    df = pd.read_csv(path)
    if "wavelength_angstrom" in df.columns:
        lam = df["wavelength_angstrom"].values
        flu = df["flux"].values
    else:
        lam = df["wavelength_A"].values
        flu = df["flux_erg_s_A"].values
    g = band_int(lam, flu, 4500, 5800)
    return lam, flu * (gH/g)

def rms_bn(lam_m, flu_m, hlam, hflu):
    selH = (hlam>=3000) & (hlam<=8000)
    hl_b, hf_b = baseline_norm(hlam[selH], hflu[selH])
    selM = (lam_m>=2900) & (lam_m<=8100)
    ml_b, mf_b = baseline_norm(lam_m[selM], flu_m[selM])
    common = (hl_b >= ml_b[0]) & (hl_b <= ml_b[-1])
    mi = np.interp(hl_b[common], ml_b, mf_b)
    return float(np.sqrt(np.mean((hf_b[common] - mi)**2)))

h = pd.read_csv(HST)
hlam, hflu = h["wavelength_angstrom"].values, h["flux_erg_s_cm2_angstrom"].values
gH = band_int(hlam, hflu, 4500, 5800)

tlam, tflu_n = load_norm(TAR,    gH, hlam, hflu)
blam, bflu_n = load_norm(FIBASE, gH, hlam, hflu)
clam, cflu_n = load_norm(FICONT, gH, hlam, hflu)
xlam, xflu_n = load_norm(FIBOTH, gH, hlam, hflu)

rms_T = rms_bn(tlam, tflu_n, hlam, hflu)
rms_B = rms_bn(blam, bflu_n, hlam, hflu)
rms_C = rms_bn(clam, cflu_n, hlam, hflu)
rms_X = rms_bn(xlam, xflu_n, hlam, hflu)

# Red-band ratio diagnostic (smooth baseline)
def red_ratio(lam, flu, lo, hi):
    sel_H = (hlam>=lo) & (hlam<=hi)
    sel_M = (lam>=lo) & (lam<=hi)
    iH = np.trapezoid(hflu[sel_H], hlam[sel_H])
    iM = np.trapezoid(flu[sel_M], lam[sel_M])
    return iM/iH

red_bands = [("opt", 5800, 6500), ("redI", 6500, 7400),
             ("redII", 7400, 8200), ("redIII", 8200, 9000)]
print(f"\n{'model':12s}  RMS_bn   " + "  ".join(f"{nm:>6s}" for nm,_,_ in red_bands))
for nm, lam, flu, rms in [
        ("FI_base",  blam, bflu_n, rms_B),
        ("FI_cont",  clam, cflu_n, rms_C),
        ("FI_both",  xlam, xflu_n, rms_X),
        ("TARDIS",   tlam, tflu_n, rms_T)]:
    ratios = [red_ratio(lam, flu, lo, hi) for _, lo, hi in red_bands]
    print(f"{nm:12s}  {rms:.4f}  " + "  ".join(f"{r:>6.3f}" for r in ratios))

TROUGHS = [
    ("Si II 4130", 3950, 4250),
    ("Si II 5972", 5700, 6050),
    ("Si II 6355", 5950, 6450),
    ("Ca II H&K", 3650, 4000),
    ("Mg II 4481", 4300, 4600),
    ("Fe II m42 5018", 4750, 5100),
    ("Fe II 5169", 4900, 5250),
    ("O I 7773",  7400, 7900),
    ("Ca II IR",  7900, 8800),
]

fig = plt.figure(figsize=(16, 11), facecolor="white")
gs  = fig.add_gridspec(4, 3, height_ratios=[1.6, 1.0, 1.0, 1.0], hspace=0.42, wspace=0.22)

ax = fig.add_subplot(gs[0, :])
ax.plot(hlam, hflu, color="black",    lw=1.2, label="HST B-max")
ax.plot(tlam, tflu_n, color="tab:blue", lw=0.9, alpha=0.7, label=f"TARDIS  {rms_T:.3f}")
ax.plot(blam, bflu_n, color="dimgray", lw=0.9, alpha=0.8, label=f"FI_base   {rms_B:.3f}")
ax.plot(clam, cflu_n, color="crimson", lw=1.0, alpha=0.95, label=f"FI_cont   {rms_C:.3f}")
ax.plot(xlam, xflu_n, color="darkorange", lw=1.0, alpha=0.9, label=f"FI_both  {rms_X:.3f}")
ax.set_xlim(2800, 9200); ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel(r"$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title("FI ablation: line-only vs (line+e-scatter+W·B(T_rad) continuum) vs (+inner-Planck dilute)")
ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
ax.grid(alpha=0.3)
for nm, lo, hi in TROUGHS:
    ax.axvspan(lo, hi, alpha=0.06, color="gold")

for i, (nm, lo, hi) in enumerate(TROUGHS):
    r = 1 + i//3; c = i%3
    ax = fig.add_subplot(gs[r, c])
    selH = (hlam>=lo) & (hlam<=hi)
    selB = (blam>=lo) & (blam<=hi)
    selC = (clam>=lo) & (clam<=hi)
    selX = (xlam>=lo) & (xlam<=hi)
    selT = (tlam>=lo) & (tlam<=hi)
    ax.plot(hlam[selH], hflu[selH],     color="black",      lw=1.1, label="HST")
    ax.plot(tlam[selT], tflu_n[selT],   color="tab:blue",   lw=0.8, alpha=0.7, label="TARDIS")
    ax.plot(blam[selB], bflu_n[selB],   color="dimgray",    lw=0.8, alpha=0.8, label="base")
    ax.plot(clam[selC], cflu_n[selC],   color="crimson",    lw=1.0, alpha=0.9, label="cont")
    ax.plot(xlam[selX], xflu_n[selX],   color="darkorange", lw=1.0, alpha=0.85, label="both")
    ax.set_title(nm, fontsize=10)
    ax.set_xlim(lo, hi)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

fig.suptitle("FI ablation — formal-integral continuum opacity validation (job 156433, 600K×10)",
             y=0.995, fontsize=12, weight="bold")
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"\nsaved: {OUT}")
