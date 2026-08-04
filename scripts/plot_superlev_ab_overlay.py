#!/usr/bin/env python3
"""Super-level vs truncation overlay (formal spectra), optical-anchor pinned [4000,6000]Å."""
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
runs = {
    "truncation (161349)": glob.glob(f"{ROOT}/logs/*superlev_ab_trunc_161349")[0],
    "super-level (161350)": glob.glob(f"{ROOT}/logs/*superlev_ab_super_161350")[0],
}
colors = {"truncation (161349)": "#888888", "super-level (161350)": "#3898EC"}

def load(run):
    d = pd.read_csv(f"{run}/lumina_spectrum_formal.csv")
    return d["wavelength_angstrom"].values, d["flux"].values

# pin each to its own mean over [4000,6000] so shapes are comparable
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1]})
specs = {}
for lab, run in runs.items():
    w, f = load(run)
    m = (w >= 4000) & (w <= 6000)
    norm = np.mean(f[m])
    fn = f / norm
    specs[lab] = (w, fn)
    ax[0].plot(w, fn, color=colors[lab], lw=0.8, label=lab)

ax[0].set_xlim(1500, 10200)
ax[0].set_ylabel("F$_\\lambda$ (pinned to mean[4000,6000])")
ax[0].set_title("SN 2002bo DDC15 — super-level vs truncation (formal, optical-anchor pinned)")
ax[0].legend(loc="upper right")
ax[0].grid(alpha=0.2)
for b in (3000, 4000, 5500, 7000, 9000):
    ax[0].axvline(b, color="k", ls=":", alpha=0.25, lw=0.6)

# ratio super/trunc on common grid
wt, ft = specs["truncation (161349)"]
ws, fs = specs["super-level (161350)"]
fs_i = np.interp(wt, ws, fs)
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(ft > 0, fs_i / ft, np.nan)
ax[1].plot(wt, ratio, color="#D97757", lw=0.7)
ax[1].axhline(1.0, color="k", ls="--", alpha=0.5)
ax[1].set_ylim(0.5, 1.5)
ax[1].set_xlabel("wavelength (Å)")
ax[1].set_ylabel("super / trunc")
ax[1].grid(alpha=0.2)
for b in (3000, 4000, 5500, 7000, 9000):
    ax[1].axvline(b, color="k", ls=":", alpha=0.25, lw=0.6)

out = f"{ROOT}/figures/superlev_ab_overlay_161349_vs_161350.png"
plt.tight_layout(); plt.savefig(out, dpi=130); print("saved:", out)
