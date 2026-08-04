#!/usr/bin/env python3
"""Compare stratV2 variants (Fe-out / Si-in / combined) vs champion + HST.

For each user wishlist band, report:
  - peak λ shift relative to champion 152761
  - peak amplitude change
  - approximate Δv inferred from λ shift
"""
import sys, glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
CHAMP = ROOT / "logs/PROD_L19_W2x010_W3x065wide_152761/lumina_spectrum_formal.csv"

# locate latest stratV2 jobs
def latest(pat):
    cands = sorted(glob.glob(str(ROOT / "logs" / pat / "lumina_spectrum_formal.csv")),
                   key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0)
    return Path(cands[-1]) if cands else None

RUNS = {
    "champion 152761":     CHAMP,
    "stratV2A (Fe out)":   latest("stratV2A_FeOut_*"),
    "stratV2B (Si in)":    latest("stratV2B_SiIn_*"),
    "stratV2C (combo)":    latest("stratV2C_combo_*"),
    "stratV2D (Fe both 0.06)":  latest("stratV2D_FeBoth_*"),
    "stratV2E (Fe both 0.03)":  latest("stratV2E_FeBoth03_*"),
    "stratV2F (Fe both 0.01)":  latest("stratV2F_FeBoth01_*"),
}

WISH = [
    ("A 3500", 3400, 3600, "lower"),
    ("B 4000", 3800, 4100, "raise"),
    ("C 4300", 4250, 4400, "+50Å red, lower"),
    ("D 4700", 4600, 4800, "-200Å BLUE"),
    ("E 5500", 5400, 5600, "lower"),
    ("F 6000", 5900, 6100, "+150Å Si II red"),
]


def band_int(lam, flu, lo, hi):
    msk = (lam >= lo) & (lam <= hi)
    return np.trapezoid(flu[msk], lam[msk])


h = pd.read_csv(HST)
hlam, hflu = h["wavelength_angstrom"].values, h["flux_erg_s_cm2_angstrom"].values

data = {}
for name, p in RUNS.items():
    if p is None or not p.exists():
        print(f"MISSING: {name}")
        continue
    df = pd.read_csv(p)
    lam, flu = df["wavelength_angstrom"].values, df["flux"].values
    norm = band_int(hlam, hflu, 4500, 5800) / band_int(lam, flu, 4500, 5800)
    data[name] = (lam, flu * norm)

if "champion 152761" not in data:
    print("ERROR: champion baseline missing"); sys.exit(1)

# peak per band
print(f"\n{'Band':<8} | {'wish':<18} | "
      + " | ".join(f"{n:<24}" for n in data.keys()))
print("-" * (8 + 21 + 27 * len(data)))
peaks = {}
for label, lo, hi, wish in WISH:
    cells = []
    for name, (lam, flu) in data.items():
        msk = (lam >= lo) & (lam <= hi)
        if msk.sum() < 3:
            cells.append("(no data)"); continue
        flu_sm = uniform_filter1d(flu[msk], size=3)
        i = int(np.argmax(flu_sm))
        peaks.setdefault(label, {})[name] = (lam[msk][i], flu_sm[i])
        cells.append(f"λ={lam[msk][i]:6.1f}  F={flu_sm[i]*1e14:5.1f}")
    print(f"{label:<8} | {wish:<18} | " + " | ".join(f"{c:<24}" for c in cells))

# Δλ vs champion + Δv
print(f"\n=== Δλ relative to champion 152761 + inferred Δv (km/s) ===")
print(f"{'Band':<8} | "
      + " | ".join(f"{n:<28}" for n in list(data.keys())[1:]))
print("-" * (8 + 31 * (len(data) - 1)))
for label, lo, hi, wish in WISH:
    if label not in peaks: continue
    base_lam, base_flu = peaks[label].get("champion 152761", (None, None))
    if base_lam is None: continue
    cells = []
    for name in list(data.keys())[1:]:
        lam_v, flu_v = peaks[label].get(name, (None, None))
        if lam_v is None:
            cells.append("(no data)"); continue
        dlam = lam_v - base_lam
        dv = (dlam / base_lam) * 3e5
        df_pct = (flu_v / base_flu - 1.0) * 100 if base_flu > 0 else 0
        cells.append(f"Δλ={dlam:+5.0f}Å Δv={dv:+5.0f} ΔF={df_pct:+5.1f}%")
    print(f"{label:<8} | " + " | ".join(f"{c:<28}" for c in cells))

# 6-band UVbl/CaK/UVtg/fluo/grn/red ratios + RMS
BANDS = [
    ("UVbl",  1700, 2900),
    ("CaK",   3700, 3950),
    ("UVtg",  2900, 3700),
    ("fluo",  3950, 4500),
    ("grn",   4500, 5800),
    ("red",   5800, 7000),
]


def ratio_metric(lam, flu):
    rats = {}
    for name, lo, hi in BANDS:
        m_int = band_int(lam, flu, lo, hi)
        h_int = band_int(hlam, hflu, lo, hi)
        rats[name] = m_int / h_int if h_int > 0 else 0
    log_r = np.array([np.log10(rats[b]) if rats[b] > 0 else -10 for b, _, _ in BANDS])
    rms = float(np.sqrt(np.mean(log_r ** 2)))
    return rats, rms


print(f"\n=== 6-band ratio + RMS log10 (vs HST) ===")
print(f"{'Run':<22} | " + " | ".join(f"{b:>5}" for b, _, _ in BANDS) + " | RMS")
for name, (lam, flu) in data.items():
    rats, rms = ratio_metric(lam, flu)
    cells = " | ".join(f"{rats[b]:5.3f}" for b, _, _ in BANDS)
    print(f"{name:<22} | {cells} | {rms:.4f}")

# plot
fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
ax = axes[0]
ax.plot(hlam, hflu * 1e14, color="black", lw=1.0, alpha=0.85, label="HST")
colors = {
    "champion 152761":   "#444444",
    "stratV2A (Fe out)": "#1f77b4",
    "stratV2B (Si in)":  "#2ca02c",
    "stratV2C (combo)":  "#d62728",
}
for name, (lam, flu) in data.items():
    ax.plot(lam, flu * 1e14, color=colors.get(name, "gray"),
            lw=1.0, alpha=0.85, label=name)
for label, lo, hi, _ in WISH:
    ax.axvspan(lo, hi, color="gold", alpha=0.10)
    ax.text(0.5*(lo+hi), 95, label, ha="center", fontsize=8)
ax.set_xlim(2500, 7000); ax.set_ylim(0, 110)
ax.set_ylabel(r"Flux (10$^{-14}$)")
ax.set_title("stratV2 vs champion 152761 + HST  (200K x 10 iter, champion W's)")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.25)

ax = axes[1]
hlam_g = np.linspace(2500, 7000, 1000)
hflu_g = np.interp(hlam_g, hlam, hflu)
for name, (lam, flu) in data.items():
    if name == "champion 152761": continue
    flu_g = np.interp(hlam_g, lam, flu)
    base_g = np.interp(hlam_g, *data["champion 152761"])
    ratio = np.where((flu_g > 0) & (base_g > 0), np.log10(flu_g / base_g), 0)
    ax.plot(hlam_g, ratio, color=colors.get(name), lw=1.0, alpha=0.85,
            label=name + " vs champion")
ax.axhline(0, color="black", lw=0.5, alpha=0.4)
ax.set_ylabel("log10(model/champion)")
ax.set_xlabel("λ (Å)")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.25)
ax.set_ylim(-0.3, 0.3)

out = ROOT / "figures/stratV2_scan.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"\nWrote {out}")
