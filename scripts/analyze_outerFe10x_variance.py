#!/usr/bin/env python3
"""outerFe 10× W1=1.0 W3=0.65 variance test: 200K (2 runs) vs 1M (2 runs)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"

RUNS = {
    "200K run #1 (154067)":            ROOT / "logs/outerFe10x_W11p0_154067/lumina_spectrum_formal.csv",
    "200K run #2 (154073)":            ROOT / "logs/outerFe10x_W11p0_W30p65_154073/lumina_spectrum_formal.csv",
    "1M run #1 (154096 arm1)":         ROOT / "logs/outerFe10x_1M_arm1_154096/lumina_spectrum_formal.csv",
    "1M run #2 (154097 arm0)":         ROOT / "logs/outerFe10x_1M_arm0_154097/lumina_spectrum_formal.csv",
}

BANDS = [("UVbl",1700,2900),("CaK",3700,3950),("UVtg",2900,3700),
         ("fluo",3950,4500),("grn",4500,5800),("red",5800,7000)]


def band_int(lam, flu, lo, hi):
    m = (lam >= lo) & (lam <= hi)
    return np.trapezoid(flu[m], lam[m])


h = pd.read_csv(HST)
hlam = h.iloc[:,0].values; hflu = h.iloc[:,1].values
m = (hlam >= 1700) & (hlam <= 8500) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]

results = {}
for name, p in RUNS.items():
    if not p.exists():
        print(f"MISSING: {p}")
        continue
    df = pd.read_csv(p)
    lam = df.iloc[:,0].values
    flu_col = "flux" if "flux" in df.columns else df.columns[1]
    flu = df[flu_col].values
    norm = band_int(hlam, hflu, 4500, 5800) / band_int(lam, flu, 4500, 5800)
    flu = flu * norm
    rats = {}
    for n, lo, hi in BANDS:
        rats[n] = band_int(lam, flu, lo, hi) / band_int(hlam, hflu, lo, hi)
    log_r = np.array([np.log10(rats[b]) if rats[b]>0 else -10 for b,_,_ in BANDS])
    rms = float(np.sqrt(np.mean(log_r**2)))
    results[name] = (rats, rms)

print("=== variance test: 200K vs 1M, identical config (W1=1.0, W3=0.65, outerFe 10×) ===")
hdr = f"{'Run':<28} | " + " | ".join(f"{b:>5}" for b,_,_ in BANDS) + " |   RMS"
print(hdr); print("-"*len(hdr))
for name, (rats, rms) in results.items():
    cells = " | ".join(f"{rats[b]:5.3f}" for b,_,_ in BANDS)
    print(f"{name:<28} | {cells} | {rms:.4f}")

# Pair statistics
def get(label):
    for n in results:
        if label in n: return results[n]
    return None

print("\n=== pair-spread Δ(band) ===")
for tag in ["200K", "1M"]:
    r1 = [v for n,v in results.items() if tag in n and "#1" in n]
    r2 = [v for n,v in results.items() if tag in n and "#2" in n]
    if not r1 or not r2: continue
    rats1, rms1 = r1[0]; rats2, rms2 = r2[0]
    print(f"\n{tag} pair Δ(run1 - run2):")
    for b,_,_ in BANDS:
        d = rats1[b] - rats2[b]
        print(f"  {b:<5} | run1={rats1[b]:.3f} run2={rats2[b]:.3f} Δ={d:+.3f}")
    print(f"  RMS  | run1={rms1:.4f} run2={rms2:.4f} Δ={rms1-rms2:+.4f}")
