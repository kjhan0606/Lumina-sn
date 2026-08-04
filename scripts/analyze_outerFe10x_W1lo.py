#!/usr/bin/env python3
"""outerFe 10× W1 LOW scan (1M packets) — drop UVbl from ~1.5 toward 1.0."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT / "data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
CHAMP = ROOT / "logs/PROD_L19_W2x010_W3x065wide_152761/lumina_spectrum_formal.csv"

RUNS = {
    "champion 152761 (no oFe, W1=1.7)":             CHAMP,
    "outerFe 10× W1=0.0  W3=0.65 (1M, 154100)":     ROOT / "logs/outerFe10x_W10p0_1M_154100/lumina_spectrum_formal.csv",
    "outerFe 10× W1=0.25 W3=0.65 (1M, 154101)":     ROOT / "logs/outerFe10x_W10p25_1M_154101/lumina_spectrum_formal.csv",
    "outerFe 10× W1=0.5  W3=0.65 (1M, 154099)":     ROOT / "logs/outerFe10x_W10p5_1M_154099/lumina_spectrum_formal.csv",
    "outerFe 10× W1=1.0  W3=0.65 (1M, 154096+97 mean)": None,  # special: mean of two 1M runs
}

BANDS = [("UVbl",1700,2900),("CaK",3700,3950),("UVtg",2900,3700),
         ("fluo",3950,4500),("grn",4500,5800),("red",5800,7000)]

FE_DIAG = [("Fe II 5169", 5169.0, 14000, 19000),
           ("Fe III 5129", 5129.2, 13000, 17000)]
C_KMS = 299792.458


def band_int(lam, flu, lo, hi):
    m = (lam >= lo) & (lam <= hi)
    return np.trapezoid(flu[m], lam[m])


def measure_trough(lam, flu_norm, lam_rest, v_lo, v_hi):
    lo = lam_rest * (1 - v_hi/C_KMS)
    hi = lam_rest * (1 - v_lo/C_KMS)
    m = (lam >= lo) & (lam <= hi)
    if m.sum() < 5: return None, None
    fs = flu_norm[m]
    if len(fs) > 11:
        fs = savgol_filter(fs, 11, 3)
    j = int(np.argmin(fs))
    lam_obs = lam[m][j]
    depth = 1.0 - fs[j]
    return depth, (lam_rest - lam_obs)/lam_rest * C_KMS


def pseudo_cont(lam, flu, win=300):
    cont = np.zeros_like(flu)
    for i in range(len(lam)):
        sel = (lam >= lam[i]-win/2) & (lam <= lam[i]+win/2)
        cont[i] = np.percentile(flu[sel], 90)
    if len(cont) > 51:
        cont = savgol_filter(cont, 51, 3)
    return cont


h = pd.read_csv(HST)
hlam = h.iloc[:,0].values; hflu = h.iloc[:,1].values
m = (hlam >= 1700) & (hlam <= 8500) & np.isfinite(hflu) & (hflu > 0)
hlam, hflu = hlam[m], hflu[m]


def measure_one(p):
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
    sm_lam = lam[(lam>=1700)&(lam<=8500)]
    sm_flu = flu[(lam>=1700)&(lam<=8500)]
    sm_cont = pseudo_cont(sm_lam, sm_flu)
    sm_norm = sm_flu / sm_cont
    fe_dep = {}
    for label, lr, vlo, vhi in FE_DIAG:
        d, v = measure_trough(sm_lam, sm_norm, lr, vlo, vhi)
        fe_dep[label] = (d, v)
    return rats, rms, fe_dep


# Mean of two W1=1.0 1M runs
p1 = ROOT / "logs/outerFe10x_1M_arm1_154096/lumina_spectrum_formal.csv"
p2 = ROOT / "logs/outerFe10x_1M_arm0_154097/lumina_spectrum_formal.csv"
r1, rms1, fe1 = measure_one(p1)
r2, rms2, fe2 = measure_one(p2)
mean_rats = {b: (r1[b]+r2[b])/2 for b,_,_ in BANDS}
log_r = np.array([np.log10(mean_rats[b]) for b,_,_ in BANDS])
mean_rms = float(np.sqrt(np.mean(log_r**2)))
mean_fe = {l: ((fe1[l][0]+fe2[l][0])/2 if fe1[l][0] else None,
               (fe1[l][1]+fe2[l][1])/2 if fe1[l][1] else None) for l,_,_,_ in FE_DIAG}

results = {}
for name, p in RUNS.items():
    if p is None:
        results[name] = (mean_rats, mean_rms, mean_fe)
        continue
    if not p.exists():
        print(f"MISSING: {p}")
        continue
    results[name] = measure_one(p)

print("=== outerFe 10× W1 LOW scan (1M packets) ===")
hdr = f"{'Run':<48} | " + " | ".join(f"{b:>5}" for b,_,_ in BANDS) + " |   RMS"
print(hdr); print("-"*len(hdr))
for name, (rats, rms, _) in results.items():
    cells = " | ".join(f"{rats[b]:5.3f}" for b,_,_ in BANDS)
    print(f"{name:<48} | {cells} | {rms:.4f}")

print("\n=== Fe II 5169 / Fe III 5129 trough depths (HST: 0.31 / 0.31) ===")
for name, (_, _, fe_dep) in results.items():
    cells = []
    for label,_,_,_ in FE_DIAG:
        d, v = fe_dep[label]
        cells.append(f"{label}: d={d:.2f} v={v:5.0f}" if d else f"{label}: --")
    print(f"  {name:<48} | " + " | ".join(cells))
