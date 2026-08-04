#!/usr/bin/env python3
"""Baseline-norm metric audit: TARDIS ref + FWHM sweep.

Hypothesis: A_ul lever family (K1/L1/KL1/M1) saturated at RMS_bn ~0.30 vs HST.
Test whether (1) using TARDIS as reference (a code-vs-code comparison)
gives a tighter signal, and (2) which Gauss FWHM for baseline removal
maximizes ΔRMS between control and best A_ul cell.

If signal is metric-blind: identify a better metric.
If signal is real but small: confirm A_ul family closed.
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TARDIS = ROOT/"data/sn2011fe/tardis_spectrum.csv"
C_KMS = 299792.458

# Champion + diagnostic cells: ctrls from each sweep + best from each
# (For control sanity: 4 ctrls should give same physics → measure MC noise)
CELLS = [
    ("L1_ctrl",              "logs/ddc15L1_156085_ddc15L1_SiII_f1p0_ctrl/lumina_spectrum_formal.csv"),
    ("L1_SiII_f0.05",        "logs/ddc15L1_156085_ddc15L1_SiII_f0p05/lumina_spectrum_formal.csv"),
    ("KL1_ctrl",             "logs/ddc15KL1_156110_ddc15KL1_ctrl/lumina_spectrum_formal.csv"),
    ("KL1_NiII_f0.3",        "logs/ddc15KL1_156110_ddc15KL1_niIIonly_f0p3/lumina_spectrum_formal.csv"),
    ("KL1_SiII_f0.05",       "logs/ddc15KL1_156110_ddc15KL1_siIIonly_f0p05/lumina_spectrum_formal.csv"),
    ("KL1_stack",            "logs/ddc15KL1_156110_ddc15KL1_stack_ni0p3_si0p05/lumina_spectrum_formal.csv"),
    ("M1_ctrl",              "logs/ddc15M1_156136_ddc15M1_FeII_f1p0_ctrl/lumina_spectrum_formal.csv"),
    ("M1_FeII_f0.5",         "logs/ddc15M1_156136_ddc15M1_FeII_f0p5/lumina_spectrum_formal.csv"),
    ("M1_FeII_f0.1",         "logs/ddc15M1_156136_ddc15M1_FeII_f0p1/lumina_spectrum_formal.csv"),
]

def load_two_col(p):
    d = pd.read_csv(p)
    lam = d.iloc[:,0].values
    col = 'flux' if 'flux' in d.columns else d.columns[1]
    return lam, d[col].values

def band_int(lam, flu, lo, hi):
    m = (lam>=lo)&(lam<=hi)
    return np.trapezoid(flu[m], lam[m]) if m.sum()>=2 else np.nan

def gauss_baseline(lam, flu, fwhm_kms, mask_lo=3000, mask_hi=8000):
    sel = (lam >= mask_lo) & (lam <= mask_hi)
    sub_lam = lam[sel]; sub_flu = flu[sel]
    if len(sub_lam) < 50: return None, None
    dlam = np.median(np.diff(sub_lam))
    lam_mid = 0.5*(mask_lo+mask_hi)
    sigma_aa = (fwhm_kms/C_KMS) * lam_mid / 2.355
    sigma_pix = sigma_aa / dlam
    base = gaussian_filter1d(sub_flu, sigma_pix, mode='nearest')
    return sub_lam, sub_flu / base

# --- Load references
hlam, hflu = load_two_col(HST)
tlam, tflu = load_two_col(TARDIS)
gH = band_int(hlam, hflu, 4500, 5800)
gT = band_int(tlam, tflu, 4500, 5800)

# --- Load all model cells, scale to HST green
data = {}
for name, rel in CELLS:
    p = ROOT/rel
    if not p.exists(): print(f"MISSING {name}"); continue
    lam, flu = load_two_col(p)
    f_hst = flu * (gH / band_int(lam, flu, 4500, 5800))
    f_tar = flu * (gT / band_int(lam, flu, 4500, 5800))
    data[name] = (lam, f_hst, f_tar)

FWHMS = [10000, 20000, 30000, 40000, 50000, 60000]

def rms_bn(mlam, mflu, rlam, rflu, fwhm):
    mlam_n, mflu_n = gauss_baseline(mlam, mflu, fwhm)
    rlam_n, rflu_n = gauss_baseline(rlam, rflu, fwhm)
    if mlam_n is None or rlam_n is None: return np.nan
    fm = interp1d(mlam_n, mflu_n, kind='linear', bounds_error=False, fill_value=np.nan)
    return float(np.sqrt(np.nanmean((fm(rlam_n) - rflu_n)**2)))

# Header
hdr_fwhm = "  ".join([f"F{fw//1000:>4d}k" for fw in FWHMS])

print("="*90)
print("=== RMS_bn vs HST  (current metric reference) ===")
print(f"{'cell':<24s}  {hdr_fwhm}")
print("-"*90)
hst_rms = {}
for name, (lam, fH, fT) in data.items():
    row = []
    for fw in FWHMS:
        r = rms_bn(lam, fH, hlam, hflu, fw)
        row.append(r)
    hst_rms[name] = row
    print(f"{name:<24s}  " + "  ".join([f"{r:6.4f}" for r in row]))

print()
print("="*90)
print("=== RMS_bn vs TARDIS  (code-vs-code) ===")
print(f"{'cell':<24s}  {hdr_fwhm}")
print("-"*90)
tar_rms = {}
for name, (lam, fH, fT) in data.items():
    row = []
    for fw in FWHMS:
        r = rms_bn(lam, fT, tlam, tflu, fw)
        row.append(r)
    tar_rms[name] = row
    print(f"{name:<24s}  " + "  ".join([f"{r:6.4f}" for r in row]))

print()
print("="*90)
print("=== ΔRMS_bn vs HST  (best A_ul cell − ctrl, per sweep, per FWHM) ===")
print("    Positive value = ctrl is BETTER (A_ul lever worsens)")
print("    Negative value = A_ul cell is BETTER")
print()
print(f"{'comparison':<32s}  {hdr_fwhm}")
print("-"*90)
PAIRS_HST = [
    ("L1: SiII_f0.05 - ctrl",       "L1_SiII_f0.05", "L1_ctrl"),
    ("KL1: NiII_f0.3 - ctrl",       "KL1_NiII_f0.3", "KL1_ctrl"),
    ("KL1: SiII_f0.05 - ctrl",      "KL1_SiII_f0.05", "KL1_ctrl"),
    ("KL1: stack - ctrl",           "KL1_stack",     "KL1_ctrl"),
    ("M1: FeII_f0.5 - ctrl",        "M1_FeII_f0.5",  "M1_ctrl"),
    ("M1: FeII_f0.1 - ctrl",        "M1_FeII_f0.1",  "M1_ctrl"),
]
print("  --- vs HST ---")
for lab, best, ctrl in PAIRS_HST:
    dr = [hst_rms[best][i] - hst_rms[ctrl][i] for i in range(len(FWHMS))]
    print(f"{lab:<32s}  " + "  ".join([f"{d:+6.4f}" for d in dr]))
print("  --- vs TARDIS ---")
for lab, best, ctrl in PAIRS_HST:
    dr = [tar_rms[best][i] - tar_rms[ctrl][i] for i in range(len(FWHMS))]
    print(f"{lab:<32s}  " + "  ".join([f"{d:+6.4f}" for d in dr]))

print()
print("="*90)
print("=== Ctrl-vs-ctrl spread = MC noise floor (4 independent 800K runs) ===")
ctrls = ["L1_ctrl", "KL1_ctrl", "M1_ctrl"]
print(f"{'metric':<20s}  {hdr_fwhm}")
print("  --- vs HST ---")
for fw_i, fw in enumerate(FWHMS):
    vals = [hst_rms[c][fw_i] for c in ctrls]
    print(f"  std(ctrl)/mean   " + f"{'':>{6 + 8*fw_i}}{np.std(vals)/np.mean(vals):>6.4f}")
    break  # just print all FWHMs in one row
rows_hst = [[hst_rms[c][i] for c in ctrls] for i in range(len(FWHMS))]
rows_tar = [[tar_rms[c][i] for c in ctrls] for i in range(len(FWHMS))]
print(f"{'std(HST)':<20s}  " + "  ".join([f"{np.std(rows_hst[i]):6.4f}" for i in range(len(FWHMS))]))
print(f"{'mean(HST)':<20s}  " + "  ".join([f"{np.mean(rows_hst[i]):6.4f}" for i in range(len(FWHMS))]))
print(f"{'std(TARDIS)':<20s}  " + "  ".join([f"{np.std(rows_tar[i]):6.4f}" for i in range(len(FWHMS))]))
print(f"{'mean(TARDIS)':<20s}  " + "  ".join([f"{np.mean(rows_tar[i]):6.4f}" for i in range(len(FWHMS))]))

print()
print("="*90)
print("=== SNR proxy: |max ΔRMS over ctrl| / std(ctrl) per FWHM ===")
print("    SNR > 2: lever is distinguishable from MC noise")
print()
print(f"{'ref':<10s}  {hdr_fwhm}")
print("-"*90)
for ref_name, ref_rms in [("HST", hst_rms), ("TARDIS", tar_rms)]:
    snrs = []
    for fw_i, fw in enumerate(FWHMS):
        ctrl_vals = [ref_rms[c][fw_i] for c in ctrls]
        ctrl_std = np.std(ctrl_vals)
        # max |Δ| across all A_ul cells vs their own ctrl
        max_d = 0.0
        for lab, best, ctrl in PAIRS_HST:
            d = abs(ref_rms[best][fw_i] - ref_rms[ctrl][fw_i])
            max_d = max(max_d, d)
        snrs.append(max_d / ctrl_std if ctrl_std > 0 else np.nan)
    print(f"{ref_name:<10s}  " + "  ".join([f"{s:6.2f}" for s in snrs]))
