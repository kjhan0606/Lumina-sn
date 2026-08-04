#!/usr/bin/env python3
"""N1 triple-stack analysis: does M1 (Fe II) stack additively on KL1?

Compares 4 cells (ctrl, K1_repl, KL1_repl, N1_triple) against HST + TARDIS
at FWHM=20k (sensitive metric from audit_baseline_metric.py).

Decision logic:
  ΔK = K1_repl - ctrl       (K1 single-lever signal)
  ΔKL = KL1_repl - ctrl     (KL1 stack signal)
  ΔN1 = N1_triple - ctrl    (triple signal)
  ΔM_from_N1 = N1 - KL1     (Fe II contribution on top of KL1)

  vs HST: if ΔM_from_N1 < 0 (better), Fe II stacks
  vs TARDIS: if ΔM_from_N1 ≈ 0 vs HST < 0, Fe II is HST-only artifact
             if ΔM_from_N1 < 0 in both, Fe II is real physics

Also reports Si II 6355 trough, band ratios, Phase D criteria.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
HST  = ROOT/"data/sn2011fe/hst_uv/sn2011fe_hst_bmax_stitched.csv"
TARDIS = ROOT/"data/sn2011fe/tardis_spectrum.csv"
C_KMS = 299792.458
SI_LAB = 6355.0
JOB = 156141

CELLS = [
    ("ctrl",       f"logs/ddc15N1_{JOB}_ddc15N1_ctrl/lumina_spectrum_formal.csv"),
    ("K1_repl",    f"logs/ddc15N1_{JOB}_ddc15N1_K1_repl/lumina_spectrum_formal.csv"),
    ("KL1_repl",   f"logs/ddc15N1_{JOB}_ddc15N1_KL1_repl/lumina_spectrum_formal.csv"),
    ("N1_triple",  f"logs/ddc15N1_{JOB}_ddc15N1_N1_triple/lumina_spectrum_formal.csv"),
]

def load(p):
    d = pd.read_csv(p); lam = d.iloc[:,0].values
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

def rms_bn(mlam, mflu, rlam, rflu, fwhm):
    mlam_n, mflu_n = gauss_baseline(mlam, mflu, fwhm)
    rlam_n, rflu_n = gauss_baseline(rlam, rflu, fwhm)
    if mlam_n is None or rlam_n is None: return np.nan
    fm = interp1d(mlam_n, mflu_n, kind='linear', bounds_error=False, fill_value=np.nan)
    return float(np.sqrt(np.nanmean((fm(rlam_n) - rflu_n)**2)))

def trough_v_depth(lam, flu, lab=SI_LAB, half_aa=200):
    m = (lam>=lab-half_aa)&(lam<=lab+half_aa)
    sl, sf = lam[m], flu[m]
    if len(sl)<5: return np.nan, np.nan, np.nan
    imin = np.argmin(sf); lam_min = sl[imin]; f_min = sf[imin]
    f_cont = sf.max()
    depth = 1.0 - f_min/f_cont if f_cont>0 else np.nan
    v_kms = (lab - lam_min)/lab * C_KMS
    return v_kms, depth, lam_min

# --- Load refs
hlam, hflu = load(HST)
tlam, tflu = load(TARDIS)
gH = band_int(hlam, hflu, 4500, 5800)
gT = band_int(tlam, tflu, 4500, 5800)

# --- Load cells, scale to HST & TARDIS green
data = {}
for name, rel in CELLS:
    p = ROOT/rel
    if not p.exists(): print(f"MISSING {name}"); continue
    lam, flu = load(p)
    g = band_int(lam, flu, 4500, 5800)
    data[name] = (lam, flu * (gH/g), flu * (gT/g))

# --- Baseline-norm RMS @ FWHM=20k and 40k vs HST and TARDIS
print("="*88)
print(f"=== Baseline-norm RMS vs HST + TARDIS, FWHM=20k & 40k (N1 job {JOB}) ===")
print(f"{'cell':<14s}  {'HST_20k':>9s} {'HST_40k':>9s} {'TAR_20k':>9s} {'TAR_40k':>9s}")
print("-"*88)
rms = {}
for name, (lam, fH, fT) in data.items():
    r_h20 = rms_bn(lam, fH, hlam, hflu, 20000)
    r_h40 = rms_bn(lam, fH, hlam, hflu, 40000)
    r_t20 = rms_bn(lam, fT, tlam, tflu, 20000)
    r_t40 = rms_bn(lam, fT, tlam, tflu, 40000)
    rms[name] = (r_h20, r_h40, r_t20, r_t40)
    print(f"{name:<14s}  {r_h20:>9.4f} {r_h40:>9.4f} {r_t20:>9.4f} {r_t40:>9.4f}")

# --- Delta analysis (the headline)
print()
print("="*88)
print("=== Δ RMS_bn (cell − ctrl) at FWHM=20k — sensitive metric ===")
print(f"{'comparison':<28s}  {'Δ HST_20k':>11s} {'Δ HST_40k':>11s} {'Δ TAR_20k':>11s} {'Δ TAR_40k':>11s}")
print("-"*88)
ctr = rms["ctrl"]
for name in ["K1_repl", "KL1_repl", "N1_triple"]:
    if name not in rms: continue
    d = tuple(rms[name][i] - ctr[i] for i in range(4))
    print(f"{name+' − ctrl':<28s}  {d[0]:>+11.4f} {d[1]:>+11.4f} {d[2]:>+11.4f} {d[3]:>+11.4f}")

# Fe II contribution on top of KL1
if "KL1_repl" in rms and "N1_triple" in rms:
    klr = rms["KL1_repl"]; n1r = rms["N1_triple"]
    d = tuple(n1r[i] - klr[i] for i in range(4))
    print(f"{'N1_triple − KL1_repl (Fe II)':<28s}  {d[0]:>+11.4f} {d[1]:>+11.4f} {d[2]:>+11.4f} {d[3]:>+11.4f}")

# --- M1 historical vs HST: was -0.0267 @ FWHM=20k. Compare.
print()
print("="*88)
print("=== Sanity: K1/KL1 from N1 cells vs original sweep (MC-noise check) ===")
print("    Old measurements at FWHM=20k vs HST:")
print("      K1 (job 156056):   Δ ≈ -0.020 (K1 paper memory)")
print("      KL1 (job 156110):  Δ = -0.0227")
print("      M1 f=0.5 (156136): Δ = -0.0267")
print("    If N1 K1/KL1 deltas differ by >0.02 = expected MC noise σ")

# --- Si II trough velocity + depth
print()
print("="*88)
print("=== Si II 6355 trough velocity, depth, λ_min (HST-scaled) ===")
hv, hd, hl = trough_v_depth(hlam, hflu)
print(f"  {'HST':<14s}  v={hv:+8.0f} km/s  depth={hd:.3f}  λ_min={hl:.1f}")
for name, (lam, fH, fT) in data.items():
    v, d, l = trough_v_depth(lam, fH)
    print(f"  {name:<14s}  v={v:+8.0f}  Δv={v-hv:+7.0f}  depth={d:.3f}  λ_min={l:.1f}  Δλ={l-hl:+5.1f}")

# --- Band ratios
print()
print("="*88)
print("=== Band ratios (model/HST, green-normalized) ===")
BANDS = [(2300,3500,"UV-mid"), (3500,4500,"blue"), (4500,5800,"green"),
         (5800,6800,"Si-red"), (6800,8000,"OI/cont"), (8000,9500,"Ca-IR")]
hdr = f"{'cell':<14s}"
for _,_,lab in BANDS: hdr += f"{lab:>10s}"
print(hdr)
for name, (lam, fH, fT) in data.items():
    row = f"{name:<14s}"
    for lo,hi,_ in BANDS:
        rH = band_int(hlam, hflu, lo, hi); rM = band_int(lam, fH, lo, hi)
        row += f"{rM/rH:>10.3f}"
    print(row)

# --- Decision summary
print()
print("="*88)
print("=== DECISION LOGIC ===")
if "N1_triple" in rms and "KL1_repl" in rms:
    dM_hst = rms["N1_triple"][0] - rms["KL1_repl"][0]
    dM_tar = rms["N1_triple"][2] - rms["KL1_repl"][2]
    dN_hst = rms["N1_triple"][0] - rms["ctrl"][0]
    dN_tar = rms["N1_triple"][2] - rms["ctrl"][2]
    print(f"  Fe II contribution (N1 − KL1) @ FWHM=20k:")
    print(f"    vs HST:    {dM_hst:+.4f}")
    print(f"    vs TARDIS: {dM_tar:+.4f}")
    print(f"  Total triple signal (N1 − ctrl) @ FWHM=20k:")
    print(f"    vs HST:    {dN_hst:+.4f}")
    print(f"    vs TARDIS: {dN_tar:+.4f}")
    print()
    if dM_hst < -0.005 and dM_tar < -0.005:
        print("  → Fe II stacks PHYSICALLY: both refs improve.")
    elif dM_hst < -0.005 and abs(dM_tar) < 0.005:
        print("  → Fe II is HST-ONLY (code-vs-data artifact): TARDIS unchanged.")
    elif dM_hst > 0.005 or dM_tar > 0.005:
        print("  → Fe II ANTAGONIZES KL1: stack breaks (antagonism with K/L levers).")
    else:
        print("  → Fe II below MC noise: contribution indistinguishable from null.")

    target = 0.20
    print()
    print(f"  Task #172 target: baseline-norm RMS ≤ {target} vs HST (now FWHM=20k canonical)")
    print(f"    N1_triple @ FWHM=20k vs HST: {rms['N1_triple'][0]:.4f}")
    if rms['N1_triple'][0] <= target:
        print("    *** TARGET MET ***")
    else:
        print(f"    gap to target: {rms['N1_triple'][0]-target:+.4f}")
