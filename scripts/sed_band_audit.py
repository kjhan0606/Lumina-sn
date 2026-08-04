#!/usr/bin/env python3
"""Per-band integrated flux audit: LUMINA r070 vs SN 2011fe p+3.7d (HST + Snifs).

Normalization: LUMINA flux is scaled so that integrated flux in [4500,5800] matches HST.
This is the same normalization used by rms_bn (b3p0 reference band).
Then we compute integrated flux in many physically-meaningful bands and report
ratios LUMINA/HST and LUMINA/Snifs."""

import glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
C_KMS = 2.998e5
FWHM = 20000.0
TAG = "phase+03.7"

# physically-meaningful bands for SN Ia p+3.7d
BANDS = [
    ("FUV",        1500, 1700, "below G230LB nominal start"),
    ("UV-blue",    1700, 2300, "Fe III continuum, weak Fe II 2090 onset"),
    ("UV-Fe2a",    2300, 2500, "Fe II UV1/UV2 (Fe II 2382)"),
    ("UV-Fe2b",    2500, 2700, "Fe II UV3 (Fe II 2600 forest)"),
    ("UV-MgII",    2700, 2900, "Mg II h+k 2796/2803 + Cr II"),
    ("UV-Fe3",     2900, 3100, "iron-peak III combined bump (#220)"),
    ("UV-CaHK",    3100, 3400, "Ca II H+K wing, Ti II onset"),
    ("Blue-line",  3400, 4000, "Ca H+K core, Fe II forest"),
    ("Blue-2",     4000, 4500, "Fe II/Co II forest, S II"),
    ("Ref",        4500, 5800, "REFERENCE band (Fe II 5169, S II W feature)"),
    ("Yellow",     5800, 6300, "Si II 5972, Na D, Fe II 6149"),
    ("Si26355",    6300, 6500, "Si II 6355 trough core"),
    ("Red-1",      6500, 7500, "Si II 6900, weak emission"),
    ("Red-2",      7500, 8200, "O I 7773 trough"),
    ("CaIR-1",     8200, 8800, "Ca II IR triplet 8498/8542/8662"),
    ("NIR",        8800, 10000, "tail, weak Mg II 9218"),
]

def band_int(lam, flu, lo, hi):
    sel = (lam >= lo) & (lam <= hi)
    if sel.sum() < 5: return float("nan")
    return float(np.trapezoid(flu[sel], lam[sel]))

def smooth(lam, flu, fwhm=FWHM):
    dl = np.median(np.diff(lam))
    mid = 0.5 * (lam[0] + lam[-1])
    sig = (fwhm / C_KMS) * mid / 2.355 / dl
    return gaussian_filter1d(flu, sig, mode="nearest")

def rms_bn(mod_lam, mod_flu, obs_lam, obs_flu, gO, wl_lo, wl_hi):
    g = band_int(mod_lam, mod_flu, 4500, 5800)
    if g <= 0 or gO <= 0: return float("nan")
    mod_flu_n = mod_flu * (gO/g)
    selO = (obs_lam>=wl_lo) & (obs_lam<=wl_hi)
    ol, of = obs_lam[selO], obs_flu[selO]
    if len(ol) < 10: return float("nan")
    sO = smooth(ol, of)
    selM = (mod_lam>=wl_lo-100) & (mod_lam<=wl_hi+100)
    ml, mf = mod_lam[selM], mod_flu_n[selM]
    if len(ml) < 10: return float("nan")
    sM = smooth(ml, mf)
    common = (ol>=ml[0]) & (ol<=ml[-1])
    mb = np.interp(ol[common], ml, mf/sM)
    return float(np.sqrt(np.mean((of[common]/sO[common] - mb)**2)))

# --- load ---
m = pd.read_csv(f"{ROOT}/logs/w1_156893_w1_r070/lumina_spectrum.csv")
mlam = m["wavelength_angstrom"].values.astype(float)
mflu = m["flux"].values.astype(float)

sn = pd.read_csv(f"{ROOT}/data/sn2011fe/epochs/sn2011fe_p3d2d.csv", comment="#")
slam = sn["wavelength_angstrom"].values
sflu = sn["flux_erg_s_cm2_angstrom"].values

hst_dir = f"{ROOT}/data/sn2011fe/hst_uv"
parts = []
for grat, w_lo, w_hi in [("G230LB", 1700, 2900), ("G430L", 2900, 5266), ("G750L", 5266, 10000)]:
    files = sorted(glob.glob(f"{hst_dir}/CCD_{grat}_*{TAG}*sx1.csv"))
    if not files: continue
    dfs = [pd.read_csv(f) for f in files]
    base = dfs[0].copy()
    if len(dfs) > 1:
        stack = np.column_stack([
            np.interp(base["wavelength_angstrom"].values, d["wavelength_angstrom"].values,
                      d["flux_erg_s_cm2_angstrom"].values) for d in dfs])
        base["flux_erg_s_cm2_angstrom"] = np.nanmean(stack, axis=1)
    sel = (base["wavelength_angstrom"] >= w_lo) & (base["wavelength_angstrom"] <= w_hi)
    parts.append(base.loc[sel, ["wavelength_angstrom", "flux_erg_s_cm2_angstrom"]])
hst = pd.concat(parts, ignore_index=True).sort_values("wavelength_angstrom").reset_index(drop=True)
hst = hst[hst["flux_erg_s_cm2_angstrom"] > 0]
hlam = hst["wavelength_angstrom"].values
hflu = hst["flux_erg_s_cm2_angstrom"].values

# also try FUV-MAMA G140L (912-1700Å) for FUV band
fuv_files = sorted(glob.glob(f"{hst_dir}/FUV-MAMA_G140L_*{TAG}*x1d.csv"))
if not fuv_files:
    fuv_files = sorted(glob.glob(f"{hst_dir}/FUV-MAMA_G140L_*phase+03.8*x1d.csv"))
if fuv_files:
    fuvd = [pd.read_csv(f) for f in fuv_files]
    fbase = fuvd[0].copy()
    if len(fuvd) > 1:
        stack = np.column_stack([
            np.interp(fbase["wavelength_angstrom"].values, d["wavelength_angstrom"].values,
                      d["flux_erg_s_cm2_angstrom"].values) for d in fuvd])
        fbase["flux_erg_s_cm2_angstrom"] = np.nanmean(stack, axis=1)
    fbase = fbase[(fbase["wavelength_angstrom"] >= 1200) & (fbase["wavelength_angstrom"] <= 1700)]
    fbase = fbase[fbase["flux_erg_s_cm2_angstrom"] > 0]
    flam = fbase["wavelength_angstrom"].values
    fflu = fbase["flux_erg_s_cm2_angstrom"].values
    print(f"FUV file: {fuv_files[0].split('/')[-1]}  ({len(flam)} points)")
else:
    flam = np.array([]); fflu = np.array([])
    print("No FUV-MAMA file")

# --- normalize LUMINA to HST [4500,5800] integrated flux ---
gH_ref = band_int(hlam, hflu, 4500, 5800)
gM_ref = band_int(mlam, mflu, 4500, 5800)
norm = gH_ref / gM_ref
mflu_n = mflu * norm
print(f"\nNormalization: HST[4500,5800] = {gH_ref:.3e},  LUMINA raw = {gM_ref:.3e},  scale = {norm:.3e}")

# --- per-band table ---
rows = []
for name, lo, hi, desc in BANDS:
    if hi <= 1700:
        oH = band_int(flam, fflu, lo, hi) if flam.size else float("nan")
    elif lo >= 1700:
        oH = band_int(hlam, hflu, lo, hi)
    else:
        # mixed: combine FUV + HST for this band
        oH_fuv = band_int(flam, fflu, lo, 1700) if flam.size else 0.0
        oH_ccd = band_int(hlam, hflu, 1700, hi)
        oH = (oH_fuv if not np.isnan(oH_fuv) else 0.0) + oH_ccd
    oS = band_int(slam, sflu, lo, hi)
    mB = band_int(mlam, mflu_n, lo, hi)
    rH = mB / oH if (oH and oH > 0 and not np.isnan(oH)) else float("nan")
    rS = mB / oS if (oS and oS > 0 and not np.isnan(oS)) else float("nan")
    # narrow-band line-shape rms_bn (only when band wide enough)
    if hi - lo >= 200 and lo >= 1700:
        rb_h = rms_bn(mlam, mflu, hlam, hflu, gH_ref, lo, hi)
    else:
        rb_h = float("nan")
    rows.append({
        "band": name, "λ_lo": lo, "λ_hi": hi,
        "HST [erg/s/cm²]": oH, "Snifs [erg/s/cm²]": oS,
        "LUMINA_norm": mB,
        "LUMINA/HST": rH, "LUMINA/Snifs": rS,
        "RMS_bn(HST)": rb_h,
        "physics": desc,
    })

# integrated totals
def total_in(lam, flu, lo, hi):
    return band_int(lam, flu, lo, hi)

print("\n=== PER-BAND INTEGRATED FLUX RATIOS (LUMINA r070, normalized to HST [4500,5800]) ===\n")
print(f"{'band':12s} {'λ range':15s} {'L/HST':>8s} {'L/Snifs':>8s} {'RMS_bn(HST)':>12s}  physics")
print("-" * 110)
for r in rows:
    s_lh = f"{r['LUMINA/HST']:6.3f}" if not np.isnan(r['LUMINA/HST']) else "  ----"
    s_ls = f"{r['LUMINA/Snifs']:6.3f}" if not np.isnan(r['LUMINA/Snifs']) else "  ----"
    s_rb = f"{r['RMS_bn(HST)']:8.4f}" if not np.isnan(r['RMS_bn(HST)']) else "    ----"
    print(f"{r['band']:12s} [{r['λ_lo']:5d},{r['λ_hi']:5d}] {s_lh:>8s} {s_ls:>8s} {s_rb:>12s}  {r['physics']}")

# totals
print()
T_lo = 1700; T_hi = 10000
o_tot = band_int(hlam, hflu, T_lo, T_hi)
s_tot = band_int(slam, sflu, max(T_lo, slam.min()), min(T_hi, slam.max()))
m_tot = band_int(mlam, mflu_n, T_lo, T_hi)
print(f"TOTAL [1700,10000]: HST={o_tot:.3e}  Snifs(in range)={s_tot:.3e}  LUMINA_norm={m_tot:.3e}")
print(f"  LUMINA / HST  (total)  = {m_tot/o_tot:.3f}")

T_lo = 3300; T_hi = 8000
o_tot = band_int(hlam, hflu, T_lo, T_hi)
s_tot = band_int(slam, sflu, T_lo, T_hi)
m_tot = band_int(mlam, mflu_n, T_lo, T_hi)
print(f"\nTOTAL [3300,8000] (Snifs range): HST={o_tot:.3e}  Snifs={s_tot:.3e}  LUMINA={m_tot:.3e}")
print(f"  LUMINA / HST   = {m_tot/o_tot:.3f}")
print(f"  LUMINA / Snifs = {m_tot/s_tot:.3f}")
print(f"  Snifs / HST    = {s_tot/o_tot:.3f}  (instrument-to-instrument check)")

# save table CSV
df = pd.DataFrame(rows)
df.to_csv(f"{ROOT}/figures/2026-05-22_W1_r070_sed_band_audit.csv", index=False)
print(f"\nsaved: {ROOT}/figures/2026-05-22_W1_r070_sed_band_audit.csv")

# --- figure: stacked bar chart of band ratios ---
fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 2]})

# panel 1: overlay spectra with band shading
ax = axes[0]
ax.plot(hlam, hflu, lw=1.0, color="black", alpha=0.80, label="HST stitched (obs)")
ax.plot(slam, sflu, lw=0.7, color="goldenrod", alpha=0.65, label="Snifs p+3.2d (obs)")
ax.plot(mlam, mflu_n, lw=1.0, color="crimson", alpha=0.85, label="LUMINA r070 (norm to HST [4500,5800])")
if flam.size:
    ax.plot(flam, fflu, lw=0.6, color="darkgreen", alpha=0.7, label="HST FUV-MAMA G140L (obs)")
ax.set_xlim(1500, 10000)
ax.set_ylim(0, max(hflu.max(), sflu.max()) * 1.2)
colors = plt.cm.tab20.colors
for i, (name, lo, hi, _) in enumerate(BANDS):
    ax.axvspan(lo, hi, alpha=0.08, color=colors[i % len(colors)])
    ax.text((lo+hi)/2, ax.get_ylim()[1]*0.96, name, fontsize=7, ha="center", rotation=60, color="dimgray")
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
ax.set_title("LUMINA r070 vs SN 2011fe p+3.7d — per-band SED audit")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)

# panel 2: band ratios as bars
ax = axes[1]
labels = [r["band"] for r in rows]
mids = [0.5*(r["λ_lo"]+r["λ_hi"]) for r in rows]
ratios_h = [r["LUMINA/HST"] for r in rows]
ratios_s = [r["LUMINA/Snifs"] for r in rows]
x = np.arange(len(labels))
w = 0.4
ax.bar(x - w/2, ratios_h, w, color="crimson", alpha=0.85, label="LUMINA / HST")
ax.bar(x + w/2, ratios_s, w, color="goldenrod", alpha=0.85, label="LUMINA / Snifs")
ax.axhline(1.0, lw=1.0, color="black", ls="--", alpha=0.7, label="perfect match")
ax.axhline(2.0, lw=0.5, color="gray", ls=":", alpha=0.5)
ax.axhline(0.5, lw=0.5, color="gray", ls=":", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("LUMINA / obs  (integrated flux ratio)")
ax.set_title("Per-band integrated flux ratios — UV under, Ref OK, Red over")
ax.set_ylim(0, 3.0)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25, axis="y")

plt.tight_layout()
out = f"{ROOT}/figures/2026-05-22_W1_r070_sed_band_audit.png"
plt.savefig(out, dpi=130)
print(f"saved: {out}")
