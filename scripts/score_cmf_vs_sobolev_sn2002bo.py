#!/usr/bin/env python3
"""CMF (paper-method) vs Sobolev formal vs SN 2002bo at B-max — the "verified map" sanity test.

Loads BOTH lumina_spectrum_cmf.csv (full comoving-frame line transfer, paper method)
and lumina_spectrum_formal.csv (Sobolev formal integral), each independently anchor-
normalized on [4000,6000]Å against the dereddened observed SN 2002bo spectrum.

Verdict logic:
  CMF recovers paper-consistent per-band ratios  -> LUMINA core (atomic/NLTE/plasma) sane;
                                                     gap localized to Sobolev transfer.
  CMF also diverges like Sobolev                 -> real atomic/NLTE/plasma bug.

Usage: score_cmf_vs_sobolev_sn2002bo.py <JOB> [TAG]
"""
import sys, glob, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
JOB = int(sys.argv[1]) if len(sys.argv) > 1 else 160067
TAG = sys.argv[2] if len(sys.argv) > 2 else "prod"

cand = sorted(glob.glob(f"{ROOT}/logs/*DDC15*2002bo_*_{TAG}_{JOB}")) or \
       sorted(glob.glob(f"{ROOT}/logs/*DDC15*2002bo_*_{JOB}")) or \
       sorted(glob.glob(f"{ROOT}/logs/*_{JOB}"))
if not cand:
    print(f"[err] no run dir for job {JOB}", file=sys.stderr); sys.exit(1)
RUN = cand[0]
print(f"RUN = {RUN}")

MU, EBV, RV = 31.90, 0.41, 3.1
A_V = RV * EBV

def ccm_a_over_av(wave_aa):
    x = 1e4 / wave_aa
    a = np.zeros_like(x); b = np.zeros_like(x)
    sel = (x >= 1.1) & (x <= 3.3); y = x[sel] - 1.82
    a[sel] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b[sel] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    sel = (x >= 0.3) & (x < 1.1)
    a[sel] =  0.574 * x[sel]**1.61; b[sel] = -0.527 * x[sel]**1.61
    sel = (x > 3.3) & (x <= 8.0); xs = x[sel]
    Fa = np.where(xs >= 5.9, -0.04473*(xs-5.9)**2 - 0.009779*(xs-5.9)**3, 0.0)
    Fb = np.where(xs >= 5.9,  0.2130*(xs-5.9)**2 + 0.1207*(xs-5.9)**3, 0.0)
    a[sel] =  1.752 - 0.316*xs - 0.104/((xs - 4.67)**2 + 0.341) + Fa
    b[sel] = -3.090 + 1.825*xs + 1.206/((xs - 4.62)**2 + 0.263) + Fb
    return a + b / RV

def deredden(wave_aa, flux):
    return flux * 10**(0.4 * A_V * ccm_a_over_av(wave_aa))

# --- obs ---
obs = pd.read_csv(f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam = obs["wavelength_angstrom"].values
oflu = deredden(olam, obs["flux_erg_s_cm2_angstrom"].values)

ANCHOR_LO, ANCHOR_HI = 4000., 6000.
sel_o_a = (olam >= ANCHOR_LO) & (olam <= ANCHOR_HI)
I_obs_anchor = float(np.trapezoid(oflu[sel_o_a], olam[sel_o_a]))
UVOIR_LO, UVOIR_HI = 1500., 10200.
sel_o2 = (olam >= max(UVOIR_LO, olam[0])) & (olam <= UVOIR_HI)
I_obs_UVOIR = float(np.trapezoid(oflu[sel_o2], olam[sel_o2]))

BANDS = [(1700,2400,"UV mid"),(2400,3000,"UV near"),(3000,4000,"UV/blue"),
         (4000,5500,"blue/green"),(5500,7000,"red"),(7000,9000,"NIR I"),(9000,10200,"NIR II")]

def load_norm(path, label):
    if not os.path.exists(path):
        print(f"[skip] {label}: {path} missing"); return None
    m = pd.read_csv(path)
    lam = m["wavelength_angstrom"].values; flu_raw = m["flux"].values
    sel_a = (lam >= ANCHOR_LO) & (lam <= ANCHOR_HI)
    K = I_obs_anchor / float(np.trapezoid(flu_raw[sel_a], lam[sel_a]))
    flu = flu_raw * K
    sel_uv = (lam >= UVOIR_LO) & (lam <= UVOIR_HI)
    Q = float(np.trapezoid(flu[sel_uv], lam[sel_uv])) / I_obs_UVOIR
    sel_fuv = (lam >= 1500) & (lam <= 3356)
    sel_all = (lam >= 1500) & (lam <= 10200)
    fuv_frac = float(np.trapezoid(flu[sel_fuv], lam[sel_fuv])) / float(np.trapezoid(flu[sel_all], lam[sel_all])) * 100
    bands = {}
    for lo, hi, name in BANDS:
        sel = (olam >= lo) & (olam <= hi)
        if sel.sum() < 5: continue
        mi = np.interp(olam[sel], lam, flu)
        bands[name] = float(np.trapezoid(mi, olam[sel])) / max(float(np.trapezoid(oflu[sel], olam[sel])), 1e-30)
    return dict(lam=lam, flu=flu, K=K, Q=Q, fuv=fuv_frac, bands=bands, label=label)

cmf = load_norm(f"{RUN}/lumina_spectrum_cmf.csv", "CMF (paper method)")
sob = load_norm(f"{RUN}/lumina_spectrum_formal.csv", "Sobolev formal")

print("\n=== Blondin SED metrics (optical anchor [4000,6000]Å pinned, F_scl≡1) ===")
print(f"  {'method':22s}  {'Q_uvoir':>8s}  {'|F-Q|':>7s}  {'FUV%[1500,3356]':>16s}")
print(f"  paper DDC15 vs 2002bo  {'1.06':>8s}  {'0.01':>7s}  {'~15':>16s}")
for s in (cmf, sob):
    if s: print(f"  {s['label']:22s}  {s['Q']:>8.3f}  {abs(1.0-s['Q']):>7.3f}  {s['fuv']:>15.1f}%")

print("\n=== Per-band model/obs ratios (dereddened obs; 1.00 = perfect) ===")
hdr = f"  {'band':12s}  {'λ range':>14s}"
if cmf: hdr += f"  {'CMF':>7s}"
if sob: hdr += f"  {'Sobolev':>8s}"
if cmf and sob: hdr += f"  {'CMF/Sob':>8s}"
print(hdr)
for lo, hi, name in BANDS:
    if (cmf and name not in cmf['bands']) and (sob and name not in sob['bands']): continue
    line = f"  {name:12s}  [{lo:5d},{hi:5d}]"
    cv = cmf['bands'].get(name) if cmf else None
    sv = sob['bands'].get(name) if sob else None
    if cmf: line += f"  {cv:>7.2f}" if cv else f"  {'-':>7s}"
    if sob: line += f"  {sv:>8.2f}" if sv else f"  {'-':>8s}"
    if cmf and sob and cv and sv: line += f"  {cv/sv:>8.2f}"
    print(line)

# --- Fig 6-style overlay ---
fig, axes = plt.subplots(3, 1, figsize=(13, 12), gridspec_kw={"height_ratios":[3,3,2.4]})
ax = axes[0]
ax.plot(olam, oflu, lw=0.9, color="black", alpha=0.85, label="SN 2002bo dereddened")
if sob: ax.plot(sob['lam'], sob['flu'], lw=0.9, color="crimson", alpha=0.8, label=f"Sobolev (Q={sob['Q']:.2f})")
if cmf: ax.plot(cmf['lam'], cmf['flu'], lw=1.0, color="navy", alpha=0.85, label=f"CMF paper-method (Q={cmf['Q']:.2f})")
ax.set_xlim(1500,10500); ax.set_xlabel("Wavelength [Å]"); ax.set_ylabel("F_λ [erg/s/cm²/Å]")
ax.set_title(f"CMF vs Sobolev vs SN 2002bo B-max (job {JOB}) — paper F_scl=1.05 Q_uvoir=1.06")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.25)

ax = axes[1]
so = (olam>=1500)&(olam<=10500)
ax.semilogy(olam[so], oflu[so], lw=0.9, color="black", alpha=0.85, label="2002bo dereddened")
if sob: sl=(sob['lam']>=1500)&(sob['lam']<=10500); ax.semilogy(sob['lam'][sl], sob['flu'][sl], lw=0.9, color="crimson", alpha=0.8, label="Sobolev")
if cmf: cl=(cmf['lam']>=1500)&(cmf['lam']<=10500); ax.semilogy(cmf['lam'][cl], cmf['flu'][cl], lw=1.0, color="navy", alpha=0.85, label="CMF")
ax.set_xlim(1500,10500); ax.set_xlabel("Wavelength [Å]"); ax.set_ylabel("log F_λ")
ax.set_title("Log F_λ — UVOIR span"); ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.25)

ax = axes[2]
ax.axhline(1.0, lw=1.0, color="black", ls="--", alpha=0.6, label="model=obs")
for s, c in ((sob,"crimson"),(cmf,"navy")):
    if not s: continue
    cm = (olam >= s['lam'][0]) & (olam <= s['lam'][-1])
    r = np.interp(olam[cm], s['lam'], s['flu']) / np.maximum(oflu[cm], 1e-30)
    ax.semilogy(olam[cm], r, lw=0.8, color=c, alpha=0.8, label=f"{s['label'].split()[0]}/obs")
for h in (2.0,0.5): ax.axhline(h, lw=0.5, color="orange", ls=":", alpha=0.4)
for h in (10.,0.1): ax.axhline(h, lw=0.5, color="red", ls=":", alpha=0.4)
ax.set_xlim(1500,10500); ax.set_ylim(0.05,200); ax.set_xlabel("Wavelength [Å]"); ax.set_ylabel("model/obs (log)")
ax.set_title("model/obs ratio"); ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.25, which="both")

plt.tight_layout()
out = f"{ROOT}/figures/2026-05-30_cmf_vs_sobolev_{JOB}_vs_sn2002bo_bmax.png"
os.makedirs(f"{ROOT}/figures", exist_ok=True)
plt.savefig(out, dpi=130)
print(f"\nsaved: {out}")
