#!/usr/bin/env python3
"""Energy budget: is the NIR over-flux explained by the UV deficit?

Anchor model+obs identically on [4000,6000] (F_scl frame), then for each band
integrate F_lambda over lambda to get band ENERGY, and report model-obs in the
same anchored units. If the NIR excess (model-obs > 0) is comparable in absolute
energy to the UV deficit (model-obs < 0), the cascade-reddening picture holds.
"""
import sys, glob
import numpy as np
import pandas as pd

EBV, RV = 0.41, 3.1
A_V = RV * EBV
ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
OBS = f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv"

BANDS = [
    ("UV mid",    1700, 2400),
    ("UV near",   2400, 3000),
    ("UV/blue",   3000, 4000),
    ("UV35-45",   3500, 4500),   # the user's hypothesized over-absorption band
    ("blue/grn",  4000, 5500),
    ("red",       5500, 7000),
    ("NIR I",     7000, 9000),
    ("NIR II",    9000,10200),
]

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

def ccm_deredden(wave, flux):
    return flux * 10**(0.4 * A_V * ccm_a_over_av(wave))

def anchor(wave, flux, lo=4000, hi=6000):
    m = (wave >= lo) & (wave <= hi)
    return flux / np.trapz(flux[m], wave[m]) * (hi - lo)

def band_energy(wave, flux, lo, hi):
    m = (wave >= lo) & (wave <= hi)
    if m.sum() < 2: return np.nan
    return np.trapz(flux[m], wave[m])

def load_model(job):
    run = glob.glob(f"{ROOT}/logs/*_{job}")[0]
    df = pd.read_csv(f"{run}/lumina_spectrum_formal.csv")
    return df["wavelength_angstrom"].values, df["flux"].values

def main():
    job = sys.argv[1]
    # obs
    obs = pd.read_csv(OBS, comment='#')
    ow = obs["wavelength_angstrom"].values
    of = ccm_deredden(ow, obs["flux_erg_s_cm2_angstrom"].values)
    of = anchor(ow, of)
    # model
    mw, mf = load_model(job)
    mf = anchor(mw, mf)

    print(f"JOB {job}  (anchored on [4000,6000], obs dereddened E(B-V)={EBV})")
    print(f"{'band':<10} {'range':>13} {'E_model':>10} {'E_obs':>10} {'M-O':>10} {'M/O':>6}")
    tot_def = 0.0; tot_exc = 0.0
    for name, lo, hi in BANDS:
        em = band_energy(mw, mf, lo, hi)
        eo = band_energy(ow, of, lo, hi)
        d = em - eo
        if name == "UV35-45":
            tag = "  <-- hypothesized over-absorption"
        else:
            tag = ""
            if lo >= 7000: tot_exc += max(d, 0.0)
            if hi <= 4000 and lo >= 1700: tot_def += min(d, 0.0)
        print(f"{name:<10} [{lo:>5},{hi:>5}] {em:10.1f} {eo:10.1f} {d:10.1f} {em/eo:6.2f}{tag}")
    print(f"\n  UV deficit (model<obs, [1700,4000]) total : {tot_def:10.1f}")
    print(f"  NIR excess (model>obs, [7000,10200]) total: {tot_exc:10.1f}")
    if tot_exc > 0:
        print(f"  |UV deficit| / NIR excess = {abs(tot_def)/tot_exc:5.2f}")

main()
