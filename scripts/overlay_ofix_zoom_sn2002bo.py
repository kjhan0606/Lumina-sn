#!/usr/bin/env python3
"""O-fix A/B trough zoom vs SN 2002bo B-max — line-shape inspection.
Panels: UV pump, Ca II H&K + Si II 4000, Si II 6355, O I 7774 / Ca II NIR triplet.
Both Sobolev formal spectra anchor-normalized on [4000,6000]A vs dereddened obs.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUNS = [("160663", "partial O-fix", "darkorange"),
        ("160756", "full O-fix", "navy")]
EBV, RV = 0.41, 3.1; A_V = RV*EBV

def ccm(w):
    x = 1e4/w; a=np.zeros_like(x); b=np.zeros_like(x)
    s=(x>=1.1)&(x<=3.3); y=x[s]-1.82
    a[s]=1+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
    b[s]=1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
    s=(x>=0.3)&(x<1.1); a[s]=0.574*x[s]**1.61; b[s]=-0.527*x[s]**1.61
    s=(x>3.3)&(x<=8.0); xs=x[s]
    Fa=np.where(xs>=5.9,-0.04473*(xs-5.9)**2-0.009779*(xs-5.9)**3,0.0)
    Fb=np.where(xs>=5.9,0.2130*(xs-5.9)**2+0.1207*(xs-5.9)**3,0.0)
    a[s]=1.752-0.316*xs-0.104/((xs-4.67)**2+0.341)+Fa
    b[s]=-3.090+1.825*xs+1.206/((xs-4.62)**2+0.263)+Fb
    return a+b/RV

obs = pd.read_csv(f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam = obs["wavelength_angstrom"].values
oflu = obs["flux_erg_s_cm2_angstrom"].values * 10**(0.4*A_V*ccm(olam))
ALO,AHI=4000.,6000.
I_obs=float(np.trapezoid(oflu[(olam>=ALO)&(olam<=AHI)], olam[(olam>=ALO)&(olam<=AHI)]))

def load(job):
    m=pd.read_csv(f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_nltedump_{job}/lumina_spectrum_formal.csv")
    lam=m["wavelength_angstrom"].values; fr=m["flux"].values
    sa=(lam>=ALO)&(lam<=AHI); K=I_obs/float(np.trapezoid(fr[sa],lam[sa]))
    return lam, fr*K
D={j:load(j) for j,_,_ in RUNS}

panels=[("UV pump [1800,3400]",1800,3400),
        ("Ca II H&K / Si 4000 [3500,4400]",3500,4400),
        ("Si II 6355 [5700,6500]",5700,6500),
        ("O I 7774 / Ca II NIR [7400,8800]",7400,8800)]
fig,axes=plt.subplots(2,2,figsize=(14,9))
for ax,(ttl,lo,hi) in zip(axes.ravel(),panels):
    so=(olam>=lo)&(olam<=hi)
    ax.plot(olam[so],oflu[so],lw=1.3,color="black",alpha=0.85,label="2002bo")
    for j,lbl,c in RUNS:
        lam,flu=D[j]; sm=(lam>=lo)&(lam<=hi)
        ax.plot(lam[sm],flu[sm],lw=1.2,color=c,alpha=0.85,label=lbl)
    ax.set_xlim(lo,hi); ax.set_title(ttl,fontsize=11)
    ax.set_xlabel("Wavelength [A]"); ax.set_ylabel("F_lambda")
    ax.legend(fontsize=8); ax.grid(True,alpha=0.25)
fig.suptitle("O-fix A/B trough zoom vs SN 2002bo B-max (anchor-normalized)",fontsize=13)
plt.tight_layout()
out=f"{ROOT}/figures/2026-05-30_ofix_ab_zoom_sn2002bo_bmax.png"
plt.savefig(out,dpi=130); print(f"saved: {out}")
