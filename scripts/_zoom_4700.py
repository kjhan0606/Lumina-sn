#!/usr/bin/env python3
"""Zoom [4300,5200]A: model configs vs dereddened SN2002bo, anchor [4000,6000].
Shows the 4700A peak morphology (peak-between-troughs?) the user flagged as ~70% over."""
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
MU,EBV,RV=31.90,0.41,3.1; A_V=RV*EBV
def ccm(w):
    x=1e4/w; a=np.zeros_like(x); b=np.zeros_like(x)
    s=(x>=1.1)&(x<=3.3); y=x[s]-1.82
    a[s]=1+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
    b[s]=1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
    s=(x>=0.3)&(x<1.1); a[s]=0.574*x[s]**1.61; b[s]=-0.527*x[s]**1.61
    s=(x>3.3)&(x<=8.0); xs=x[s]
    a[s]=1.752-0.316*xs-0.104/((xs-4.67)**2+0.341); b[s]=-3.090+1.825*xs+1.206/((xs-4.62)**2+0.263)
    return a+b/RV
def dered(w,f): return f*10**(0.4*A_V*ccm(w))

obs=pd.read_csv(f"{ROOT}/data/sn2002bo/epochs/sn2002bo_m0d0.csv",comment="#")
olam=obs["wavelength_angstrom"].values; oflu=dered(olam,obs["flux_erg_s_cm2_angstrom"].values)
ALO,AHI=4000.,6000.
Ioa=float(np.trapezoid(oflu[(olam>=ALO)&(olam<=AHI)],olam[(olam>=ALO)&(olam<=AHI)]))

RUNS=[(161427,"radeq_off_n12","trunc OFF","#1f77b4"),
      (161421,"radeq_milne_n12","trunc radeqON","#2ca02c"),
      (161630,"superSE_wcap_n12","super+Wcap OFF","#9467bd"),
      (161616,"superSE_n12","super noWcap OFF","#ff7f0e")]
fig,ax=plt.subplots(figsize=(13,7))
ax.plot(olam,oflu,lw=2.2,color="black",label="SN2002bo dered")
for job,tag,lbl,col in RUNS:
    c=sorted(glob.glob(f"{ROOT}/logs/*DDC15*2002bo_*_{tag}_{job}"))
    if not c: continue
    m=pd.read_csv(f"{c[0]}/lumina_spectrum_formal.csv")
    ml=m["wavelength_angstrom"].values; mr=m["flux"].values
    K=Ioa/float(np.trapezoid(mr[(ml>=ALO)&(ml<=AHI)],ml[(ml>=ALO)&(ml<=AHI)]))
    s=(ml>=4250)&(ml<=5250)
    ax.plot(ml[s],mr[s]*K,lw=1.3,color=col,alpha=0.9,label=lbl)
for x in (4700,): ax.axvline(x,lw=0.8,color="red",ls="--",alpha=0.5)
ax.axvspan(4600,4800,color="red",alpha=0.07)
ax.set_xlim(4250,5250); ax.set_xlabel("Wavelength [A]"); ax.set_ylabel("F_lambda (anchor-pinned)")
ax.set_title("4700A peak morphology — model configs vs SN2002bo (user: ~70% over in all configs)")
ax.legend(fontsize=9); ax.grid(True,alpha=0.25)
plt.tight_layout()
out=f"{ROOT}/figures/2026-06-01_zoom4700_vs_sn2002bo.png"; plt.savefig(out,dpi=140)
print("saved:",out)
# print band ratio in narrow 4600-4800
for job,tag,lbl,col in RUNS:
    c=sorted(glob.glob(f"{ROOT}/logs/*DDC15*2002bo_*_{tag}_{job}"))
    if not c: continue
    m=pd.read_csv(f"{c[0]}/lumina_spectrum_formal.csv")
    ml=m["wavelength_angstrom"].values; mr=m["flux"].values
    K=Ioa/float(np.trapezoid(mr[(ml>=ALO)&(ml<=AHI)],ml[(ml>=ALO)&(ml<=AHI)]))
    sel=(olam>=4600)&(olam<=4800); mi=np.interp(olam[sel],ml,mr*K)
    r=float(np.trapezoid(mi,olam[sel]))/float(np.trapezoid(oflu[sel],olam[sel]))
    print(f"  {lbl:20s} 4600-4800 mod/obs = {r:.2f}")
