#!/usr/bin/env python3
"""Capped-macroatom (cap=200 smoke 160855) vs scatter baseline (160767),
overlaid on SN 2002bo B-max. Both Sobolev formal spectra, anchor-normalized on
[4000,6000]A vs dereddened obs. NOTE: macroatom run truncates ~4% of packets at
the interaction cap (luminosity bias) and is a 200k-pkt 6-iter SMOKE, not the
converged 800k/15-iter production -- treat as a direction check, not a champion.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUNS = [("ofix_160767",              "scatter (baseline, converged)", "darkorange"),
        ("macroatom_smoke_160855",   "macroatom cap=200 (smoke, 4% trunc)", "navy")]
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

def load(tag):
    p=f"{ROOT}/logs/paperDDC15einsteinFix_2002bo_vi9019_L1p0_{tag}/lumina_spectrum_formal.csv"
    m=pd.read_csv(p); lam=m["wavelength_angstrom"].values; fr=m["flux"].values
    sa=(lam>=ALO)&(lam<=AHI); K=I_obs/float(np.trapezoid(fr[sa],lam[sa]))
    return lam, fr*K
D={t:load(t) for t,_,_ in RUNS}

BANDS=[("UV [1800,3000]",1800,3000),("blue <3700 [3000,3700]",3000,3700),
       ("4050-4250",4050,4250),("4300 peak [4250,4400]",4250,4400),
       ("opt [4400,5700]",4400,5700),("Si6355 [5700,6500]",5700,6500),
       ("red [6500,7400]",6500,7400),("CaNIR [7400,8800]",7400,8800)]
print(f"{'band':24s}" + "".join(f"{l:>26s}" for _,l,_ in RUNS))
for ttl,lo,hi in BANDS:
    so=(olam>=lo)&(olam<=hi); Io=float(np.trapezoid(oflu[so],olam[so]))
    row=f"{ttl:24s}"
    for t,_,_ in RUNS:
        lam,flu=D[t]; sm=(lam>=lo)&(lam<=hi)
        Im=float(np.trapezoid(flu[sm],lam[sm]))
        row+=f"{Im/Io:>26.3f}"
    print(row)

fig,axes=plt.subplots(2,1,figsize=(13,9),gridspec_kw={"height_ratios":[3,2]})
ax=axes[0]; ax.plot(olam,oflu,lw=1.0,color="black",alpha=0.85,label="SN 2002bo dereddened")
for t,lbl,c in RUNS:
    lam,flu=D[t]; ax.plot(lam,flu,lw=1.0,color=c,alpha=0.85,label=lbl)
ax.set_xlim(1500,10500); ax.set_ylabel("F_lambda"); ax.legend(fontsize=10); ax.grid(True,alpha=0.25)
ax.set_title("macroatom cap=200 smoke vs scatter baseline — SN 2002bo B-max [4000,6000]A anchor")
ax=axes[1]; ax.axhline(1.0,lw=1.0,color="black",ls="--",alpha=0.6)
for t,lbl,c in RUNS:
    lam,flu=D[t]; cm=(olam>=lam[0])&(olam<=lam[-1])
    r=np.interp(olam[cm],lam,flu)/np.maximum(oflu[cm],1e-30)
    ax.semilogy(olam[cm],r,lw=0.9,color=c,alpha=0.85,label=f"{lbl}/obs")
ax.set_xlim(1500,10500); ax.set_ylim(0.05,50); ax.set_xlabel("Wavelength [A]"); ax.set_ylabel("model/obs")
ax.legend(fontsize=9); ax.grid(True,alpha=0.25,which="both")
plt.tight_layout()
out1=f"{ROOT}/figures/2026-05-31_macroatom_cap200_vs_scatter_sn2002bo_bmax.png"
plt.savefig(out1,dpi=130); print(f"saved: {out1}")

panels=[("P2 UV/blue [1800,4400]",1800,4400),
        ("P2 detail 3700-4400",3700,4400),
        ("P3 Si II 6355 [5700,6500]",5700,6500),
        ("P4 Ca II NIR [7400,8800]",7400,8800)]
fig,axes=plt.subplots(2,2,figsize=(14,9))
for ax,(ttl,lo,hi) in zip(axes.ravel(),panels):
    so=(olam>=lo)&(olam<=hi)
    ax.plot(olam[so],oflu[so],lw=1.3,color="black",alpha=0.85,label="2002bo")
    for t,lbl,c in RUNS:
        lam,flu=D[t]; sm=(lam>=lo)&(lam<=hi)
        ax.plot(lam[sm],flu[sm],lw=1.2,color=c,alpha=0.85,label=lbl)
    ax.set_xlim(lo,hi); ax.set_title(ttl,fontsize=11)
    ax.set_xlabel("Wavelength [A]"); ax.set_ylabel("F_lambda")
    ax.legend(fontsize=8); ax.grid(True,alpha=0.25)
fig.suptitle("macroatom cap=200 smoke vs scatter — user panels (P2 UV/blue, P3 Si6355, P4 CaNIR)",fontsize=13)
plt.tight_layout()
out2=f"{ROOT}/figures/2026-05-31_macroatom_cap200_vs_scatter_zoom_sn2002bo_bmax.png"
plt.savefig(out2,dpi=130); print(f"saved: {out2}")
