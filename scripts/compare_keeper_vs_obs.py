#!/usr/bin/env python3
"""Full-range overlay + per-band ratio table: LUMINA keeper vs dereddened SN 2002bo."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn")
EBV, RV = 0.41, 3.1; A_V = RV * EBV
JOB = sys.argv[1] if len(sys.argv) > 1 else "161737"
LABEL = sys.argv[2] if len(sys.argv) > 2 else f"keeper {JOB} (scatter, radeq-OFF)"

def ccm(w):
    x = 1e4/w; a=np.zeros_like(x); b=np.zeros_like(x)
    s=(x>=1.1)&(x<=3.3); y=x[s]-1.82
    a[s]=1+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
    b[s]=1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
    s=(x>=0.3)&(x<1.1); a[s]=0.574*x[s]**1.61; b[s]=-0.527*x[s]**1.61
    s=(x>3.3)&(x<=8.0); xs=x[s]
    Fa=np.where(xs>=5.9,-0.04473*(xs-5.9)**2-0.009779*(xs-5.9)**3,0.0)
    Fb=np.where(xs>=5.9, 0.2130*(xs-5.9)**2+0.1207*(xs-5.9)**3,0.0)
    a[s]=1.752-0.316*xs-0.104/((xs-4.67)**2+0.341)+Fa
    b[s]=-3.090+1.825*xs+1.206/((xs-4.62)**2+0.263)+Fb
    return a+b/RV

obs = pd.read_csv(ROOT/"data/sn2002bo/epochs/sn2002bo_m0d0.csv", comment="#")
olam = obs["wavelength_angstrom"].values
oflu = obs["flux_erg_s_cm2_angstrom"].values * 10**(0.4*A_V*ccm(olam))

wd = sorted(ROOT.glob(f"logs/*_{JOB}"))[-1]
m = pd.read_csv(wd/"lumina_spectrum.csv")
lam = m["wavelength_angstrom"].values; flu = m["flux"].values
sa=(lam>=4000)&(lam<=6000); oa=(olam>=4000)&(olam<=6000)
K = np.trapezoid(oflu[oa],olam[oa])/np.trapezoid(flu[sa],lam[sa])
flu = flu*K

BANDS=[("UV <3400",1000,3400),("blue 3400-4500",3400,4500),
       ("opt 4500-5600",4500,5600),("red 5600-7000",5600,7000),
       ("NIR I 7000-9000",7000,9000),("NIR II 9000-12000",9000,12000)]
def band(lo,hi):
    sm=(lam>=lo)&(lam<=hi); so=(olam>=lo)&(olam<=hi)
    if sm.sum()<2 or so.sum()<2: return np.nan
    return np.trapezoid(flu[sm],lam[sm])/np.trapezoid(oflu[so],olam[so])

print(f"# {LABEL}  vs SN 2002bo (dereddened, B-max), optical[4000,6000]-anchored")
print(f"{'band':22s} {'model/obs':>10s}")
for nm,lo,hi in BANDS:
    print(f"{nm:22s} {band(lo,hi):10.2f}")

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(13,8),sharex=False)
ax1.plot(olam,oflu,color="black",lw=1.0,label="SN 2002bo (dered, B-max)")
ax1.plot(lam,flu,color="crimson",lw=1.0,alpha=0.85,label=LABEL)
ax1.set_xlim(1500,12000); ax1.set_ylim(0,None)
ax1.set_xlabel("wavelength (Å)"); ax1.set_ylabel("F_λ (opt-anchored)")
ax1.legend(fontsize=9); ax1.set_title("Full range")
ax2.plot(olam,oflu,color="black",lw=1.1,label="SN 2002bo")
ax2.plot(lam,flu,color="crimson",lw=1.1,alpha=0.85,label=LABEL)
for w in (4896,5307,5474): ax2.axvline(w,color="gray",ls=":",lw=0.7)
ax2.set_xlim(3500,7000); ax2.set_ylim(0,None)
ax2.set_xlabel("wavelength (Å)"); ax2.set_ylabel("F_λ"); ax2.legend(fontsize=9)
ax2.set_title("Optical zoom (W-trough markers 4896/5307/5474)")
out=ROOT/f"figures/2026-06-02_compare_{JOB}_vs_sn2002bo.png"
out.parent.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(out,dpi=130)
print("saved:",out)
