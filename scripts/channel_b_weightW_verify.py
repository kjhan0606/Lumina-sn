#!/usr/bin/env python3
"""Channel-(b) verification: how much does the dilute weight=W in tau_sobolev
(plasma.c:704-709) DIRECTLY inflate the [3500-4500] over-absorption?

For each (Z,ion) in a wavelength window, compute the opacity-faithful Sobolev
Sum_tau two ways on a converged NLTE dump:
  (1) actual  : non-metastable lower levels carry weight=W (per-shell W from dump)
  (2) W=1     : counterfactual with W:=1 everywhere (numerator AND Z_part)
ratio = (1)/(2) = NET channel-(b) inflation (Z_part partially compensates).

Also splits window tau by metastable vs non-metastable lower level: only the
non-metastable part feels W, so a high non-meta fraction = channel (b) operative.

Usage: channel_b_weightW_verify.py <run_dir> <iter> <lam_lo> <lam_hi> [shells]
"""
import sys, os, csv
from collections import defaultdict
import numpy as np
import pandas as pd

RUN=sys.argv[1]; ITER=sys.argv[2]
LAM_LO=float(sys.argv[3]); LAM_HI=float(sys.argv[4])
SHELLS=[int(x) for x in sys.argv[5].split(",")] if len(sys.argv)>5 else [0,5,15]

EC=4.8032068e-10; ME=9.1093837e-28; C_CGS=2.99792458e10
SIGMA_PRE=np.pi*EC*EC/(ME*C_CGS); KB_EV=8.617333262e-5
ELEM={6:"C",8:"O",12:"Mg",13:"Al",14:"Si",16:"S",20:"Ca",21:"Sc",22:"Ti",
      24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni"}
ROMAN={0:"I",1:"II",2:"III",3:"IV",4:"V"}

ref=os.path.join(RUN,"ref")
import json
t_exp=json.load(open(os.path.join(ref,"config.json")))["time_explosion_s"]

shell_TW={}; nion={}
with open(os.path.join(RUN,f"nlte_levels_iter{ITER}.csv")) as f:
    r=csv.reader(f); next(r)
    for row in r:
        Z=int(row[0]);ion=int(row[1]);s=int(row[2])
        Te=float(row[8]);Tr=float(row[9]);W=float(row[10]);nt=float(row[11])
        shell_TW[s]=(Te,Tr,W); nion[(Z,ion,s)]=nt

lev=pd.read_csv(os.path.join(ref,"levels.csv"))
lev_g={}
for (Z,ion),grp in lev.groupby(["atomic_number","ion_number"]):
    lev_g[(Z,ion)]={int(r.level_number):(float(r.energy_eV),float(r.g),int(r.metastable))
                    for r in grp.itertuples()}

_ZP={}
def zpart(Z,ion,Tr,W):
    key=(Z,ion,round(Tr,2),round(W,4))
    if key in _ZP: return _ZP[key]
    d=lev_g.get((Z,ion))
    if not d: return None
    z=0.0
    for (E,g,meta) in d.values():
        b=E/(KB_EV*Tr)
        if b>500: continue
        z+=(1.0 if meta else W)*g*np.exp(-b)
    z=max(z,1e-300); _ZP[key]=z; return z

def n_lower(Z,ion,s,lvl,Tr,W):
    d=lev_g.get((Z,ion))
    if not d or lvl not in d: return 0.0,1
    E,g,meta=d[lvl]; nt=nion.get((Z,ion,s),0.0)
    if nt<=0: return 0.0,meta
    b=E/(KB_EV*Tr)
    if b>500: return 0.0,meta
    zp=zpart(Z,ion,Tr,W)
    return nt*(1.0 if meta else W)*g*np.exp(-b)/zp, meta

ll=pd.read_csv(os.path.join(ref,"line_list.csv"))
sel=(ll.wavelength>=LAM_LO)&(ll.wavelength<=LAM_HI)
lw=ll[sel].reset_index(drop=True)
print(f"# run={os.path.basename(RUN)} iter{ITER} window=[{LAM_LO:.0f},{LAM_HI:.0f}]A lines={len(lw):,}")
lam_cm=lw.wavelength.values*1e-8; flu=lw.f_lu.values
Zc=lw.atomic_number.values.astype(int); ionc=lw.ion_number.values.astype(int)
lloc=lw.level_number_lower.values.astype(int)

for s in SHELLS:
    Te,Tr,W=shell_TW.get(s,(0,0,0))
    tauW=defaultdict(float); tau1=defaultdict(float)
    tau_meta=defaultdict(float); tau_nonmeta=defaultdict(float)
    for i in range(len(lw)):
        nlo_W,meta=n_lower(Zc[i],ionc[i],s,lloc[i],Tr,W)
        if nlo_W<=0: continue
        pre=SIGMA_PRE*flu[i]*lam_cm[i]*t_exp
        tW=pre*nlo_W
        nlo_1,_=n_lower(Zc[i],ionc[i],s,lloc[i],Tr,1.0)
        t1=pre*nlo_1
        k=(Zc[i],ionc[i])
        tauW[k]+=tW; tau1[k]+=t1
        if meta: tau_meta[k]+=tW
        else: tau_nonmeta[k]+=tW
    totW=sum(tauW.values()); tot1=sum(tau1.values())
    print(f"\n=== shell {s}: T_e={Te:.0f} T_rad={Tr:.0f} W={W:.3f}  "
          f"Sum_tau(W)={totW:.3e}  Sum_tau(W=1)={tot1:.3e}  NET inflate x{totW/max(tot1,1e-300):.2f} ===")
    print(f"  {'ion':9s}{'frac%':>7s}{'tauW':>11s}{'tau(W=1)':>11s}{'x_infl':>8s}{'nonmeta%':>9s}")
    for (Z,ion),tv in sorted(tauW.items(),key=lambda kv:-kv[1])[:10]:
        nm=tau_nonmeta[(Z,ion)]; t1=tau1[(Z,ion)]
        name=f"{ELEM.get(Z,Z)} {ROMAN.get(ion,ion+1)}"
        print(f"  {name:9s}{100*tv/max(totW,1e-300):6.1f}%{tv:11.2e}{t1:11.2e}"
              f"{tv/max(t1,1e-300):8.2f}{100*nm/max(tv,1e-300):8.1f}%")
