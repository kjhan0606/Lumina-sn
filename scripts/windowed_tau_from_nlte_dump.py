#!/usr/bin/env python3
"""Windowed per-(Z,ion) Sobolev tau attribution from an NLTE level dump.

Unlike per_ion_tau_attr_si2_6355.py (locked to the 8-element TARDIS schema and
nebular Z_part), this reads the ACTUAL NLTE level populations (n_pop) written by
LUMINA_NLTE_LEVEL_DUMP, so it covers every element in the dump (incl. the near-UV
blanketers Sc/Ti/Cr/Mg/Al) and needs no partition-function reconstruction.

  tau_Sob(line, shell) = (pi e^2 / m_e c) * f_lu * lam_rest * t_exp * n_pop(lower)

n_pop(lower) is looked up directly from the dump at (Z, ion, shell, level_lower).

Usage: windowed_tau_from_nlte_dump.py <run_dir> <iter> <lam_lo> <lam_hi> [shells]
"""
import sys, os, csv
from collections import defaultdict
import numpy as np
import pandas as pd

RUN = sys.argv[1]
ITER = sys.argv[2]
LAM_LO = float(sys.argv[3])
LAM_HI = float(sys.argv[4])
SHELLS = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 5, 15]

EC=4.8032068e-10; ME=9.1093837e-28; C_CGS=2.99792458e10
SIGMA_PRE = np.pi*EC*EC/(ME*C_CGS)

ELEM={6:"C",8:"O",12:"Mg",13:"Al",14:"Si",16:"S",20:"Ca",21:"Sc",22:"Ti",
      24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni"}
ROMAN={0:"I",1:"II",2:"III",3:"IV",4:"V"}

ref = os.path.join(RUN, "ref")
import json
cfg = json.load(open(os.path.join(ref,"config.json")))
t_exp = cfg["time_explosion_s"]
KB_EV = 8.617333262e-5

# --- load NLTE dump: per-shell T_e/T_rad/W, n_ion_total (nebular, feeds opacity) ---
shell_TW = {}
nion = {}
with open(os.path.join(RUN, f"nlte_levels_iter{ITER}.csv")) as f:
    r = csv.reader(f); next(r)
    for row in r:
        Z=int(row[0]); ion=int(row[1]); s=int(row[2]); lvl=int(row[3])
        Te=float(row[8]); Tr=float(row[9]); W=float(row[10]); nt=float(row[11])
        shell_TW[s] = (Te,Tr,W)
        nion[(Z,ion,s)] = nt

# --- levels.csv -> per (Z,ion) arrays of (level,E_eV,g,metastable) for Z_part + n_lower ---
lev = pd.read_csv(os.path.join(ref,"levels.csv"))
lev_g = {}      # (Z,ion) -> dict[level]=(E,g,meta)
for (Z,ion),grp in lev.groupby(["atomic_number","ion_number"]):
    lev_g[(Z,ion)] = {int(r.level_number):(float(r.energy_eV),float(r.g),int(r.metastable))
                      for r in grp.itertuples()}

_ZP = {}
def zpart(Z,ion,Tr,W):
    key=(Z,ion,round(Tr,2),round(W,4))
    if key in _ZP: return _ZP[key]
    d = lev_g.get((Z,ion))
    if not d: return None
    z=0.0
    for (E,g,meta) in d.values():
        b = E/(KB_EV*Tr)
        if b>500: continue
        z += (1.0 if meta else W)*g*np.exp(-b)
    z=max(z,1e-300); _ZP[key]=z; return z

# nebular n_lower(line) = n_ion * weight * g_lo * exp(-E_lo/kTr) / Z_part  (opacity-faithful)
def n_lower_neb(Z,ion,s,lvl,Tr,W):
    d = lev_g.get((Z,ion))
    if not d or lvl not in d: return 0.0
    E,g,meta = d[lvl]
    nt = nion.get((Z,ion,s),0.0)
    if nt<=0: return 0.0
    b = E/(KB_EV*Tr)
    if b>500: return 0.0
    zp = zpart(Z,ion,Tr,W)
    return nt*(1.0 if meta else W)*g*np.exp(-b)/zp

# --- line list in window ---
ll = pd.read_csv(os.path.join(ref,"line_list.csv"))
sel = (ll.wavelength>=LAM_LO)&(ll.wavelength<=LAM_HI)
lw = ll[sel].reset_index(drop=True)
print(f"# run={os.path.basename(RUN)} iter{ITER}  window=[{LAM_LO:.0f},{LAM_HI:.0f}]A  lines={len(lw):,}")

lam_cm = lw.wavelength.values*1e-8
flu = lw.f_lu.values
Zc = lw.atomic_number.values.astype(int)
ionc = lw.ion_number.values.astype(int)
lloc = lw.level_number_lower.values.astype(int)

for s in SHELLS:
    Te,Tr,W = shell_TW.get(s,(0,0,0))
    tau_by_ion = defaultdict(float)
    nthick = defaultdict(int)
    top_line = {}  # (Z,ion)->(tau,lam,flu)
    zp_cache = {}
    for i in range(len(lw)):
        n_lo = n_lower_neb(Zc[i],ionc[i],s,lloc[i],Tr,W)
        if n_lo<=0: continue
        tau = SIGMA_PRE*flu[i]*lam_cm[i]*t_exp*n_lo
        k=(Zc[i],ionc[i])
        tau_by_ion[k]+=tau
        if tau>1.0: nthick[k]+=1
        if k not in top_line or tau>top_line[k][0]:
            top_line[k]=(tau,lw.wavelength.values[i],flu[i])
    total=sum(tau_by_ion.values())
    print(f"\n=== shell {s}: T_e={Te:.0f} T_rad={Tr:.0f} W={W:.3f}  Sum_tau={total:.3e} ===")
    print(f"  {'ion':9s}{'frac%':>7s}{'Sum_tau':>11s}{'n_thick':>8s}{'tau_top':>10s}{'lam_top':>9s}{'flu_top':>9s}{'f_ion_II':>9s}")
    items=sorted(tau_by_ion.items(),key=lambda kv:-kv[1])
    for (Z,ion),tv in items[:12]:
        name=f"{ELEM.get(Z,Z)} {ROMAN.get(ion,ion+1)}"
        tt,tl,tf=top_line[(Z,ion)]
        # ion fraction II vs III for this element at this shell
        nII=nion.get((Z,1,s),0.0); nIII=nion.get((Z,2,s),0.0)
        fII = nII/(nII+nIII) if (nII+nIII)>0 else float('nan')
        print(f"  {name:9s}{100*tv/max(total,1e-300):6.1f}%{tv:11.2e}{nthick[(Z,ion)]:8d}{tt:10.2e}{tl:9.1f}{tf:9.2e}{fII:9.3f}")
