#!/usr/bin/env python3
"""Offline reconstruction of the P1/P2 T_e pathologies in the kpr composed-repair run.
Uses ONLY the run's own outputs (field csv, plasma state, ion pops). No GPU/run."""
import csv, math
from collections import defaultdict

RUN="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_kpr"
H=6.62607015e-27; C=2.99792458e10; KB=1.380649e-16
CLA=2.99792458e18  # c in Angstrom/s

# ---- plasma state ----
state={}
with open(f"{RUN}/lumina_plasma_state.csv") as f:
    for r in csv.DictReader(f):
        s=int(r["shell_id"]); state[s]=dict(W=float(r["W"]),Trad=float(r["T_rad"]),
                                            ne=float(r["n_e"]),Te=float(r["T_e"]))

# ---- field: per shell, sum cs_J / mc_J in bands ----
# bands in Angstrom
bands={"EUV_300_450":(300,450),"FUV_918_1290":(918,1290),
       "ion_<912":(0,912),"valley_1650_2100":(1650,2100)}
bandsum=defaultdict(lambda: defaultdict(lambda:[0.0,0.0]))  # [cs,mc]
# also compute sum(J*dnu) with dnu from wavelength bin; approximate dnu per bin
rows=defaultdict(list)
with open(f"{RUN}/lumina_coevolve_field.csv") as f:
    for r in csv.DictReader(f):
        s=int(r["shell"]); lam=float(r["wavelength_A"])
        rows[s].append((lam,float(r["cs_J"]),float(r["mc_J"])))

def band_int(s):
    # integrate J dnu over band (nu = c/lam). rows sorted by bin (descending lam).
    out=defaultdict(lambda:[0.0,0.0])
    rr=sorted(rows[s])  # ascending lambda
    for i,(lam,cs,mc) in enumerate(rr):
        nu=CLA/lam
        # dnu from neighbor spacing
        if i+1<len(rr):
            nu2=CLA/rr[i+1][0]
            dnu=abs(nu-nu2)
        else:
            dnu=abs(nu-CLA/rr[i-1][0])
        for bn,(lo,hi) in bands.items():
            if lo<=lam<hi:
                out[bn][0]+=cs*dnu; out[bn][1]+=mc*dnu
    return out

print("=== Split-field & flood: band-integrated J dnu [erg/s/cm2/sr], mc/cs ratio ===")
print(f"{'s':>2} {'W':>6} {'Te':>7} | "
      +" | ".join(f"{bn}(mc, mc/cs)" for bn in bands))
for s in range(11):
    bi=band_int(s)
    cells=[]
    for bn in bands:
        cs,mc=bi[bn]
        ratio = mc/cs if cs>0 else float('inf')
        cells.append(f"{mc:.2e}({ratio:5.1f})")
    print(f"{s:2d} {state[s]['W']:.4f} {state[s]['Te']:7.0f} | "+" | ".join(cells))

# ---- Planck B(T_e) in ionizing band: shows the self-emission a hot shell dumps ----
def planck_band_int(Te,lo,hi,n=200):
    # integrate B_nu dnu over [lo,hi] Angstrom
    nu_lo=CLA/hi; nu_hi=CLA/lo
    tot=0.0
    for k in range(n):
        nu=nu_lo+(nu_hi-nu_lo)*(k+0.5)/n
        x=H*nu/(KB*Te)
        if x>700: continue
        B=(2*H*nu**3/C**2)/math.expm1(x)
        tot+=B*(nu_hi-nu_lo)/n
    return tot

print("\n=== B(T_e) self-emission in ionizing bands vs a 13kK reference ===")
print(f"{'s':>2} {'Te':>7} {'B_EUV(Te)':>11} {'B_EUV(13k)':>11} {'ratio':>7} "
      f"{'B_FUV(Te)':>11} {'B_FUV(13k)':>11} {'ratio':>7}")
for s in range(11):
    Te=state[s]['Te']
    be=planck_band_int(Te,300,450); be0=planck_band_int(13120,300,450)
    bf=planck_band_int(Te,918,1290); bf0=planck_band_int(13120,918,1290)
    print(f"{s:2d} {Te:7.0f} {be:11.3e} {be0:11.3e} {be/be0:7.1f} "
          f"{bf:11.3e} {bf0:11.3e} {bf/bf0:7.1f}")

# ---- f(FeIV) per shell from ion pops (downstream over-ionization proof) ----
ion=defaultdict(lambda:defaultdict(float))  # ion[s][(Z,stage)]=n
with open(f"{RUN}/lumina_ion_pops.csv") as f:
    for r in csv.DictReader(f):
        s=int(r["shell_id"]); Z=int(r["Z"]); st=int(r["stage"]); ion[s][(Z,st)]=float(r["n_ion"])
def frac_stage(s,Z,stage):
    tot=sum(v for (zz,ss),v in ion[s].items() if zz==Z)
    return (ion[s].get((Z,stage),0.0)/tot) if tot>0 else 0.0
print("\n=== Over-ionization: f(FeIV)=Fe(3)/Fe_tot per shell (Fe stage index: 0=I) ===")
print(f"{'s':>2} {'Te':>7} {'f(FeIII)':>9} {'f(FeIV)':>9} {'f(FeV)':>8}")
for s in range(11):
    print(f"{s:2d} {state[s]['Te']:7.0f} {frac_stage(s,26,2):9.3f} "
          f"{frac_stage(s,26,3):9.3f} {frac_stage(s,26,4):8.3f}")
