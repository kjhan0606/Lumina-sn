#!/usr/bin/env python3
"""Bake REAL Co III sigma_bf from the CMFGEN term-grouped phot file that toy06 links.

Co III is Kramers (nu^-3) in every baked bin because carsus never matched its
'Very crude', 386-entry, term-grouped phot_data (Split J levels=False).  But the
term->level mapping is 100% clean (osc level name minus [J] == phot config).  So
we bake the real cross-sections directly here.

CMFGEN Type-1 Seaton form (newsubs/sub_phot_gen.f:411, CROSS_TYPE==1):
    RU = EDGE/nu                          # = nu_threshold/nu, per LEVEL's own edge
    sigma(nu) = CONV_FAC * A0 * ( A1 + (1-A1)*RU ) * RU^A2
with the file's params [A0=sigma_0(Mb), A1=1.0, A2=2.0] this is sigma_0*(nu_th/nu)^2.
Unit is Megabarns (file '!Cross-section unit'), so *1e-18 -> cm^2 (identical to the
carsus type-1 path scripts/bake_cmfgen_sigma_bf_carsus.py:146-151 that baked FeIII).

Output: an npz side-file with 1000 Co III rows on the SAME 1000-bin log-nu grid the
sigma_bf.bin uses, plus a matched-flag per level.  Overlaid at runtime by
db_photoion_calc.py when LUMINA_SIGMA_COIII_PATCH points at it.
"""
import re, csv, math, sys
import numpy as np

OSC  = "/gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/18oct00/coiii_osc.dat"
PHOT = "/gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/18oct00/phot_data.dat"
LEVELS = "data/tardis_reference_toy06_19p48d/levels.csv"
IONIZ  = "data/tardis_reference_toy06_19p48d/ionization_energies.csv"
OUT    = "data/coiii_real_sigma_patch.npz"

H=6.62607015e-27; EV=1.602176634e-12
# grid identical to the sigma_bf.bin (db_photoion_calc)
NUMIN, NUMAX, NFREQ = 1.5e14, 3.0e16, 1000
lnmin=math.log(NUMIN); dlog=(math.log(NUMAX)-lnmin)/NFREQ
nu_c=np.exp(lnmin+(np.arange(NFREQ)+0.5)*dlog)

# ---- phot terms ----
pl=open(PHOT).read().split("\n")
for k,ln in enumerate(pl):
    if "Total number of data pairs" in ln: start=k+1; break
photnorm={}; k=start
while k<len(pl):
    if "!Configuration name" in pl[k]:
        name=pl[k].split()[0]; npts=int(pl[k+2].split()[0])
        params=[float(pl[k+3+j].split()[0].replace("D","E").replace("d","e")) for j in range(npts)]
        photnorm[name.strip().lower()]=params
        k=k+3+npts
    else: k+=1
print(f"[phot] terms={len(photnorm)}")

# ---- osc level names (order = levels.csv level_number) ----
ol=open(OSC).read().split("\n")
for j,ln in enumerate(ol):
    m=re.match(r"\s*(\d+)\s+!Number of energy levels",ln)
    if m: nlev=int(m.group(1)); he=j; break
TERMRE=re.compile(r"\[[^\[\]]*\]\s*$"); numre=re.compile(r"^[-+]?\d*\.?\d+$")
names=[]
for ln in ol[he+1:]:
    s=ln.split()
    if len(s)<7: continue
    try: float(s[1]); float(s[2]); int(float(s[6]))
    except: continue
    if numre.match(s[0]): continue
    names.append(s[0])
    if len(names)>=nlev: break
print(f"[osc] levels={len(names)}")

# ---- CoIII level energies (levels.csv) + chi0 ----
E={}
for r in csv.DictReader(open(LEVELS)):
    if int(r["atomic_number"])==27 and int(r["ion_number"])==2:
        E[int(r["level_number"])]=float(r["energy_eV"])
chi0=None
for r in csv.DictReader(open(IONIZ)):
    if int(r["atomic_number"])==27 and int(r["ion_number"])==2:
        chi0=float(r["ionization_energy_eV"])
print(f"[chi0] CoIII = {chi0} eV")

# ---- bake ----
N=min(len(names), 1000)
SIG=np.zeros((1000, NFREQ), dtype="<f8")
matched=np.zeros(1000, dtype=np.int8)
n_nothr=0
for kk in range(N):
    term=TERMRE.sub("",names[kk]).strip().lower()
    p=photnorm.get(term)
    if p is None: continue
    s0,a1,a2=p[0],p[1],p[2]
    chi_l=chi0 - E[kk]
    if chi_l<=0: n_nothr+=1; continue
    nu_th=chi_l*EV/H
    RU=nu_th/nu_c
    m=nu_c>=nu_th
    sig=np.zeros(NFREQ)
    sig[m]=s0*(a1+(1.0-a1)*RU[m])*(RU[m]**a2)*1e-18
    SIG[kk]=sig
    matched[kk]=1
print(f"[bake] matched {int(matched.sum())}/1000 levels; {n_nothr} with chi_l<=0 (above ioniz)")

np.savez(OUT, sigma=SIG, matched=matched, level_number=np.arange(1000),
         numin=NUMIN, numax=NUMAX, nfreq=NFREQ, Z=27, ion=2)
print(f"[out] {OUT}  (sigma {SIG.shape}, {int(matched.sum())} matched)")

# ---- quick provenance check ----
g0=SIG[0]; nz=np.where(g0>0)[0]
if len(nz)>3:
    sl=np.polyfit(np.log(nu_c[nz]),np.log(g0[nz]),1)[0]
    print(f"[check] ground(level0) slope={sl:+.3f} (expect ~-2, NOT Kramers -3), "
          f"sigma_thr={g0[nz[0]]:.3e} cm^2")
