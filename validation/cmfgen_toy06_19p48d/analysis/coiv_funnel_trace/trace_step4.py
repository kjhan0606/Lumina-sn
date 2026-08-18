#!/usr/bin/env python3
"""Step 4 — entry side + Fe III control.
(a) Sobolev-tau spectrum at s0 by ion in FUV(918-1290) and PILE(1290-2000),
    nebular pops (consistent w/ compute_tau_sobolev non-NLTE path) -> which ion
    dominates a-priori line opacity. (Authoritative *realized* absorption is the
    event ledger; this is the opacity cross-check.)
(b) Atomic-structure control: downward emission lines available from a strong
    Fe III UV upper level vs Co IV level 144 -> does Fe III have OPTICAL exits
    (fluorescence UV->optical) that Co IV (all-UV) lacks?
"""
import numpy as np, csv, os
from collections import defaultdict
REF="data/tardis_reference_toy06_19p48d"; LOG="logs/coevolve_consume_a10_kx_gphall"
OUT="validation/cmfgen_toy06_19p48d/analysis/coiv_funnel_trace"
SOBOLEV_COEFF=2.6540281e-02; K_B=1.380649e-16; EV=1.602176634e-12
C=2.99792458e10; t_exp=1.683072e6
W=0.2978587262; T_rad=10470.093240; beta_rad=1.0/(K_B*T_rad)

# levels
lev={}
levlist=defaultdict(list)
for row in csv.DictReader(open(os.path.join(REF,"levels.csv"))):
    Z=int(row["atomic_number"]); ion=int(row["ion_number"]); n=int(row["level_number"])
    E=float(row["energy_eV"]); g=int(row["g"]); m=int(row["metastable"])
    lev[(Z,ion,n)]=(E,g,m); levlist[(Z,ion)].append((E,g,m))
# nebular partition per ion at s0
Zpart={}
for key,arr in levlist.items():
    tot=0.0
    for (E,g,m) in arr:
        b=E*EV*beta_rad
        if b<500: tot+=(1.0 if m else W)*g*np.exp(-b)
    Zpart[key]=tot
# ion pops s0
nion={}
for row in csv.DictReader(open(os.path.join(LOG,"lumina_ion_pops.csv"))):
    if int(row["shell_id"])==0: nion[(int(row["Z"]),int(row["stage"]))]=float(row["n_ion"])
# lines
LL=np.genfromtxt(os.path.join(REF,"line_list.csv"),delimiter=",",names=True)
Z_=LL["atomic_number"].astype(int); ion_=LL["ion_number"].astype(int)
lo_=LL["level_number_lower"].astype(int); up_=LL["level_number_upper"].astype(int)
flu_=LL["f_lu"]; nu_=LL["nu"]; Aul_=LL["A_ul"]; lamcm_=LL["wavelength_cm"]; lam_=C/nu_*1e8
N=len(Z_)

def nebtau(i):
    key=(Z_[i],ion_[i]); ni=nion.get(key,0.0); Zp=Zpart.get(key,0.0)
    if ni<=0 or Zp<=0: return 0.0
    llo=lev.get((Z_[i],ion_[i],lo_[i])); lup=lev.get((Z_[i],ion_[i],up_[i]))
    if llo is None or lup is None: return 0.0
    Elo,glo,mlo=llo; Eup,gup,mup=lup
    blo=Elo*EV*beta_rad
    if blo>=500: return 0.0
    nl=ni*(1.0 if mlo else W)*glo*np.exp(-blo)/Zp
    bup=Eup*EV*beta_rad
    nu2=ni*(1.0 if mup else W)*gup*np.exp(-bup)/Zp if bup<500 else 0.0
    stim=1.0
    if nl>0 and nu2>0:
        stim=1.0-(glo*nu2)/(gup*nl); stim=max(stim,0.0)
    return max(SOBOLEV_COEFF*flu_[i]*lamcm_[i]*t_exp*nl*stim,0.0)

# bands
def sel(lo,hi): return np.where((lam_>=lo)&(lam_<hi))[0]
for label,lo,hi in [("FUV 918-1290",918,1290),("PILE 1290-2000",1290,2000),
                     ("subPILE 1490-1650",1490,1650)]:
    idx=sel(lo,hi)
    agg=defaultdict(lambda:[0.0,0])  # (Z,ion)->[sum(1-e^-tau), count active]
    for i in idx:
        tau=nebtau(i)
        if tau<=0: continue
        agg[(Z_[i],ion_[i])][0]+=(1.0-np.exp(-tau)); agg[(Z_[i],ion_[i])][1]+=1
    tot=sum(v[0] for v in agg.values())
    print("\n=== %s : sum(1-e^-tau) by ion (a-priori line opacity, nebular s0) ==="%label)
    print("  total sum(1-e^-tau) = %.4g over %d active lines"%(tot,sum(v[1] for v in agg.values())))
    top=sorted(agg.items(),key=lambda z:-z[1][0])[:8]
    for (Zk,ik),(sm,cnt) in top:
        print("   Z=%2d ion=%d  sum(1-e^-tau)=%9.4g  frac=%5.1f%%  nlines=%d"%(Zk,ik,sm,100*sm/tot if tot>0 else 0,cnt))

# ---- (b) Fe III control vs Co IV: downward emission structure of a strong UV upper level ----
def downward_emit_lines(Z,ion,upper):
    m=(Z_==Z)&(ion_==ion)&(up_==upper)
    out=[(lam_[i],Aul_[i],lo_[i]) for i in np.where(m)[0]]
    out.sort(key=lambda z:-z[1])
    return out
def band(l):
    if l<912:return "EUV"
    if l<1290:return "FUV"
    if l<2000:return "NUV"
    if l<4500:return "nearUV/blue"
    return "opt/IR"

print("\n\n=== CONTROL: downward emission structure ===")
# Co IV level 144 (upper of 1526)
co=downward_emit_lines(27,3,144)
print("\nCo IV upper level 144 (E=%.2f eV): %d downward lines"%(lev[(27,3,144)][0],len(co)))
bb=defaultdict(lambda:[0.0,0])
for (l,A,dl) in co: bb[band(l)][0]+=A; bb[band(l)][1]+=1
for k in ["EUV","FUV","NUV","nearUV/blue","opt/IR"]:
    print("   %-12s nlines=%3d  sum_A_ul=%.3g"%(k,bb[k][1],bb[k][0]))
print("   strongest 5:", [(round(l,1),("%.2g"%A)) for (l,A,dl) in co[:5]])

# pick strong Fe III UV lines and inspect their upper levels' downward structure
print("\nFe III (Z=26 ion=2) strong UV lines (1290-2000) and their upper-level cascade:")
mfe=(Z_==26)&(ion_==2)&(lam_>=1290)&(lam_<2000)
order=np.argsort(-(Aul_[mfe]*flu_[mfe]))
feidx=np.where(mfe)[0][order][:4]
for i in feidx:
    up=up_[i]
    dl=downward_emit_lines(26,2,up)
    bb=defaultdict(lambda:[0.0,0])
    for (l,A,dd) in dl: bb[band(l)][0]+=A; bb[band(l)][1]+=1
    opt_A=bb["opt/IR"][0]+bb["nearUV/blue"][0]; uv_A=bb["EUV"][0]+bb["FUV"][0]+bb["NUV"][0]
    print("  line lam=%.1f up=%d (E=%.2f eV): %d down-lines | EUV/FUV/NUV sumA=%.3g  blue+opt sumA=%.3g  -> P(opt-ward)~%.2f"%(
        lam_[i],up,lev.get((26,2,up),(0,0,0))[0],len(dl),uv_A,opt_A, opt_A/(uv_A+opt_A) if (uv_A+opt_A)>0 else 0))
    print("     down-line bands:", {k:bb[k][1] for k in ["EUV","FUV","NUV","nearUV/blue","opt/IR"] if bb[k][1]>0})
