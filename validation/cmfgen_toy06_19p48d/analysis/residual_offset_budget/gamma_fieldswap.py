#!/usr/bin/env python3
"""gamma_fieldswap.py -- Wien-shadow hypothesis test for kpr4.

Compute Gamma(FeIII->IV) (and CoIII->IV) at s0/s2/s4/s6/s8 with THREE fields on
the identical 1000-bin log-nu grid (numin 1.5e14, numax 3e16):
  (A) kpr4 mc_J  (its own coevolve field; Gph consumer field)
  (B) kpr4 cs_J  (the pump/scatter deterministic field)
  (C) CMFGEN jtable (data/cmfgen_jtable_toy06_19p48d.bin) -- ground truth field

Decompose Gamma by band and isolate the FIELD effect: r=G/(ne*alpha), f_up=r/(1+r).
alpha (Milne) is field-independent, so r(fieldX)/r(fieldY)=G(X)/G(Y) at fixed Te,pops
-> immune to the alpha calibration. Predict f_up under CMFGEN field from kpr4's OWN
measured f(FeIV) (ion_pops) times the G-ratio.

Read-only. Every number reproducible from committed data.
"""
import os, sys, struct, math
import numpy as np

REPO='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
os.environ['LUMINA_REF_DIR']=f'{REPO}/data/tardis_reference_toy06_19p48d'
os.environ['LUMINA_SIGMA_BIN']=f'{REPO}/data/tardis_reference_toy06_19p48d/cmfgen_sigma_bf.bin'
os.environ['LUMINA_SIGMA_COIII_PATCH']=f'{REPO}/data/coiii_real_sigma_patch.npz'
sys.path.insert(0,f'{REPO}/scripts')
import db_photoion_calc as dbp

RUN=f'{REPO}/logs/coevolve_consume_a10_kx_kpr4'
NFB=1000
H=dbp.H; KB=dbp.KB; C=dbp.C; PI=dbp.PI; EV=dbp.EV; ME=dbp.ME; KB_EV=dbp.KB_EV
nu_c=dbp.nu_c; dnu=dbp.dnu
lam_c=2.99792458e18/nu_c   # Angstrom

SHELLS=[0,2,4,6,8]
CMF_TE={0:18760.22,2:16351.43,4:13657.24,6:11929.14,8:10383.0}

# ---- load kpr4 fields (mc_J col4, cs_J col3) into (50,1000) ----
mcJ=np.full((50,NFB),0.0); csJ=np.full((50,NFB),0.0)
with open(f'{RUN}/lumina_coevolve_field.csv') as f:
    next(f)
    for line in f:
        p=line.split(',')
        s=int(p[0]); b=int(p[1]); csJ[s,b]=float(p[3]); mcJ[s,b]=float(p[4])
# field the Gph loop actually consumes: mc_J if >0 else cs_J (db_photoion.field())
gphJ=np.where(mcJ>0,mcJ,csJ)

# ---- CMFGEN jtable ----
with open(f'{REPO}/data/cmfgen_jtable_toy06_19p48d.bin','rb') as fh:
    mg,ver,nsh,nfbj=struct.unpack('<4i',fh.read(16))
    jt=np.frombuffer(fh.read(),np.float64).reshape(nsh,nfbj)
jt=np.where(jt>0,jt,0.0)

# ---- band masks on the shared grid ----
# Fe III ground edge chi=30.65 eV -> 404.5 A. bands by wavelength:
band={
 'EUV_FeIIIedge(<=405A)': lam_c<=405.0,
 '300-450A'             : (lam_c>=300.0)&(lam_c<=450.0),
 'FUV(912-2000A)'       : (lam_c>=912.0)&(lam_c<=2000.0),
 'opt+(>2000A)'         : lam_c>2000.0,
}

def band_avg(J,mask):
    v=J[mask]; v=v[v>0]
    return float(np.exp(np.mean(np.log(v)))) if v.size else 0.0

def gamma(J,Te,ne,Z,ion,pops):
    """G three ways + alpha(Milne) + band-decomposed G_nlte. Mirrors db_photoion.analyze."""
    chi0=dbp.CHI[(Z,ion)]; kT=KB*Te
    idx=np.where((dbp.levZ==Z)&(dbp.levI==ion))[0]
    x=dbp.levE[idx]/(KB_EV*Te)
    U=float(np.sum(np.where(x<50,dbp.levG[idx]*np.exp(-np.minimum(x,50)),0.0)))
    idxu=np.where((dbp.levZ==Z)&(dbp.levI==ion+1))[0]
    xu=dbp.levE[idxu]/(KB_EV*Te)
    Uu=float(np.sum(np.where(xu<50,dbp.levG[idxu]*np.exp(-np.minimum(xu,50)),0.0)))
    if Uu<1: Uu=max(1.0,dbp.levG[idxu[0]] if len(idxu) else 1.0)
    lam3=(H*H/(2*PI*ME*KB*Te))**1.5
    ntot=sum(v[0] for v in pops.values())
    G_gnd=G_b=G_n=alpha=0.0
    Gband={k:0.0 for k in band}     # band decomposition of G_nlte
    for gl in idx:
        Rb,chi_l=dbp.R_planck(gl,Te,chi0)
        if chi_l>0 and dbp.flags[gl]:
            alpha+=Rb*lam3*dbp.levG[gl]/(2*Uu)*math.exp(min(chi_l/kT,300))
        # field-driven R + per-band decomposition
        E=dbp.levE[gl]; chil=(chi0-E)*EV
        if chil<=0 or not dbp.flags[gl]: continue
        nu_th=chil/H
        m=(nu_c>=nu_th)&(dbp.SIG[gl]>0)&(J>0)
        if not m.any(): continue
        integ=4*PI*dbp.SIG[gl][m]*J[m]/(H*nu_c[m])*dnu[m]
        R=float(np.sum(integ))
        if R<=0: continue
        n=dbp.levN[gl]; xl=E/(KB_EV*Te)
        if xl>=50: continue
        pb=dbp.levG[gl]*math.exp(-xl)/U
        nk=pops.get(n,(0.0,))[0]; pn=nk/ntot if ntot>0 else 0.0
        if n==0: G_gnd+=R
        G_b+=pb*R; G_n+=pn*R
        # band split of the NLTE-weighted contribution
        for k,bmask in band.items():
            Gband[k]+=pn*float(np.sum(integ[bmask[m]]))
    return dict(G_gnd=G_gnd,G_b=G_b,G_n=G_n,alpha=alpha,ne=ne,U=U,Uu=Uu,Gband=Gband)

def f_up(r): return r/(1.0+r)

# ---- kpr4 actual ion fractions from ion_pops ----
ionpop={}
with open(f'{RUN}/lumina_ion_pops.csv') as f:
    next(f)
    for line in f:
        s,Z,st,n=line.split(','); ionpop[(int(s),int(Z),int(st))]=float(n)
def actual_f(s,Z,st):
    tot=sum(ionpop.get((s,Z,k),0.0) for k in range(7))
    return ionpop.get((s,Z,st),0.0)/tot if tot>0 else 0.0
def actual_r(s,Z,st_lo):  # n(lo+1)/n(lo)
    return ionpop.get((s,Z,st_lo+1),0.0)/max(ionpop.get((s,Z,st_lo),0.0),1e-300)

st=dbp  # plasma reader
def plasma(s):
    return dbp.plasma(RUN,s)   # (Te,ne)

print("="*118)
print("FIELD band-averages (geometric mean) : kpr4 mc_J / kpr4 cs_J / kpr4 gphJ(mc||cs) / CMFGEN jtable  + excess ratio gphJ/CMFGEN")
print("="*118)
for s in SHELLS:
    print(f"\n-- s{s}  (Te_kpr4={plasma(s)[0]:.0f}  Te_cmf={CMF_TE[s]:.0f}) --")
    print(f"   {'band':>22} {'mc_J':>11} {'cs_J':>11} {'gphJ':>11} {'CMFGEN':>11} {'gphJ/CMF':>10}")
    for k,bmask in band.items():
        a_mc=band_avg(mcJ[s],bmask); a_cs=band_avg(csJ[s],bmask)
        a_g=band_avg(gphJ[s],bmask); a_c=band_avg(jt[s],bmask)
        rr=a_g/a_c if a_c>0 else float('inf')
        print(f"   {k:>22} {a_mc:>11.3e} {a_cs:>11.3e} {a_g:>11.3e} {a_c:>11.3e} {rr:>10.2e}")

for (Z,ion,st_lo,name) in [(26,2,2,'FeIII->IV'),(27,2,2,'CoIII->IV')]:
    print("\n"+"="*118)
    print(f"GAMMA & f_up : {name}   (pops=kpr4 NLTE levelpops; Te,ne=kpr4; field swapped)")
    print("  NOTE CoIII sigma may be Kramers-patched; Fe III is real CMFGEN sigma.")
    print("="*118)
    hdr=f"{'shell':>5} {'Te':>7} {'ne':>9} {'f_act(IV)':>9} {'r_act':>9} | "\
        f"{'G(mcJ)':>9} {'G(gphJ)':>9} {'G(CMF)':>9} {'alpha':>9} | "\
        f"{'r_gph':>9} {'f_gph':>7} {'r_CMF':>9} {'f_CMF':>7} | {'Gcmf/Ggph':>9} {'f_pred_CMF':>10}"
    print(hdr)
    for s in SHELLS:
        Te,ne=plasma(s)
        pops=dbp.levelpops(RUN,Z,ion,s)
        gA=gamma(mcJ[s],Te,ne,Z,ion,pops)
        gG=gamma(gphJ[s],Te,ne,Z,ion,pops)
        gC=gamma(jt[s],Te,ne,Z,ion,pops)
        alpha=gG['alpha']
        r_gph=gG['G_n']/(ne*alpha) if alpha>0 else float('nan')
        r_cmf=gC['G_n']/(ne*alpha) if alpha>0 else float('nan')
        f_act=actual_f(s,Z,st_lo+1); r_act=actual_r(s,Z,st_lo)
        Gratio=gC['G_n']/gG['G_n'] if gG['G_n']>0 else float('nan')
        r_pred=r_act*Gratio
        print(f"{s:>5} {Te:>7.0f} {ne:>9.2e} {f_act:>9.4f} {r_act:>9.2e} | "
              f"{gA['G_n']:>9.2e} {gG['G_n']:>9.2e} {gC['G_n']:>9.2e} {alpha:>9.2e} | "
              f"{r_gph:>9.2e} {f_up(r_gph):>7.4f} {r_cmf:>9.2e} {f_up(r_cmf):>7.4f} | "
              f"{Gratio:>9.2e} {f_up(r_pred):>10.4f}")
    # band decomposition of G_nlte for the gph field vs CMFGEN, Fe only
    if Z==26:
        print(f"\n  -- band decomposition of G_nlte({name}) : gphJ vs CMFGEN --")
        print(f"  {'shell':>5} {'field':>7} " + " ".join(f"{k[:16]:>16}" for k in band) + f"  {'sum':>10}")
        for s in SHELLS:
            Te,ne=plasma(s); pops=dbp.levelpops(RUN,Z,ion,s)
            for tag,J in (('gphJ',gphJ[s]),('CMFGEN',jt[s])):
                g=gamma(J,Te,ne,Z,ion,pops)
                print(f"  {s:>5} {tag:>7} "+" ".join(f"{g['Gband'][k]:>16.3e}" for k in band)+f"  {g['G_n']:>10.3e}")
