#!/usr/bin/env python3
"""
OFFLINE per-level Fe III photoionization (Gph) decomposition at shell s8.

Replicates the EXACT formula the code uses in lumina_plasma.c radeq_simul_all
all-level Gph loop (LUMINA_GPH_ALLLEVEL=1, LUMINA_GPH_SIGMA_CMFGEN=1):

  Gph_perion = Sum_over_FeIII_levels[ pop_l * Sum_bb 4pi*sigma(l,bb)*J(bb)/(h*nu_bb)*dnu_bb ]

with, for shell s8 in THIS run (W(s8)=0.039 < STAGE4_GPH_WTHR=0.13 and
GPH_ALLLEVEL_NLTE unset  ==> use_nlte=0):
  pop_l = g_l * exp(-E_l/kT_e) / U_ion        (BOLTZMANN weight at T_e -- code's actual s8 weight)

J field: COEVOLVE_PHOTOION_ALPHA=1.0 => J = mc_J exactly (pure Monte-Carlo shadow field).
sigma:  per-level CMFGEN sigma_bf from cmfgen_sigma_bf.bin (global-level indexed).
grid:   sigma grid == NLTE field grid == coevolve_field grid (1000 log bins 1.5e14..3.0e16 Hz).

We ALSO compute the counterfactual n_k (actual NLTE population) weighting to expose
whether the metastable trap would dominate IF the depth gate let it feed Gph.
"""
import struct, math, numpy as np, sys, os

RUN   = "logs/coevolve_consume_a10_kx_euv461"
BASE  = "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
LEVELS= os.path.join(BASE,"data/tardis_reference_toy06_19p48d/levels.csv")
SIGMA = os.path.join(BASE,"data/tardis_reference_toy06_19p48d/cmfgen_sigma_bf.bin")
FIELD = os.path.join(BASE,RUN,"lumina_coevolve_field.csv")
LEVP  = os.path.join(BASE,RUN,"lumina_levelpop.csv")
PLAS  = os.path.join(BASE,RUN,"lumina_plasma_state.csv")
CMFJ  = os.path.join(BASE,"data/cmfgen_jtable_toy06_19p48d.bin")

# constants (match lumina.h exactly)
H=6.62607015e-27; KB=1.380649e-16; EV=1.602176634e-12; C=2.99792458e10; PI=3.14159265358979323846
KB_eV=8.617333262e-5
CHI_FeIII = 30.6513735284   # eV, Fe III -> Fe IV threshold (ionization_energies.csv ion_number=2)
S=8
Zt,IONt = 26,2              # Fe III

# ---- grid (identical for sigma / field) ----
NFB=1000; NUMIN=1.5e14; NUMAX=3.0e16
dln=math.log(NUMAX/NUMIN)/NFB
bb=np.arange(NFB)
lo=math.log(NUMIN)+bb*dln
nu =np.exp(lo+0.5*dln)            # bin centers (Hz)
dnu=np.exp(lo+dln)-np.exp(lo)     # bin widths (Hz)
lam=C/nu*1e8                      # Angstrom

# ---- plasma state s8 ----
Te=None
with open(PLAS) as f:
    f.readline()
    for ln in f:
        p=ln.split(',');
        if int(p[0])==S: Te=float(p[4]); W=float(p[1]); ne=float(p[3]); break
kT=KB*Te; kT_eV=KB_eV*Te
print(f"# s{S}: T_e={Te:.2f} K  W={W:.4f}  n_e={ne:.3e}  kT={kT_eV:.4f} eV")

# ---- levels.csv -> global index for Fe III ----
Zc=[];ionc=[];lnum=[];Eev=[];gg=[];meta=[]
with open(LEVELS) as f:
    f.readline()
    for ln in f:
        p=ln.split(',')
        Zc.append(int(p[0]));ionc.append(int(p[1]));lnum.append(int(p[2]))
        Eev.append(float(p[3]));gg.append(int(p[4]));meta.append(int(p[5]))
Zc=np.array(Zc);ionc=np.array(ionc);lnum=np.array(lnum)
Eev=np.array(Eev);gg=np.array(gg);meta=np.array(meta)
nlev=len(Zc)
gl_fe=np.where((Zc==Zt)&(ionc==IONt))[0]
print(f"# levels.csv total={nlev}  Fe III levels={len(gl_fe)}  gl range=[{gl_fe.min()},{gl_fe.max()}]")

# ---- sigma_bf.bin: header + has_cmfgen + Fe III sigma rows ----
with open(SIGMA,'rb') as f:
    magic,ver,slev,sfreq=struct.unpack('<IIii',f.read(16))
    snumin,snumax=struct.unpack('<dd',f.read(16))
    assert slev==nlev and sfreq==NFB, (slev,nlev,sfreq)
    has8=np.frombuffer(f.read(nlev),dtype=np.int8).astype(int)
    pad=(8-(nlev%8))%8; f.read(pad)
    sig_off=f.tell()
    g0,g1=gl_fe.min(),gl_fe.max()+1
    f.seek(sig_off+g0*NFB*8)
    sig_fe=np.frombuffer(f.read((g1-g0)*NFB*8),dtype='<f8').reshape(g1-g0,NFB)
print(f"# sigma.bin nu_min={snumin:.3e} nu_max={snumax:.3e} has_cmfgen(FeIII)={has8[g0:g1].sum()}")

# ---- field s8: mc_J, cs_J ----
mcJ=np.zeros(NFB); csJ=np.zeros(NFB)
with open(FIELD) as f:
    f.readline()
    for ln in f:
        p=ln.split(',')
        if int(p[0])==S:
            b=int(p[1]); csJ[b]=float(p[3]); mcJ[b]=float(p[4])
# verify grid vs csv wavelength
# (bin0 lam ~19933) -- trust reconstruction

# ---- CMFGEN J benchmark (JTAB) s8 ----
cmfJ=None
if os.path.exists(CMFJ):
    with open(CMFJ,'rb') as f:
        hmagic,hver,hns,hnf=struct.unpack('<iiii',f.read(16))
        if hmagic==0x4A544142 and hns>=S+1 and hnf==NFB:
            buf=np.frombuffer(f.read(hns*hnf*8),dtype='<f8').reshape(hns,hnf)
            cmfJ=buf[S].copy()
    print(f"# CMFGEN jtable: magic ok, shells={hns} nfb={hnf}  nonzero(s8)={np.count_nonzero(cmfJ)}")
else:
    print("# CMFGEN jtable NOT FOUND")

# ---- levelpop n_k / b_k for Fe III s8 (keyed by level_number) ----
nk={}; bk={}
with open(LEVP) as f:
    f.readline()
    for ln in f:
        p=ln.split(',')
        if int(p[0])==S and int(p[1])==Zt and int(p[2])==IONt:
            L=int(p[3]); nk[L]=float(p[6]); bk[L]=float(p[8])

# ---- partition function U_ion over Fe III levels (code: x<50) ----
U=0.0
for gl in gl_fe:
    x=Eev[gl]*EV/kT
    if x<50.0: U+=gg[gl]*math.exp(-x)

# ---- per-level Gph contribution ----
rows=[]
Gtot_boltz_mc=Gtot_boltz_cs=Gtot_boltz_cmf=0.0
Gtot_nk_mc=0.0
for k,gl in enumerate(gl_fe):
    if not has8[gl]: continue
    E=Eev[gl]
    chi_l=CHI_FeIII-E
    if chi_l<=0: continue
    nu_l=chi_l*EV/H
    x=E*EV/kT
    if x>=50.0: continue
    pop_b=gg[gl]*math.exp(-x)/U
    srow=sig_fe[gl-g0]
    m=(nu>=nu_l)&(srow>0.0)
    if not m.any(): continue
    kern=4.0*PI*srow[m]/(H*nu[m])*dnu[m]
    I_mc=float((kern*mcJ[m]).sum())
    I_cs=float((kern*csJ[m]).sum())
    I_cmf=float((kern*cmfJ[m]).sum()) if cmfJ is not None else 0.0
    # integrand-weighted mean wavelength of the MC contribution
    contr_bins=kern*mcJ[m]
    lam_mean=float((lam[m]*contr_bins).sum()/contr_bins.sum()) if contr_bins.sum()>0 else float('nan')
    L=int(lnum[gl])
    nkl=nk.get(L,0.0); bkl=bk.get(L,-1.0)
    cb_mc=pop_b*I_mc; cb_cs=pop_b*I_cs; cb_cmf=pop_b*I_cmf
    cnk_mc=nkl*I_mc
    Gtot_boltz_mc+=cb_mc; Gtot_boltz_cs+=cb_cs; Gtot_boltz_cmf+=cb_cmf
    Gtot_nk_mc+=cnk_mc
    rows.append(dict(L=L,gl=int(gl),E=E,edge=12398.419/chi_l,g=gg[gl],meta=meta[gl],
                     pop_b=pop_b,nk=nkl,bk=bkl,I_mc=I_mc,
                     cb_mc=cb_mc,cnk_mc=cnk_mc,lam_mean=lam_mean))

print(f"\n# U_ion(FeIII,Boltz@T_e)={U:.3f}")
print(f"# Gph_perion(Boltzmann,mc_J)  = {Gtot_boltz_mc:.4e}  [code's ACTUAL s8 weight]")
print(f"# Gph_perion(Boltzmann,cs_J)  = {Gtot_boltz_cs:.4e}  (thermal ref field)")
if cmfJ is not None:
    print(f"# Gph_perion(Boltzmann,CMFGEN)= {Gtot_boltz_cmf:.4e}  (CMFGEN J benchmark)")
    print(f"#   amplification mc/CMFGEN   = {Gtot_boltz_mc/Gtot_boltz_cmf:.2f}x   mc/cs = {Gtot_boltz_mc/Gtot_boltz_cs:.2f}x")
print(f"# SUM n_k*I_mc (NLTE-weighted, absolute) = {Gtot_nk_mc:.4e}")

def show(rows,key,tot,label,n=12):
    print(f"\n===== TOP {n} Fe III levels by {label} =====")
    print(f"{'lnum':>5} {'gl':>6} {'E_eV':>7} {'edge_A':>8} {'g':>3} {'meta':>4} "
          f"{'pop_b':>10} {'b_k':>9} {'n_k':>10} {'lam_wt':>7} {'%'+key:>8}")
    sr=sorted(rows,key=lambda r:r[key],reverse=True)
    acc=0.0
    for r in sr[:n]:
        pct=100.0*r[key]/tot
        acc+=pct
        print(f"{r['L']:>5} {r['gl']:>6} {r['E']:>7.3f} {r['edge']:>8.1f} {r['g']:>3} "
              f"{r['meta']:>4} {r['pop_b']:>10.3e} {r['bk']:>9.2f} {r['nk']:>10.3e} "
              f"{r['lam_mean']:>7.1f} {pct:>7.2f}%")
    print(f"  (top {n} cumulative = {acc:.2f}%)")
    return sr

sr_code=show(rows,'cb_mc',Gtot_boltz_mc,"CODE's ACTUAL Gph (Boltzmann x mc_J)")
sr_nk  =show(rows,'cnk_mc',Gtot_nk_mc,"COUNTERFACTUAL Gph (actual n_k x mc_J)")

# ---- edge/band aggregation of the CODE's actual Gph ----
print("\n===== CODE Gph by EDGE-wavelength band (Boltzmann x mc_J) =====")
bands=[(400,420),(420,440),(440,470),(470,520),(520,620),(620,900),(900,20000)]
for a,b in bands:
    tot=sum(r['cb_mc'] for r in rows if a<=r['edge']<b)
    n=sum(1 for r in rows if a<=r['edge']<b)
    print(f"  edge {a:>5}-{b:<5} A : {100*tot/Gtot_boltz_mc:6.2f}%  ({n} levels)")

# ---- field anomaly: mc vs cs vs CMFGEN in the dominant edge/integrand band ----
print("\n===== FIELD ANOMALY: J(mc) vs J(cs) vs J(CMFGEN) per wavelength band =====")
print(f"{'band_A':>14} {'mc_J':>11} {'cs_J':>11} {'CMFGEN_J':>11} {'mc/cs':>7} {'mc/cmf':>7}")
wbands=[(404,420),(420,461),(461,520),(520,620),(620,800),(800,1290),(1290,3000)]
for a,b in wbands:
    m=(lam>=a)&(lam<b)
    if not m.any(): continue
    # geometric-mean-ish: use mean of J over bins in band
    mc=mcJ[m].mean(); cs=csJ[m].mean(); cf=cmfJ[m].mean() if cmfJ is not None else float('nan')
    r1=mc/cs if cs>0 else float('inf')
    r2=mc/cf if (cmfJ is not None and cf>0) else float('nan')
    print(f"{a:>6}-{b:<6} {mc:>11.3e} {cs:>11.3e} {cf:>11.3e} {r1:>7.2f} {r2:>7.2f}")

# metastable levels of Fe III
print("\n===== Fe III metastable levels (levels.csv meta=1) =====")
for gl in gl_fe:
    if meta[gl]==1:
        L=int(lnum[gl]); chi_l=CHI_FeIII-Eev[gl]
        print(f"  lnum={L} E={Eev[gl]:.3f}eV edge={12398.419/chi_l:.1f}A g={gg[gl]} "
              f"b_k={bk.get(L,-1):.1f} n_k={nk.get(L,0):.3e} pop_boltz_frac={gg[gl]*math.exp(-Eev[gl]*EV/kT)/U:.3e}")

# ---- targeted summary numbers ----
print("\n===== TARGETED SUMMARY =====")
g_ground=next((r for r in rows if r['L']==0),None)
g_l17   =next((r for r in rows if r['L']==17),None)
if g_ground: print(f"GROUND (lnum0, edge404.5): CODE Gph share = {100*g_ground['cb_mc']/Gtot_boltz_mc:.2f}%  (pop_boltz={g_ground['pop_b']:.3e})")
if g_l17:    print(f"META lnum17 (edge460.6,b_k=1040): CODE Gph share = {100*g_l17['cb_mc']/Gtot_boltz_mc:.2f}%  |  counterfactual n_k share = {100*g_l17['cnk_mc']/Gtot_nk_mc:.2f}%")
# thermal vs trapped split of the CODE's actual Gph
for thr in (2.0,5.0):
    trap=sum(r['cb_mc'] for r in rows if r['bk']>thr)
    print(f"CODE Gph from levels with b_k>{thr:.0f} (over-populated traps) = {100*trap/Gtot_boltz_mc:.2f}%")
# integrand (absorbed-flux) wavelength band of CODE Gph -- recompute per level accumulation
absb={(404,461):0.0,(461,520):0.0,(520,900):0.0}
for k,gl in enumerate(gl_fe):
    if not has8[gl]: continue
    E=Eev[gl]; chi_l=CHI_FeIII-E
    if chi_l<=0: continue
    nu_l=chi_l*EV/H; x=E*EV/kT
    if x>=50: continue
    pop_b=gg[gl]*math.exp(-x)/U
    srow=sig_fe[gl-g0]; m=(nu>=nu_l)&(srow>0.0)
    if not m.any(): continue
    contr=pop_b*(4*PI*srow[m]/(H*nu[m])*dnu[m]*mcJ[m])
    lm=lam[m]
    for (a,b) in absb: absb[(a,b)]+=contr[(lm>=a)&(lm<b)].sum()
tota=sum(absb.values())
print("CODE Gph by ABSORBED-flux (integrand) wavelength band:")
for (a,b),v in absb.items(): print(f"   {a}-{b} A : {100*v/tota:.2f}%")
