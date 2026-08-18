#!/usr/bin/env python3
"""Offline reproduction of the SIMUL-ladder III<->IV balance for Fe and Co
under the twin run's pins (T_e table + Gph J-table).  r = Gph/(n_e*alpha)
exactly as lumina_plasma.c:4840, with Gph the all-level Boltzmann-weighted
photoionization (5435-5480) and alpha = frozenin_alpha_rr (2748-2879)+DR.
"""
import struct, numpy as np, csv, os

ROOT="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF=ROOT+"/data/tardis_reference_toy06_19p48d"

# ---- constants (lumina.h) ----
C=2.99792458e10; H=6.62607015e-27; K=1.380649e-16
M_PI=3.14159265358979323846; EV=1.602176634e-12; ME=9.1093837015e-28

# ---- frequency grid (NLTE = CMFGEN sigma = jtable, all identical) ----
NUMIN=1.5e14; NUMAX=3.0e16; NFB=1000
DLN=np.log(NUMAX/NUMIN)/NFB
bb=np.arange(NFB)
lo=np.log(NUMIN)+bb*DLN
nu=np.exp(lo+0.5*DLN)            # bin centers
dnu=np.exp(lo+DLN)-np.exp(lo)

# ---- levels.csv (row index = global level = sigma_bf index) ----
Z=[];ION=[];NUM=[];E=[];G=[]
with open(REF+"/levels.csv") as f:
    r=csv.reader(f);hdr=next(r)
    for row in r:
        Z.append(int(row[0]));ION.append(int(row[1]));NUM.append(int(row[2]))
        E.append(float(row[3]));G.append(int(row[4]))
Z=np.array(Z);ION=np.array(ION);E=np.array(E,float);G=np.array(G,float)
NLEV=len(Z)

# ---- sigma_bf.bin ----
with open(REF+"/cmfgen_sigma_bf.bin","rb") as f:
    magic,ver,nlev,nfreq=struct.unpack('<IIii',f.read(16))
    numin,numax=struct.unpack('<dd',f.read(16))
    assert nlev==NLEV and nfreq==NFB
    flag=np.frombuffer(f.read(nlev),dtype=np.int8).astype(int)
    pad=(8-(nlev%8))%8; f.read(pad)
    off_sig=f.tell()
sig_mm=np.memmap(REF+"/cmfgen_sigma_bf.bin",dtype='<f8',mode='r',
                 offset=off_sig,shape=(nlev,nfreq))
HAS=flag

# ---- ionization energies ----
IE={}
with open(REF+"/ionization_energies.csv") as f:
    r=csv.reader(f);next(r)
    for row in r: IE[(int(row[0]),int(row[1]))]=float(row[2])

# ---- jtable ----
with open(ROOT+"/data/cmfgen_jtable_toy06_19p48d.bin","rb") as f:
    h=struct.unpack('<iiii',f.read(16)); ns,nfb=h[2],h[3]
    JT=np.frombuffer(f.read(ns*nfb*8),dtype='<f8').reshape(ns,nfb).copy()

# ---- Te table & plasma_state (n_e) ----
TE={}
with open(ROOT+"/data/cmfgen_te_table_toy06_19p48d.csv") as f:
    for line in f:
        if line.startswith('#'):continue
        c=line.split(','); TE[int(c[0])]=float(c[2])
NE={}
with open(ROOT+"/logs/coevolve_consume_a10_kx_tetab_jtab/lumina_plasma_state.csv") as f:
    r=csv.reader(f);hh=next(r)
    for row in r: NE[int(row[0])]=float(row[3])

# ---- DR (FROZENIN_DR=1, boost=1.0) ----
DR={
 (26,3):([2.762952e-07,6.854108e-07,1.112431e-05,3.071916e-05,2.100067e-06,9.690480e-04],
         [9.502922e+01,7.045133e+02,1.554180e+04,4.609481e+04,3.392230e+03,2.483458e+05]),
 (27,3):([2.7336e-06,9.9735e-06,1.2109e-05],[6.1307e+02,4.3869e+03,9.7077e+03]),
}
def dr_alpha(Z,ion_recomb,T):
    if (Z,ion_recomb) not in DR: return 0.0
    c,Ei=DR[(Z,ion_recomb)]; s=0.0
    for ci,ei in zip(c,Ei):
        a=-ei/T
        if a<-700: continue
        s+=ci*np.exp(a)
    return s*T**-1.5

def levels_of(Zz,ionn):
    return np.where((Z==Zz)&(ION==ionn))[0]

def U_partition(Zz,ionn,T):
    kT=K*T; idx=levels_of(Zz,ionn)
    x=E[idx]*EV/kT
    m=x<50.0
    u=np.sum(G[idx][m]*np.exp(-x[m]))
    return u

def Gph(Zz,shell):
    """all-level Boltzmann photoionization rate of ion (Zz,ion=2)=III (5435-5480)."""
    T=TE[shell]; kT=K*T
    chi_g=IE[(Zz,2)]*EV   # III->IV ground threshold (erg)
    U=U_partition(Zz,2,T); U=U if U>=1.0 else 1.0
    Jrow=JT[shell]
    idx=levels_of(Zz,2)
    Gtot=0.0
    for l in idx:
        if not HAS[l]: continue
        El=E[l]*EV
        chi_l=chi_g-El
        if chi_l<=0: continue
        nu_l=chi_l/H
        x_l=El/kT
        if x_l>=50: continue
        pop_l=G[l]*np.exp(-x_l)/U
        if pop_l<=0: continue
        s=sig_mm[l]
        mask=(nu>=nu_l)&(s>0)&(Jrow>0)
        if not mask.any(): continue
        w=4.0*M_PI*s[mask]*Jrow[mask]/(H*nu[mask])*dnu[mask]
        Gtot+=pop_l*np.sum(w)
    return Gtot

def alpha_rad(Zz,shell):
    """frozenin_alpha_rr(III,IV,T): Milne recomb into III levels (2748-2879)."""
    T=TE[shell]; kT=K*T
    chi_ion=IE[(Zz,2)]*EV
    Uiv=U_partition(Zz,3,T)
    if Uiv<1.0:
        g_iv=levels_of(Zz,3); Uiv=G[g_iv[0]] if len(g_iv) else 1.0
    lam3=(H*H/(2.0*M_PI*ME*K*T))**1.5
    idx=levels_of(Zz,2)
    a_tot=0.0
    x_all=H*nu/kT
    for gl in idx:
        if not HAS[gl]: continue
        El=E[gl]*EV
        chi_l=chi_ion-El
        if chi_l<=0: continue
        nu_th=chi_l/H
        s=sig_mm[gl]
        mask=(nu>=nu_th)&(s>0)&(x_all<=700.0)
        if not mask.any(): continue
        xm=x_all[mask]
        B=(2.0*H*nu[mask]**3/(C*C))/np.expm1(xm)
        Rbf=np.sum(4.0*M_PI*B*s[mask]/(H*nu[mask])*dnu[mask])
        a_l=Rbf*lam3*G[gl]/(2.0*Uiv)*np.exp(chi_l/kT)
        a_tot+=a_l
    return a_tot

print("shell  Z   T_e     n_e       Gph        a_rad      a_DR       a_tot      "
      "r=N(IV)/N(III)")
results={}
for shell in [2,6,8]:
    for Zz,name in [(26,'Fe'),(27,'Co')]:
        T=TE[shell]; ne=NE[shell]
        g=Gph(Zz,shell); ar=alpha_rad(Zz,shell); adr=dr_alpha(Zz,3,T)
        at=ar+adr
        r=g/(ne*at) if at>0 else float('inf')
        results[(shell,Zz)]=(g,ar,adr,at,r)
        print("%3d  %s%3d  %7.1f  %.3e  %.3e  %.3e  %.3e  %.3e  %.4e"%(
            shell,name,Zz,T,ne,g,ar,adr,at,r))

# ---- compare with twin observed and CMFGEN ----
print("\n=== twin observed n(IV)/n(III) from lumina_ion_pops.csv ===")
pops={}
with open(ROOT+"/logs/coevolve_consume_a10_kx_tetab_jtab/lumina_ion_pops.csv") as f:
    r=csv.reader(f);next(r)
    for row in r:
        s=int(row[0]);z=int(row[1]);st=int(row[2]);n=float(row[3])
        pops[(s,z,st)]=n
for shell in [2,6,8]:
    for Zz,name in [(26,'Fe'),(27,'Co')]:
        n3=pops.get((shell,Zz,2),0);n4=pops.get((shell,Zz,3),0)
        robs=n4/n3 if n3>0 else float('inf')
        rrep=results[(shell,Zz)][4]
        print("  s%d %s: twin r_obs=%.4e   repro r=%.4e   ratio(repro/obs)=%.2f"%(
            shell,name,robs,rrep,rrep/robs if robs>0 else 0))
