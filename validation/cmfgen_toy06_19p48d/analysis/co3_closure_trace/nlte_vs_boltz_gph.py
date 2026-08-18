#!/usr/bin/env python3
"""Does the twin's own NLTE Co III / Fe III population (b_k) rescue Gph vs the
Boltzmann weighting that GPH_ALLLEVEL actually uses?  Recompute Gph with NLTE
population fractions (the GPH_ALLLEVEL_NLTE path, 5377-5433) and compare."""
import struct, numpy as np, csv
ROOT="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF=ROOT+"/data/tardis_reference_toy06_19p48d"
LOG=ROOT+"/logs/coevolve_consume_a10_kx_tetab_jtab"
C=2.99792458e10;H=6.62607015e-27;K=1.380649e-16
M_PI=3.14159265358979323846;EV=1.602176634e-12
NUMIN=1.5e14;NUMAX=3.0e16;NFB=1000
DLN=np.log(NUMAX/NUMIN)/NFB
bb=np.arange(NFB);lo=np.log(NUMIN)+bb*DLN
nu=np.exp(lo+0.5*DLN);dnu=np.exp(lo+DLN)-np.exp(lo)

Z=[];ION=[];NUM=[];E=[];G=[]
with open(REF+"/levels.csv") as f:
    r=csv.reader(f);next(r)
    for row in r:
        Z.append(int(row[0]));ION.append(int(row[1]));NUM.append(int(row[2]))
        E.append(float(row[3]));G.append(int(row[4]))
Z=np.array(Z);ION=np.array(ION);NUM=np.array(NUM);E=np.array(E,float);G=np.array(G,float)
NLEV=len(Z)
with open(REF+"/cmfgen_sigma_bf.bin","rb") as f:
    _=f.read(32);flag=np.frombuffer(f.read(NLEV),dtype=np.int8).astype(int)
    pad=(8-(NLEV%8))%8;f.read(pad);off=f.tell()
sig_mm=np.memmap(REF+"/cmfgen_sigma_bf.bin",dtype='<f8',mode='r',offset=off,shape=(NLEV,NFB))
HAS=flag
IE={}
with open(REF+"/ionization_energies.csv") as f:
    r=csv.reader(f);next(r)
    for row in r: IE[(int(row[0]),int(row[1]))]=float(row[2])
with open(ROOT+"/data/cmfgen_jtable_toy06_19p48d.bin","rb") as f:
    h=struct.unpack('<iiii',f.read(16));ns,nfb=h[2],h[3]
    JT=np.frombuffer(f.read(ns*nfb*8),dtype='<f8').reshape(ns,nfb).copy()
TE={}
with open(ROOT+"/data/cmfgen_te_table_toy06_19p48d.csv") as f:
    for line in f:
        if line.startswith('#'):continue
        c=line.split(',');TE[int(c[0])]=float(c[2])

# --- NLTE n_k from levelpop: nk[(shell,Z,ion,level_num)] ---
nk={}
with open(LOG+"/lumina_levelpop.csv") as f:
    r=csv.reader(f);hd=next(r)
    for row in r:
        s=int(row[0]);z=int(row[1]);io=int(row[2]);ln=int(row[3]);n=float(row[6])
        if z in (26,27) and io==2 and s in (6,8):
            nk[(s,z,ln)]=n

def lev(Zz,ii): return np.where((Z==Zz)&(ION==ii))[0]
def U(Zz,ii,T):
    kT=K*T;idx=lev(Zz,ii);x=E[idx]*EV/kT;m=x<50.0
    u=np.sum(G[idx][m]*np.exp(-x[m]));return u if u>=1 else 1.0

def gph(Zz,shell,weight):
    """weight='boltz' (GPH_ALLLEVEL) or 'nlte' (GPH_ALLLEVEL_NLTE)."""
    T=TE[shell];kT=K*T;chi_g=IE[(Zz,2)]*EV;Uion=U(Zz,2,T)
    Jrow=JT[shell];idx=lev(Zz,2)
    if weight=='nlte':
        ntot=0.0
        for k,l in enumerate(idx):
            ntot+=nk.get((shell,Zz,NUM[l]),0.0)
        if ntot<=0: return None
    Gtot=0.0
    for k,l in enumerate(idx):
        if not HAS[l]:continue
        El=E[l]*EV;chi_l=chi_g-El
        if chi_l<=0:continue
        nu_l=chi_l/H;x_l=El/kT
        if x_l>=50 and weight=='boltz':continue
        if weight=='boltz':
            pop_l=G[l]*np.exp(-x_l)/Uion
        else:
            pop_l=nk.get((shell,Zz,NUM[l]),0.0)/ntot
        if pop_l<=0:continue
        s=sig_mm[l];mask=(nu>=nu_l)&(s>0)&(Jrow>0)
        if not mask.any():continue
        w=4.0*M_PI*s[mask]*Jrow[mask]/(H*nu[mask])*dnu[mask]
        Gtot+=pop_l*np.sum(w)
    return Gtot

print("shell  ion     Gph_boltz    Gph_nlte    nlte/boltz")
for shell in [6,8]:
    for Zz,nm in [(26,'Fe III'),(27,'Co III')]:
        gb=gph(Zz,shell,'boltz');gn=gph(Zz,shell,'nlte')
        print("  s%d  %s  %.3e  %.3e   %.2f"%(shell,nm,gb,gn,gn/gb if gb>0 else -1))

# report b_k of the low-threshold Co III levels (E 5-11 eV) at s8
print("\n=== Co III b_k at s8 for the low-threshold excited levels ===")
bkrows={}
with open(LOG+"/lumina_levelpop.csv") as f:
    r=csv.reader(f);next(r)
    for row in r:
        s=int(row[0]);z=int(row[1]);io=int(row[2]);ln=int(row[3])
        if s==8 and z==27 and io==2:
            bkrows[ln]=(float(row[4]),float(row[8]),float(row[6]))  # E_eV,b_k,n_k
for ln in [0,1,17,29,39,43,55]:
    if ln in bkrows:
        Eev,bk,nkk=bkrows[ln]
        print("  Co III lvl %3d  E=%6.3f eV  b_k=%.3e  n_k=%.3e"%(ln,Eev,bk,nkk))
print("\n=== Fe III b_k at s8 for its dominant levels ===")
bkf={}
with open(LOG+"/lumina_levelpop.csv") as f:
    r=csv.reader(f);next(r)
    for row in r:
        s=int(row[0]);z=int(row[1]);io=int(row[2]);ln=int(row[3])
        if s==8 and z==26 and io==2:
            bkf[ln]=(float(row[4]),float(row[8]))
for ln in [0,6,7,8,14,18]:
    if ln in bkf:
        Eev,bk=bkf[ln]
        print("  Fe III lvl %3d  E=%6.3f eV  b_k=%.3e"%(ln,Eev,bk))
