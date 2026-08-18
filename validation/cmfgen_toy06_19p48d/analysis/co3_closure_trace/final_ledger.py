#!/usr/bin/env python3
"""Consolidated ledger: reproduce twin r=(Gph+gnt)/(n_e*alpha) with a COMMON
gnt solved per shell, then decompose the Fe-vs-Co differential and the deficit
vs CMFGEN.  Writes results CSV to this dir."""
import struct, numpy as np, csv
ROOT="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF=ROOT+"/data/tardis_reference_toy06_19p48d"
LOG=ROOT+"/logs/coevolve_consume_a10_kx_tetab_jtab"
OUT=ROOT+"/validation/cmfgen_toy06_19p48d/analysis/co3_closure_trace"
C=2.99792458e10;H=6.62607015e-27;K=1.380649e-16
M_PI=3.14159265358979323846;EV=1.602176634e-12;ME=9.1093837015e-28
NUMIN=1.5e14;NUMAX=3.0e16;NFB=1000;DLN=np.log(NUMAX/NUMIN)/NFB
bb=np.arange(NFB);lo=np.log(NUMIN)+bb*DLN
nu=np.exp(lo+0.5*DLN);dnu=np.exp(lo+DLN)-np.exp(lo)
Z=[];ION=[];E=[];G=[]
with open(REF+"/levels.csv") as f:
    r=csv.reader(f);next(r)
    for row in r:
        Z.append(int(row[0]));ION.append(int(row[1]));E.append(float(row[3]));G.append(int(row[4]))
Z=np.array(Z);ION=np.array(ION);E=np.array(E,float);G=np.array(G,float);NLEV=len(Z)
with open(REF+"/cmfgen_sigma_bf.bin","rb") as f:
    _=f.read(32);flag=np.frombuffer(f.read(NLEV),dtype=np.int8).astype(int)
    f.read((8-(NLEV%8))%8);off=f.tell()
sig=np.memmap(REF+"/cmfgen_sigma_bf.bin",dtype='<f8',mode='r',offset=off,shape=(NLEV,NFB));HAS=flag
IE={}
with open(REF+"/ionization_energies.csv") as f:
    r=csv.reader(f);next(r)
    for row in r: IE[(int(row[0]),int(row[1]))]=float(row[2])
with open(ROOT+"/data/cmfgen_jtable_toy06_19p48d.bin","rb") as f:
    h=struct.unpack('<iiii',f.read(16));JT=np.frombuffer(f.read(h[2]*h[3]*8),dtype='<f8').reshape(h[2],h[3]).copy()
TE={};NE={}
with open(ROOT+"/data/cmfgen_te_table_toy06_19p48d.csv") as f:
    for line in f:
        if not line.startswith('#'):
            c=line.split(',');TE[int(c[0])]=float(c[2])
with open(LOG+"/lumina_plasma_state.csv") as f:
    r=csv.reader(f);next(r)
    for row in r: NE[int(row[0])]=float(row[3])
DR={(26,3):([2.762952e-07,6.854108e-07,1.112431e-05,3.071916e-05,2.100067e-06,9.690480e-04],
            [9.502922e+01,7.045133e+02,1.554180e+04,4.609481e+04,3.392230e+03,2.483458e+05]),
    (27,3):([2.7336e-06,9.9735e-06,1.2109e-05],[6.1307e+02,4.3869e+03,9.7077e+03])}
def dr_a(Zz,T):
    c,Ei=DR[(Zz,3)];return sum(ci*np.exp(-ei/T) for ci,ei in zip(c,Ei) if -ei/T>-700)*T**-1.5
def lv(Zz,ii): return np.where((Z==Zz)&(ION==ii))[0]
def U(Zz,ii,T):
    kT=K*T;idx=lv(Zz,ii);x=E[idx]*EV/kT;m=x<50;u=np.sum(G[idx][m]*np.exp(-x[m]));return u if u>=1 else 1.0
def Gph(Zz,s):
    T=TE[s];kT=K*T;chi=IE[(Zz,2)]*EV;Ui=U(Zz,2,T);Jr=JT[s];tot=0.0
    for l in lv(Zz,2):
        if not HAS[l]:continue
        chil=chi-E[l]*EV
        if chil<=0:continue
        nul=chil/H;xl=E[l]*EV/kT
        if xl>=50:continue
        pop=G[l]*np.exp(-xl)/Ui
        s_=sig[l];m=(nu>=nul)&(s_>0)&(Jr>0)
        if m.any():tot+=pop*np.sum(4*M_PI*s_[m]*Jr[m]/(H*nu[m])*dnu[m])
    return tot
def alpha(Zz,s):
    T=TE[s];kT=K*T;chi=IE[(Zz,2)]*EV;Uiv=U(Zz,3,T);lam3=(H*H/(2*M_PI*ME*K*T))**1.5
    xall=H*nu/kT;a=0.0
    for gl in lv(Zz,2):
        if not HAS[gl]:continue
        chil=chi-E[gl]*EV
        if chil<=0:continue
        nth=chil/H;s_=sig[gl];m=(nu>=nth)&(s_>0)&(xall<=700)
        if not m.any():continue
        B=(2*H*nu[m]**3/(C*C))/np.expm1(xall[m])
        Rbf=np.sum(4*M_PI*B*s_[m]/(H*nu[m])*dnu[m])
        a+=Rbf*lam3*G[gl]/(2*Uiv)*np.exp(chil/kT)
    return a
pops={}
with open(LOG+"/lumina_ion_pops.csv") as f:
    r=csv.reader(f);next(r)
    for row in r: pops[(int(row[0]),int(row[1]),int(row[2]))]=float(row[3])
# CMFGEN r (velocity-matched, hand-read)
CMF={(2,26):210.8,(6,26):0.0743,(8,26):0.02234,(2,27):85.9,(6,27):0.167,(8,27):0.1098}
rows=[["shell","Z","T_e","n_e","Gph","alpha_rad","alpha_DR","alpha_tot",
       "gnt_solved","r_repro_with_gnt","r_twin_obs","r_cmfgen","deficit_cmf/twin"]]
print("%-5s %-3s %8s %10s %10s %10s %10s %10s %10s %10s %10s %8s"%(
      "sh","Z","Te","Gph","a_rad","a_DR","a_tot","gnt","r_repro","r_obs","r_cmf","def"))
for s in [2,6,8]:
    T=TE[s];ne=NE[s]
    dat={}
    for Zz in (26,27):
        g=Gph(Zz,s);ar=alpha(Zz,s);ad=dr_a(Zz,T);at=ar+ad
        n3=pops.get((s,Zz,2),0);n4=pops.get((s,Zz,3),0);robs=n4/n3 if n3>0 else float('nan')
        gnt=robs*ne*at-g               # solve common gnt from observed r
        dat[Zz]=(g,ar,ad,at,gnt,robs)
    # verify gnt agreement
    gnt_fe=dat[26][4];gnt_co=dat[27][4];gnt=0.5*(gnt_fe+gnt_co)
    for Zz in (26,27):
        g,ar,ad,at,gntz,robs=dat[Zz]
        rrep=(g+gnt)/(ne*at)
        rcmf=CMF[(s,Zz)];defc=rcmf/robs if robs>0 else float('nan')
        print("%-5d %-3d %8.0f %10.3e %10.3e %10.3e %10.3e %10.3e %10.4e %10.4e %10.4e %8.2f"%(
              s,Zz,T,g,ar,ad,at,gntz,rrep,robs,rcmf,defc))
        rows.append([s,Zz,T,ne,g,ar,ad,at,gntz,rrep,robs,rcmf,defc])
    print("      gnt(Fe)=%.3e gnt(Co)=%.3e agree=%.1f%%"%(gnt_fe,gnt_co,100*gnt_co/gnt_fe))
with open(OUT+"/final_ledger.csv","w",newline="") as f:
    csv.writer(f).writerows(rows)
print("\nwrote",OUT+"/final_ledger.csv")
