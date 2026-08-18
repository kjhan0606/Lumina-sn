#!/usr/bin/env python3
"""Decompose Gph(Fe III) vs Gph(Co III): coverage, threshold/J-sampling,
sigma magnitude, and ground-vs-all-level weighting.  Also solve for the
common non-thermal ionization rate gnt that closes the twin balance."""
import struct, numpy as np, csv
ROOT="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
REF=ROOT+"/data/tardis_reference_toy06_19p48d"
C=2.99792458e10;H=6.62607015e-27;K=1.380649e-16
M_PI=3.14159265358979323846;EV=1.602176634e-12;ME=9.1093837015e-28
NUMIN=1.5e14;NUMAX=3.0e16;NFB=1000
DLN=np.log(NUMAX/NUMIN)/NFB
bb=np.arange(NFB);lo=np.log(NUMIN)+bb*DLN
nu=np.exp(lo+0.5*DLN);dnu=np.exp(lo+DLN)-np.exp(lo)
lamA=2.99792458e18/nu

Z=[];ION=[];E=[];G=[]
with open(REF+"/levels.csv") as f:
    r=csv.reader(f);next(r)
    for row in r:
        Z.append(int(row[0]));ION.append(int(row[1]))
        E.append(float(row[3]));G.append(int(row[4]))
Z=np.array(Z);ION=np.array(ION);E=np.array(E,float);G=np.array(G,float)
NLEV=len(Z)
with open(REF+"/cmfgen_sigma_bf.bin","rb") as f:
    _=f.read(32)
    flag=np.frombuffer(f.read(NLEV),dtype=np.int8).astype(int)
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
TE={};NE={}
with open(ROOT+"/data/cmfgen_te_table_toy06_19p48d.csv") as f:
    for line in f:
        if line.startswith('#'):continue
        c=line.split(',');TE[int(c[0])]=float(c[2])
with open(ROOT+"/logs/coevolve_consume_a10_kx_tetab_jtab/lumina_plasma_state.csv") as f:
    r=csv.reader(f);next(r)
    for row in r: NE[int(row[0])]=float(row[3])

def lev(Zz,ii): return np.where((Z==Zz)&(ION==ii))[0]
def U(Zz,ii,T):
    kT=K*T;idx=lev(Zz,ii);x=E[idx]*EV/kT;m=x<50.0
    u=np.sum(G[idx][m]*np.exp(-x[m]));return u if u>=1 else 1.0

def gph_detail(Zz,shell):
    T=TE[shell];kT=K*T;chi_g=IE[(Zz,2)]*EV;Uion=U(Zz,2,T)
    Jrow=JT[shell];idx=lev(Zz,2)
    ncov=0;ntot=len(idx);G_gnd=0.0;G_all=0.0;gnd_edge_sig=None;gnd_edge_J=None
    contribs=[]
    for k,l in enumerate(idx):
        if not HAS[l]:continue
        ncov+=1
        El=E[l]*EV;chi_l=chi_g-El
        if chi_l<=0:continue
        nu_l=chi_l/H;x_l=El/kT
        if x_l>=50:continue
        pop_l=G[l]*np.exp(-x_l)/Uion
        if pop_l<=0:continue
        s=sig_mm[l];mask=(nu>=nu_l)&(s>0)&(Jrow>0)
        if not mask.any():continue
        w=4.0*M_PI*s[mask]*Jrow[mask]/(H*nu[mask])*dnu[mask]
        gl=pop_l*np.sum(w)
        G_all+=gl
        if k==0:
            G_gnd=gl
            # ground threshold bin sigma + J
            ib=np.argmax(nu>=nu_l)
            gnd_edge_sig=s[ib];gnd_edge_J=Jrow[ib]
        contribs.append((k,E[l],G[l],pop_l,nu_l,H*nu_l/EV,gl))
    contribs.sort(key=lambda t:-t[6])
    return dict(ncov=ncov,ntot=ntot,G_gnd=G_gnd,G_all=G_all,
                chi_g_eV=chi_g/EV,gnd_nu=chi_g/H,gnd_lamA=2.99792458e18/(chi_g/H),
                gnd_edge_sig=gnd_edge_sig,gnd_edge_J=gnd_edge_J,top=contribs[:6])

print("=== Gph decomposition at s6, s8 ===")
for shell in [6,8]:
    print("\n--- shell %d  T_e=%.0f ---"%(shell,TE[shell]))
    for Zz,nm in [(26,'Fe III'),(27,'Co III')]:
        d=gph_detail(Zz,shell)
        print(" %s: chi=%.3f eV  edge=%.1f A  cov=%d/%d  sig_edge=%.3e  J_edge=%.3e"%(
            nm,d['chi_g_eV'],d['gnd_lamA'],d['ncov'],d['ntot'],
            d['gnd_edge_sig'] if d['gnd_edge_sig'] is not None else -1,
            d['gnd_edge_J'] if d['gnd_edge_J'] is not None else -1))
        print("      G_gnd=%.3e  G_all=%.3e  all/gnd=%.2f"%(
            d['G_gnd'],d['G_all'],d['G_all']/d['G_gnd'] if d['G_gnd']>0 else -1))
        print("      top-6 level contribs (k,E_eV,g,pop,thr_eV,Gph_l):")
        for k,El,gl,pop,nul,thre,glph in d['top']:
            print("        k=%4d E=%6.3f g=%3d pop=%.2e thr=%5.2feV Gph_l=%.3e"%(
                k,El,gl,pop,thre,glph))

# cross-element ground-threshold J comparison (same shell, isolate threshold effect)
print("\n=== Isolate threshold effect: J at Fe-edge vs Co-edge (same J field) ===")
for shell in [6,8]:
    feE=IE[(26,2)];coE=IE[(27,2)]
    fnu=feE*EV/H;cnu=coE*EV/H
    fi=np.argmax(nu>=fnu);ci=np.argmax(nu>=cnu)
    print(" s%d: Fe edge %.2feV/%.1fA J=%.3e ; Co edge %.2feV/%.1fA J=%.3e ; J_Co/J_Fe=%.3f"%(
        shell,feE,2.99792458e18/fnu,JT[shell][fi],
        coE,2.99792458e18/cnu,JT[shell][ci],JT[shell][ci]/JT[shell][fi]))

# solve common gnt from the two elements' observed r (verify common-mode)
print("\n=== Solve gnt from twin observed r (should agree Fe vs Co) ===")
import importlib.util
spec=importlib.util.spec_from_file_location("rb",
    ROOT+"/validation/cmfgen_toy06_19p48d/analysis/co3_closure_trace/repro_balance.py")
pops={}
with open(ROOT+"/logs/coevolve_consume_a10_kx_tetab_jtab/lumina_ion_pops.csv") as f:
    r=csv.reader(f);next(r)
    for row in r: pops[(int(row[0]),int(row[1]),int(row[2]))]=float(row[3])
