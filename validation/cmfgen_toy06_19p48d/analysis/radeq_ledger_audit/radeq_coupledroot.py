#!/usr/bin/env python3
"""Coupled root: re-solve ionization vs T (calibrated to run's committed s0 pops;
Gph fixed during bisection, alpha~T^-0.8 => r_j(T)=r_j0*(T/Te0)^0.8), recompute
line cooling(cs.J pump) + fb + ff + ad, find LOWEST root (upward march = code)."""
import csv, math, numpy as np, pandas as pd
H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; EV=1.602176634e-12
A_RAD=7.5657e-15; T_EXP=19.48*86400.0; SOB=2.6540281e-2
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN=f"{REPO}/logs/coevolve_consume_a10_kx_gphall"; BASE=f"{REPO}/data/tardis_reference_toy06_19p48d"
S=0; Te0=13119.874754; ne=4.426076e9; NU_MIN=1.5e14; NU_MAX=3e16; NFB=1000
dln=math.log(NU_MAX/NU_MIN)/NFB; H_dep=1.506865e-03

fld=pd.read_csv(f"{RUN}/lumina_coevolve_field.csv"); f0=fld[fld['shell']==S]
csJ_bin=np.full(NFB,1e-30); mcJ_bin=np.full(NFB,1e-30)
for _,r in f0.iterrows(): b=int(r['bin']); csJ_bin[b]=r['cs_J']; mcJ_bin[b]=r['mc_J']
def Jbin(nu,arr):
    b=np.clip(np.floor(np.log(nu/NU_MIN)/dln).astype(int),0,NFB-1); out=arr[b].copy()
    out[(nu<=NU_MIN)|(nu>=NU_MAX)]=1e-30; return out
def Bnu(nu,T): x=H*nu/(KB*T); return 2*H*nu**3/C**2/np.expm1(np.minimum(x,700))

# committed pops per (Z,stage) at s0 -> per-element stage vectors
pops={}
for r in csv.DictReader(open(f"{RUN}/lumina_ion_pops.csv")):
    if int(r['shell_id'])!=S: continue
    pops[(int(r['Z']),int(r['stage']))]=float(r['n_ion'])
elems=sorted(set(z for (z,s) in pops))
frac0={}; nel={}
for Zc in elems:
    v=np.array([pops.get((Zc,st),0.0) for st in range(8)])
    tot=v.sum()
    if tot<=0: continue
    nel[Zc]=tot; frac0[Zc]=v/tot
CHI={}
for r in csv.DictReader(open(f"{BASE}/ionization_energies.csv")):
    CHI[(int(r['atomic_number']),int(r['ion_number']))]=float(r['ionization_energy_eV'])

# ladder: r_j0 = frac0[j+1]/frac0[j]; r_j(T)=r_j0*(T/Te0)^0.8
def fracT(Zc,T):
    f0v=frac0[Zc]; sc=(T/Te0)**0.8
    y=np.zeros(8); y[0]=1.0
    for j in range(7):
        rj=(f0v[j+1]/f0v[j]) if f0v[j]>0 else 0.0
        y[j+1]=y[j]*rj*sc
    y/=y.sum(); return y
def pops_at(T):
    return {(Zc,st):nel[Zc]*fracT(Zc,T)[st] for Zc in nel for st in range(8)}

# lines
L=pd.read_csv(f"{BASE}/line_list.csv",
   usecols=['atomic_number','ion_number','level_number_lower','level_number_upper','f_lu','A_ul','nu'])
Z=L['atomic_number'].values; ion=L['ion_number'].values
llo=L['level_number_lower'].values; lup=L['level_number_upper'].values
f_lu=L['f_lu'].values; A_ul=L['A_ul'].values; nu=L['nu'].values.astype(float)
NL=len(Z)
tmp={}
for r in csv.DictReader(open(f"{BASE}/levels.csv")):
    key=(int(r['atomic_number']),int(r['ion_number'])); tmp.setdefault(key,{})[int(r['level_number'])]=(float(r['energy_eV']),float(r['g']))
levarr={}
for key,d in tmp.items():
    n=max(d)+1; E=np.zeros(n); g=np.ones(n)
    for k,(e,gg) in d.items(): E[k]=e; g[k]=gg
    levarr[key]=(E,g)
E_lo=np.zeros(NL); g_lo=np.ones(NL); g_up=np.ones(NL); valid=np.zeros(NL,bool)
lineZ=Z.copy(); lineion=ion.copy()
for key in set(zip(Z,ion)):
    Zk,ik=key
    if key not in levarr: continue
    E,g=levarr[key]; nmax=len(E); m=(Z==Zk)&(ion==ik); lo=llo[m]; up=lup[m]; ok=(lo<nmax)&(up<nmax)
    idx=np.where(m)[0][ok]; E_lo[idx]=E[lo[ok]]; g_lo[idx]=g[lo[ok]]; g_up[idx]=g[up[ok]]; valid[idx]=True
keep=valid&(nu>0)&np.isin(Z,list(nel.keys()))
Z=Z[keep];ion=ion[keep];f_lu=f_lu[keep];A_ul=A_ul[keep];nu=nu[keep];E_lo=E_lo[keep];g_lo=g_lo[keep];g_up=g_up[keep]
dE=H*nu; NLk=len(Z)
gbar=0.2;om_floor=1.0; ry_de=np.minimum(13.605693/(dE/EV),136.0)
coeff=np.where(f_lu>1e-10,np.maximum(8.63e-6*14.5*gbar*f_lu*g_lo*ry_de,8.63e-6*om_floor),8.63e-6*om_floor)
ftau=SOB*f_lu*(C/nu)*T_EXP
ionkeys=list(set(zip(Z.tolist(),ion.tolist())))
Jb_cs=Jbin(nu,csJ_bin)   # fixed (lagged field)
Bul=(C*C/(2*H*nu**3))*A_ul; Blu=Bul*(g_up/g_lo)
def U_of(key,T): E,g=levarr[key]; x=E*EV/(KB*T); return float(np.sum(g*np.exp(-np.minimum(x,300.0))))
def beta_esc(tau):
    return np.where(tau<=1e-6,1.0,np.where(tau>700,1.0/np.maximum(tau,1e-30),(1.0-np.exp(-np.minimum(tau,700)))/np.maximum(tau,1e-30)))
def alpha_rr(Zc,zrec,T): return 2.6e-13*zrec**1.6*(T/1e4)**-0.8

def Lambda_and_fb(T, Jb):
    P=pops_at(T)
    nion_lo=np.array([P.get((Z[i],ion[i]),0.0) for i in range(NLk)])  # slow; vectorize below
    return None
# vectorize nion_lo mapping per T
lineidx={}
for key in ionkeys:
    Zk,ik=key; lineidx[key]=np.where((Z==Zk)&(ion==ik))[0]
def nion_lo_of(T):
    P=pops_at(T); arr=np.zeros(NLk)
    for key in ionkeys: arr[lineidx[key]]=P.get(key,0.0)
    return arr,P
def r_of(T,Jb):
    nion_lo,P=nion_lo_of(T)
    U=np.ones(NLk)
    for key in ionkeys: U[lineidx[key]]=U_of(key,T)
    x=E_lo*EV/(KB*T); nlo=np.where(x<300,nion_lo*g_lo*np.exp(-np.minimum(x,300))/U,0.0)
    invsq=1/math.sqrt(T); exb=np.exp(-np.minimum(dE/(KB*T),300.0))
    qlu=coeff/g_lo*invsq*exb; qul=coeff/g_up*invsq; Clu=ne*qlu; Cul=ne*qul
    tau=ftau*nlo; be=beta_esc(tau); Rul=(A_ul+Bul*Jb)*be; Rlu=Blu*Jb*be; den=Cul+Rul
    nup=np.where(den>0,nlo*(Clu+Rlu)/np.maximum(den,1e-300),0.0)
    lam=float(np.sum(dE*(nlo*qlu*ne-nup*qul*ne)))
    # fb
    cfb=0.0
    for (Zc,st),nlo_i in P.items():
        nx=P.get((Zc,st+1),0.0)
        if nx<=0 or (Zc,st) not in CHI: continue
        cfb+=ne*nx*alpha_rr(Zc,st+1,T)*(CHI[(Zc,st)]*EV+KB*T)
    cff=1.426e-27*1.2*ne*ne*math.sqrt(T); cad=1.5*ne*KB*T*(3.0/T_EXP)
    Hph=7.2e-7
    return (H_dep+Hph)-(cff+cad+cfb+lam), lam, cfb

print("# COUPLED (ionization re-solved vs T) r(T) with cs.J cooling pump:")
print(f"# {'T':>7} {'r':>11} {'Lam':>11} {'Cfb':>11} {'Fe_III_frac':>11} {'Fe_IV_frac':>10}")
for T in [8000,10000,11000,12000,13120,14000,15000,16000,18000,20000,24000]:
    r,lam,cfb=r_of(T,Jb_cs); fT=fracT(26,T)
    print(f"# {T:>7} {r:+.3e} {lam:+.3e} {cfb:.3e} {fT[2]:11.3f} {fT[3]:10.3f}")

# lowest-root upward march (mimic code: Tlo=3500, first + -> - crossing)
def lowest_root(Jb):
    Ts=np.geomspace(3500,140000,25); prev=None; Ta=None
    for T in Ts:
        f,_,_=r_of(T,Jb)
        if prev is not None and prev>0 and f<=0:
            lo,hi=Ta,T
            for _ in range(40):
                Tm=0.5*(lo+hi); fm,_,_=r_of(Tm,Jb)
                if fm>0: lo=Tm
                else: hi=Tm
            return 0.5*(lo+hi)
        prev=f; Ta=T
    f0,_,_=r_of(3500,Jb)
    return "pin_lo(f_lo<=0)" if f0<=0 else "pin_hi(no crossing)"
# CMFGEN jtable interp to line freqs
import struct
with open(f"{REPO}/data/cmfgen_jtable_toy06_19p48d.bin",'rb') as fh:
    mg,ver,nsh,nfbj=struct.unpack('4i',fh.read(16)); jt=np.frombuffer(fh.read(),np.float64).reshape(nsh,nfbj)
jt_nu=np.array([NU_MIN*math.exp((b+0.5)*dln) for b in range(nfbj)]); jt0=jt[S]; mjt=jt0>0
Jb_cmf=np.interp(nu, jt_nu[mjt], jt0[mjt])
Jb_mc=Jbin(nu,mcJ_bin)
print(f"\n# LOWEST-ROOT (coupled, zero pump)            = {lowest_root(np.full(NLk,1e-30))}   [run T_e[0]=13120]")
print(f"# LOWEST-ROOT (coupled, thermal B(Te) pump)   = {lowest_root(Jbin(nu, np.array([Bnu(NU_MIN*math.exp((b+0.5)*dln),Te0) for b in range(NFB)])))}")
print(f"# LOWEST-ROOT (coupled, run mc_J pump)        = {lowest_root(Jb_mc)}")
print(f"# LOWEST-ROOT (coupled, run cs.J pump)        = {lowest_root(Jb_cs)}")
print(f"# LOWEST-ROOT (coupled, CMFGEN jtable pump)   = {lowest_root(Jb_cmf)}   [CMFGEN truth T=18760]")
