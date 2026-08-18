#!/usr/bin/env python3
"""Diagnose line pumping at s0: exact bin lookup (match nlte_get_J_at_nu),
per-line heating/cooling breakdown, cs.J vs B(Te) at cooling-line freqs."""
import csv, math, numpy as np, struct, pandas as pd
H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; EV=1.602176634e-12
A_RAD=7.5657e-15; T_EXP=19.48*86400.0; SOB=2.6540281e-2
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN=f"{REPO}/logs/coevolve_consume_a10_kx_gphall"; BASE=f"{REPO}/data/tardis_reference_toy06_19p48d"
S=0; Te=13119.874754; ne=4.426076e9; NU_MIN=1.5e14; NU_MAX=3e16; NFB=1000
dln=math.log(NU_MAX/NU_MIN)/NFB

# field s0: cs_J per bin (this IS nlte.J_nu). exact bin lookup like nlte_get_J_at_nu.
fld=pd.read_csv(f"{RUN}/lumina_coevolve_field.csv"); f0=fld[fld['shell']==S]
csJ_bin=np.full(NFB,1e-30); mcJ_bin=np.full(NFB,1e-30)
for _,r in f0.iterrows():
    b=int(r['bin']); csJ_bin[b]=r['cs_J']; mcJ_bin[b]=r['mc_J']
def Jbin(nu, arr):
    b=np.floor(np.log(nu/NU_MIN)/dln).astype(int); b=np.clip(b,0,NFB-1)
    out=arr[b].copy()
    out[(nu<=NU_MIN)|(nu>=NU_MAX)]=1e-30
    return out
# cs.J color check at s0: UV2500 vs opt6000
def binof(lam):
    nu=C/(lam*1e-8); return min(NFB-1,max(0,int(math.log(nu/NU_MIN)/dln)))
b2500=binof(2500.); b6000=binof(6000.); b1200=binof(1200.)
def B(nu,T):
    x=H*nu/(KB*T); return 2*H*nu**3/C**2/np.expm1(np.minimum(x,700))
print(f"# s0 cs.J color UV2500/opt6000 = {csJ_bin[b2500]/csJ_bin[b6000]:.3f}")
print(f"# s0 cs.J/B(Te) at 2500A = {csJ_bin[b2500]/B(C/2500e-8,Te):.3f}  (>1 => super-thermal => pumps)")
print(f"# s0 cs.J/B(Te) at 6000A = {csJ_bin[b6000]/B(C/6000e-8,Te):.3f}")
print(f"# s0 cs.J/B(Te) at 1200A = {csJ_bin[b1200]/B(C/1200e-8,Te):.3e}  (FUV)")
print(f"# s0 mc.J/B(Te) at 2500A = {mcJ_bin[b2500]/B(C/2500e-8,Te):.3f}")

# pops + levels + lines (reuse cooling.py machinery, dominant coolants only for speed: Fe Co Ni)
npops={}
for r in csv.DictReader(open(f"{RUN}/lumina_ion_pops.csv")):
    if int(r['shell_id'])!=S: continue
    npops[(int(r['Z']),int(r['stage']))]=float(r['n_ion'])
levarr={}
tmp={}
for r in csv.DictReader(open(f"{BASE}/levels.csv")):
    key=(int(r['atomic_number']),int(r['ion_number']))
    tmp.setdefault(key,{})[int(r['level_number'])]=(float(r['energy_eV']),float(r['g']))
for key,d in tmp.items():
    n=max(d)+1; E=np.zeros(n); g=np.ones(n)
    for k,(e,gg) in d.items(): E[k]=e; g[k]=gg
    levarr[key]=(E,g)
def U_of(key,T):
    E,g=levarr[key]; x=E*EV/(KB*T); return float(np.sum(g*np.exp(-np.minimum(x,300.0))))

L=pd.read_csv(f"{BASE}/line_list.csv",
   usecols=['atomic_number','ion_number','level_number_lower','level_number_upper','f_lu','A_ul','nu'])
Z=L['atomic_number'].values; ion=L['ion_number'].values
llo=L['level_number_lower'].values; lup=L['level_number_upper'].values
f_lu=L['f_lu'].values; A_ul=L['A_ul'].values; nu=L['nu'].values.astype(float)
NL=len(Z)
E_lo=np.zeros(NL); g_lo=np.ones(NL); g_up=np.ones(NL); nion_lo=np.zeros(NL); valid=np.zeros(NL,bool)
for key in set(zip(Z,ion)):
    Zk,ik=key
    if key not in levarr: continue
    E,g=levarr[key]; nmax=len(E)
    m=(Z==Zk)&(ion==ik); lo=llo[m]; up=lup[m]; ok=(lo<nmax)&(up<nmax)
    idx=np.where(m)[0][ok]
    E_lo[idx]=E[lo[ok]]; g_lo[idx]=g[lo[ok]]; g_up[idx]=g[up[ok]]; nion_lo[idx]=npops.get((Zk,ik),0.0); valid[idx]=True
keep=valid&(nion_lo>0)&(nu>0)
Z=Z[keep];ion=ion[keep];f_lu=f_lu[keep];A_ul=A_ul[keep];nu=nu[keep]
E_lo=E_lo[keep];g_lo=g_lo[keep];g_up=g_up[keep];nion_lo=nion_lo[keep]
dE=H*nu; NLk=len(Z)
gbar=0.2;om_floor=1.0
ry_de=np.minimum(13.605693/(dE/EV),136.0)
coeff=np.where(f_lu>1e-10,np.maximum(8.63e-6*14.5*gbar*f_lu*g_lo*ry_de,8.63e-6*om_floor),8.63e-6*om_floor)
ftau=SOB*f_lu*(C/nu)*T_EXP
ionkeys=list(set(zip(Z.tolist(),ion.tolist())))
def beta_esc(tau):
    return np.where(tau<=1e-6,1.0,np.where(tau>700.0,1.0/np.maximum(tau,1e-30),(1.0-np.exp(-np.minimum(tau,700)))/np.maximum(tau,1e-30)))
def netcool(T, arr, use_bin=True):
    invsq=1/math.sqrt(T)
    U=np.ones(NLk)
    for key in ionkeys:
        Zk,ik=key; m=(Z==Zk)&(ion==ik); U[m]=U_of((Zk,ik),T)
    x=E_lo*EV/(KB*T); nlo=np.where(x<300,nion_lo*g_lo*np.exp(-np.minimum(x,300))/U,0.0)
    exb=np.exp(-np.minimum(dE/(KB*T),300.0))
    qlu=coeff/g_lo*invsq*exb; qul=coeff/g_up*invsq
    Clu=ne*qlu;Cul=ne*qul; tau=ftau*nlo; be=beta_esc(tau)
    Jb=Jbin(nu,arr) if use_bin else np.interp(nu, np.sort(nu), arr)  # bin lookup
    Bul=(C*C/(2*H*nu**3))*A_ul; Blu=Bul*(g_up/g_lo)
    Rul=(A_ul+Bul*Jb)*be; Rlu=Blu*Jb*be; den=Cul+Rul
    nup=np.where(den>0,nlo*(Clu+Rlu)/np.maximum(den,1e-300),0.0)
    net=dE*(nlo*qlu*ne-nup*qul*ne)
    return net,Jb,nlo,be,tau

# exact-bin cs pumping
net,Jb,nlo,be,tau=netcool(Te,csJ_bin)
print(f"\n# EXACT-BIN cs.J pumping: Lambda_line(Te) = {net.sum():.3e}  (neg=heating)")
print(f"#   sum of COOLING lines (net>0) = {net[net>0].sum():.3e}")
print(f"#   sum of HEATING lines (net<0) = {net[net<0].sum():.3e}")
# concentration: top heating lines
order=np.argsort(net)  # most negative first
print(f"\n# TOP-15 HEATING lines (exact-bin cs.J):")
print(f"#  {'Z':>3}{'ion':>4} {'lamA':>9} {'net':>11} {'cs/B(Te)':>9} {'beta':>8} {'tau':>9} {'f_lu':>8}")
lam=C/nu*1e8
for i in order[:15]:
    r=Jb[i]/B(nu[i],Te)
    print(f"#  {Z[i]:>3}{ion[i]:>4} {lam[i]:9.1f} {net[i]:11.3e} {r:9.2f} {be[i]:8.2e} {tau[i]:9.2e} {f_lu[i]:8.1e}")
# how many lines carry 90% of heating
h=np.sort(net[net<0]); csum=np.cumsum(h); tot=csum[-1]
n90=np.searchsorted(csum, 0.9*tot)
print(f"#\n# heating carried by {n90} lines (of {np.sum(net<0)} heaters) to reach 90%")
# fraction of heating from thick (beta<0.1) vs thin lines
mheat=net<0
print(f"# heating from beta<0.1 (thick) lines = {net[mheat&(be<0.1)].sum():.3e}  ({100*net[mheat&(be<0.1)].sum()/net[mheat].sum():.0f}%)")
print(f"# heating from beta>0.1 (thin)  lines = {net[mheat&(be>=0.1)].sum():.3e}")
# what if we use B(Te) as pump (thermal) - should be ~0
netT,_,_,_,_=netcool(Te, np.array([B(NU_MIN*math.exp((b+0.5)*dln),Te) for b in range(NFB)]))
print(f"\n# thermal-pump (Jb=B(Te) per bin) Lambda_line = {netT.sum():.3e} (should be ~0)")
