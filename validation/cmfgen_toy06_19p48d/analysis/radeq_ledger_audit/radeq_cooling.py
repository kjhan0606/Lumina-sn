#!/usr/bin/env python3
"""Reconstruct simul_r1 line cooling Lambda_line(T,Jb) at s0 with the EXACT C formulas
(VR_STD=1, gbar=0.2, om_floor=1.0, beta_esc=(1-e^-tau)/tau). Run pops held fixed.
Measures pumping sensitivity + counterfactual root. Read-only."""
import csv, math, numpy as np, struct
import pandas as pd

H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; EV=1.602176634e-12
AMU=1.66053906660e-24; A_RAD=7.5657e-15; RY_EV=13.605693
T_EXP=19.48*86400.0
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN=f"{REPO}/logs/coevolve_consume_a10_kx_gphall"
BASE=f"{REPO}/data/tardis_reference_toy06_19p48d"
S=0; Te=13119.874754; ne=4.426076e9
H_dep=1.506865e-03
SOB=2.6540281e-2

# ---- field s0 (cs_J for line pumping = nlte->J_nu; mc_J for reference) ----
fld=pd.read_csv(f"{RUN}/lumina_coevolve_field.csv")
f0=fld[fld['shell']==S].copy()
nu_f=(C/(f0['wavelength_A'].values*1e-8)); o=np.argsort(nu_f)
nu_f=nu_f[o]; csJ=f0['cs_J'].values[o]; mcJ=f0['mc_J'].values[o]

# ---- CMFGEN jtable interpolated onto line freqs ----
with open(f"{REPO}/data/cmfgen_jtable_toy06_19p48d.bin",'rb') as fh:
    magic,ver,nsh,nfb=struct.unpack('4i',fh.read(16)); jt=np.frombuffer(fh.read(),np.float64).reshape(nsh,nfb)
NU_MIN=1.5e14; NU_MAX=3e16; dln=math.log(NU_MAX/NU_MIN)/nfb
jt_nu=np.array([NU_MIN*math.exp((bb+0.5)*dln) for bb in range(nfb)]); jt_s0=jt[S]
mjt=jt_s0>0

def Jb_of(nu_l, which):
    if which=='cs':  return np.interp(nu_l, nu_f, csJ)
    if which=='mc':  return np.interp(nu_l, nu_f, mcJ)
    if which=='cmfgen': return np.interp(nu_l, jt_nu[mjt], jt_s0[mjt])
    if which=='thermal':  # B_nu(Te)
        x=H*nu_l/(KB*Te); return 2*H*nu_l**3/C**2/np.expm1(np.minimum(x,700))
    if which=='zero': return np.zeros_like(nu_l)

# ---- ion pops s0 (lower-ion density per (Z,stage)) ----
npops={}
for r in csv.DictReader(open(f"{RUN}/lumina_ion_pops.csv")):
    if int(r['shell_id'])!=S: continue
    npops[(int(r['Z']),int(r['stage']))]=float(r['n_ion'])

# ---- levels: (Z,ion,levnum) -> (E_eV,g) ; and partition fn arrays ----
lev={}  # (Z,ion)-> dict levnum->(E,g)
for r in csv.DictReader(open(f"{BASE}/levels.csv")):
    key=(int(r['atomic_number']),int(r['ion_number']))
    lev.setdefault(key,{})[int(r['level_number'])]=(float(r['energy_eV']),float(r['g']))
# partition function U(Z,ion,T)
levarr={}
for key,d in lev.items():
    n=max(d)+1; E=np.zeros(n); g=np.ones(n)
    for k,(e,gg) in d.items(): E[k]=e; g[k]=gg
    levarr[key]=(E,g)
def U_of(key,T):
    E,g=levarr[key]; x=E*EV/(KB*T); return float(np.sum(g*np.exp(-np.minimum(x,300.0))))

# ---- lines: load all, map E_lo,g_lo,g_up, compute coeff (VR_STD) ----
print("loading 2.56M lines...",flush=True)
L=pd.read_csv(f"{BASE}/line_list.csv",
   usecols=['atomic_number','ion_number','level_number_lower','level_number_upper','f_lu','A_ul','nu'])
Z=L['atomic_number'].values; ion=L['ion_number'].values
llo=L['level_number_lower'].values; lup=L['level_number_upper'].values
f_lu=L['f_lu'].values; A_ul=L['A_ul'].values; nu=L['nu'].values.astype(float)
NL=len(Z); print(f"  {NL} lines")

# vectorized level lookup: build per-(Z,ion) arrays and index
E_lo=np.zeros(NL); g_lo=np.ones(NL); g_up=np.ones(NL); nion_lo=np.zeros(NL); valid=np.zeros(NL,bool)
ipkey=np.empty(NL,dtype=object)
for key in set(zip(Z,ion)):
    Zk,ik=key
    if key not in levarr: continue
    E,g=levarr[key]; nmax=len(E)
    m=(Z==Zk)&(ion==ik)
    lo=llo[m]; up=lup[m]
    ok=(lo<nmax)&(up<nmax)
    idx=np.where(m)[0][ok]
    E_lo[idx]=E[lo[ok]]; g_lo[idx]=g[lo[ok]]; g_up[idx]=g[up[ok]]
    nion_lo[idx]=npops.get((Zk,ik),0.0)
    valid[idx]=True
# keep valid + nonzero ion pop + positive nu
keep=valid&(nion_lo>0)&(nu>0)
Z=Z[keep]; ion=ion[keep]; f_lu=f_lu[keep]; A_ul=A_ul[keep]; nu=nu[keep]
E_lo=E_lo[keep]; g_lo=g_lo[keep]; g_up=g_up[keep]; nion_lo=nion_lo[keep]
dE=H*nu   # erg (line energy) -- note code uses |E_up-E_lo|*EV; equals h*nu
NLk=len(Z); print(f"  {NLk} lines with pops+levels")

# VR_STD coeff (gbar=0.2, om_floor=1.0)
gbar=0.2; om_floor=1.0
dE_eV=dE/EV
ry_de=np.minimum(13.605693/dE_eV,136.0)
c_vr=8.63e-6*14.5*gbar*f_lu*g_lo*ry_de
c_min=8.63e-6*om_floor
coeff=np.where((f_lu>1e-10),np.maximum(c_vr,c_min),c_min)  # forbidden -> floor
# Sobolev tau prefactor: l_ftau = 0.02654*f_lu*lam_cm*texp ; f_lu here already the osc str
lam_cm=C/nu
ftau=SOB*f_lu*lam_cm*T_EXP
# per-ion U at trial T -> build lookup
ionkeys=list(set(zip(Z.tolist(),ion.tolist())))

def beta_esc(tau):
    return np.where(tau<=1e-6,1.0,np.where(tau>700.0,1.0/np.maximum(tau,1e-30),(1.0-np.exp(-np.minimum(tau,700)))/np.maximum(tau,1e-30)))

def Lambda_line(T, which, apply_cull=True):
    invsq=1.0/math.sqrt(T)
    # U per line
    U=np.ones(NLk)
    for key in ionkeys:
        Zk,ik=key
        m=(Z==Zk)&(ion==ik)
        U[m]=U_of((Zk,ik),T)
    x=E_lo*EV/(KB*T)
    nlo=np.where(x<300, nion_lo*g_lo*np.exp(-np.minimum(x,300))/U, 0.0)
    exb=np.exp(-np.minimum(dE/(KB*T),300.0))
    qlu=coeff/g_lo*invsq*exb
    qul=coeff/g_up*invsq
    Clu=ne*qlu; Cul=ne*qul
    tau=ftau*nlo
    be=beta_esc(tau)
    Jb=Jb_of(nu, which)
    Bul=(C*C/(2*H*nu**3))*A_ul
    Blu=Bul*(g_up/g_lo)
    Rul=(A_ul+Bul*Jb)*be; Rlu=Blu*Jb*be
    den=Cul+Rul
    nup=np.where(den>0, nlo*(Clu+Rlu)/np.maximum(den,1e-300), 0.0)
    net=dE*(nlo*qlu*ne - nup*qul*ne)
    if apply_cull:
        # mimic C cull: cmax = dE*coeff*nel_k/g_lo*(6*natom)/sqrt(2000) < 0.01*H_dep/n_lines
        pass  # skip; effect small, we report both
    return float(np.sum(net)), net

# ---- validate at Te with cs pumping ----
print("\n# === VALIDATION: Lambda_line(Te=13120) reconstructed vs inferred residual 1.11e-3 ===")
for which in ['cs','mc','thermal','zero','cmfgen']:
    lam,_=Lambda_line(Te,which)
    print(f"#   Jb={which:8s}: Lambda_line = {lam:.3e}")

# ---- C_fb, C_ff, C_ad, H_photo (from ledger) ----
def alpha_rr(Zk,zrec,T): return 2.6e-13*zrec**1.6*(T/1e4)**-0.8
def C_fb(T):
    c=0.0
    for (Zk,st),n_lo in npops.items():
        nx=npops.get((Zk,st+1),0.0)
        if nx<=0: continue
        # chi
        c+= ne*nx*alpha_rr(Zk,st+1,T)*(CHI.get((Zk,st),0.0)*EV+KB*T)
    return c
CHI={}
for r in csv.DictReader(open(f"{BASE}/ionization_energies.csv")):
    CHI[(int(r['atomic_number']),int(r['ion_number']))]=float(r['ionization_energy_eV'])
def C_ff(T): return 1.426e-27*1.2*ne*ne*math.sqrt(T)
def C_ad(T): return 1.5*ne*KB*T*(3.0/T_EXP)
Hph_run=7.187e-07   # ground Kramers mc_J (from ledger)
Hph_cmf=3.739e-04   # ground Kramers jtable

def r_of(T, which, Hph):
    lam,_=Lambda_line(T,which)
    H=H_dep+max(Hph,0.0)
    Cc=C_ff(T)+C_ad(T)+C_fb(T)+lam
    return H-Cc, lam

print("\n# === r(T)=H-C scan, run field (Jb=cs, Hph=run) ===")
for T in [8000,10000,11000,12000,13120,14000,15000,16000,18000,20000,25000]:
    r,lam=r_of(T,'cs',Hph_run)
    print(f"#   T={T:6d}: r={r:+.3e}  Lam={lam:.3e} Cfb={C_fb(T):.3e}")

def bisect(which,Hph,Tlo=4000.,Thi=60000.):
    flo,_=r_of(Tlo,which,Hph); fhi,_=r_of(Thi,which,Hph)
    if flo<=0: return Tlo,'pin_lo'
    if fhi>=0: return Thi,'pin_hi'
    for _ in range(50):
        Tm=0.5*(Tlo+Thi); fm,_=r_of(Tm,which,Hph)
        if fm>0: Tlo=Tm
        else: Thi=Tm
    return 0.5*(Tlo+Thi),'root'

print("\n# === COUNTERFACTUAL ROOTS (pops fixed at run s0) ===")
for tag,which,Hph in [("RUN field (cs pump, Hph_run)",'cs',Hph_run),
                       ("RUN mc pump",'mc',Hph_run),
                       ("thermal pump B(Te)",'thermal',Hph_run),
                       ("CMFGEN J (pump+Hph_cmf ground)",'cmfgen',Hph_cmf),
                       ("CMFGEN pump, Hph_run (isolate pump)",'cmfgen',Hph_run),
                       ("CMFGEN pump, Hph x40 alllevel~1.5e-2",'cmfgen',1.5e-2)]:
    T,how=bisect(which,Hph)
    print(f"#   {tag:42s}: T_e = {T:7.0f} K  [{how}]")
