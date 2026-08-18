#!/usr/bin/env python3
"""kpr2 T_e warm-bias budget reconstruction (OFFLINE). Mirrors simul_r1 terms.
Shells s0/s4/s8. Uses kpr2's own committed field (cs_J line pump, mc_J Gph/Hex),
committed ion pops, and CMFGEN sigma_bf. No rerun."""
import csv, math, struct, numpy as np

H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; EV=1.602176634e-12
A_RAD=7.5657e-15; T_EXP=19.48*86400.0; SOB=2.6540281e-2
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN=f"{REPO}/logs/coevolve_consume_a10_kx_kpr2"
BASE=f"{REPO}/data/tardis_reference_toy06_19p48d"
NU_MIN=1.5e14; NU_MAX=3.0e16; NFB=1000
DLN=math.log(NU_MAX/NU_MIN)/NFB
nu_lo =NU_MIN*np.exp(np.arange(NFB)*DLN)
nu_mid=NU_MIN*np.exp((np.arange(NFB)+0.5)*DLN)
nu_hi =NU_MIN*np.exp((np.arange(NFB)+1)*DLN)
dnu   =nu_hi-nu_lo
hnu_mid=H*nu_mid

# ---- ionization energies ----
CHI={}
for r in csv.DictReader(open(f"{BASE}/ionization_energies.csv")):
    CHI[(int(r['atomic_number']),int(r['ion_number']))]=float(r['ionization_energy_eV'])*EV

# ---- levels.csv in GLOBAL order (matches sigma_bf rows) ----
lev_Z=[];lev_ion=[];lev_E=[];lev_g=[];lev_ln=[]
for r in csv.DictReader(open(f"{BASE}/levels.csv")):
    lev_Z.append(int(r['atomic_number']));lev_ion.append(int(r['ion_number']))
    lev_E.append(float(r['energy_eV'])*EV);lev_g.append(float(r['g']))
    lev_ln.append(int(r['level_number']))
lev_Z=np.array(lev_Z);lev_ion=np.array(lev_ion);lev_E=np.array(lev_E);lev_g=np.array(lev_g)
lev_ln=np.array(lev_ln)
NLEV=len(lev_Z)
# ion -> global level indices
ion_levels={}
for i in range(NLEV):
    ion_levels.setdefault((lev_Z[i],lev_ion[i]),[]).append(i)

# ---- sigma_bf.bin ----
with open(f"{BASE}/cmfgen_sigma_bf.bin",'rb') as f:
    magic,ver,nlv,nfr=struct.unpack('<IIii',f.read(16))
    numin,numax=struct.unpack('<dd',f.read(16))
    assert nlv==NLEV and nfr==NFB, (nlv,NLEV,nfr,NFB)
    flag8=np.frombuffer(f.read(nlv),dtype=np.int8)
    pad=(8-(nlv%8))%8; f.read(pad)
    sig=np.frombuffer(f.read(nlv*nfr*8),dtype=np.float64).reshape(nlv,nfr)
has_sig=flag8.astype(bool)
print(f"# sigma_bf: {NLEV} levels, {has_sig.sum()} with cmfgen sigma")

# ---- field (cs_J, mc_J per bin) per shell ----
field={}
import collections
tmp=collections.defaultdict(lambda:[np.full(NFB,1e-30),np.full(NFB,1e-30)])
with open(f"{RUN}/lumina_coevolve_field.csv") as f:
    rd=csv.reader(f);next(rd)
    for row in rd:
        s=int(row[0]);b=int(row[1])
        tmp[s][0][b]=float(row[3]);tmp[s][1][b]=float(row[4])
for s in (0,4,8): field[s]=tmp[s]

# ---- committed state ----
state={}
for r in csv.DictReader(open(f"{RUN}/lumina_plasma_state.csv")):
    state[int(r['shell_id'])]=(float(r['W']),float(r['T_rad']),float(r['n_e']),float(r['T_e']))

# ---- committed ion pops per shell (Z,stage)->n ----
pops=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{RUN}/lumina_ion_pops.csv")):
    pops[int(r['shell_id'])][(int(r['Z']),int(r['stage']))]=float(r['n_ion'])

# ---- deposition ----
Hdep={}
for r in csv.DictReader(open(f"{BASE}/deposition_cmfgen.csv")):
    Hdep[int(r['shell_id'])]=float(r['heating_rate'])

# ---- CMFGEN Te truth ----
CMFTe={0:18760.22,4:13657.24,8:10382.79}

# ---- levelpop NLTE n_k per (shell,Z,stage,level_num) ----
nk_map=collections.defaultdict(dict)   # shell -> (Z,stage,lnum)->n_k
with open(f"{RUN}/lumina_levelpop.csv") as f:
    rd=csv.reader(f);next(rd)
    for row in rd:
        s=int(row[0])
        if s not in (0,4,8): continue
        nk_map[s][(int(row[1]),int(row[2]),int(row[3]))]=float(row[6])
print("# levelpop n_k loaded for shells 0/4/8")

# ---- CMFGEN jtable field (per shell,bin) ----
with open(f"{REPO}/data/cmfgen_jtable_toy06_19p48d.bin",'rb') as fh:
    mg,ver2,nsh,nfbj=struct.unpack('<4i',fh.read(16))
    jt=np.frombuffer(fh.read(),np.float64).reshape(nsh,nfbj)
def cmfJ(s):
    a=jt[s].copy();a[a<=0]=1e-30;return a  # already on NLTE grid (nfbj bins)

def Bwien(T): return 2*H*nu_mid**3/C**2*np.exp(-np.minimum(hnu_mid/(KB*T),700))
def Bplanck(T):
    x=np.minimum(hnu_mid/(KB*T),700); return 2*H*nu_mid**3/C**2/np.expm1(x)

# ---------------- bf integral (H_photo / C_fb) ----------------
# Precompute per-shell per-pair the level list with sigma, chi_l, threshold bin.
def build_bf(s):
    """Per-pair level table with ABSOLUTE level populations n_level.
    Use NLTE n_k from levelpop where available; else Boltzmann*nion_committed."""
    W,Trad,ne,Te=state[s]; kT=KB*Te
    pairs=[]
    Zs=sorted(set(z for (z,st) in pops[s] if pops[s][(z,st)]>0))
    for Z in Zs:
        stages=sorted(st for (zz,st) in pops[s] if zz==Z and pops[s][(zz,st)]>0)
        for j in stages:
            chi_j=CHI.get((Z,j))
            if chi_j is None: continue
            nlo=pops[s].get((Z,j),0.0)
            if nlo<=0: continue
            gidx=ion_levels.get((Z,j),[])
            if not gidx: continue
            gidx=np.array(gidx)
            E=lev_E[gidx]; g=lev_g[gidx]; hs=has_sig[gidx]; ln=lev_ln[gidx]
            x=E/kT
            U=np.sum(g[x<50]*np.exp(-np.minimum(x[x<50],300)))
            if U<1.0: U=1.0
            sel=hs & (x<50) & ((chi_j-E)>0)
            if not sel.any(): continue
            gi=gidx[sel]; Es=E[sel]; gs=g[sel]; lns=ln[sel]
            chil=chi_j-Es; nul=chil/H
            # absolute level population
            nlev=np.zeros(len(gi))
            for k in range(len(gi)):
                v=nk_map[s].get((Z,j,int(lns[k])))
                nlev[k]=v if v is not None else nlo*gs[k]*math.exp(-Es[k]/kT)/U
            thr_bin=np.clip(np.floor(np.log(nul/NU_MIN)/DLN).astype(int),0,NFB-1)
            pairs.append((Z,j,gi,chil,nlev,thr_bin))
    return pairs,Te,ne

def bf_term(pairs,Jarr,per_pair=None):
    """sum over levels of n_level * integral 4pi sig (hnu-chi_l)/hnu J dnu."""
    tot=0.0
    for pi,(Z,j,gi,chil,nlev,thr) in enumerate(pairs):
        acc=0.0
        for k in range(len(gi)):
            l=gi[k]; b0=thr[k]
            sr=sig[l,b0:]
            m=sr>0
            if not m.any(): continue
            bb=np.arange(b0,NFB)[m]
            w=4*math.pi*sr[m]*Jarr[bb]/hnu_mid[bb]*dnu[bb]
            acc+=nlev[k]*np.sum(w*(hnu_mid[bb]-chil[k]))
        if per_pair is not None: per_pair[(Z,j)]=acc
        tot+=acc
    return tot

# ---------------- line cooling Lambda(cs_J,T) ----------------
import pandas as pd
print("# loading line list ...")
L=pd.read_csv(f"{BASE}/line_list.csv",
   usecols=['atomic_number','ion_number','level_number_lower','level_number_upper','f_lu','A_ul','nu'])
LZ=L['atomic_number'].values;Lion=L['ion_number'].values
llo=L['level_number_lower'].values;lup=L['level_number_upper'].values
f_lu=L['f_lu'].values;A_ul=L['A_ul'].values;lnu=L['nu'].values.astype(float)
# level arrays per (Z,ion) from levels.csv (level_number local)
tmpL={}
for i,(z,io,ln,e,gg) in enumerate(zip(lev_Z,lev_ion,
       [int(r['level_number']) for r in csv.DictReader(open(f"{BASE}/levels.csv"))], lev_E,lev_g)):
    tmpL.setdefault((z,io),{})[ln]=(e,gg)
levarr={}
for key,d in tmpL.items():
    n=max(d)+1;E=np.zeros(n);g=np.ones(n)
    for k,(e,gg) in d.items():E[k]=e;g[k]=gg
    levarr[key]=(E,g)
NL=len(LZ);E_lo=np.zeros(NL);g_lo=np.ones(NL);g_up=np.ones(NL);valid=np.zeros(NL,bool)
for key in set(zip(LZ,Lion)):
    if key not in levarr: continue
    E,g=levarr[key];nmax=len(E);m=(LZ==key[0])&(Lion==key[1])
    lo=llo[m];up=lup[m];ok=(lo<nmax)&(up<nmax);idx=np.where(m)[0][ok]
    E_lo[idx]=E[lo[ok]];g_lo[idx]=g[lo[ok]];g_up[idx]=g[up[ok]];valid[idx]=True
keep=valid&(lnu>0)
LZ=LZ[keep];Lion=Lion[keep];f_lu=f_lu[keep];A_ul=A_ul[keep];lnu=lnu[keep]
E_lo=E_lo[keep];g_lo=g_lo[keep];g_up=g_up[keep]
dE=H*lnu;NLk=len(LZ)
gbar=0.2;om_floor=1.0;ry_de=np.minimum(13.605693/(dE/EV),136.0)
coeff=np.where(f_lu>1e-10,np.maximum(8.63e-6*14.5*gbar*f_lu*g_lo*ry_de,8.63e-6*om_floor),8.63e-6*om_floor)
ftau=SOB*f_lu*(C/lnu)*T_EXP
Bul=(C*C/(2*H*lnu**3))*A_ul;Blu=Bul*(g_up/g_lo)
ionkeys=list(set(zip(LZ.tolist(),Lion.tolist())))
lineidx={key:np.where((LZ==key[0])&(Lion==key[1]))[0] for key in ionkeys}
def beta_esc(tau):
    return np.where(tau<=1e-6,1.0,np.where(tau>700,1.0/np.maximum(tau,1e-30),
                   (1.0-np.exp(-np.minimum(tau,700)))/np.maximum(tau,1e-30)))
def Jbin_at(nu,arr):
    b=np.clip(np.floor(np.log(nu/NU_MIN)/DLN).astype(int),0,NFB-1);out=arr[b].copy()
    out[(nu<=NU_MIN)|(nu>=NU_MAX)]=1e-30;return out
def U_of(key,T):
    E,g=levarr[key];x=E*EV*0+E/(KB*T);return float(np.sum(g*np.exp(-np.minimum(x,300))))

def Lambda(s,T,pop_map,Jarr):
    """two-level ETLA line cooling; pop_map: (Z,ion)->nion; Jb from Jarr (bin)."""
    W,Trad,ne,Te=state[s]
    nion_lo=np.zeros(NLk)
    for key in ionkeys: nion_lo[lineidx[key]]=pop_map.get(key,0.0)
    U=np.ones(NLk)
    for key in ionkeys: U[lineidx[key]]=U_of(key,T)
    x=E_lo/(KB*T);nlo=np.where(x<300,nion_lo*g_lo*np.exp(-np.minimum(x,300))/U,0.0)
    invsq=1/math.sqrt(T);exb=np.exp(-np.minimum(dE/(KB*T),300))
    qlu=coeff/g_lo*invsq*exb;qul=coeff/g_up*invsq;Clu=ne*qlu;Cul=ne*qul
    Jb=Jbin_at(lnu,Jarr)
    tau=ftau*nlo;be=beta_esc(tau);Rul=(A_ul+Bul*Jb)*be;Rlu=Blu*Jb*be;den=Cul+Rul
    nup=np.where(den>0,nlo*(Clu+Rlu)/np.maximum(den,1e-300),0.0)
    return float(np.sum(dE*(nlo*qlu*ne-nup*qul*ne)))

def Cff(s,T): _,_,ne,_=state[s];return 1.426e-27*1.2*ne*ne*math.sqrt(T)
def Cad(s,T): _,_,ne,_=state[s];return 1.5*ne*KB*T*(3.0/T_EXP)

# ---------------- main ----------------
for s in (0,4,8):
    W,Trad,ne,Te=state[s]
    print(f"\n{'='*70}\n### SHELL s{s}: committed T_e={Te:.0f} K  CMFGEN={CMFTe[s]:.0f} K  bias={Te-CMFTe[s]:+.0f} K")
    print(f"    n_e={ne:.3e}  W={W:.4f}  H_dep={Hdep[s]:.3e}")
    pairs,Te2,ne2=build_bf(s)
    csJ,mcJ=field[s];cmJ=cmfJ(s)
    pp={}
    Hph  =bf_term(pairs,mcJ,pp)      # heating (mc_J)
    Hph_cs=bf_term(pairs,csJ)        # heating if Gph read cs_J
    Hph_cmf=bf_term(pairs,cmJ)       # heating if Gph read CMFGEN J
    Cfb_W=bf_term(pairs,Bwien(Te))   # DBFB cooling (Wien)
    Cfb_P=bf_term(pairs,Bplanck(Te)) # full-Planck partner
    pop_map={(Z,st):pops[s].get((Z,st),0.0) for (Z,st) in pops[s]}
    Lam_cs=Lambda(s,Te,pop_map,csJ)
    Lam_mc=Lambda(s,Te,pop_map,mcJ)
    cff=Cff(s,Te);cad=Cad(s,Te)
    print(f"    -- terms @committed Te (erg/cm3/s), Hdep={Hdep[s]:.3e} --")
    print(f"    H_photo(mc_J)   = {Hph:+.3e} ({100*Hph/Hdep[s]:+6.1f}% Hdep)   H_photo(cs_J)={Hph_cs:+.3e}  H_photo(CMFGEN_J)={Hph_cmf:+.3e}")
    print(f"    C_fb(Wien,Te)   = {Cfb_W:.3e} ({100*Cfb_W/Hdep[s]:6.1f}% Hdep)   C_fb(Planck)={Cfb_P:.3e}  Wien-defect={Cfb_P-Cfb_W:.3e} ({100*(Cfb_P-Cfb_W)/max(Cfb_W,1e-30):.2f}% of Cfb)")
    print(f"    net_bf(mc-Wien) = {Hph-Cfb_W:+.3e}   net_bf(CMFGENj-Wien)={Hph_cmf-Cfb_W:+.3e}")
    print(f"    Lambda(cs_J)    = {Lam_cs:+.3e} ({100*Lam_cs/Hdep[s]:+6.1f}% Hdep)   Lambda(mc_J)={Lam_mc:+.3e}  (split cs-mc={Lam_cs-Lam_mc:+.3e})")
    print(f"    C_ff={cff:.3e}  C_ad={cad:.3e}")
    # top pairs by |H_photo| contribution
    top=sorted(pp.items(),key=lambda kv:-abs(kv[1]))[:5]
    print("    top bf pairs (H_photo/ion contribution):",", ".join(f"Z{z}+{st}:{v:.2e}" for (z,st),v in top))
    r_committed=Hdep[s]+Hph-cff-cad-Cfb_W-Lam_cs
    print(f"    residual r(Te)  = {r_committed:+.3e}  ({100*r_committed/Hdep[s]:+.1f}% Hdep) [~0 if faithful]")
    def r_of(T):
        cW=bf_term(pairs,Bwien(T))
        lam=Lambda(s,T,pop_map,csJ)
        return Hdep[s]+Hph-Cff(s,T)-Cad(s,T)-cW-lam
    dT=500.0
    slope=(r_of(Te+dT)-r_of(Te-dT))/(2*dT)
    # reconstructed root
    Tlo,Thi=6000.0,Te+2000
    rl=r_of(Tlo)
    root=None
    if rl>0:
        for _ in range(45):
            Tm=0.5*(Tlo+Thi)
            if r_of(Tm)>0: Tlo=Tm
            else: Thi=Tm
        root=0.5*(Tlo+Thi)
    print(f"    dr/dT={slope:+.3e}/K   reconstructed root={root if root is None else f'{root:.0f}'} K (committed {Te:.0f})")
    dr_C1=-(Cfb_P-Cfb_W); dT_C1=-dr_C1/slope
    dr_C2=-(Lam_mc-Lam_cs); dT_C2=-dr_C2/slope
    dr_C2h=-(Hph_cmf-Hph); dT_C2h=-dr_C2h/slope  # Gph/Hex mc_J -> CMFGEN J
    print(f"    >> C1 Wien->Planck:      dr={dr_C1:+.2e} -> dTe={dT_C1:+.0f} K")
    print(f"    >> C2a pump cs->mc:      dr={dr_C2:+.2e} -> dTe={dT_C2:+.0f} K")
    print(f"    >> C2b Gph/Hex mc->CMFGEN: dr={dr_C2h:+.2e} -> dTe={dT_C2h:+.0f} K")
