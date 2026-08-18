#!/usr/bin/env python3
"""bistability.py -- the photospheric warm-loop bistability discriminator (OFFLINE, kpr5).

Reconstructs the CODE's simul_r1 energy residual r(T)=H-C AND the code's simul_ladder
ionization re-solve, faithfully to src/lumina_plasma.c:
  * ionization:  r_j = (Gph_j(T) + gnt_j) / (n_e * alpha_j(T))       [simul_ladder 5010-5048]
  * Gph_j(T)  = sum_l (g_l e^{-E_l/kT}/U_j(T)) * R_l   [Boltzmann loop 5871-5924, W<0.13]
  * alpha_j(T)= Milne RR (R_planck) + DR (dr_alpha_eval)             [frozenin_alpha_rr 2827-2958]
  * energy:    r = Hdep + sum_p n_p Hex_p - Cff - Cad - Cfb(DBFB) - Lambda_ETLA [simul_r1 5076-5142]

Field = kpr5's OWN gphJ = mc_J if mc_J>0 else cs_J (db_photoion.field convention).
Two modes at s8/s6/s2 over T in [9000,22000]:
  (a) FROZEN   : ion fractions held at kpr5 committed; Boltzmann weights, Cfb, Cff, Lambda vary with T
  (b) COUPLED  : re-solve the full ladder at each T (Gph(T),alpha(T)) -> new fractions -> full ledger
Plus the CMFGEN-state residual (T=10383, CMFGEN pops, CMFGEN jtable field).
Every number reproducible from committed data. No rerun/edit/GPU.
"""
import csv, math, struct, re, sys, collections
import numpy as np

H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10; ME=9.1093837015e-28
EV=1.602176634e-12; PI=math.pi; KB_EV=8.617333262e-5
T_EXP=19.48*86400.0; SOB=2.6540281e-2
ETA_NT=0.05; W_ION_ERG=35.0*EV; NTP=2.0
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN=f"{REPO}/logs/coevolve_consume_a10_kx_kpr5"
BASE=f"{REPO}/data/tardis_reference_toy06_19p48d"
OUT=f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/bistability_test"
NU_MIN=1.5e14; NU_MAX=3.0e16; NFB=1000; DLN=math.log(NU_MAX/NU_MIN)/NFB
nu_lo=NU_MIN*np.exp(np.arange(NFB)*DLN); nu_mid=NU_MIN*np.exp((np.arange(NFB)+0.5)*DLN)
nu_hi=NU_MIN*np.exp((np.arange(NFB)+1)*DLN); dnu=nu_hi-nu_lo; hnu_mid=H*nu_mid
SHELLS=[8,6,2]
CMFTe={0:18760.22,2:16351.43,4:13657.24,6:11929.14,8:10383.0}
VEL={0:4264.0,2:5720.0,4:7176.0,6:8632.0,8:10088.0}

# ---------- ionization energies ----------
CHI={}
for r in csv.DictReader(open(f"{BASE}/ionization_energies.csv")):
    CHI[(int(r['atomic_number']),int(r['ion_number']))]=float(r['ionization_energy_eV'])*EV

# ---------- levels (canonical order == sigma rows) ----------
lev_Z=[];lev_ion=[];lev_E=[];lev_g=[];lev_ln=[]
for r in csv.DictReader(open(f"{BASE}/levels.csv")):
    lev_Z.append(int(r['atomic_number']));lev_ion.append(int(r['ion_number']))
    lev_E.append(float(r['energy_eV'])*EV);lev_g.append(float(r['g']));lev_ln.append(int(r['level_number']))
lev_Z=np.array(lev_Z);lev_ion=np.array(lev_ion);lev_E=np.array(lev_E);lev_g=np.array(lev_g);lev_ln=np.array(lev_ln)
NLEV=len(lev_Z); ion_levels={}
for i in range(NLEV): ion_levels.setdefault((lev_Z[i],lev_ion[i]),[]).append(i)
# partition-function level set per ion (ALL levels, x<50) for U(T)
ion_all=ion_levels

with open(f"{BASE}/cmfgen_sigma_bf.bin",'rb') as f:
    magic,ver,nlv,nfr=struct.unpack('<IIii',f.read(16)); numin,numax=struct.unpack('<dd',f.read(16))
    assert nlv==NLEV and nfr==NFB
    flag8=np.frombuffer(f.read(nlv),dtype=np.int8); pad=(8-(nlv%8))%8; f.read(pad)
    sig=np.frombuffer(f.read(nlv*nfr*8),dtype=np.float64).reshape(nlv,nfr)
has_sig=flag8.astype(bool)

# ---------- kpr5 field: gphJ = mc if mc>0 else cs ----------
csJ=np.full((50,NFB),1e-30); mcJ=np.full((50,NFB),1e-30)
with open(f"{RUN}/lumina_coevolve_field.csv") as f:
    rd=csv.reader(f); next(rd)
    for row in rd:
        s=int(row[0]); b=int(row[1]); csJ[s,b]=float(row[3]); mcJ[s,b]=float(row[4])
gphJ=np.where(mcJ>0,mcJ,csJ)

# ---------- state, ion pops, deposition, CMFGEN jtable ----------
state={}
for r in csv.DictReader(open(f"{RUN}/lumina_plasma_state.csv")):
    state[int(r['shell_id'])]=(float(r['W']),float(r['T_rad']),float(r['n_e']),float(r['T_e']))
pops=collections.defaultdict(dict); natom_sh=collections.defaultdict(float)
for r in csv.DictReader(open(f"{RUN}/lumina_ion_pops.csv")):
    s=int(r['shell_id']); pops[s][(int(r['Z']),int(r['stage']))]=float(r['n_ion']); natom_sh[s]+=float(r['n_ion'])
Hdep={}
for r in csv.DictReader(open(f"{BASE}/deposition_cmfgen.csv")):
    Hdep[int(r['shell_id'])]=float(r['heating_rate'])
with open(f"{REPO}/data/cmfgen_jtable_toy06_19p48d.bin",'rb') as fh:
    struct.unpack('<4i',fh.read(16)); jt=np.frombuffer(fh.read(),np.float64).reshape(50,NFB)
def cmfJ(s): a=jt[s].copy(); a[a<=0]=1e-30; return a

# ---------- parse DR_TABLE from source ----------
src=open(f"{REPO}/src/lumina_plasma.c").read()
m=re.search(r"static const DRCoefficient DR_TABLE\[\] = \{(.*?)\n\};",src,re.S)
block=m.group(1)
# each entry: {Z, ion_recomb, n, {c...}, {E...}, DR_SOURCE_X}
DR={}
for em in re.finditer(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*,\s*(DR_SOURCE_\w+)\s*\}",block):
    Z=int(em.group(1)); ion_recomb=int(em.group(2)); n=int(em.group(3))
    ci=[float(x) for x in em.group(4).split(',') if x.strip()]
    Ei=[float(x) for x in em.group(5).split(',') if x.strip()]
    DR[(Z,ion_recomb)]=(np.array(ci),np.array(Ei))   # boost=1.0 (kpr5: no DR_BOOST)
def alpha_dr(Z,stage,T):
    # frozenin_alpha_rr: dr_lookup(Z, stage+1); stage = lower-ion stage index
    key=(Z,stage+1)
    if key not in DR: return 0.0
    ci,Ei=DR[key]; Tc=max(T,1.0)
    return float(np.sum(ci*np.exp(np.maximum(-Ei/Tc,-700))))*Tc**-1.5

# ---------- Milne RR alpha (frozenin_alpha_rr, RR part) ----------
def U_of(Z,ion,T):
    idx=ion_all.get((Z,ion),[])
    if not idx: return 1.0
    x=lev_E[np.array(idx)]/(KB*T); u=float(np.sum(np.where(x<50,lev_g[np.array(idx)]*np.exp(-np.minimum(x,50)),0.0)))
    return u if u>=1.0 else max(1.0,lev_g[idx[0]])
# precompute per-ion Milne contributors (level masks) so alpha(T) is a few vectorized sums
_alpha_lev={}
def _alpha_prep(Z,ion):
    if (Z,ion) in _alpha_lev: return _alpha_lev[(Z,ion)]
    chi0=CHI.get((Z,ion)); out=[]
    if chi0 is not None:
        for gl in ion_all.get((Z,ion),[]):
            if not has_sig[gl]: continue
            chi_l=chi0-lev_E[gl]
            if chi_l<=0: continue
            nu_th=chi_l/H; mk=(nu_mid>=nu_th)&(sig[gl]>0)
            if not mk.any(): continue
            out.append((float(chi_l),float(lev_g[gl]),nu_mid[mk],sig[gl][mk],dnu[mk]))
    _alpha_lev[(Z,ion)]=out; return out
def alpha_milne(Z,ion,T):
    levs=_alpha_prep(Z,ion)
    if not levs: return 0.0
    Uup=U_of(Z,ion+1,T); lam3=(H*H/(2*PI*ME*KB*T))**1.5; kT=KB*T
    a=0.0
    for chi_l,g,nuM,sM,dM in levs:
        x=H*nuM/(KB*T); ok=x<700
        if not ok.any(): continue
        B=2*H*nuM[ok]**3/C**2/np.expm1(x[ok])
        Rbf=float(np.sum(4*PI*B*sM[ok]/(H*nuM[ok])*dM[ok]))
        a+=Rbf*lam3*g/(2*Uup)*math.exp(min(chi_l/kT,300))
    return a
def alpha_full(Z,ion,T): return alpha_milne(Z,ion,T)+alpha_dr(Z,ion,T)   # RR+DR, FROZENIN_DR=1

# ---------- per-pair precompute: level list + field-driven R_l (ioniz) and Rexc_l (heat) ----------
# built once per shell from gphJ (field frozen); T only enters through Boltzmann weights.
def build_pairs(s,Jarr):
    W,Trad,ne,Te=state[s]
    pairs={}   # (Z,ion) -> dict of arrays
    Zs=sorted(set(z for (z,st) in pops[s] if pops[s][(z,st)]>0))
    for Z in Zs:
        stages=sorted(st for (zz,st) in pops[s] if zz==Z and pops[s][(zz,st)]>0)
        for j in stages:
            chi_j=CHI.get((Z,j))
            if chi_j is None: continue
            gidx=np.array(ion_all.get((Z,j),[]))
            if gidx.size==0: continue
            E=lev_E[gidx]; g=lev_g[gidx]; hs=has_sig[gidx]
            sel=hs&((chi_j-E)>0)
            if not sel.any(): continue
            gi=gidx[sel]; Es=E[sel]; gs=g[sel]; chil=chi_j-Es
            Rl=np.zeros(len(gi)); Rexc=np.zeros(len(gi))   # field-driven, per unit level pop
            cb_l=[];cb_nu=[];cb_sig=[];cb_dnu=[];cb_chi=[]   # concat for vectorized DBFB cooling
            for k in range(len(gi)):
                l=gi[k]; nu_th=chil[k]/H
                mk=(nu_mid>=nu_th)&(sig[l]>0)&(Jarr>0)
                if mk.any():
                    w=4*PI*sig[l][mk]*Jarr[mk]/(H*nu_mid[mk])*dnu[mk]
                    Rl[k]=np.sum(w); Rexc[k]=np.sum(w*(hnu_mid[mk]-chil[k]))
                mc=(nu_mid>=nu_th)&(sig[l]>0)
                if mc.any():
                    nn=int(mc.sum())
                    cb_l.append(np.full(nn,k)); cb_nu.append(nu_mid[mc]); cb_sig.append(sig[l][mc])
                    cb_dnu.append(dnu[mc]); cb_chi.append(np.full(nn,chil[k]))
            if cb_nu:
                cb_l=np.concatenate(cb_l);cb_nu=np.concatenate(cb_nu);cb_sig=np.concatenate(cb_sig)
                cb_dnu=np.concatenate(cb_dnu);cb_chi=np.concatenate(cb_chi)
            else:
                cb_l=np.array([],int);cb_nu=cb_sig=cb_dnu=cb_chi=np.array([])
            pairs[(Z,j)]=dict(gi=gi,E=Es,g=gs,chil=chil,Rl=Rl,Rexc=Rexc,
                              cb_l=cb_l,cb_nu=cb_nu,cb_sig=cb_sig,cb_dnu=cb_dnu,cb_chi=cb_chi,cb_hnu=H*cb_nu)
    return pairs
# Cfb (DBFB) per pair at trial T: sum_l boltz_l(T) * [4pi int sig Bwien(T)(hnu-chi)/hnu dnu]
def Bwien(T): return 2*H*nu_mid**3/C**2*np.exp(-np.minimum(hnu_mid/(KB*T),700))
def pair_boltz(pr,Z,j,T):
    U=U_of(Z,j,T); x=pr['E']/(KB*T)
    return np.where(x<50,pr['g']*np.exp(-np.minimum(x,50))/U,0.0)
def gph_of(pr,Z,j,T):     # per-lower-ion ionization rate [s^-1]
    b=pair_boltz(pr,Z,j,T); return float(np.sum(b*pr['Rl']))
def hexfrac_of(pr,Z,j,T): # per-lower-ion bf excess heating [erg/s]
    b=pair_boltz(pr,Z,j,T); return float(np.sum(b*pr['Rexc']))
def cfbfrac_of(pr,Z,j,T): # per-lower-ion DBFB cooling [erg/s]  (Wien partner, vectorized)
    if pr['cb_nu'].size==0: return 0.0
    b=pair_boltz(pr,Z,j,T)
    Bw=2*H*pr['cb_nu']**3/C**2*np.exp(-np.minimum(pr['cb_hnu']/(KB*T),700))
    integ=4*PI*pr['cb_sig']*Bw/pr['cb_hnu']*(pr['cb_hnu']-pr['cb_chi'])*pr['cb_dnu']
    return float(np.sum(b[pr['cb_l']]*integ))

# ---------- ETLA line cooling Lambda (ported from residual_ledger_kpr4.py, validated) ----------
import pandas as pd
L=pd.read_csv(f"{BASE}/line_list.csv",usecols=['atomic_number','ion_number','level_number_lower','level_number_upper','f_lu','A_ul','nu'])
LZ=L['atomic_number'].values;Lion=L['ion_number'].values;llo=L['level_number_lower'].values;lup=L['level_number_upper'].values
f_lu=L['f_lu'].values;A_ul=L['A_ul'].values;lnu=L['nu'].values.astype(float)
tmpL={}
for i,(z,io,ln,e,gg) in enumerate(zip(lev_Z,lev_ion,lev_ln,lev_E,lev_g)):
    tmpL.setdefault((z,io),{})[ln]=(e,gg)
levarr={}
for key,d in tmpL.items():
    n=max(d)+1;E=np.zeros(n);g=np.ones(n)
    for k,(e,gg) in d.items():E[k]=e;g[k]=gg
    levarr[key]=(E,g)
NL=len(LZ);E_lo=np.zeros(NL);g_lo=np.ones(NL);g_up=np.ones(NL);valid=np.zeros(NL,bool)
for key in set(zip(LZ,Lion)):
    if key not in levarr: continue
    E,g=levarr[key];nmax=len(E);m=(LZ==key[0])&(Lion==key[1]);lo=llo[m];up=lup[m];ok=(lo<nmax)&(up<nmax);idx=np.where(m)[0][ok]
    E_lo[idx]=E[lo[ok]];g_lo[idx]=g[lo[ok]];g_up[idx]=g[up[ok]];valid[idx]=True
keep=valid&(lnu>0)
LZ=LZ[keep];Lion=Lion[keep];f_lu=f_lu[keep];A_ul=A_ul[keep];lnu=lnu[keep];E_lo=E_lo[keep];g_lo=g_lo[keep];g_up=g_up[keep]
dE=H*lnu;NLk=len(LZ)
gbar=0.2;om_floor=1.0;ry_de=np.minimum(13.605693/(dE/EV),136.0)
coeff=np.where(f_lu>1e-10,np.maximum(8.63e-6*14.5*gbar*f_lu*g_lo*ry_de,8.63e-6*om_floor),8.63e-6*om_floor)
ftau=SOB*f_lu*(C/lnu)*T_EXP;Bul=(C*C/(2*H*lnu**3))*A_ul;Blu=Bul*(g_up/g_lo)
ionkeys=list(set(zip(LZ.tolist(),Lion.tolist())));lineidx={key:np.where((LZ==key[0])&(Lion==key[1]))[0] for key in ionkeys}
def beta_esc(tau): return np.where(tau<=1e-6,1.0,np.where(tau>700,1.0/np.maximum(tau,1e-30),(1.0-np.exp(-np.minimum(tau,700)))/np.maximum(tau,1e-30)))
def Jbin_at(nu,arr):
    b=np.clip(np.floor(np.log(nu/NU_MIN)/DLN).astype(int),0,NFB-1);out=arr[b].copy();out[(nu<=NU_MIN)|(nu>=NU_MAX)]=1e-30;return out
def Uline(key,T): E,g=levarr[key];return float(np.sum(g*np.exp(-np.minimum(E/(KB*T),300))))
def Lambda(s,T,pop_map,Jarr,ne):
    nion_lo=np.zeros(NLk)
    for key in ionkeys: nion_lo[lineidx[key]]=pop_map.get(key,0.0)
    U=np.ones(NLk)
    for key in ionkeys: U[lineidx[key]]=Uline(key,T)
    x=E_lo/(KB*T);nlo=np.where(x<300,nion_lo*g_lo*np.exp(-np.minimum(x,300))/U,0.0)
    invsq=1/math.sqrt(T);exb=np.exp(-np.minimum(dE/(KB*T),300))
    qlu=coeff/g_lo*invsq*exb;qul=coeff/g_up*invsq;Clu=ne*qlu;Cul=ne*qul
    Jb=Jbin_at(lnu,Jarr);tau=ftau*nlo;be=beta_esc(tau)
    Rul=(A_ul+Bul*Jb)*be;Rlu=Blu*Jb*be;den=Cul+Rul
    nup=np.where(den>0,nlo*(Clu+Rlu)/np.maximum(den,1e-300),0.0)
    return float(np.sum(dE*(nlo*qlu*ne-nup*qul*ne)))

# ---------- ladder re-solve (simul_ladder mirror) ----------
def elements_of(s): return sorted(set(z for (z,st) in pops[s]))
def gnt_atom(s):
    return ETA_NT*Hdep[s]/W_ION_ERG/natom_sh[s]
def resolve_ladder(s,pairs,T,ne0,alpha_lut,fix_ne=False):
    """Return dict (Z,stage)->n_abs, and n_e. Mirrors simul_ladder: r=(Gph+gnt)/(ne*alpha)."""
    W,Trad,ne_c,Te=state[s]; ne=ne0
    gnt=gnt_atom(s)
    # element -> ordered stage list and n_element
    elem={}
    for (Z,st),n in pops[s].items():
        elem.setdefault(Z,{})[st]=n
    eldata={}
    for Z,d in elem.items():
        stg=sorted(d); nel=sum(d.values()); eldata[Z]=(stg,nel)
    frac_out={}
    for _ in range(30):
        ne_new=0.0
        for Z,(stg,nel) in eldata.items():
            y=[1.0]
            for a in range(len(stg)-1):
                j=stg[a]
                if (Z,j) in pairs:
                    G=gph_of(pairs[(Z,j)],Z,j,T)
                else:
                    G=0.0
                chi_eV=CHI.get((Z,j),None)
                gnt_j=gnt*((35.0/(chi_eV/EV))**NTP if chi_eV and chi_eV/EV>1.0 else 0.0)
                al=alpha_lut.get((Z,j),0.0)
                r=(G+gnt_j)/(ne*al) if (al>0 and ne>0) else 0.0
                if not math.isfinite(r) or r<0: r=0.0
                if r>1e28: r=1e28
                # top-ion Saha cap: if next stage has no levels, truncate
                if (Z,j+1) not in ion_all or len(ion_all.get((Z,j+1),[]))==0: r=0.0
                y.append(y[-1]*r)
            ys=sum(y)
            zbar=0.0
            for a,j in enumerate(stg):
                fr=y[a]/ys if ys>0 else (1.0 if a==0 else 0.0)
                frac_out[(Z,j)]=nel*fr
                zbar+=j*fr
            ne_new+=nel*zbar
        if fix_ne:
            ne=ne_c; break
        ne_next=0.5*(ne+max(ne_new,1e-6*natom_sh[s]))
        if abs(ne_next-ne)<1e-3*ne: ne=ne_next; break
        ne=ne_next
    return frac_out,ne

# ---------- energy residual ----------
def energy_residual(s,pairs,T,nabs,ne,Jarr):
    Hd=Hdep[s]; Hph=0.0; Cfb=0.0
    for (Z,j),pr in pairs.items():
        n_low=nabs.get((Z,j),0.0)
        if n_low<=0: continue
        Hph+=n_low*hexfrac_of(pr,Z,j,T)
        Cfb+=n_low*cfbfrac_of(pr,Z,j,T)
    Cff=1.426e-27*1.2*ne*ne*math.sqrt(T)
    Cad=1.5*ne*KB*T*(3.0/T_EXP)
    pop_map={(Z,st):nabs.get((Z,st),0.0) for (Z,st) in pops[s]}
    Lam=Lambda(s,T,pop_map,Jarr,ne)
    r=Hd+Hph-Cff-Cad-Cfb-Lam
    return dict(r=r,Hph=Hph,Cfb=Cfb,Cff=Cff,Cad=Cad,Lam=Lam,Hd=Hd)

def find_roots(Tgrid,rvals):
    roots=[]
    for i in range(len(Tgrid)-1):
        if rvals[i]==0: roots.append(Tgrid[i])
        elif rvals[i]*rvals[i+1]<0:
            t0,t1,r0,r1=Tgrid[i],Tgrid[i+1],rvals[i],rvals[i+1]
            roots.append(t0-r0*(t1-t0)/(r1-r0))
    return roots

# ======================= RUN =======================
Tgrid=np.linspace(9000,22000,27)   # 500 K steps (root-detection + directional)
print("="*100)
print("BISTABILITY DISCRIMINATOR (kpr5, own gphJ field). r(T)=H-C.  gnt(NT) included (negligible for Fe).")
print("="*100)
val_rows=[]; curve_rows=[]; ion_rows=[]
for s in SHELLS:
    W,Trad,ne_c,Te=state[s]
    pairs=build_pairs(s,gphJ[s])
    # ---- round-trip validation: committed r34 (Fe III->IV) vs reconstructed at Te ----
    nFeIII=pops[s].get((26,2),0.0); nFeIV=pops[s].get((26,3),0.0)
    r34_comm=nFeIV/nFeIII if nFeIII>0 else float('nan')
    G_FeIII=gph_of(pairs[(26,2)],26,2,Te) if (26,2) in pairs else 0.0
    al_FeIII=alpha_full(26,2,Te); al_rr=alpha_milne(26,2,Te); al_drv=alpha_dr(26,2,Te)
    gnt=gnt_atom(s); gnt_Fe=gnt*(35.0/30.6514)**NTP
    r34_recon=(G_FeIII+gnt_Fe)/(ne_c*al_FeIII) if al_FeIII>0 else float('nan')
    print(f"\n{'#'*90}\n### s{s}: W={W:.4f} n_e={ne_c:.3e} T_e(kpr5)={Te:.0f} CMFGEN={CMFTe[s]:.0f} (offset {Te-CMFTe[s]:+.0f})")
    print(f"  [VALIDATION Fe III->IV]  committed r34={r34_comm:.3e} (f_IV={r34_comm/(1+r34_comm):.4f})")
    print(f"     reconstructed r34={r34_recon:.3e} (f_IV={r34_recon/(1+r34_recon):.4f})  ratio recon/comm={r34_recon/r34_comm:.3f}")
    print(f"     Gph_FeIII(Te)={G_FeIII:.3e}/s  gnt_Fe={gnt_Fe:.3e}/s ({100*gnt_Fe/(G_FeIII+1e-30):.2f}% of Gph)  alpha=RR {al_rr:.3e}+DR {al_drv:.3e}={al_FeIII:.3e}")
    val_rows.append(dict(s=s,W=W,ne=ne_c,Te=Te,cmf=CMFTe[s],r34_comm=r34_comm,r34_recon=r34_recon,
                         Gph_FeIII=G_FeIII,gnt_Fe=gnt_Fe,alpha_rr=al_rr,alpha_dr=al_drv))
    # committed absolute pops for frozen mode
    nabs_comm={(Z,st):pops[s].get((Z,st),0.0) for (Z,st) in pops[s]}
    # ---- MODE A (frozen ionization) and MODE B (coupled) over T ----
    rA=[]; rB=[]; f4A=[]; f4B=[]; neB=[]
    detA=[]; detB=[]
    ions_here=sorted(set((Z,st) for (Z,st) in pops[s]))
    for T in Tgrid:
        alut={(Z,st):alpha_full(Z,st,T) for (Z,st) in ions_here}
        eA=energy_residual(s,pairs,T,nabs_comm,ne_c,gphJ[s]); rA.append(eA['r']); detA.append(eA)
        f4A.append(r34_comm/(1+r34_comm))  # frozen fraction (const)
        nabsB,neb=resolve_ladder(s,pairs,T,ne_c,alut)
        eB=energy_residual(s,pairs,T,nabsB,neb,gphJ[s]); rB.append(eB['r']); detB.append(eB)
        nfe3=nabsB.get((26,2),0.0); nfe4=nabsB.get((26,3),0.0)
        f4B.append(nfe4/max(nfe3+nfe4,1e-30)); neB.append(neb)
    rA=np.array(rA); rB=np.array(rB)
    rootsA=find_roots(Tgrid,rA); rootsB=find_roots(Tgrid,rB)
    print(f"  MODE A (frozen)  roots(K): {['%.0f'%x for x in rootsA]}   r(9000)={rA[0]:+.2e} r(10383)={np.interp(10383,Tgrid,rA):+.2e} r(Te)={np.interp(Te,Tgrid,rA):+.2e}")
    print(f"  MODE B (coupled) roots(K): {['%.0f'%x for x in rootsB]}   r(9000)={rB[0]:+.2e} r(10383)={np.interp(10383,Tgrid,rB):+.2e} r(Te)={np.interp(Te,Tgrid,rB):+.2e}")
    print(f"     coupled f(FeIV): T=9000->{f4B[0]:.4f}  T=10383->{np.interp(10383,Tgrid,f4B):.4f}  T=12208->{np.interp(12208,Tgrid,f4B):.4f}  T=22000->{f4B[-1]:.4f}")
    # term breakdown at CMFGEN T (guilty-term analysis): what keeps r>0 at cold T?
    ic=int(np.argmin(abs(Tgrid-CMFTe[s])))
    dB=detB[ic]
    print(f"     [terms @T={Tgrid[ic]:.0f}~CMFGEN, MODE B]  Hdep={Hdep[s]:.3e}  Hph={dB['Hph']:+.3e}  Cfb={dB['Cfb']:.3e}  Lam={dB['Lam']:+.3e}  Cff={dB['Cff']:.2e}  Cad={dB['Cad']:.2e}  => r={dB['r']:+.3e}")
    print(f"        net_bf(Hph-Cfb)={dB['Hph']-dB['Cfb']:+.3e}  radiated(Cfb+Lam+Cff+Cad)={dB['Cfb']+dB['Lam']+dB['Cff']+dB['Cad']:.3e}  vs (Hdep+Hph)={Hdep[s]+dB['Hph']:.3e}")
    for k,T in enumerate(Tgrid):
        curve_rows.append(dict(s=s,T=T,rA=rA[k],rB=rB[k],f4A=f4A[k],f4B=f4B[k],neB=neB[k],
                               HphA=detA[k]['Hph'],CfbA=detA[k]['Cfb'],LamA=detA[k]['Lam'],
                               HphB=detB[k]['Hph'],CfbB=detB[k]['Cfb'],LamB=detB[k]['Lam'],
                               CffB=detB[k]['Cff'],CadB=detB[k]['Cad']))

# ---- CMFGEN-state residual: T=CMFTe, CMFGEN field, CMFGEN Fe/Co/Ni fractions ----
def parse_ionfrac(fn,tgt=19.48):
    Ln=open(fn).read().splitlines();times=[];st=[]
    for i,l in enumerate(Ln):
        mm=re.match(r'#TIME:\s*([\d.]+)',l)
        if mm:times.append(float(mm.group(1)));st.append(i)
    k=int(np.argmin(abs(np.array(times)-tgt)));i0=st[k];i1=st[k+1] if k+1<len(st) else len(Ln)
    cols=None;v=[];d=[]
    for l in Ln[i0:i1]:
        if l.startswith('#vel_mid'):cols=l.strip().lstrip('#').split()[1:];continue
        if l.startswith('#'):continue
        p=l.split()
        try:vals=[float(x) for x in p]
        except:continue
        if len(p)>=2:v.append(vals[0]);d.append(vals[1:])
    return np.array(v),{c:np.array(d)[:,j] for j,c in enumerate(cols)}
elem_files={26:'fe',27:'co',28:'ni',14:'si',16:'s',20:'ca'}
cmf_frac={}
for Z,tag in elem_files.items():
    vcm,dcm=parse_ionfrac(f"{REPO}/data/standart_data1/toy06/ionfrac_{tag}_toy06_cmfgen.txt")
    cmf_frac[Z]=(vcm,dcm)
def cmf_ion_abs(s):
    """CMFGEN ion fractions -> absolute pops using kpr5 n_element (Fe/Co/Ni/Si/S/Ca CMFGEN; others kpr5)."""
    out=dict(pops[s])
    for Z,(vcm,dcm) in cmf_frac.items():
        nel=sum(pops[s].get((Z,st),0.0) for st in range(7))
        fr={}
        tot=0.0
        for stg in range(6):
            key=f'{elem_files[Z]}{stg}'
            if key in dcm: fr[stg]=max(np.interp(VEL[s],vcm,dcm[key]),0.0); tot+=fr[stg]
        if tot>0:
            for stg,fv in fr.items(): out[(Z,stg)]=nel*fv/tot
    return out
print(f"\n{'='*100}\nCMFGEN-STATE RESIDUAL: is CMFGEN's own state a root of OUR ledger? (T=CMFTe, CMFGEN jtable field, CMFGEN Fe/Co/Ni/Si/S/Ca fractions)")
print(f"{'='*100}")
cmf_state_rows=[]
for s in SHELLS:
    W,Trad,ne_c,Te=state[s]; Tc=CMFTe[s]
    pairs_cmf=build_pairs(s,cmfJ(s))
    nabs_cmf=cmf_ion_abs(s)
    # n_e from CMFGEN fractions
    ne_cmf=sum(st*nabs_cmf.get((Z,st),0.0) for (Z,st) in nabs_cmf)
    e=energy_residual(s,pairs_cmf,Tc,nabs_cmf,ne_cmf,cmfJ(s))
    nfe3=nabs_cmf.get((26,2),0.0);nfe4=nabs_cmf.get((26,3),0.0)
    print(f"  s{s}: T={Tc:.0f} ne_cmf={ne_cmf:.3e} f(FeIV)={nfe4/max(nfe3+nfe4,1e-30):.4f} | "
          f"r={e['r']:+.3e} ({100*e['r']/Hdep[s]:+.1f}% Hdep)  Hph={e['Hph']:.2e} Cfb={e['Cfb']:.2e} Lam={e['Lam']:.2e} Cff={e['Cff']:.2e}")
    cmf_state_rows.append(dict(s=s,T=Tc,ne=ne_cmf,fFeIV=nfe4/max(nfe3+nfe4,1e-30),r=e['r'],rpct=100*e['r']/Hdep[s],
                               Hph=e['Hph'],Cfb=e['Cfb'],Lam=e['Lam'],Cff=e['Cff'],Cad=e['Cad'],Hd=Hdep[s]))

# ---------- write CSVs ----------
def wcsv(fn,rows):
    if not rows: return
    keys=list(rows[0].keys())
    with open(fn,'w') as f:
        f.write(",".join(keys)+"\n")
        for r in rows: f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k],float) else str(r[k]) for k in keys)+"\n")
wcsv(f"{OUT}/rT_curves.csv",curve_rows)
wcsv(f"{OUT}/validation_roundtrip.csv",val_rows)
wcsv(f"{OUT}/cmfgen_state_residual.csv",cmf_state_rows)
print(f"\n[out] rT_curves.csv, validation_roundtrip.csv, cmfgen_state_residual.csv written to {OUT}")
