#!/usr/bin/env python3
"""Calibrated field-swap budget for kpr2 warm bias. C_fb(Wien) normalization is
calibrated per shell so the baseline root == committed T_e (absorbs the pop-weight
uncertainty, assumed field-independent); then field-swaps give clean root shifts."""
import csv,math,struct,re,collections,numpy as np,pandas as pd
H=6.62607015e-27;KB=1.380649e-16;C=2.99792458e10;EV=1.602176634e-12
T_EXP=19.48*86400.0;SOB=2.6540281e-2
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN=f"{REPO}/logs/coevolve_consume_a10_kx_kpr2";BASE=f"{REPO}/data/tardis_reference_toy06_19p48d"
NU_MIN=1.5e14;NU_MAX=3.0e16;NFB=1000;DLN=math.log(NU_MAX/NU_MIN)/NFB
nu_mid=NU_MIN*np.exp((np.arange(NFB)+0.5)*DLN);hnu_mid=H*nu_mid
nu_lo=NU_MIN*np.exp(np.arange(NFB)*DLN);dnu=NU_MIN*np.exp((np.arange(NFB)+1)*DLN)-nu_lo
CMFTe={0:18760.22,4:13657.24,8:10382.79};VEL={0:4264.0,4:7176.0,8:10088.0}
CMFbias={0:1785,4:3789,8:6402}

CHI={}
for r in csv.DictReader(open(f"{BASE}/ionization_energies.csv")):
    CHI[(int(r['atomic_number']),int(r['ion_number']))]=float(r['ionization_energy_eV'])*EV
lev_Z=[];lev_ion=[];lev_E=[];lev_g=[];lev_ln=[]
for r in csv.DictReader(open(f"{BASE}/levels.csv")):
    lev_Z.append(int(r['atomic_number']));lev_ion.append(int(r['ion_number']))
    lev_E.append(float(r['energy_eV'])*EV);lev_g.append(float(r['g']));lev_ln.append(int(r['level_number']))
lev_Z=np.array(lev_Z);lev_ion=np.array(lev_ion);lev_E=np.array(lev_E);lev_g=np.array(lev_g);lev_ln=np.array(lev_ln)
NLEV=len(lev_Z);ion_levels={}
for i in range(NLEV):ion_levels.setdefault((lev_Z[i],lev_ion[i]),[]).append(i)
with open(f"{BASE}/cmfgen_sigma_bf.bin",'rb') as f:
    magic,ver,nlv,nfr=struct.unpack('<IIii',f.read(16));numin,numax=struct.unpack('<dd',f.read(16))
    flag8=np.frombuffer(f.read(nlv),dtype=np.int8);f.read((8-(nlv%8))%8)
    sig=np.frombuffer(f.read(nlv*nfr*8),dtype=np.float64).reshape(nlv,nfr)
has_sig=flag8.astype(bool)

csJ={};mcJ={}
t1=collections.defaultdict(lambda:np.full(NFB,1e-30));t2=collections.defaultdict(lambda:np.full(NFB,1e-30))
with open(f"{RUN}/lumina_coevolve_field.csv") as f:
    rd=csv.reader(f);next(rd)
    for row in rd:
        s=int(row[0])
        if s in(0,4,8):t1[s][int(row[1])]=float(row[3]);t2[s][int(row[1])]=float(row[4])
for s in(0,4,8):csJ[s]=t1[s];mcJ[s]=t2[s]
with open(f"{REPO}/data/cmfgen_jtable_toy06_19p48d.bin",'rb') as fh:
    mg,v2,nsh,nfbj=struct.unpack('<4i',fh.read(16));jt=np.frombuffer(fh.read(),np.float64).reshape(nsh,nfbj)
cmfJ={s:np.where(jt[s]>0,jt[s],1e-30) for s in(0,4,8)}
state={};pops=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{RUN}/lumina_plasma_state.csv")):state[int(r['shell_id'])]=(float(r['n_e']),float(r['T_e']))
for r in csv.DictReader(open(f"{RUN}/lumina_ion_pops.csv")):pops[int(r['shell_id'])][(int(r['Z']),int(r['stage']))]=float(r['n_ion'])
Hdep={}
for r in csv.DictReader(open(f"{BASE}/deposition_cmfgen.csv")):Hdep[int(r['shell_id'])]=float(r['heating_rate'])
nk_map=collections.defaultdict(dict)
with open(f"{RUN}/lumina_levelpop.csv") as f:
    rd=csv.reader(f);next(rd)
    for row in rd:
        s=int(row[0])
        if s in(0,4,8):nk_map[s][(int(row[1]),int(row[2]),int(row[3]))]=float(row[6])
# CMFGEN ion fractions
def parse(fn,tgt=19.48):
    L=open(fn).read().splitlines();times=[];st=[]
    for i,l in enumerate(L):
        m=re.match(r'#TIME:\s*([\d.]+)',l)
        if m:times.append(float(m.group(1)));st.append(i)
    k=int(np.argmin(abs(np.array(times)-tgt)));i0=st[k];i1=st[k+1] if k+1<len(st) else len(L)
    cols=None;v=[];d=[]
    for l in L[i0:i1]:
        if l.startswith('#vel_mid'):cols=l.strip().lstrip('#').split()[1:];continue
        if l.startswith('#'):continue
        p=l.split()
        try:vals=[float(x) for x in p]
        except:continue
        if len(p)>=2:v.append(vals[0]);d.append(vals[1:])
    return np.array(v),{c:np.array(d)[:,j] for j,c in enumerate(cols)}
cmf_frac=collections.defaultdict(dict)
for el,Zc in [('fe',26),('co',27),('ni',28),('si',14),('s',16),('ca',20)]:
    try:v,dd=parse(f"{REPO}/data/standart_data1/toy06/ionfrac_{el}_toy06_cmfgen.txt")
    except:continue
    for s,vt in VEL.items():
        tot=0.0;row={}
        for st in range(6):
            if f"{el}{st}" in dd:row[st]=max(np.interp(vt,v,dd[f"{el}{st}"]),0.0);tot+=row[st]
        for st,val in row.items():
            if tot>0:cmf_frac[s][(Zc,st)]=val/tot

def Bwien(T):return 2*H*nu_mid**3/C**2*np.exp(-np.minimum(hnu_mid/(KB*T),700))
def Bplanck(T):return 2*H*nu_mid**3/C**2/np.expm1(np.minimum(hnu_mid/(KB*T),700))

def build_A(s):
    """Precompute A[bb] = sum_l n_level * 4pi sig[l,bb] dnu/hnu (hnu-chi_l) (T-indep).
    Then any bf integral over field J is just A . J  (O(nfb))."""
    ne,Te=state[s];kT=KB*Te;A=np.zeros(NFB)
    for Z in sorted(set(z for (z,st) in pops[s] if pops[s][(z,st)]>0)):
        for j in sorted(st for (zz,st) in pops[s] if zz==Z and pops[s][(zz,st)]>0):
            chi_j=CHI.get((Z,j))
            if chi_j is None:continue
            nlo=pops[s].get((Z,j),0.0)
            gidx=ion_levels.get((Z,j),[])
            if nlo<=0 or not gidx:continue
            gidx=np.array(gidx);E=lev_E[gidx];g=lev_g[gidx];hs=has_sig[gidx];ln=lev_ln[gidx];x=E/kT
            U=max(np.sum(g[x<50]*np.exp(-np.minimum(x[x<50],300))),1.0)
            sel=hs&(x<50)&((chi_j-E)>0)
            if not sel.any():continue
            gi=gidx[sel];Es=E[sel];gs=g[sel];lns=ln[sel];chil=chi_j-Es
            nlev=np.array([nk_map[s].get((Z,j,int(lns[k])), nlo*gs[k]*math.exp(-Es[k]/kT)/U) for k in range(len(gi))])
            thr=np.clip(np.floor(np.log((chil/H)/NU_MIN)/DLN).astype(int),0,NFB-1)
            base=4*math.pi*dnu/hnu_mid
            for k in range(len(gi)):
                l=gi[k];b0=thr[k];sr=sig[l];m=np.zeros(NFB,bool);m[b0:]=sr[b0:]>0
                A[m]+=nlev[k]*sr[m]*base[m]*(hnu_mid[m]-chil[k])
    return A
def bf_term(A,Jarr):return float(np.dot(A,Jarr))

# lines
L=pd.read_csv(f"{BASE}/line_list.csv",usecols=['atomic_number','ion_number','level_number_lower','level_number_upper','f_lu','A_ul','nu'])
LZ=L['atomic_number'].values;Lion=L['ion_number'].values;llo=L['level_number_lower'].values;lup=L['level_number_upper'].values
f_lu=L['f_lu'].values;A_ul=L['A_ul'].values;lnu=L['nu'].values.astype(float)
tmpL={}
for r in csv.DictReader(open(f"{BASE}/levels.csv")):
    tmpL.setdefault((int(r['atomic_number']),int(r['ion_number'])),{})[int(r['level_number'])]=(float(r['energy_eV']),float(r['g']))
levarr={}
for key,d in tmpL.items():
    n=max(d)+1;E=np.zeros(n);g=np.ones(n)
    for k,(e,gg) in d.items():E[k]=e;g[k]=gg
    levarr[key]=(E,g)
NL=len(LZ);E_lo=np.zeros(NL);g_lo=np.ones(NL);g_up=np.ones(NL);valid=np.zeros(NL,bool)
for key in set(zip(LZ,Lion)):
    if key not in levarr:continue
    E,g=levarr[key];nmax=len(E);m=(LZ==key[0])&(Lion==key[1]);lo=llo[m];up=lup[m];ok=(lo<nmax)&(up<nmax)
    idx=np.where(m)[0][ok];E_lo[idx]=E[lo[ok]];g_lo[idx]=g[lo[ok]];g_up[idx]=g[up[ok]];valid[idx]=True
keep=valid&(lnu>0)
LZ=LZ[keep];Lion=Lion[keep];f_lu=f_lu[keep];A_ul=A_ul[keep];lnu=lnu[keep];E_lo=E_lo[keep];g_lo=g_lo[keep];g_up=g_up[keep]
dE=H*lnu;NLk=len(LZ);gbar=0.2;ry_de=np.minimum(13.605693/(dE/EV),136.0)
coeff=np.where(f_lu>1e-10,np.maximum(8.63e-6*14.5*gbar*f_lu*g_lo*ry_de,8.63e-6),8.63e-6)
ftau=SOB*f_lu*(C/lnu)*T_EXP;Bul=(C*C/(2*H*lnu**3))*A_ul;Blu=Bul*(g_up/g_lo)
ionkeys=list(set(zip(LZ.tolist(),Lion.tolist())));lineidx={k:np.where((LZ==k[0])&(Lion==k[1]))[0] for k in ionkeys}
def beta_esc(tau):return np.where(tau<=1e-6,1.0,np.where(tau>700,1.0/np.maximum(tau,1e-30),(1.0-np.exp(-np.minimum(tau,700)))/np.maximum(tau,1e-30)))
def Jbin_at(nu,arr):
    b=np.clip(np.floor(np.log(nu/NU_MIN)/DLN).astype(int),0,NFB-1);out=arr[b].copy();out[(nu<=NU_MIN)|(nu>=NU_MAX)]=1e-30;return out
def U_of(key,T):E,g=levarr[key];return float(np.sum(g*np.exp(-np.minimum(E/(KB*T)*EV,300))))
def Lambda(pop_map,T,ne,Jb):
    nion_lo=np.zeros(NLk)
    for key in ionkeys:nion_lo[lineidx[key]]=pop_map.get(key,0.0)
    U=np.ones(NLk)
    for key in ionkeys:U[lineidx[key]]=U_of(key,T)
    x=E_lo*EV/(KB*T);nlo=np.where(x<300,nion_lo*g_lo*np.exp(-np.minimum(x,300))/U,0.0)
    invsq=1/math.sqrt(T);exb=np.exp(-np.minimum(dE/(KB*T),300))
    qlu=coeff/g_lo*invsq*exb;qul=coeff/g_up*invsq;Clu=ne*qlu;Cul=ne*qul
    tau=ftau*nlo;be=beta_esc(tau);Rul=(A_ul+Bul*Jb)*be;Rlu=Blu*Jb*be;den=Cul+Rul
    nup=np.where(den>0,nlo*(Clu+Rlu)/np.maximum(den,1e-300),0.0)
    return float(np.sum(dE*(nlo*qlu*ne-nup*qul*ne)))
def Cff(ne,T):return 1.426e-27*1.2*ne*ne*math.sqrt(T)
def Cad(ne,T):return 1.5*ne*KB*T*(3.0/T_EXP)
def cmf_popmap(s):
    pc={}
    for Z in set(Z for (Z,st) in pops[s] if pops[s][(Z,st)]>0):
        Ntot=sum(pops[s].get((Z,st),0.0) for st in range(7));fr={st:cmf_frac[s].get((Z,st),0.0) for st in range(6)};fs=sum(fr.values())
        if fs<=0:
            for st in range(7):pc[(Z,st)]=pops[s].get((Z,st),0.0)
        else:
            for st in range(6):pc[(Z,st)]=Ntot*fr[st]/fs
    return pc

print("### CALIBRATED FIELD-SWAP BUDGET (root shifts)")
rows=[["shell","Te_kpr2","CMFGEN","bias","kappa","root_base","dTe_pump","dTe_gph","dTe_full","dTe_C1"]]
for s in(0,4,8):
    ne,Te=state[s];A=build_A(s);pk={(Z,st):pops[s].get((Z,st),0.0) for (Z,st) in pops[s]}
    Jline_cs=Jbin_at(lnu,csJ[s]);Jline_cmf=Jbin_at(lnu,cmfJ[s])
    Hph_mc=bf_term(A,mcJ[s]);Hph_cmf=bf_term(A,cmfJ[s]);CfbW_Te=bf_term(A,Bwien(Te))
    Lam_base=Lambda(pk,Te,ne,Jline_cs)
    kappa=(Hdep[s]+Hph_mc-Cff(ne,Te)-Cad(ne,Te)-Lam_base)/max(CfbW_Te,1e-30)
    def root(Hph,pump_pop,pump_J):
        def r(T):return Hdep[s]+Hph-Cff(ne,T)-Cad(ne,T)-kappa*bf_term(A,Bwien(T))-Lambda(pump_pop,T,ne,pump_J)
        lo,hi=6000.0,max(Te,26000.0)+3000
        if r(lo)<=0:return float('nan')
        for _ in range(30):
            mid=0.5*(lo+hi)
            if r(mid)>0:lo=mid
            else:hi=mid
        return 0.5*(lo+hi)
    r_base=root(Hph_mc,pk,Jline_cs)
    r_pump=root(Hph_mc,pk,Jline_cmf)
    r_gph =root(Hph_cmf,pk,Jline_cs)
    r_full=root(Hph_cmf,cmf_popmap(s),Jline_cmf)
    dC1=kappa*(bf_term(A,Bplanck(Te))-CfbW_Te)
    def rb(T):return Hdep[s]+Hph_mc-Cff(ne,T)-Cad(ne,T)-kappa*bf_term(A,Bwien(T))-Lambda(pk,T,ne,Jline_cs)
    slope=(rb(Te+400)-rb(Te-400))/800
    dTe_C1=dC1/slope
    print(f"\n s{s}: committed Te={Te:.0f}  CMFGEN={CMFTe[s]:.0f}  bias=+{CMFbias[s]}  kappa={kappa:.3f}")
    print(f"   root(baseline pump=cs_J,Gph=mc_J) = {r_base:.0f}  [calib target {Te:.0f}]")
    print(f"   root(pump->CMFj)   = {r_pump:.0f}   dTe(pump arm)     = {r_pump-r_base:+.0f}")
    print(f"   root(Gph->CMFj)    = {r_gph:.0f}   dTe(ioniz/bf arm) = {r_gph-r_base:+.0f}")
    print(f"   root(full CMFGEN)  = {r_full:.0f}   dTe(total field)  = {r_full-r_base:+.0f}  [CMFGEN {CMFTe[s]:.0f}]")
    print(f"   C1 Wien->Planck:   dTe = {dTe_C1:+.0f} K")
    rows.append([s,int(Te),int(CMFTe[s]),CMFbias[s],round(kappa,3),round(r_base),round(r_pump-r_base),round(r_gph-r_base),round(r_full-r_base),round(dTe_C1)])
import csv as _csv
with open(f"{REPO}/validation/cmfgen_toy06_19p48d/analysis/te_bias_budget/budget_shells.csv","w",newline="") as f:
    _csv.writer(f).writerows(rows)
print("\n# wrote budget_shells.csv")
