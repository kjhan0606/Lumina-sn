#!/usr/bin/env python3
"""coolant_burnout_kpr4.py -- the dominant offset closer: Lambda_line lost to over-ionization.
Lambda(kpr4 pops) vs Lambda(CMFGEN-frac pops) at kpr4 field & Te -> dLambda -> dTe via ledger slope.
Fast (no root bisection). Slopes from residual_ledger_kpr4.py."""
import csv,math,re,collections,numpy as np,pandas as pd,struct
H=6.62607015e-27;KB=1.380649e-16;C=2.99792458e10;EV=1.602176634e-12
T_EXP=19.48*86400.0;SOB=2.6540281e-2
REPO="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn"
RUN=f"{REPO}/logs/coevolve_consume_a10_kx_kpr4";BASE=f"{REPO}/data/tardis_reference_toy06_19p48d"
NU_MIN=1.5e14;NU_MAX=3.0e16;NFB=1000;DLN=math.log(NU_MAX/NU_MIN)/NFB
SHELLS=[0,2,4,6,8]
CMFTe={0:18760.22,2:16351.43,4:13657.24,6:11929.14,8:10383.0}
VEL={0:4264.0,2:5720.0,4:7176.0,6:8632.0,8:10088.0}
SLOPE={0:-1.171e-6,2:-3.610e-7,4:-5.806e-7,6:-6.885e-7,8:-1.144e-7}  # dr/dT from ledger

mc=collections.defaultdict(lambda:np.full(NFB,1e-30))
with open(f"{RUN}/lumina_coevolve_field.csv") as f:
    rd=csv.reader(f);next(rd)
    for row in rd:
        s=int(row[0])
        if s in SHELLS: mc[s][int(row[1])]=float(row[4])
state={}
for r in csv.DictReader(open(f"{RUN}/lumina_plasma_state.csv")):
    state[int(r['shell_id'])]=(float(r['n_e']),float(r['T_e']))
pops=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{RUN}/lumina_ion_pops.csv")):
    pops[int(r['shell_id'])][(int(r['Z']),int(r['stage']))]=float(r['n_ion'])

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
    try: v,dd=parse(f"{REPO}/data/standart_data1/toy06/ionfrac_{el}_toy06_cmfgen.txt")
    except: continue
    for s,vt in VEL.items():
        tot=0.0;row={}
        for stg in range(6):
            key=f"{el}{stg}"
            if key in dd: row[stg]=max(np.interp(vt,v,dd[key]),0.0);tot+=row[stg]
        for stg,val in row.items():
            if tot>0: cmf_frac[s][(Zc,stg)]=val/tot

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
dE=H*lnu;NLk=len(LZ)
gbar=0.2;om_floor=1.0;ry_de=np.minimum(13.605693/(dE/EV),136.0)
coeff=np.where(f_lu>1e-10,np.maximum(8.63e-6*14.5*gbar*f_lu*g_lo*ry_de,8.63e-6*om_floor),8.63e-6*om_floor)
ftau=SOB*f_lu*(C/lnu)*T_EXP;Bul=(C*C/(2*H*lnu**3))*A_ul;Blu=Bul*(g_up/g_lo)
ionkeys=list(set(zip(LZ.tolist(),Lion.tolist())));lineidx={k:np.where((LZ==k[0])&(Lion==k[1]))[0] for k in ionkeys}
def beta_esc(tau):return np.where(tau<=1e-6,1.0,np.where(tau>700,1.0/np.maximum(tau,1e-30),(1.0-np.exp(-np.minimum(tau,700)))/np.maximum(tau,1e-30)))
def Jbin_at(nu,arr):
    b=np.clip(np.floor(np.log(nu/NU_MIN)/DLN).astype(int),0,NFB-1);out=arr[b].copy();out[(nu<=NU_MIN)|(nu>=NU_MAX)]=1e-30;return out
def U_of(key,T):E,g=levarr[key];return float(np.sum(g*np.exp(-np.minimum(E/(KB*T),300))))
def Lambda(pop_map,T,ne,Jb):
    nion_lo=np.zeros(NLk)
    for key in ionkeys:nion_lo[lineidx[key]]=pop_map.get(key,0.0)
    U=np.ones(NLk)
    for key in ionkeys:U[lineidx[key]]=U_of(key,T)
    x=E_lo/(KB*T);nlo=np.where(x<300,nion_lo*g_lo*np.exp(-np.minimum(x,300))/U,0.0)
    invsq=1/math.sqrt(T);exb=np.exp(-np.minimum(dE/(KB*T),300))
    qlu=coeff/g_lo*invsq*exb;qul=coeff/g_up*invsq;Clu=ne*qlu;Cul=ne*qul
    tau=ftau*nlo;be=beta_esc(tau);Rul=(A_ul+Bul*Jb)*be;Rlu=Blu*Jb*be;den=Cul+Rul
    nup=np.where(den>0,nlo*(Clu+Rlu)/np.maximum(den,1e-300),0.0)
    return float(np.sum(dE*(nlo*qlu*ne-nup*qul*ne)))

print("### COOLANT BURNOUT (kpr4): Lambda_line at kpr4 pops vs CMFGEN-frac pops, kpr4 field & Te")
print(f"{'s':>3} {'Te':>6} {'ne':>9} {'FeIV_kpr4':>9} {'FeIV_cmf':>9} {'Lam_kpr4':>10} {'Lam_cmfpop':>10} {'dLam(restore)':>13} {'dTe_coolant':>11}")
rows=[]
for s in SHELLS:
    ne,Te=state[s];Jb=Jbin_at(lnu,mc[s])
    pk={(Z,st):pops[s].get((Z,st),0.0) for (Z,st) in pops[s]}
    elems=set(Z for (Z,st) in pops[s] if pops[s][(Z,st)]>0)
    pc={}
    for Z in elems:
        Ntot=sum(pops[s].get((Z,st),0.0) for st in range(7))
        fr={st:cmf_frac[s].get((Z,st),0.0) for st in range(6)};fsum=sum(fr.values())
        if fsum<=0:
            for st in range(7):pc[(Z,st)]=pops[s].get((Z,st),0.0)
        else:
            for st in range(6):pc[(Z,st)]=Ntot*fr[st]/fsum
    Lk=Lambda(pk,Te,ne,Jb);Lc=Lambda(pc,Te,ne,Jb)
    dLam=Lc-Lk                   # extra cooling if coolant restored (positive=more cooling)
    dTe=-(-dLam)/SLOPE[s]         # dr=-dLam(added cooling reduces r) ; dTe=-dr/slope = dLam/slope
    dTe=dLam/SLOPE[s]
    fe4k=pops[s].get((26,3),0)/max(sum(pops[s].get((26,st),0)for st in range(7)),1e-30)
    fe4c=cmf_frac[s].get((26,3),0)
    print(f"{s:>3} {Te:>6.0f} {ne:>9.2e} {fe4k:>9.4f} {fe4c:>9.4f} {Lk:>+10.3e} {Lc:>+10.3e} {dLam:>+13.3e} {dTe:>+11.0f}")
    rows.append((s,Te,CMFTe[s],Te-CMFTe[s],fe4k,fe4c,Lk,Lc,dLam,dTe))
print("\ndTe_coolant = cooling gained by restoring CMFGEN ionization (III coolant) -> shifts root DOWN (negative=cools).")
