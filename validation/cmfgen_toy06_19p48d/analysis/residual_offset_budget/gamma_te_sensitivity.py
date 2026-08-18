#!/usr/bin/env python3
"""gamma_te_sensitivity.py -- isolate T_e vs field vs recombination for f(FeIV).

Self-consistent photoionization detailed balance with BOLTZMANN level pops (removes
the kpr4-NLTE-pop dependence): r = G_b/(ne*alpha), both G_b and alpha at the SAME Te.
Cross the field {kpr4 gphJ, CMFGEN jtable} with Te {kpr4, CMFGEN}. Compare to CMFGEN's
measured f(FeIV). If even (CMFGEN field, CMFGEN Te) gives f~0.99, the radiative-Milne
balance CANNOT produce CMFGEN's 0.022 -> the missing lever is RECOMBINATION (dielectronic),
not the field or T_e. Also reports the recombination deficit factor implied.
"""
import os, sys, struct, math
import numpy as np
REPO='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
os.environ['LUMINA_REF_DIR']=f'{REPO}/data/tardis_reference_toy06_19p48d'
os.environ['LUMINA_SIGMA_BIN']=f'{REPO}/data/tardis_reference_toy06_19p48d/cmfgen_sigma_bf.bin'
os.environ['LUMINA_SIGMA_COIII_PATCH']=f'{REPO}/data/coiii_real_sigma_patch.npz'
sys.path.insert(0,f'{REPO}/scripts')
import db_photoion_calc as dbp
RUN=f'{REPO}/logs/coevolve_consume_a10_kx_kpr4'
H=dbp.H;KB=dbp.KB;C=dbp.C;PI=dbp.PI;EV=dbp.EV;ME=dbp.ME;KB_EV=dbp.KB_EV
nu_c=dbp.nu_c;dnu=dbp.dnu;NFB=1000
SHELLS=[0,2,4,6,8]
CMF_TE={0:18760.22,2:16351.43,4:13657.24,6:11929.14,8:10383.0}

mcJ=np.zeros((50,NFB));csJ=np.zeros((50,NFB))
with open(f'{RUN}/lumina_coevolve_field.csv') as f:
    next(f)
    for line in f:
        p=line.split(','); s=int(p[0]);b=int(p[1]);csJ[s,b]=float(p[3]);mcJ[s,b]=float(p[4])
gphJ=np.where(mcJ>0,mcJ,csJ)
with open(f'{REPO}/data/cmfgen_jtable_toy06_19p48d.bin','rb') as fh:
    struct.unpack('<4i',fh.read(16)); jt=np.frombuffer(fh.read(),np.float64).reshape(50,NFB)
jt=np.where(jt>0,jt,0.0)

# CMFGEN measured Fe ion fractions at target shells (from standart ionfrac, cmfgen col)
import re
def parse_ionfrac(fn,tgt=19.48):
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
vcm,dcm=parse_ionfrac(f'{REPO}/data/standart_data1/toy06/ionfrac_fe_toy06_cmfgen.txt')
VEL={0:4264.0,2:5720.0,4:7176.0,6:8632.0,8:10088.0}
def cmf_ffe(s):  # returns dict stage->frac at shell velocity
    out={};tot=0.0
    for stg in range(6):
        key=f'fe{stg}'
        if key in dcm: out[stg]=max(np.interp(VEL[s],vcm,dcm[key]),0.0);tot+=out[stg]
    return {k:v/tot for k,v in out.items()} if tot>0 else out

Z,ion=26,2; chi0=dbp.CHI[(Z,ion)]
idx=np.where((dbp.levZ==Z)&(dbp.levI==ion))[0]
idxu=np.where((dbp.levZ==Z)&(dbp.levI==ion+1))[0]

def balance_boltz(J,Te,ne):
    kT=KB*Te
    x=dbp.levE[idx]/(KB_EV*Te)
    U=float(np.sum(np.where(x<50,dbp.levG[idx]*np.exp(-np.minimum(x,50)),0.0)))
    xu=dbp.levE[idxu]/(KB_EV*Te)
    Uu=float(np.sum(np.where(xu<50,dbp.levG[idxu]*np.exp(-np.minimum(xu,50)),0.0)))
    if Uu<1: Uu=max(1.0,dbp.levG[idxu[0]] if len(idxu) else 1.0)
    lam3=(H*H/(2*PI*ME*KB*Te))**1.5
    G_b=alpha=0.0
    for gl in idx:
        Rb,chi_l=dbp.R_planck(gl,Te,chi0)
        if chi_l>0 and dbp.flags[gl]:
            alpha+=Rb*lam3*dbp.levG[gl]/(2*Uu)*math.exp(min(chi_l/kT,300))
        R,_=dbp.R_of_level(gl,J,chi0)
        if R<=0: continue
        xl=dbp.levE[gl]/(KB_EV*Te)
        if xl>=50: continue
        pb=dbp.levG[gl]*math.exp(-xl)/U
        G_b+=pb*R
    return G_b,alpha,U,Uu

print("="*120)
print("Fe III->IV photoionization detailed balance, BOLTZMANN pops (self-consistent at each Te).")
print("r = G_b/(ne*alpha);  f_up=r/(1+r).  Field x Te cross.  Compare last col to CMFGEN measured f(FeIV).")
print("="*120)
print(f"{'shell':>5} {'ne':>9} | {'Te_kpr4':>7} {'Te_cmf':>7} |"
      f" {'f[gphJ,Tk]':>10} {'f[CMF,Tk]':>10} {'f[gphJ,Tc]':>10} {'f[CMF,Tc]':>10} |"
      f" {'f_CMF_meas':>10} {'alpha_defc':>10}")
rows=[]
for s in SHELLS:
    Te,ne=dbp.plasma(RUN,s); Tc=CMF_TE[s]
    def fpred(J,T):
        Gb,al,U,Uu=balance_boltz(J,T,ne); r=Gb/(ne*al) if al>0 else float('nan'); return r/(1+r),Gb,al
    f_gk,_,_=fpred(gphJ[s],Te)
    f_ck,Gb_ck,al_ck=fpred(jt[s],Te)
    f_gc,_,_=fpred(gphJ[s],Tc)
    f_cc,Gb_cc,al_cc=fpred(jt[s],Tc)
    # CMFGEN measured
    ff=cmf_ffe(s); f_meas=ff.get(3,0.0)  # Fe IV = stage index 3
    r_meas=f_meas/max(ff.get(2,1e-30),1e-30)
    # recomb deficit implied: to match r_meas at CMFGEN field+Te with fixed Gamma,
    # alpha must be multiplied by r_pred/r_meas
    r_cc=f_cc/max(1-f_cc,1e-30)
    defc=r_cc/max(r_meas,1e-30)
    rows.append((s,Te,Tc,ne,f_gk,f_ck,f_gc,f_cc,f_meas,defc))
    print(f"{s:>5} {ne:>9.2e} | {Te:>7.0f} {Tc:>7.0f} |"
          f" {f_gk:>10.4f} {f_ck:>10.4f} {f_gc:>10.4f} {f_cc:>10.4f} |"
          f" {f_meas:>10.4f} {defc:>10.2e}")
print("\nalpha_defc = r_pred(CMFGEN field,CMFGEN Te)/r_meas = the factor by which recombination")
print("(alpha_rad+alpha_DR+...) must exceed the radiative-Milne alpha to reproduce CMFGEN's f(FeIV).")
print("\nCMFGEN Fe stage fractions at each shell (measured):")
for s in SHELLS:
    ff=cmf_ffe(s)
    print(f"  s{s}: "+" ".join(f"Fe{st+1}={ff.get(st,0):.4f}" for st in range(5)))
