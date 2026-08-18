#!/usr/bin/env python3
"""FINAL decomposition of Lumina(kpr6) over-recombination vs CMFGEN(jnu4+published)
at kpr6 shells s4 (v=7176) and s6 (v=8632). All CMFGEN quantities interpolated to the
exact kpr6 shell mid-velocity. Same db_photoion_calc machinery / same sigma for both codes.
  R = n(FeIV)/n(FeIII) = Gamma / (ne * alpha)
  R_L/R_C = [Gamma_L/Gamma_C] * [alpha_C/alpha_L] * [ne_C/ne_L]
"""
import os, sys, math, numpy as np
REPO='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
os.environ['LUMINA_REF_DIR']=f'{REPO}/data/tardis_reference_toy06_19p48d'
os.environ['LUMINA_SIGMA_BIN']=f'{REPO}/data/tardis_reference_toy06_19p48d/cmfgen_sigma_bf.bin'
sys.path.insert(0,f'{REPO}/scripts'); sys.path.insert(0,f'{REPO}/validation/cmfgen_toy06_19p48d/analysis')
import db_photoion_calc as dbp, gamma_from_cmfgen_jnu as G, csv

JNU4='/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4'
KPR6=f'{REPO}/logs/coevolve_consume_a10_kx_kpr6'
STD=f'{REPO}/data/standart_data1/toy06'

# ---- CMFGEN field (jnu4) ----
Jc,nuc,ND,_=G.read_edd(f'{JNU4}/EDDFACTOR'); rt=open(f'{JNU4}/RVTJ').read()
Vc=G.rvtj_block(rt,'Velocity (km/s)',ND); Tc=G.rvtj_block(rt,'Temperature (10^4K)',ND)*1e4
Nec=G.rvtj_block(rt,'Electron density',ND)
order=np.argsort(Vc)  # ascending v for interp

def cmf_J_at(v):
    # linear interp of J[:,d] across the two bracketing depths in velocity
    vi=Vc[order]; i=np.searchsorted(vi,v); i=min(max(i,1),len(vi)-1)
    v0,v1=vi[i-1],vi[i]; w=(v-v0)/(v1-v0)
    d0,d1=order[i-1],order[i]
    return (1-w)*Jc[:,d0]+w*Jc[:,d1], (1-w)*Tc[d0]+w*Tc[d1], (1-w)*Nec[d0]+w*Nec[d1]

def milne_alpha(Te,Z=26,ion=2):
    idx=np.where((dbp.levZ==Z)&(dbp.levI==ion))[0]; idxu=np.where((dbp.levZ==Z)&(dbp.levI==ion+1))[0]
    xu=dbp.levE[idxu]/(dbp.KB_EV*Te); Uu=float(np.sum(np.where(xu<50,dbp.levG[idxu]*np.exp(-np.minimum(xu,50)),0)))
    if Uu<1: Uu=max(1.0,dbp.levG[idxu[0]] if len(idxu) else 1.0)
    lam3=(dbp.H*dbp.H/(2*dbp.PI*dbp.ME*dbp.KB*Te))**1.5; chi0=dbp.CHI[(Z,ion)]; kT=dbp.KB*Te; a=0.0
    for gl in idx:
        Rb,chi_l=dbp.R_planck(gl,Te,chi0)
        if chi_l>0 and dbp.flags[gl]: a+=Rb*lam3*dbp.levG[gl]/(2*Uu)*math.exp(min(chi_l/kT,300))
    return a

def gamma_field(J,Te,Z=26,ion=2):
    chi0=dbp.CHI[(Z,ion)]; idx=np.where((dbp.levZ==Z)&(dbp.levI==ion))[0]
    x=dbp.levE[idx]/(dbp.KB_EV*Te); U=float(np.sum(np.where(x<50,dbp.levG[idx]*np.exp(-np.minimum(x,50)),0)))
    Gg=Gb=0.0
    for gl in idx:
        R,_=dbp.R_of_level(gl,J,chi0)
        if R<=0: continue
        xl=dbp.levE[gl]/(dbp.KB_EV*Te)
        if xl>=50: continue
        pb=dbp.levG[gl]*math.exp(-xl)/U; Gb+=pb*R
        if dbp.levN[gl]==0: Gg+=R
    return Gg,Gb

# ---- published CMFGEN Fe ionfrac ----
def cmf_block(path,t=19.480):
    L=open(path).read().splitlines(); s=None
    for i,ln in enumerate(L):
        if ln.startswith('#TIME:') and abs(float(ln.split()[1])-t)<1e-3: s=i;break
    rows=[]; j=s+1
    while j<len(L):
        t2=L[j].strip()
        if t2.startswith('#TIME'): break
        if t2 and not t2.startswith('#'):
            try: rows.append([float(x) for x in t2.split()])
            except: pass
        j+=1
    return np.array(rows)
fe=cmf_block(f'{STD}/ionfrac_fe_toy06_cmfgen.txt')  # v,fe0..fe5 ; FeIII=col3, FeIV=col4
def cmf_R(v):
    f3=np.interp(v,fe[:,0],fe[:,3]); f4=np.interp(v,fe[:,0],fe[:,4]); return f4/f3

# ---- Lumina kpr6 ----
def lum(shell):
    Te,ne=dbp.plasma(KPR6,shell); J=dbp.field(KPR6,shell)
    Gg,Gb=gamma_field(J,Te); a=milne_alpha(Te)
    return Te,ne,Gg,Gb,a
pops={}
for r in csv.DictReader(open(f'{KPR6}/lumina_ion_pops.csv')):
    pops[(int(r['shell_id']),int(r['Z']),int(r['stage']))]=float(r['n_ion'])
def lum_R(shell,Z=26):
    return pops[(shell,Z,3)]/pops[(shell,Z,2)]

for shell,v in ((4,7176.0),(6,8632.0)):
    Jc_v,Te_C,ne_C=cmf_J_at(v)
    Jc_g=G.J_on_sigma_grid(Jc_v,nuc)
    GgC,GbC=gamma_field(Jc_g,Te_C); aC=milne_alpha(Te_C); R_C=cmf_R(v); f_C=R_C/(1+R_C)
    Te_L,ne_L,GgL,GbL,aL=lum(shell); R_L=lum_R(shell); f_L=R_L/(1+R_L)
    print(f"\n========== kpr6 s{shell}  v={v:.0f} km/s ==========")
    print(f"  {'':13}{'LUMINA(kpr6)':>16}{'CMFGEN(jnu4/pub)':>18}{'ratio L/C':>12}")
    print(f"  {'Te [K]':13}{Te_L:>16.0f}{Te_C:>18.0f}{Te_L/Te_C:>12.3f}")
    print(f"  {'ne [/cm3]':13}{ne_L:>16.3e}{ne_C:>18.3e}{ne_L/ne_C:>12.3f}")
    print(f"  {'alpha[cm3/s]':13}{aL:>16.3e}{aC:>18.3e}{aL/aC:>12.3f}")
    print(f"  {'Gph_gnd[/s]':13}{GgL:>16.3e}{GgC:>18.3e}{GgL/GgC:>12.3e}")
    print(f"  {'Gph_boltz[/s]':13}{GbL:>16.3e}{GbC:>18.3e}{GbL/GbC:>12.3e}")
    print(f"  {'G_bz/G_gnd':13}{GbL/GgL:>16.1f}{GbC/GgC:>18.1f}")
    print(f"  {'R=nIV/nIII':13}{R_L:>16.4f}{R_C:>18.4f}{R_L/R_C:>12.4f}")
    print(f"  {'f(FeIV)':13}{f_L:>16.4f}{f_C:>18.4f}")
    # decomposition of R_L/R_C
    gfac=GbL/GbC; afac=aC/aL; nefac=ne_C/ne_L
    print(f"  DECOMP  R_L/R_C = [Gph_L/Gph_C]*[a_C/a_L]*[ne_C/ne_L]")
    print(f"          {R_L/R_C:.4f}  =  [{gfac:.4f}] * [{afac:.4f}] * [{nefac:.4f}]  = {gfac*afac*nefac:.4f} (photo-balance)")
    print(f"          => Gamma deficit {1/gfac:.1f}x ; alpha {afac:.2f}x ; ne {nefac:.2f}x")
