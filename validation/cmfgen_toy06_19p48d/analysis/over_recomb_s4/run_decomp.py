#!/usr/bin/env python3
"""over-recomb s4 decomposition: Lumina alpha_L & Gamma_L (Fe III->IV) from the
kpr6 field, via the SAME db_photoion_calc machinery that built the CMFGEN gamma CSV.
Read-only: reads logs/coevolve_consume_a10_kx_kpr6/* and data/tardis_reference*.
"""
import os, math, numpy as np
REPO='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
os.environ['LUMINA_REF_DIR']=f'{REPO}/data/tardis_reference_toy06_19p48d'
os.environ['LUMINA_SIGMA_BIN']=f'{REPO}/data/tardis_reference_toy06_19p48d/cmfgen_sigma_bf.bin'
import sys; sys.path.insert(0,f'{REPO}/scripts')
import db_photoion_calc as dbp

KPR6=f'{REPO}/logs/coevolve_consume_a10_kx_kpr6'

def compute(shell, Z=26, ion=2):
    Te,ne=dbp.plasma(KPR6,shell)
    J=dbp.field(KPR6,shell)
    chi0=dbp.CHI[(Z,ion)]; kT=dbp.KB*Te
    idx=np.where((dbp.levZ==Z)&(dbp.levI==ion))[0]
    x=dbp.levE[idx]/(dbp.KB_EV*Te)
    U=float(np.sum(np.where(x<50,dbp.levG[idx]*np.exp(-np.minimum(x,50)),0)))
    idxu=np.where((dbp.levZ==Z)&(dbp.levI==ion+1))[0]
    xu=dbp.levE[idxu]/(dbp.KB_EV*Te)
    Uu=float(np.sum(np.where(xu<50,dbp.levG[idxu]*np.exp(-np.minimum(xu,50)),0)))
    if Uu<1: Uu=max(1.0,dbp.levG[idxu[0]] if len(idxu) else 1.0)
    lam3=(dbp.H*dbp.H/(2*dbp.PI*dbp.ME*dbp.KB*Te))**1.5
    G_gnd=G_b=alpha=alpha_planckfield_G=0.0
    Gb_planck=0.0
    for gl in idx:
        Rb,chi_l=dbp.R_planck(gl,Te,chi0)   # Planck field rate (for alpha Milne + flat-field ref)
        if chi_l>0 and dbp.flags[gl]:
            alpha+=Rb*lam3*dbp.levG[gl]/(2*Uu)*math.exp(min(chi_l/kT,300))
        R,_=dbp.R_of_level(gl,J,chi0)        # actual kpr6-field rate
        if R>0:
            xl=dbp.levE[gl]/(dbp.KB_EV*Te)
            if xl<50:
                pb=dbp.levG[gl]*math.exp(-xl)/U
                G_b+=pb*R
                if dbp.levN[gl]==0: G_gnd+=R
        if Rb>0:
            xl=dbp.levE[gl]/(dbp.KB_EV*Te)
            if xl<50:
                pb=dbp.levG[gl]*math.exp(-xl)/U
                Gb_planck+=pb*Rb
    r_b=G_b/(ne*alpha) if alpha>0 else float('nan')
    f_b=r_b/(1+r_b)
    return dict(shell=shell,Te=Te,ne=ne,U=U,Uu=Uu,alpha=alpha,
                G_gnd=G_gnd,G_b=G_b,Gb_planck=Gb_planck,r_b=r_b,f_b=f_b)

print("=== kpr6 Fe III (26,2->3) : Lumina alpha & Gamma from ACTUAL field ===")
for s in (0,4,6):
    d=compute(s)
    print(f"s{s}: Te={d['Te']:.0f} ne={d['ne']:.3e} U={d['U']:.1f} Uu={d['Uu']:.1f}")
    print(f"     alpha_L(Milne,FeIV->III) = {d['alpha']:.4e} cm3/s")
    print(f"     Gamma_L G_gnd={d['G_gnd']:.4e}  G_boltz(field)={d['G_b']:.4e}  G_boltz(PlanckTe)={d['Gb_planck']:.4e} s^-1")
    print(f"     photo-balance r_b=n(IV)/n(III)={d['r_b']:.4e}  f(FeIV)={d['f_b']:.4f}")
    print()
