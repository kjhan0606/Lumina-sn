#!/usr/bin/env python3
"""CMFGEN Gamma_C(Fe III) + Te + ne at EXACT kpr6 shell velocities (v_mid).
Reuses gamma_from_cmfgen_jnu machinery. Also computes Milne alpha_C at CMFGEN Te
(same db machinery as Lumina) to test whether the Milne alpha itself diverges."""
import os, sys, math, numpy as np
REPO='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn'
os.environ['LUMINA_REF_DIR']=f'{REPO}/data/tardis_reference_toy06_19p48d'
os.environ['LUMINA_SIGMA_BIN']=f'{REPO}/data/tardis_reference_toy06_19p48d/cmfgen_sigma_bf.bin'
sys.path.insert(0,f'{REPO}/scripts')
sys.path.insert(0,f'{REPO}/validation/cmfgen_toy06_19p48d/analysis')
import db_photoion_calc as dbp
import gamma_from_cmfgen_jnu as G

EDD=f'/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR'
RVTJ=f'/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ'

J,nu,ND,fin=G.read_edd(EDD)
rt=open(RVTJ).read()
V =G.rvtj_block(rt,'Velocity (km/s)',ND)
T =G.rvtj_block(rt,'Temperature (10^4K)',ND)*1e4
NE=G.rvtj_block(rt,'Electron density',ND)
print(f"[edd] ND={ND} nfreq={J.shape[0]}  [rvtj] V[{V.min():.0f}..{V.max():.0f}] Te0={T[0]:.0f} ne0={NE[0]:.2e}")

def milne_alpha(Te,Z=26,ion=2):
    chi0=dbp.CHI[(Z,ion)]; kT=dbp.KB*Te
    idx=np.where((dbp.levZ==Z)&(dbp.levI==ion))[0]
    idxu=np.where((dbp.levZ==Z)&(dbp.levI==ion+1))[0]
    xu=dbp.levE[idxu]/(dbp.KB_EV*Te)
    Uu=float(np.sum(np.where(xu<50,dbp.levG[idxu]*np.exp(-np.minimum(xu,50)),0)))
    if Uu<1: Uu=max(1.0,dbp.levG[idxu[0]] if len(idxu) else 1.0)
    lam3=(dbp.H*dbp.H/(2*dbp.PI*dbp.ME*dbp.KB*Te))**1.5
    a=0.0
    for gl in idx:
        Rb,chi_l=dbp.R_planck(gl,Te,chi0)
        if chi_l>0 and dbp.flags[gl]:
            a+=Rb*lam3*dbp.levG[gl]/(2*Uu)*math.exp(min(chi_l/kT,300))
    return a

for vt in (7176, 8632, 6448, 7904):
    d=int(np.argmin(np.abs(V-vt)))
    Jg=G.J_on_sigma_grid(J[:,d], nu)
    fe_g,fe_b,fen,fek=G.gamma_ion(Jg,T[d],26,2)
    aC=milne_alpha(T[d])
    print(f"v_target={vt:5d}  depth={d+1:3d} v_cmf={V[d]:7.1f}  Te_C={T[d]:8.1f}  ne_C={NE[d]:.4e}  "
          f"Gamma_C(gnd)={fe_g:.4e} Gamma_C(boltz)={fe_b:.4e}  alpha_C(Milne)={aC:.4e}")
