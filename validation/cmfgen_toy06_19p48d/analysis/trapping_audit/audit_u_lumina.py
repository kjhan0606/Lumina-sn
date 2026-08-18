#!/usr/bin/env python3
"""Audit U (Lumina side): u(s)=(4pi/c) INT mc_J dnu over the 1000-bin field grid.
Field CSV cols: shell,bin,wavelength_A,cs_J,mc_J. mc_J in erg/cm2/s/Hz/sr (same
convention as CMFGEN J_nu; cross-calibrated by the FUV band anchor 5.81e-6 vs 2.02e-4).
"""
import numpy as np, csv, sys

C_CM = 2.99792458e10
C_A  = 2.99792458e18   # A/s
FOURPI_OVER_C = 4*np.pi/C_CM
RUNS = {
 'Brun':   'logs/coevolve_consume_a10_kx_gphall/lumina_coevolve_field.csv',
 'tetab':  'logs/coevolve_consume_a10_kx_tetab/lumina_coevolve_field.csv',
 'tincol': 'logs/coevolve_consume_a10_kx_tincol/lumina_coevolve_field.csv',
}
BASE='/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/'
NU_LO, NU_HI = 1.5e14, 3.0e16

def load(path):
    d={}
    with open(BASE+path) as f:
        r=csv.reader(f); next(r)
        for row in r:
            s=int(row[0]); lam=float(row[2]); cs=float(row[3]); mc=float(row[4])
            d.setdefault(s,[]).append((lam,cs,mc))
    return d

def u_of_shell(rows, which):
    a=np.array(rows)              # [nbin, 3] lam,cs,mc
    lam=a[:,0]; val=a[:,1] if which=='cs' else a[:,2]
    nu=C_A/lam
    o=np.argsort(nu); nu=nu[o]; val=val[o]
    u_full=FOURPI_OVER_C*np.trapz(val,nu)
    m=(nu>=NU_LO)&(nu<=NU_HI)
    u_ovl=FOURPI_OVER_C*np.trapz(val[m],nu[m])
    return u_full,u_ovl,nu.min(),nu.max()

out={}
for name,path in RUNS.items():
    d=load(path)
    res={}
    for s in range(0,11):
        if s in d:
            uf,uo,nlo,nhi=u_of_shell(d[s],'mc')
            ufc,uoc,_,_=u_of_shell(d[s],'cs')
            res[s]=(uf,uo,ufc)
    out[name]=res
    print(f"\n# {name}: nu grid {nlo:.3e}..{nhi:.3e} Hz")
    print(f"{'shell':>5} {'u_mc_full':>11} {'u_mc_ovl':>11} {'u_cs_full':>11}")
    for s in range(0,11):
        if s in res:
            print(f"{s:>5} {res[s][0]:>11.4e} {res[s][1]:>11.4e} {res[s][2]:>11.4e}")

# save
import json
with open('/tmp/claude-10396/-home-kjhan-BACKUP-Eunha-A1-Claude-Lumina-sn/ab44c098-ca8e-4ede-a602-b922281a5709/scratchpad/u_lumina.json','w') as f:
    json.dump({k:{str(s):v for s,v in r.items()} for k,r in out.items()}, f)
print("\n[wrote] u_lumina.json")
