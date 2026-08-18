#!/usr/bin/env python3
"""Ledger cross-cut: field super-thermality J_band / B_nu(T_e_local) at s6/s7/s8.
Discriminator for over-ionization: is the photospheric EUV field above (super-thermal,
non-thermal line dump) or below (thermal, absorbed) the local Planck function?
Reads field_bands.csv (J per band per side) + each side's own T_e.
"""
import os,sys,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,'..','formation_map'))
import cmfgen_formation_map as M
CL=2997.92458; H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10
BANDS=[(300,450),(450,912),(912,2000),(2000,4500),(4500,7000),(7000,3e4)]
BLBL=["300-450","450-912","912-2000","2000-4500","4500-7000","7000+"]
SHELLS={6:8632,7:9360,8:10088}
BRUN="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_gphall"
KPR8="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_kpr8"

def Bnu(nu,T):
    x=H*nu/(KB*T); return (2*H*nu**3/C**2)/np.expm1(np.clip(x,1e-30,700))
def bandB(lo,hi,T):
    lam=np.linspace(lo,hi,400); nu=CL/lam*1e15; nu=np.sort(nu)
    return np.trapz(Bnu(nu,T),nu)/(nu[-1]-nu[0])

def te(rundir):
    d={}
    for line in open(os.path.join(rundir,'lumina_plasma_state.csv')).read().splitlines()[1:]:
        p=line.split(','); s=int(p[0])
        if s in SHELLS: d[s]=float(p[4])
    return d
def Te_cmfgen():
    rt=open("/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ").read()
    V=M.parse_rvtj(rt,'Velocity (km/s)',90); T=M.parse_rvtj(rt,'Temperature (10^4K)',90)*1e4
    return {s:float(np.interp(v,V[::-1],T[::-1])) for s,v in SHELLS.items()}

def read_fields():
    F={}
    for line in open(os.path.join(HERE,'field_bands.csv')).read().splitlines()[1:]:
        p=line.split(','); s=p[0]; band=p[2]
        F[(s,band)]=dict(C=float(p[3]),Bm=float(p[4]),Km=float(p[6]))
    return F

def main():
    F=read_fields(); TeB=te(BRUN); TeK=te(KPR8); TeC=Te_cmfgen()
    lines=["shell,band,Te_C,Te_B,Te_K,JC/B,mcB/B,mcK/B"]
    print("shell band        Te(C/B/K)          JC/B    mcB/B   mcK/B   (>1 super-thermal)")
    for s,v in SHELLS.items():
        for lbl,(lo,hi) in zip(BLBL,BANDS):
            d=F[(f's{s}',lbl)]
            bC=bandB(lo,hi,TeC[s]); bB=bandB(lo,hi,TeB[s]); bK=bandB(lo,hi,TeK[s])
            rC=d['C']/bC; rB=d['Bm']/bB; rK=d['Km']/bK
            lines.append(f"s{s},{lbl},{TeC[s]:.0f},{TeB[s]:.0f},{TeK[s]:.0f},{rC:.3e},{rB:.3e},{rK:.3e}")
            print(f"s{s} {lbl:10s} {TeC[s]:.0f}/{TeB[s]:.0f}/{TeK[s]:.0f}  {rC:8.2e} {rB:8.2e} {rK:8.2e}")
    with open(os.path.join(HERE,'thermalization.csv'),'w') as f:
        f.write("\n".join(lines)+"\n")
    print("\n[wrote] thermalization.csv")

if __name__=='__main__':
    main()
