#!/usr/bin/env python3
"""Ledger item 2: radiation field J_nu band-by-band at s6/s7/s8.
CMFGEN(EDDFACTOR jnu4) vs Lumina mc_J and cs_J (B-run & kpr8).
Bands (CMF Angstrom): 300-450,450-912,912-2000,2000-4500,4500-7000,7000+.
Band value = nu-weighted mean J_nu (trapz over nu / delta-nu).
Outputs field_bands.csv.
"""
import os, sys, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,'..','formation_map'))
import cmfgen_formation_map as M
CL=2997.92458
JNU4="/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"
BRUN="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_gphall"
KPR8="/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_kpr8"
SHELLS={6:8632,7:9360,8:10088}
BANDS=[(300,450),(450,912),(912,2000),(2000,4500),(4500,7000),(7000,1e12)]
BLBL=["300-450","450-912","912-2000","2000-4500","4500-7000","7000+"]

def band_mean_nu(lam, y, lo, hi):
    """nu-weighted mean of y over [lo,hi] Angstrom. lam,y unsorted."""
    m=(lam>=lo)&(lam<hi)
    if m.sum()<2: return np.nan
    nu=CL/lam[m]*1e15  # Hz  (CL is in 1e15 units -> nu Hz)
    yy=y[m]
    o=np.argsort(nu); nu=nu[o]; yy=yy[o]
    dn=nu[-1]-nu[0]
    if dn<=0: return np.nan
    return np.trapz(yy,nu)/dn

def cmfgen_J():
    # read EDDFACTOR (J_nu) records
    data,ND,fin=M.read_edd_records(JNU4+'/EDDFACTOR')
    NU=data[:,ND]; good=np.isfinite(data[:,:ND]).all(1)&(NU>0)
    J=data[good,:ND]; NU=NU[good]; lam=CL/NU
    rt=open(JNU4+'/RVTJ').read()
    V=M.parse_rvtj(rt,'Velocity (km/s)',ND)
    T=M.parse_rvtj(rt,'Temperature (10^4K)',ND)*1e4
    print(f"[cmfgen J] ND={ND} nfreq={J.shape[0]} lam {lam.min():.1f}-{lam.max():.3e}A FINISH={fin}")
    # per depth band means, then interp in V to target
    out={}
    for s,vt in SHELLS.items():
        # interpolate each band from the two depths bracketing vt (V descending)
        row={}
        # precompute band means at all depths is heavy; instead build J at target depth by
        # per-frequency log-interp in V (like extract_jnu) then band-mean.
        dv=np.argsort(V); Vasc=V[dv]
        Jt=np.empty(J.shape[0])
        Jsub=J[:,dv]
        # vectorized per-freq linear interp in V of log10 J (clip nonpos)
        idx=np.searchsorted(Vasc,vt); idx=np.clip(idx,1,len(Vasc)-1)
        x0,x1=Vasc[idx-1],Vasc[idx]; w=(vt-x0)/(x1-x0)
        a=Jsub[:,idx-1]; b=Jsub[:,idx]
        with np.errstate(divide='ignore',invalid='ignore'):
            la=np.log10(np.where(a>0,a,np.nan)); lb=np.log10(np.where(b>0,b,np.nan))
            Jt=10**((1-w)*la+w*lb)
        Jt=np.where(np.isfinite(Jt),Jt,0.0)
        Tt=np.interp(vt,Vasc,T[dv])
        for lbl,(lo,hi) in zip(BLBL,BANDS):
            row[lbl]=band_mean_nu(lam,Jt,lo,hi)
        row['T_e']=Tt
        out[s]=row
    return out

def lumina_field(rundir):
    """read coevolve_field.csv -> per shell arrays lam, cs_J, mc_J for shells 6/7/8."""
    d={s:{'lam':[],'cs':[],'mc':[]} for s in SHELLS}
    with open(os.path.join(rundir,'lumina_coevolve_field.csv')) as f:
        next(f)
        for line in f:
            p=line.split(',')
            s=int(p[0])
            if s not in SHELLS: continue
            d[s]['lam'].append(float(p[2])); d[s]['cs'].append(float(p[3])); d[s]['mc'].append(float(p[4]))
    out={}
    for s in SHELLS:
        lam=np.array(d[s]['lam']); cs=np.array(d[s]['cs']); mc=np.array(d[s]['mc'])
        row={}
        for lbl,(lo,hi) in zip(BLBL,BANDS):
            row['mc_'+lbl]=band_mean_nu(lam,mc,lo,hi)
            row['cs_'+lbl]=band_mean_nu(lam,cs,lo,hi)
        row['lam_min']=lam.min(); row['lam_max']=lam.max()
        out[s]=row
    return out

def main():
    C=cmfgen_J(); B=lumina_field(BRUN); K=lumina_field(KPR8)
    print(f"[lumina lam coverage] Brun s6: {B[6]['lam_min']:.1f}-{B[6]['lam_max']:.1f}A")
    lines=["shell,v_kms,band,cmfgen_J,Brun_mcJ,Brun_csJ,kpr8_mcJ,kpr8_csJ,mcB/C,mcK/C"]
    print("\nshell band        CMFGEN     Brun_mc    kpr8_mc    Brun_cs    | mcB/C   mcK/C")
    for s,v in SHELLS.items():
        for lbl in BLBL:
            c=C[s][lbl]; bm=B[s]['mc_'+lbl]; bc=B[s]['cs_'+lbl]; km=K[s]['mc_'+lbl]; kc=K[s]['cs_'+lbl]
            rb=bm/c if (c and np.isfinite(c) and c>0) else np.nan
            rk=km/c if (c and np.isfinite(c) and c>0) else np.nan
            lines.append(f"s{s},{v},{lbl},{c:.4e},{bm:.4e},{bc:.4e},{km:.4e},{kc:.4e},{rb:.4e},{rk:.4e}")
            print(f"s{s} {lbl:10s} {c:.3e}  {bm:.3e}  {km:.3e}  {bc:.3e}  | {rb:6.3f}  {rk:6.3f}")
    with open(os.path.join(HERE,'field_bands.csv'),'w') as f:
        f.write("\n".join(lines)+"\n")
    print("\n[wrote] field_bands.csv")

if __name__=='__main__':
    main()
