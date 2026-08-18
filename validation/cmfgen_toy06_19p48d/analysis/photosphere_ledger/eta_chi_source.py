#!/usr/bin/env python3
"""Ledger items 3+4 (THE load-bearing test): CMFGEN photospheric emissivity eta,
opacity chi, source function S=eta/chi, vs B(T_e), band-by-band at s6/s7/s8.
Also line/continuum decomposition (rolling low-pct floor).
Tests: (a) is CMFGEN's photospheric UV field a THERMAL continuum (S~B)? (b) is it line
or continuum? -> discriminates 'Lumina lacks a thermal bf continuum' vs 'Lumina lacks
UV opacity / thermalization'.
Uses cmf_flux ETA_DATA/CHI_DATA at the depth interpolated to each shell velocity.
"""
import os,sys,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,'..','formation_map'))
import cmfgen_formation_map as M
CL=2997.92458
D="/gpfs/kjhan/cmfgen_runs/toy06_19.48d_cmfflux"
H=6.62607015e-27; KB=1.380649e-16; C=2.99792458e10
SIGMA_T=6.6524587e-25
SHELLS={6:8632,7:9360,8:10088}
BANDS=[(300,450),(450,912),(912,1290),(1290,2000),(2000,4500),(4500,7000),(7000,1e12)]
BLBL=["300-450","450-912","912-1290","1290-2000","2000-4500","4500-7000","7000+"]

def Bnu(nu,T):
    x=H*nu/(KB*T)
    return (2*H*nu**3/C**2)/np.expm1(np.clip(x,1e-30,700))

def floor_env(x,win=151,pct=20):
    n=x.size; out=np.empty(n); h=win//2
    for i in range(n):
        a=max(0,i-h); b=min(n,i+h+1); out[i]=np.percentile(x[a:b],pct)
    return out

def main():
    eta_r,ND,_=M.read_edd_records(D+'/ETA_DATA')
    chi_r,_,_=M.read_edd_records(D+'/CHI_DATA')
    nrec=min(eta_r.shape[0],chi_r.shape[0]); eta_r=eta_r[:nrec]; chi_r=chi_r[:nrec]
    NU=eta_r[:,ND]; NUc=chi_r[:,ND]
    good=np.isfinite(eta_r).all(1)&np.isfinite(chi_r).all(1)&(NU>0)&(np.abs(NU-NUc)<1e-6*NU)
    eta=eta_r[good,:ND]; chi=chi_r[good,:ND]; NU=NU[good]; lam=CL/NU
    rt=open(D+'/RVTJ').read()
    V=M.parse_rvtj(rt,'Velocity (km/s)',ND)
    ED=M.parse_rvtj(rt,'Electron density',ND)
    T=M.parse_rvtj(rt,'Temperature (10^4K)',ND)*1e4
    print(f"[eta/chi] ND={ND} nfreq={good.sum()} lam {lam.min():.1f}-{lam.max():.2e}A")
    # sort by nu ascending once
    o=np.argsort(NU); NU=NU[o]; lam=lam[o]; eta=eta[o]; chi=chi[o]; nuHz=NU*1e15
    lines=["shell,v_kms,T_e,band,eta_mean,chi_mean,S_eta_over_chi,B_nu_Te,S_over_B,"
           "cont_frac_eta,line_frac_eta,S_cont,Scont_over_B"]
    print("\n shell band      S=eta/chi   B(Te)       S/B    contFrac  Scont/B")
    for s,vt in SHELLS.items():
        # nearest two depths bracketing vt (V descending with depth)
        d=int(np.argmin(np.abs(V-vt))); Te=float(np.interp(vt,V[::-1],T[::-1]))
        ne=float(np.interp(vt,V[::-1],ED[::-1]))
        # use nearest depth column (interp of full spectra across depth is costly; nearest is fine,
        # grid spacing ~550 km/s). report actual depth V.
        e=eta[:,d]; c=chi[:,d]
        chi_es=ne*SIGMA_T*1e10
        for lbl,(lo,hi) in zip(BLBL,BANDS):
            m=(lam>=lo)&(lam<hi)
            if m.sum()<3:
                lines.append(f"s{s},{vt},{Te:.0f},{lbl},nan,nan,nan,nan,nan,nan,nan,nan,nan"); continue
            nub=nuHz[m]; eb=e[m]; cb=c[m]
            dn=nub[-1]-nub[0]
            eta_m=np.trapz(eb,nub)/dn; chi_m=np.trapz(cb,nub)/dn
            S=eta_m/chi_m if chi_m>0 else np.nan
            Bm=np.trapz(Bnu(nub,Te),nub)/dn
            SoB=S/Bm if Bm>0 else np.nan
            # line/continuum split of emissivity
            ef=np.minimum(floor_env(eb),eb); cf=np.minimum(floor_env(cb),cb)
            E_tot=np.trapz(eb,nub); E_cont=np.trapz(ef,nub); E_line=E_tot-E_cont
            cont_frac=E_cont/E_tot if E_tot>0 else np.nan
            # continuum source function (floor eta / floor chi), es-removed from chi floor
            chi_cont_therm=np.clip(np.trapz(cf,nub)/dn - chi_es,1e-99,None)
            eta_cont_m=np.trapz(ef,nub)/dn
            # remove es emissivity ~ chi_es*J ~ chi_es*S (approx): eta_therm = eta_cont - chi_es*S
            eta_cont_therm=max(eta_cont_m - chi_es*S, 1e-99)
            S_cont=eta_cont_therm/chi_cont_therm if chi_cont_therm>0 else np.nan
            ScB=S_cont/Bm if Bm>0 else np.nan
            lines.append(f"s{s},{vt},{Te:.0f},{lbl},{eta_m:.4e},{chi_m:.4e},{S:.4e},{Bm:.4e},"
                         f"{SoB:.4e},{cont_frac:.4f},{1-cont_frac:.4f},{S_cont:.4e},{ScB:.4e}")
            print(f" s{s} {lbl:10s} {S:.3e}  {Bm:.3e}  {SoB:7.3f}  {cont_frac:6.3f}  {ScB:7.3f}")
        print(f"    (depth {d}, V_grid={V[d]:.0f}, T_e={Te:.0f}K, n_e={ne:.2e}, chi_es={chi_es:.2e})")
    with open(os.path.join(HERE,'eta_chi_source.csv'),'w') as f:
        f.write("\n".join(lines)+"\n")
    print("\n[wrote] eta_chi_source.csv")

if __name__=='__main__':
    main()
